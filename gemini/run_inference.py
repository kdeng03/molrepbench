#!/usr/bin/env python3
"""
Gemini 3 Flash Preview inference script for Molecular Representation Benchmark Suite.

Thin wrapper around the shared utilities in scripts/utils/.
Calls Google's Gemini generateContent REST API with thinking mode enabled.
Outputs JSONL checkpoints compatible with scripts/03_evaluate.py.

Usage:
    python gemini/run_inference.py --benchmark 1 --representation canonical_smiles --num_samples 5
    python gemini/run_inference.py --benchmark all
    python gemini/run_inference.py --benchmark 6 --representation moljson --num_samples 1 --dry_run
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — import shared utilities from scripts/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config import (
    BENCHMARK_NUM_TO_NAME,
    BENCHMARK_TOKEN_BUDGETS,
    DATA_DIR,
    ELEMENTS,
    FUNCTIONAL_GROUPS,
    GENERATION_SAMPLE_SIZE,
    PROPERTIES,
    REPRESENTATIONS,
    REPR_DISPLAY_NAMES,
    STOP_SEQUENCES,
)
from utils import chemistry, parsing, prompts, representations

# Gemini-specific config
import importlib.util
_gemini_config_spec = importlib.util.spec_from_file_location("gemini_config", PROJECT_ROOT / "gemini" / "config.py")
_gemini_config = importlib.util.module_from_spec(_gemini_config_spec)
_gemini_config_spec.loader.exec_module(_gemini_config)
MAX_CONCURRENT_REQUESTS = _gemini_config.MAX_CONCURRENT_REQUESTS
MAX_RETRIES = _gemini_config.MAX_RETRIES
MODEL = _gemini_config.MODEL
REQUEST_TIMEOUT = _gemini_config.REQUEST_TIMEOUT
RESULTS_DIR = _gemini_config.RESULTS_DIR
SKIP_BENCHMARKS = _gemini_config.SKIP_BENCHMARKS

# MolJSON schema
moljson_dir = PROJECT_ROOT / "moljson"
sys.path.insert(0, str(moljson_dir))
from schema import GetSchema as GetMolJSONSchema

import requests as http_requests  # avoid shadowing

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = MODEL["id"]
THINKING = True
THINKING_STR = "thinking_on"


# ============================================================================
# Gemini Structured Output Schema Helpers
# ============================================================================


UNSUPPORTED_KEYS = {"minLength", "maxLength", "pattern",
                     "exclusiveMinimum", "exclusiveMaximum",
                     "additionalProperties"}


def strip_unsupported_constraints(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively remove JSON schema constraints not supported by Gemini."""
    schema = deepcopy(schema)
    for key in UNSUPPORTED_KEYS:
        schema.pop(key, None)
    # Gemini requires all enum values to be strings; drop non-string enums
    if "enum" in schema and schema["enum"] and not isinstance(schema["enum"][0], str):
        schema.pop("enum")
    # Gemini doesn't support union types like ["array", "null"]; keep non-null type
    if isinstance(schema.get("type"), list):
        non_null = [t for t in schema["type"] if t != "null"]
        schema["type"] = non_null[0] if non_null else schema["type"][0]
        schema["nullable"] = True
    if "properties" in schema:
        for k, v in schema["properties"].items():
            if isinstance(v, dict):
                schema["properties"][k] = strip_unsupported_constraints(v)
    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = strip_unsupported_constraints(schema["items"])
    if "anyOf" in schema:
        schema["anyOf"] = [strip_unsupported_constraints(b) for b in schema["anyOf"]]
    return schema


def get_response_schema_for_representation(repr_name: str) -> Optional[Dict[str, Any]]:
    """
    Build a Gemini responseSchema dict for structured output.

    Gemini REST API uses responseMimeType + responseSchema in generationConfig.
    Returns the schema dict (caller sets responseMimeType separately).
    """
    if repr_name == "moljson":
        raw_schema = GetMolJSONSchema()
        raw_schema = strip_unsupported_constraints(raw_schema)
        return raw_schema

    if repr_name in [
        "canonical_smiles", "isomeric_smiles", "randomized_smiles",
        "deepsmiles", "iupac", "selfies",
    ]:
        return {
            "type": "object",
            "properties": {
                "molecule": {
                    "type": "string",
                    "description": (
                        f"Molecule written as {repr_name} ONLY. "
                        "Do not ask clarifying questions. Do not write any comments."
                    ),
                }
            },
            "required": ["molecule"],
        }

    return None


# ============================================================================
# Gemini generateContent API Caller
# ============================================================================


def call_gemini(
    messages: List[Dict[str, str]],
    generation_params: Dict[str, Any],
    thinking_level: str = "high",
    response_format: Optional[Dict[str, Any]] = None,
    stop: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Call Gemini generateContent API with thinking mode and retry logic.

    Args:
        messages: List of dicts with "role" and "content" keys (OpenAI-style).
                  Converted to Gemini's contents format internally.
        generation_params: Dict with temperature, top_p, max_tokens.
        thinking_level: Thinking level (minimal, low, medium, high).
        response_format: Gemini responseFormat dict for structured output.
        stop: Stop sequences.
        dry_run: If True, print request details without calling API.

    Returns:
        Dict with raw_response, parsed_answer, model, thinking, latency_ms
        or None on failure.
    """
    if dry_run:
        print(f"\n{'='*80}")
        print(f"Model: {MODEL['name']} | Thinking: {THINKING_STR}")
        print(f"Thinking level: {thinking_level}")
        print(f"Messages:")
        for msg in messages:
            print(f"  [{msg['role']}]: {msg['content'][:500]}...")
        print(f"Generation Params: {generation_params}")
        if response_format:
            print(f"Response Format: JSON schema")
        if stop:
            print(f"Stop Sequences: {stop}")
        print(f"{'='*80}\n")
        sys.stdout.flush()
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    api_url = f"{MODEL['api_base']}/models/{MODEL['api_model']}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    # Convert OpenAI-style messages to Gemini contents format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}],
        })

    # Build generationConfig
    generation_config = {
        "temperature": generation_params.get("temperature", 1.0),
        "topP": generation_params.get("top_p", 0.95),
        "maxOutputTokens": generation_params.get("max_tokens", 65536),
        "thinkingConfig": {
            "thinkingLevel": thinking_level.upper(),
            "includeThoughts": True,
        },
    }

    # Structured output via responseMimeType + responseSchema
    if response_format:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_format

    # Stop sequences
    if stop:
        generation_config["stopSequences"] = stop

    request_data = {
        "contents": contents,
        "generationConfig": generation_config,
    }

    # Retry loop with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            start_time = time.time()
            response = http_requests.post(
                api_url, json=request_data, headers=headers, timeout=REQUEST_TIMEOUT,
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 2 ** attempt))
                logger.warning(f"Rate limited (429). Retrying in {retry_after:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(retry_after)
                continue

            if response.status_code >= 500:
                wait = 2 ** attempt
                logger.warning(f"Server error {response.status_code}. Retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            if response.status_code == 400:
                logger.error(f"Bad request (400): {response.text[:500]}")
                return None

            if response.status_code != 200:
                logger.error(f"API error {response.status_code}: {response.text[:300]}")
                return None

            response_json = response.json()

            # Extract text from candidates[0].content.parts[]
            # Gemini returns parts with a "thought" boolean flag:
            #   thought=True  -> reasoning/thinking text
            #   thought=False -> final answer text
            candidates = response_json.get("candidates", [])
            if not candidates:
                logger.error(f"No candidates in response: {response.text[:300]}")
                return None

            parts = candidates[0].get("content", {}).get("parts", [])

            thinking_text = ""
            raw_response = ""

            for part in parts:
                if part.get("thought", False):
                    thinking_text += part.get("text", "")
                else:
                    raw_response += part.get("text", "")

            # Parse answer (strip thinking tags if model includes them in text)
            _, parsed_answer = parsing.extract_answer_from_thinking_response(raw_response)

            return {
                "raw_response": raw_response,
                "parsed_answer": parsed_answer,
                "model": MODEL_ID,
                "thinking": THINKING,
                "latency_ms": latency_ms,
            }

        except http_requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                logger.warning(f"Timeout. Retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            logger.error(f"API timeout after {MAX_RETRIES} retries")
            return None

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    logger.error(f"All {MAX_RETRIES} retries exhausted")
    return None


# ============================================================================
# Checkpoint Helpers
# ============================================================================


def load_checkpoint(checkpoint_path: Path) -> Set[Tuple]:
    """Load completed (molecule_id, representation) pairs from JSONL.

    For B2 (functional_groups) and B3 (property_estimation), the skip key is
    (molecule_id, "{group_or_prop}_{representation}") so we reconstruct that
    compound key when a 'group' or 'property' field is present in the record.
    """
    completed = set()
    if not checkpoint_path.exists():
        return completed
    with open(checkpoint_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record.get("raw_response") is None:
                continue  # skip failed rows so they get retried
            mol_id = record["molecule_id"]
            repr_name = record["representation"]
            if "group" in record:
                completed.add((mol_id, f"{record['group']}_{repr_name}"))
            elif "property" in record:
                completed.add((mol_id, f"{record['property']}_{repr_name}"))
            else:
                completed.add((mol_id, repr_name))
    logger.info(f"Loaded {len(completed)} completed items from {checkpoint_path}")
    return completed


def load_checkpoint_pairs(checkpoint_path: Path) -> Set[Tuple]:
    """Load completed (pair_id, representation) pairs from JSONL (B5, B9, B10)."""
    completed = set()
    if not checkpoint_path.exists():
        return completed
    with open(checkpoint_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record.get("raw_response") is None:
                continue  # skip failed rows so they get retried
            completed.add((record["pair_id"], record["representation"]))
    logger.info(f"Loaded {len(completed)} completed items from {checkpoint_path}")
    return completed


def save_to_checkpoint(record: Dict[str, Any], checkpoint_path: Path, run_id: int = 1):
    """Append a record to the JSONL checkpoint file."""
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps({**record, "run_id": run_id}) + "\n")


def load_test_dataset(dataset_path: Path):
    """Load test dataset, return (DataFrame, id_col, smiles_col)."""
    df = pd.read_csv(dataset_path)
    id_col = "CID" if "CID" in df.columns else "molecule_id"
    smiles_col = "smiles" if "smiles" in df.columns else "SMILES"
    return df, id_col, smiles_col


def get_generation_params(benchmark_name: str) -> Dict[str, Any]:
    """Get generation params for Gemini 3 Flash Preview."""
    return MODEL["generation_params"].copy()


def get_repr_string(mol_row, repr_name, smiles_col):
    """Get representation string for a molecule from a dataset row."""
    if repr_name == "iupac":
        return mol_row["iupac"] if "iupac" in mol_row and pd.notna(mol_row["iupac"]) else None
    elif repr_name == "selfies":
        if "selfies" in mol_row and pd.notna(mol_row["selfies"]):
            return mol_row["selfies"]
        if "SELFIES" in mol_row and pd.notna(mol_row["SELFIES"]):
            return mol_row["SELFIES"]
        return None
    else:
        return representations.convert_to_representation(mol_row[smiles_col], repr_name)


# ============================================================================
# Generic batch processor
# ============================================================================


def process_batch(all_tasks, gen_params, stop_seqs=None, max_workers=MAX_CONCURRENT_REQUESTS):
    """
    Process tasks with ThreadPoolExecutor, calling Gemini API.

    Each task must have a "messages" key and optionally "response_format".
    Returns list of results aligned with all_tasks.
    """
    results = [None] * len(all_tasks)

    thinking_level = MODEL["thinking_on_config"]["thinking_level"]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, task in enumerate(all_tasks):
            future = executor.submit(
                call_gemini,
                messages=task["messages"],
                generation_params=gen_params,
                thinking_level=thinking_level,
                response_format=task.get("response_format"),
                stop=stop_seqs,
            )
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Task {idx} raised: {e}")
                results[idx] = None

    return results


# ============================================================================
# Benchmark 1: Atom Counting
# ============================================================================


def run_benchmark_1(num_samples=None, dry_run=False, batch_size=MAX_CONCURRENT_REQUESTS, run_id=1):
    logger.info(f"\n{'='*80}\nBenchmark 1: Atom Counting | {MODEL_ID} | {THINKING_STR}\n{'='*80}")

    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_1_{MODEL_ID}_{THINKING_STR}_run{run_id}.jsonl"
    completed = load_checkpoint(checkpoint_path)

    subset_df = pd.read_csv(DATA_DIR / "comprehension_subset_ids.csv")
    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")
    molecule_ids = subset_df["molecule_id"].tolist()
    if num_samples:
        molecule_ids = molecule_ids[:num_samples]

    from rdkit import Chem
    all_tasks = []
    for mol_id in molecule_ids:
        mol_row = test_df[test_df[id_col] == mol_id].iloc[0]
        smiles = mol_row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        element_counts = chemistry.get_atom_counts(mol)
        available = [e for e in ELEMENTS if element_counts.get(e, 0) > 0]
        if not available:
            continue
        random.seed(hash(mol_id) % (2**32))
        element = random.choice(available)
        ground_truth = element_counts[element]

        for repr_name in REPRESENTATIONS:
            if (mol_id, repr_name) in completed:
                continue
            repr_string = get_repr_string(mol_row, repr_name, smiles_col)
            if repr_string is None:
                continue
            prompt_text = prompts.create_atom_counting_prompt(repr_string, element, repr_name)
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "mol_id": mol_id, "repr_name": repr_name,
                "element": element, "ground_truth": ground_truth,
            })

    logger.info(f"Pending tasks: {len(all_tasks)}")
    if dry_run:
        for task in all_tasks[:3]:
            call_gemini(task["messages"], get_generation_params("atom_counting"),
                        MODEL["thinking_on_config"]["thinking_level"], dry_run=True)
        return

    gen_params = get_generation_params("atom_counting")
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        results = process_batch(batch, gen_params)
        for task, result in zip(batch, results):
            mol_id, repr_name = task["mol_id"], task["repr_name"]
            if result is None:
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "element": task["element"], "raw_response": None,
                    "parsed_answer": None, "predicted": None, "ground_truth": task["ground_truth"],
                    "latency_ms": None,
                }
            else:
                predicted = parsing.extract_integer(result["parsed_answer"])
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "element": task["element"],
                    "raw_response": result["raw_response"], "parsed_answer": result["parsed_answer"],
                    "predicted": predicted, "ground_truth": task["ground_truth"],
                    "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((mol_id, repr_name))

    logger.info(f"Benchmark 1 complete. Results: {checkpoint_path}")


# ============================================================================
# Benchmark 2: Functional Group Identification
# ============================================================================


def run_benchmark_2(num_samples=None, dry_run=False, batch_size=MAX_CONCURRENT_REQUESTS, run_id=1):
    logger.info(f"\n{'='*80}\nBenchmark 2: Functional Groups | {MODEL_ID} | {THINKING_STR}\n{'='*80}")

    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_2_{MODEL_ID}_{THINKING_STR}_run{run_id}.jsonl"
    completed = load_checkpoint(checkpoint_path)

    subset_df = pd.read_csv(DATA_DIR / "comprehension_subset_ids.csv")
    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")
    molecule_ids = subset_df["molecule_id"].tolist()
    if num_samples:
        molecule_ids = molecule_ids[:num_samples]

    from rdkit import Chem
    all_tasks = []
    for mol_id in molecule_ids:
        mol_row = test_df[test_df[id_col] == mol_id].iloc[0]
        smiles = mol_row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        ground_truths = {fg_key: chemistry.has_substructure(mol, fg_config["smarts"])
                         for fg_key, fg_config in FUNCTIONAL_GROUPS.items()}
        for fg_key, fg_config in FUNCTIONAL_GROUPS.items():
            fg_name = fg_config["name"]
            ground_truth = ground_truths[fg_key]
            for repr_name in REPRESENTATIONS:
                if (mol_id, f"{fg_key}_{repr_name}") in completed:
                    continue
                repr_string = get_repr_string(mol_row, repr_name, smiles_col)
                if repr_string is None:
                    continue
                prompt_text = prompts.create_functional_group_prompt(repr_string, fg_name, repr_name)
                all_tasks.append({
                    "messages": [{"role": "user", "content": prompt_text}],
                    "mol_id": mol_id, "repr_name": repr_name,
                    "fg_key": fg_key, "ground_truth": ground_truth,
                })

    logger.info(f"Pending tasks: {len(all_tasks)}")
    if dry_run:
        for task in all_tasks[:3]:
            call_gemini(task["messages"], get_generation_params("functional_groups"),
                        MODEL["thinking_on_config"]["thinking_level"], dry_run=True)
        return

    gen_params = get_generation_params("functional_groups")
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        results = process_batch(batch, gen_params)
        for task, result in zip(batch, results):
            mol_id, repr_name = task["mol_id"], task["repr_name"]
            fg_key, ground_truth = task["fg_key"], task["ground_truth"]
            if result is None:
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "group": fg_key, "raw_response": None,
                    "parsed_answer": None, "predicted": None, "ground_truth": ground_truth,
                    "latency_ms": None,
                }
            else:
                predicted = parsing.extract_yes_no(result["parsed_answer"])
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "group": fg_key,
                    "raw_response": result["raw_response"], "parsed_answer": result["parsed_answer"],
                    "predicted": predicted, "ground_truth": ground_truth,
                    "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((mol_id, f"{fg_key}_{repr_name}"))

    logger.info(f"Benchmark 2 complete. Results: {checkpoint_path}")


# ============================================================================
# Benchmark 3: Property Estimation
# ============================================================================


def run_benchmark_3(num_samples=None, dry_run=False, batch_size=MAX_CONCURRENT_REQUESTS, run_id=1):
    logger.info(f"\n{'='*80}\nBenchmark 3: Property Estimation | {MODEL_ID} | {THINKING_STR}\n{'='*80}")

    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_3_{MODEL_ID}_{THINKING_STR}_run{run_id}.jsonl"
    completed = load_checkpoint(checkpoint_path)

    subset_df = pd.read_csv(DATA_DIR / "comprehension_subset_ids.csv")
    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")
    molecule_ids = subset_df["molecule_id"].tolist()
    if num_samples:
        molecule_ids = molecule_ids[:num_samples]

    from rdkit import Chem
    all_tasks = []
    for mol_id in molecule_ids:
        mol_row = test_df[test_df[id_col] == mol_id].iloc[0]
        smiles = mol_row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        props = chemistry.calculate_all_properties(mol)
        for prop_key, prop_config in PROPERTIES.items():
            prop_name = prop_config["name"]
            ground_truth = props.get(prop_key)
            if ground_truth is None:
                continue
            for repr_name in REPRESENTATIONS:
                if (mol_id, f"{prop_key}_{repr_name}") in completed:
                    continue
                repr_string = get_repr_string(mol_row, repr_name, smiles_col)
                if repr_string is None:
                    continue
                prompt_text = prompts.create_property_estimation_prompt(repr_string, prop_name, repr_name)
                all_tasks.append({
                    "messages": [{"role": "user", "content": prompt_text}],
                    "mol_id": mol_id, "repr_name": repr_name,
                    "prop_key": prop_key, "ground_truth": ground_truth,
                })

    logger.info(f"Pending tasks: {len(all_tasks)}")
    if dry_run:
        for task in all_tasks[:3]:
            call_gemini(task["messages"], get_generation_params("property_estimation"),
                        MODEL["thinking_on_config"]["thinking_level"], dry_run=True)
        return

    gen_params = get_generation_params("property_estimation")
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        results = process_batch(batch, gen_params)
        for task, result in zip(batch, results):
            mol_id, repr_name = task["mol_id"], task["repr_name"]
            prop_key, ground_truth = task["prop_key"], task["ground_truth"]
            if result is None:
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "property": prop_key, "raw_response": None,
                    "parsed_answer": None, "predicted": None, "ground_truth": ground_truth,
                    "latency_ms": None,
                }
            else:
                if prop_key in ["logp", "tpsa"]:
                    predicted = parsing.extract_float(result["parsed_answer"])
                else:
                    predicted = parsing.extract_integer(result["parsed_answer"])
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "property": prop_key,
                    "raw_response": result["raw_response"], "parsed_answer": result["parsed_answer"],
                    "predicted": predicted, "ground_truth": ground_truth,
                    "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((mol_id, f"{prop_key}_{repr_name}"))

    logger.info(f"Benchmark 3 complete. Results: {checkpoint_path}")


# ============================================================================
# Benchmark 4: Molecule Retrieval
# ============================================================================


def run_benchmark_4(num_samples=None, dry_run=False, batch_size=MAX_CONCURRENT_REQUESTS, run_id=1):
    logger.info(f"\n{'='*80}\nBenchmark 4: Molecule Retrieval | {MODEL_ID} | {THINKING_STR}\n{'='*80}")

    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_4_{MODEL_ID}_{THINKING_STR}_run{run_id}.jsonl"
    completed = load_checkpoint(checkpoint_path)

    subset_df = pd.read_csv(DATA_DIR / "comprehension_subset_ids.csv")
    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")
    distractors_df = pd.read_csv(DATA_DIR / "retrieval_distractors.csv")
    molecule_ids = subset_df["molecule_id"].tolist()
    if num_samples:
        molecule_ids = molecule_ids[:num_samples]

    all_tasks = []
    for mol_id in molecule_ids:
        mol_row = test_df[test_df[id_col] == mol_id].iloc[0]
        description = mol_row["description"]
        correct_smiles = mol_row[smiles_col]
        distractor_rows_df = distractors_df[distractors_df["target_molecule_id"] == mol_id]
        if len(distractor_rows_df) == 0:
            continue
        distractor_ids = distractor_rows_df["distractor_molecule_id"].tolist()
        distractor_mol_rows = []
        for dist_id in distractor_ids:
            dist_row = test_df[test_df[id_col] == dist_id]
            if len(dist_row) == 0:
                break
            distractor_mol_rows.append(dist_row.iloc[0])
        if len(distractor_mol_rows) != 3:
            continue

        for repr_name in REPRESENTATIONS:
            if (mol_id, repr_name) in completed:
                continue
            correct_repr = get_repr_string(mol_row, repr_name, smiles_col)
            if correct_repr is None:
                continue
            distractor_reprs = []
            for dist_row in distractor_mol_rows:
                dr = get_repr_string(dist_row, repr_name, smiles_col)
                if dr is None:
                    break
                distractor_reprs.append(dr)
            if len(distractor_reprs) != 3:
                continue
            choices = [correct_repr] + distractor_reprs
            random.seed(hash(f"{mol_id}_{repr_name}") % (2**32))
            random.shuffle(choices)
            correct_letter = "ABCD"[choices.index(correct_repr)]
            molecules_dict = {letter: mol for letter, mol in zip("ABCD", choices)}
            prompt_text = prompts.create_retrieval_prompt(description, molecules_dict, repr_name)
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "mol_id": mol_id, "repr_name": repr_name,
                "correct_letter": correct_letter,
            })

    logger.info(f"Pending tasks: {len(all_tasks)}")
    if dry_run:
        for task in all_tasks[:3]:
            call_gemini(task["messages"], get_generation_params("retrieval"),
                        MODEL["thinking_on_config"]["thinking_level"], dry_run=True)
        return

    gen_params = get_generation_params("retrieval")
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        results = process_batch(batch, gen_params)
        for task, result in zip(batch, results):
            mol_id, repr_name = task["mol_id"], task["repr_name"]
            correct_letter = task["correct_letter"]
            if result is None:
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "raw_response": None, "parsed_answer": None,
                    "predicted_letter": None, "correct_letter": correct_letter, "latency_ms": None,
                }
            else:
                predicted_letter = parsing.extract_multiple_choice(result["parsed_answer"])
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "raw_response": result["raw_response"],
                    "parsed_answer": result["parsed_answer"],
                    "predicted_letter": predicted_letter, "correct_letter": correct_letter,
                    "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((mol_id, repr_name))

    logger.info(f"Benchmark 4 complete. Results: {checkpoint_path}")


# ============================================================================
# Benchmark 5: Isomer Discrimination
# ============================================================================


def run_benchmark_5(num_samples=None, dry_run=False, batch_size=MAX_CONCURRENT_REQUESTS, run_id=1):
    logger.info(f"\n{'='*80}\nBenchmark 5: Isomer Discrimination | {MODEL_ID} | {THINKING_STR}\n{'='*80}")

    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_5_{MODEL_ID}_{THINKING_STR}_run{run_id}.jsonl"
    completed = load_checkpoint_pairs(checkpoint_path)

    pairs_df = pd.read_csv(DATA_DIR / "isomer_pairs.csv")
    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")
    if num_samples:
        pairs_df = pairs_df.head(num_samples)

    all_tasks = []
    for idx, row in pairs_df.iterrows():
        pair_id = idx
        smiles1, smiles2 = row["smiles_1"], row["smiles_2"]
        pair_type = row["type"] if "type" in row else row.get("pair_type", "unknown")
        ground_truth = pair_type in ("natural", "stereoisomer")
        mol_row_1 = test_df[test_df[id_col] == row["molecule_id_1"]]
        mol_row_2 = test_df[test_df[id_col] == row["molecule_id_2"]]

        for repr_name in REPRESENTATIONS:
            if (pair_id, repr_name) in completed:
                continue
            if repr_name == "iupac":
                repr1 = row.get("iupac_1") if pd.notna(row.get("iupac_1", None)) else None
                repr2 = row.get("iupac_2") if pd.notna(row.get("iupac_2", None)) else None
                if repr1 is None and len(mol_row_1) > 0:
                    r = mol_row_1.iloc[0]
                    repr1 = r.get("iupac") if pd.notna(r.get("iupac", None)) else None
                if repr2 is None and len(mol_row_2) > 0:
                    r = mol_row_2.iloc[0]
                    repr2 = r.get("iupac") if pd.notna(r.get("iupac", None)) else None
            elif repr_name == "selfies":
                repr1 = row.get("selfies_1") if pd.notna(row.get("selfies_1", None)) else None
                repr2 = row.get("selfies_2") if pd.notna(row.get("selfies_2", None)) else None
                if repr1 is None and len(mol_row_1) > 0:
                    r = mol_row_1.iloc[0]
                    repr1 = r.get("selfies") if pd.notna(r.get("selfies", None)) else (
                        r.get("SELFIES") if pd.notna(r.get("SELFIES", None)) else None)
                if repr2 is None and len(mol_row_2) > 0:
                    r = mol_row_2.iloc[0]
                    repr2 = r.get("selfies") if pd.notna(r.get("selfies", None)) else (
                        r.get("SELFIES") if pd.notna(r.get("SELFIES", None)) else None)
            else:
                repr1 = representations.convert_to_representation(smiles1, repr_name)
                repr2 = representations.convert_to_representation(smiles2, repr_name)

            if repr1 is None or repr2 is None:
                continue
            prompt_text = prompts.create_isomer_discrimination_prompt(repr1, repr2, repr_name)
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "pair_id": pair_id, "repr_name": repr_name,
                "pair_type": pair_type, "ground_truth": ground_truth,
            })

    logger.info(f"Pending tasks: {len(all_tasks)}")
    if dry_run:
        for task in all_tasks[:3]:
            call_gemini(task["messages"], get_generation_params("isomer_discrimination"),
                        MODEL["thinking_on_config"]["thinking_level"], dry_run=True)
        return

    gen_params = get_generation_params("isomer_discrimination")
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        results = process_batch(batch, gen_params)
        for task, result in zip(batch, results):
            pair_id, repr_name = task["pair_id"], task["repr_name"]
            pair_type, ground_truth = task["pair_type"], task["ground_truth"]
            if result is None:
                record = {
                    "pair_id": pair_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "pair_type": pair_type, "raw_response": None,
                    "parsed_answer": None, "predicted": None, "ground_truth": ground_truth,
                    "latency_ms": None,
                }
            else:
                predicted = parsing.extract_yes_no(result["parsed_answer"])
                record = {
                    "pair_id": pair_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "pair_type": pair_type,
                    "raw_response": result["raw_response"], "parsed_answer": result["parsed_answer"],
                    "predicted": predicted, "ground_truth": ground_truth,
                    "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((pair_id, repr_name))

    logger.info(f"Benchmark 5 complete. Results: {checkpoint_path}")


# ============================================================================
# Benchmark 6: Caption-to-Molecule Generation (with structured output)
# ============================================================================


def run_benchmark_6(num_samples=None, dry_run=False, batch_size=MAX_CONCURRENT_REQUESTS, run_id=1):
    logger.info(f"\n{'='*80}\nBenchmark 6: Caption-to-Molecule Generation | {MODEL_ID} | {THINKING_STR}\n{'='*80}")

    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_6_{MODEL_ID}_{THINKING_STR}_run{run_id}.jsonl"
    completed = load_checkpoint(checkpoint_path)

    test_df, id_col, smiles_col = load_test_dataset(DATA_DIR / "chebi20_test.csv")
    cap = num_samples or GENERATION_SAMPLE_SIZE
    if cap:
        test_df = test_df.head(cap)

    all_tasks = []
    for idx, row in test_df.iterrows():
        mol_id = row[id_col]
        description = row["description"]
        for repr_name in REPRESENTATIONS:
            if (mol_id, repr_name) in completed:
                continue
            response_format = get_response_schema_for_representation(repr_name)
            use_json = response_format is not None
            prompt_text = prompts.create_generation_prompt(
                description, repr_name, few_shot_examples=None, use_json_format=use_json,
            )
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "mol_id": mol_id, "repr_name": repr_name,
                "description": description, "response_format": response_format,
            })

    logger.info(f"Pending tasks: {len(all_tasks)}")
    if dry_run:
        for task in all_tasks[:3]:
            call_gemini(task["messages"], get_generation_params("generation"),
                        MODEL["thinking_on_config"]["thinking_level"],
                        response_format=task.get("response_format"), dry_run=True)
        return

    gen_params = get_generation_params("generation")
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        results = process_batch(batch, gen_params)
        for task, result in zip(batch, results):
            mol_id, repr_name = task["mol_id"], task["repr_name"]
            description = task["description"]
            if result is None:
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "description": description, "raw_response": None,
                    "parsed_answer": None, "generated_string": None, "latency_ms": None,
                }
            else:
                parsed_answer = result["parsed_answer"]
                # Extract molecule from structured JSON output
                response_format = task.get("response_format")
                if response_format is not None:
                    try:
                        json_obj = json.loads(parsed_answer)
                        if "molecule" in json_obj:
                            parsed_answer = json_obj["molecule"]
                        elif "atoms" in json_obj and "bonds" in json_obj:
                            parsed_answer = json.dumps(json_obj)
                    except (json.JSONDecodeError, TypeError):
                        pass
                generated_string = parsing.extract_molecule_string(parsed_answer, repr_name)
                record = {
                    "molecule_id": mol_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "description": description,
                    "raw_response": result["raw_response"], "parsed_answer": parsed_answer,
                    "generated_string": generated_string, "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((mol_id, repr_name))

    logger.info(f"Benchmark 6 complete. Results: {checkpoint_path}")


# ============================================================================
# Benchmark 9: Tautomer Recognition
# ============================================================================


def run_benchmark_9(num_samples=None, dry_run=False, batch_size=MAX_CONCURRENT_REQUESTS, run_id=1):
    logger.info(f"\n{'='*80}\nBenchmark 9: Tautomer Recognition | {MODEL_ID} | {THINKING_STR}\n{'='*80}")

    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_9_{MODEL_ID}_{THINKING_STR}_run{run_id}.jsonl"
    completed = load_checkpoint_pairs(checkpoint_path)

    pairs_df = pd.read_csv(DATA_DIR / "tautomer_pairs250.csv")
    if num_samples:
        pairs_df = pairs_df.head(num_samples)

    all_tasks = []
    for idx, row in pairs_df.iterrows():
        pair_id = int(row["pair_id"])
        smiles1, smiles2 = row["smiles_1"], row["smiles_2"]
        pair_type = row.get("pair_type", "unknown")
        tautomer_class = row.get("tautomer_class", "unknown")
        ground_truth = str(row["ground_truth"]).strip().lower() == "yes"

        for repr_name in REPRESENTATIONS:
            if (pair_id, repr_name) in completed:
                continue
            if repr_name == "randomized_smiles":
                from rdkit import Chem
                mol1 = Chem.MolFromSmiles(smiles1)
                mol2 = Chem.MolFromSmiles(smiles2)
                repr1 = Chem.MolToSmiles(mol1, doRandom=True) if mol1 else None
                repr2 = Chem.MolToSmiles(mol2, doRandom=True) if mol2 else None
            elif repr_name == "iupac":
                repr1 = row.get("iupac_1") if pd.notna(row.get("iupac_1", None)) else None
                repr2 = row.get("iupac_2") if pd.notna(row.get("iupac_2", None)) else None
            elif repr_name == "selfies":
                repr1 = row.get("selfies_1") if pd.notna(row.get("selfies_1", None)) else None
                repr2 = row.get("selfies_2") if pd.notna(row.get("selfies_2", None)) else None
            else:
                repr1 = representations.convert_to_representation(smiles1, repr_name)
                repr2 = representations.convert_to_representation(smiles2, repr_name)

            if repr1 is None or repr2 is None:
                continue
            prompt_text = prompts.create_tautomer_recognition_prompt(repr1, repr2, repr_name)
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "pair_id": pair_id, "repr_name": repr_name,
                "pair_type": pair_type, "tautomer_class": tautomer_class,
                "ground_truth": ground_truth,
            })

    logger.info(f"Pending tasks: {len(all_tasks)}")
    if dry_run:
        for task in all_tasks[:3]:
            call_gemini(task["messages"], get_generation_params("tautomer_recognition"),
                        MODEL["thinking_on_config"]["thinking_level"], dry_run=True)
        return

    gen_params = get_generation_params("tautomer_recognition")
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        results = process_batch(batch, gen_params)
        for task, result in zip(batch, results):
            pair_id, repr_name = task["pair_id"], task["repr_name"]
            pair_type = task["pair_type"]
            tautomer_class = task["tautomer_class"]
            ground_truth = task["ground_truth"]
            if result is None:
                record = {
                    "pair_id": pair_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "pair_type": pair_type, "tautomer_class": tautomer_class,
                    "raw_response": None, "parsed_answer": None, "predicted": None,
                    "ground_truth": ground_truth, "latency_ms": None,
                }
            else:
                predicted = parsing.extract_yes_no(result["parsed_answer"])
                record = {
                    "pair_id": pair_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "pair_type": pair_type, "tautomer_class": tautomer_class,
                    "raw_response": result["raw_response"], "parsed_answer": result["parsed_answer"],
                    "predicted": predicted, "ground_truth": ground_truth,
                    "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((pair_id, repr_name))

    logger.info(f"Benchmark 9 complete. Results: {checkpoint_path}")


# ============================================================================
# Benchmark 10: Protonation State Recognition
# ============================================================================


def run_benchmark_10(num_samples=None, dry_run=False, batch_size=MAX_CONCURRENT_REQUESTS, run_id=1):
    logger.info(f"\n{'='*80}\nBenchmark 10: Protonation State Recognition | {MODEL_ID} | {THINKING_STR}\n{'='*80}")

    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"benchmark_10_{MODEL_ID}_{THINKING_STR}_run{run_id}.jsonl"
    completed = load_checkpoint_pairs(checkpoint_path)

    pairs_df = pd.read_csv(DATA_DIR / "protonation_pairs250.csv")
    if num_samples:
        pairs_df = pairs_df.head(num_samples)

    all_tasks = []
    for idx, row in pairs_df.iterrows():
        pair_id = int(row["pair_id"])
        smiles1, smiles2 = row["smiles_1"], row["smiles_2"]
        pair_type = row.get("pair_type", "unknown")
        ionizable_group = row.get("ionizable_group", "none")
        charge_1 = row.get("charge_1", 0)
        charge_2 = row.get("charge_2", 0)
        ground_truth = str(row["ground_truth"]).strip().lower() == "yes"

        for repr_name in REPRESENTATIONS:
            if (pair_id, repr_name) in completed:
                continue
            if repr_name == "randomized_smiles":
                from rdkit import Chem
                mol1 = Chem.MolFromSmiles(smiles1)
                mol2 = Chem.MolFromSmiles(smiles2)
                repr1 = Chem.MolToSmiles(mol1, doRandom=True) if mol1 else None
                repr2 = Chem.MolToSmiles(mol2, doRandom=True) if mol2 else None
            elif repr_name == "iupac":
                repr1 = row.get("iupac_1") if pd.notna(row.get("iupac_1", None)) else None
                repr2 = row.get("iupac_2") if pd.notna(row.get("iupac_2", None)) else None
            elif repr_name == "selfies":
                repr1 = row.get("selfies_1") if pd.notna(row.get("selfies_1", None)) else None
                repr2 = row.get("selfies_2") if pd.notna(row.get("selfies_2", None)) else None
            else:
                repr1 = representations.convert_to_representation(smiles1, repr_name)
                repr2 = representations.convert_to_representation(smiles2, repr_name)

            if repr1 is None or repr2 is None:
                continue
            prompt_text = prompts.create_protonation_recognition_prompt(repr1, repr2, repr_name)
            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "pair_id": pair_id, "repr_name": repr_name,
                "pair_type": pair_type, "ionizable_group": ionizable_group,
                "charge_1": charge_1, "charge_2": charge_2,
                "ground_truth": ground_truth,
            })

    logger.info(f"Pending tasks: {len(all_tasks)}")
    if dry_run:
        for task in all_tasks[:3]:
            call_gemini(task["messages"], get_generation_params("protonation_recognition"),
                        MODEL["thinking_on_config"]["thinking_level"], dry_run=True)
        return

    gen_params = get_generation_params("protonation_recognition")
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start:batch_start + batch_size]
        results = process_batch(batch, gen_params)
        for task, result in zip(batch, results):
            pair_id, repr_name = task["pair_id"], task["repr_name"]
            pair_type = task["pair_type"]
            ionizable_group = task["ionizable_group"]
            charge_1, charge_2 = task["charge_1"], task["charge_2"]
            ground_truth = task["ground_truth"]
            if result is None:
                record = {
                    "pair_id": pair_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "pair_type": pair_type,
                    "ionizable_group": ionizable_group, "charge_1": charge_1, "charge_2": charge_2,
                    "raw_response": None, "parsed_answer": None, "predicted": None,
                    "ground_truth": ground_truth, "latency_ms": None,
                }
            else:
                predicted = parsing.extract_yes_no(result["parsed_answer"])
                record = {
                    "pair_id": pair_id, "representation": repr_name, "model": MODEL_ID,
                    "thinking": THINKING, "pair_type": pair_type,
                    "ionizable_group": ionizable_group, "charge_1": charge_1, "charge_2": charge_2,
                    "raw_response": result["raw_response"], "parsed_answer": result["parsed_answer"],
                    "predicted": predicted, "ground_truth": ground_truth,
                    "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((pair_id, repr_name))

    logger.info(f"Benchmark 10 complete. Results: {checkpoint_path}")


# ============================================================================
# Dispatcher & CLI
# ============================================================================

BENCHMARK_RUNNERS = {
    1: run_benchmark_1,
    2: run_benchmark_2,
    3: run_benchmark_3,
    4: run_benchmark_4,
    5: run_benchmark_5,
    6: run_benchmark_6,
    9: run_benchmark_9,
    10: run_benchmark_10,
}


def main():
    parser = argparse.ArgumentParser(description="Run Gemini 3 Flash Preview inference on molecular benchmarks")
    parser.add_argument("--benchmark", type=str, default="all",
                        help="Benchmark number (1-6, 9-10) or 'all'")
    parser.add_argument("--representation", type=str, default="all",
                        help="Representation name or 'all'")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of molecules (for testing)")
    parser.add_argument("--run_id", type=int, default=1,
                        help="Run ID for replication (1/2/3)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print prompts without calling API")
    parser.add_argument("--batch_size", type=int, default=MAX_CONCURRENT_REQUESTS,
                        help=f"Max concurrent API requests (default: {MAX_CONCURRENT_REQUESTS})")
    args = parser.parse_args()

    # Validate API key (unless dry run)
    if not args.dry_run and not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable is not set. Set it or use --dry_run.")
        sys.exit(1)

    # Filter representations if specified
    global REPRESENTATIONS
    if args.representation != "all":
        if args.representation not in REPRESENTATIONS:
            logger.error(f"Unknown representation: {args.representation}. Choose from: {REPRESENTATIONS}")
            sys.exit(1)
        REPRESENTATIONS = [args.representation]

    # Determine which benchmarks to run
    if args.benchmark == "all":
        benchmark_nums = [n for n in BENCHMARK_RUNNERS if n not in SKIP_BENCHMARKS]
    else:
        try:
            benchmark_nums = [int(args.benchmark)]
        except ValueError:
            logger.error(f"Invalid benchmark: {args.benchmark}. Use a number or 'all'.")
            sys.exit(1)
        if benchmark_nums[0] in SKIP_BENCHMARKS:
            logger.error(f"Benchmark {benchmark_nums[0]} is skipped (B7 completion not supported).")
            sys.exit(1)
        if benchmark_nums[0] not in BENCHMARK_RUNNERS:
            logger.error(f"Unknown benchmark: {benchmark_nums[0]}. Available: {list(BENCHMARK_RUNNERS.keys())}")
            sys.exit(1)

    logger.info(f"Model: {MODEL['name']} | Thinking: ON | Level: {MODEL['thinking_on_config']['thinking_level']}")
    logger.info(f"Benchmarks: {benchmark_nums}")
    logger.info(f"Representations: {REPRESENTATIONS}")
    logger.info(f"Results dir: {RESULTS_DIR}")
    if args.num_samples:
        logger.info(f"Sample limit: {args.num_samples}")

    for bench_num in benchmark_nums:
        runner = BENCHMARK_RUNNERS[bench_num]
        runner(
            num_samples=args.num_samples,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            run_id=args.run_id,
        )

    logger.info("\nAll benchmarks complete.")


if __name__ == "__main__":
    main()
