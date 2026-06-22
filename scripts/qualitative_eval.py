#!/usr/bin/env python3
"""
07_qualitative_eval.py — LLM-as-Judge qualitative evaluation of B6 results.

For each (molecule, representation) pair, batches ALL 14 models' responses into
a single Gemini API call for cross-model comparison. Evaluates chemical reasoning
quality, error patterns, representation faithfulness, overall quality, and provides
concrete qualitative insights.

Usage:
    python scripts/07_qualitative_eval.py                      # full run
    python scripts/07_qualitative_eval.py --dry-run            # print first prompt only
    python scripts/07_qualitative_eval.py --summarize          # print summary from existing results
    python scripts/07_qualitative_eval.py --n-molecules 5      # fewer molecules
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config import (
    CHEBI20_TEST,
    REPRESENTATIONS,
    REPR_DISPLAY_NAMES,
    SEED,
)

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
# Gemini judge configuration
# ---------------------------------------------------------------------------
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
JUDGE_MODEL = "gemini-3-flash-preview"
MAX_RETRIES = 5
REQUEST_TIMEOUT = 300  # seconds — large batched prompts may take a while
RAW_RESPONSE_TRUNCATE = 2000  # chars per model reasoning excerpt

# ---------------------------------------------------------------------------
# Model configs (14 main B6 run1 files)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = [
    {"label": "chemdfm-r-14b_on",       "path": PROJECT_ROOT / "results_small/raw/benchmark_6_chemdfm-r-14b_thinking_on_run1.jsonl"},
    {"label": "chemdfm-v2.0-14b_off",   "path": PROJECT_ROOT / "results_small/raw/benchmark_6_chemdfm-v2.0-14b_thinking_off_run1.jsonl"},
    {"label": "ether0-24b_on",           "path": PROJECT_ROOT / "results_small/raw/benchmark_6_ether0-24b_thinking_on_run1.jsonl"},
    {"label": "olmo-3.1-32b-instruct_off", "path": PROJECT_ROOT / "results_small/raw/benchmark_6_olmo-3.1-32b-instruct_thinking_off_run1.jsonl"},
    {"label": "olmo-3.1-32b-think_on",  "path": PROJECT_ROOT / "results_small/raw/benchmark_6_olmo-3.1-32b-think_thinking_on_run1.jsonl"},
    {"label": "phi-4_on",               "path": PROJECT_ROOT / "results_small/raw/benchmark_6_phi-4_thinking_on_run1.jsonl"},
    {"label": "phi-4-reasoning_on",     "path": PROJECT_ROOT / "results_small/raw/benchmark_6_phi-4-reasoning_thinking_on_run1.jsonl"},
    {"label": "phi-4-reasoning-plus_on", "path": PROJECT_ROOT / "results_small/raw/benchmark_6_phi-4-reasoning-plus_thinking_on_run1.jsonl"},
    {"label": "qwen3-30b-a3b_on",       "path": PROJECT_ROOT / "results_small/raw/benchmark_6_qwen3-30b-a3b-thinking-2507_thinking_on_run1.jsonl"},
    {"label": "qwen3-30b-a3b_off",      "path": PROJECT_ROOT / "results_small/raw/benchmark_6_qwen3-30b-a3b-thinking-2507_thinking_off_run1.jsonl"},
    {"label": "qwen3-4b_on",            "path": PROJECT_ROOT / "results_small/raw/benchmark_6_qwen3-4b-thinking-2507_thinking_on_run1.jsonl"},
    {"label": "qwen3-4b_off",           "path": PROJECT_ROOT / "results_small/raw/benchmark_6_qwen3-4b-thinking-2507_thinking_off_run1.jsonl"},
    {"label": "gpt-5.4-mini_on",        "path": PROJECT_ROOT / "results_gpt/raw/benchmark_6_gpt-5.4-mini_thinking_on_run1.jsonl"},
    {"label": "claude-haiku-4.5_on",    "path": PROJECT_ROOT / "results_claude/raw/benchmark_6_claude-haiku-4.5_thinking_on_run1.jsonl"},
    {"label": "mistral-small-24b_on",   "path": PROJECT_ROOT / "results_small/raw/benchmark_6_mistral-small-24b_thinking_on_run1.jsonl"},
    {"label": "qwen2.5-14b_off",        "path": PROJECT_ROOT / "results_small/raw/benchmark_6_qwen2.5-14b_thinking_off_run1.jsonl"},
]

# ---------------------------------------------------------------------------
# Gemini structured output schema
# ---------------------------------------------------------------------------
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "model_evaluations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "model_label": {"type": "STRING"},
                    "chemical_reasoning_quality": {"type": "INTEGER"},
                    "error_pattern": {
                        "type": "STRING",
                        "enum": [
                            "none",
                            "syntax_error",
                            "wrong_scaffold",
                            "missing_substituents",
                            "stereochemistry_error",
                            "hallucination",
                        ],
                    },
                    "representation_faithfulness": {"type": "INTEGER"},
                    "overall_quality": {"type": "INTEGER"},
                    "qualitative_insights": {"type": "STRING"},
                },
                "required": [
                    "model_label",
                    "chemical_reasoning_quality",
                    "error_pattern",
                    "representation_faithfulness",
                    "overall_quality",
                    "qualitative_insights",
                ],
            },
        },
        "comparative_summary": {"type": "STRING"},
    },
    "required": ["model_evaluations", "comparative_summary"],
}


# ============================================================================
# Data loading
# ============================================================================

def load_ground_truth():
    """Load ChEBI-20 test set as CID -> {smiles, iupac, description} dict."""
    df = pd.read_csv(CHEBI20_TEST)
    gt = {}
    for _, row in df.iterrows():
        gt[int(row["CID"])] = {
            "smiles": row["SMILES"],
            "iupac": str(row.get("iupacname", "")),
            "description": row["description"],
        }
    return gt


def sample_molecule_ids(n_molecules, seed):
    """Sample n molecule_ids from the first available result file."""
    for cfg in MODEL_CONFIGS:
        if cfg["path"].exists():
            ids = set()
            with open(cfg["path"]) as f:
                for line in f:
                    rec = json.loads(line)
                    ids.add(rec["molecule_id"])
            ids = sorted(ids)
            random.seed(seed)
            sampled = sorted(random.sample(ids, min(n_molecules, len(ids))))
            logger.info(f"Sampled {len(sampled)} molecule_ids: {sampled}")
            return sampled
    raise FileNotFoundError("No result files found")


def load_results(sampled_ids):
    """Load all model results for sampled molecules.

    Returns dict: (representation, molecule_id) -> [{model_label, raw_response, generated_string, ...}, ...]
    """
    sampled_set = set(sampled_ids)
    # key: (rep, mol_id) -> list of model responses
    batched = {}

    for cfg in MODEL_CONFIGS:
        if not cfg["path"].exists():
            logger.warning(f"File not found, skipping: {cfg['path']}")
            continue

        with open(cfg["path"]) as f:
            for line in f:
                rec = json.loads(line)
                if rec["molecule_id"] not in sampled_set:
                    continue
                key = (rec["representation"], rec["molecule_id"])
                if key not in batched:
                    batched[key] = []
                batched[key].append({
                    "model_label": cfg["label"],
                    "raw_response": rec.get("raw_response", ""),
                    "generated_string": rec.get("generated_string", ""),
                    "description": rec.get("description", ""),
                })

    logger.info(f"Loaded {sum(len(v) for v in batched.values())} records across {len(batched)} (rep, molecule) pairs")
    return batched


# ============================================================================
# Prompt building
# ============================================================================

def build_prompt(representation, molecule_id, model_responses, gt):
    """Build the batched evaluation prompt for one (representation, molecule_id)."""
    rep_display = REPR_DISPLAY_NAMES.get(representation, representation)
    gt_info = gt.get(molecule_id, {})
    description = model_responses[0]["description"] if model_responses else gt_info.get("description", "N/A")

    parts = [
        "You are an expert chemist evaluating AI models' attempts to generate a molecule "
        f"in **{rep_display}** format from a natural language description.\n",
        "## Task",
        f"**Description:** {description}\n",
        f"**Ground Truth (Canonical SMILES):** {gt_info.get('smiles', 'N/A')}",
        f"**Ground Truth (IUPAC):** {gt_info.get('iupac', 'N/A')}",
        f"**Requested output format:** {rep_display}\n",
        "## Model Responses\n",
    ]

    for i, resp in enumerate(model_responses):
        raw = resp["raw_response"] or ""
        if len(raw) > RAW_RESPONSE_TRUNCATE:
            raw = raw[:RAW_RESPONSE_TRUNCATE] + "... [truncated]"
        parts.append(f"### Model {chr(65+i)}: {resp['model_label']}")
        parts.append(f"**Reasoning (excerpt):**\n{raw}\n")
        parts.append(f"**Final answer:** {resp['generated_string']}\n")

    parts.append("""## Evaluation Instructions
For EACH model listed above, provide:

1. **chemical_reasoning_quality** (1-5): Does the reasoning demonstrate correct understanding of functional groups, ring systems, stereochemistry, and molecular topology? 1=no reasoning or completely wrong, 3=partially correct, 5=expert-level.

2. **error_pattern**: Classify the PRIMARY failure mode. Choose exactly one:
   - "none" — correct or near-correct molecule
   - "syntax_error" — invalid representation syntax
   - "wrong_scaffold" — fundamentally wrong core structure
   - "missing_substituents" — right scaffold but wrong/missing functional groups
   - "stereochemistry_error" — correct structure but wrong stereochemistry
   - "hallucination" — invents chemical features not described

3. **representation_faithfulness** (1-5): Does the output conform to the requested format? 1=completely wrong format, 5=perfect format compliance.

4. **overall_quality** (1-5): Holistic assessment. 1=completely wrong, 5=perfect.

5. **qualitative_insights**: 2-3 sentences of CONCRETE chemical observations. Mention specific functional groups, ring counts, atom counts, or structural features that were correct or incorrect. Be specific, not generic.

After evaluating all models, provide a **comparative_summary**: Which models demonstrated the strongest chemical reasoning? What common failure patterns emerged? Any surprising observations across models? Be concrete and specific.""")

    return "\n".join(parts)


# ============================================================================
# Gemini API call
# ============================================================================

def call_gemini_judge(prompt):
    """Call Gemini with structured output for judge evaluation.

    Returns parsed JSON dict on success, None on failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    api_url = f"{API_BASE}/models/{JUDGE_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    request_data = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]},
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.95,
            "maxOutputTokens": 65536,
            "thinkingConfig": {
                "thinkingLevel": "LOW",
                "includeThoughts": False,
            },
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }

    for attempt in range(MAX_RETRIES):
        try:
            start = time.time()
            resp = requests.post(
                api_url, json=request_data, headers=headers, timeout=REQUEST_TIMEOUT,
            )
            latency_ms = (time.time() - start) * 1000

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                logger.warning(f"Rate limited (429). Retrying in {retry_after:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(retry_after)
                continue

            if resp.status_code >= 500:
                wait = 2 ** attempt
                logger.warning(f"Server error {resp.status_code}. Retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            if resp.status_code == 400:
                logger.error(f"Bad request (400): {resp.text[:500]}")
                return None

            if resp.status_code != 200:
                logger.error(f"API error {resp.status_code}: {resp.text[:300]}")
                return None

            resp_json = resp.json()
            candidates = resp_json.get("candidates", [])
            if not candidates:
                logger.error(f"No candidates in response: {resp.text[:300]}")
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            # Extract the non-thought text (structured JSON output)
            text = ""
            for part in parts:
                if not part.get("thought", False):
                    text += part.get("text", "")

            result = json.loads(text)
            logger.debug(f"Judge response ({latency_ms:.0f}ms): {len(result.get('model_evaluations', []))} evaluations")
            return result

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                logger.warning(f"Timeout. Retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            logger.error(f"Timeout after {MAX_RETRIES} retries")
            return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse judge response as JSON: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    logger.error(f"All {MAX_RETRIES} retries exhausted")
    return None


# ============================================================================
# Checkpointing
# ============================================================================

def load_completed(output_path):
    """Return set of (representation, molecule_id) already evaluated."""
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                completed.add((rec["representation"], rec["molecule_id"]))
    return completed


def save_results(output_path, records):
    """Append evaluation records to output JSONL."""
    with open(output_path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ============================================================================
# Summary
# ============================================================================

def print_summary(output_path):
    """Print aggregated summary from completed evaluations."""
    df = pd.read_json(output_path, lines=True)
    n_records = len(df)
    n_models = df["model_label"].nunique()
    n_reps = df["representation"].nunique()
    n_mols = df["molecule_id"].nunique()
    print(f"\n{'='*70}")
    print(f"Qualitative Evaluation Summary: {n_records} evaluations")
    print(f"  {n_models} models x {n_reps} representations x {n_mols} molecules")
    print(f"{'='*70}\n")

    # Table 1: Mean scores by model
    print("--- Mean Scores by Model ---")
    model_summary = df.groupby("model_label").agg({
        "chemical_reasoning_quality": "mean",
        "representation_faithfulness": "mean",
        "overall_quality": "mean",
    }).round(2).sort_values("overall_quality", ascending=False)
    print(model_summary.to_string())
    print()

    # Table 2: Error pattern distribution by model
    print("--- Error Pattern Distribution by Model ---")
    error_dist = pd.crosstab(
        df["model_label"], df["error_pattern"], normalize="index"
    ).round(2)
    print(error_dist.to_string())
    print()

    # Table 3: Mean scores by representation
    print("--- Mean Scores by Representation ---")
    repr_summary = df.groupby("representation").agg({
        "chemical_reasoning_quality": "mean",
        "representation_faithfulness": "mean",
        "overall_quality": "mean",
    }).round(2).sort_values("overall_quality", ascending=False)
    print(repr_summary.to_string())
    print()

    # Comparative summaries (unique ones)
    summaries = df.drop_duplicates(subset=["representation", "molecule_id"])["comparative_summary"].dropna().tolist()
    if summaries:
        print("--- Sample Comparative Summaries ---")
        for i, s in enumerate(summaries[:5]):
            print(f"\n[{i+1}] {s}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="B6 Qualitative LLM-as-Judge Evaluation")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results_small" / "qualitative_eval_b6_v2.jsonl"),
                        help="Output JSONL path")
    parser.add_argument("--n-molecules", type=int, default=10, help="Number of molecules to sample")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for molecule sampling")
    parser.add_argument("--summarize", action="store_true", help="Print summary from existing results and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print first prompt without calling API")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.summarize:
        if not output_path.exists():
            logger.error(f"Output file not found: {output_path}")
            sys.exit(1)
        print_summary(output_path)
        return

    # Step 1: Load ground truth
    logger.info("Loading ground truth from ChEBI-20 test set...")
    gt = load_ground_truth()
    logger.info(f"Loaded {len(gt)} ground truth molecules")

    # Step 2: Sample molecule IDs
    sampled_ids = sample_molecule_ids(args.n_molecules, args.seed)

    # Step 3: Load all model results for sampled molecules
    batched = load_results(sampled_ids)

    # Step 4: Build task list
    tasks = []
    for rep in REPRESENTATIONS:
        for mol_id in sampled_ids:
            key = (rep, mol_id)
            if key in batched:
                tasks.append(key)

    logger.info(f"Total evaluation tasks: {len(tasks)} (representation x molecule pairs)")

    # Step 5: Check for completed evaluations
    completed = load_completed(output_path)
    remaining = [(rep, mol_id) for rep, mol_id in tasks if (rep, mol_id) not in completed]
    logger.info(f"Already completed: {len(completed)}, remaining: {len(remaining)}")

    if not remaining:
        logger.info("All evaluations already completed!")
        print_summary(output_path)
        return

    # Dry run: print first prompt
    if args.dry_run:
        rep, mol_id = remaining[0]
        prompt = build_prompt(rep, mol_id, batched[(rep, mol_id)], gt)
        print(f"\n{'='*70}")
        print(f"DRY RUN — Prompt for ({rep}, {mol_id})")
        print(f"Models in batch: {len(batched[(rep, mol_id)])}")
        print(f"Prompt length: {len(prompt)} chars")
        print(f"{'='*70}\n")
        print(prompt)
        return

    # Step 6: Run evaluations
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable is not set")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for rep, mol_id in tqdm(remaining, desc="Evaluating"):
        model_responses = batched[(rep, mol_id)]
        prompt = build_prompt(rep, mol_id, model_responses, gt)

        result = call_gemini_judge(prompt)
        if result is None:
            logger.warning(f"Failed evaluation for ({rep}, {mol_id}), skipping")
            continue

        # Flatten: one JSONL record per model evaluation
        comparative_summary = result.get("comparative_summary", "")
        records = []
        for eval_item in result.get("model_evaluations", []):
            records.append({
                "molecule_id": mol_id,
                "representation": rep,
                "model_label": eval_item.get("model_label", ""),
                "chemical_reasoning_quality": eval_item.get("chemical_reasoning_quality"),
                "error_pattern": eval_item.get("error_pattern", ""),
                "representation_faithfulness": eval_item.get("representation_faithfulness"),
                "overall_quality": eval_item.get("overall_quality"),
                "qualitative_insights": eval_item.get("qualitative_insights", ""),
                "comparative_summary": comparative_summary,
            })

        save_results(output_path, records)
        time.sleep(1)  # rate-limit courtesy

    logger.info("Evaluation complete!")
    print_summary(output_path)


if __name__ == "__main__":
    main()
