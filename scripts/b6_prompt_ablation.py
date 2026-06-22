"""
b6_prompt_ablation.py - Ablative study of prompting formats for B6 (Caption-to-Molecule Generation)

Tests two additional prompting strategies against the baseline (zero-shot) for qwen3-4b:
  1. few_shot   — 2-shot ICL using precomputed TF-IDF similar examples
  2. decompose  — Structured decomposition: identify functional groups first, then generate

Usage:
    python scripts/b6_prompt_ablation.py --prompt_format few_shot --thinking on --vllm_url http://localhost:8000/v1
    python scripts/b6_prompt_ablation.py --prompt_format decompose --thinking off
    python scripts/b6_prompt_ablation.py --prompt_format all --thinking on   # run both formats
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR,
    GENERATION_SAMPLE_SIZE,
    MODELS,
    REPRESENTATIONS,
    REPR_DISPLAY_NAMES,
    RESULTS_DIR,
    STOP_SEQUENCES,
)
from utils import parsing, prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Fixed model for this ablation
QWEN3_4B_CONFIG = next(m for m in MODELS if m["id"] == "qwen3-4b-thinking-2507")

PROMPT_FORMATS = ["few_shot", "decompose"]

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _repr_display(representation: str) -> str:
    return REPR_DISPLAY_NAMES.get(representation, representation)


def create_fewshot_prompt(
    description: str,
    representation: str,
    few_shot_examples: List[Dict[str, str]],
    use_json_format: bool = False,
) -> str:
    """Delegate to the existing prompt builder with few-shot examples."""
    return prompts.create_generation_prompt(
        description,
        representation,
        few_shot_examples=few_shot_examples,
        use_json_format=use_json_format,
    )


def create_decompose_prompt(
    description: str,
    representation: str,
    use_json_format: bool = False,
) -> str:
    """Structured decomposition: identify substructures, then assemble."""
    rep_name = _repr_display(representation)

    instruction = f"""You are an expert chemist. Given a description of a molecule, generate the corresponding {rep_name} string by following these steps:

Step 1 — Identify the core scaffold and key functional groups mentioned in the description.
Step 2 — Determine how they connect (ring fusions, substituent positions, stereochemistry).
Step 3 — Write the final {rep_name} string.

Be concise in steps 1-2. Output tokens are limited."""

    if use_json_format:
        if representation == "moljson":
            format_instruction = '''Your final output must be ONLY a valid JSON object with "atoms" and "bonds" arrays.

Format:
{"atoms": [{"id": "C1", "element": "C"}, ...], "bonds": [{"source": "C1", "target": "C2", "order": 1.0}, ...], "charges": null, "aromatic_n_h": null}'''
        else:
            format_instruction = f'''Your final output must be a JSON object in this format:
{{"molecule": "..."}}

Where the value is the complete {rep_name} string.'''
    else:
        format_instruction = "Provide your final answer in the format: Final answer: <molecule_string>"

    return f"{instruction}\n\nDescription: {description}\n\n{format_instruction}"


# ---------------------------------------------------------------------------
# Few-shot example lookup
# ---------------------------------------------------------------------------


def load_fewshot_lookup(
    fewshot_path: Path, train_df: pd.DataFrame, id_col: str
) -> Dict[str, List[Dict[str, str]]]:
    """
    Build a mapping: test_molecule_id -> list of train example rows.
    Each example row is a dict with all representation columns + 'description'.
    """
    fewshot_df = pd.read_csv(fewshot_path)
    train_indexed = train_df.set_index(id_col)
    lookup = {}
    for _, row in fewshot_df.iterrows():
        test_id = row["test_molecule_id"]
        examples = []
        for i in range(1, 3):  # example_1_id, example_2_id
            ex_id = row[f"example_{i}_id"]
            if ex_id in train_indexed.index:
                examples.append(train_indexed.loc[ex_id])
        lookup[test_id] = examples
    return lookup


def get_fewshot_for_repr(
    examples: list, representation: str
) -> List[Dict[str, str]]:
    """Convert raw example rows to the format expected by prompts.create_generation_prompt."""
    out = []
    for ex in examples:
        mol_str = ex.get(representation, "")
        desc = ex.get("description", "")
        if mol_str and desc:
            out.append({"description": desc, "molecule": str(mol_str)})
    return out


# ---------------------------------------------------------------------------
# Reuse call_model from inference script (import-safe: no model instantiation)
# ---------------------------------------------------------------------------

# We import call_model, get_generation_params, get_json_schema_for_representation,
# load_checkpoint, save_to_checkpoint directly.
import importlib

_inf_module = importlib.import_module("02_run_inference")
call_model = _inf_module.call_model
get_generation_params = _inf_module.get_generation_params
get_json_schema_for_representation = _inf_module.get_json_schema_for_representation
load_checkpoint = _inf_module.load_checkpoint
save_to_checkpoint = _inf_module.save_to_checkpoint


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_ablation(
    prompt_format: str,
    thinking: bool,
    vllm_url: str,
    num_samples: Optional[int] = None,
    batch_size: int = 50,
    run_id: int = 1,
    dry_run: bool = False,
):
    model_config = QWEN3_4B_CONFIG
    model_id = model_config["id"]
    thinking_str = "thinking_on" if thinking else "thinking_off"

    logger.info(f"{'='*80}")
    logger.info(f"B6 Prompt Ablation: format={prompt_format}")
    logger.info(f"Model: {model_id} | Thinking: {thinking_str}")
    logger.info(f"{'='*80}")

    # Checkpoint path includes prompt_format to separate from baseline
    checkpoint_dir = RESULTS_DIR / "raw"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        checkpoint_dir
        / f"benchmark_6_{model_id}_{thinking_str}_{prompt_format}_run{run_id}.jsonl"
    )
    completed = load_checkpoint(checkpoint_path)

    # Load test data
    test_df = pd.read_csv(DATA_DIR / "chebi20_test.csv")
    id_col = "molecule_id" if "molecule_id" in test_df.columns else "CID"

    cap = num_samples or GENERATION_SAMPLE_SIZE
    if cap:
        test_df = test_df.head(cap)
    logger.info(f"Processing {len(test_df)} molecules")

    # Load few-shot lookup if needed
    fewshot_lookup = {}
    if prompt_format == "few_shot":
        train_df = pd.read_csv(DATA_DIR / "chebi20_train.csv")
        train_id_col = "molecule_id" if "molecule_id" in train_df.columns else "CID"
        fewshot_path = DATA_DIR / "fewshot_examples.csv"
        if not fewshot_path.exists():
            logger.error(f"fewshot_examples.csv not found at {fewshot_path}. Run 01_prepare_dataset.py first.")
            return
        fewshot_lookup = load_fewshot_lookup(fewshot_path, train_df, train_id_col)
        logger.info(f"Loaded few-shot mappings for {len(fewshot_lookup)} test molecules")

    # Pre-collect pending tasks
    all_tasks = []
    for _, row in test_df.iterrows():
        mol_id = row[id_col]
        description = row["description"]
        for repr_name in REPRESENTATIONS:
            if (mol_id, repr_name) in completed:
                continue

            json_schema = get_json_schema_for_representation(repr_name)
            use_json = json_schema is not None

            if prompt_format == "few_shot":
                examples_raw = fewshot_lookup.get(mol_id, [])
                fs_examples = get_fewshot_for_repr(examples_raw, repr_name)
                prompt_text = create_fewshot_prompt(
                    description, repr_name, fs_examples, use_json_format=use_json
                )
            elif prompt_format == "decompose":
                prompt_text = create_decompose_prompt(
                    description, repr_name, use_json_format=use_json
                )
            else:
                raise ValueError(f"Unknown prompt_format: {prompt_format}")

            all_tasks.append({
                "messages": [{"role": "user", "content": prompt_text}],
                "mol_id": mol_id,
                "repr_name": repr_name,
                "description": description,
                "json_schema": json_schema,
            })

    logger.info(f"Total pending tasks: {len(all_tasks)}")
    if not all_tasks:
        logger.info("Nothing to do — all tasks already checkpointed.")
        return

    gen_params = get_generation_params(model_config, "generation", thinking)
    stop_seqs = STOP_SEQUENCES.get("generation", [])

    if dry_run:
        for task in all_tasks[:3]:
            call_model(
                messages=task["messages"],
                model_config=model_config,
                thinking=thinking,
                vllm_url=vllm_url,
                generation_params=gen_params,
                stop_sequences=stop_seqs,
                json_schema=task["json_schema"],
                dry_run=True,
            )
        logger.info("Dry run complete (3 prompts shown).")
        return

    # Process in batches
    for batch_start in tqdm(range(0, len(all_tasks), batch_size), desc="Batches"):
        batch = all_tasks[batch_start : batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=min(len(batch), 32)) as executor:
            future_to_idx = {
                executor.submit(
                    call_model,
                    t["messages"],
                    model_config,
                    thinking,
                    vllm_url,
                    gen_params,
                    stop_seqs,
                    t["json_schema"],
                ): i
                for i, t in enumerate(batch)
            }
            results = [None] * len(batch)
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()

        for task, result in zip(batch, results):
            mol_id, repr_name = task["mol_id"], task["repr_name"]
            description = task["description"]

            if result is None:
                record = {
                    "molecule_id": mol_id,
                    "representation": repr_name,
                    "model": model_id,
                    "thinking": thinking,
                    "prompt_format": prompt_format,
                    "description": description,
                    "raw_response": None,
                    "parsed_answer": None,
                    "generated_string": None,
                    "latency_ms": None,
                }
            else:
                parsed_answer = result["parsed_answer"]
                json_schema = task["json_schema"]
                if json_schema is not None:
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
                    "molecule_id": mol_id,
                    "representation": repr_name,
                    "model": model_id,
                    "thinking": thinking,
                    "prompt_format": prompt_format,
                    "description": description,
                    "raw_response": result["raw_response"],
                    "parsed_answer": parsed_answer,
                    "generated_string": generated_string,
                    "latency_ms": result["latency_ms"],
                }
            save_to_checkpoint(record, checkpoint_path, run_id)
            completed.add((mol_id, repr_name))

    logger.info(f"Ablation complete: {prompt_format} | {model_id} | {thinking_str}")
    logger.info(f"Results: {checkpoint_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="B6 prompt ablation study for qwen3-4b")
    parser.add_argument(
        "--prompt_format",
        type=str,
        required=True,
        choices=PROMPT_FORMATS + ["all"],
        help="Prompting format to test (few_shot, decompose, or all)",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Enable/disable thinking mode",
    )
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--num_samples", type=int, default=None, help="Override sample cap")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true", help="Print prompts without calling API")
    args = parser.parse_args()

    thinking = args.thinking == "on"
    formats = PROMPT_FORMATS if args.prompt_format == "all" else [args.prompt_format]

    for fmt in formats:
        run_ablation(
            prompt_format=fmt,
            thinking=thinking,
            vllm_url=args.vllm_url,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            run_id=args.run_id,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
