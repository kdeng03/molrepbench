"""
03_evaluate.py - Evaluate benchmark results and compute metrics

This script:
- Reads consolidated CSVs from results/benchmark_*.csv
- Computes all metrics (accuracy, F1, MAE, validity, Tanimoto, etc.)
- Saves scored CSVs to results/scored/
- Saves aggregated metrics to results/metrics/
- Computes cross-cutting analyses to results/cross/

NO model/GPU dependencies - pure pandas + RDKit + scipy
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import BENCHMARKS, RESULTS_DIR, DATA_DIR, REPRESENTATIONS
from utils import chemistry, representations, parsing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Helper Functions
# ============================================================================


def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def decode_generated_molecule(
    generated_string: str, repr_name: str
) -> Tuple[Optional[str], bool]:
    """
    Decode a generated molecule string to canonical SMILES.

    Args:
        generated_string: The generated molecule string
        repr_name: Representation name

    Returns:
        (canonical_smiles, is_valid) tuple
    """
    if generated_string is None or pd.isna(generated_string):
        return None, False

    # Reject pathologically long strings before passing to any decoder.
    # Real molecules in this benchmark are at most a few hundred characters;
    # models occasionally hallucinate thousands of tokens which can hang or OOM.
    MAX_REPR_LEN = 2000
    if len(str(generated_string)) > MAX_REPR_LEN:
        logger.debug(f"Skipping oversized {repr_name} string (len={len(str(generated_string))})")
        return None, False

    try:
        # Parse the representation back to a molecule
        mol = representations.parse_representation(generated_string, repr_name)
        if mol is None:
            return None, False

        # Convert to canonical SMILES
        from rdkit import Chem

        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return canonical, True

    except Exception as e:
        logger.debug(f"Failed to decode {repr_name} '{generated_string}': {e}")
        return None, False


def compute_tanimoto_similarity(
    smiles1: str, smiles2: str, fp_type: str = "morgan"
) -> Optional[float]:
    """
    Compute Tanimoto similarity between two SMILES.

    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        fp_type: Fingerprint type ("morgan" or "maccs")

    Returns:
        Tanimoto similarity (0-1) or None if computation fails
    """
    from rdkit import Chem

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return None

        return chemistry.calculate_tanimoto_similarity(mol1, mol2, fp_type)

    except Exception as e:
        logger.debug(f"Failed to compute Tanimoto: {e}")
        return None


def compute_molecule_complexity(smiles: str) -> Dict[str, any]:
    """
    Compute molecular complexity metrics.

    Args:
        smiles: SMILES string

    Returns:
        Dict with complexity metrics
    """
    from rdkit import Chem

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "num_heavy_atoms": None,
                "num_rings": None,
                "has_stereocenters": None,
                "molecular_weight": None,
            }

        return {
            "num_heavy_atoms": chemistry.count_heavy_atoms(mol),
            "num_rings": chemistry.count_rings(mol),
            "has_stereocenters": chemistry.has_stereocenters(mol),
            "molecular_weight": chemistry.calculate_molecular_weight(mol),
        }

    except Exception as e:
        logger.debug(f"Failed to compute complexity: {e}")
        return {
            "num_heavy_atoms": None,
            "num_rings": None,
            "has_stereocenters": None,
            "molecular_weight": None,
        }


# ============================================================================
# Helper: Load raw JSONL files from results/raw/
# ============================================================================


def load_raw_benchmark(benchmark_num: int) -> Optional[pd.DataFrame]:
    """
    Load and consolidate all JSONL files for a given benchmark number.

    Inference saves per-model JSONL files to results/raw/benchmark_{N}_*.jsonl.
    This helper merges them into a single DataFrame for evaluation.
    """
    raw_dir = RESULTS_DIR / "raw"
    pattern = f"benchmark_{benchmark_num}_*.jsonl"
    files = list(raw_dir.glob(pattern))
    if not files:
        logger.warning(f"Raw results not found: {raw_dir / pattern}")
        return None
    dfs = []
    for f in files:
        rows = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            dfs.append(pd.DataFrame(rows))
    if not dfs:
        logger.warning(f"All JSONL files empty for benchmark {benchmark_num}")
        return None
    df = pd.concat(dfs, ignore_index=True)
    # Normalise thinking column to bool
    if "thinking" in df.columns:
        df["thinking"] = df["thinking"].astype(bool)
    return df


# ============================================================================
# Benchmark 1: Atom Counting
# ============================================================================


def evaluate_benchmark_1():
    """Evaluate Benchmark 1: Atom Counting."""
    logger.info("Evaluating Benchmark 1: Atom Counting")

    # Load raw results
    df = load_raw_benchmark(1)
    if df is None:
        return

    # Compute correctness
    df["correct"] = df["predicted"] == df["ground_truth"]

    # Add molecule size bins (for breakdown analysis)
    # Load test set to get molecule properties
    test_df = pd.read_csv(DATA_DIR / "chebi_prepared_500.csv")
    test_df["molecule_id"] = test_df["CID"].astype(str)  # Map CID to molecule_id
    test_df["smiles"] = test_df["SMILES"]  # Map SMILES column

    # Compute heavy atom counts
    heavy_atom_counts = {}
    for _, row in test_df.iterrows():
        complexity = compute_molecule_complexity(row["smiles"])
        heavy_atom_counts[row["molecule_id"]] = complexity["num_heavy_atoms"]

    df["num_heavy_atoms"] = df["molecule_id"].map(heavy_atom_counts)

    # Create size bins
    df["size_bin"] = pd.cut(
        df["num_heavy_atoms"],
        bins=[0, 20, 35, 100],
        labels=["small (<20)", "medium (20-35)", "large (35+)"],
    )

    # Save scored results
    scored_dir = RESULTS_DIR / "scored"
    ensure_dir(scored_dir)
    scored_path = scored_dir / "benchmark_1_scored.csv"
    df.to_csv(scored_path, index=False)
    logger.info(f"Saved scored results to {scored_path}")

    # Compute aggregate metrics
    metrics_rows = []

    for (representation, model, thinking), group in df.groupby(
        ["representation", "model", "thinking"]
    ):
        # Overall accuracy
        accuracy = group["correct"].mean()

        # Accuracy by size bin
        accuracy_by_size = {}
        for size_bin in ["small (<20)", "medium (20-35)", "large (35+)"]:
            size_group = group[group["size_bin"] == size_bin]
            if len(size_group) > 0:
                accuracy_by_size[size_bin] = size_group["correct"].mean()
            else:
                accuracy_by_size[size_bin] = None

        metrics_rows.append(
            {
                "representation": representation,
                "model": model,
                "thinking": thinking,
                "accuracy": accuracy,
                "accuracy_small": accuracy_by_size["small (<20)"],
                "accuracy_medium": accuracy_by_size["medium (20-35)"],
                "accuracy_large": accuracy_by_size["large (35+)"],
                "n_samples": len(group),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    # Save aggregate metrics
    metrics_dir = RESULTS_DIR / "metrics"
    ensure_dir(metrics_dir)
    metrics_path = metrics_dir / "benchmark_1_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved aggregate metrics to {metrics_path}")

    # Print summary
    logger.info(f"Mean accuracy: {metrics_df['accuracy'].mean():.3f}")
    logger.info(
        f"Best: {metrics_df.loc[metrics_df['accuracy'].idxmax(), 'representation']} "
        f"({metrics_df['accuracy'].max():.3f})"
    )


# ============================================================================
# Benchmark 2: Functional Group Identification
# ============================================================================


def evaluate_benchmark_2():
    """Evaluate Benchmark 2: Functional Group Identification."""
    logger.info("Evaluating Benchmark 2: Functional Group Identification")

    # Load raw results
    df = load_raw_benchmark(2)
    if df is None:
        return

    # Compute correctness
    df["correct"] = df["predicted"] == df["ground_truth"]

    # Save scored results
    scored_dir = RESULTS_DIR / "scored"
    ensure_dir(scored_dir)
    scored_path = scored_dir / "benchmark_2_scored.csv"
    df.to_csv(scored_path, index=False)
    logger.info(f"Saved scored results to {scored_path}")

    # Compute aggregate metrics
    metrics_rows = []

    for (representation, model, thinking, group_name), group in df.groupby(
        ["representation", "model", "thinking", "group"]
    ):
        # Compute TP, FP, TN, FN
        tp = ((group["predicted"] == True) & (group["ground_truth"] == True)).sum()
        fp = ((group["predicted"] == True) & (group["ground_truth"] == False)).sum()
        tn = ((group["predicted"] == False) & (group["ground_truth"] == False)).sum()
        fn = ((group["predicted"] == False) & (group["ground_truth"] == True)).sum()

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

        metrics_rows.append(
            {
                "representation": representation,
                "model": model,
                "thinking": thinking,
                "group": group_name,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "n_samples": len(group),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    # Compute macro-averaged F1
    macro_f1_rows = []
    for (representation, model, thinking), group in metrics_df.groupby(
        ["representation", "model", "thinking"]
    ):
        macro_f1 = group["f1"].mean()
        macro_f1_rows.append(
            {
                "representation": representation,
                "model": model,
                "thinking": thinking,
                "macro_f1": macro_f1,
            }
        )

    macro_f1_df = pd.DataFrame(macro_f1_rows)

    # Save aggregate metrics
    metrics_dir = RESULTS_DIR / "metrics"
    ensure_dir(metrics_dir)
    metrics_path = metrics_dir / "benchmark_2_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved per-group metrics to {metrics_path}")

    macro_path = metrics_dir / "benchmark_2_macro_metrics.csv"
    macro_f1_df.to_csv(macro_path, index=False)
    logger.info(f"Saved macro metrics to {macro_path}")

    # Print summary
    logger.info(f"Mean macro F1: {macro_f1_df['macro_f1'].mean():.3f}")


# ============================================================================
# Benchmark 3: Property Estimation
# ============================================================================


def evaluate_benchmark_3():
    """Evaluate Benchmark 3: Property Estimation."""
    logger.info("Evaluating Benchmark 3: Property Estimation")

    # Load raw results
    df = load_raw_benchmark(3)
    if df is None:
        return

    # Compute error metrics
    df["error"] = df["predicted"] - df["ground_truth"]
    df["abs_error"] = df["error"].abs()

    # For HBD/HBA (integers), compute exact match
    df["exact_match"] = df["predicted"] == df["ground_truth"]

    # Save scored results
    scored_dir = RESULTS_DIR / "scored"
    ensure_dir(scored_dir)
    scored_path = scored_dir / "benchmark_3_scored.csv"
    df.to_csv(scored_path, index=False)
    logger.info(f"Saved scored results to {scored_path}")

    # Compute aggregate metrics
    metrics_rows = []

    for (representation, model, thinking, prop), group in df.groupby(
        ["representation", "model", "thinking", "property"]
    ):
        # Remove null predictions
        valid_group = group.dropna(subset=["predicted"])

        if len(valid_group) == 0:
            continue

        # MAE
        mae = valid_group["abs_error"].mean()

        # For continuous properties (logp, tpsa): compute Spearman correlation
        if prop in ["logp", "tpsa"]:
            spearman_rho, spearman_p = stats.spearmanr(
                valid_group["predicted"], valid_group["ground_truth"]
            )
            exact_match_acc = None
        else:  # For discrete properties (hbd, hba): compute exact match accuracy
            spearman_rho = None
            spearman_p = None
            exact_match_acc = valid_group["exact_match"].mean()

        metrics_rows.append(
            {
                "representation": representation,
                "model": model,
                "thinking": thinking,
                "property": prop,
                "mae": mae,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "exact_match_accuracy": exact_match_acc,
                "n_samples": len(valid_group),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    # Save aggregate metrics
    metrics_dir = RESULTS_DIR / "metrics"
    ensure_dir(metrics_dir)
    metrics_path = metrics_dir / "benchmark_3_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved aggregate metrics to {metrics_path}")

    # Print summary
    logger.info(f"Mean MAE: {metrics_df['mae'].mean():.3f}")


# ============================================================================
# Benchmark 4: Retrieval
# ============================================================================


def evaluate_benchmark_4():
    """Evaluate Benchmark 4: Retrieval."""
    logger.info("Evaluating Benchmark 4: Retrieval")

    # Load raw results
    df = load_raw_benchmark(4)
    if df is None:
        return

    # Compute correctness
    df["correct"] = df["predicted_letter"] == df["correct_letter"]

    # Save scored results
    scored_dir = RESULTS_DIR / "scored"
    ensure_dir(scored_dir)
    scored_path = scored_dir / "benchmark_4_scored.csv"
    df.to_csv(scored_path, index=False)
    logger.info(f"Saved scored results to {scored_path}")

    # Compute aggregate metrics
    metrics_rows = []

    for (representation, model, thinking), group in df.groupby(
        ["representation", "model", "thinking"]
    ):
        accuracy = group["correct"].mean()

        metrics_rows.append(
            {
                "representation": representation,
                "model": model,
                "thinking": thinking,
                "accuracy": accuracy,
                "n_samples": len(group),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    # Save aggregate metrics
    metrics_dir = RESULTS_DIR / "metrics"
    ensure_dir(metrics_dir)
    metrics_path = metrics_dir / "benchmark_4_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved aggregate metrics to {metrics_path}")

    # Print summary
    logger.info(f"Mean accuracy: {metrics_df['accuracy'].mean():.3f}")
    logger.info(f"Chance level: 0.25 (4-choice)")


# ============================================================================
# Benchmark 5: Isomer Discrimination
# ============================================================================


def evaluate_benchmark_5():
    """Evaluate Benchmark 5: Isomer Discrimination."""
    logger.info("Evaluating Benchmark 5: Isomer Discrimination")

    # Load raw results
    df = load_raw_benchmark(5)
    if df is None:
        return

    # ground_truth in the JSONL is wrong (mol_id_1 == mol_id_2 is always False).
    # Recompute: natural and stereoisomer pairs ARE isomers; test_pair pairs are not.
    df = df.copy()
    df["ground_truth"] = df["pair_type"].isin(["natural", "stereoisomer"])

    # Compute correctness
    df["correct"] = df["predicted"] == df["ground_truth"]

    # Save scored results
    scored_dir = RESULTS_DIR / "scored"
    ensure_dir(scored_dir)
    scored_path = scored_dir / "benchmark_5_scored.csv"
    df.to_csv(scored_path, index=False)
    logger.info(f"Saved scored results to {scored_path}")

    # Compute aggregate metrics (overall and by pair_type)
    metrics_rows = []

    for (representation, model, thinking), group in df.groupby(
        ["representation", "model", "thinking"]
    ):
        # Overall metrics
        tp = ((group["predicted"] == True) & (group["ground_truth"] == True)).sum()
        fp = ((group["predicted"] == True) & (group["ground_truth"] == False)).sum()
        tn = ((group["predicted"] == False) & (group["ground_truth"] == False)).sum()
        fn = ((group["predicted"] == False) & (group["ground_truth"] == True)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

        metrics_rows.append(
            {
                "representation": representation,
                "model": model,
                "thinking": thinking,
                "pair_type": "overall",
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_samples": len(group),
            }
        )

        # Breakdown by pair_type
        for pair_type, pair_group in group.groupby("pair_type"):
            pair_accuracy = pair_group["correct"].mean()

            metrics_rows.append(
                {
                    "representation": representation,
                    "model": model,
                    "thinking": thinking,
                    "pair_type": pair_type,
                    "accuracy": pair_accuracy,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "n_samples": len(pair_group),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)

    # Save aggregate metrics
    metrics_dir = RESULTS_DIR / "metrics"
    ensure_dir(metrics_dir)
    metrics_path = metrics_dir / "benchmark_5_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved aggregate metrics to {metrics_path}")

    # Print summary
    overall_metrics = metrics_df[metrics_df["pair_type"] == "overall"]
    logger.info(f"Mean accuracy: {overall_metrics['accuracy'].mean():.3f}")
    logger.info(f"Mean F1: {overall_metrics['f1'].mean():.3f}")


# ============================================================================
# Benchmark 6: Generation
# ============================================================================


def evaluate_benchmark_6():
    """Evaluate Benchmark 6: Generation."""
    logger.info("Evaluating Benchmark 6: Generation")

    # Load raw results
    df = load_raw_benchmark(6)
    if df is None:
        return

    # Load ground truth — use cutdown set, then fall back to full chebi20_test
    test_df = pd.read_csv(DATA_DIR / "chebi_prepared_500.csv")
    test_df["molecule_id"] = test_df["CID"].astype(str)
    test_df["smiles"] = test_df["SMILES"]
    ground_truth_map = dict(zip(test_df["molecule_id"], test_df["smiles"]))
    full_test = DATA_DIR / "chebi20_test.csv"
    if full_test.exists():
        full_df = pd.read_csv(full_test)
        full_df["molecule_id"] = full_df["CID"].astype(str)
        for _, row in full_df.iterrows():
            if row["molecule_id"] not in ground_truth_map:
                ground_truth_map[row["molecule_id"]] = row["SMILES"]

    # Decode generated molecules and compute metrics
    logger.info("Decoding generated molecules (parallel, 20 workers)...")

    def _decode_row(row):
        from rdkit import Chem

        # Re-extract generated_string from raw_response using the current parser.
        # This recovers answers for models (e.g. phi-4 base) whose prose-style output
        # was not correctly parsed at inference time.
        raw = row.get("raw_response") or ""
        repr_name = row["representation"]
        if raw:
            _, cleaned = parsing.extract_answer_from_thinking_response(raw)
            # Try JSON schema extraction on the cleaned answer
            try:
                json_obj = json.loads(cleaned)
                if "molecule" in json_obj:
                    cleaned = json_obj["molecule"]
                elif "atoms" in json_obj and "bonds" in json_obj:
                    cleaned = json.dumps(json_obj)
            except (json.JSONDecodeError, TypeError):
                pass
            re_extracted = parsing.extract_molecule_string(cleaned, repr_name)
        else:
            re_extracted = None
        generated_str = re_extracted or row.get("generated_string")

        canonical_generated, is_valid = decode_generated_molecule(
            generated_str, repr_name
        )

        ground_truth_smiles = ground_truth_map.get(str(row["molecule_id"]))
        if ground_truth_smiles:
            gt_mol = Chem.MolFromSmiles(ground_truth_smiles)
            canonical_ground_truth = (
                Chem.MolToSmiles(gt_mol, canonical=True, isomericSmiles=False)
                if gt_mol
                else None
            )
        else:
            canonical_ground_truth = None

        exact_match = (
            canonical_generated == canonical_ground_truth
            if is_valid and canonical_ground_truth
            else False
        )

        tanimoto = None
        maccs_tanimoto = None
        if is_valid and canonical_ground_truth:
            tanimoto = compute_tanimoto_similarity(
                canonical_generated, canonical_ground_truth, "morgan"
            )
            maccs_tanimoto = compute_tanimoto_similarity(
                canonical_generated, canonical_ground_truth, "maccs"
            )

        complexity = compute_molecule_complexity(ground_truth_smiles)

        return {
            **row.to_dict(),
            "valid": is_valid,
            "canonical_generated": canonical_generated,
            "canonical_ground_truth": canonical_ground_truth,
            "exact_match": exact_match,
            "tanimoto": tanimoto,
            "maccs_tanimoto": maccs_tanimoto,
            **complexity,
        }

    rows_list = [row for _, row in df.iterrows()]
    results = [None] * len(rows_list)

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_idx = {executor.submit(_decode_row, row): i for i, row in enumerate(rows_list)}
        for future in tqdm(as_completed(future_to_idx), total=len(rows_list), desc="Decoding"):
            idx = future_to_idx[future]
            results[idx] = future.result()

    # Flush IUPAC cache after all parallel work is done
    import utils.representations as _repr_mod
    if _repr_mod._iupac_cache_dirty:
        _repr_mod._save_iupac_cache()

    scored_df = pd.DataFrame(results)

    # Save scored results
    scored_dir = RESULTS_DIR / "scored"
    ensure_dir(scored_dir)
    scored_path = scored_dir / "benchmark_6_scored.csv"
    scored_df.to_csv(scored_path, index=False)
    logger.info(f"Saved scored results to {scored_path}")

    # Compute aggregate metrics
    metrics_rows = []

    for (representation, model, thinking), group in scored_df.groupby(
        ["representation", "model", "thinking"]
    ):
        validity_rate = group["valid"].mean()
        exact_match_rate = group["exact_match"].mean()

        valid_group = group[group["valid"] == True]
        mean_tanimoto = valid_group["tanimoto"].mean() if len(valid_group) > 0 else 0.0
        mean_maccs_tanimoto = (
            valid_group["maccs_tanimoto"].mean() if len(valid_group) > 0 else 0.0
        )

        # Composite score: weighted combination of all generation metrics
        composite_score = (
            0.2 * validity_rate
            + 0.4 * mean_tanimoto
            + 0.2 * mean_maccs_tanimoto
            + 0.2 * exact_match_rate
        )

        metrics_rows.append(
            {
                "representation": representation,
                "model": model,
                "thinking": thinking,
                "validity_rate": validity_rate,
                "exact_match_rate": exact_match_rate,
                "mean_tanimoto": mean_tanimoto,
                "mean_maccs_tanimoto": mean_maccs_tanimoto,
                "composite_score": composite_score,
                "n_samples": len(group),
                "n_valid": len(valid_group),
            }
        )


    metrics_df = pd.DataFrame(metrics_rows)

    # Save aggregate metrics
    metrics_dir = RESULTS_DIR / "metrics"
    ensure_dir(metrics_dir)
    metrics_path = metrics_dir / "benchmark_6_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved aggregate metrics to {metrics_path}")

    # Print summary
    logger.info(f"Mean validity rate: {metrics_df['validity_rate'].mean():.3f}")
    logger.info(f"Mean exact match rate: {metrics_df['exact_match_rate'].mean():.3f}")
    logger.info(f"Mean Tanimoto: {metrics_df['mean_tanimoto'].mean():.3f}")


# ============================================================================
# Benchmark 7: Completion
# ============================================================================


def evaluate_benchmark_7():
    """Evaluate Benchmark 7: Completion."""
    logger.info("Evaluating Benchmark 7: Completion")

    # Load raw results
    df = load_raw_benchmark(7)
    if df is None:
        return

    # Load ground truth from the ZINC dataset (benchmark 7 uses ZINC IDs)
    test_df = pd.read_csv(DATA_DIR / "zinc_prepared_500.csv")
    ground_truth_map = dict(zip(test_df["molecule_id"], test_df["smiles"]))

    # Decode completed molecules and compute metrics
    logger.info("Decoding completed molecules (parallel, 20 workers)...")

    def _decode_b7_row(row):
        from rdkit import Chem

        # Re-extract generated_string from raw_response using the current parser
        raw = row.get("raw_response") or ""
        repr_name = row["representation"]
        if raw:
            _, cleaned = parsing.extract_answer_from_thinking_response(raw)
            try:
                json_obj = json.loads(cleaned)
                if "molecule" in json_obj:
                    cleaned = json_obj["molecule"]
                elif "atoms" in json_obj and "bonds" in json_obj:
                    cleaned = json.dumps(json_obj)
            except (json.JSONDecodeError, TypeError):
                pass
            re_extracted = parsing.extract_molecule_string(cleaned, repr_name)
        else:
            re_extracted = None
        generated_str = re_extracted or row.get("generated_string")

        canonical_generated, is_valid = decode_generated_molecule(generated_str, repr_name)

        ground_truth_smiles = ground_truth_map.get(row["molecule_id"])
        if ground_truth_smiles:
            gt_mol = Chem.MolFromSmiles(ground_truth_smiles)
            canonical_ground_truth = (
                Chem.MolToSmiles(gt_mol, canonical=True, isomericSmiles=False)
                if gt_mol else None
            )
        else:
            canonical_ground_truth = None

        recovery = (
            canonical_generated == canonical_ground_truth
            if is_valid and canonical_ground_truth else False
        )

        tanimoto = None
        if is_valid and canonical_ground_truth:
            tanimoto = compute_tanimoto_similarity(
                canonical_generated, canonical_ground_truth, "morgan"
            )

        return {
            **row.to_dict(),
            "valid": is_valid,
            "canonical_generated": canonical_generated,
            "canonical_ground_truth": canonical_ground_truth,
            "recovery": recovery,
            "tanimoto": tanimoto,
        }

    rows_list = [row for _, row in df.iterrows()]
    results = [None] * len(rows_list)

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_idx = {executor.submit(_decode_b7_row, row): i for i, row in enumerate(rows_list)}
        for future in tqdm(as_completed(future_to_idx), total=len(rows_list), desc="Decoding"):
            idx = future_to_idx[future]
            results[idx] = future.result()

    import utils.representations as _repr_mod
    if _repr_mod._iupac_cache_dirty:
        _repr_mod._save_iupac_cache()

    scored_df = pd.DataFrame(results)

    # Save scored results
    scored_dir = RESULTS_DIR / "scored"
    ensure_dir(scored_dir)
    scored_path = scored_dir / "benchmark_7_scored.csv"
    scored_df.to_csv(scored_path, index=False)
    logger.info(f"Saved scored results to {scored_path}")

    # Compute aggregate metrics
    metrics_rows = []

    for (representation, model, thinking), group in scored_df.groupby(
        ["representation", "model", "thinking"]
    ):
        validity_rate = group["valid"].mean()
        recovery_rate = group["recovery"].mean()

        valid_group = group[group["valid"] == True]
        mean_tanimoto = valid_group["tanimoto"].mean() if len(valid_group) > 0 else 0.0

        metrics_rows.append(
            {
                "representation": representation,
                "model": model,
                "thinking": thinking,
                "validity_rate": validity_rate,
                "recovery_rate": recovery_rate,
                "mean_tanimoto": mean_tanimoto,
                "n_samples": len(group),
                "n_valid": len(valid_group),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    # Save aggregate metrics
    metrics_dir = RESULTS_DIR / "metrics"
    ensure_dir(metrics_dir)
    metrics_path = metrics_dir / "benchmark_7_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved aggregate metrics to {metrics_path}")

    # Print summary
    logger.info(f"Mean validity rate: {metrics_df['validity_rate'].mean():.3f}")
    logger.info(f"Mean recovery rate: {metrics_df['recovery_rate'].mean():.3f}")
    logger.info(f"Mean Tanimoto: {metrics_df['mean_tanimoto'].mean():.3f}")


# ============================================================================
# Benchmark 9: Tautomer Recognition
# ============================================================================


def _compute_discrimination_metrics(df, benchmark_num, benchmark_name, extra_breakdown_col=None):
    """
    Shared evaluation logic for Yes/No pair-discrimination benchmarks (B5, B9, B10).

    Computes accuracy / precision / recall / F1 overall and by pair_type.
    Optionally also breaks down by extra_breakdown_col (e.g. tautomer_class).

    Returns metrics_df.
    """
    # Map string "Yes"/"No" ground truth to bool for consistency with B5
    if df["ground_truth"].dtype == object:
        df = df.copy()
        df["ground_truth_bool"] = df["ground_truth"].map({"Yes": True, "No": False})
    else:
        df = df.copy()
        df["ground_truth_bool"] = df["ground_truth"].astype(bool)

    if df["predicted"].dtype == object:
        df["predicted_bool"] = df["predicted"].map({"Yes": True, "No": False, True: True, False: False})
    else:
        df["predicted_bool"] = df["predicted"].astype(bool)

    df["correct"] = df["predicted_bool"] == df["ground_truth_bool"]

    metrics_rows = []

    for (representation, model, thinking), group in df.groupby(
        ["representation", "model", "thinking"]
    ):
        # Overall metrics
        tp = ((group["predicted_bool"] == True) & (group["ground_truth_bool"] == True)).sum()
        fp = ((group["predicted_bool"] == True) & (group["ground_truth_bool"] == False)).sum()
        tn = ((group["predicted_bool"] == False) & (group["ground_truth_bool"] == False)).sum()
        fn = ((group["predicted_bool"] == False) & (group["ground_truth_bool"] == True)).sum()

        # precision is undefined (NaN) when the model never predicts positive (tp+fp=0);
        # recall is 0 when there are no positive ground-truth examples (tp+fn=0).
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (pd.notna(precision) and (precision + recall) > 0) else float("nan") if pd.isna(precision) else 0.0
        )
        accuracy = (tp + tn) / len(group) if len(group) > 0 else 0.0

        metrics_rows.append({
            "representation": representation,
            "model": model,
            "thinking": thinking,
            "pair_type": "overall",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_samples": len(group),
        })

        # Breakdown by pair_type
        for pair_type, pt_group in group.groupby("pair_type"):
            metrics_rows.append({
                "representation": representation,
                "model": model,
                "thinking": thinking,
                "pair_type": pair_type,
                "accuracy": pt_group["correct"].mean(),
                "precision": None,
                "recall": None,
                "f1": None,
                "n_samples": len(pt_group),
            })

        # Optional additional breakdown column (e.g. tautomer_class or ionizable_group)
        if extra_breakdown_col and extra_breakdown_col in group.columns:
            for col_val, col_group in group.groupby(extra_breakdown_col):
                if str(col_val) == "none":
                    continue
                metrics_rows.append({
                    "representation": representation,
                    "model": model,
                    "thinking": thinking,
                    "pair_type": f"{extra_breakdown_col}:{col_val}",
                    "accuracy": col_group["correct"].mean(),
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "n_samples": len(col_group),
                })

    return pd.DataFrame(metrics_rows), df


def evaluate_benchmark_9():
    """Evaluate Benchmark 9: Tautomer Recognition."""
    logger.info("Evaluating Benchmark 9: Tautomer Recognition")

    df = load_raw_benchmark(9)
    if df is None:
        return

    metrics_df, scored_df = _compute_discrimination_metrics(
        df, benchmark_num=9, benchmark_name="tautomer_recognition",
        extra_breakdown_col="tautomer_class",
    )

    # Save scored results
    scored_dir = RESULTS_DIR / "scored"
    ensure_dir(scored_dir)
    scored_df.to_csv(scored_dir / "benchmark_9_scored.csv", index=False)
    logger.info(f"Saved scored results to {scored_dir / 'benchmark_9_scored.csv'}")

    # Save aggregate metrics
    metrics_dir = RESULTS_DIR / "metrics"
    ensure_dir(metrics_dir)
    metrics_path = metrics_dir / "benchmark_9_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved aggregate metrics to {metrics_path}")

    # Print summary
    overall = metrics_df[metrics_df["pair_type"] == "overall"]
    logger.info(f"Mean accuracy: {overall['accuracy'].mean():.3f}")
    logger.info(f"Mean F1: {overall['f1'].mean():.3f}")


# ============================================================================
# Benchmark 10: Protonation State Recognition
# ============================================================================


def evaluate_benchmark_10():
    """Evaluate Benchmark 10: Protonation State Recognition."""
    logger.info("Evaluating Benchmark 10: Protonation State Recognition")

    df = load_raw_benchmark(10)
    if df is None:
        return

    metrics_df, scored_df = _compute_discrimination_metrics(
        df, benchmark_num=10, benchmark_name="protonation_recognition",
        extra_breakdown_col="ionizable_group",
    )

    # Save scored results
    scored_dir = RESULTS_DIR / "scored"
    ensure_dir(scored_dir)
    scored_df.to_csv(scored_dir / "benchmark_10_scored.csv", index=False)
    logger.info(f"Saved scored results to {scored_dir / 'benchmark_10_scored.csv'}")

    # Save aggregate metrics
    metrics_dir = RESULTS_DIR / "metrics"
    ensure_dir(metrics_dir)
    metrics_path = metrics_dir / "benchmark_10_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved aggregate metrics to {metrics_path}")

    # Print summary
    overall = metrics_df[metrics_df["pair_type"] == "overall"]
    logger.info(f"Mean accuracy: {overall['accuracy'].mean():.3f}")
    logger.info(f"Mean F1: {overall['f1'].mean():.3f}")


# ============================================================================
# Cross-Cutting Analyses
# ============================================================================


def compute_cross_cutting_analyses():
    """Compute cross-cutting analyses."""
    logger.info("Computing cross-cutting analyses")

    cross_dir = RESULTS_DIR / "cross"
    ensure_dir(cross_dir)

    # Load all benchmark metrics
    metrics_dir = RESULTS_DIR / "metrics"

    try:
        b1 = pd.read_csv(metrics_dir / "benchmark_1_metrics.csv")
        b2 = pd.read_csv(metrics_dir / "benchmark_2_macro_metrics.csv")
        b3 = pd.read_csv(metrics_dir / "benchmark_3_metrics.csv")
        b4 = pd.read_csv(metrics_dir / "benchmark_4_metrics.csv")
        b5 = pd.read_csv(metrics_dir / "benchmark_5_metrics.csv")
        b6 = pd.read_csv(metrics_dir / "benchmark_6_metrics.csv")
        # b7 = pd.read_csv(metrics_dir / "benchmark_7_metrics.csv")
    except FileNotFoundError as e:
        logger.warning(f"Missing benchmark metrics: {e}")
        return

    # ========================================================================
    # 1. Comprehension vs Generation Correlation
    # ========================================================================
    logger.info("Computing comprehension vs generation correlation...")

    # Normalize benchmarks to 0-1
    # B1: accuracy already 0-1
    b1_norm = b1[["representation", "model", "thinking", "accuracy"]].copy()
    b1_norm.rename(columns={"accuracy": "b1_score"}, inplace=True)

    # B2: macro_f1 already 0-1
    b2_norm = b2[["representation", "model", "thinking", "macro_f1"]].copy()
    b2_norm.rename(columns={"macro_f1": "b2_score"}, inplace=True)

    # B3: normalize MAE to 0-1 (need overall metric - take mean across properties)
    b3_overall = (
        b3.groupby(["representation", "model", "thinking"])["mae"].mean().reset_index()
    )
    max_mae = b3_overall["mae"].max()
    b3_overall["b3_score"] = 1 - (b3_overall["mae"] / max_mae)
    b3_norm = b3_overall[["representation", "model", "thinking", "b3_score"]]

    # B4: accuracy already 0-1
    b4_norm = b4[["representation", "model", "thinking", "accuracy"]].copy()
    b4_norm.rename(columns={"accuracy": "b4_score"}, inplace=True)

    # B5: f1 already 0-1 (take overall)
    b5_overall = b5[b5["pair_type"] == "overall"][
        ["representation", "model", "thinking", "f1"]
    ].copy()
    b5_overall.rename(columns={"f1": "b5_score"}, inplace=True)

    # B6: composite_score already 0-1
    b6_norm = b6[["representation", "model", "thinking", "composite_score"]].copy()
    b6_norm.rename(columns={"composite_score": "b6_score"}, inplace=True)

    # Merge all
    corr_df = b1_norm
    for df in [b2_norm, b3_norm, b4_norm, b5_overall, b6_norm]:
        corr_df = corr_df.merge(df, on=["representation", "model", "thinking"])

    # Compute mean comprehension (B1-B5)
    corr_df["mean_comprehension"] = corr_df[
        ["b1_score", "b2_score", "b3_score", "b4_score", "b5_score"]
    ].mean(axis=1)

    # Mean generation is B6 composite score
    corr_df["mean_generation"] = corr_df["b6_score"]

    # Save
    corr_path = cross_dir / "comprehension_vs_generation.csv"
    corr_df.to_csv(corr_path, index=False)
    logger.info(f"Saved comprehension vs generation to {corr_path}")

    # Compute correlation
    valid_corr = corr_df.dropna(subset=["mean_comprehension", "mean_generation"])
    if len(valid_corr) > 0:
        pearson_r, pearson_p = stats.pearsonr(
            valid_corr["mean_comprehension"], valid_corr["mean_generation"]
        )
        spearman_r, spearman_p = stats.spearmanr(
            valid_corr["mean_comprehension"], valid_corr["mean_generation"]
        )
        logger.info(f"Pearson r={pearson_r:.3f}, p={pearson_p:.3f}")
        logger.info(f"Spearman r={spearman_r:.3f}, p={spearman_p:.3f}")

    # ========================================================================
    # 2. Thinking Ablation
    # ========================================================================
    logger.info("Computing thinking ablation...")

    # For each benchmark, compute delta = thinking_on - thinking_off
    ablation_rows = []

    # Helper to compute delta for a benchmark
    def compute_delta(df, metric_col, benchmark_name):
        rows = []
        for (representation, model), group in df.groupby(["representation", "model"]):
            thinking_on = group[group["thinking"] == True]
            thinking_off = group[group["thinking"] == False]

            if len(thinking_on) > 0 and len(thinking_off) > 0:
                on_score = thinking_on[metric_col].iloc[0]
                off_score = thinking_off[metric_col].iloc[0]
                delta = on_score - off_score

                rows.append(
                    {
                        "representation": representation,
                        "model": model,
                        "benchmark": benchmark_name,
                        "thinking_on_score": on_score,
                        "thinking_off_score": off_score,
                        "delta": delta,
                    }
                )
        return rows

    ablation_rows.extend(compute_delta(b1, "accuracy", "B1_atom_counting"))
    ablation_rows.extend(compute_delta(b2, "macro_f1", "B2_functional_groups"))

    # B3: use overall MAE
    b3_overall = (
        b3.groupby(["representation", "model", "thinking"])["mae"].mean().reset_index()
    )
    ablation_rows.extend(compute_delta(b3_overall, "mae", "B3_properties"))

    ablation_rows.extend(compute_delta(b4, "accuracy", "B4_retrieval"))

    b5_overall = b5[b5["pair_type"] == "overall"]
    ablation_rows.extend(compute_delta(b5_overall, "f1", "B5_isomer_discrimination"))

    ablation_rows.extend(
        compute_delta(b6, "exact_match_rate", "B6_generation_exact_match")
    )
    ablation_rows.extend(compute_delta(b6, "validity_rate", "B6_generation_validity"))

    # ablation_rows.extend(compute_delta(b7, "recovery_rate", "B7_completion_recovery"))
    # ablation_rows.extend(compute_delta(b7, "validity_rate", "B7_completion_validity"))

    ablation_df = pd.DataFrame(ablation_rows)

    # Save
    ablation_path = cross_dir / "thinking_ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)
    logger.info(f"Saved thinking ablation to {ablation_path}")

    # ========================================================================
    # 3. Representation Rankings
    # ========================================================================
    logger.info("Computing representation rankings...")

    # For each benchmark, rank representations (averaged across models and thinking)
    ranking_rows = []

    def rank_representations(df, metric_col, benchmark_name, higher_is_better=True):
        # Average across model and thinking
        avg = (
            df.groupby("representation")[metric_col].mean().reset_index()
        )

        # Rank
        avg["rank"] = avg[metric_col].rank(ascending=not higher_is_better)

        for _, row in avg.iterrows():
            ranking_rows.append(
                {
                    "representation": row["representation"],
                    "benchmark": benchmark_name,
                    "score": row[metric_col],
                    "rank": row["rank"],
                }
            )

    rank_representations(b1, "accuracy", "B1_atom_counting", higher_is_better=True)
    rank_representations(b2, "macro_f1", "B2_functional_groups", higher_is_better=True)
    rank_representations(b3_overall, "mae", "B3_properties", higher_is_better=False)  # Lower MAE is better
    rank_representations(b4, "accuracy", "B4_retrieval", higher_is_better=True)
    rank_representations(b5_overall, "f1", "B5_isomer_discrimination", higher_is_better=True)
    rank_representations(b6, "composite_score", "B6_generation", higher_is_better=True)
    # rank_representations(b7, "recovery_rate", "B7_completion", higher_is_better=True)

    ranking_df = pd.DataFrame(ranking_rows)

    # Compute mean rank per representation
    mean_ranks = ranking_df.groupby("representation")["rank"].mean().reset_index()
    mean_ranks.rename(columns={"rank": "mean_rank"}, inplace=True)
    mean_ranks = mean_ranks.sort_values("mean_rank")

    # Save
    ranking_path = cross_dir / "representation_rankings.csv"
    ranking_df.to_csv(ranking_path, index=False)
    logger.info(f"Saved representation rankings to {ranking_path}")

    mean_rank_path = cross_dir / "representation_mean_ranks.csv"
    mean_ranks.to_csv(mean_rank_path, index=False)
    logger.info(f"Saved mean ranks to {mean_rank_path}")

    logger.info("\nMean ranks:")
    for _, row in mean_ranks.iterrows():
        logger.info(f"  {row['representation']}: {row['mean_rank']:.2f}")


# ============================================================================
# Main
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate benchmark results")

    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        help='Benchmark to evaluate (1-7, 9, 10, or "all")',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse benchmark selection
    if args.benchmark == "all":
        benchmark_nums = list(range(1, 7)) + [9, 10]
    else:
        benchmark_nums = [int(args.benchmark)]

    # Map to evaluation functions
    evaluators = {
        1: evaluate_benchmark_1,
        2: evaluate_benchmark_2,
        3: evaluate_benchmark_3,
        4: evaluate_benchmark_4,
        5: evaluate_benchmark_5,
        6: evaluate_benchmark_6,
        # 7: evaluate_benchmark_7,
        9: evaluate_benchmark_9,
        10: evaluate_benchmark_10,
    }

    # Run evaluations
    for benchmark_num in benchmark_nums:
        try:
            evaluators[benchmark_num]()
        except Exception as e:
            logger.error(f"Error evaluating benchmark {benchmark_num}: {e}")
            import traceback

            traceback.print_exc()

    # Compute cross-cutting analyses if all benchmarks evaluated
    if args.benchmark == "all":
        try:
            compute_cross_cutting_analyses()
        except Exception as e:
            logger.error(f"Error computing cross-cutting analyses: {e}")
            import traceback

            traceback.print_exc()

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
