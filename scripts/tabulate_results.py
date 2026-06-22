"""
06_tabulate_results.py - Tabulate benchmark results into CSV and LaTeX tables

Reads raw JSONL files (run 1 only), scores them, and produces:
  1. results_small/tables/full_results.csv   (per model/thinking/benchmark/representation)
  2. results_small/tables/summary_table.tex  (one table per benchmark, per representation)

NO model/GPU dependencies - pure pandas + RDKit + scipy

Usage:
    source /tier1/home/arunraja/rl/.venv/bin/activate
    python scripts/06_tabulate_results.py
"""

import importlib.util
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BENCHMARK_DISPLAY_NAMES,
    BENCHMARK_NUM_TO_NAME,
    MODEL_NAMES,
    REPRESENTATIONS,
    REPR_DISPLAY_NAMES,
    RESULTS_DIR,
    DATA_DIR,
)
from utils import parsing, representations as repr_mod

# Import helpers from evaluate.py
_spec = importlib.util.spec_from_file_location(
    "evaluate_03", Path(__file__).parent / "evaluate.py"
)
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
decode_generated_molecule = _eval_mod.decode_generated_molecule
compute_tanimoto_similarity = _eval_mod.compute_tanimoto_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

RAW_DIR = RESULTS_DIR / "raw"
TABLE_DIR = RESULTS_DIR / "tables"

# Additional raw directories for API models
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIRS = [
    RAW_DIR,
    PROJECT_ROOT / "results_gpt" / "raw",
    PROJECT_ROOT / "results_claude" / "raw",
]

# Display names for API models (not in scripts/config.py MODEL_NAMES)
API_MODEL_NAMES = {
    "gpt-5.4-mini": "GPT-5.4-mini",
    "claude-haiku-4.5": "Claude-Haiku-4.5",
}

# Benchmarks to evaluate (B7 excluded)
BENCH_NUMS = [1, 2, 3, 4, 5, 6, 9, 10]

BENCH_PRIMARY_METRIC = {
    1: "accuracy",
    2: "macro_f1",
    3: "spearman_rho",  # per-property; see score_b3
    4: "accuracy",
    5: "accuracy",
    6: "composite_score",
    9: "accuracy",
    10: "accuracy",
}

# Model display order (by family then size)
MODEL_ORDER = [
    "qwen3-4b-thinking-2507",
    "qwen3-30b-a3b-thinking-2507",
    "phi-4",
    "phi-4-reasoning",
    "phi-4-reasoning-plus",
    "qwen2.5-14b",
    "chemdfm-v2.0-14b",
    "chemdfm-r-14b",
    "mistral-small-24b",
    "ether0-24b",
    "olmo-3.1-32b-instruct",
    "olmo-3.1-32b-think",
    "gpt-5.4-mini",
    "claude-haiku-4.5",
]

# Models that are non-reasoning (only have thinking=off, never thinking=on)
NON_REASONING_MODELS = {
    "qwen2.5-14b",
    "mistral-small-24b",
}

# Models to skip entirely
SKIP_MODELS = {
    "olmo-3-1125-32b",
    "qwen3-235b-a22b-thinking-2507",
    "qwen3-4b-thinking-2507-moljson",
    "qwen3-4b-thinking-2507-moljson-v2",
    "gpt-5.5",
    "claude-sonnet-4.6",
}


# ============================================================================
# File Discovery
# ============================================================================


def discover_run1_files() -> Dict[Tuple[int, str, str], Path]:
    """Find all run1 JSONL files across all raw dirs, return dict of (bench_num, model_id, thinking) -> path."""
    pattern = re.compile(r"^benchmark_(\d+)_(.+)_thinking_(on|off)_run1\.jsonl$")
    files = {}
    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            continue
        for f in sorted(raw_dir.glob("benchmark_*_run1.jsonl")):
            m = pattern.match(f.name)
            if m:
                bench_num = int(m.group(1))
                model_id = m.group(2)
                thinking = m.group(3)
                if bench_num in BENCH_NUMS and model_id not in SKIP_MODELS:
                    files[(bench_num, model_id, thinking)] = f
    return files


def load_jsonl(filepath: Path) -> pd.DataFrame:
    """Load a single JSONL file into a DataFrame."""
    rows = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "thinking" in df.columns:
        df["thinking"] = df["thinking"].astype(bool)
    return df


# ============================================================================
# Scoring Functions
# ============================================================================


def score_b1(df: pd.DataFrame) -> List[dict]:
    """B1: Atom Counting — accuracy."""
    df["correct"] = (df["predicted"] == df["ground_truth"]).astype(float)
    rows = []
    for (rep, model, thinking), g in df.groupby(["representation", "model", "thinking"]):
        scores = g["correct"].values
        rows.append({
            "representation": rep, "model": model, "thinking": thinking,
            "metric_name": "accuracy",
            "value": float(np.mean(scores)),
            "sem": float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0,
            "n_questions": len(scores),
        })
    return rows


def score_b2(df: pd.DataFrame) -> List[dict]:
    """B2: Functional Groups — macro-F1 across 5 groups."""
    df["correct"] = (df["predicted"] == df["ground_truth"])
    rows = []
    for (rep, model, thinking), g in df.groupby(["representation", "model", "thinking"]):
        # Compute F1 per functional group
        group_f1s = []
        for group_name, gg in g.groupby("group"):
            tp = ((gg["predicted"] == True) & (gg["ground_truth"] == True)).sum()
            fp = ((gg["predicted"] == True) & (gg["ground_truth"] == False)).sum()
            fn = ((gg["predicted"] == False) & (gg["ground_truth"] == True)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            group_f1s.append(f1)
        macro_f1 = float(np.mean(group_f1s)) if group_f1s else 0.0

        # Bootstrap std of macro-F1
        rng = np.random.default_rng(42)
        molecules = g["molecule_id"].unique()
        n_boot = 1000
        boot_f1s = []
        for _ in range(n_boot):
            sample_mols = rng.choice(molecules, size=len(molecules), replace=True)
            boot_g = g[g["molecule_id"].isin(sample_mols)]
            gf1s = []
            for gn, gg in boot_g.groupby("group"):
                tp = ((gg["predicted"] == True) & (gg["ground_truth"] == True)).sum()
                fp = ((gg["predicted"] == True) & (gg["ground_truth"] == False)).sum()
                fn = ((gg["predicted"] == False) & (gg["ground_truth"] == True)).sum()
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                gf1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
            boot_f1s.append(np.mean(gf1s))

        rows.append({
            "representation": rep, "model": model, "thinking": thinking,
            "metric_name": "macro_f1",
            "value": macro_f1,
            "sem": float(np.std(boot_f1s, ddof=1)),
            "n_questions": len(g),
        })
    return rows


def score_b3(df: pd.DataFrame) -> List[dict]:
    """B3: Property Estimation — per-property metrics.

    logp, tpsa: Spearman ρ and MAE
    hbd, hba:   Spearman ρ, exact-match accuracy, and MAE
    """
    rows = []
    rng = np.random.default_rng(42)
    n_boot = 1000

    for (rep, model, thinking), g in df.groupby(["representation", "model", "thinking"]):
        for prop in ["logp", "tpsa", "hbd", "hba"]:
            pg = g[g["property"] == prop].dropna(subset=["predicted"])
            if len(pg) == 0:
                continue

            pred = pd.to_numeric(pg["predicted"], errors="coerce")
            gt = pd.to_numeric(pg["ground_truth"], errors="coerce")
            valid = pred.notna() & gt.notna()
            pred, gt = pred[valid], gt[valid]
            if len(pred) == 0:
                continue

            molecules = pg.loc[valid.values, "molecule_id"].unique() if "molecule_id" in pg.columns else np.arange(len(pred))
            n_q = len(pred)
            metric_name_prefix = f"{prop}"

            # --- Spearman ρ ---
            if len(pred) >= 3:
                rho, _ = stats.spearmanr(pred, gt)
                boot_rhos = []
                for _ in range(n_boot):
                    idx = rng.choice(len(pred), size=len(pred), replace=True)
                    if len(np.unique(pred.values[idx])) < 2:
                        continue
                    r, _ = stats.spearmanr(pred.values[idx], gt.values[idx])
                    boot_rhos.append(r)
                sem_rho = float(np.std(boot_rhos, ddof=1)) if boot_rhos else 0.0
                rows.append({
                    "representation": rep, "model": model, "thinking": thinking,
                    "metric_name": f"{metric_name_prefix}_spearman_rho",
                    "value": float(rho),
                    "sem": sem_rho,
                    "n_questions": n_q,
                })

            # --- MAE ---
            mae = float(np.mean(np.abs(pred.values - gt.values)))
            boot_maes = []
            for _ in range(n_boot):
                idx = rng.choice(len(pred), size=len(pred), replace=True)
                boot_maes.append(np.mean(np.abs(pred.values[idx] - gt.values[idx])))
            rows.append({
                "representation": rep, "model": model, "thinking": thinking,
                "metric_name": f"{metric_name_prefix}_mae",
                "value": mae,
                "sem": float(np.std(boot_maes, ddof=1)),
                "n_questions": n_q,
            })

            # --- Exact match (hbd/hba only) ---
            if prop in ["hbd", "hba"]:
                exact = float((pred.values == gt.values).mean())
                boot_exacts = []
                for _ in range(n_boot):
                    idx = rng.choice(len(pred), size=len(pred), replace=True)
                    boot_exacts.append(float((pred.values[idx] == gt.values[idx]).mean()))
                rows.append({
                    "representation": rep, "model": model, "thinking": thinking,
                    "metric_name": f"{metric_name_prefix}_exact_match",
                    "value": exact,
                    "sem": float(np.std(boot_exacts, ddof=1)),
                    "n_questions": n_q,
                })

    return rows


def score_b4(df: pd.DataFrame) -> List[dict]:
    """B4: Retrieval — accuracy."""
    df["correct"] = (df["predicted_letter"] == df["correct_letter"]).astype(float)
    rows = []
    for (rep, model, thinking), g in df.groupby(["representation", "model", "thinking"]):
        scores = g["correct"].values
        rows.append({
            "representation": rep, "model": model, "thinking": thinking,
            "metric_name": "accuracy",
            "value": float(np.mean(scores)),
            "sem": float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0,
            "n_questions": len(scores),
        })
    return rows


def score_b5(df: pd.DataFrame) -> List[dict]:
    """B5: Isomer Discrimination — accuracy (recompute ground_truth from pair_type)."""
    df = df.copy()
    df["ground_truth"] = df["pair_type"].isin(["natural", "stereoisomer"])
    df["correct"] = (df["predicted"] == df["ground_truth"]).astype(float)
    rows = []
    for (rep, model, thinking), g in df.groupby(["representation", "model", "thinking"]):
        scores = g["correct"].values
        rows.append({
            "representation": rep, "model": model, "thinking": thinking,
            "metric_name": "accuracy",
            "value": float(np.mean(scores)),
            "sem": float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0,
            "n_questions": len(scores),
        })
    return rows


def score_b6(df: pd.DataFrame) -> List[dict]:
    """B6: Generation — multiple metrics."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import RDKFingerprint
    import fcd

    # Load ground truth — use cutdown set, then fall back to full chebi20_test
    test_df = pd.read_csv(DATA_DIR / "chebi_prepared_500.csv")
    test_df["molecule_id"] = test_df["CID"].astype(str)
    ground_truth_map = dict(zip(test_df["molecule_id"], test_df["SMILES"]))
    full_test = DATA_DIR / "chebi20_test.csv"
    if full_test.exists():
        full_df = pd.read_csv(full_test)
        full_df["molecule_id"] = full_df["CID"].astype(str)
        # Only add molecules not already in the cutdown set
        for _, row in full_df.iterrows():
            if row["molecule_id"] not in ground_truth_map:
                ground_truth_map[row["molecule_id"]] = row["SMILES"]

    def _compute_rdkit_tanimoto(smi1, smi2):
        """RDKit topological fingerprint Tanimoto."""
        try:
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)
            if mol1 is None or mol2 is None:
                return None
            fp1 = RDKFingerprint(mol1)
            fp2 = RDKFingerprint(mol2)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception:
            return None

    def _process_row(row):
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

        canonical_gen, is_valid = decode_generated_molecule(generated_str, repr_name)

        gt_smiles = ground_truth_map.get(str(row["molecule_id"]))
        if gt_smiles:
            gt_mol = Chem.MolFromSmiles(gt_smiles)
            canonical_gt = Chem.MolToSmiles(gt_mol, canonical=True, isomericSmiles=False) if gt_mol else None
        else:
            canonical_gt = None

        tanimoto_morgan = None
        tanimoto_maccs = None
        tanimoto_rdkit = None
        exact_match = False
        if is_valid and canonical_gt:
            tanimoto_morgan = compute_tanimoto_similarity(canonical_gen, canonical_gt, "morgan")
            tanimoto_maccs = compute_tanimoto_similarity(canonical_gen, canonical_gt, "maccs")
            tanimoto_rdkit = _compute_rdkit_tanimoto(canonical_gen, canonical_gt)
            exact_match = (canonical_gen == canonical_gt)

        return {
            "molecule_id": row["molecule_id"],
            "representation": repr_name,
            "model": row["model"],
            "thinking": row["thinking"],
            "valid": is_valid,
            "exact_match": exact_match,
            "tanimoto_morgan": tanimoto_morgan,
            "tanimoto_maccs": tanimoto_maccs,
            "tanimoto_rdkit": tanimoto_rdkit,
            "canonical_gen": canonical_gen if is_valid else None,
            "canonical_gt": canonical_gt,
        }

    logger.info("Scoring B6: decoding generated molecules...")
    rows_list = [row for _, row in df.iterrows()]
    results = [None] * len(rows_list)
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_idx = {executor.submit(_process_row, row): i for i, row in enumerate(rows_list)}
        for future in tqdm(as_completed(future_to_idx), total=len(rows_list), desc="Decoding B6"):
            idx = future_to_idx[future]
            results[idx] = future.result()

    # Flush IUPAC cache if dirty
    if repr_mod._iupac_cache_dirty:
        repr_mod._save_iupac_cache()

    scored = pd.DataFrame(results)

    out_rows = []
    for (rep, model, thinking), g in scored.groupby(["representation", "model", "thinking"]):
        valid_g = g[g["valid"] == True]
        n = len(g)

        # Validity rate
        valid_scores = g["valid"].astype(float).values
        out_rows.append({
            "representation": rep, "model": model, "thinking": thinking,
            "metric_name": "validity_rate",
            "value": float(np.mean(valid_scores)),
            "sem": float(np.std(valid_scores, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n_questions": n,
        })

        # Exact match rate
        exact_scores = g["exact_match"].astype(float).values
        out_rows.append({
            "representation": rep, "model": model, "thinking": thinking,
            "metric_name": "exact_match",
            "value": float(np.mean(exact_scores)),
            "sem": float(np.std(exact_scores, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n_questions": n,
        })

        # Morgan Tanimoto (over ALL molecules; invalid = 0)
        morgan_all = g["tanimoto_morgan"].fillna(0.0).values
        out_rows.append({
            "representation": rep, "model": model, "thinking": thinking,
            "metric_name": "mean_tanimoto",
            "value": float(np.mean(morgan_all)),
            "sem": float(np.std(morgan_all, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n_questions": n,
        })

        # MACCS Tanimoto (over ALL molecules; invalid = 0)
        maccs_all = g["tanimoto_maccs"].fillna(0.0).values
        out_rows.append({
            "representation": rep, "model": model, "thinking": thinking,
            "metric_name": "maccs_tanimoto",
            "value": float(np.mean(maccs_all)),
            "sem": float(np.std(maccs_all, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n_questions": n,
        })

        # RDKit Tanimoto (over ALL molecules; invalid = 0)
        rdkit_all = g["tanimoto_rdkit"].fillna(0.0).values
        out_rows.append({
            "representation": rep, "model": model, "thinking": thinking,
            "metric_name": "rdkit_tanimoto",
            "value": float(np.mean(rdkit_all)),
            "sem": float(np.std(rdkit_all, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n_questions": n,
        })

        # FCD (computed per group — needs list of SMILES)
        gen_smiles = valid_g["canonical_gen"].dropna().tolist()
        gt_smiles = g["canonical_gt"].dropna().tolist()
        if len(gen_smiles) >= 2 and len(gt_smiles) >= 2:
            try:
                fcd_score = fcd.get_fcd(gt_smiles, gen_smiles)
            except Exception:
                fcd_score = np.nan
        else:
            fcd_score = np.nan
        out_rows.append({
            "representation": rep, "model": model, "thinking": thinking,
            "metric_name": "fcd",
            "value": float(fcd_score) if not np.isnan(fcd_score) else np.nan,
            "sem": 0.0,  # FCD is a single distributional metric, no per-question SEM
            "n_questions": n,
        })

        # Composite score: 0.2*validity + 0.4*morgan_tanimoto + 0.2*maccs_tanimoto + 0.2*exact_match
        validity_val = float(np.mean(valid_scores))
        exact_val = float(np.mean(exact_scores))
        morgan_val = float(np.mean(morgan_all))
        maccs_val = float(np.mean(maccs_all))
        composite = 0.2 * validity_val + 0.4 * morgan_val + 0.2 * maccs_val + 0.2 * exact_val
        out_rows.append({
            "representation": rep, "model": model, "thinking": thinking,
            "metric_name": "composite_score",
            "value": composite,
            "sem": 0.0,  # Derived metric, no direct SEM
            "n_questions": n,
        })

    return out_rows


def score_b9_b10(df: pd.DataFrame) -> List[dict]:
    """B9/B10: Yes/No discrimination — accuracy."""
    df = df.copy()
    if df["ground_truth"].dtype == object:
        df["ground_truth_bool"] = df["ground_truth"].map({"Yes": True, "No": False})
    else:
        df["ground_truth_bool"] = df["ground_truth"].astype(bool)
    if df["predicted"].dtype == object:
        df["predicted_bool"] = df["predicted"].map({"Yes": True, "No": False, True: True, False: False})
    else:
        df["predicted_bool"] = df["predicted"].astype(bool)
    df["correct"] = (df["predicted_bool"] == df["ground_truth_bool"]).astype(float)

    rows = []
    for (rep, model, thinking), g in df.groupby(["representation", "model", "thinking"]):
        scores = g["correct"].values
        rows.append({
            "representation": rep, "model": model, "thinking": thinking,
            "metric_name": "accuracy",
            "value": float(np.mean(scores)),
            "sem": float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0,
            "n_questions": len(scores),
        })
    return rows


SCORERS = {
    1: score_b1,
    2: score_b2,
    3: score_b3,
    4: score_b4,
    5: score_b5,
    6: score_b6,
    9: score_b9_b10,
    10: score_b9_b10,
}


# ============================================================================
# LaTeX Generation
# ============================================================================


BENCH_DISPLAY_CLEAN = {
    "atom_counting": "Atom Counting",
    "functional_groups": "Functional Groups",
    "property_estimation": "Property Estimation",
    "retrieval": "Molecule Retrieval",
    "isomer_discrimination": "Isomer Discrimination",
    "generation": "Caption-to-Molecule",
    "completion": "Molecular Completion",
    "tautomer_recognition": "Tautomer Recognition",
    "protonation_recognition": "Protonation State Recognition",
}


def generate_latex(results_df: pd.DataFrame) -> str:
    """Generate one LaTeX table per benchmark."""

    # Determine model order based on what's in the data
    model_thinking_pairs = []
    seen = set()
    for mid in MODEL_ORDER:
        for th in ["on", "off"]:
            key = (mid, th)
            subset = results_df[(results_df["model"] == mid) & (results_df["thinking"] == th)]
            if len(subset) > 0 and key not in seen:
                model_thinking_pairs.append(key)
                seen.add(key)
    # Add any models not in MODEL_ORDER
    for _, row in results_df[["model", "thinking"]].drop_duplicates().iterrows():
        key = (row["model"], row["thinking"])
        if key not in seen:
            model_thinking_pairs.append(key)
            seen.add(key)

    rep_cols = REPRESENTATIONS
    rep_headers = [REPR_DISPLAY_NAMES.get(r, r) for r in rep_cols]

    # Short rep headers for table width
    short_rep = {
        "canonical_smiles": "Canonical SMILES",
        "isomeric_smiles": "Isomeric SMILES",
        "randomized_smiles": "Randomized SMILES",
        "deepsmiles": "DeepSMILES",
        "iupac": "IUPAC",
        "selfies": "SELFIES",
        "moljson": "MolJSON",
        "cml": "CML",
        "inchi": "InChI",
    }

    latex_parts = []
    latex_parts.append("% Auto-generated by 06_tabulate_results.py\n")

    # Build list of (bench_num, metric_name) pairs to generate tables for
    table_specs = []
    for bench_num in BENCH_NUMS:
        if bench_num == 6:
            table_specs.append((6, "composite_score"))
            table_specs.append((6, "validity_rate"))
            table_specs.append((6, "exact_match"))
            table_specs.append((6, "mean_tanimoto"))
            table_specs.append((6, "maccs_tanimoto"))
            table_specs.append((6, "rdkit_tanimoto"))
            table_specs.append((6, "fcd"))
        elif bench_num == 3:
            # Separate table per property with appropriate metric
            table_specs.append((3, "logp_spearman_rho"))
            table_specs.append((3, "tpsa_spearman_rho"))
            table_specs.append((3, "hbd_exact_match"))
            table_specs.append((3, "hba_exact_match"))
        else:
            table_specs.append((bench_num, BENCH_PRIMARY_METRIC[bench_num]))

    for bench_num, metric_name in table_specs:
        bench_name = BENCHMARK_NUM_TO_NAME.get(bench_num, f"benchmark_{bench_num}")
        bench_display = BENCH_DISPLAY_CLEAN.get(bench_name, bench_name)
        bench_df = results_df[
            (results_df["benchmark"] == bench_num)
            & (results_df["metric_name"] == metric_name)
        ]

        if len(bench_df) == 0:
            continue

        # Build lookup: (model, thinking, representation) -> value
        lookup = {}
        for _, row in bench_df.iterrows():
            lookup[(row["model"], row["thinking"], row["representation"])] = (
                row["value"], row["sem"]
            )

        # Find best value per column (representation) for bolding
        # FCD and MAE are lower-is-better; all others are higher-is-better
        lower_is_better = (metric_name == "fcd" or metric_name.endswith("_mae"))
        best_per_rep = {}
        for rep in rep_cols:
            vals = []
            for (mid, th) in model_thinking_pairs:
                if (mid, th, rep) in lookup:
                    v, _ = lookup[(mid, th, rep)]
                    if not np.isnan(v):
                        vals.append(v)
            if vals:
                best_per_rep[rep] = min(vals) if lower_is_better else max(vals)
            else:
                best_per_rep[rep] = None

        n_reps = len(rep_cols)
        col_spec = "ll" + "c" * n_reps
        header_row = " & ".join(
            ["Model", "Reasoning"] + [short_rep.get(r, r) for r in rep_cols]
        )

        # Clean metric name for caption (replace underscores with hyphens)
        metric_display = metric_name.replace("_", "-")
        caption_text = f"{bench_display} ({metric_display})"
        label_suffix = f"{bench_num}_{metric_name}" if bench_num == 6 else str(bench_num)

        table_lines = []
        table_lines.append(f"\\begin{{table*}}[htbp]")
        table_lines.append(f"\\centering")
        table_lines.append(f"\\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{{col_spec}}}")
        table_lines.append(f"\\toprule")
        table_lines.append(f"{header_row} \\\\")
        table_lines.append(f"\\midrule")

        # For non-reasoning models, collapse on/off into a single row
        non_reasoning_seen = set()
        for (mid, th) in model_thinking_pairs:
            if mid in NON_REASONING_MODELS:
                if mid in non_reasoning_seen:
                    continue
                non_reasoning_seen.add(mid)
            display_name = MODEL_NAMES.get(mid, API_MODEL_NAMES.get(mid, mid))
            # Shorten display name for table width
            display_name = display_name.replace("-Thinking-2507", "")
            display_name = display_name.replace("-A3B", "")
            display_name = display_name.replace("-A22B", "")
            if mid in NON_REASONING_MODELS:
                think_str = "\\texttimes"
            else:
                think_str = "\\checkmark" if th == "on" else "\\texttimes"

            cells = []
            for rep in rep_cols:
                # For non-reasoning models, check both on/off keys
                if mid in NON_REASONING_MODELS:
                    lk = lookup.get((mid, "on", rep)) or lookup.get((mid, "off", rep))
                else:
                    lk = lookup.get((mid, th, rep))
                if lk is not None:
                    v, s = lk
                    if np.isnan(v):
                        cells.append("N/A")
                    else:
                        val_str = f"{v:.3f}"
                        if s > 0:
                            cell = f"{val_str}{{\\scriptsize$\\pm${s:.3f}}}"
                        else:
                            cell = val_str
                        if best_per_rep.get(rep) is not None and abs(v - best_per_rep[rep]) < 1e-6:
                            cell = f"\\textbf{{{cell}}}"
                        cells.append(cell)
                else:
                    cells.append("N/A")

            row_str = " & ".join([display_name, think_str] + cells) + " \\\\"
            table_lines.append(row_str)

        table_lines.append(f"\\bottomrule")
        table_lines.append(f"\\end{{tabular}}}}")
        table_lines.append(f"\\caption{{{caption_text}}}")
        # Label derived from caption: e.g. "Atom Counting (accuracy)" -> "tab:atom_counting_accuracy"
        label_str = caption_text.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        table_lines.append(f"\\label{{tab:{label_str}}}")
        table_lines.append(f"\\end{{table*}}")
        table_lines.append("")

        latex_parts.append("\n".join(table_lines))

    return "\n\n".join(latex_parts)


# ============================================================================
# Main
# ============================================================================


def _score_one_file(args):
    """Score a single file — top-level function for ProcessPoolExecutor."""
    bench_num, model_id, thinking, filepath = args
    filepath = Path(filepath)
    df = load_jsonl(filepath)
    if len(df) == 0:
        return []
    scorer = SCORERS.get(bench_num)
    if scorer is None:
        return []
    scored_rows = scorer(df)
    for r in scored_rows:
        r["benchmark"] = bench_num
        r["benchmark_name"] = BENCHMARK_NUM_TO_NAME.get(bench_num, f"b{bench_num}")
    return scored_rows


def _build_per_question_scores(bench_num: int, run1_files: Dict[Tuple[int, str, str], Path]) -> Optional[pd.DataFrame]:
    """Build per-question scores from run-1 raw JSONL files (all models incl. API).

    Returns DataFrame with columns: [_id, representation, model, thinking, score]
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Collect all run-1 files for this benchmark
    bench_files = {k: v for k, v in run1_files.items() if k[0] == bench_num}
    if not bench_files:
        return None

    dfs = []
    for (_, model_id, thinking_str), filepath in bench_files.items():
        dfs.append(load_jsonl(filepath))
    df = pd.concat(dfs, ignore_index=True)

    id_col = "pair_id" if bench_num in [5, 9, 10] else "molecule_id"

    if bench_num == 1:
        # Atom counting: numeric comparison (predicted can be float, ground_truth int)
        df["pred_num"] = pd.to_numeric(df["predicted"], errors="coerce")
        df["gt_num"] = pd.to_numeric(df["ground_truth"], errors="coerce")
        df["correct"] = (df["pred_num"] == df["gt_num"]).astype(float)
        per_mol = df.groupby([id_col, "representation", "model", "thinking"])["correct"].mean().reset_index()
        per_mol["score"] = per_mol["correct"].astype(float)

    elif bench_num == 5:
        # Isomer discrimination: Yes/No string comparison
        df["correct"] = (df["predicted"].astype(str).str.strip().str.lower() == df["ground_truth"].astype(str).str.strip().str.lower()).astype(float)
        per_mol = df.groupby([id_col, "representation", "model", "thinking"])["correct"].mean().reset_index()
        per_mol["score"] = per_mol["correct"].astype(float)

    elif bench_num == 4:
        # Retrieval: predicted_letter == correct_letter
        df["correct"] = (df["predicted_letter"].astype(str).str.strip() == df["correct_letter"].astype(str).str.strip()).astype(float)
        per_mol = df.groupby([id_col, "representation", "model", "thinking"])["correct"].mean().reset_index()
        per_mol["score"] = per_mol["correct"].astype(float)

    elif bench_num == 2:
        # Functional groups: boolean comparison per (molecule, group)
        bool_map = {"True": True, "False": False, True: True, False: False}
        df["pred_bool"] = df["predicted"].map(bool_map)
        df["gt_bool"] = df["ground_truth"].map(bool_map)
        df["correct"] = (df["pred_bool"] == df["gt_bool"]).astype(float)
        per_mol = df.groupby([id_col, "representation", "model", "thinking"])["correct"].mean().reset_index()
        per_mol["score"] = per_mol["correct"].astype(float)

    elif bench_num == 3:
        # Property estimation: return dict of property -> per-molecule scores
        # logp/tpsa: per-molecule absolute error (negated so higher = better for bootstrap)
        # hbd/hba: per-molecule exact match
        df["predicted_num"] = pd.to_numeric(df["predicted"], errors="coerce")
        df["gt_num"] = pd.to_numeric(df["ground_truth"], errors="coerce")
        valid = df["predicted_num"].notna() & df["gt_num"].notna()
        df = df[valid].copy()

        prop_results = {}
        for prop in ["logp", "tpsa", "hbd", "hba"]:
            pg = df[df["property"] == prop].copy()
            if len(pg) == 0:
                continue
            if prop in ["logp", "tpsa"]:
                # Use negative absolute error so higher = better (for paired bootstrap)
                pg["score"] = -np.abs(pg["predicted_num"] - pg["gt_num"])
            else:
                pg["score"] = (pg["predicted_num"] == pg["gt_num"]).astype(float)
            pm = pg.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()
            pm["_id"] = pm[id_col]
            prop_results[(3, prop)] = pm[["_id", "representation", "model", "thinking", "score"]]

        return prop_results

    elif bench_num == 6:
        # Generation: per-molecule composite from validity + tanimoto
        # Re-use score_b6 which does the heavy lifting, but we need per-molecule scores
        # Compute inline: valid(float) * 0.5 + morgan_tanimoto * 0.5 as a quick proxy
        scored = pd.DataFrame(score_b6(df))
        # score_b6 returns aggregated rows, not per-molecule — need to score per molecule
        # Fall back to computing per-molecule validity as the per-question score
        from rdkit import Chem
        results = []
        for _, row in df.iterrows():
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
            canonical_gen, is_valid = decode_generated_molecule(generated_str, repr_name)

            gt_smiles = None
            test_path = DATA_DIR / "chebi_prepared_500.csv"
            if not hasattr(_build_per_question_scores, "_gt_map"):
                gt_df = pd.read_csv(test_path)
                gt_df["molecule_id"] = gt_df["CID"].astype(str)
                _build_per_question_scores._gt_map = dict(zip(gt_df["molecule_id"], gt_df["SMILES"]))
            gt_smiles = _build_per_question_scores._gt_map.get(str(row["molecule_id"]))

            tanimoto = 0.0
            exact = 0.0
            if is_valid and gt_smiles:
                tanimoto = compute_tanimoto_similarity(canonical_gen, gt_smiles, "morgan") or 0.0
                gt_mol = Chem.MolFromSmiles(gt_smiles)
                canonical_gt = Chem.MolToSmiles(gt_mol, canonical=True, isomericSmiles=False) if gt_mol else None
                exact = float(canonical_gen == canonical_gt) if canonical_gt else 0.0

            # Composite: 0.2*valid + 0.4*morgan + 0.2*maccs + 0.2*exact (simplified: skip maccs)
            composite = 0.2 * float(is_valid) + 0.6 * tanimoto + 0.2 * exact
            results.append({
                "_id": row["molecule_id"],
                "representation": repr_name,
                "model": row["model"],
                "thinking": row["thinking"],
                "score": composite,
            })
        per_mol = pd.DataFrame(results)
        return per_mol[["_id", "representation", "model", "thinking", "score"]]

    elif bench_num in [9, 10]:
        # Yes/No discrimination
        gt_map = {"Yes": "True", "No": "False", True: "True", False: "False"}
        df["gt_str"] = df["ground_truth"].map(gt_map).fillna(df["ground_truth"].astype(str))
        df["pred_str"] = df["predicted"].map(gt_map).fillna(df["predicted"].astype(str))
        df["correct"] = (df["pred_str"] == df["gt_str"]).astype(float)
        per_mol = df.groupby([id_col, "representation", "model", "thinking"])["correct"].mean().reset_index()
        per_mol["score"] = per_mol["correct"].astype(float)

    else:
        return None

    per_mol["_id"] = per_mol[id_col]
    return per_mol[["_id", "representation", "model", "thinking", "score"]]


def _find_tied_reprs(per_mol: pd.DataFrame, model: str, thinking: str,
                     alpha: float = 0.05, n_bootstrap: int = 10000, seed: int = 42) -> List[str]:
    """Return list of representations tied for best (paired bootstrap, CI includes 0)."""
    rng = np.random.RandomState(seed)

    grp = per_mol[(per_mol["model"] == model) & (per_mol["thinking"] == thinking)]
    repr_means = grp.groupby("representation")["score"].mean().sort_values(ascending=False)
    if len(repr_means) < 2:
        return [repr_means.index[0]] if len(repr_means) == 1 else []

    best_repr = repr_means.index[0]
    tied = [best_repr]

    for other_repr in repr_means.index[1:]:
        # Skip representations that scored 0 on every question
        if repr_means[other_repr] == 0:
            continue
        pivot = grp[grp["representation"].isin([best_repr, other_repr])].pivot_table(
            index="_id", columns="representation", values="score", aggfunc="mean"
        ).dropna()
        if len(pivot) < 10:
            continue
        diff = pivot[best_repr].values.astype(float) - pivot[other_repr].values.astype(float)
        if np.all(diff == 0):
            tied.append(other_repr)
        else:
            # Paired bootstrap: resample paired differences, check if CI includes 0
            n = len(diff)
            boot_means = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                idx = rng.randint(0, n, size=n)
                boot_means[b] = diff[idx].mean()
            lo = np.percentile(boot_means, 100 * alpha / 2)
            # If lower bound of CI <= 0, the difference is not significant
            if lo <= 0:
                tied.append(other_repr)

    return tied


def generate_best_repr_table(results_df: pd.DataFrame) -> str:
    """Generate 'best representation per model' and 'win counts' LaTeX tables.

    Uses paired bootstrap test to identify ties: when the best representation
    is not significantly better (p >= 0.05) than another, both are listed.
    Win counts award a point to every tied representation.
    """

    # Benchmark columns in display order
    # B3 is split into 4 sub-columns keyed as (3, "logp"), (3, "tpsa"), etc.
    # Other benchmarks are keyed as plain ints.
    B3_PROPERTIES = ["logp", "tpsa", "hbd", "hba"]
    bench_order = [1, 2, (3, "logp"), (3, "tpsa"), (3, "hbd"), (3, "hba"), 4, 5, 6, 9, 10]
    bench_short_names = {
        1: "Atom Count.",
        2: "Func. Groups",
        (3, "logp"): "logP",
        (3, "tpsa"): "TPSA",
        (3, "hbd"): "HBD",
        (3, "hba"): "HBA",
        4: "Retrieval",
        5: "Isomer Disc.",
        6: "Cap.-to-Mol.",
        9: "Tautomer",
        10: "Protonation",
    }

    short_rep = {
        "canonical_smiles": "Canonical SMILES",
        "isomeric_smiles": "Isomeric SMILES",
        "randomized_smiles": "Randomized SMILES",
        "deepsmiles": "DeepSMILES",
        "iupac": "IUPAC",
        "selfies": "SELFIES",
        "moljson": "MolJSON",
        "cml": "CML",
        "inchi": "InChI",
    }

    # Model family groupings for midrule separators
    MODEL_FAMILIES = [
        ["qwen3-4b-thinking-2507", "qwen3-30b-a3b-thinking-2507", "qwen2.5-14b"],
        ["phi-4", "phi-4-reasoning", "phi-4-reasoning-plus"],
        ["chemdfm-v2.0-14b", "chemdfm-r-14b"],
        ["mistral-small-24b", "ether0-24b"],
        ["olmo-3.1-32b-instruct", "olmo-3.1-32b-think"],
        ["gpt-5.4-mini", "claude-haiku-4.5"],
    ]

    # Load per-question scores for paired bootstrap tests (from run-1 raw JSONLs, all models)
    run1_files = discover_run1_files()
    per_q_data = {}
    # Collect unique raw benchmark numbers to score
    raw_bench_nums = sorted(set(b if isinstance(b, int) else b[0] for b in bench_order))
    for bench_num in raw_bench_nums:
        pq = _build_per_question_scores(bench_num, run1_files)
        if pq is not None:
            if isinstance(pq, dict):
                # B3 returns {(3, "logp"): df, (3, "tpsa"): df, ...}
                per_q_data.update(pq)
            else:
                per_q_data[bench_num] = pq

    # Build lookup: (bench_key, model, thinking) -> list of tied representation names
    best_repr_lookup = {}
    for bench_key in bench_order:
        if bench_key in per_q_data:
            pq = per_q_data[bench_key]
            for (model, thinking), _ in pq.groupby(["model", "thinking"]):
                tied = _find_tied_reprs(pq, model, thinking)
                th_str = "on" if thinking is True or thinking == "on" or thinking == True else "off"
                best_repr_lookup[(bench_key, model, th_str)] = tied

    # Determine model/thinking pairs present in data, in order
    model_thinking_pairs = []
    seen = set()
    for family in MODEL_FAMILIES:
        for mid in family:
            for th in ["on", "off"]:
                key = (mid, th)
                subset = results_df[(results_df["model"] == mid) & (results_df["thinking"] == th)]
                if len(subset) > 0 and key not in seen:
                    model_thinking_pairs.append(key)
                    seen.add(key)
    # Add any remaining models not in MODEL_FAMILIES
    for _, row in results_df[["model", "thinking"]].drop_duplicates().iterrows():
        key = (row["model"], row["thinking"])
        if key not in seen:
            model_thinking_pairs.append(key)
            seen.add(key)

    # --- Table 1: Best representation per model ---
    n_bench = len(bench_order)

    # Build two-row header: top row has \multicolumn{4}{c}{Prop. Est.} spanning B3 sub-cols
    # Bottom row has the individual column names
    top_header_parts = ["Model", "Reasoning"]
    bottom_header_parts = ["", ""]
    for b in bench_order:
        if isinstance(b, tuple) and b[0] == 3:
            # Skip — handled by multicolumn below
            pass
        else:
            top_header_parts.append(bench_short_names[b])
            bottom_header_parts.append("")
        # Insert multicolumn once at the first B3 sub-col
        if b == (3, "logp"):
            top_header_parts.append(r"\multicolumn{4}{c}{Prop.\ Est.}")
            for prop_key in [(3, "logp"), (3, "tpsa"), (3, "hbd"), (3, "hba")]:
                bottom_header_parts.append(bench_short_names[prop_key])

    top_header = " & ".join(top_header_parts)
    bottom_header = " & ".join(bottom_header_parts)

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\tiny")
    lines.append(r"\resizebox{\textwidth}{!}{\begin{tabular}{ll" + "p{1.6cm}" * n_bench + "}")
    lines.append(r"\toprule")
    lines.append(f"{top_header} \\\\")
    # Add cmidrule under the Prop. Est. multicolumn
    # Find the column positions: Model(1) + Reasoning(2) + B1(3) + B2(4) + B3 logp(5)..hba(8)
    b3_start = 3 + bench_order.index((3, "logp"))
    b3_end = b3_start + 3
    lines.append(f"\\cmidrule(lr){{{b3_start}-{b3_end}}}")
    lines.append(f"{bottom_header} \\\\")
    lines.append(r"\midrule")

    family_idx = 0
    family_set = set(MODEL_FAMILIES[0]) if MODEL_FAMILIES else set()
    non_reasoning_seen = set()
    for i, (mid, th) in enumerate(model_thinking_pairs):
        # For non-reasoning models, collapse to a single row
        if mid in NON_REASONING_MODELS:
            if mid in non_reasoning_seen:
                continue
            non_reasoning_seen.add(mid)

        # Insert midrule between families
        if i > 0:
            prev_mid = model_thinking_pairs[i - 1][0]
            if mid not in family_set:
                for fi, fam in enumerate(MODEL_FAMILIES):
                    if mid in fam:
                        family_idx = fi
                        family_set = set(fam)
                        lines.append(r"\midrule")
                        break

        display_name = MODEL_NAMES.get(mid, API_MODEL_NAMES.get(mid, mid))
        display_name = display_name.replace("-Thinking-2507", "")
        display_name = display_name.replace("-A3B", "")
        display_name = display_name.replace("-A22B", "")
        if mid in NON_REASONING_MODELS:
            think_str = "\\texttimes"
        else:
            think_str = "\\checkmark" if th == "on" else "\\texttimes"

        cells = []
        for bench_key in bench_order:
            if mid in NON_REASONING_MODELS:
                tied = best_repr_lookup.get((bench_key, mid, "on")) or best_repr_lookup.get((bench_key, mid, "off"))
            else:
                tied = best_repr_lookup.get((bench_key, mid, th))
            if not tied:
                cells.append("N/A")
            elif len(tied) == 1:
                cells.append(short_rep.get(tied[0], tied[0]))
            else:
                # Multiple tied: bold the first (highest mean), list others after comma
                tied_names = [short_rep.get(r, r) for r in tied]
                cells.append(", ".join(tied_names))

        row_str = " & ".join([display_name, think_str] + cells) + " \\\\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\caption{Best representation per model for each benchmark (primary metric). "
                 r"Multiple representations listed when the paired bootstrap test "
                 r"($p \geq 0.05$) cannot distinguish them from the best.}")
    lines.append(r"\label{tab:best_repr_per_model}")
    lines.append(r"\end{table*}")

    # --- Table 2: Win counts per representation ---
    # Each tied representation gets a point
    rep_order = [
        "canonical_smiles", "isomeric_smiles", "randomized_smiles", "deepsmiles",
        "iupac", "selfies", "cml", "inchi", "moljson",
    ]
    win_counts = {rep: {b: 0 for b in bench_order} for rep in rep_order}

    for (mid, th) in model_thinking_pairs:
        for bench_key in bench_order:
            tied = best_repr_lookup.get((bench_key, mid, th))
            if tied:
                for rep in tied:
                    if rep in win_counts:
                        win_counts[rep][bench_key] += 1

    lines.append("")
    lines.append("")
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\tiny")
    lines.append(r"\caption{Number of model configurations where each representation is among the best "
                 r"(paired bootstrap, 95\% CI includes zero vs.\ top scorer). "
                 r"Ties award a point to every indistinguishable representation.}")
    lines.append(r"\label{tab:repr_win_counts}")
    lines.append(r"\resizebox{\textwidth}{!}{\begin{tabular}{l" + "c" * n_bench + "|c}")
    lines.append(r"\toprule")
    # Two-row header for win counts table too
    win_top_parts = ["Representation"]
    win_bottom_parts = [""]
    for b in bench_order:
        if isinstance(b, tuple) and b[0] == 3:
            pass
        else:
            win_top_parts.append(bench_short_names[b])
            win_bottom_parts.append("")
        if b == (3, "logp"):
            win_top_parts.append(r"\multicolumn{4}{c}{Prop.\ Est.}")
            for prop_key in [(3, "logp"), (3, "tpsa"), (3, "hbd"), (3, "hba")]:
                win_bottom_parts.append(bench_short_names[prop_key])
    win_top_parts.append("Total")
    win_bottom_parts.append("")
    lines.append(f"{' & '.join(win_top_parts)} \\\\")
    # cmidrule under Prop. Est. — column positions: Repr(1) + B1(2) + B2(3) + B3 logp(4)..hba(7)
    win_b3_start = 2 + bench_order.index((3, "logp"))
    win_b3_end = win_b3_start + 3
    lines.append(f"\\cmidrule(lr){{{win_b3_start}-{win_b3_end}}}")
    lines.append(f"{' & '.join(win_bottom_parts)} \\\\")
    lines.append(r"\midrule")

    for rep in rep_order:
        counts = [win_counts[rep][b] for b in bench_order]
        total = sum(counts)
        row_str = " & ".join([short_rep[rep]] + [str(c) for c in counts] + [str(total)]) + " \\\\"
        lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tabulate benchmark results into CSV and LaTeX tables")
    parser.add_argument("--best-repr", action="store_true",
                        help="Generate best_representation_tables.tex (best repr per model + win counts)")
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    files = discover_run1_files()
    logger.info(f"Discovered {len(files)} run-1 files across benchmarks {sorted(set(b for b, _, _ in files))}")

    # Build task list
    tasks = [
        (bench_num, model_id, thinking, str(filepath))
        for (bench_num, model_id, thinking), filepath in sorted(files.items())
    ]

    import multiprocessing
    n_workers = min(len(tasks), multiprocessing.cpu_count(), 32)
    logger.info(f"Scoring {len(tasks)} files with {n_workers} parallel workers")

    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_score_one_file, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring files"):
            key = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
            except Exception as e:
                logger.error(f"Error scoring {key}: {e}")

    results_df = pd.DataFrame(all_rows)
    # Normalize thinking to string for clean output
    results_df["thinking"] = results_df["thinking"].map({True: "on", False: "off"})

    # Save CSV
    csv_path = TABLE_DIR / "full_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path} ({len(results_df)} rows)")

    # Generate LaTeX
    latex_str = generate_latex(results_df)
    tex_path = TABLE_DIR / "summary_table.tex"
    tex_path.write_text(latex_str)
    logger.info(f"Saved LaTeX: {tex_path}")

    # Generate best representation table if requested
    if args.best_repr:
        best_repr_str = generate_best_repr_table(results_df)
        best_repr_path = TABLE_DIR / "best_representation_tables.tex"
        best_repr_path.write_text(best_repr_str)
        logger.info(f"Saved best-repr LaTeX: {best_repr_path}")

    # Print summary stats
    logger.info("=== Summary ===")
    for bn in BENCH_NUMS:
        bdf = results_df[results_df["benchmark"] == bn]
        if len(bdf) > 0:
            logger.info(
                f"  B{bn}: {len(bdf)} results, "
                f"mean {BENCH_PRIMARY_METRIC[bn]} = {bdf['value'].mean():.3f}"
            )


if __name__ == "__main__":
    main()
