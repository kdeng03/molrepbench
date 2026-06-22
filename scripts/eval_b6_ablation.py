"""
eval_b6_ablation.py - Evaluate B6 prompt ablation results and generate a LaTeX table.

Reads the raw JSONL files for three B6 prompt formats (baseline/zero-shot, few_shot,
decompose) for qwen3-4b, computes per-representation metrics, and outputs a .tex table.

Metrics per (prompt_format, representation):
  - Validity rate
  - Exact match rate
  - Mean Morgan Tanimoto (valid molecules only)
  - Mean MACCS Tanimoto (valid molecules only)

Usage:
    python scripts/eval_b6_ablation.py
    python scripts/eval_b6_ablation.py --run_id 1 --output paper/b6_ablation_table.tex
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR,
    REPRESENTATIONS,
    REPR_DISPLAY_NAMES,
    RESULTS_DIR,
)
from utils import chemistry, parsing, representations

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "qwen3-4b-thinking-2507"

PROMPT_FORMAT_LABELS = {
    "zero_shot": "Zero-shot",
    "few_shot": "2-shot ICL",
    "decompose": "Decomposition",
}

# ---------------------------------------------------------------------------
# Helpers (reuse logic from 03_evaluate.py without importing it)
# ---------------------------------------------------------------------------


def decode_generated_molecule(
    generated_string: str, repr_name: str
) -> Tuple[Optional[str], bool]:
    if generated_string is None or (isinstance(generated_string, float) and pd.isna(generated_string)):
        return None, False
    if len(str(generated_string)) > 2000:
        return None, False
    try:
        from rdkit import Chem
        mol = representations.parse_representation(generated_string, repr_name)
        if mol is None:
            return None, False
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return canonical, True
    except Exception:
        return None, False


def compute_tanimoto(smiles1: str, smiles2: str, fp_type: str = "morgan") -> Optional[float]:
    from rdkit import Chem
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return None
        return chemistry.calculate_tanimoto_similarity(mol1, mol2, fp_type)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Load and score
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def re_extract(row: pd.Series) -> str:
    """Re-extract generated string from raw_response (mirrors 03_evaluate logic)."""
    raw = row.get("raw_response")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raw = ""
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
        extracted = parsing.extract_molecule_string(cleaned, repr_name)
    else:
        extracted = None
    return extracted or row.get("generated_string")


def score_row(row: pd.Series, ground_truth_map: Dict[str, str]) -> Dict:
    generated_str = re_extract(row)
    repr_name = row["representation"]
    canonical_gen, is_valid = decode_generated_molecule(generated_str, repr_name)

    gt_smiles = ground_truth_map.get(str(row["molecule_id"]))
    canonical_gt = None
    if gt_smiles:
        from rdkit import Chem
        gt_mol = Chem.MolFromSmiles(gt_smiles)
        if gt_mol:
            canonical_gt = Chem.MolToSmiles(gt_mol, canonical=True, isomericSmiles=False)

    exact = (canonical_gen == canonical_gt) if (is_valid and canonical_gt) else False
    tanimoto = compute_tanimoto(canonical_gen, canonical_gt, "morgan") if (is_valid and canonical_gt) else None
    maccs = compute_tanimoto(canonical_gen, canonical_gt, "maccs") if (is_valid and canonical_gt) else None

    return {
        "valid": is_valid,
        "exact_match": exact,
        "tanimoto": tanimoto,
        "maccs_tanimoto": maccs,
    }


def score_dataframe(df: pd.DataFrame, ground_truth_map: Dict[str, str]) -> pd.DataFrame:
    logger.info(f"  Scoring {len(df)} rows...")
    rows = [row for _, row in df.iterrows()]
    # Score CML rows single-threaded (Open Babel is not thread-safe)
    has_cml = any(r["representation"] == "cml" for r in rows)
    if has_cml:
        results = [score_row(r, ground_truth_map) for r in rows]
    else:
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda r: score_row(r, ground_truth_map), rows))
    scores_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), scores_df], axis=1)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def aggregate_metrics(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per representation."""
    rows = []
    for repr_name in REPRESENTATIONS:
        grp = scored_df[scored_df["representation"] == repr_name]
        if len(grp) == 0:
            continue
        valid_grp = grp[grp["valid"] == True]
        rows.append({
            "representation": repr_name,
            "n": len(grp),
            "validity": grp["valid"].mean(),
            "exact_match": grp["exact_match"].mean(),
            "tanimoto": valid_grp["tanimoto"].mean() if len(valid_grp) > 0 else 0.0,
            "maccs_tanimoto": valid_grp["maccs_tanimoto"].mean() if len(valid_grp) > 0 else 0.0,
        })
    # Overall row
    valid_all = scored_df[scored_df["valid"] == True]
    rows.append({
        "representation": "Overall",
        "n": len(scored_df),
        "validity": scored_df["valid"].mean(),
        "exact_match": scored_df["exact_match"].mean(),
        "tanimoto": valid_all["tanimoto"].mean() if len(valid_all) > 0 else 0.0,
        "maccs_tanimoto": valid_all["maccs_tanimoto"].mean() if len(valid_all) > 0 else 0.0,
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------


def _fmt(val: float) -> str:
    import math
    if val is None or (isinstance(val, float) and math.isnan(val)):
        val = 0.0
    return f"{val * 100:.2f}"


def _bold_max(vals: List[float], idx: int) -> str:
    import math
    clean = [0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in vals]
    s = _fmt(clean[idx])
    if clean[idx] == max(clean):
        return rf"\textbf{{{s}}}"
    return s


def generate_latex_table(
    metrics: Dict[str, pd.DataFrame],
    output_path: Path,
):
    """Generate a LaTeX table comparing prompt formats across representations.

    Rows: representations (+ overall)
    Columns: grouped by metric, sub-columns per prompt format
    Uses resizebox to fit within page margins.
    """
    fmt_keys = list(metrics.keys())
    fmt_labels = [PROMPT_FORMAT_LABELS[k] for k in fmt_keys]
    n_fmt = len(fmt_keys)

    repr_order = REPRESENTATIONS + ["Overall"]

    metric_cols = ["validity", "exact_match", "tanimoto", "maccs_tanimoto"]
    metric_short = ["Valid.", "Exact", "Morgan", "MACCS"]

    # Column spec: l ccc | ccc | ccc | ccc
    col_spec = "l " + " | ".join(["c" * n_fmt for _ in range(4)])

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{B6 caption-to-molecule generation: prompt format ablation for Qwen3-4B (thinking on). Best per row in \textbf{bold}. All values are percentages.}")
    lines.append(r"\label{tab:b6_ablation}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row 1: metric group names
    header1_parts = [""]
    for ms in metric_short:
        header1_parts.append(rf"\multicolumn{{{n_fmt}}}{{c}}{{{ms}}}")
    lines.append(" & ".join(header1_parts) + r" \\")

    # Header row 2: prompt format abbreviations
    fmt_short = ["ZS", "2S", "Dec"]
    header2_parts = ["Representation"]
    for _ in range(4):
        header2_parts.extend(fmt_short)
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for repr_name in repr_order:
        display = REPR_DISPLAY_NAMES.get(repr_name, repr_name)
        if repr_name == "Overall":
            lines.append(r"\midrule")
            display = r"\textit{Overall}"

        cells = [display]
        for metric_col in metric_cols:
            vals = []
            for fk in fmt_keys:
                row = metrics[fk][metrics[fk]["representation"] == repr_name]
                vals.append(row[metric_col].values[0] if len(row) > 0 else 0.0)
            for i in range(n_fmt):
                cells.append(_bold_max(vals, i))

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}%")  # close resizebox
    # Footer note
    lines.append(r"\vspace{2pt}")
    lines.append(r"\raggedright\footnotesize ZS = Zero-shot, 2S = 2-shot ICL, Dec = Decomposition.")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex)
    logger.info(f"LaTeX table saved to {output_path}")
    print("\n" + tex + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate B6 prompt ablation and generate LaTeX table")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .tex path (default: paper/b6_ablation_table.tex)")
    args = parser.parse_args()

    run_id = args.run_id
    thinking_str = "thinking_on"
    raw_dir = RESULTS_DIR / "raw"

    file_map = {
        "zero_shot": raw_dir / f"benchmark_6_{MODEL_ID}_{thinking_str}_run{run_id}.jsonl",
        "few_shot": raw_dir / f"benchmark_6_{MODEL_ID}_{thinking_str}_few_shot_run{run_id}.jsonl",
        "decompose": raw_dir / f"benchmark_6_{MODEL_ID}_{thinking_str}_decompose_run{run_id}.jsonl",
    }

    for key, path in file_map.items():
        if not path.exists():
            logger.error(f"Missing result file for '{key}': {path}")
            sys.exit(1)

    # Load ground truth
    gt_df = pd.read_csv(DATA_DIR / "chebi_prepared_500.csv")
    gt_df["molecule_id"] = gt_df["CID"].astype(str)
    ground_truth_map = dict(zip(gt_df["molecule_id"], gt_df["SMILES"]))
    logger.info(f"Ground truth: {len(ground_truth_map)} molecules")

    # Score each format
    all_metrics = {}
    all_scored = {}
    for key, path in file_map.items():
        logger.info(f"Processing: {key} ({path.name})")
        df = load_jsonl(path)
        scored = score_dataframe(df, ground_truth_map)
        all_scored[key] = scored
        metrics = aggregate_metrics(scored)
        all_metrics[key] = metrics

        overall = metrics[metrics["representation"] == "Overall"].iloc[0]
        logger.info(
            f"  {key}: validity={overall['validity']:.1%}  exact={overall['exact_match']:.1%}  "
            f"tanimoto={overall['tanimoto']:.3f}  maccs={overall['maccs_tanimoto']:.3f}"
        )

    # Save scored CSVs
    scored_dir = RESULTS_DIR / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    for key, scored in all_scored.items():
        out_path = scored_dir / f"benchmark_6_{MODEL_ID}_{thinking_str}_{key}_scored.csv"
        scored.to_csv(out_path, index=False)
        logger.info(f"Scored CSV: {out_path}")

    # Generate LaTeX table
    output_path = Path(args.output) if args.output else Path("paper/b6_ablation_table.tex")
    generate_latex_table(all_metrics, output_path)


if __name__ == "__main__":
    main()
