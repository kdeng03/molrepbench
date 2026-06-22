"""
05_statistical_tests.py - Statistical significance testing for benchmark results

Performs pairwise Wilcoxon signed-rank tests between representations,
applies Bonferroni correction, and tests thinking/model effects.

Usage:
    python scripts/05_statistical_tests.py --all
    python scripts/05_statistical_tests.py --test pairwise
    python scripts/05_statistical_tests.py --test thinking
"""

import sys
from pathlib import Path
import argparse
import logging
from itertools import combinations
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare

# Import configuration
from config import (
    REPRESENTATIONS,
    REPR_DISPLAY_NAMES,
    MODELS,
    BENCHMARKS,
    BENCHMARK_DISPLAY_NAMES,
    RESULTS_DIR,
    ALPHA,
    BONFERRONI_ALPHA,
    N_COMPARISONS,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def load_scored_data() -> Dict[str, pd.DataFrame]:
    """Load all scored benchmark data."""
    data = {}

    for i in range(1, 8):
        scored_path = RESULTS_DIR / "scored" / f"benchmark_{i}_scored.csv"
        if scored_path.exists():
            data[f"b{i}"] = pd.read_csv(scored_path)
            logger.info(f"Loaded B{i} scored data: {len(data[f'b{i}'])} rows")
        else:
            logger.warning(f"Missing: {scored_path}")

    return data


def compute_normalized_score(df: pd.DataFrame, benchmark_num: int) -> pd.Series:
    """
    Compute normalized score (0-1) for a benchmark.

    Args:
        df: DataFrame with scored data
        benchmark_num: Benchmark number (1-7)

    Returns:
        Series with normalized scores
    """
    if benchmark_num == 1:
        # Atom counting: use 'correct' column (boolean)
        return df["correct"].astype(float)

    elif benchmark_num == 2:
        # Functional groups: use 'correct' column
        return df["correct"].astype(float)

    elif benchmark_num == 3:
        # Property estimation: 1 - (abs_error / max_abs_error)
        # Group by property type first
        scores = []
        for prop in df["property"].unique():
            prop_df = df[df["property"] == prop].copy()
            max_error = prop_df["abs_error"].max()
            if max_error > 0:
                prop_df["normalized"] = 1 - (prop_df["abs_error"] / max_error)
            else:
                prop_df["normalized"] = 1.0
            scores.append(prop_df)

        combined = pd.concat(scores)
        return combined["normalized"]

    elif benchmark_num == 4:
        # Retrieval: use 'correct' column
        return df["correct"].astype(float)

    elif benchmark_num == 5:
        # Isomer discrimination: use 'correct' column
        return df["correct"].astype(float)

    elif benchmark_num == 6:
        # Generation: use 'exact_match' column
        return df["exact_match"].astype(float)

    elif benchmark_num == 7:
        # Completion: use 'recovery' column
        return df["recovery"].astype(float)

    else:
        raise ValueError(f"Unknown benchmark number: {benchmark_num}")


# ============================================================================
# Test 1: Pairwise Representation Comparisons
# ============================================================================


def test_pairwise_representations(data: Dict[str, pd.DataFrame]) -> None:
    """
    Run pairwise Wilcoxon signed-rank tests between all representation pairs.

    For each benchmark, compare representations paired by molecule_id,
    averaging across model conditions first.

    Applies Bonferroni correction for multiple comparisons.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Test 1: Pairwise Representation Comparisons")
    logger.info("=" * 80 + "\n")

    stats_dir = RESULTS_DIR / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # For each benchmark
    for bench_num in range(1, 8):
        bench_key = f"b{bench_num}"
        bench_name = BENCHMARKS[bench_num - 1]

        if bench_key not in data:
            logger.warning(f"Skipping B{bench_num}: no data")
            continue

        logger.info(f"\nBenchmark {bench_num}: {BENCHMARK_DISPLAY_NAMES[bench_name]}")

        df = data[bench_key].copy()

        # Add normalized score
        df["score"] = compute_normalized_score(df, bench_num)

        # Determine ID column (molecule_id for most, pair_id for B5)
        id_col = "pair_id" if bench_num == 5 else "molecule_id"

        # Average across models and thinking conditions for each (id_col, representation)
        # For B2 and B3, also average across groups/properties
        if bench_num == 2:
            group_cols = [id_col, "representation", "model", "thinking", "group"]
            agg_df = df.groupby(group_cols)["score"].mean().reset_index()
            # Now average across groups
            agg_df = agg_df.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()
        elif bench_num == 3:
            prop_cols = [id_col, "representation", "model", "thinking", "property"]
            agg_df = df.groupby(prop_cols)["score"].mean().reset_index()
            # Now average across properties
            agg_df = agg_df.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()
        else:
            agg_df = df.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()

        # Average across models and thinking
        molecule_repr_scores = agg_df.groupby([id_col, "representation"])["score"].mean().reset_index()

        # Run pairwise tests
        results = []

        for repr1, repr2 in combinations(REPRESENTATIONS, 2):
            # Get scores for both representations
            repr1_df = molecule_repr_scores[molecule_repr_scores["representation"] == repr1]
            repr2_df = molecule_repr_scores[molecule_repr_scores["representation"] == repr2]

            # Merge on id_col to get paired data
            paired = repr1_df.merge(
                repr2_df,
                on=id_col,
                suffixes=("_1", "_2")
            )

            if len(paired) < 3:
                logger.warning(f"  {repr1} vs {repr2}: insufficient paired samples ({len(paired)})")
                continue

            # Wilcoxon signed-rank test (paired)
            try:
                statistic, p_value = wilcoxon(
                    paired["score_1"],
                    paired["score_2"],
                    alternative="two-sided"
                )

                # Bonferroni correction
                p_adjusted = min(p_value * N_COMPARISONS, 1.0)
                significant = p_adjusted < ALPHA

                # Effect size (mean difference)
                mean_diff = paired["score_1"].mean() - paired["score_2"].mean()

                results.append({
                    "benchmark": f"B{bench_num}",
                    "representation_1": repr1,
                    "representation_2": repr2,
                    "n_pairs": len(paired),
                    "mean_1": paired["score_1"].mean(),
                    "mean_2": paired["score_2"].mean(),
                    "mean_diff": mean_diff,
                    "statistic": statistic,
                    "p_value": p_value,
                    "p_adjusted": p_adjusted,
                    "significant": significant,
                })

                sig_marker = "***" if significant else ""
                logger.info(
                    f"  {REPR_DISPLAY_NAMES[repr1]:20s} vs {REPR_DISPLAY_NAMES[repr2]:20s}: "
                    f"Δ={mean_diff:+.3f}, p={p_value:.4f}, p_adj={p_adjusted:.4f} {sig_marker}"
                )

            except Exception as e:
                logger.warning(f"  {repr1} vs {repr2}: test failed - {str(e)}")

        # Save results
        if results:
            results_df = pd.DataFrame(results)
            output_path = stats_dir / f"pairwise_{bench_name}.csv"
            results_df.to_csv(output_path, index=False)
            logger.info(f"  Saved: {output_path}")

            # Count significant differences
            n_significant = results_df["significant"].sum()
            logger.info(f"  Significant differences: {n_significant}/{len(results)} pairs")

    # Create aggregated comprehension and generation tests
    logger.info("\n" + "-" * 80)
    logger.info("Aggregated Tests")
    logger.info("-" * 80 + "\n")

    # Comprehension (B1-B5 average)
    logger.info("Comprehension (B1-B5 average):")
    comprehension_scores = {}

    for bench_num in range(1, 6):
        bench_key = f"b{bench_num}"
        if bench_key not in data:
            continue

        df = data[bench_key].copy()
        df["score"] = compute_normalized_score(df, bench_num)

        # Determine ID column (pair_id for B5, molecule_id for others)
        id_col = "pair_id" if bench_num == 5 else "molecule_id"

        # Average as before
        if bench_num == 2:
            agg_df = df.groupby([id_col, "representation", "model", "thinking", "group"])["score"].mean().reset_index()
            agg_df = agg_df.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()
        elif bench_num == 3:
            agg_df = df.groupby([id_col, "representation", "model", "thinking", "property"])["score"].mean().reset_index()
            agg_df = agg_df.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()
        else:
            agg_df = df.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()

        molecule_repr = agg_df.groupby([id_col, "representation"])["score"].mean().reset_index()
        molecule_repr["benchmark"] = bench_num
        # Rename id_col to standard name for combining
        molecule_repr = molecule_repr.rename(columns={id_col: "id"})

        comprehension_scores[bench_num] = molecule_repr

    if comprehension_scores:
        # Combine and average
        comp_combined = pd.concat(comprehension_scores.values())
        comp_avg = comp_combined.groupby(["id", "representation"])["score"].mean().reset_index()

        # Pairwise tests
        comp_results = []

        for repr1, repr2 in combinations(REPRESENTATIONS, 2):
            repr1_df = comp_avg[comp_avg["representation"] == repr1]
            repr2_df = comp_avg[comp_avg["representation"] == repr2]

            paired = repr1_df.merge(repr2_df, on="id", suffixes=("_1", "_2"))

            if len(paired) >= 3:
                try:
                    statistic, p_value = wilcoxon(
                        paired["score_1"],
                        paired["score_2"],
                        alternative="two-sided"
                    )

                    p_adjusted = min(p_value * N_COMPARISONS, 1.0)
                    significant = p_adjusted < ALPHA
                    mean_diff = paired["score_1"].mean() - paired["score_2"].mean()

                    comp_results.append({
                        "representation_1": repr1,
                        "representation_2": repr2,
                        "n_pairs": len(paired),
                        "mean_1": paired["score_1"].mean(),
                        "mean_2": paired["score_2"].mean(),
                        "mean_diff": mean_diff,
                        "statistic": statistic,
                        "p_value": p_value,
                        "p_adjusted": p_adjusted,
                        "significant": significant,
                    })

                    sig_marker = "***" if significant else ""
                    logger.info(
                        f"  {REPR_DISPLAY_NAMES[repr1]:20s} vs {REPR_DISPLAY_NAMES[repr2]:20s}: "
                        f"Δ={mean_diff:+.3f}, p={p_value:.4f}, p_adj={p_adjusted:.4f} {sig_marker}"
                    )
                except Exception as e:
                    logger.warning(f"  {repr1} vs {repr2}: test failed - {str(e)}")

        if comp_results:
            comp_df = pd.DataFrame(comp_results)
            comp_path = stats_dir / "pairwise_tests_comprehension.csv"
            comp_df.to_csv(comp_path, index=False)
            logger.info(f"  Saved: {comp_path}")

            n_significant = comp_df["significant"].sum()
            logger.info(f"  Significant differences: {n_significant}/{len(comp_results)} pairs")

    # Generation (B6)
    logger.info("\nGeneration (B6 exact match):")

    if "b6" in data:
        df = data["b6"].copy()
        df["score"] = compute_normalized_score(df, 6)

        # B6 uses molecule_id
        agg_df = df.groupby(["molecule_id", "representation", "model", "thinking"])["score"].mean().reset_index()
        gen_avg = agg_df.groupby(["molecule_id", "representation"])["score"].mean().reset_index()

        gen_results = []

        for repr1, repr2 in combinations(REPRESENTATIONS, 2):
            repr1_df = gen_avg[gen_avg["representation"] == repr1]
            repr2_df = gen_avg[gen_avg["representation"] == repr2]

            paired = repr1_df.merge(repr2_df, on="molecule_id", suffixes=("_1", "_2"))

            if len(paired) >= 3:
                try:
                    statistic, p_value = wilcoxon(
                        paired["score_1"],
                        paired["score_2"],
                        alternative="two-sided"
                    )

                    p_adjusted = min(p_value * N_COMPARISONS, 1.0)
                    significant = p_adjusted < ALPHA
                    mean_diff = paired["score_1"].mean() - paired["score_2"].mean()

                    gen_results.append({
                        "representation_1": repr1,
                        "representation_2": repr2,
                        "n_pairs": len(paired),
                        "mean_1": paired["score_1"].mean(),
                        "mean_2": paired["score_2"].mean(),
                        "mean_diff": mean_diff,
                        "statistic": statistic,
                        "p_value": p_value,
                        "p_adjusted": p_adjusted,
                        "significant": significant,
                    })

                    sig_marker = "***" if significant else ""
                    logger.info(
                        f"  {REPR_DISPLAY_NAMES[repr1]:20s} vs {REPR_DISPLAY_NAMES[repr2]:20s}: "
                        f"Δ={mean_diff:+.3f}, p={p_value:.4f}, p_adj={p_adjusted:.4f} {sig_marker}"
                    )
                except Exception as e:
                    logger.warning(f"  {repr1} vs {repr2}: test failed - {str(e)}")

        if gen_results:
            gen_df = pd.DataFrame(gen_results)
            gen_path = stats_dir / "pairwise_tests_generation.csv"
            gen_df.to_csv(gen_path, index=False)
            logger.info(f"  Saved: {gen_path}")

            n_significant = gen_df["significant"].sum()
            logger.info(f"  Significant differences: {n_significant}/{len(gen_results)} pairs")


# ============================================================================
# Test 2: Thinking Ablation Significance
# ============================================================================


def test_thinking_significance(data: Dict[str, pd.DataFrame]) -> None:
    """
    Test whether thinking significantly affects performance.

    For each (representation, benchmark), run paired Wilcoxon test
    comparing thinking ON vs OFF across molecules, averaged across models.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Test 2: Thinking Ablation Significance")
    logger.info("=" * 80 + "\n")

    stats_dir = RESULTS_DIR / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for bench_num in range(1, 8):
        bench_key = f"b{bench_num}"
        bench_name = BENCHMARKS[bench_num - 1]

        if bench_key not in data:
            logger.warning(f"Skipping B{bench_num}: no data")
            continue

        logger.info(f"\nBenchmark {bench_num}: {BENCHMARK_DISPLAY_NAMES[bench_name]}")

        df = data[bench_key].copy()
        df["score"] = compute_normalized_score(df, bench_num)

        # Determine ID column
        id_col = "pair_id" if bench_num == 5 else "molecule_id"

        # Average as before
        if bench_num == 2:
            agg_df = df.groupby([id_col, "representation", "model", "thinking", "group"])["score"].mean().reset_index()
            agg_df = agg_df.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()
        elif bench_num == 3:
            agg_df = df.groupby([id_col, "representation", "model", "thinking", "property"])["score"].mean().reset_index()
            agg_df = agg_df.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()
        else:
            agg_df = df.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()

        # Average across models
        molecule_repr_thinking = agg_df.groupby([id_col, "representation", "thinking"])["score"].mean().reset_index()

        # Test for each representation
        for repr_name in REPRESENTATIONS:
            repr_df = molecule_repr_thinking[molecule_repr_thinking["representation"] == repr_name]

            # Split by thinking
            thinking_on = repr_df[repr_df["thinking"] == True]
            thinking_off = repr_df[repr_df["thinking"] == False]

            # Merge on id column
            paired = thinking_on.merge(
                thinking_off,
                on=[id_col, "representation"],
                suffixes=("_on", "_off")
            )

            if len(paired) < 3:
                logger.warning(f"  {REPR_DISPLAY_NAMES[repr_name]}: insufficient paired samples ({len(paired)})")
                continue

            try:
                statistic, p_value = wilcoxon(
                    paired["score_on"],
                    paired["score_off"],
                    alternative="two-sided"
                )

                significant = p_value < ALPHA
                mean_on = paired["score_on"].mean()
                mean_off = paired["score_off"].mean()
                delta = mean_on - mean_off

                all_results.append({
                    "benchmark": f"B{bench_num}",
                    "benchmark_name": bench_name,
                    "representation": repr_name,
                    "n_pairs": len(paired),
                    "mean_thinking_on": mean_on,
                    "mean_thinking_off": mean_off,
                    "delta": delta,
                    "statistic": statistic,
                    "p_value": p_value,
                    "significant": significant,
                    "direction": "helps" if delta > 0 else "hurts",
                })

                sig_marker = "***" if significant else ""
                direction = "↑" if delta > 0 else "↓"
                logger.info(
                    f"  {REPR_DISPLAY_NAMES[repr_name]:20s}: "
                    f"Δ={delta:+.3f} {direction}, p={p_value:.4f} {sig_marker}"
                )

            except Exception as e:
                logger.warning(f"  {repr_name}: test failed - {str(e)}")

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = stats_dir / "thinking_significance.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nSaved: {output_path}")

        # Summary
        logger.info("\n" + "-" * 80)
        logger.info("Thinking Effect Summary")
        logger.info("-" * 80 + "\n")

        significant_df = results_df[results_df["significant"]]

        if len(significant_df) > 0:
            helps = significant_df[significant_df["delta"] > 0]
            hurts = significant_df[significant_df["delta"] < 0]

            logger.info(f"Significant positive effects (thinking helps): {len(helps)}")
            for _, row in helps.iterrows():
                logger.info(
                    f"  {row['benchmark']:3s} {REPR_DISPLAY_NAMES[row['representation']]:20s}: "
                    f"Δ={row['delta']:+.3f}, p={row['p_value']:.4f}"
                )

            logger.info(f"\nSignificant negative effects (thinking hurts): {len(hurts)}")
            for _, row in hurts.iterrows():
                logger.info(
                    f"  {row['benchmark']:3s} {REPR_DISPLAY_NAMES[row['representation']]:20s}: "
                    f"Δ={row['delta']:+.3f}, p={row['p_value']:.4f}"
                )
        else:
            logger.info("No significant thinking effects found")


# ============================================================================
# Test 3: Model Differences
# ============================================================================


def test_model_differences(data: Dict[str, pd.DataFrame]) -> None:
    """
    Test whether different models perform significantly differently.

    For each (representation, benchmark), run Friedman test across 3 models.
    If significant, run post-hoc pairwise Wilcoxon tests.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Test 3: Model Differences")
    logger.info("=" * 80 + "\n")

    stats_dir = RESULTS_DIR / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    model_ids = [m["id"] for m in MODELS]

    all_results = []

    for bench_num in range(1, 8):
        bench_key = f"b{bench_num}"
        bench_name = BENCHMARKS[bench_num - 1]

        if bench_key not in data:
            logger.warning(f"Skipping B{bench_num}: no data")
            continue

        logger.info(f"\nBenchmark {bench_num}: {BENCHMARK_DISPLAY_NAMES[bench_name]}")

        df = data[bench_key].copy()
        df["score"] = compute_normalized_score(df, bench_num)

        # Determine ID column (pair_id for B5, molecule_id for others)
        id_col = "pair_id" if bench_num == 5 else "molecule_id"

        # Average as before
        if bench_num == 2:
            agg_df = df.groupby([id_col, "representation", "model", "thinking", "group"])["score"].mean().reset_index()
            agg_df = agg_df.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()
        elif bench_num == 3:
            agg_df = df.groupby([id_col, "representation", "model", "thinking", "property"])["score"].mean().reset_index()
            agg_df = agg_df.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()
        else:
            agg_df = df.groupby([id_col, "representation", "model", "thinking"])["score"].mean().reset_index()

        # Average across thinking
        molecule_repr_model = agg_df.groupby([id_col, "representation", "model"])["score"].mean().reset_index()

        # Test for each representation
        for repr_name in REPRESENTATIONS:
            repr_df = molecule_repr_model[molecule_repr_model["representation"] == repr_name]

            # Get scores for each model
            model_scores = []
            for model_id in model_ids:
                model_df = repr_df[repr_df["model"] == model_id]
                if len(model_df) > 0:
                    model_scores.append(model_df.set_index(id_col)["score"])
                else:
                    model_scores.append(None)

            # Check if we have data for all models
            if any(s is None for s in model_scores):
                logger.warning(f"  {REPR_DISPLAY_NAMES[repr_name]}: missing data for some models")
                continue

            # Align by molecule_id
            try:
                aligned = pd.DataFrame({
                    model_ids[0]: model_scores[0],
                    model_ids[1]: model_scores[1],
                    model_ids[2]: model_scores[2],
                })

                # Remove rows with NaN
                aligned = aligned.dropna()

                if len(aligned) < 3:
                    logger.warning(f"  {REPR_DISPLAY_NAMES[repr_name]}: insufficient samples ({len(aligned)})")
                    continue

                # Friedman test
                statistic, p_value = friedmanchisquare(
                    aligned[model_ids[0]],
                    aligned[model_ids[1]],
                    aligned[model_ids[2]],
                )

                significant = p_value < ALPHA

                means = [aligned[mid].mean() for mid in model_ids]

                result = {
                    "benchmark": f"B{bench_num}",
                    "benchmark_name": bench_name,
                    "representation": repr_name,
                    "n_samples": len(aligned),
                    "mean_model_1": means[0],
                    "mean_model_2": means[1],
                    "mean_model_3": means[2],
                    "friedman_statistic": statistic,
                    "p_value": p_value,
                    "significant": significant,
                }

                # If significant, do post-hoc pairwise tests
                if significant:
                    pairwise_results = []

                    for i, j in combinations(range(3), 2):
                        stat_pw, p_pw = wilcoxon(
                            aligned[model_ids[i]],
                            aligned[model_ids[j]],
                            alternative="two-sided"
                        )

                        # Bonferroni for 3 comparisons
                        p_adj = min(p_pw * 3, 1.0)

                        pairwise_results.append({
                            "model_1": model_ids[i],
                            "model_2": model_ids[j],
                            "p_value": p_pw,
                            "p_adjusted": p_adj,
                            "significant": p_adj < ALPHA,
                        })

                    # Add to result
                    for idx, pw in enumerate(pairwise_results):
                        result[f"pairwise_{idx+1}_models"] = f"{pw['model_1']}_vs_{pw['model_2']}"
                        result[f"pairwise_{idx+1}_p_adjusted"] = pw["p_adjusted"]
                        result[f"pairwise_{idx+1}_significant"] = pw["significant"]

                all_results.append(result)

                sig_marker = "***" if significant else ""
                logger.info(
                    f"  {REPR_DISPLAY_NAMES[repr_name]:20s}: "
                    f"χ²={statistic:.2f}, p={p_value:.4f} {sig_marker}"
                )

                if significant:
                    for pw in pairwise_results:
                        if pw["significant"]:
                            logger.info(
                                f"    {pw['model_1']} vs {pw['model_2']}: "
                                f"p_adj={pw['p_adjusted']:.4f} ***"
                            )

            except Exception as e:
                logger.warning(f"  {repr_name}: test failed - {str(e)}")

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = stats_dir / "model_differences.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nSaved: {output_path}")

        # Summary
        significant_df = results_df[results_df["significant"]]
        logger.info(f"\nSignificant model differences: {len(significant_df)}/{len(results_df)}")


# ============================================================================
# Main Function
# ============================================================================


def main():
    """Run statistical tests."""
    parser = argparse.ArgumentParser(
        description="Run statistical significance tests on benchmark results"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "pairwise", "thinking", "models"],
        help="Which test to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests",
    )

    args = parser.parse_args()

    # Load data
    logger.info("Loading scored data...")
    data = load_scored_data()

    if len(data) == 0:
        logger.error("No data loaded! Run evaluation script first.")
        return

    # Run tests
    tests_to_run = ["pairwise", "thinking", "models"] if args.all or args.test == "all" else [args.test]

    if "pairwise" in tests_to_run:
        test_pairwise_representations(data)

    if "thinking" in tests_to_run:
        test_thinking_significance(data)

    if "models" in tests_to_run:
        test_model_differences(data)

    logger.info("\n" + "=" * 80)
    logger.info("✅ Statistical testing complete!")
    logger.info(f"Results saved to: {RESULTS_DIR / 'statistics'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
