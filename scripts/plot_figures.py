"""
04_plot_figures.py - Generate all 17 figures for the molecular representation benchmark

This script reads the evaluated results (scored CSVs and metrics CSVs) and generates
all figures specified in docs/molecular_benchmark_spec.md.

All configuration (colors, sizes, model markers) is imported from config.py.

Usage:
    python scripts/04_plot_figures.py --all
    python scripts/04_plot_figures.py --figure fig1_main_heatmap
    python scripts/04_plot_figures.py --figure fig3_comprehension_vs_generation_scatter
"""

import sys
from pathlib import Path
import argparse
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional

# Import configuration
from config import (
    REPRESENTATIONS,
    REPR_DISPLAY_NAMES,
    REPR_COLORS,
    MODELS,
    MODEL_NAMES,
    BENCHMARKS,
    BENCHMARK_DISPLAY_NAMES,
    RESULTS_DIR,
    FIGURES_DIR,
    PLOT_STYLE,
    FONT_SCALE,
    PLOT_DPI,
    FIGURE_SIZES,
    HEATMAP_CMAP,
    THINKING_STYLES,
    FUNCTIONAL_GROUPS,
    PROPERTIES,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Set global plotting style
sns.set_style(PLOT_STYLE)
plt.rcParams["font.size"] = 12
sns.set_context("paper", font_scale=FONT_SCALE)


# ============================================================================
# Helper Functions
# ============================================================================


def save_figure(fig: plt.Figure, name: str) -> None:
    """
    Save a figure to both PDF and PNG formats.

    Args:
        fig: Matplotlib Figure object
        name: Figure name (without extension)
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"

    fig.savefig(pdf_path, dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")

    logger.info(f"Saved figure: {name} (.pdf and .png)")


def load_data() -> Dict[str, pd.DataFrame]:
    """
    Load all scored CSVs and metrics CSVs.

    Returns:
        Dictionary mapping data keys to DataFrames
    """
    data = {}

    # Load scored CSVs (row-level with computed metrics)
    for i in range(1, 8):
        scored_path = RESULTS_DIR / "scored" / f"benchmark_{i}_scored.csv"
        if scored_path.exists():
            key = f"b{i}_scored"
            data[key] = pd.read_csv(scored_path)
            logger.info(f"Loaded {key}: {len(data[key])} rows")
        else:
            logger.warning(f"Missing: {scored_path}")

    # Load metrics CSVs (aggregated per condition)
    for i in range(1, 8):
        metrics_path = RESULTS_DIR / "metrics" / f"benchmark_{i}_metrics.csv"
        if metrics_path.exists():
            key = f"b{i}_metrics"
            data[key] = pd.read_csv(metrics_path)
            logger.info(f"Loaded {key}: {len(data[key])} rows")
        else:
            logger.warning(f"Missing: {metrics_path}")

    # Load B2 macro metrics
    b2_macro_path = RESULTS_DIR / "metrics" / "benchmark_2_macro_metrics.csv"
    if b2_macro_path.exists():
        data["b2_macro"] = pd.read_csv(b2_macro_path)
        logger.info(f"Loaded b2_macro: {len(data['b2_macro'])} rows")

    # Load cross-cutting analyses
    cross_files = [
        "comprehension_vs_generation.csv",
        "thinking_ablation.csv",
        "representation_rankings.csv",
        "representation_mean_ranks.csv",
    ]

    for filename in cross_files:
        cross_path = RESULTS_DIR / "cross" / filename
        if cross_path.exists():
            key = filename.replace(".csv", "")
            data[key] = pd.read_csv(cross_path)
            logger.info(f"Loaded {key}: {len(data[key])} rows")
        else:
            logger.warning(f"Missing: {cross_path}")

    return data


def normalize_benchmark_score(df: pd.DataFrame, benchmark_num: int, metric_col: str) -> pd.Series:
    """
    Normalize benchmark scores to 0-1 scale.

    Args:
        df: DataFrame with metric values
        benchmark_num: Benchmark number (1-7)
        metric_col: Column name containing the metric

    Returns:
        Normalized scores (0-1)
    """
    if benchmark_num == 3:
        # B3: 1 - (MAE / max_MAE)
        max_mae = df[metric_col].max()
        if max_mae > 0:
            return 1 - (df[metric_col] / max_mae)
        else:
            return pd.Series([1.0] * len(df))
    else:
        # All others are already 0-1 (accuracy, F1, exact_match, recovery)
        return df[metric_col]


# ============================================================================
# Figure 1: Main Results Heatmap
# ============================================================================


def plot_fig1_main_heatmap(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 1: Main Results Heatmap

    Annotated heatmap showing normalized performance (0-1) for each
    representation × benchmark combination, averaged across all models
    and thinking conditions.

    Spec reference: Figure 1
    """
    # Collect data for heatmap
    heatmap_data = []

    for i, bench_name in enumerate(BENCHMARKS, 1):
        key = f"b{i}_metrics"
        if key not in data:
            continue

        df = data[key].copy()

        # Determine metric column based on benchmark
        if i == 1:
            metric_col = "accuracy"
        elif i == 2:
            # Use macro metrics
            if "b2_macro" in data:
                df = data["b2_macro"].copy()
                metric_col = "macro_f1"
            else:
                continue
        elif i == 3:
            metric_col = "mae"
        elif i == 4:
            metric_col = "accuracy"
        elif i == 5:
            metric_col = "f1"
        elif i == 6:
            metric_col = "exact_match_rate"
        elif i == 7:
            metric_col = "recovery_rate"
        else:
            continue

        # Normalize scores
        df["normalized_score"] = normalize_benchmark_score(df, i, metric_col)

        # Average across models and thinking
        repr_scores = df.groupby("representation")["normalized_score"].mean()

        for repr_name in REPRESENTATIONS:
            if repr_name in repr_scores.index:
                heatmap_data.append({
                    "representation": repr_name,
                    "benchmark": BENCHMARK_DISPLAY_NAMES[bench_name],
                    "score": repr_scores[repr_name],
                })

    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(heatmap_data)

    # Pivot to matrix form
    heatmap_matrix = heatmap_df.pivot(
        index="representation",
        columns="benchmark",
        values="score"
    )

    # Reorder representations to match config order
    heatmap_matrix = heatmap_matrix.reindex(REPRESENTATIONS)

    # Map to display names
    heatmap_matrix.index = [REPR_DISPLAY_NAMES[r] for r in heatmap_matrix.index]

    # Create figure
    figsize = FIGURE_SIZES["fig1_main_heatmap"]
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".2f",
        cmap=HEATMAP_CMAP,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Normalized Score"},
        ax=ax,
    )

    ax.set_xlabel("Benchmark", fontsize=12, fontweight="bold")
    ax.set_ylabel("Representation", fontsize=12, fontweight="bold")
    ax.set_title("Main Results: Representation Performance Across All Benchmarks",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 2: Generation Performance Bar Plot
# ============================================================================


def plot_fig2_generation_barplot(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 2: Generation Performance Bar Plot

    Grouped bar chart showing validity, exact match, and Tanimoto for
    Benchmark 6 (generation), with 3 subplots (one per model).

    Spec reference: Figure 2
    """
    if "b6_metrics" not in data:
        logger.warning("Missing B6 metrics for Figure 2")
        return plt.figure()

    df = data["b6_metrics"].copy()

    # Create 3 subplots (one per model)
    figsize = FIGURE_SIZES["fig2_generation_barplot"]
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    model_ids = [m["id"] for m in MODELS]

    for ax_idx, model_id in enumerate(model_ids):
        ax = axes[ax_idx]
        model_name = MODEL_NAMES[model_id]

        # Filter for this model
        model_df = df[df["model"] == model_id].copy()

        if len(model_df) == 0:
            ax.text(0.5, 0.5, f"No data for {model_name}",
                   ha="center", va="center", transform=ax.transAxes)
            continue

        # Setup x positions
        x = np.arange(len(REPRESENTATIONS))
        width = 0.12  # Width of each bar

        metrics = ["validity_rate", "exact_match_rate", "mean_tanimoto"]
        metric_labels = ["Validity", "Exact Match", "Tanimoto"]

        for thinking_idx, thinking in enumerate([False, True]):
            thinking_df = model_df[model_df["thinking"] == thinking]

            for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
                offset = (thinking_idx * len(metrics) + metric_idx - 2.5) * width

                values = []
                for repr_name in REPRESENTATIONS:
                    repr_df = thinking_df[thinking_df["representation"] == repr_name]
                    if len(repr_df) > 0:
                        values.append(repr_df[metric].iloc[0])
                    else:
                        values.append(0)

                # Bar style: solid for thinking ON, hatched for thinking OFF
                if thinking:
                    hatch = None
                    alpha = 0.8
                    label_suffix = " (Thinking ON)"
                else:
                    hatch = "///"
                    alpha = 0.6
                    label_suffix = " (Thinking OFF)"

                # Only add label for first subplot
                if ax_idx == 0:
                    bar_label = label + label_suffix
                else:
                    bar_label = None

                ax.bar(
                    x + offset,
                    values,
                    width,
                    label=bar_label,
                    alpha=alpha,
                    hatch=hatch,
                    color=f"C{metric_idx}",
                )

        ax.set_xlabel("Representation", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Score", fontsize=10)
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([REPR_DISPLAY_NAMES[r] for r in REPRESENTATIONS],
                          rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    # Add legend to first subplot
    if len(model_df) > 0:
        axes[0].legend(loc="upper left", fontsize=7, ncol=1)

    fig.suptitle("Generation Performance (Benchmark 6)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig


# ============================================================================
# Figure 3: Comprehension vs Generation Scatter
# ============================================================================


def plot_fig3_comprehension_vs_generation_scatter(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 3: Comprehension vs Generation Scatter

    Scatter plot showing correlation between mean comprehension score
    (average of B1-B5) and generation exact match (B6).

    Spec reference: Figure 3
    """
    if "comprehension_vs_generation" not in data:
        logger.warning("Missing comprehension vs generation data for Figure 3")
        return plt.figure()

    df = data["comprehension_vs_generation"].copy()

    figsize = FIGURE_SIZES["fig3_comprehension_vs_generation_scatter"]
    fig, ax = plt.subplots(figsize=figsize)

    # Plot points colored by representation, shaped by model, filled by thinking
    for repr_name in REPRESENTATIONS:
        repr_df = df[df["representation"] == repr_name]

        for model_dict in MODELS:
            model_id = model_dict["id"]
            marker = model_dict["marker"]

            model_df = repr_df[repr_df["model"] == model_id]

            for thinking in [False, True]:
                thinking_df = model_df[model_df["thinking"] == thinking]

                if len(thinking_df) == 0:
                    continue

                # Marker style: filled for thinking ON, open for thinking OFF
                if thinking:
                    facecolor = REPR_COLORS[repr_name]
                    edgecolor = REPR_COLORS[repr_name]
                else:
                    facecolor = "none"
                    edgecolor = REPR_COLORS[repr_name]

                ax.scatter(
                    thinking_df["mean_comprehension"],
                    thinking_df["mean_generation"],
                    marker=marker,
                    s=100,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidths=2,
                    alpha=0.7,
                    label=None,  # We'll add a custom legend
                )

    # Add regression line
    valid_data = df[df["mean_comprehension"].notna() & df["mean_generation"].notna()]
    if len(valid_data) > 2:
        x = valid_data["mean_comprehension"].values
        y = valid_data["mean_generation"].values

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Plot regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", linewidth=2, alpha=0.5, label="Linear fit")

        # Add correlation annotation
        ax.text(
            0.05, 0.95,
            f"Pearson r = {r_value:.3f}\np = {p_value:.3e}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Mean Comprehension Score (B1-B5)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Generation Exact Match (B6)", fontsize=12, fontweight="bold")
    ax.set_title("Does Comprehension Predict Generation?", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    # Create custom legend
    legend_elements = []
    for repr_name in REPRESENTATIONS:
        legend_elements.append(
            mpatches.Patch(facecolor=REPR_COLORS[repr_name],
                          label=REPR_DISPLAY_NAMES[repr_name])
        )

    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, ncol=2)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 4: Thinking Ablation Delta Plot
# ============================================================================


def plot_fig4_thinking_ablation_delta(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 4: Thinking Ablation Delta Plot

    Diverging horizontal bar chart showing Δ = (thinking ON) - (thinking OFF)
    for each benchmark, representation, and model.

    Spec reference: Figure 4
    """
    if "thinking_ablation" not in data:
        logger.warning("Missing thinking ablation data for Figure 4")
        return plt.figure()

    df = data["thinking_ablation"].copy()

    figsize = FIGURE_SIZES["fig4_thinking_ablation_delta"]
    fig, axes = plt.subplots(7, 1, figsize=figsize, sharex=True)

    benchmarks = [f"B{i}" for i in range(1, 8)]

    for ax_idx, benchmark in enumerate(benchmarks):
        ax = axes[ax_idx]
        bench_df = df[df["benchmark"] == benchmark]

        if len(bench_df) == 0:
            ax.text(0.5, 0.5, f"No data for {benchmark}",
                   ha="center", va="center", transform=ax.transAxes)
            continue

        # Setup y positions
        y = np.arange(len(REPRESENTATIONS))
        height = 0.25

        for model_idx, model_dict in enumerate(MODELS):
            model_id = model_dict["id"]
            offset = (model_idx - 1) * height

            deltas = []
            for repr_name in REPRESENTATIONS:
                repr_df = bench_df[
                    (bench_df["representation"] == repr_name) &
                    (bench_df["model"] == model_id)
                ]
                if len(repr_df) > 0:
                    deltas.append(repr_df["delta"].iloc[0])
                else:
                    deltas.append(0)

            # Color bars by model
            bars = ax.barh(
                y + offset,
                deltas,
                height,
                label=MODEL_NAMES[model_id] if ax_idx == 0 else None,
                color=f"C{model_idx}",
                alpha=0.7,
            )

            # Add delta values at end of bars
            for bar, delta in zip(bars, deltas):
                if abs(delta) > 0.001:
                    x_pos = delta + (0.01 if delta > 0 else -0.01)
                    ax.text(
                        x_pos,
                        bar.get_y() + bar.get_height() / 2,
                        f"{delta:.2f}",
                        ha="left" if delta > 0 else "right",
                        va="center",
                        fontsize=7,
                    )

        ax.axvline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([REPR_DISPLAY_NAMES[r] for r in REPRESENTATIONS], fontsize=9)
        ax.set_ylabel(benchmark, fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=9, ncol=3)
    axes[-1].set_xlabel("Δ Score (Thinking ON - Thinking OFF)", fontsize=12, fontweight="bold")
    fig.suptitle("Thinking Ablation: Effect Across Benchmarks", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 5: Representation Radar Plot
# ============================================================================


def plot_fig5_representation_radar(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 5: Representation Radar Plot

    Spider/radar chart showing the strength profile of each representation
    across all 7 benchmarks.

    Spec reference: Figure 5
    """
    # Collect normalized scores for each representation × benchmark
    radar_data = {repr_name: [] for repr_name in REPRESENTATIONS}
    benchmark_labels = []

    for i, bench_name in enumerate(BENCHMARKS, 1):
        key = f"b{i}_metrics"
        if key not in data:
            continue

        df = data[key].copy()

        # Determine metric column
        if i == 1:
            metric_col = "accuracy"
        elif i == 2:
            if "b2_macro" in data:
                df = data["b2_macro"].copy()
                metric_col = "macro_f1"
            else:
                continue
        elif i == 3:
            metric_col = "mae"
        elif i == 4:
            metric_col = "accuracy"
        elif i == 5:
            metric_col = "f1"
        elif i == 6:
            metric_col = "exact_match_rate"
        elif i == 7:
            metric_col = "recovery_rate"
        else:
            continue

        # Normalize and average across models and thinking
        df["normalized_score"] = normalize_benchmark_score(df, i, metric_col)
        repr_scores = df.groupby("representation")["normalized_score"].mean()

        for repr_name in REPRESENTATIONS:
            if repr_name in repr_scores.index:
                radar_data[repr_name].append(repr_scores[repr_name])
            else:
                radar_data[repr_name].append(0)

        benchmark_labels.append(f"B{i}")

    # Create radar chart
    figsize = FIGURE_SIZES["fig5_representation_radar"]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")

    # Number of variables
    num_vars = len(benchmark_labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the circle
    angles += angles[:1]

    # Plot each representation
    for repr_name in REPRESENTATIONS:
        values = radar_data[repr_name]
        values += values[:1]  # Complete the circle

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=REPR_DISPLAY_NAMES[repr_name],
            color=REPR_COLORS[repr_name],
        )
        ax.fill(angles, values, alpha=0.15, color=REPR_COLORS[repr_name])

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(benchmark_labels, fontsize=11)

    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.5)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.title("Representation Strength Profiles", fontsize=14, fontweight="bold", pad=20)

    return fig


# ============================================================================
# Figure 6: Atom Counting Accuracy by Molecule Size
# ============================================================================


def plot_fig6_atom_counting_by_size(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 6: Atom Counting Accuracy by Molecule Size

    Line plot showing how atom counting accuracy varies with molecular size
    (number of heavy atoms), with error bands.

    Spec reference: Figure 6
    """
    if "b1_scored" not in data:
        logger.warning("Missing B1 scored data for Figure 6")
        return plt.figure()

    df = data["b1_scored"].copy()

    # Bin by heavy atoms (assuming num_heavy_atoms column exists)
    if "num_heavy_atoms" not in df.columns:
        logger.warning("Missing num_heavy_atoms column in B1 scored data")
        return plt.figure()

    figsize = FIGURE_SIZES["fig6_atom_counting_by_size"]
    fig, ax = plt.subplots(figsize=figsize)

    # Define bins
    bins = [5, 10, 15, 20, 25, 30, 35, 100]
    bin_labels = ["5-10", "11-15", "16-20", "21-25", "26-30", "31-35", "35+"]

    df["size_bin"] = pd.cut(df["num_heavy_atoms"], bins=bins, labels=bin_labels)

    # Plot line for each representation
    for repr_name in REPRESENTATIONS:
        repr_df = df[df["representation"] == repr_name]

        # Compute accuracy per bin
        bin_stats = []
        for bin_label in bin_labels:
            bin_df = repr_df[repr_df["size_bin"] == bin_label]
            if len(bin_df) > 0:
                mean_acc = bin_df["correct"].mean()
                std_acc = bin_df["correct"].std()
                bin_stats.append({"bin": bin_label, "mean": mean_acc, "std": std_acc})
            else:
                bin_stats.append({"bin": bin_label, "mean": np.nan, "std": np.nan})

        bin_df = pd.DataFrame(bin_stats)

        # Plot line with error band
        x = np.arange(len(bin_labels))
        ax.plot(
            x,
            bin_df["mean"],
            marker="o",
            linewidth=2,
            label=REPR_DISPLAY_NAMES[repr_name],
            color=REPR_COLORS[repr_name],
        )

        # Add error band (±1 std)
        ax.fill_between(
            x,
            bin_df["mean"] - bin_df["std"],
            bin_df["mean"] + bin_df["std"],
            alpha=0.2,
            color=REPR_COLORS[repr_name],
        )

    ax.set_xlabel("Number of Heavy Atoms", fontsize=12, fontweight="bold")
    ax.set_ylabel("Atom Counting Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy vs Molecular Complexity", fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 7: Functional Group Breakdown
# ============================================================================


def plot_fig7_functional_group_breakdown(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 7: Functional Group Breakdown

    Grouped bar chart showing F1 score for each of 10 functional groups
    across all representations.

    Spec reference: Figure 7
    """
    if "b2_metrics" not in data:
        logger.warning("Missing B2 metrics for Figure 7")
        return plt.figure()

    df = data["b2_metrics"].copy()

    figsize = FIGURE_SIZES["fig7_functional_group_breakdown"]
    fig, ax = plt.subplots(figsize=figsize)

    # Average across models and thinking
    group_scores = df.groupby(["representation", "group"])["f1"].mean().reset_index()

    # Setup plot
    x = np.arange(len(FUNCTIONAL_GROUPS))
    width = 0.13

    for repr_idx, repr_name in enumerate(REPRESENTATIONS):
        repr_df = group_scores[group_scores["representation"] == repr_name]

        values = []
        for fg_key in FUNCTIONAL_GROUPS.keys():
            fg_df = repr_df[repr_df["group"] == fg_key]
            if len(fg_df) > 0:
                values.append(fg_df["f1"].iloc[0])
            else:
                values.append(0)

        offset = (repr_idx - 2.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=REPR_DISPLAY_NAMES[repr_name],
            color=REPR_COLORS[repr_name],
            alpha=0.8,
        )

    ax.set_xlabel("Functional Group", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    ax.set_title("Functional Group Identification Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)

    # Get functional group display names
    fg_labels = [fg_config["name"] for fg_config in FUNCTIONAL_GROUPS.values()]
    ax.set_xticklabels(fg_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 8: Property Estimation Scatter
# ============================================================================


def plot_fig8_property_estimation_scatter(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 8: Property Estimation Scatter

    2×6 grid of scatter plots showing predicted vs ground truth for
    LogP and TPSA across all representations.

    Spec reference: Figure 8
    """
    if "b3_scored" not in data:
        logger.warning("Missing B3 scored data for Figure 8")
        return plt.figure()

    df = data["b3_scored"].copy()

    figsize = FIGURE_SIZES["fig8_property_estimation_scatter"]
    fig, axes = plt.subplots(2, 6, figsize=figsize, sharex="row", sharey="row")

    properties = ["logp", "tpsa"]

    for prop_idx, prop_key in enumerate(properties):
        prop_df = df[df["property"] == prop_key]

        for repr_idx, repr_name in enumerate(REPRESENTATIONS):
            ax = axes[prop_idx, repr_idx]
            repr_df = prop_df[prop_df["representation"] == repr_name]

            if len(repr_df) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                       transform=ax.transAxes, fontsize=8)
                continue

            # Remove NaN values
            valid_df = repr_df[repr_df["predicted"].notna() & repr_df["ground_truth"].notna()]

            if len(valid_df) < 2:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                       transform=ax.transAxes, fontsize=8)
                continue

            x = valid_df["ground_truth"].values
            y = valid_df["predicted"].values

            # Scatter plot
            ax.scatter(x, y, alpha=0.5, s=10, color=REPR_COLORS[repr_name])

            # Identity line
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, alpha=0.5)

            # Compute MAE and Spearman
            mae = np.mean(np.abs(x - y))
            if len(x) > 2:
                spearman_r, _ = stats.spearmanr(x, y)
                stat_text = f"MAE={mae:.2f}\nρ={spearman_r:.2f}"
            else:
                stat_text = f"MAE={mae:.2f}"

            ax.text(
                0.05, 0.95,
                stat_text,
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, pad=0.3),
            )

            # Labels
            if repr_idx == 0:
                ax.set_ylabel(f"{PROPERTIES[prop_key]['display']}\nPredicted", fontsize=9)
            if prop_idx == 1:
                ax.set_xlabel("Ground Truth", fontsize=9)
            if prop_idx == 0:
                ax.set_title(REPR_DISPLAY_NAMES[repr_name], fontsize=9, fontweight="bold")

            ax.grid(alpha=0.3)

    fig.suptitle("Property Estimation: Predicted vs Ground Truth",
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig


# ============================================================================
# Figure 9: Generation Validity by Molecular Complexity
# ============================================================================


def plot_fig9_validity_by_complexity(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 9: Generation Validity by Molecular Complexity

    Two panels showing how generation validity varies with molecular complexity
    (number of rings and number of heavy atoms).

    Spec reference: Figure 9
    """
    if "b6_scored" not in data:
        logger.warning("Missing B6 scored data for Figure 9")
        return plt.figure()

    df = data["b6_scored"].copy()

    figsize = FIGURE_SIZES["fig9_validity_by_complexity"]
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left panel: Number of rings
    if "num_rings" in df.columns:
        ax = axes[0]

        # Bin rings
        df["ring_bin"] = df["num_rings"].clip(upper=4)
        ring_labels = ["0", "1", "2", "3", "4+"]

        for repr_name in REPRESENTATIONS:
            repr_df = df[df["representation"] == repr_name]

            ring_validity = []
            for ring_count in range(5):
                ring_df = repr_df[repr_df["ring_bin"] == ring_count]
                if len(ring_df) > 0:
                    validity = ring_df["valid"].mean()
                    ring_validity.append(validity)
                else:
                    ring_validity.append(np.nan)

            ax.plot(
                range(5),
                ring_validity,
                marker="o",
                linewidth=2,
                label=REPR_DISPLAY_NAMES[repr_name],
                color=REPR_COLORS[repr_name],
            )

        ax.set_xlabel("Number of Rings", fontsize=11, fontweight="bold")
        ax.set_ylabel("Validity Rate", fontsize=11, fontweight="bold")
        ax.set_title("Validity vs Ring Count", fontsize=12, fontweight="bold")
        ax.set_xticks(range(5))
        ax.set_xticklabels(ring_labels)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)

    # Right panel: Number of heavy atoms
    if "num_heavy_atoms" in df.columns:
        ax = axes[1]

        # Bin by heavy atoms
        bins = [0, 10, 20, 30, 40, 100]
        bin_labels = ["<10", "10-20", "20-30", "30-40", "40+"]
        df["atom_bin"] = pd.cut(df["num_heavy_atoms"], bins=bins, labels=bin_labels)

        for repr_name in REPRESENTATIONS:
            repr_df = df[df["representation"] == repr_name]

            atom_validity = []
            for bin_label in bin_labels:
                bin_df = repr_df[repr_df["atom_bin"] == bin_label]
                if len(bin_df) > 0:
                    validity = bin_df["valid"].mean()
                    atom_validity.append(validity)
                else:
                    atom_validity.append(np.nan)

            ax.plot(
                range(len(bin_labels)),
                atom_validity,
                marker="o",
                linewidth=2,
                label=REPR_DISPLAY_NAMES[repr_name],
                color=REPR_COLORS[repr_name],
            )

        ax.set_xlabel("Number of Heavy Atoms", fontsize=11, fontweight="bold")
        ax.set_ylabel("Validity Rate", fontsize=11, fontweight="bold")
        ax.set_title("Validity vs Heavy Atoms", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 10: Generation Tanimoto Violin Plot
# ============================================================================


def plot_fig10_generation_tanimoto_violin(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 10: Generation Tanimoto Violin Plot

    Split violin plot showing distribution of Tanimoto similarity for
    valid generations, split by thinking ON/OFF.

    Spec reference: Figure 10
    """
    if "b6_scored" not in data:
        logger.warning("Missing B6 scored data for Figure 10")
        return plt.figure()

    df = data["b6_scored"].copy()

    # Filter to valid generations only
    df = df[df["valid"] == True].copy()

    figsize = FIGURE_SIZES["fig10_generation_tanimoto_violin"]
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for split violin
    # We need to create a DataFrame suitable for seaborn
    plot_df = df[["representation", "thinking", "tanimoto"]].copy()
    plot_df["representation_display"] = plot_df["representation"].map(REPR_DISPLAY_NAMES)
    plot_df["thinking_str"] = plot_df["thinking"].map({True: "ON", False: "OFF"})

    # Create split violin plot
    sns.violinplot(
        data=plot_df,
        x="representation_display",
        y="tanimoto",
        hue="thinking_str",
        split=True,
        inner="quartile",
        palette={"OFF": "lightblue", "ON": "lightcoral"},
        ax=ax,
    )

    ax.set_xlabel("Representation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tanimoto Similarity", fontsize=12, fontweight="bold")
    ax.set_title("Distribution of Generation Quality (Valid Molecules Only)",
                fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Thinking", loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 11: Thinking × Benchmark Interaction
# ============================================================================


def plot_fig11_thinking_vs_benchmark_interaction(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 11: Thinking × Benchmark Interaction

    Line plot showing mean performance across all benchmarks,
    comparing thinking ON vs OFF.

    Spec reference: Figure 11
    """
    # Collect normalized scores for each benchmark × thinking condition
    interaction_data = []

    for i, bench_name in enumerate(BENCHMARKS, 1):
        key = f"b{i}_metrics"
        if key not in data:
            continue

        df = data[key].copy()

        # Determine metric column
        if i == 1:
            metric_col = "accuracy"
        elif i == 2:
            if "b2_macro" in data:
                df = data["b2_macro"].copy()
                metric_col = "macro_f1"
            else:
                continue
        elif i == 3:
            metric_col = "mae"
        elif i == 4:
            metric_col = "accuracy"
        elif i == 5:
            metric_col = "f1"
        elif i == 6:
            metric_col = "exact_match_rate"
        elif i == 7:
            metric_col = "recovery_rate"
        else:
            continue

        # Normalize scores
        df["normalized_score"] = normalize_benchmark_score(df, i, metric_col)

        # Average across models and representations, grouped by thinking
        for thinking in [False, True]:
            thinking_df = df[df["thinking"] == thinking]
            mean_score = thinking_df["normalized_score"].mean()
            std_score = thinking_df["normalized_score"].std()

            interaction_data.append({
                "benchmark": f"B{i}",
                "thinking": thinking,
                "mean_score": mean_score,
                "std_score": std_score,
            })

    interaction_df = pd.DataFrame(interaction_data)

    figsize = FIGURE_SIZES["fig11_thinking_vs_benchmark_interaction"]
    fig, ax = plt.subplots(figsize=figsize)

    # Plot lines for thinking ON and OFF
    for thinking in [False, True]:
        thinking_df = interaction_df[interaction_df["thinking"] == thinking]

        if thinking:
            linestyle = "-"
            marker = "o"
            label = "Thinking ON"
            color = "darkred"
        else:
            linestyle = "--"
            marker = "o"
            markerfacecolor = "none"
            label = "Thinking OFF"
            color = "darkblue"

        x = range(len(thinking_df))
        y = thinking_df["mean_score"].values
        yerr = thinking_df["std_score"].values

        if thinking:
            ax.plot(x, y, linestyle=linestyle, marker=marker, linewidth=2,
                   markersize=8, label=label, color=color)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)
        else:
            ax.plot(x, y, linestyle=linestyle, marker=marker, linewidth=2,
                   markersize=8, markerfacecolor="none", markeredgewidth=2,
                   label=label, color=color)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.1, color=color)

    ax.set_xlabel("Benchmark", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Normalized Score", fontsize=12, fontweight="bold")
    ax.set_title("Thinking Effect Across Benchmarks", fontsize=14, fontweight="bold")
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"B{i}" for i in range(1, 8)])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 12: Isomer Discrimination Breakdown
# ============================================================================


def plot_fig12_isomer_discrimination_breakdown(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 12: Isomer Discrimination Breakdown

    Grouped bar chart showing accuracy for different pair types
    (positive, stereoisomer, substitution) across representations.

    Spec reference: Figure 12
    """
    if "b5_metrics" not in data:
        logger.warning("Missing B5 metrics for Figure 12")
        return plt.figure()

    df = data["b5_metrics"].copy()

    # Check if pair_type breakdown exists
    if "pair_type" not in df.columns:
        logger.warning("Missing pair_type column in B5 metrics")
        return plt.figure()

    figsize = FIGURE_SIZES["fig12_isomer_discrimination_breakdown"]
    fig, ax = plt.subplots(figsize=figsize)

    # Average across models and thinking
    pair_scores = df.groupby(["representation", "pair_type"])["accuracy"].mean().reset_index()

    # Get unique pair types
    pair_types = pair_scores["pair_type"].unique()

    x = np.arange(len(pair_types))
    width = 0.13

    for repr_idx, repr_name in enumerate(REPRESENTATIONS):
        repr_df = pair_scores[pair_scores["representation"] == repr_name]

        values = []
        for pair_type in pair_types:
            type_df = repr_df[repr_df["pair_type"] == pair_type]
            if len(type_df) > 0:
                values.append(type_df["accuracy"].iloc[0])
            else:
                values.append(0)

        offset = (repr_idx - 2.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=REPR_DISPLAY_NAMES[repr_name],
            color=REPR_COLORS[repr_name],
            alpha=0.8,
        )

    ax.set_xlabel("Pair Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Isomer Discrimination by Pair Type", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_types, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 13: Completion Validity vs Recovery
# ============================================================================


def plot_fig13_completion_validity_vs_recovery(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 13: Completion Validity vs Recovery

    Scatter plot showing the tradeoff between validity and recovery
    for molecular completion (Benchmark 7).

    Spec reference: Figure 13
    """
    if "b7_metrics" not in data:
        logger.warning("Missing B7 metrics for Figure 13")
        return plt.figure()

    df = data["b7_metrics"].copy()

    figsize = FIGURE_SIZES["fig13_completion_validity_vs_recovery"]
    fig, ax = plt.subplots(figsize=figsize)

    # Plot points colored by representation, shaped by model
    for repr_name in REPRESENTATIONS:
        repr_df = df[df["representation"] == repr_name]

        for model_dict in MODELS:
            model_id = model_dict["id"]
            marker = model_dict["marker"]

            model_df = repr_df[repr_df["model"] == model_id]

            for thinking in [False, True]:
                thinking_df = model_df[model_df["thinking"] == thinking]

                if len(thinking_df) == 0:
                    continue

                # Marker style
                if thinking:
                    facecolor = REPR_COLORS[repr_name]
                    edgecolor = REPR_COLORS[repr_name]
                else:
                    facecolor = "none"
                    edgecolor = REPR_COLORS[repr_name]

                ax.scatter(
                    thinking_df["validity_rate"],
                    thinking_df["recovery_rate"],
                    marker=marker,
                    s=100,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidths=2,
                    alpha=0.7,
                )

    # Add diagonal line (recovery ≤ validity)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="recovery = validity")

    ax.set_xlabel("Validity Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recovery Rate", fontsize=12, fontweight="bold")
    ax.set_title("Completion: Validity vs Recovery Tradeoff", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # Create custom legend
    legend_elements = []
    for repr_name in REPRESENTATIONS:
        legend_elements.append(
            mpatches.Patch(facecolor=REPR_COLORS[repr_name],
                          label=REPR_DISPLAY_NAMES[repr_name])
        )

    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, ncol=2)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 14: Retrieval Distractor Confusion
# ============================================================================


def plot_fig14_retrieval_distractor_confusion(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 14: Retrieval Distractor Confusion

    Stacked bar chart showing which types of distractors fool the model
    when retrieval fails (Benchmark 4).

    Spec reference: Figure 14
    """
    if "b4_scored" not in data:
        logger.warning("Missing B4 scored data for Figure 14")
        return plt.figure()

    df = data["b4_scored"].copy()

    # Filter to incorrect answers only
    df = df[df["correct"] == False].copy()

    if len(df) == 0:
        logger.warning("No incorrect answers in B4 for Figure 14")
        return plt.figure()

    # Check if distractor_type column exists
    # If not, we'd need to infer it from the data structure
    # For now, create a placeholder

    figsize = FIGURE_SIZES["fig14_retrieval_distractor_confusion"]
    fig, ax = plt.subplots(figsize=figsize)

    # This figure requires distractor type information which may not be in the scored data
    # For a complete implementation, we'd need to track which distractor was chosen
    # For now, show a placeholder

    ax.text(
        0.5, 0.5,
        "Figure 14: Retrieval Distractor Confusion\n\n"
        "Requires distractor type tracking in scored data.\n"
        "Implementation depends on data structure.",
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 15: Statistical Significance Matrix
# ============================================================================


def plot_fig15_statistical_significance_matrix(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 15: Statistical Significance Matrix

    Lower-triangular heatmap showing p-values from pairwise Wilcoxon tests
    between representations, for comprehension and generation separately.

    Spec reference: Figure 15

    Note: This requires the statistical tests to be run first (script 05).
    This is a placeholder that will work once those results are available.
    """
    figsize = FIGURE_SIZES["fig15_statistical_significance_matrix"]
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Check if statistical test results exist
    stats_comp_path = RESULTS_DIR / "statistics" / "pairwise_tests_comprehension.csv"
    stats_gen_path = RESULTS_DIR / "statistics" / "pairwise_tests_generation.csv"

    if not stats_comp_path.exists() or not stats_gen_path.exists():
        for ax in axes:
            ax.text(
                0.5, 0.5,
                "Figure 15: Statistical Significance Matrix\n\n"
                "Requires statistical tests to be run first.\n"
                "Run scripts/05_statistical_tests.py",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=11,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        plt.tight_layout()
        return fig

    # Load statistical test results
    comp_df = pd.read_csv(stats_comp_path)
    gen_df = pd.read_csv(stats_gen_path)

    # Create p-value matrices
    for ax_idx, (ax, df, title) in enumerate([
        (axes[0], comp_df, "Comprehension (B1-B5)"),
        (axes[1], gen_df, "Generation (B6)"),
    ]):
        # Create matrix
        n_repr = len(REPRESENTATIONS)
        p_matrix = np.full((n_repr, n_repr), np.nan)

        for _, row in df.iterrows():
            repr1_idx = REPRESENTATIONS.index(row["representation_1"])
            repr2_idx = REPRESENTATIONS.index(row["representation_2"])

            # Lower triangular
            if repr1_idx > repr2_idx:
                p_matrix[repr1_idx, repr2_idx] = row["p_value"]

        # Mask upper triangle
        mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=1)

        # Plot heatmap
        sns.heatmap(
            p_matrix,
            mask=mask,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=0.05,
            cbar_kws={"label": "p-value"},
            xticklabels=[REPR_DISPLAY_NAMES[r] for r in REPRESENTATIONS],
            yticklabels=[REPR_DISPLAY_NAMES[r] for r in REPRESENTATIONS],
            ax=ax,
        )

        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle("Statistical Significance Matrix (Pairwise Wilcoxon Tests)",
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig


# ============================================================================
# Figure 16: Token Length vs Performance
# ============================================================================


def plot_fig16_token_length_vs_performance(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 16: Token Length vs Performance

    Scatter plot with marginal histograms showing relationship between
    representation token length and generation performance.

    Spec reference: Figure 16

    Note: Requires token length to be computed for each representation.
    """
    # Check if token length data exists
    token_length_path = RESULTS_DIR / "metrics" / "token_lengths.csv"

    if not token_length_path.exists() or "b6_metrics" not in data:
        figsize = FIGURE_SIZES["fig16_token_length_vs_performance"]
        fig, ax = plt.subplots(figsize=figsize)

        ax.text(
            0.5, 0.5,
            "Figure 16: Token Length vs Performance\n\n"
            "Requires token length computation.\n"
            "Token lengths need to be measured per representation.",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        return fig

    # Load token lengths and generation metrics
    token_df = pd.read_csv(token_length_path)
    gen_df = data["b6_metrics"].copy()

    # Merge
    merged = gen_df.merge(token_df, on=["representation", "model"])

    # Create jointplot using seaborn
    g = sns.jointplot(
        data=merged,
        x="mean_token_length",
        y="exact_match_rate",
        hue="representation",
        palette=REPR_COLORS,
        height=FIGURE_SIZES["fig16_token_length_vs_performance"][0],
        ratio=5,
        marginal_kws=dict(bins=20, fill=True),
    )

    # Add correlation
    x = merged["mean_token_length"].values
    y = merged["exact_match_rate"].values
    r, p = stats.pearsonr(x, y)

    g.ax_joint.text(
        0.05, 0.95,
        f"Pearson r = {r:.3f}\np = {p:.3e}",
        transform=g.ax_joint.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    g.set_axis_labels(
        "Mean Token Length",
        "Generation Exact Match Rate",
        fontsize=12,
        fontweight="bold",
    )

    g.fig.suptitle("Token Efficiency vs Performance", fontsize=14, fontweight="bold", y=1.02)

    return g.fig


# ============================================================================
# Figure 17: Summary Table
# ============================================================================


def plot_fig_summary_table(data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Figure 17: Summary Table

    Styled table showing mean performance across all benchmarks for each
    representation, with best values highlighted.

    Spec reference: Summary Table Figure
    """
    # Collect summary statistics
    summary_data = []

    for repr_name in REPRESENTATIONS:
        row = {"Representation": REPR_DISPLAY_NAMES[repr_name]}

        # B1: Accuracy
        if "b1_metrics" in data:
            b1_df = data["b1_metrics"]
            repr_df = b1_df[b1_df["representation"] == repr_name]
            if len(repr_df) > 0:
                row["B1 Acc"] = repr_df["accuracy"].mean()

        # B2: Macro F1
        if "b2_macro" in data:
            b2_df = data["b2_macro"]
            repr_df = b2_df[b2_df["representation"] == repr_name]
            if len(repr_df) > 0:
                row["B2 F1"] = repr_df["macro_f1"].mean()

        # B3: Normalized score
        if "b3_metrics" in data:
            b3_df = data["b3_metrics"].copy()
            b3_df["score"] = normalize_benchmark_score(b3_df, 3, "mae")
            repr_df = b3_df[b3_df["representation"] == repr_name]
            if len(repr_df) > 0:
                row["B3 Score"] = repr_df["score"].mean()

        # B4: Accuracy
        if "b4_metrics" in data:
            b4_df = data["b4_metrics"]
            repr_df = b4_df[b4_df["representation"] == repr_name]
            if len(repr_df) > 0:
                row["B4 Acc"] = repr_df["accuracy"].mean()

        # B5: F1
        if "b5_metrics" in data:
            b5_df = data["b5_metrics"]
            repr_df = b5_df[b5_df["representation"] == repr_name]
            if len(repr_df) > 0:
                row["B5 F1"] = repr_df["f1"].mean()

        # B6: Exact match and validity
        if "b6_metrics" in data:
            b6_df = data["b6_metrics"]
            repr_df = b6_df[b6_df["representation"] == repr_name]
            if len(repr_df) > 0:
                row["B6 Exact"] = repr_df["exact_match_rate"].mean()
                row["B6 Valid"] = repr_df["validity_rate"].mean()

        # B7: Recovery
        if "b7_metrics" in data:
            b7_df = data["b7_metrics"]
            repr_df = b7_df[b7_df["representation"] == repr_name]
            if len(repr_df) > 0:
                row["B7 Recovery"] = repr_df["recovery_rate"].mean()

        summary_data.append(row)

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Compute mean rank
    if "representation_mean_ranks" in data:
        ranks_df = data["representation_mean_ranks"]
        for repr_name in REPRESENTATIONS:
            repr_df = ranks_df[ranks_df["representation"] == repr_name]
            if len(repr_df) > 0:
                idx = summary_df[summary_df["Representation"] == REPR_DISPLAY_NAMES[repr_name]].index[0]
                summary_df.loc[idx, "Mean Rank"] = repr_df["mean_rank"].iloc[0]

    # Create figure with table
    figsize = FIGURE_SIZES["fig_summary_table"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Create table
    table_data = summary_df.values
    col_labels = summary_df.columns

    table = ax.table(
        cellText=[[f"{val:.3f}" if isinstance(val, (int, float)) and not pd.isna(val) else str(val)
                  for val in row] for row in table_data],
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight best values in each column
    for col_idx in range(1, len(col_labels)):
        col_name = col_labels[col_idx]
        if col_name == "Representation":
            continue

        try:
            values = [float(table[(row_idx + 1, col_idx)].get_text().get_text())
                     for row_idx in range(len(table_data))]

            # For rank, lower is better; for others, higher is better
            if col_name == "Mean Rank":
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)

            table[(best_idx + 1, col_idx)].set_facecolor("#90EE90")
            table[(best_idx + 1, col_idx)].set_text_props(weight="bold")
        except:
            pass

    ax.set_title("Summary Table: Mean Performance Across All Conditions",
                fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


# ============================================================================
# Main Function
# ============================================================================


def main():
    """Generate all figures or a specific figure."""
    parser = argparse.ArgumentParser(description="Generate figures for molecular representation benchmark")
    parser.add_argument(
        "--figure",
        type=str,
        default="all",
        help="Figure to generate (e.g., 'fig1_main_heatmap' or 'all')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all figures",
    )

    args = parser.parse_args()

    # Load all data
    logger.info("Loading data...")
    data = load_data()

    if len(data) == 0:
        logger.error("No data loaded! Run evaluation script first.")
        return

    # Define all figure functions
    figure_functions = {
        "fig1_main_heatmap": plot_fig1_main_heatmap,
        "fig2_generation_barplot": plot_fig2_generation_barplot,
        "fig3_comprehension_vs_generation_scatter": plot_fig3_comprehension_vs_generation_scatter,
        "fig4_thinking_ablation_delta": plot_fig4_thinking_ablation_delta,
        "fig5_representation_radar": plot_fig5_representation_radar,
        "fig6_atom_counting_by_size": plot_fig6_atom_counting_by_size,
        "fig7_functional_group_breakdown": plot_fig7_functional_group_breakdown,
        "fig8_property_estimation_scatter": plot_fig8_property_estimation_scatter,
        "fig9_validity_by_complexity": plot_fig9_validity_by_complexity,
        "fig10_generation_tanimoto_violin": plot_fig10_generation_tanimoto_violin,
        "fig11_thinking_vs_benchmark_interaction": plot_fig11_thinking_vs_benchmark_interaction,
        "fig12_isomer_discrimination_breakdown": plot_fig12_isomer_discrimination_breakdown,
        "fig13_completion_validity_vs_recovery": plot_fig13_completion_validity_vs_recovery,
        "fig14_retrieval_distractor_confusion": plot_fig14_retrieval_distractor_confusion,
        "fig15_statistical_significance_matrix": plot_fig15_statistical_significance_matrix,
        "fig16_token_length_vs_performance": plot_fig16_token_length_vs_performance,
        "fig_summary_table": plot_fig_summary_table,
    }

    # Determine which figures to generate
    if args.all or args.figure == "all":
        figures_to_generate = list(figure_functions.keys())
    else:
        if args.figure not in figure_functions:
            logger.error(f"Unknown figure: {args.figure}")
            logger.info(f"Available figures: {', '.join(figure_functions.keys())}")
            return
        figures_to_generate = [args.figure]

    # Generate figures
    logger.info(f"Generating {len(figures_to_generate)} figure(s)...")

    for fig_name in figures_to_generate:
        try:
            logger.info(f"\nGenerating {fig_name}...")
            fig = figure_functions[fig_name](data)
            save_figure(fig, fig_name)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error generating {fig_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    logger.info(f"\n✅ Done! Generated {len(figures_to_generate)} figure(s)")
    logger.info(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
