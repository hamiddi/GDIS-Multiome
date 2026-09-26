#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
45_main_figure_5_exact_circular_shift_null.py
=============================================

Rebuild Main Figure 5 directly from the frozen Phase F18 exact
dependence-preserving circular-shift null values.

Why this script exists
----------------------
The primary F18 alignment test uses EXACT admissible circular shifts of the
complete ATAC transition-energy profile. Therefore, the publication figure
should display those exact null values directly rather than a synthetic-looking
large histogram.

Panels
------
A. Exact circular-shift null values for absolute Q50 separation.
B. Exact circular-shift null values for Spearman correlation.
C. Exact circular-shift null values for Jensen-Shannon divergence.
D. Compact summary of primary circular-shift and secondary block-permutation
   p-values.

Each point in panels A-C is ONE admissible exact circular shift.
The observed statistic is shown as a vertical reference line.

Inputs
------
data/GSE140203/representations_f18/
    f18_circular_shift_null.tsv.gz
    f18_alignment_summary.tsv

The script also validates that:
    1. the number of rows in the exact-null table matches the frozen summary;
    2. all three empirical circular-shift p-values recomputed from the exact
       values match the frozen F18 summary;
    3. the primary evidence classification implied by the exact p-values
       matches the frozen F18 result.

Outputs
-------
data/GSE140203/figures_f25/
    Main_Figure_5_exact_circular_shift_alignment_null.png
    Main_Figure_5_exact_circular_shift_alignment_null.pdf
    Main_Figure_5_exact_circular_shift_validation.tsv

Example
-------
python 45_main_figure_5_exact_circular_shift_null.py

Optional custom paths:
python 45_main_figure_5_exact_circular_shift_null.py \
    --f18-dir data/GSE140203/representations_f18 \
    --out-dir data/GSE140203/figures_f25
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# FROZEN PLOTTING SETTINGS
# ============================================================================

DPI = 600
FIGSIZE = (13.0, 8.3)
ALPHA = 0.05

TITLE_SIZE = 12.0
LABEL_SIZE = 10.5
TICK_SIZE = 9.0
ANNOTATION_SIZE = 9.0
PANEL_LABEL_SIZE = 13.0


# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild Main Figure 5 from the exact F18 circular-shift null."
        )
    )
    parser.add_argument(
        "--f18-dir",
        default="data/GSE140203/representations_f18",
        help="Directory containing frozen F18 outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/GSE140203/figures_f25",
        help="Directory for publication figure outputs.",
    )
    return parser.parse_args()


# ============================================================================
# HELPERS
# ============================================================================

def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label} is missing required columns:\n  " + "\n  ".join(missing)
        )


def empirical_p_lower(observed: float, null_values: np.ndarray) -> float:
    """
    F18 lower-tail empirical p-value:
        p = (1 + number(null <= observed)) / (1 + N)
    """
    null_values = np.asarray(null_values, dtype=float)
    return float(
        (1 + np.sum(null_values <= observed))
        / (1 + len(null_values))
    )


def empirical_p_upper(observed: float, null_values: np.ndarray) -> float:
    """
    F18 upper-tail empirical p-value:
        p = (1 + number(null >= observed)) / (1 + N)
    """
    null_values = np.asarray(null_values, dtype=float)
    return float(
        (1 + np.sum(null_values >= observed))
        / (1 + len(null_values))
    )


def evidence_label(n_significant: int) -> str:
    if n_significant >= 3:
        return "STRONG"
    if n_significant == 2:
        return "PARTIAL"
    return "LIMITED_OR_NONE"


def format_p(value: float) -> str:
    return f"{float(value):.4f}"


def apply_axes(ax):
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(axis="x", alpha=0.22, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_exact_shift_panel(
    ax,
    circular: pd.DataFrame,
    metric_column: str,
    observed_value: float,
    empirical_p: float,
    title: str,
    xlabel: str,
):
    """
    Plot EVERY admissible exact circular shift explicitly.

    x = null statistic
    y = circular-shift offset in windows

    This avoids implying that the primary null was generated by hundreds or
    thousands of random permutations.
    """
    x = circular[metric_column].to_numpy(dtype=float)
    y = circular["offset_windows"].to_numpy(dtype=float)

    ax.scatter(
        x,
        y,
        s=28,
        alpha=0.82,
        edgecolors="black",
        linewidths=0.35,
        label="Exact admissible shifts",
    )

    ax.axvline(
        observed_value,
        linestyle="--",
        linewidth=1.8,
        label="Observed",
    )

    ax.set_title(title, fontsize=TITLE_SIZE, pad=8)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel("ATAC circular shift (windows)", fontsize=LABEL_SIZE)

    annotation = (
        f"Observed = {observed_value:.6f}\n"
        f"Exact shifts = {len(circular)}\n"
        f"Empirical p = {empirical_p:.4f}"
    )
    ax.text(
        0.98,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_SIZE,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
        ),
    )

    ax.legend(
        loc="lower right",
        fontsize=8.1,
        frameon=True,
    )
    apply_axes(ax)


def build_summary_table(ax, summary: pd.Series):
    """
    Panel D: compact comparison of the frozen primary and sensitivity tests.
    """
    ax.axis("off")

    metrics = [
        (
            "Abs. Q50 separation",
            float(summary["observed_absolute_q50_separation"]),
            float(summary["circular_p_abs_q50_separation"]),
            float(summary["block_p_abs_q50_separation"]),
        ),
        (
            "Spearman rho",
            float(summary["observed_spearman_rho"]),
            float(summary["circular_p_spearman_rho"]),
            float(summary["block_p_spearman_rho"]),
        ),
        (
            "JS divergence",
            float(summary["observed_JS_divergence_base2"]),
            float(summary["circular_p_JS_divergence"]),
            float(summary["block_p_JS_divergence"]),
        ),
    ]

    cell_text = []
    for metric, observed, p_circular, p_block in metrics:
        cell_text.append(
            [
                metric,
                f"{observed:.6f}",
                f"{p_circular:.4f}",
                "Yes" if p_circular <= ALPHA else "No",
                f"{p_block:.4f}",
                "Yes" if p_block <= ALPHA else "No",
            ]
        )

    columns = [
        "Metric",
        "Observed",
        "Circular p",
        "Sig.",
        "Block p",
        "Sig.",
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=[0.00, 0.49, 1.00, 0.43],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(fontweight="bold")

    circular_class = str(summary["circular_alignment_evidence_class"])
    block_class = str(summary["block_alignment_evidence_class"])
    n_shifts = int(summary["circular_n_admissible_shifts"])
    n_block = int(summary["block_n_permutations"])

    conclusion = (
        f"Primary exact circular-shift null: {n_shifts} admissible shifts\n"
        f"Primary evidence class: {circular_class}\n\n"
        f"Secondary block-permutation sensitivity: {n_block:,} permutations\n"
        f"Sensitivity evidence class: {block_class}\n\n"
        "Interpretation: localized ATAC-earlier timing does not imply\n"
        "broad RNA–ATAC profile alignment."
    )

    ax.text(
        0.02,
        0.40,
        conclusion,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        linespacing=1.25,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="black",
            linewidth=0.7,
        ),
    )

    ax.set_title(
        "Primary vs sensitivity null summary",
        fontsize=TITLE_SIZE,
        pad=8,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    f18_dir = Path(args.f18_dir)
    out_dir = Path(args.out_dir)

    circular_path = f18_dir / "f18_circular_shift_null.tsv.gz"
    summary_path = f18_dir / "f18_alignment_summary.tsv"

    output_png = out_dir / "Main_Figure_5_exact_circular_shift_alignment_null.png"
    output_pdf = out_dir / "Main_Figure_5_exact_circular_shift_alignment_null.pdf"
    output_validation = out_dir / "Main_Figure_5_exact_circular_shift_validation.tsv"

    for path in [circular_path, summary_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required frozen F18 file not found:\n{path}\n\n"
                "Run Phase F18 first or supply --f18-dir pointing to the "
                "directory containing the frozen F18 outputs."
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    circular = pd.read_csv(
        circular_path,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )
    summary_df = pd.read_csv(
        summary_path,
        sep="\t",
        low_memory=False,
    )

    if len(summary_df) != 1:
        raise RuntimeError(
            "f18_alignment_summary.tsv must contain exactly one frozen summary row."
        )
    summary = summary_df.iloc[0]

    require_columns(
        circular,
        [
            "offset_windows",
            "circular_distance_from_zero",
            "absolute_q50_separation",
            "spearman_rho",
            "JS_divergence_base2",
        ],
        "f18_circular_shift_null.tsv.gz",
    )

    require_columns(
        summary_df,
        [
            "observed_absolute_q50_separation",
            "observed_spearman_rho",
            "observed_JS_divergence_base2",
            "circular_n_admissible_shifts",
            "circular_p_abs_q50_separation",
            "circular_p_spearman_rho",
            "circular_p_JS_divergence",
            "circular_alignment_evidence_class",
            "block_n_permutations",
            "block_p_abs_q50_separation",
            "block_p_spearman_rho",
            "block_p_JS_divergence",
            "block_alignment_evidence_class",
        ],
        "f18_alignment_summary.tsv",
    )

    # ------------------------------------------------------------------------
    # Recompute the exact empirical p-values from the actual circular-shift
    # values. This is an important publication integrity check.
    # ------------------------------------------------------------------------

    observed_abs = float(summary["observed_absolute_q50_separation"])
    observed_rho = float(summary["observed_spearman_rho"])
    observed_js = float(summary["observed_JS_divergence_base2"])

    recomputed_abs_p = empirical_p_lower(
        observed_abs,
        circular["absolute_q50_separation"].to_numpy(dtype=float),
    )
    recomputed_rho_p = empirical_p_upper(
        observed_rho,
        circular["spearman_rho"].to_numpy(dtype=float),
    )
    recomputed_js_p = empirical_p_lower(
        observed_js,
        circular["JS_divergence_base2"].to_numpy(dtype=float),
    )

    frozen_abs_p = float(summary["circular_p_abs_q50_separation"])
    frozen_rho_p = float(summary["circular_p_spearman_rho"])
    frozen_js_p = float(summary["circular_p_JS_divergence"])

    expected_n = int(summary["circular_n_admissible_shifts"])
    actual_n = len(circular)

    p_tolerance = 1e-12

    checks = [
        {
            "check": "Exact-null row count matches frozen summary",
            "pass": actual_n == expected_n,
            "observed": actual_n,
            "expected": expected_n,
        },
        {
            "check": "Absolute-Q50 circular p-value matches frozen F18",
            "pass": abs(recomputed_abs_p - frozen_abs_p) <= p_tolerance,
            "observed": recomputed_abs_p,
            "expected": frozen_abs_p,
        },
        {
            "check": "Spearman circular p-value matches frozen F18",
            "pass": abs(recomputed_rho_p - frozen_rho_p) <= p_tolerance,
            "observed": recomputed_rho_p,
            "expected": frozen_rho_p,
        },
        {
            "check": "JS circular p-value matches frozen F18",
            "pass": abs(recomputed_js_p - frozen_js_p) <= p_tolerance,
            "observed": recomputed_js_p,
            "expected": frozen_js_p,
        },
    ]

    n_sig = int(
        (recomputed_abs_p <= ALPHA)
        + (recomputed_rho_p <= ALPHA)
        + (recomputed_js_p <= ALPHA)
    )
    recomputed_class = evidence_label(n_sig)
    frozen_class = str(summary["circular_alignment_evidence_class"])

    checks.append(
        {
            "check": "Primary evidence class matches frozen F18",
            "pass": recomputed_class == frozen_class,
            "observed": recomputed_class,
            "expected": frozen_class,
        }
    )

    validation = pd.DataFrame(checks)
    validation.to_csv(output_validation, sep="\t", index=False)

    if not bool(validation["pass"].all()):
        print(validation.to_string(index=False))
        raise RuntimeError(
            "One or more F18 integrity checks failed. "
            "The figure was not generated."
        )

    # ------------------------------------------------------------------------
    # Publication figure.
    # ------------------------------------------------------------------------

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        bottom=0.085,
        top=0.965,
        wspace=0.27,
        hspace=0.34,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_exact_shift_panel(
        ax_a,
        circular,
        "absolute_q50_separation",
        observed_abs,
        frozen_abs_p,
        "Exact circular-shift null: absolute Q50 separation",
        "Absolute Q50 separation",
    )
    panel_label(ax_a, "A")

    plot_exact_shift_panel(
        ax_b,
        circular,
        "spearman_rho",
        observed_rho,
        frozen_rho_p,
        "Exact circular-shift null: Spearman correlation",
        "Spearman correlation",
    )
    panel_label(ax_b, "B")

    plot_exact_shift_panel(
        ax_c,
        circular,
        "JS_divergence_base2",
        observed_js,
        frozen_js_p,
        "Exact circular-shift null: Jensen-Shannon divergence",
        "Jensen-Shannon divergence",
    )
    panel_label(ax_c, "C")

    build_summary_table(ax_d, summary)
    panel_label(ax_d, "D")

    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        output_pdf,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)

    print("=" * 92)
    print("MAIN FIGURE 5 — EXACT F18 CIRCULAR-SHIFT NULL")
    print("=" * 92)
    print(f"Exact admissible shifts: {actual_n}")
    print()
    print("Primary circular-shift results:")
    print(
        f"  Absolute Q50 separation: observed={observed_abs:.6f}, "
        f"p={frozen_abs_p:.6f}"
    )
    print(
        f"  Spearman rho:           observed={observed_rho:.6f}, "
        f"p={frozen_rho_p:.6f}"
    )
    print(
        f"  JS divergence:          observed={observed_js:.6f}, "
        f"p={frozen_js_p:.6f}"
    )
    print(f"  Evidence class:         {frozen_class}")
    print()
    print("Saved:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")
    print(f"  {output_validation}")
    print()
    print("All F18 integrity checks passed.")


if __name__ == "__main__":
    main()

