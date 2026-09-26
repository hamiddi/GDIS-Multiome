#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
42_supplementary_figure_s2_discovery_alignment_null.py
======================================================

Publication-only generator for Supplementary Figure S2.

Purpose
-------
Reproduce the frozen Phase F5 discovery cross-modal alignment null analysis:

A. Null distribution of absolute Q50 center separation.
B. Null distribution of RNA/ATAC sustained-GDIS Spearman correlation.
C. Null distribution of Jensen-Shannon distance.
D. Compact summary of observed values and empirical P values.

The null model is the exact one used in canonical Phase F5:
ATAC contiguous 100-cell block permutation along the RNA-derived common
pseudotime ordering. RNA and pseudotime are held fixed.

Scientific guardrails
---------------------
- This is a cross-modal alignment test, not a chromatin-priming test.
- Phase F4 peak-based timing remains inconclusive.
- Phase F4c region timing does not support a robust ATAC-leading shift.
- No new statistic, null model, threshold, or parameter is introduced.

Run from the canonical GDIS-Multiome project root:
    python 42_supplementary_figure_s2_discovery_alignment_null.py

Outputs
-------
figures/supplementary/Supplementary_Figure_S2_discovery_alignment_null.png
figures/supplementary/Supplementary_Figure_S2_discovery_alignment_null.pdf
figures/supplementary/Supplementary_Figure_S2_null_distribution.tsv.gz
figures/supplementary/Supplementary_Figure_S2_null_comparison.tsv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mudata as mu


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

if (HERE / "data").exists():
    PROJECT_ROOT = HERE
elif (HERE.parent / "data").exists():
    PROJECT_ROOT = HERE.parent
else:
    PROJECT_ROOT = HERE

DATA_FILE = PROJECT_ROOT / "data" / "GSE275562_mudata_with_annotation_all.h5mu"
OUTDIR = PROJECT_ROOT / "figures" / "supplementary"
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUTDIR / "Supplementary_Figure_S2_discovery_alignment_null.png"
PDF_OUT = OUTDIR / "Supplementary_Figure_S2_discovery_alignment_null.pdf"
NULL_TABLE = OUTDIR / "Supplementary_Figure_S2_null_distribution.tsv.gz"
COMPARISON_TABLE = OUTDIR / "Supplementary_Figure_S2_null_comparison.tsv"

DPI = 600
FIGSIZE = (15.5, 11.0)

PANEL_LABEL_FONTSIZE = 22
PANEL_TITLE_FONTSIZE = 17
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 12
ANNOTATION_FONTSIZE = 12


def find_script(filename: str) -> Path:
    candidates = [
        HERE / filename,
        PROJECT_ROOT / filename,
        PROJECT_ROOT / "scripts" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not locate canonical script {filename}. "
        f"Searched: {', '.join(str(p) for p in candidates)}"
    )


F5_SCRIPT = find_script("12_phase_f5_cross_modal_alignment_null.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inputs(f5):
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {DATA_FILE}\n"
            "Place the canonical GSE275562 H5MU file under ./data."
        )

    try:
        mdata = mu.read_h5mu(DATA_FILE, backed="r")
    except TypeError:
        mdata = mu.read_h5mu(DATA_FILE)

    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]
    obs_full = rna.obs.copy()

    beta_mask = (
        obs_full["cell_type"].astype(str).isin(f5.BETA_LINEAGE)
    ).to_numpy()

    obs = obs_full.loc[beta_mask].copy().reset_index(drop=True)
    obs.index = pd.Index([f"beta_{i}" for i in range(len(obs))])

    X_rna = f5.as_numpy(
        rna.obsm[f5.RNA_REPRESENTATION]
    )[beta_mask, : f5.N_LATENT_DIMS]

    X_atac = f5.as_numpy(
        atac.obsm[f5.ATAC_REPRESENTATION]
    )[beta_mask, : f5.N_LATENT_DIMS]

    X_rna_z = f5.zscore_columns(X_rna)
    X_atac_z = f5.zscore_columns(X_atac)

    pseudotime = f5.reconstruct_common_pseudotime(X_rna, obs)
    medians = f5.get_state_medians(pseudotime, obs)

    frozen_rna_center, frozen_atac_center, critical_value = (
        f5.derive_frozen_centers(
            X_rna_z,
            X_atac_z,
            pseudotime,
            medians,
        )
    )

    region_lower = max(
        0.0,
        frozen_atac_center - f5.FAMILY_MATCH_TOLERANCE,
    )
    region_upper = min(
        1.0,
        frozen_rna_center + f5.FAMILY_MATCH_TOLERANCE,
    )

    return (
        obs,
        X_rna_z,
        X_atac_z,
        pseudotime,
        critical_value,
        region_lower,
        region_upper,
    )


def reproduce_f5_null(
    f5,
    X_rna_z,
    X_atac_z,
    pseudotime,
    critical_value,
    region_lower,
    region_upper,
):
    valid = np.isfinite(pseudotime)
    order = np.argsort(pseudotime[valid], kind="mergesort")

    pt_ordered = pseudotime[valid][order]
    X_rna_ordered = X_rna_z[valid][order]
    X_atac_ordered = X_atac_z[valid][order]

    rna_traj, parameters = f5.create_windows_from_ordered(
        X_rna_ordered,
        pt_ordered,
        f5.PRIMARY_WINDOW_SIZE,
        f5.PRIMARY_STEP_SIZE,
    )
    atac_traj, atac_parameters = f5.create_windows_from_ordered(
        X_atac_ordered,
        pt_ordered,
        f5.PRIMARY_WINDOW_SIZE,
        f5.PRIMARY_STEP_SIZE,
    )

    rna_x, rna_y = f5.sustained_gdis_curve(
        rna_traj, parameters, critical_value
    )
    atac_x, atac_y = f5.sustained_gdis_curve(
        atac_traj, atac_parameters, critical_value
    )

    observed = f5.alignment_metrics(
        rna_x,
        rna_y,
        atac_x,
        atac_y,
        region_lower,
        region_upper,
    )
    if observed is None:
        raise RuntimeError("Observed F5 alignment metrics could not be reproduced.")

    rng = np.random.default_rng(f5.RANDOM_SEED)
    records = []

    for i in range(1, f5.N_PERMUTATIONS + 1):
        X_atac_null = f5.permute_atac_blocks(X_atac_ordered, rng)

        atac_null_traj, atac_null_parameters = (
            f5.create_windows_from_ordered(
                X_atac_null,
                pt_ordered,
                f5.PRIMARY_WINDOW_SIZE,
                f5.PRIMARY_STEP_SIZE,
            )
        )

        atac_null_x, atac_null_y = f5.sustained_gdis_curve(
            atac_null_traj,
            atac_null_parameters,
            critical_value,
        )

        metrics = f5.alignment_metrics(
            rna_x,
            rna_y,
            atac_null_x,
            atac_null_y,
            region_lower,
            region_upper,
        )

        if metrics is not None:
            metrics["permutation_id"] = i
            records.append(metrics)

        if i % 25 == 0 or i == f5.N_PERMUTATIONS:
            print(f"F5 block permutations: {i}/{f5.N_PERMUTATIONS}")

    null = pd.DataFrame(records)

    if len(null) < int(0.95 * f5.N_PERMUTATIONS):
        raise RuntimeError("Too many F5 null permutations failed.")

    specs = [
        ("absolute_center_separation", "smaller"),
        ("curve_spearman_rho", "larger"),
        ("jensen_shannon_distance", "smaller"),
    ]

    rows = []
    for metric, direction in specs:
        obs_value = float(observed[metric])
        null_values = null[metric].to_numpy(dtype=float)
        summary = f5.null_summary(null_values)

        if direction == "smaller":
            p_value = f5.empirical_p_smaller(obs_value, null_values)
        else:
            p_value = f5.empirical_p_larger(obs_value, null_values)

        rows.append({
            "metric": metric,
            "observed": obs_value,
            "better_direction": direction,
            "null_mean": summary["mean"],
            "null_median": summary["median"],
            "null_q2.5": summary["q2.5"],
            "null_q25": summary["q25"],
            "null_q75": summary["q75"],
            "null_q97.5": summary["q97.5"],
            "empirical_p": p_value,
        })

    comparison = pd.DataFrame(rows).set_index("metric")
    return observed, null, comparison


def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )


def plot_null_panel(
    ax,
    values,
    observed,
    title,
    xlabel,
    p_value,
):
    ax.hist(values, bins=30, alpha=0.80, edgecolor="white")
    ax.axvline(
        observed,
        linewidth=2.2,
        label=f"Observed = {observed:.4f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Null permutations")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.97,
        0.84,
        f"Empirical p = {p_value:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
    )


def make_figure(null, comparison, region_lower, region_upper):
    plt.rcParams.update({
        "axes.titlesize": PANEL_TITLE_FONTSIZE,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "xtick.labelsize": TICK_LABEL_FONTSIZE,
        "ytick.labelsize": TICK_LABEL_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
    })

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.04,
        h_pad=0.05,
        hspace=0.04,
        wspace=0.04,
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A
    row = comparison.loc["absolute_center_separation"]
    plot_null_panel(
        ax_a,
        null["absolute_center_separation"].to_numpy(dtype=float),
        float(row["observed"]),
        "Absolute event-region center separation",
        r"$|Q50_{\mathrm{RNA}} - Q50_{\mathrm{ATAC}}|$",
        float(row["empirical_p"]),
    )
    add_panel_label(ax_a, "A")

    # B
    row = comparison.loc["curve_spearman_rho"]
    plot_null_panel(
        ax_b,
        null["curve_spearman_rho"].to_numpy(dtype=float),
        float(row["observed"]),
        "Cross-modal profile correlation",
        "Spearman correlation",
        float(row["empirical_p"]),
    )
    add_panel_label(ax_b, "B")

    # C
    row = comparison.loc["jensen_shannon_distance"]
    plot_null_panel(
        ax_c,
        null["jensen_shannon_distance"].to_numpy(dtype=float),
        float(row["observed"]),
        "Cross-modal distributional distance",
        "Jensen–Shannon distance",
        float(row["empirical_p"]),
    )
    add_panel_label(ax_c, "C")

    # D: compact inferential summary
    ax_d.axis("off")
    add_panel_label(ax_d, "D")
    ax_d.set_title("Primary null-test summary", fontweight="bold", pad=14)

    labels = {
        "absolute_center_separation": "Absolute Q50 separation",
        "curve_spearman_rho": "Spearman correlation",
        "jensen_shannon_distance": "Jensen–Shannon distance",
    }

    # Three metric summaries are vertically separated from the conclusion box.
    y_positions = [0.78, 0.56, 0.34]
    for metric, y in zip(
        [
            "absolute_center_separation",
            "curve_spearman_rho",
            "jensen_shannon_distance",
        ],
        y_positions,
    ):
        row = comparison.loc[metric]
        ax_d.text(
            0.05,
            y,
            labels[metric],
            transform=ax_d.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="top",
        )
        ax_d.text(
            0.05,
            y - 0.075,
            (
                f"Observed = {float(row['observed']):.4f}    "
                f"Null 95% = [{float(row['null_q2.5']):.4f}, "
                f"{float(row['null_q97.5']):.4f}]    "
                f"p = {float(row['empirical_p']):.4f}"
            ),
            transform=ax_d.transAxes,
            fontsize=ANNOTATION_FONTSIZE,
            ha="left",
            va="top",
        )

    n_sig = int((comparison["empirical_p"].astype(float) < 0.05).sum())
    ax_d.text(
        0.05,
        0.095,
        (
            f"Primary conclusion: {n_sig}/3 alignment metrics significant at α = 0.05.\n"
            "No evidence of broad RNA–ATAC alignment beyond the frozen null."
        ),
        transform=ax_d.transAxes,
        fontsize=12,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "white",
            "edgecolor": "0.7",
        },
    )

    fig.savefig(PNG_OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    print("Loading frozen Phase F5 discovery module...")
    f5 = load_module("gdis_f5", F5_SCRIPT)

    print("Loading GSE275562 and reproducing the frozen event region...")
    (
        obs,
        X_rna_z,
        X_atac_z,
        pseudotime,
        critical_value,
        region_lower,
        region_upper,
    ) = load_inputs(f5)

    print(f"Cells: {len(obs):,}")
    print(
        f"Frozen shared event region: "
        f"[{region_lower:.6f}, {region_upper:.6f}]"
    )

    if NULL_TABLE.exists() and COMPARISON_TABLE.exists():
        print("\nExisting F5 null tables detected; reusing them for figure generation.")
        null = pd.read_csv(NULL_TABLE, sep="\t")
        comparison = pd.read_csv(COMPARISON_TABLE, sep="\t").set_index("metric")
        observed = {
            metric: float(row["observed"])
            for metric, row in comparison.iterrows()
        }
    else:
        print("\nReproducing the frozen F5 block-permutation null...")
        observed, null, comparison = reproduce_f5_null(
            f5,
            X_rna_z,
            X_atac_z,
            pseudotime,
            critical_value,
            region_lower,
            region_upper,
        )

        null.to_csv(NULL_TABLE, sep="\t", index=False, compression="gzip")
        comparison.reset_index().to_csv(
            COMPARISON_TABLE,
            sep="\t",
            index=False,
        )

    print("\nGenerating Supplementary Figure S2...")
    make_figure(
        null,
        comparison,
        region_lower,
        region_upper,
    )

    print("\nSupplementary Figure S2 generated successfully.")
    print(f"PNG (600 dpi): {PNG_OUT}")
    print(f"PDF (vector):  {PDF_OUT}")
    print(f"Null table:    {NULL_TABLE}")
    print(f"Comparison:    {COMPARISON_TABLE}")

    print("\nObserved F5 metrics:")
    for metric, row in comparison.iterrows():
        print(
            f"  {metric}: observed={float(row['observed']):.6f}, "
            f"p={float(row['empirical_p']):.6f}"
        )


if __name__ == "__main__":
    main()

