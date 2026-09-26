#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
41_supplementary_figure_s1_discovery_diagnostics.py
===================================================

Publication-only generator for Supplementary Figure S1 (corrected v2).

Purpose
-------
Reproduce the frozen GSE275562 discovery diagnostics without changing any
analytical decision:

A. Diagnostic frozen-family detection rates from Phase F4b.
B. Bootstrap distributions of tracked RNA/ATAC peak locations.
C. Bootstrap distributions of tracked RNA/ATAC relative prominence.
D. Exploratory event-region timing from Phase F4c, summarized as bootstrap
   medians and 95% percentile intervals for Q25, Q50, and Q75 lead descriptors.

Scientific guardrails
---------------------
- Phase F4 remains the primary peak-based inferential result.
- Phase F4b is diagnostic only.
- Phase F4c is exploratory only.
- No thresholds, frozen event families, or GDIS settings are changed.
- No new inferential test is introduced.
- Completed F4b/F4c bootstrap tables are reused automatically when present.

Run from the canonical GDIS-Multiome project root:
    python 41_supplementary_figure_s1_discovery_diagnostics.py

The script can also be placed in ./scripts/; project paths are resolved
automatically.

Outputs
-------
figures/supplementary/Supplementary_Figure_S1_discovery_diagnostics.png
figures/supplementary/Supplementary_Figure_S1_discovery_diagnostics.pdf
figures/supplementary/Supplementary_Figure_S1_F4b_bootstrap.tsv.gz
figures/supplementary/Supplementary_Figure_S1_F4c_region_bootstrap.tsv.gz
figures/supplementary/Supplementary_Figure_S1_summary.tsv
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

PNG_OUT = OUTDIR / "Supplementary_Figure_S1_discovery_diagnostics.png"
PDF_OUT = OUTDIR / "Supplementary_Figure_S1_discovery_diagnostics.pdf"
F4B_TABLE = OUTDIR / "Supplementary_Figure_S1_F4b_bootstrap.tsv.gz"
F4C_TABLE = OUTDIR / "Supplementary_Figure_S1_F4c_region_bootstrap.tsv.gz"
SUMMARY_TABLE = OUTDIR / "Supplementary_Figure_S1_summary.tsv"

DPI = 600
FIGSIZE = (15.5, 11.0)

PANEL_LABEL_FONTSIZE = 22
PANEL_TITLE_FONTSIZE = 17
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 12
ANNOTATION_FONTSIZE = 11


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


F4B_SCRIPT = find_script("10_phase_f4b_peak_stability_diagnostics.py")
F4C_SCRIPT = find_script("11_phase_f4c_event_region_timing_diagnostics.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_discovery_data(f4b):
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
        obs_full["cell_type"].astype(str).isin(f4b.BETA_LINEAGE)
    ).to_numpy()

    obs = obs_full.loc[beta_mask].copy().reset_index(drop=True)
    obs.index = pd.Index([f"beta_{i}" for i in range(len(obs))])

    X_rna = f4b.as_numpy(
        rna.obsm[f4b.RNA_REPRESENTATION]
    )[beta_mask, : f4b.N_LATENT_DIMS]

    X_atac = f4b.as_numpy(
        atac.obsm[f4b.ATAC_REPRESENTATION]
    )[beta_mask, : f4b.N_LATENT_DIMS]

    if X_rna.shape != X_atac.shape:
        raise RuntimeError("RNA and ATAC latent matrices are not paired.")

    X_rna_z = f4b.zscore_columns(X_rna)
    X_atac_z = f4b.zscore_columns(X_atac)

    pseudotime = f4b.reconstruct_common_pseudotime(X_rna, obs)
    medians = f4b.get_state_medians(pseudotime, obs)

    frozen_rna_center, frozen_atac_center, critical_value = (
        f4b.derive_frozen_centers(
            X_rna_z,
            X_atac_z,
            pseudotime,
            medians,
        )
    )

    return (
        obs,
        X_rna_z,
        X_atac_z,
        pseudotime,
        frozen_rna_center,
        frozen_atac_center,
        critical_value,
    )


def run_f4b_diagnostic(
    f4b,
    X_rna_z,
    X_atac_z,
    pseudotime,
    frozen_rna_center,
    frozen_atac_center,
    critical_value,
):
    finite_indices = np.flatnonzero(np.isfinite(pseudotime))
    ordered_indices = finite_indices[
        np.argsort(pseudotime[finite_indices], kind="mergesort")
    ]

    rng = np.random.default_rng(f4b.RANDOM_SEED)
    records = []

    for b in range(1, f4b.N_BOOTSTRAPS + 1):
        row = f4b.diagnostic_bootstrap_replicate(
            X_rna_z=X_rna_z,
            X_atac_z=X_atac_z,
            pseudotime=pseudotime,
            ordered_indices=ordered_indices,
            frozen_rna_center=frozen_rna_center,
            frozen_atac_center=frozen_atac_center,
            critical_value=critical_value,
            rng=rng,
        )
        row["bootstrap_id"] = b
        records.append(row)

        if b % 20 == 0 or b == f4b.N_BOOTSTRAPS:
            print(f"F4b diagnostic bootstrap: {b}/{f4b.N_BOOTSTRAPS}")

    return pd.DataFrame(records)


def run_f4c_region_bootstrap(
    f4c,
    X_rna_z,
    X_atac_z,
    pseudotime,
    frozen_rna_center,
    frozen_atac_center,
    critical_value,
):
    region_lower = max(
        0.0,
        frozen_atac_center - f4c.FAMILY_MATCH_TOLERANCE,
    )
    region_upper = min(
        1.0,
        frozen_rna_center + f4c.FAMILY_MATCH_TOLERANCE,
    )

    finite_indices = np.flatnonzero(np.isfinite(pseudotime))
    ordered_indices = finite_indices[
        np.argsort(pseudotime[finite_indices], kind="mergesort")
    ]

    rng = np.random.default_rng(f4c.RANDOM_SEED)
    records = []

    for b in range(1, f4c.N_BOOTSTRAPS + 1):
        row = f4c.bootstrap_region_replicate(
            X_rna_z=X_rna_z,
            X_atac_z=X_atac_z,
            pseudotime=pseudotime,
            ordered_indices=ordered_indices,
            critical_value=critical_value,
            region_lower=region_lower,
            region_upper=region_upper,
            rng=rng,
        )
        row["bootstrap_id"] = b
        records.append(row)

        if b % 50 == 0 or b == f4c.N_BOOTSTRAPS:
            print(f"F4c region bootstrap: {b}/{f4c.N_BOOTSTRAPS}")

    result = pd.DataFrame(records)
    valid = result.loc[result["valid"].astype(bool)].copy()

    if len(valid) < 20:
        raise RuntimeError("Too few valid F4c region-timing bootstrap replicates.")

    return result, valid, region_lower, region_upper


def percentile_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "q2.5": float(np.quantile(values, 0.025)),
        "q97.5": float(np.quantile(values, 0.975)),
        "p_gt_zero": float(np.mean(values > 0.0)),
    }


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


def make_figure(
    f4b_results,
    f4c_valid,
    frozen_rna_center,
    frozen_atac_center,
    region_lower,
    region_upper,
    materiality_threshold,
):
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

    # A. Diagnostic frozen-family detection rates (F4b)
    detection_labels = ["RNA", "ATAC", "Paired"]
    detection_rates = [
        float(f4b_results["rna_family_detected"].astype(bool).mean()),
        float(f4b_results["atac_family_detected"].astype(bool).mean()),
        float(f4b_results["paired_family_detected"].astype(bool).mean()),
    ]

    bars = ax_a.bar(detection_labels, detection_rates)
    ax_a.axhline(
        0.80,
        linestyle="--",
        linewidth=1.4,
        label="Support threshold = 0.80",
    )
    for bar, rate in zip(bars, detection_rates):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            min(rate + 0.025, 0.97),
            f"{rate:.3f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )
    ax_a.set_ylim(0, 1.0)
    ax_a.set_ylabel("Detection rate")
    ax_a.set_title(
        f"Frozen-family recovery (F4b; n = {len(f4b_results)})",
        fontweight="bold",
    )
    ax_a.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        fontsize=ANNOTATION_FONTSIZE,
    )
    ax_a.grid(axis="y", alpha=0.2)
    add_panel_label(ax_a, "A")

    # B. Tracked peak locations
    rna_pt = pd.to_numeric(
        f4b_results["rna_tracked_peak_pt"], errors="coerce"
    ).dropna().to_numpy()
    atac_pt = pd.to_numeric(
        f4b_results["atac_tracked_peak_pt"], errors="coerce"
    ).dropna().to_numpy()

    bins = np.linspace(
        min(np.min(rna_pt), np.min(atac_pt)),
        max(np.max(rna_pt), np.max(atac_pt)),
        24,
    )
    ax_b.hist(rna_pt, bins=bins, histtype="step", linewidth=2.0, label="RNA")
    ax_b.hist(atac_pt, bins=bins, histtype="step", linewidth=2.0, label="ATAC")
    ax_b.axvline(
        frozen_rna_center,
        linestyle="--",
        linewidth=1.4,
        label=f"Frozen RNA = {frozen_rna_center:.3f}",
    )
    ax_b.axvline(
        frozen_atac_center,
        linestyle=":",
        linewidth=1.8,
        label=f"Frozen ATAC = {frozen_atac_center:.3f}",
    )
    ax_b.set_xlabel("Tracked peak pseudotime")
    ax_b.set_ylabel("Bootstrap replicates")
    ax_b.set_title("Tracked peak-location stability", fontweight="bold")
    ax_b.legend(frameon=False, ncol=2)
    ax_b.grid(axis="y", alpha=0.2)
    add_panel_label(ax_b, "B")

    # C. Tracked peak relative prominence
    rna_prom = pd.to_numeric(
        f4b_results["rna_tracked_relative_prominence"], errors="coerce"
    ).dropna().to_numpy()
    atac_prom = pd.to_numeric(
        f4b_results["atac_tracked_relative_prominence"], errors="coerce"
    ).dropna().to_numpy()

    bins_prom = np.linspace(
        min(np.min(rna_prom), np.min(atac_prom)),
        max(np.max(rna_prom), np.max(atac_prom)),
        24,
    )
    ax_c.hist(
        rna_prom,
        bins=bins_prom,
        histtype="step",
        linewidth=2.0,
        label="RNA",
    )
    ax_c.hist(
        atac_prom,
        bins=bins_prom,
        histtype="step",
        linewidth=2.0,
        label="ATAC",
    )
    ax_c.axvline(
        materiality_threshold,
        linestyle="--",
        linewidth=1.4,
        label=f"Materiality threshold = {materiality_threshold:.2f}",
    )
    ax_c.set_xlabel("Tracked relative prominence")
    ax_c.set_ylabel("Bootstrap replicates")
    ax_c.set_title("Tracked peak-prominence stability", fontweight="bold")
    ax_c.legend(frameon=False)
    ax_c.grid(axis="y", alpha=0.2)
    add_panel_label(ax_c, "C")

    # D. Region-based timing uncertainty (F4c)
    metrics = [
        ("early_lead_q25", "Q25"),
        ("center_lead_q50", "Q50"),
        ("late_lead_q75", "Q75"),
    ]

    x = np.arange(len(metrics), dtype=float)
    medians = []
    lower_err = []
    upper_err = []
    annotations = []

    for col, label in metrics:
        stats = percentile_summary(f4c_valid[col])
        medians.append(stats["median"])
        lower_err.append(stats["median"] - stats["q2.5"])
        upper_err.append(stats["q97.5"] - stats["median"])
        annotations.append(
            f"{label}: {stats['median']:.3f}\n"
            f"[{stats['q2.5']:.3f}, {stats['q97.5']:.3f}]"
        )

    ax_d.errorbar(
        x,
        medians,
        yerr=np.vstack([lower_err, upper_err]),
        fmt="o",
        capsize=6,
        linewidth=1.8,
        markersize=7,
    )
    ax_d.axhline(0.0, linestyle="--", linewidth=1.5)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(["Early Q25", "Center Q50", "Late Q75"])
    ax_d.set_ylabel(
        r"RNA $-$ ATAC regional timing (pseudotime)"
    )
    ax_d.set_title(
        f"Exploratory event-region timing (F4c; n = {len(f4c_valid)})",
        fontweight="bold",
    )
    ax_d.grid(axis="y", alpha=0.2)

    # Add headroom so CI annotations remain inside the axes and do not overlap the title.
    current_ymin, current_ymax = ax_d.get_ylim()
    data_upper = max(m + e for m, e in zip(medians, upper_err))
    data_lower = min(m - e for m, e in zip(medians, lower_err))
    y_span = max(data_upper - data_lower, 1e-6)
    new_ymax = data_upper + 0.30 * y_span
    new_ymin = min(current_ymin, data_lower - 0.08 * y_span)
    ax_d.set_ylim(new_ymin, new_ymax)

    annotation_offset = 0.035 * (new_ymax - new_ymin)
    for xi, med, up_err, text in zip(x, medians, upper_err, annotations):
        y_text = med + up_err + annotation_offset
        ax_d.text(
            xi,
            y_text,
            text,
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )
    add_panel_label(ax_d, "D")

    fig.savefig(PNG_OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    print("Loading frozen F4b/F4c discovery modules...")
    f4b = load_module("gdis_f4b", F4B_SCRIPT)
    f4c = load_module("gdis_f4c", F4C_SCRIPT)

    print("Loading GSE275562 and reproducing the frozen discovery trajectory...")
    (
        obs,
        X_rna_z,
        X_atac_z,
        pseudotime,
        frozen_rna_center,
        frozen_atac_center,
        critical_value,
    ) = load_discovery_data(f4b)

    print(f"Cells: {len(obs):,}")
    print(f"Frozen RNA center:  {frozen_rna_center:.6f}")
    print(f"Frozen ATAC center: {frozen_atac_center:.6f}")

    # Reuse completed bootstrap tables when available. This is especially
    # useful if a previous run completed F4b/F4c but failed during plotting.
    region_lower = max(
        0.0, frozen_atac_center - f4c.FAMILY_MATCH_TOLERANCE
    )
    region_upper = min(
        1.0, frozen_rna_center + f4c.FAMILY_MATCH_TOLERANCE
    )

    if F4B_TABLE.exists() and F4C_TABLE.exists():
        print("\nExisting F4b/F4c bootstrap tables detected; reusing them.")
        print(f"  F4b: {F4B_TABLE}")
        print(f"  F4c: {F4C_TABLE}")
        f4b_results = pd.read_csv(F4B_TABLE, sep="\t")
        f4c_all = pd.read_csv(F4C_TABLE, sep="\t")
        f4c_valid = f4c_all.loc[f4c_all["valid"].astype(bool)].copy()
    else:
        print("\nRunning the frozen F4b diagnostic bootstrap...")
        f4b_results = run_f4b_diagnostic(
            f4b,
            X_rna_z,
            X_atac_z,
            pseudotime,
            frozen_rna_center,
            frozen_atac_center,
            critical_value,
        )

        print("\nRunning the frozen F4c region-timing bootstrap...")
        f4c_all, f4c_valid, region_lower, region_upper = (
            run_f4c_region_bootstrap(
                f4c,
                X_rna_z,
                X_atac_z,
                pseudotime,
                frozen_rna_center,
                frozen_atac_center,
                critical_value,
            )
        )

        f4b_results.to_csv(
            F4B_TABLE, sep="\t", index=False, compression="gzip"
        )
        f4c_all.to_csv(
            F4C_TABLE, sep="\t", index=False, compression="gzip"
        )

    summary_rows = [
        {
            "metric": "F4b RNA frozen-family detection rate",
            "value": float(
                f4b_results["rna_family_detected"].astype(bool).mean()
            ),
        },
        {
            "metric": "F4b ATAC frozen-family detection rate",
            "value": float(
                f4b_results["atac_family_detected"].astype(bool).mean()
            ),
        },
        {
            "metric": "F4b paired frozen-family detection rate",
            "value": float(
                f4b_results["paired_family_detected"].astype(bool).mean()
            ),
        },
        {
            "metric": "F4c valid region-timing rate",
            "value": float(len(f4c_valid) / f4c.N_BOOTSTRAPS),
        },
    ]

    for col, label in [
        ("early_lead_q25", "F4c Q25 lead"),
        ("center_lead_q50", "F4c Q50 lead"),
        ("late_lead_q75", "F4c Q75 lead"),
    ]:
        stats = percentile_summary(f4c_valid[col])
        summary_rows.extend([
            {"metric": f"{label} median", "value": stats["median"]},
            {"metric": f"{label} 95% CI lower", "value": stats["q2.5"]},
            {"metric": f"{label} 95% CI upper", "value": stats["q97.5"]},
            {"metric": f"{label} P(lead>0)", "value": stats["p_gt_zero"]},
        ])

    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_TABLE, sep="\t", index=False
    )

    print("\nGenerating Supplementary Figure S1...")
    make_figure(
        f4b_results,
        f4c_valid,
        frozen_rna_center,
        frozen_atac_center,
        region_lower,
        region_upper,
        f4b.MIN_RELATIVE_PROMINENCE,
    )

    print("\nSupplementary Figure S1 generated successfully.")
    print(f"PNG (600 dpi): {PNG_OUT}")
    print(f"PDF (vector):  {PDF_OUT}")
    print(f"F4b table:     {F4B_TABLE}")
    print(f"F4c table:     {F4C_TABLE}")
    print(f"Summary:       {SUMMARY_TABLE}")


if __name__ == "__main__":
    main()

