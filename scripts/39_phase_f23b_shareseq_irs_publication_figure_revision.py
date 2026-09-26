#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
39_phase_f23b_shareseq_irs_publication_figure_revision.py
=========================================================

Presentation-only revision of F23 figures.

NO new scientific/statistical analysis is performed.

Revisions:
1. Main Figure A uses ACTUAL primary 400/100 frozen event peaks:
       ATAC = 0.123472
       RNA  = 0.154080
   rather than three-window family centers.

2. Main Figure B clearly distinguishes the zero/synchronous reference line.

3. Supplementary Figure S2 explicitly states that positive RNA-ATAC timing
   means ATAC is earlier.

4. Supplementary Figures S3-S5 display the already-frozen empirical
   circular-shift P values.

5. Supplementary Figure S6 labels only the leading nominal motif to avoid
   overlapping annotations. No enrichment threshold is changed.

All outputs remain:
    PNG
    600 dpi
    black text
    large fonts

RUN
---
python 39_phase_f23b_shareseq_irs_publication_figure_revision.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# PATHS
# =====================================================================

DATA_DIR = Path("data") / "GSE140203"

F14_DIR = DATA_DIR / "representations_f14"
F17_DIR = DATA_DIR / "representations_f17"
F18_DIR = DATA_DIR / "representations_f18"
F21_DIR = DATA_DIR / "representations_f21"
F22_DIR = DATA_DIR / "representations_f22"

OUT_DIR = DATA_DIR / "figures_f23b"

F14_PROFILES = F14_DIR / "f14_gdis_profiles.tsv.gz"
F14_LANDMARKS = F14_DIR / "f14_independent_state_landmarks.tsv"

F17_REPLICATES = F17_DIR / "f17_bootstrap_replicates.tsv.gz"
F17_SUMMARY = F17_DIR / "f17_bootstrap_summary.tsv"

F18_CIRCULAR = F18_DIR / "f18_circular_shift_null.tsv.gz"
F18_SUMMARY = F18_DIR / "f18_alignment_summary.tsv"

F21_ENRICHMENT = F21_DIR / "f21_motif_enrichment.tsv"

F22_WINDOWS = F22_DIR / "f22_window_specific_timing.tsv"
F22_MANIFEST = F22_DIR / "f22_manifest.json"

MAIN_A = OUT_DIR / "Main_Figure_A_SHAREseq_transition_energy_profiles_REVISED.png"
MAIN_B = OUT_DIR / "Main_Figure_B_SHAREseq_bootstrap_CMIL_REVISED.png"

SUPP_S1 = OUT_DIR / "Supplementary_Figure_S1_sustained_GDIS_profiles_REVISED.png"
SUPP_S2 = OUT_DIR / "Supplementary_Figure_S2_window_specific_timing_REVISED.png"
SUPP_S3 = OUT_DIR / "Supplementary_Figure_S3_alignment_null_spearman_REVISED.png"
SUPP_S4 = OUT_DIR / "Supplementary_Figure_S4_alignment_null_q50_REVISED.png"
SUPP_S5 = OUT_DIR / "Supplementary_Figure_S5_alignment_null_JS_REVISED.png"
SUPP_S6 = OUT_DIR / "Supplementary_Figure_S6_motif_enrichment_diagnostic_REVISED.png"

OUTPUT_MANIFEST = OUT_DIR / "f23b_manifest.json"


# =====================================================================
# SETTINGS
# =====================================================================

PRIMARY_PROFILE_ROLE = "primary_multiome"
PRIMARY_WINDOW_CONFIG = "primary_w400_s100"

DPI = 600

FIGSIZE_WIDE = (8.8, 5.8)
FIGSIZE_STANDARD = (7.2, 5.8)

FONT_SIZE = 15
LABEL_SIZE = 16
TITLE_SIZE = 17
LEGEND_SIZE = 12.5
TICK_SIZE = 13

HIST_BINS = 35


# =====================================================================
# UTILITIES
# =====================================================================

def section(title):
    print("\n" + "=" * 124)
    print(title)
    print("=" * 124)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def require_inputs():
    required = [
        F14_PROFILES,
        F14_LANDMARKS,
        F17_REPLICATES,
        F17_SUMMARY,
        F18_CIRCULAR,
        F18_SUMMARY,
        F21_ENRICHMENT,
        F22_WINDOWS,
        F22_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required frozen result(s):\n"
            + "\n".join(f"  {path}" for path in missing)
        )


def apply_publication_axes(ax):
    ax.tick_params(
        axis="both",
        labelsize=TICK_SIZE,
        width=1.2,
    )

    for label in ax.get_xticklabels():
        label.set_color("black")

    for label in ax.get_yticklabels():
        label.set_color("black")

    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def save_figure(fig, path):
    fig.tight_layout()

    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(path)


def load_results():
    profiles = pd.read_csv(
        F14_PROFILES,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    landmarks = pd.read_csv(
        F14_LANDMARKS,
        sep="\t",
        low_memory=False,
    )

    replicates = pd.read_csv(
        F17_REPLICATES,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    bootstrap = pd.read_csv(
        F17_SUMMARY,
        sep="\t",
        low_memory=False,
    )

    circular = pd.read_csv(
        F18_CIRCULAR,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    alignment = pd.read_csv(
        F18_SUMMARY,
        sep="\t",
        low_memory=False,
    )

    motifs = pd.read_csv(
        F21_ENRICHMENT,
        sep="\t",
        low_memory=False,
    )

    windows = pd.read_csv(
        F22_WINDOWS,
        sep="\t",
        low_memory=False,
    )

    if len(bootstrap) != 1:
        raise RuntimeError("F17 summary must contain exactly one row.")

    if len(alignment) != 1:
        raise RuntimeError("F18 summary must contain exactly one row.")

    return {
        "profiles": profiles,
        "landmarks": landmarks,
        "replicates": replicates,
        "bootstrap": bootstrap.iloc[0],
        "circular": circular,
        "alignment": alignment.iloc[0],
        "motifs": motifs,
        "windows": windows,
    }


# =====================================================================
# MAIN FIGURE A
# =====================================================================

def plot_transition_energy_profiles(data):
    profiles = data["profiles"]
    landmarks = data["landmarks"]
    windows = data["windows"]

    primary = profiles.loc[
        (profiles["profile_role"].astype(str) == PRIMARY_PROFILE_ROLE)
        & (profiles["window_config"].astype(str) == PRIMARY_WINDOW_CONFIG)
    ].copy()

    rna = primary.loc[
        primary["modality"].astype(str) == "RNA"
    ].sort_values("parameter")

    atac = primary.loc[
        primary["modality"].astype(str) == "ATAC"
    ].sort_values("parameter")

    tac2 = landmarks.loc[
        landmarks["state"].astype(str) == "TAC-2"
    ]

    if len(tac2) != 1:
        raise RuntimeError("Could not identify unique TAC-2 landmark.")

    tac2 = tac2.iloc[0]

    primary_pair = windows.loc[
        windows["window_config"].astype(str) == PRIMARY_WINDOW_CONFIG
    ]

    if len(primary_pair) != 1:
        raise RuntimeError(
            "Could not identify unique primary 400/100 F22 timing row."
        )

    primary_pair = primary_pair.iloc[0]

    atac_event = float(
        primary_pair["ATAC_peak_parameter"]
    )

    rna_event = float(
        primary_pair["RNA_peak_parameter"]
    )

    observed_cmil = float(
        primary_pair["RNA_minus_ATAC_parameter"]
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE_WIDE
    )

    ax.plot(
        rna["parameter"],
        rna["transition_energy"],
        linewidth=2.3,
        marker="o",
        markersize=3.5,
        label="RNA transition energy",
    )

    ax.plot(
        atac["parameter"],
        atac["transition_energy"],
        linewidth=2.3,
        marker="s",
        markersize=3.5,
        label="ATAC transition energy",
    )

    ax.axvspan(
        float(tac2["q25"]),
        float(tac2["q75"]),
        alpha=0.09,
        label="TAC-2 IQR",
    )

    ax.axvline(
        atac_event,
        linestyle="--",
        linewidth=1.8,
        label="Frozen ATAC event (400/100)",
    )

    ax.axvline(
        rna_event,
        linestyle=":",
        linewidth=2.0,
        label="Frozen RNA event (400/100)",
    )

    ymax = max(
        float(rna["transition_energy"].max()),
        float(atac["transition_energy"].max()),
    )

    ax.text(
        (atac_event + rna_event) / 2.0,
        0.92 * ymax,
        f"Δ = {observed_cmil:.4f}",
        ha="center",
        va="top",
        fontsize=12.5,
        color="black",
    )

    ax.set_xlabel(
        "RNA-derived common pseudotime",
        fontsize=LABEL_SIZE,
    )

    ax.set_ylabel(
        "Transition energy",
        fontsize=LABEL_SIZE,
    )

    ax.set_title(
        "Localized RNA and ATAC transition-energy events on the validated TAC→IRS trajectory",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    ax.legend(
        fontsize=LEGEND_SIZE,
        frameon=False,
    )

    apply_publication_axes(ax)

    save_figure(
        fig,
        MAIN_A,
    )


# =====================================================================
# MAIN FIGURE B
# =====================================================================

def plot_bootstrap_cmil(data):
    replicates = data["replicates"]
    boot = data["bootstrap"]

    detected = replicates.loc[
        replicates["paired_detected"].astype(bool)
    ].copy()

    values = detected[
        "CMIL_RNA_minus_ATAC"
    ].to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE_STANDARD
    )

    ax.hist(
        values,
        bins=HIST_BINS,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    ax.axvline(
        float(
            boot["observed_primary_CMIL_RNA_minus_ATAC"]
        ),
        linewidth=2.0,
        label="Observed primary CMIL",
    )

    ax.axvline(
        float(
            boot["bootstrap_CMIL_median"]
        ),
        linestyle="--",
        linewidth=2.0,
        label="Bootstrap median",
    )

    ax.axvline(
        float(
            boot["bootstrap_CMIL_ci025"]
        ),
        linestyle=":",
        linewidth=1.8,
        label="95% CI",
    )

    ax.axvline(
        float(
            boot["bootstrap_CMIL_ci975"]
        ),
        linestyle=":",
        linewidth=1.8,
    )

    # Distinct line style so x=0 cannot be mistaken for observed CMIL.
    ax.axvline(
        0.0,
        linestyle="-.",
        linewidth=1.2,
        alpha=0.65,
        label="0 (synchronous)",
    )

    ax.set_xlabel(
        "CMIL = RNA event pseudotime − ATAC event pseudotime\n"
        "(positive values indicate ATAC earlier)",
        fontsize=LABEL_SIZE,
    )

    ax.set_ylabel(
        "Bootstrap replicates",
        fontsize=LABEL_SIZE,
    )

    ax.set_title(
        "Bootstrap support for the frozen ATAC-earlier localized event",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    annotation = (
        f"Paired detections = "
        f"{int(boot['paired_detection_n'])}/"
        f"{int(boot['n_bootstrap'])}\n"
        f"Median = {float(boot['bootstrap_CMIL_median']):.4f}\n"
        f"95% CI = "
        f"[{float(boot['bootstrap_CMIL_ci025']):.4f}, "
        f"{float(boot['bootstrap_CMIL_ci975']):.4f}]\n"
        f"P(CMIL > 0) = "
        f"{float(boot['P_boot_CMIL_gt_0']):.4f}"
    )

    ax.text(
        0.98,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        color="black",
    )

    ax.legend(
        fontsize=11.5,
        frameon=False,
        loc="upper left",
    )

    apply_publication_axes(ax)

    save_figure(
        fig,
        MAIN_B,
    )


# =====================================================================
# SUPPLEMENTARY S1
# =====================================================================

def plot_sustained_gdis(data):
    profiles = data["profiles"]

    primary = profiles.loc[
        (profiles["profile_role"].astype(str) == PRIMARY_PROFILE_ROLE)
        & (profiles["window_config"].astype(str) == PRIMARY_WINDOW_CONFIG)
    ].copy()

    fig, ax = plt.subplots(
        figsize=FIGSIZE_WIDE
    )

    for modality, marker in [
        ("RNA", "o"),
        ("ATAC", "s"),
    ]:
        subset = primary.loc[
            primary["modality"].astype(str) == modality
        ].sort_values("parameter")

        ax.plot(
            subset["parameter"],
            subset["gdis"],
            linewidth=2.3,
            marker=marker,
            markersize=3.5,
            label=modality,
        )

    ax.set_xlabel(
        "RNA-derived common pseudotime",
        fontsize=LABEL_SIZE,
    )

    ax.set_ylabel(
        "Sustained GDIS (λₜ = 0)",
        fontsize=LABEL_SIZE,
    )

    ax.set_title(
        "Distinct sustained-instability architectures in RNA and ATAC",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    ax.legend(
        fontsize=LEGEND_SIZE,
        frameon=False,
    )

    apply_publication_axes(ax)

    save_figure(
        fig,
        SUPP_S1,
    )


# =====================================================================
# SUPPLEMENTARY S2
# =====================================================================

def plot_window_specific_timing(data):
    windows = data["windows"].copy()

    label_map = {
        "sensitivity_w300_s75": "300 / 75",
        "primary_w400_s100": "400 / 100",
        "sensitivity_w500_s125": "500 / 125",
    }

    windows["display_label"] = windows[
        "window_config"
    ].map(
        label_map
    )

    order = [
        "300 / 75",
        "400 / 100",
        "500 / 125",
    ]

    windows["display_label"] = pd.Categorical(
        windows["display_label"],
        categories=order,
        ordered=True,
    )

    windows = windows.sort_values(
        "display_label"
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE_STANDARD
    )

    x = np.arange(
        len(windows)
    )

    ax.plot(
        x,
        windows[
            "RNA_minus_ATAC_parameter"
        ],
        marker="o",
        linewidth=2.0,
        markersize=7,
    )

    ax.axhline(
        0.0,
        linestyle="-.",
        linewidth=1.1,
        alpha=0.65,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        windows["display_label"].astype(str),
        fontsize=TICK_SIZE,
    )

    ax.set_xlabel(
        "Window size / step",
        fontsize=LABEL_SIZE,
    )

    ax.set_ylabel(
        "RNA − ATAC event pseudotime\n"
        "(positive = ATAC earlier)",
        fontsize=LABEL_SIZE,
    )

    ax.set_title(
        "Frozen event timing is directionally consistent across window designs",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    for x_value, value in zip(
        x,
        windows["RNA_minus_ATAC_parameter"],
    ):
        ax.text(
            x_value,
            float(value) + 0.0015,
            f"{float(value):.4f}",
            ha="center",
            va="bottom",
            fontsize=12.5,
            color="black",
        )

    apply_publication_axes(ax)

    save_figure(
        fig,
        SUPP_S2,
    )


# =====================================================================
# SUPPLEMENTARY NULL PANELS
# =====================================================================

def plot_null_histogram(
    circular,
    metric_column,
    observed_value,
    empirical_p,
    xlabel,
    title,
    output_path,
):
    values = circular[
        metric_column
    ].to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE_STANDARD
    )

    ax.hist(
        values,
        bins=20,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    ax.axvline(
        float(observed_value),
        linewidth=2.1,
        label="Observed",
    )

    ax.text(
        0.97,
        0.95,
        f"Empirical p = {float(empirical_p):.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        color="black",
    )

    ax.set_xlabel(
        xlabel,
        fontsize=LABEL_SIZE,
    )

    ax.set_ylabel(
        "Admissible circular shifts",
        fontsize=LABEL_SIZE,
    )

    ax.set_title(
        title,
        fontsize=TITLE_SIZE,
        pad=12,
    )

    ax.legend(
        fontsize=LEGEND_SIZE,
        frameon=False,
    )

    apply_publication_axes(ax)

    save_figure(
        fig,
        output_path,
    )


# =====================================================================
# SUPPLEMENTARY S6
# =====================================================================

def plot_motif_diagnostic(data):
    motifs = data["motifs"].copy()

    finite = (
        np.isfinite(
            motifs["odds_ratio"].to_numpy(dtype=float)
        )
        & np.isfinite(
            motifs["BH_FDR"].to_numpy(dtype=float)
        )
        & (
            motifs["odds_ratio"].to_numpy(dtype=float)
            > 0
        )
        & (
            motifs["BH_FDR"].to_numpy(dtype=float)
            > 0
        )
    )

    motifs = motifs.loc[
        finite
    ].copy()

    x = np.log2(
        motifs["odds_ratio"].to_numpy(
            dtype=float
        )
    )

    y = -np.log10(
        motifs["BH_FDR"].to_numpy(
            dtype=float
        )
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE_STANDARD
    )

    ax.scatter(
        x,
        y,
        s=26,
        alpha=0.75,
    )

    ax.axvline(
        np.log2(1.5),
        linestyle="--",
        linewidth=1.5,
        label="OR = 1.5",
    )

    ax.axhline(
        -np.log10(0.05),
        linestyle=":",
        linewidth=1.5,
        label="BH-FDR = 0.05",
    )

    ax.set_xlabel(
        "log₂ odds ratio",
        fontsize=LABEL_SIZE,
    )

    ax.set_ylabel(
        "−log₁₀(BH-FDR)",
        fontsize=LABEL_SIZE,
    )

    ax.set_title(
        "No expressed single-TF motif passes the prespecified enrichment criteria",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    # Label only the leading nominal motif; all others remain visible as points.
    best = motifs.sort_values(
        [
            "BH_FDR",
            "fisher_p",
        ],
        ascending=[
            True,
            True,
        ],
    ).iloc[0]

    best_x = np.log2(
        float(
            best["odds_ratio"]
        )
    )

    best_y = -np.log10(
        float(
            best["BH_FDR"]
        )
    )

    ax.annotate(
        (
            f"{best['jaspar_tf_name']}\n"
            f"FDR={float(best['BH_FDR']):.3f}"
        ),
        xy=(
            best_x,
            best_y,
        ),
        xytext=(
            12,
            12,
        ),
        textcoords="offset points",
        fontsize=11.5,
        color="black",
    )

    ax.legend(
        fontsize=11.5,
        frameon=False,
        loc="upper left",
    )

    apply_publication_axes(ax)

    save_figure(
        fig,
        SUPP_S6,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F23b — PRESENTATION-ONLY REVISION OF SHARE-seq FIGURES"
    )

    print(
        "No scientific/statistical inference is recomputed."
    )

    require_inputs()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
        }
    )

    data = load_results()

    section(
        "1. REVISED MAIN-TEXT FIGURES"
    )

    plot_transition_energy_profiles(
        data
    )

    plot_bootstrap_cmil(
        data
    )

    section(
        "2. REVISED SUPPLEMENTARY FIGURES"
    )

    plot_sustained_gdis(
        data
    )

    plot_window_specific_timing(
        data
    )

    align = data["alignment"]
    circular = data["circular"]

    plot_null_histogram(
        circular,
        "spearman_rho",
        float(
            align["observed_spearman_rho"]
        ),
        float(
            align["circular_p_spearman_rho"]
        ),
        "Spearman correlation",
        "Primary circular-shift null: RNA–ATAC transition-energy correlation",
        SUPP_S3,
    )

    plot_null_histogram(
        circular,
        "absolute_q50_separation",
        float(
            align["observed_absolute_q50_separation"]
        ),
        float(
            align["circular_p_abs_q50_separation"]
        ),
        "Absolute q50 separation",
        "Primary circular-shift null: transition-energy median-location separation",
        SUPP_S4,
    )

    plot_null_histogram(
        circular,
        "JS_divergence_base2",
        float(
            align["observed_JS_divergence_base2"]
        ),
        float(
            align["circular_p_JS_divergence"]
        ),
        "Jensen–Shannon divergence",
        "Primary circular-shift null: RNA–ATAC transition-energy divergence",
        SUPP_S5,
    )

    plot_motif_diagnostic(
        data
    )

    # -----------------------------------------------------------------
    # Manifest.
    # -----------------------------------------------------------------

    section(
        "3. SAVE F23b MANIFEST"
    )

    outputs = [
        MAIN_A,
        MAIN_B,
        SUPP_S1,
        SUPP_S2,
        SUPP_S3,
        SUPP_S4,
        SUPP_S5,
        SUPP_S6,
    ]

    for path in outputs:
        if not path.exists():
            raise RuntimeError(
                f"Expected revised figure missing: {path}"
            )

    payload = {
        "dataset_accession": "GSE140203",
        "phase": "F23b",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "purpose": (
            "Presentation-only revision of frozen F23 publication figures."
        ),
        "scientific_analysis_changed": False,
        "revisions": [
            (
                "Main A uses primary 400/100 event peaks rather than "
                "three-window family centers."
            ),
            "Main B distinguishes zero/synchronous reference line.",
            "S2 explicitly defines positive timing as ATAC earlier.",
            "S3-S5 show frozen empirical circular-shift p-values.",
            "S6 removes overlapping motif labels and labels only leading nominal motif.",
        ],
        "figure_policy": {
            "format": "PNG",
            "dpi": DPI,
            "black_text": True,
            "large_fonts": True,
        },
        "outputs": {
            str(path): {
                "sha256": sha256_file(
                    path
                )
            }
            for path in outputs
        },
        "guardrails": {
            "new_GDIS_calculation": False,
            "new_pseudotime": False,
            "new_event_selection": False,
            "new_bootstrap": False,
            "new_null": False,
            "new_motif_scan": False,
            "new_pvalue": False,
            "threshold_changed": False,
        },
    }

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    print(
        OUTPUT_MANIFEST
    )

    section(
        "4. PHASE F23b DECISION"
    )

    print(
        "PHASE F23b VERDICT: GO — REVISED PUBLICATION FIGURES GENERATED "
        "WITHOUT CHANGING ANY FROZEN SCIENTIFIC RESULT"
    )


if __name__ == "__main__":
    main()

