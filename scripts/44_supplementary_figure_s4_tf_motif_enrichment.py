#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
44_supplementary_figure_s4_tf_motif_enrichment.py
=================================================

Generate Supplementary Figure S4:
TF-motif enrichment diagnostic for the SHARE-seq TAC->IRS mechanistic follow-up.

Figure design
-------------
A compact volcano/effect-size plot showing:
    - no embedded figure title (caption carries the title for publication)
    - x-axis: odds ratio
    - y-axis: -log10(BH FDR)
    - KLF8 labeled as the strongest nominal signal
    - prespecified enrichment thresholds:
          BH_FDR <= 0.05
          odds_ratio >= 1.5
          candidate_hit_n >= 5
    - explicit visual emphasis that no motif crosses the final criteria

Input
-----
data/GSE140203/representations_f21/f21_motif_enrichment.tsv

Primary output
--------------
data/GSE140203/figures_f24/
    Supplementary_Figure_S4_TF_motif_enrichment_diagnostic.png
    Supplementary_Figure_S4_TF_motif_enrichment_diagnostic.pdf
    Supplementary_Figure_S4_TF_motif_enrichment_summary.tsv

Run
---
python 44_supplementary_figure_s4_tf_motif_enrichment.py
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ACCESSION = "GSE140203"
DATA_DIR = Path("data") / ACCESSION
F21_DIR = DATA_DIR / "representations_f21"
FIG_DIR = DATA_DIR / "figures_f24"

INPUT_ENRICHMENT = F21_DIR / "f21_motif_enrichment.tsv"
OUTPUT_PNG = FIG_DIR / "Supplementary_Figure_S4_TF_motif_enrichment_diagnostic.png"
OUTPUT_PDF = FIG_DIR / "Supplementary_Figure_S4_TF_motif_enrichment_diagnostic.pdf"
OUTPUT_SUMMARY = FIG_DIR / "Supplementary_Figure_S4_TF_motif_enrichment_summary.tsv"


# -----------------------------------------------------------------------------
# Frozen thresholds from Phase F21
# -----------------------------------------------------------------------------

ENRICHMENT_FDR = 0.05
ENRICHMENT_MIN_OR = 1.5
ENRICHMENT_MIN_CANDIDATE_HITS = 5

# Plot styling
DPI = 600
FIGSIZE = (8.8, 6.8)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file was not found:\n{path}\n\n"
            "Run Phase F21 first to create the motif-enrichment table."
        )

def find_klf8_row(df: pd.DataFrame) -> pd.Series:
    # Prefer gene-symbol match, then JASPAR TF name match.
    gene_match = df["mapped_gene_symbol"].astype(str).str.upper().eq("KLF8")
    name_match = df["jaspar_tf_name"].astype(str).str.upper().str.contains("KLF8", regex=False)

    candidates = df.loc[gene_match | name_match].copy()
    if candidates.empty:
        # fallback: strongest nominal row
        return df.sort_values(["fisher_p", "BH_FDR", "odds_ratio"], ascending=[True, True, False]).iloc[0]
    return candidates.sort_values(["fisher_p", "BH_FDR", "odds_ratio"], ascending=[True, True, False]).iloc[0]

def prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "odds_ratio",
        "BH_FDR",
        "fisher_p",
        "candidate_hit_n",
        "candidate_n",
        "background_hit_n",
        "background_n",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace invalid OR values for plotting if needed.
    df["odds_ratio_plot"] = df["odds_ratio"].replace([np.inf, -np.inf], np.nan)
    finite_or = df["odds_ratio_plot"].dropna()
    if finite_or.empty:
        raise ValueError("No finite odds-ratio values were found in the motif table.")

    # BH_FDR should be in (0, 1]; floor tiny values for plotting stability.
    floor = 1e-300
    df["BH_FDR_plot"] = df["BH_FDR"].clip(lower=floor)
    df["neglog10_BH_FDR"] = -np.log10(df["BH_FDR_plot"])

    # Final prespecified significance.
    df["passes_all"] = (
        (df["BH_FDR"] <= ENRICHMENT_FDR)
        & (df["odds_ratio"] >= ENRICHMENT_MIN_OR)
        & (df["candidate_hit_n"] >= ENRICHMENT_MIN_CANDIDATE_HITS)
    )

    # "Nominal" effect-size candidate (for optional styling only)
    df["passes_or_and_hits"] = (
        (df["odds_ratio"] >= ENRICHMENT_MIN_OR)
        & (df["candidate_hit_n"] >= ENRICHMENT_MIN_CANDIDATE_HITS)
    )

    return df

def make_summary(df: pd.DataFrame, klf8: pd.Series) -> pd.DataFrame:
    n_total = len(df)
    n_fdr = int((df["BH_FDR"] <= ENRICHMENT_FDR).sum())
    n_or = int((df["odds_ratio"] >= ENRICHMENT_MIN_OR).sum())
    n_hits = int((df["candidate_hit_n"] >= ENRICHMENT_MIN_CANDIDATE_HITS).sum())
    n_all = int(df["passes_all"].sum())

    summary_rows = [
        ["eligible_motifs", n_total],
        ["motifs_with_BH_FDR_le_0.05", n_fdr],
        ["motifs_with_OR_ge_1.5", n_or],
        ["motifs_with_candidate_hits_ge_5", n_hits],
        ["motifs_passing_all_prespecified_criteria", n_all],
        ["KLF8_odds_ratio", float(klf8["odds_ratio"])],
        ["KLF8_fisher_p", float(klf8["fisher_p"])],
        ["KLF8_BH_FDR", float(klf8["BH_FDR"])],
        ["KLF8_candidate_hit_n", int(klf8["candidate_hit_n"])],
    ]
    return pd.DataFrame(summary_rows, columns=["metric", "value"])

def plot_figure(df: pd.DataFrame, klf8: pd.Series) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Scatter layers
    nonsig = df.loc[~df["passes_all"]]
    sig = df.loc[df["passes_all"]]

    ax.scatter(
        nonsig["odds_ratio_plot"],
        nonsig["neglog10_BH_FDR"],
        s=28,
        alpha=0.72,
        edgecolors="none",
        label="Eligible TF motifs",
    )

    if not sig.empty:
        ax.scatter(
            sig["odds_ratio_plot"],
            sig["neglog10_BH_FDR"],
            s=42,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.4,
            label="Pass all final criteria",
        )

    # Threshold lines
    ax.axvline(
        ENRICHMENT_MIN_OR,
        linestyle="--",
        linewidth=1.3,
    )
    ax.axhline(
        -math.log10(ENRICHMENT_FDR),
        linestyle="--",
        linewidth=1.3,
    )

    # Highlight KLF8
    kx = float(klf8["odds_ratio_plot"])
    ky = float(klf8["neglog10_BH_FDR"])
    ax.scatter(
        [kx],
        [ky],
        s=80,
        marker="o",
        edgecolors="black",
        linewidths=0.7,
        zorder=5,
        label="KLF8 (strongest nominal signal)",
    )
    klf8_label = (
        "KLF8\n"
        f"OR={float(klf8['odds_ratio']):.2f}\n"
        f"FDR={float(klf8['BH_FDR']):.3f}"
    )
    ax.annotate(
        klf8_label,
        xy=(kx, ky),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=9.5,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.7),
        arrowprops=dict(arrowstyle="-", linewidth=0.8),
    )

    # Axis labels
    ax.set_xlabel("Motif enrichment odds ratio", fontsize=11)
    ax.set_ylabel(r"$-\log_{10}(\mathrm{BH\ FDR})$", fontsize=11)

    # Set x-limits with a bit of margin
    finite_or = df["odds_ratio_plot"].dropna()
    xmin = max(0, float(finite_or.min()) * 0.92)
    xmax = float(finite_or.max()) * 1.08
    if xmax <= 0:
        xmax = 2.0
    ax.set_xlim(xmin, xmax)

    # y-limit
    ymax = max(float(df["neglog10_BH_FDR"].max()) * 1.12, -math.log10(ENRICHMENT_FDR) * 1.6)
    ax.set_ylim(0, ymax)

    # Summary box
    n_total = len(df)
    n_all = int(df["passes_all"].sum())
    text = (
        f"Eligible motifs: {n_total}\n"
        f"Final enriched motifs: {n_all}\n"
        "Prespecified final criteria:\n"
        f"  BH FDR ≤ {ENRICHMENT_FDR}\n"
        f"  OR ≥ {ENRICHMENT_MIN_OR}\n"
        f"  candidate hits ≥ {ENRICHMENT_MIN_CANDIDATE_HITS}\n"
        "\n"
        "No motif crossed all final criteria."
    )
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.3,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", linewidth=0.7),
    )

    # Compact threshold note (use caption for the full figure title)
    threshold_note = (
        f"Thresholds: OR ≥ {ENRICHMENT_MIN_OR}; "
        f"BH FDR ≤ {ENRICHMENT_FDR}; "
        f"candidate hits ≥ {ENRICHMENT_MIN_CANDIDATE_HITS}"
    )
    ax.text(
        0.02,
        0.02,
        threshold_note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.5),
    )

    ax.legend(loc="lower right", frameon=True, fontsize=8.8)
    fig.tight_layout()

    fig.savefig(OUTPUT_PNG, dpi=DPI)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)

def main():
    print("=" * 110)
    print("SUPPLEMENTARY FIGURE S4 — TF-MOTIF ENRICHMENT DIAGNOSTIC")
    print("=" * 110)
    print(f"Input motif table: {INPUT_ENRICHMENT}")
    print("Output directory: ", FIG_DIR)
    print()

    require_file(INPUT_ENRICHMENT)
    df = pd.read_csv(INPUT_ENRICHMENT, sep="\t")
    df = prepare_table(df)

    klf8 = find_klf8_row(df)

    summary = make_summary(df, klf8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_SUMMARY, sep="\t", index=False)

    print(f"Eligible motifs: {len(df)}")
    print(f"Motifs passing all final criteria: {int(df['passes_all'].sum())}")
    print("Strongest nominal signal selected for labeling:")
    print(
        f"  {klf8['mapped_gene_symbol']} | motif_id={klf8['motif_id']} | "
        f"OR={float(klf8['odds_ratio']):.4f} | "
        f"p={float(klf8['fisher_p']):.4g} | "
        f"BH FDR={float(klf8['BH_FDR']):.4g}"
    )
    print()

    plot_figure(df, klf8)

    print("Saved:")
    print(f"  {OUTPUT_PNG}")
    print(f"  {OUTPUT_PDF}")
    print(f"  {OUTPUT_SUMMARY}")
    print()
    print("Interpretation:")
    print("  This is a diagnostic visualization only.")
    print("  The figure emphasizes that no motif crosses the full prespecified enrichment criteria.")

if __name__ == "__main__":
    main()

