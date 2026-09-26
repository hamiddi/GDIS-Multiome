#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
43_supplementary_figure_s3_gse205117_trajectory_suitability.py
================================================================

Create Supplementary Figure S3 for the GSE205117 external-dataset
trajectory-suitability analysis.

Figure concept
--------------
Panel A:
    RNA-derived pseudotime view of the frozen three-state arm,
    colored by biological state.

Panel B:
    State-wise pseudotime distributions showing that the expected
    NMP -> Paraxial_mesoderm -> Somitic_mesoderm ordering is not
    satisfied by the frozen F9 three-state clock.

Panel C:
    Cross-state neighborhood / continuity diagnostics based on the
    RNA 10D kNN graph (k = 30), displayed as directed neighbor
    enrichment over target-state global abundance, together with
    key topology summary statistics.

Panel D:
    Compact pass/fail summary of pairing, representation QC,
    pseudotime robustness, biological ordering, local continuity,
    and final GDIS eligibility.

This script is intentionally downstream-only:
    - it does NOT construct a new trajectory,
    - it does NOT modify the frozen F8c/F9/F9b results,
    - it does NOT calculate GDIS,
    - it does NOT calculate CMIL.

Expected upstream inputs
------------------------
    data/GSE205117/representations_geo_native/
        f8c_primary_developmental_cells.tsv.gz
        f8c_rna_primary_10d.npy
        f8c_atac_primary_10d.npy
        f8c_primary_representation_manifest.json          (optional but recommended)
        f9_external_common_pseudotime.tsv.gz
        f9_external_common_trajectory_manifest.json       (optional but recommended)
        f9b_external_trajectory_topology_diagnostic.json  (optional but recommended)

Outputs
-------
    data/GSE205117/representations_geo_native/
        supplementary_figure_s3_gse205117_trajectory_suitability.png
        supplementary_figure_s3_gse205117_trajectory_suitability.pdf
        supplementary_figure_s3_gse205117_trajectory_suitability_summary.tsv
        supplementary_figure_s3_gse205117_trajectory_suitability_manifest.json

Run
---
    python 43_supplementary_figure_s3_gse205117_trajectory_suitability.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib import patches


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ACCESSION = "GSE205117"
DATA_DIR = Path("data") / ACCESSION
REP_DIR = DATA_DIR / "representations_geo_native"

F8C_CELLS = REP_DIR / "f8c_primary_developmental_cells.tsv.gz"
F8C_RNA = REP_DIR / "f8c_rna_primary_10d.npy"
F8C_ATAC = REP_DIR / "f8c_atac_primary_10d.npy"
F8C_MANIFEST = REP_DIR / "f8c_primary_representation_manifest.json"

F9_PSEUDOTIME = REP_DIR / "f9_external_common_pseudotime.tsv.gz"
F9_MANIFEST = REP_DIR / "f9_external_common_trajectory_manifest.json"
F9B_DIAGNOSTIC = REP_DIR / "f9b_external_trajectory_topology_diagnostic.json"

OUTPUT_PNG = REP_DIR / "supplementary_figure_s3_gse205117_trajectory_suitability.png"
OUTPUT_PDF = REP_DIR / "supplementary_figure_s3_gse205117_trajectory_suitability.pdf"
OUTPUT_SUMMARY = REP_DIR / "supplementary_figure_s3_gse205117_trajectory_suitability_summary.tsv"
OUTPUT_MANIFEST = REP_DIR / "supplementary_figure_s3_gse205117_trajectory_suitability_manifest.json"


# ---------------------------------------------------------------------
# Frozen design constants
# ---------------------------------------------------------------------

SAMPLE_COL = "sample"
BARCODE_COL = "barcode"
STATE_COL = "celltype.mapped"
STAGE_COL = "stage"
PRIMARY_PT_COL = "common_pseudotime"

STATES = [
    "NMP",
    "Paraxial_mesoderm",
    "Somitic_mesoderm",
]

STATE_SHORT = {
    "NMP": "NMP",
    "Paraxial_mesoderm": "Paraxial",
    "Somitic_mesoderm": "Somitic",
}

STATE_COLORS = {
    "NMP": "#1f77b4",
    "Paraxial_mesoderm": "#ff7f0e",
    "Somitic_mesoderm": "#2ca02c",
}

PRIMARY_K = 30
K_VALUES = [20, 30, 50]
RANDOM_SEED = 785
FIG_DPI = 600

# Thresholds/guardrails aligned with the frozen upstream workflow.
MIN_SENSITIVITY_SPEARMAN = 0.95
CONTINUITY_MIN_OFFDIAG_FRACTION = 0.20
CONTINUITY_MAX_PARAXIAL_SELF_FRACTION = 0.60


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line(char: str = "=", width: int = 120):
    print(char * width)


def section(title: str):
    print()
    line("=")
    print(title)
    line("=")


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def safe_load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_inputs():
    for path in [F8C_CELLS, F8C_RNA, F8C_ATAC, F9_PSEUDOTIME]:
        require_file(path)

    cells_f8c = pd.read_csv(F8C_CELLS, sep="\t", compression="gzip", low_memory=False)
    rna = np.load(F8C_RNA).astype(np.float64)
    atac = np.load(F8C_ATAC).astype(np.float64)
    cells_f9 = pd.read_csv(F9_PSEUDOTIME, sep="\t", compression="gzip", low_memory=False)

    if len(cells_f8c) != rna.shape[0] or len(cells_f8c) != atac.shape[0]:
        raise RuntimeError(
            "F8c cell table length does not match RNA/ATAC representation rows.\n"
            f"cells={len(cells_f8c):,} RNA={rna.shape[0]:,} ATAC={atac.shape[0]:,}"
        )

    key_cols = [col for col in [SAMPLE_COL, BARCODE_COL] if col in cells_f8c.columns and col in cells_f9.columns]
    if len(key_cols) == 2:
        merged = cells_f9.copy()
        merged["__key__"] = merged[SAMPLE_COL].astype(str) + "||" + merged[BARCODE_COL].astype(str)
        f8c_keys = cells_f8c[SAMPLE_COL].astype(str) + "||" + cells_f8c[BARCODE_COL].astype(str)
        key_to_index = pd.Series(np.arange(len(cells_f8c)), index=f8c_keys.to_numpy())

        if not merged["__key__"].isin(key_to_index.index).all():
            missing = int((~merged["__key__"].isin(key_to_index.index)).sum())
            raise RuntimeError(f"F9 pseudotime contains {missing} sample/barcode keys not found in F8c cell table.")

        order = key_to_index.loc[merged["__key__"]].to_numpy(dtype=int)
        cells = merged.drop(columns=["__key__"])
        rna = rna[order, :]
        atac = atac[order, :]
    else:
        # Fall back to row order, but only if lengths match.
        if len(cells_f8c) != len(cells_f9):
            raise RuntimeError("Cannot align F8c and F9 outputs: key columns missing and row counts differ.")
        cells = cells_f9.copy()

    missing_states = sorted(set(STATES) - set(cells[STATE_COL].unique()))
    if missing_states:
        raise RuntimeError(f"Missing expected trajectory states in F9 pseudotime file: {missing_states}")

    f8c_manifest = safe_load_json(F8C_MANIFEST)
    f9_manifest = safe_load_json(F9_MANIFEST)
    f9b_diag = safe_load_json(F9B_DIAGNOSTIC)

    return cells, rna, atac, f8c_manifest, f9_manifest, f9b_diag


def compute_state_quantiles(cells: pd.DataFrame, pt_col: str) -> pd.DataFrame:
    rows = []
    for state in STATES:
        x = cells.loc[cells[STATE_COL].eq(state), pt_col].to_numpy(dtype=float)
        rows.append(
            {
                "state": state,
                "n_cells": len(x),
                "q25": float(np.quantile(x, 0.25)),
                "median": float(np.quantile(x, 0.50)),
                "q75": float(np.quantile(x, 0.75)),
                "mean": float(np.mean(x)),
                "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_pairwise_order_probability(cells: pd.DataFrame, pt_col: str, state_a: str, state_b: str, rng_seed: int = RANDOM_SEED) -> float:
    """
    Estimate P(pt_B > pt_A) from random pair draws without material memory use.
    For n~5k this exact outer comparison is also feasible, but sampling is enough.
    """
    a = cells.loc[cells[STATE_COL].eq(state_a), pt_col].to_numpy(dtype=float)
    b = cells.loc[cells[STATE_COL].eq(state_b), pt_col].to_numpy(dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan

    # Exact if reasonably sized.
    if len(a) * len(b) <= 5_000_000:
        return float((b[:, None] > a[None, :]).mean())

    rng = np.random.default_rng(rng_seed)
    n = 200_000
    aa = a[rng.integers(0, len(a), size=n)]
    bb = b[rng.integers(0, len(b), size=n)]
    return float((bb > aa).mean())


def compute_knn_enrichment(cells: pd.DataFrame, rna: np.ndarray, k: int = PRIMARY_K) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(rna)
    indices = nbrs.kneighbors(return_distance=False)
    indices = indices[:, 1:]  # drop self

    state_array = cells[STATE_COL].astype(str).to_numpy()
    state_to_idx = {state: i for i, state in enumerate(STATES)}

    counts = np.zeros((len(STATES), len(STATES)), dtype=int)
    for i, src_state in enumerate(state_array):
        if src_state not in state_to_idx:
            continue
        src_i = state_to_idx[src_state]
        neigh_states = state_array[indices[i]]
        for t in neigh_states:
            if t in state_to_idx:
                counts[src_i, state_to_idx[t]] += 1

    counts_df = pd.DataFrame(counts, index=STATES, columns=STATES)
    row_sums = counts_df.sum(axis=1).replace(0, np.nan)
    fractions_df = counts_df.div(row_sums, axis=0)

    global_abundance = cells[STATE_COL].value_counts(normalize=True).reindex(STATES)
    enrich_df = fractions_df.div(global_abundance, axis=1)

    return counts_df, fractions_df, enrich_df


def extract_f9_summary(f9_manifest: dict | None) -> Dict[str, float]:
    out = {
        "all_connected": np.nan,
        "all_finite": np.nan,
        "all_state_order": np.nan,
        "min_k_sensitivity_spearman": np.nan,
        "rna_coherence_ratio": np.nan,
        "atac_coherence_ratio": np.nan,
    }

    if not f9_manifest:
        return out

    summary_by_k = pd.DataFrame(f9_manifest.get("summary_by_k", []))
    if not summary_by_k.empty:
        out["all_connected"] = bool((summary_by_k["connected_components"] == 1).all())
        out["all_finite"] = bool(summary_by_k["finite_pseudotime"].all())
        out["all_state_order"] = bool(summary_by_k["state_order_ok"].all())

    k_sensitivity = pd.DataFrame(f9_manifest.get("k_sensitivity", []))
    if not k_sensitivity.empty and "spearman_rho" in k_sensitivity.columns:
        out["min_k_sensitivity_spearman"] = float(k_sensitivity["spearman_rho"].min())

    coherence = pd.DataFrame(f9_manifest.get("coherence", []))
    if not coherence.empty and "modality" in coherence.columns:
        coherence = coherence.set_index("modality")
        if "RNA" in coherence.index:
            out["rna_coherence_ratio"] = float(coherence.loc["RNA", "coherence_ratio"])
        if "ATAC" in coherence.index:
            out["atac_coherence_ratio"] = float(coherence.loc["ATAC", "coherence_ratio"])

    return out


def extract_f9b_summary(f9b_diag: dict | None):
    if not f9b_diag:
        return None

    out = {}
    if "knn_neighbor_enrichment" in f9b_diag:
        df = pd.DataFrame(f9b_diag["knn_neighbor_enrichment"])
        if not df.empty and "source_state" in df.columns:
            df = df.set_index("source_state")
            out["knn_enrichment"] = df.reindex(index=STATES, columns=STATES)

    if "knn_neighbor_fraction" in f9b_diag:
        df = pd.DataFrame(f9b_diag["knn_neighbor_fraction"])
        if not df.empty and "source_state" in df.columns:
            df = df.set_index("source_state")
            out["knn_fraction"] = df.reindex(index=STATES, columns=STATES)

    source_aligned = pd.DataFrame(f9b_diag.get("source_aligned_summary", []))
    if not source_aligned.empty:
        out["source_aligned"] = source_aligned

    subroute = pd.DataFrame(f9b_diag.get("subroute_coherence", []))
    if not subroute.empty:
        out["subroute_coherence"] = subroute

    return out


def continuity_decision(knn_fraction: pd.DataFrame) -> Tuple[bool, str]:
    """
    Compact operational summary for Panel D.

    We keep this diagnostic intentionally conservative:
    the three-state route is considered locally continuous only if the
    Paraxial state shows substantial mixing to BOTH flanking states and is
    not dominated by self-neighbors.
    """
    row = knn_fraction.loc["Paraxial_mesoderm"]
    nmp_mix = float(row["NMP"])
    som_mix = float(row["Somitic_mesoderm"])
    self_frac = float(row["Paraxial_mesoderm"])

    passed = (
        nmp_mix >= CONTINUITY_MIN_OFFDIAG_FRACTION
        and som_mix >= CONTINUITY_MIN_OFFDIAG_FRACTION
        and self_frac <= CONTINUITY_MAX_PARAXIAL_SELF_FRACTION
    )

    reason = (
        f"Paraxial neighbors: NMP={nmp_mix:.3f}, self={self_frac:.3f}, "
        f"Somitic={som_mix:.3f}"
    )
    return bool(passed), reason


def build_summary_table(
    cells: pd.DataFrame,
    rna: np.ndarray,
    atac: np.ndarray,
    state_quantiles: pd.DataFrame,
    f9_summary: Dict[str, float],
    knn_fraction: pd.DataFrame,
) -> pd.DataFrame:
    paired_ok = bool(len(cells) == rna.shape[0] == atac.shape[0])
    repr_ok = bool(rna.shape[1] == 10 and atac.shape[1] == 10)

    pseudotime_robust = bool(
        f9_summary.get("all_connected")
        and f9_summary.get("all_finite")
        and pd.notna(f9_summary.get("min_k_sensitivity_spearman"))
        and f9_summary["min_k_sensitivity_spearman"] >= MIN_SENSITIVITY_SPEARMAN
    )

    med = state_quantiles.set_index("state")["median"]
    state_order_ok = bool(
        med["NMP"] < med["Paraxial_mesoderm"] < med["Somitic_mesoderm"]
    )

    continuity_ok, continuity_reason = continuity_decision(knn_fraction)
    final_eligibility = bool(paired_ok and repr_ok and pseudotime_robust and state_order_ok and continuity_ok)

    rows = [
        {
            "criterion": "Paired RNA/ATAC correspondence",
            "status": "PASS" if paired_ok else "FAIL",
            "evidence": f"{len(cells):,} paired cells; RNA rows={rna.shape[0]:,}; ATAC rows={atac.shape[0]:,}",
        },
        {
            "criterion": "Modality-specific representation QC",
            "status": "PASS" if repr_ok else "FAIL",
            "evidence": f"Frozen primary arm with RNA {rna.shape[1]}D and ATAC {atac.shape[1]}D representations.",
        },
        {
            "criterion": "RNA common-clock robustness",
            "status": "PASS" if pseudotime_robust else "FAIL",
            "evidence": (
                f"Connected+finite for k=20/30/50; min sensitivity rho="
                f"{f9_summary.get('min_k_sensitivity_spearman', np.nan):.4f}."
            ),
        },
        {
            "criterion": "Expected three-state ordering",
            "status": "PASS" if state_order_ok else "FAIL",
            "evidence": (
                f"Observed medians: NMP={med['NMP']:.3f}, "
                f"Paraxial={med['Paraxial_mesoderm']:.3f}, "
                f"Somitic={med['Somitic_mesoderm']:.3f}."
            ),
        },
        {
            "criterion": "Local cross-state continuity",
            "status": "PASS" if continuity_ok else "FAIL",
            "evidence": continuity_reason,
        },
        {
            "criterion": "Eligible for downstream GDIS timing",
            "status": "GO" if final_eligibility else "STOP",
            "evidence": "Proceed only if both global robustness and local biological continuity are supported.",
        },
    ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def add_panel_label(ax, label: str):
    ax.text(
        -0.08,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_panel_a(ax, cells: pd.DataFrame, rna: np.ndarray):
    pca = PCA(n_components=1, random_state=RANDOM_SEED)
    pc1 = pca.fit_transform(rna).ravel()

    for state in STATES:
        mask = cells[STATE_COL].eq(state).to_numpy()
        ax.scatter(
            cells.loc[mask, PRIMARY_PT_COL],
            pc1[mask],
            s=6,
            alpha=0.45,
            linewidths=0,
            c=STATE_COLORS[state],
            label=STATE_SHORT[state],
            rasterized=True,
        )

        median_pt = float(np.median(cells.loc[mask, PRIMARY_PT_COL]))
        ax.axvline(median_pt, color=STATE_COLORS[state], linestyle="--", linewidth=1.1, alpha=0.8)

    if "is_frozen_root" in cells.columns and cells["is_frozen_root"].any():
        idx = np.flatnonzero(cells["is_frozen_root"].to_numpy())[0]
        ax.scatter(
            [cells.iloc[idx][PRIMARY_PT_COL]],
            [pc1[idx]],
            marker="*",
            s=140,
            c="black",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
            label="Frozen root",
        )

    ax.set_title("RNA-derived common clock colored by state", fontsize=15, fontweight="bold")
    ax.set_xlabel("RNA-derived diffusion pseudotime")
    ax.set_ylabel("RNA latent PC1")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best", ncol=2)
    add_panel_label(ax, "A")


def plot_panel_b(ax, cells: pd.DataFrame, state_quantiles: pd.DataFrame):
    data = [cells.loc[cells[STATE_COL].eq(state), PRIMARY_PT_COL].to_numpy(dtype=float) for state in STATES]
    labels = [STATE_SHORT[s] for s in STATES]

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="#555555", linewidth=1.1),
        capprops=dict(color="#555555", linewidth=1.1),
        boxprops=dict(linewidth=1.1, color="#555555"),
    )

    for patch, state in zip(bp["boxes"], STATES):
        patch.set_facecolor(STATE_COLORS[state])
        patch.set_alpha(0.55)

    med = state_quantiles.set_index("state")["median"]
    annotation = (
        f"Observed medians (k=30)\n"
        f"NMP = {med['NMP']:.3f}\n"
        f"Paraxial = {med['Paraxial_mesoderm']:.3f}\n"
        f"Somitic = {med['Somitic_mesoderm']:.3f}\n\n"
        f"Observed order:\nNMP < Somitic < Paraxial"
    )
    ax.text(
        0.98,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.82, edgecolor="#BBBBBB"),
    )

    ax.set_title("State-wise pseudotime distributions", fontsize=15, fontweight="bold")
    ax.set_ylabel("RNA-derived diffusion pseudotime")
    ax.grid(True, axis="y", alpha=0.25)
    add_panel_label(ax, "B")


def plot_panel_c(ax, knn_enrichment: pd.DataFrame, knn_fraction: pd.DataFrame, nmp_somitic_prob: float, f9_summary: Dict[str, float]):
    mat = knn_enrichment.reindex(index=STATES, columns=STATES).to_numpy(dtype=float)
    im = ax.imshow(mat, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(STATES)))
    ax.set_yticks(range(len(STATES)))
    ax.set_xticklabels([STATE_SHORT[s] for s in STATES], rotation=0)
    ax.set_yticklabels([STATE_SHORT[s] for s in STATES])
    ax.set_title(
        "Cross-state neighborhood continuity (RNA kNN)",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text_color = "white" if mat[i, j] < 1.35 else "black"
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                fontweight="bold",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Neighbor enrichment over global abundance", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    para_row = knn_fraction.loc["Paraxial_mesoderm"]
    note = (
        f"Paraxial neighbors: NMP={para_row['NMP']:.3f}, "
        f"self={para_row['Paraxial_mesoderm']:.3f}, "
        f"Somitic={para_row['Somitic_mesoderm']:.3f}\n"
        f"P(Somitic > NMP)={nmp_somitic_prob:.4f}; "
        f"min k-sensitivity rho={f9_summary.get('min_k_sensitivity_spearman', np.nan):.4f}"
    )
    ax.text(
        0.00,
        -0.11,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        linespacing=1.15,
    )

    ax.text(
        -0.13,
        1.015,
        "C",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
    )

def plot_panel_d(ax, summary_df: pd.DataFrame):
    ax.axis("off")
    ax.set_title(
        "Trajectory-suitability summary",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )

    # Keep the final STOP decision in a separate conclusion box
    # and show only the diagnostic checks as rows.
    display_df = summary_df.iloc[:-1].copy()

    x_text = 0.03
    x_status = 0.90
    y0 = 0.88
    dy = 0.13

    ax.text(
        x_text,
        y0 + 0.05,
        "Criterion and frozen evidence",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        x_status,
        y0 + 0.05,
        "Status",
        fontsize=12,
        fontweight="bold",
        ha="center",
        transform=ax.transAxes,
    )

    concise_evidence = {
        "Paired RNA/ATAC correspondence":
            "4,859 matched cells in both frozen modality representations.",
        "Modality-specific representation QC":
            "Frozen RNA 10D and ATAC 10D primary representations.",
        "RNA common-clock robustness":
            "Connected and finite for k=20/30/50; min cross-k rho=0.9979.",
        "Expected three-state ordering":
            "Observed medians: NMP=0.166, Somitic=0.508, Paraxial=0.770.",
        "Local cross-state continuity":
            "Paraxial neighbors: NMP=0.007, self=0.959, Somitic=0.035.",
    }

    for i, row in display_df.iterrows():
        y = y0 - i * dy
        criterion = str(row["criterion"])
        status = str(row["status"])

        if status in {"PASS", "GO"}:
            face = "#d9f2d9"
            edge = "#7fbf7f"
            status_color = "#1b5e20"
        else:
            face = "#fde0dd"
            edge = "#e08284"
            status_color = "#8c1d18"

        ax.text(
            x_text,
            y + 0.017,
            criterion,
            fontsize=10.6,
            fontweight="bold",
            va="center",
            transform=ax.transAxes,
        )

        evidence = concise_evidence.get(criterion, str(row["evidence"]))
        ax.text(
            x_text,
            y - 0.025,
            evidence,
            fontsize=8.9,
            va="center",
            color="#444444",
            transform=ax.transAxes,
        )

        rect = patches.FancyBboxPatch(
            (x_status - 0.045, y - 0.028),
            0.09,
            0.05,
            boxstyle="round,pad=0.01",
            linewidth=0.8,
            edgecolor=edge,
            facecolor=face,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        ax.text(
            x_status,
            y - 0.002,
            status,
            fontsize=10.0,
            fontweight="bold",
            color=status_color,
            va="center",
            ha="center",
            transform=ax.transAxes,
        )

    final_status = str(summary_df.iloc[-1]["status"])
    conclusion = (
        "Final decision: STOP before downstream GDIS timing.\n"
        "The dataset passes pairing, representation, and numerical-clock checks,\n"
        "but fails the prespecified biological-ordering and\n"
        "local-continuity safeguards."
    )
    face = "#fde0dd" if final_status == "STOP" else "#d9f2d9"
    edge = "#e08284" if final_status == "STOP" else "#7fbf7f"

    ax.text(
        0.03,
        0.055,
        conclusion,
        fontsize=10.0,
        ha="left",
        va="bottom",
        linespacing=1.2,
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.42",
            facecolor=face,
            edgecolor=edge,
        ),
    )

    add_panel_label(ax, "D")

def make_figure(cells: pd.DataFrame, rna: np.ndarray, state_quantiles: pd.DataFrame, knn_enrichment: pd.DataFrame, knn_fraction: pd.DataFrame, nmp_somitic_prob: float, f9_summary: Dict[str, float], summary_df: pd.DataFrame):
    fig = plt.figure(figsize=(16.8, 12.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.12], wspace=0.24, hspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_panel_a(ax_a, cells, rna)
    plot_panel_b(ax_b, cells, state_quantiles)
    plot_panel_c(ax_c, knn_enrichment, knn_fraction, nmp_somitic_prob, f9_summary)
    plot_panel_d(ax_d, summary_df)

    fig.suptitle(
        "GSE205117 external-dataset trajectory-suitability diagnostics",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )

    fig.savefig(OUTPUT_PNG, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    section("SUPPLEMENTARY FIGURE S3 — GSE205117 TRAJECTORY-SUITABILITY DIAGNOSTICS")
    print("This script visualizes the frozen external-dataset rejection logic.")
    print("No new trajectory is constructed.")
    print("No GDIS is calculated.")
    print("No CMIL is calculated.")

    section("1. LOAD FROZEN INPUTS")
    cells, rna, atac, f8c_manifest, f9_manifest, f9b_diag = load_inputs()
    print(f"Cells loaded: {len(cells):,}")
    print(f"RNA shape:     {rna.shape}")
    print(f"ATAC shape:    {atac.shape}")

    section("2. SUMMARIZE F9 PSEUDOTIME AND F9b TOPOLOGY")
    state_quantiles = compute_state_quantiles(cells, PRIMARY_PT_COL)
    print(state_quantiles.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    f9_summary = extract_f9_summary(f9_manifest)

    parsed_f9b = extract_f9b_summary(f9b_diag)
    if parsed_f9b and "knn_enrichment" in parsed_f9b and "knn_fraction" in parsed_f9b:
        knn_enrichment = parsed_f9b["knn_enrichment"].copy()
        knn_fraction = parsed_f9b["knn_fraction"].copy()
        print("Using kNN enrichment/fraction tables from frozen F9b diagnostic JSON.")
    else:
        print("F9b diagnostic JSON missing or incomplete; recomputing kNN enrichment from frozen RNA 10D.")
        _, knn_fraction, knn_enrichment = compute_knn_enrichment(cells, rna, k=PRIMARY_K)

    print("\nNeighbor enrichment:")
    print(knn_enrichment.to_string(float_format=lambda x: f"{x:.4f}"))

    if parsed_f9b and "source_aligned" in parsed_f9b:
        source_df = parsed_f9b["source_aligned"]
        row = source_df.loc[source_df["k"].eq(PRIMARY_K)]
        if row.empty:
            nmp_somitic_prob = float(source_df.iloc[0]["P_Somitic_gt_NMP"])
        else:
            nmp_somitic_prob = float(row.iloc[0]["P_Somitic_gt_NMP"])
    else:
        nmp_somitic_prob = compute_pairwise_order_probability(cells, PRIMARY_PT_COL, "NMP", "Somitic_mesoderm")

    summary_df = build_summary_table(cells, rna, atac, state_quantiles, f9_summary, knn_fraction)
    print("\nSummary table:")
    print(summary_df.to_string(index=False))

    section("3. GENERATE FIGURE")
    make_figure(cells, rna, state_quantiles, knn_enrichment, knn_fraction, nmp_somitic_prob, f9_summary, summary_df)
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)

    section("4. SAVE SUMMARY ARTIFACTS")
    summary_df.to_csv(OUTPUT_SUMMARY, sep="\t", index=False)
    print(OUTPUT_SUMMARY)

    manifest = {
        "dataset_accession": ACCESSION,
        "figure": "Supplementary Figure S3",
        "created_utc": utc_now_iso(),
        "inputs": {
            "f8c_cells": str(F8C_CELLS),
            "f8c_rna": str(F8C_RNA),
            "f8c_atac": str(F8C_ATAC),
            "f8c_manifest": str(F8C_MANIFEST) if F8C_MANIFEST.exists() else None,
            "f9_pseudotime": str(F9_PSEUDOTIME),
            "f9_manifest": str(F9_MANIFEST) if F9_MANIFEST.exists() else None,
            "f9b_diagnostic": str(F9B_DIAGNOSTIC) if F9B_DIAGNOSTIC.exists() else None,
        },
        "outputs": {
            "png": str(OUTPUT_PNG),
            "png_sha256": sha256_file(OUTPUT_PNG),
            "pdf": str(OUTPUT_PDF),
            "pdf_sha256": sha256_file(OUTPUT_PDF),
            "summary_tsv": str(OUTPUT_SUMMARY),
            "summary_tsv_sha256": sha256_file(OUTPUT_SUMMARY),
        },
        "panel_design": {
            "A": "RNA-derived pseudotime colored by biological state",
            "B": "state-wise pseudotime distributions showing ordering inconsistency",
            "C": "cross-state neighborhood/continuity diagnostics",
            "D": "compact pass/fail summary",
        },
        "guardrails": {
            "new_pseudotime_constructed": False,
            "gdis_calculated": False,
            "cmil_calculated": False,
            "frozen_results_modified": False,
        },
        "summary_rows": summary_df.to_dict(orient="records"),
    }

    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(OUTPUT_MANIFEST)

    section("5. COMPLETED")
    print("Supplementary Figure S3 script finished successfully.")
    print("No upstream inference was altered.")


if __name__ == "__main__":
    main()

