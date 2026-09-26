#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
40_phase_f24_discovery_publication_figure.py
===========================================

Publication-only Figure 2 generator for the GDIS-Multiome manuscript.

IMPORTANT SCIENTIFIC SCOPE
--------------------------
This script does NOT change any analytical decision. It reproduces the frozen
GSE275562 discovery analysis using the accepted F2-F4 settings and generates a
four-panel manuscript figure:

A. RNA-derived endocrine DPT by biological state.
B. RNA and ATAC sustained-GDIS (lambda_t = 0) profiles across all three frozen
   sliding-window configurations.
C. Sustained-instability peak topology across resolutions, with robust families
   and the prospectively frozen RNA/ATAC event pair marked.
D. Paired pseudotime-stratified bootstrap distribution of CMIL for the frozen
   discovery event pair.

Why Panel C is NOT called "transition-energy topology"
-----------------------------------------------------
The discovery event families used in F3b/F3c/F4 were derived from the
lambda_t=0 sustained-GDIS profiles. The frozen RNA ~0.288 and ATAC ~0.228
families and the F4 bootstrap therefore belong to the sustained-instability
analysis. Calling these transition-energy families would be scientifically
incorrect.

The script imports helper functions directly from the canonical discovery
scripts so that trajectory construction, GDIS calculation, event tracking,
and bootstrap behavior remain identical to the frozen analysis.

Run from the canonical project root
-----------------------------------
    python 40_phase_f24_discovery_publication_figure.py

Expected input
--------------
    data/GSE275562_mudata_with_annotation_all.h5mu

Required canonical scripts in the same directory
-------------------------------------------------
    06_phase_f3_gdis_rna_atac.py
    09_phase_f4_event_specific_cmil_bootstrap.py

Outputs
-------
    figures/Figure_2_GSE275562_discovery.png       (600 dpi)
    figures/Figure_2_GSE275562_discovery.pdf       (vector)
    figures/Figure_2_GSE275562_bootstrap.tsv.gz    (frozen plotted bootstrap)
    figures/Figure_2_GSE275562_profiles.tsv.gz     (plotted profile values)
    figures/Figure_2_GSE275562_peak_topology.tsv   (plotted peak topology)

No parameter optimization, event reselection, or new inferential test is
performed here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mudata as mu


# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
DATA_FILE = PROJECT_ROOT / "data" / "GSE275562_mudata_with_annotation_all.h5mu"
F3_SCRIPT = SCRIPT_DIR / "06_phase_f3_gdis_rna_atac.py"
F4_SCRIPT = SCRIPT_DIR / "09_phase_f4_event_specific_cmil_bootstrap.py"
OUTDIR = PROJECT_ROOT / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUTDIR / "Figure_2_GSE275562_discovery.png"
PDF_OUT = OUTDIR / "Figure_2_GSE275562_discovery.pdf"
BOOT_OUT = OUTDIR / "Figure_2_GSE275562_bootstrap.tsv.gz"
PROFILE_OUT = OUTDIR / "Figure_2_GSE275562_profiles.tsv.gz"
PEAK_OUT = OUTDIR / "Figure_2_GSE275562_peak_topology.tsv"


# -----------------------------------------------------------------------------
# FROZEN DISPLAY / REPRODUCIBILITY SETTINGS
# -----------------------------------------------------------------------------

FIGSIZE = (15.5, 11.0)
DPI = 600
JITTER_SEED = 20260920
POINT_SIZE = 4.0
POINT_ALPHA = 0.32

# Publication typography tuned for two-column journal reduction.
SUPTITLE_FONTSIZE = 22
PANEL_LABEL_FONTSIZE = 22
PANEL_TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12
ANNOTATION_FONTSIZE = 12
STATSBOX_FONTSIZE = 12

STATE_LABELS = [
    "Ngn3 low",
    "Ngn3 high",
    "Fev+",
    "Fev+ Beta",
    "Beta",
]

WINDOW_LABELS = {
    "300_75": "300/75",
    "400_100": "400/100",
    "500_125": "500/125",
}

# Different line styles identify the window designs without relying on color.
WINDOW_LINESTYLES = {
    "300_75": ":",
    "400_100": "-",
    "500_125": "--",
}


# -----------------------------------------------------------------------------
# MODULE LOADING
# -----------------------------------------------------------------------------

def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required canonical script not found: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module specification for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# LOAD FROZEN DISCOVERY DATA
# -----------------------------------------------------------------------------

def load_discovery_inputs(f3):
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"GSE275562 input not found: {DATA_FILE}\n"
            "Place/download the canonical H5MU file under ./data and rerun."
        )

    try:
        mdata = mu.read_h5mu(DATA_FILE, backed="r")
    except TypeError:
        mdata = mu.read_h5mu(DATA_FILE)

    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]
    obs_full = rna.obs.copy()

    beta_mask = (
        obs_full["cell_type"].astype(str).isin(f3.BETA_LINEAGE)
    ).to_numpy()

    obs = obs_full.loc[beta_mask].copy().reset_index(drop=True)
    obs.index = pd.Index([f"beta_{i}" for i in range(len(obs))])

    X_rna = f3.as_numpy(rna.obsm[f3.RNA_REPRESENTATION])[
        beta_mask, : f3.N_LATENT_DIMS
    ]
    X_atac = f3.as_numpy(atac.obsm[f3.ATAC_REPRESENTATION])[
        beta_mask, : f3.N_LATENT_DIMS
    ]

    if X_rna.shape != X_atac.shape:
        raise RuntimeError("RNA and ATAC latent matrices are not paired.")

    pseudotime = f3.reconstruct_common_pseudotime(X_rna=X_rna, obs=obs)
    if np.isfinite(pseudotime).mean() < 0.99:
        raise RuntimeError("Frozen RNA-derived pseudotime was not reproduced.")

    return obs, X_rna, X_atac, pseudotime


# -----------------------------------------------------------------------------
# REPRODUCE FROZEN GDIS PROFILES
# -----------------------------------------------------------------------------

def compute_profiles(f3, X_rna, X_atac, pseudotime, obs):
    X_rna_z = f3.zscore_columns(X_rna)
    X_atac_z = f3.zscore_columns(X_atac)

    critical_value, _, _ = f3.biological_critical_landmark(pseudotime, obs)

    profiles = {}
    rows = []

    for design_name, window_size, step_size in f3.WINDOW_DESIGNS:
        rna_traj, parameters, _ = f3.create_windows(
            X_rna_z, pseudotime, window_size, step_size
        )
        atac_traj, atac_parameters, _ = f3.create_windows(
            X_atac_z, pseudotime, window_size, step_size
        )

        if not np.allclose(parameters, atac_parameters, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"RNA/ATAC window centers differ for {design_name}.")

        rna_result, rna_primary, _ = f3.run_gdis(
            rna_traj, parameters, critical_value
        )
        atac_result, atac_primary, _ = f3.run_gdis(
            atac_traj, parameters, critical_value
        )

        profiles[("RNA", design_name)] = {
            "parameter": parameters.copy(),
            "gdis": rna_primary["gdis"].to_numpy(dtype=float),
            "result": rna_result,
        }
        profiles[("ATAC", design_name)] = {
            "parameter": parameters.copy(),
            "gdis": atac_primary["gdis"].to_numpy(dtype=float),
            "result": atac_result,
        }

        for modality in ("RNA", "ATAC"):
            p = profiles[(modality, design_name)]["parameter"]
            g = profiles[(modality, design_name)]["gdis"]
            rows.extend(
                {
                    "modality": modality,
                    "window_design": design_name,
                    "window_center_pseudotime": float(x),
                    "sustained_gdis_lambda0": float(y),
                }
                for x, y in zip(p, g)
            )

    profile_table = pd.DataFrame(rows)
    return X_rna_z, X_atac_z, critical_value, profiles, profile_table


# -----------------------------------------------------------------------------
# REPRODUCE FROZEN SUSTAINED-INSTABILITY PEAK TOPOLOGY
# -----------------------------------------------------------------------------

def compute_peak_topology(f4, profiles):
    peak_frames = []

    for modality in ("RNA", "ATAC"):
        for design_name, _, _ in f4.WINDOW_DESIGNS:
            prof = profiles[(modality, design_name)]
            peaks = f4.detect_local_peaks(
                prof["parameter"],
                prof["gdis"],
                modality,
                design_name,
            )
            if not peaks.empty:
                peak_frames.append(peaks)

    peaks = pd.concat(peak_frames, ignore_index=True)

    family_tables = {}
    member_tables = {}

    for modality in ("RNA", "ATAC"):
        modality_peaks = peaks.loc[peaks["modality"] == modality].copy()
        clusters = f4.cluster_peak_families(modality_peaks)
        families, members = f4.summarize_families(modality, clusters)
        family_tables[modality] = families
        member_tables[modality] = members

    families = pd.concat(family_tables.values(), ignore_index=True)
    members = pd.concat(member_tables.values(), ignore_index=True)

    return peaks, families, members


# -----------------------------------------------------------------------------
# REPRODUCE FROZEN EVENT PAIR AND BOOTSTRAP
# -----------------------------------------------------------------------------

def compute_bootstrap(f4, X_rna_z, X_atac_z, pseudotime, obs, critical_value):
    medians = f4.get_state_medians(pseudotime, obs)

    rna_family, atac_family, rna_members, atac_members = f4.derive_frozen_event_pair(
        X_rna_z,
        X_atac_z,
        pseudotime,
        medians,
    )

    frozen_rna_center = float(rna_family["median_peak_pt"])
    frozen_atac_center = float(atac_family["median_peak_pt"])

    rna_event_members = rna_members.loc[
        rna_members["family_id"] == int(rna_family["family_id"])
    ][["window_design", "peak_pt"]].rename(columns={"peak_pt": "t_rna"})

    atac_event_members = atac_members.loc[
        atac_members["family_id"] == int(atac_family["family_id"])
    ][["window_design", "peak_pt"]].rename(columns={"peak_pt": "t_atac"})

    observed = rna_event_members.merge(atac_event_members, on="window_design", how="inner")
    observed["cmil"] = observed["t_rna"] - observed["t_atac"]

    primary_row = observed.loc[
        observed["window_design"] == f4.PRIMARY_WINDOW_DESIGN
    ]
    if len(primary_row) != 1:
        raise RuntimeError("Primary frozen discovery event pair was not reproduced.")
    observed_primary_cmil = float(primary_row.iloc[0]["cmil"])

    finite_indices = np.flatnonzero(np.isfinite(pseudotime))
    ordered_indices = finite_indices[
        np.argsort(pseudotime[finite_indices], kind="mergesort")
    ]

    rng = np.random.default_rng(f4.RANDOM_SEED)
    records = []

    for b in range(1, f4.N_BOOTSTRAPS + 1):
        result = f4.bootstrap_one_replicate(
            X_rna_z=X_rna_z,
            X_atac_z=X_atac_z,
            pseudotime=pseudotime,
            ordered_indices=ordered_indices,
            frozen_rna_center=frozen_rna_center,
            frozen_atac_center=frozen_atac_center,
            critical_value=critical_value,
            rng=rng,
        )
        result["bootstrap_id"] = b
        records.append(result)

        if b % 50 == 0 or b == f4.N_BOOTSTRAPS:
            n_detected = int(sum(bool(x["detected"]) for x in records))
            print(
                f"Bootstrap {b:>3}/{f4.N_BOOTSTRAPS}: "
                f"paired event detected in {n_detected}/{b} replicates"
            )

    bootstrap = pd.DataFrame(records)
    detected = bootstrap.loc[bootstrap["detected"]].copy()

    if len(detected) < 20:
        raise RuntimeError("Too few successful bootstrap detections.")

    cmil = detected["cmil"].to_numpy(dtype=float)
    ci_low, ci_high = f4.percentile_interval(cmil)

    summary = {
        "frozen_rna_center": frozen_rna_center,
        "frozen_atac_center": frozen_atac_center,
        "observed_primary_cmil": observed_primary_cmil,
        "n_detected": int(len(detected)),
        "detection_rate": float(len(detected) / f4.N_BOOTSTRAPS),
        "median": float(np.median(cmil)),
        "mean": float(np.mean(cmil)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_gt_zero": float(np.mean(cmil > 0.0)),
    }

    return bootstrap, detected, observed, summary


# -----------------------------------------------------------------------------
# FIGURE
# -----------------------------------------------------------------------------

def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def make_figure(obs, pseudotime, profiles, peaks, families, detected, observed, boot_summary):
    plt.rcParams.update({
        "axes.titlesize": PANEL_TITLE_FONTSIZE,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "xtick.labelsize": TICK_LABEL_FONTSIZE,
        "ytick.labelsize": TICK_LABEL_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
    })
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.05, hspace=0.03, wspace=0.03)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # ------------------------------------------------------------------
    # A. Common RNA-derived DPT by biological state
    # ------------------------------------------------------------------
    rng = np.random.default_rng(JITTER_SEED)

    for y, state in enumerate(STATE_LABELS):
        mask = (obs["cell_type"].astype(str) == state).to_numpy()
        x = pseudotime[mask]
        x = x[np.isfinite(x)]
        jitter = rng.normal(loc=0.0, scale=0.075, size=len(x))
        ax_a.scatter(
            x,
            np.full(len(x), y, dtype=float) + jitter,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            linewidths=0,
            label=state,
        )
        ax_a.plot(
            [np.median(x)],
            [y],
            marker="D",
            markersize=6,
            linestyle="None",
        )

    ax_a.set_yticks(range(len(STATE_LABELS)))
    ax_a.set_yticklabels(STATE_LABELS)
    ax_a.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax_a.set_xlabel("RNA-derived diffusion pseudotime", fontsize=AXIS_LABEL_FONTSIZE)
    ax_a.set_ylabel("Endocrine state", fontsize=AXIS_LABEL_FONTSIZE)
    ax_a.set_title("Common developmental ordering", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    ax_a.set_xlim(0, 1)
    ax_a.grid(axis="x", alpha=0.2)
    add_panel_label(ax_a, "A")

    # ------------------------------------------------------------------
    # B. Sustained-GDIS profiles across window designs
    # ------------------------------------------------------------------
    for modality in ("RNA", "ATAC"):
        for design_name, _, _ in [("300_75", 300, 75), ("400_100", 400, 100), ("500_125", 500, 125)]:
            prof = profiles[(modality, design_name)]
            linewidth = 2.4 if design_name == "400_100" else 1.5
            ax_b.plot(
                prof["parameter"],
                prof["gdis"],
                linestyle=WINDOW_LINESTYLES[design_name],
                linewidth=linewidth,
                alpha=0.95 if design_name == "400_100" else 0.70,
                label=f"{modality} {WINDOW_LABELS[design_name]}",
            )

    ax_b.set_xlabel("Common pseudotime", fontsize=AXIS_LABEL_FONTSIZE)
    ax_b.set_ylabel("Sustained GDIS (λₜ = 0)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_b.set_title("Modality-specific sustained-instability profiles", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    ax_b.set_xlim(0, 0.90)
    ax_b.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax_b.grid(alpha=0.2)
    ax_b.legend(frameon=False, fontsize=LEGEND_FONTSIZE, ncol=2)
    add_panel_label(ax_b, "B")

    # ------------------------------------------------------------------
    # C. Peak topology and frozen pair
    # ------------------------------------------------------------------
    y_positions = {
        ("RNA", "300_75"): 5,
        ("RNA", "400_100"): 4,
        ("RNA", "500_125"): 3,
        ("ATAC", "300_75"): 2,
        ("ATAC", "400_100"): 1,
        ("ATAC", "500_125"): 0,
    }

    for _, row in peaks.iterrows():
        y = y_positions[(str(row["modality"]), str(row["window_design"]))]
        ax_c.scatter(
            float(row["peak_pt"]),
            y,
            s=35 + 90 * float(row["relative_prominence"]),
            alpha=0.80,
        )

    # Robust-family centers.
    robust = families.loc[families["robust"]].copy()
    for _, row in robust.iterrows():
        center = float(row["median_peak_pt"])
        if str(row["modality"]) == "RNA":
            ymin, ymax = 0.52, 0.97
        else:
            ymin, ymax = 0.03, 0.48
        ax_c.axvline(center, ymin=ymin, ymax=ymax, linestyle="--", linewidth=1.1, alpha=0.65)

    # Frozen pair emphasized with arrows/labels using the actual reproduced centers.
    rna_center = boot_summary["frozen_rna_center"]
    atac_center = boot_summary["frozen_atac_center"]
    ax_c.axvline(rna_center, linewidth=2.0, alpha=0.9)
    ax_c.axvline(atac_center, linewidth=2.0, alpha=0.9)
    ax_c.annotate(
        f"Frozen RNA family\n{rna_center:.3f}",
        xy=(rna_center, 4.18),
        xytext=(rna_center + 0.085, 4.95),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=ANNOTATION_FONTSIZE,
        ha="left",
        va="center",
    )
    ax_c.annotate(
        f"Frozen ATAC family\n{atac_center:.3f}",
        xy=(atac_center, 1.02),
        xytext=(atac_center + 0.085, 0.35),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=ANNOTATION_FONTSIZE,
        ha="left",
        va="center",
    )

    ax_c.set_yticks([5, 4, 3, 2, 1, 0])
    ax_c.set_yticklabels([
        "RNA 300/75",
        "RNA 400/100",
        "RNA 500/125",
        "ATAC 300/75",
        "ATAC 400/100",
        "ATAC 500/125",
    ])
    ax_c.set_xlabel("Common pseudotime", fontsize=AXIS_LABEL_FONTSIZE)
    ax_c.set_title("Sustained-instability peak topology", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    ax_c.set_xlim(0, 0.90)
    ax_c.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax_c.grid(axis="x", alpha=0.2)
    add_panel_label(ax_c, "C")

    # ------------------------------------------------------------------
    # D. Frozen-pair bootstrap CMIL distribution
    # ------------------------------------------------------------------
    cmil = detected["cmil"].to_numpy(dtype=float)
    ax_d.hist(cmil, bins=28, alpha=0.80, edgecolor="white")
    ax_d.axvline(0.0, linestyle="--", linewidth=1.8, label="No lead (CMIL = 0)")
    ax_d.axvline(
        boot_summary["observed_primary_cmil"],
        linestyle="-",
        linewidth=2.0,
        label=f"Observed 400/100 = {boot_summary['observed_primary_cmil']:.3f}",
    )

    text = (
        f"Paired detection = {boot_summary['detection_rate']:.3f}\n"
        f"Median = {boot_summary['median']:.3f}\n"
        f"95% CI = [{boot_summary['ci_low']:.3f}, {boot_summary['ci_high']:.3f}]\n"
        f"P(CMIL > 0) = {boot_summary['p_gt_zero']:.3f}"
    )
    ax_d.text(
        0.98,
        0.96,
        text,
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=STATSBOX_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )

    ax_d.set_xlabel(r"CMIL = $t_{\mathrm{RNA}} - t_{\mathrm{ATAC}}$ (pseudotime)", fontsize=AXIS_LABEL_FONTSIZE)
    ax_d.set_ylabel("Bootstrap replicates", fontsize=AXIS_LABEL_FONTSIZE)
    ax_d.set_title("Frozen discovery event-pair uncertainty", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    ax_d.grid(axis="y", alpha=0.2)
    ax_d.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax_d.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left")
    add_panel_label(ax_d, "D")

    fig.savefig(PNG_OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("Loading canonical discovery helpers...")
    f3 = load_module("gdis_f3", F3_SCRIPT)
    f4 = load_module("gdis_f4", F4_SCRIPT)

    print("Loading GSE275562 and reconstructing frozen RNA-only pseudotime...")
    obs, X_rna, X_atac, pseudotime = load_discovery_inputs(f3)

    print("Reproducing frozen sustained-GDIS profiles...")
    X_rna_z, X_atac_z, critical_value, profiles, profile_table = compute_profiles(
        f3, X_rna, X_atac, pseudotime, obs
    )

    print("Reproducing sustained-instability peak topology...")
    peaks, families, members = compute_peak_topology(f4, profiles)

    print("Reproducing frozen event pair and 500-replicate paired bootstrap...")
    bootstrap, detected, observed, boot_summary = compute_bootstrap(
        f4,
        X_rna_z,
        X_atac_z,
        pseudotime,
        obs,
        critical_value,
    )

    print("Writing plotted source tables...")
    profile_table.to_csv(PROFILE_OUT, sep="\t", index=False, compression="gzip")
    bootstrap.to_csv(BOOT_OUT, sep="\t", index=False, compression="gzip")

    peaks_out = peaks.merge(
        members[["modality", "window_design", "peak_pt", "family_id"]],
        on=["modality", "window_design", "peak_pt"],
        how="left",
    )
    peaks_out.to_csv(PEAK_OUT, sep="\t", index=False)

    print("Generating publication figure...")
    make_figure(
        obs,
        pseudotime,
        profiles,
        peaks,
        families,
        detected,
        observed,
        boot_summary,
    )

    print("\nFigure 2 generated successfully.")
    print(f"PNG (600 dpi): {PNG_OUT}")
    print(f"PDF (vector):  {PDF_OUT}")
    print(f"Profiles:      {PROFILE_OUT}")
    print(f"Bootstrap:     {BOOT_OUT}")
    print(f"Peak topology: {PEAK_OUT}")
    print("\nFrozen discovery bootstrap summary reproduced for the figure:")
    for key, value in boot_summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

