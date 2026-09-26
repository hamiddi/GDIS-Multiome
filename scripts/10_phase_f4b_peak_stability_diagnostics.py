#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
10_phase_f4b_peak_stability_diagnostics.py
===========================================

Project:
Cross-Modal Dynamical Instability Reveals Chromatin Priming
before Transcriptional Cell-Fate Transitions

Phase F4b:
Diagnose why the frozen peak-based CMIL estimator failed the Phase F4
bootstrap stability criteria.

IMPORTANT
---------
Phase F4 is FROZEN as a negative/inconclusive primary result:

Observed event-specific CMIL was positive in all three prespecified
window designs, but:
    bootstrap paired-event detection rate = 0.738
    P(CMIL > 0) = 0.772
    95% percentile interval crossed zero

This script does NOT:
    - change the frozen event pair;
    - relax the prominence threshold;
    - redefine CMIL;
    - claim chromatin priming;
    - perform a null/permutation significance test;
    - write files.

It only diagnoses peak-estimator behavior.

FROZEN EVENT
------------
RNA:
    Ngn3 high -> Fev+ robust family

ATAC:
    proximal upstream robust family

PRIMARY REPRESENTATIONS
-----------------------
RNA  = X_scVI, 10D
ATAC = X_poissonvi, 10D

PRIMARY GDIS WINDOW
-------------------
400 cells / 100-cell step

BOOTSTRAP
---------
Paired pseudotime-stratified resampling using the same design as Phase F4,
but with detailed event-detection diagnostics.

The common RNA-derived pseudotime remains fixed.

DIAGNOSTICS
-----------
For each bootstrap replicate and modality, record:

1. Was any local peak detected?
2. Was any local peak inside the frozen +/-0.08 tracking interval?
3. How many eligible peaks were inside that interval?
4. Location of the closest eligible peak.
5. Relative prominence of the closest eligible peak.
6. Strongest peak location even if outside the frozen family.
7. Distance of the strongest peak from the frozen family center.
8. Local event-region score range.
9. Local event-region roughness (mean absolute first difference).
10. Local event-region plateau width above 90% of its local maximum.

This distinguishes:
    - event disappearance;
    - family escape / peak hopping;
    - multiple competing local peaks;
    - broad plateaus;
    - unstable RNA versus ATAC localization.

Run:
    python 10_phase_f4b_peak_stability_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import mudata as mu
import anndata as ad
import scanpy as sc

from scipy.signal import find_peaks, peak_prominences

try:
    from gdis import GDIS, transition_weight_sensitivity
except ImportError:
    raise SystemExit(
        "\nERROR: pyGDIS is not installed.\n"
        "Install with:\n"
        "    python -m pip install pygdis\n"
    )


# =====================================================================
# SETTINGS
# =====================================================================

DATA_FILE = Path("data/GSE275562_mudata_with_annotation_all.h5mu")

BETA_LINEAGE = [
    "Ngn3 low",
    "Ngn3 high",
    "Fev+",
    "Fev+ Beta",
    "Beta",
]

RNA_REPRESENTATION = "X_scVI"
ATAC_REPRESENTATION = "X_poissonvi"
N_LATENT_DIMS = 10

DPT_NEIGHBORS = 30
N_DIFFUSION_COMPONENTS = 10
RANDOM_SEED = 20260920

WINDOW_DESIGNS = [
    ("300_75", 300, 75),
    ("400_100", 400, 100),
    ("500_125", 500, 125),
]

PRIMARY_WINDOW_SIZE = 400
PRIMARY_STEP_SIZE = 100

PRIMARY_TRANSITION_WEIGHT = 0.0
REFERENCE_TRANSITION_WEIGHT = 0.18

MIN_RELATIVE_PROMINENCE = 0.05
MIN_PEAK_DISTANCE_WINDOWS = 3

FAMILY_MATCH_TOLERANCE = 0.08
MAX_ROBUST_FAMILY_SPREAD = 0.10
REQUIRED_WINDOW_DESIGNS = 3

# Diagnostic bootstrap. This is intentionally smaller than the inferential
# Phase F4 bootstrap because its purpose is failure-mode characterization.
N_BOOTSTRAPS = 200
BOOTSTRAP_STRATUM_SIZE = 100
PROGRESS_EVERY = 20

PLATEAU_FRACTION = 0.90


# =====================================================================
# DISPLAY HELPERS
# =====================================================================

def line(char="=", width=114):
    print(char * width)


def section(title):
    print()
    line("=")
    print(title)
    line("=")


def print_df(df, digits=6):
    if df.empty:
        print("(empty)")
        return

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 340,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string(index=True))


# =====================================================================
# BASIC HELPERS
# =====================================================================

def as_numpy(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def zscore_columns(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return (X - mean) / std


# =====================================================================
# COMMON RNA-ONLY PSEUDOTIME
# =====================================================================

def choose_root_medoid(X, obs):
    mask = (
        (obs["stage"].astype(str) == "E14.5")
        & (obs["cell_type"].astype(str) == "Ngn3 low")
    ).to_numpy()

    indices = np.flatnonzero(mask)

    if len(indices) == 0:
        raise RuntimeError("No E14.5 Ngn3-low root candidates.")

    candidates = X[indices]
    centroid = candidates.mean(axis=0)
    distances = np.linalg.norm(candidates - centroid, axis=1)

    return int(indices[np.argmin(distances)])


def reconstruct_common_pseudotime(X_rna, obs):
    X = zscore_columns(X_rna)

    adata = ad.AnnData(
        X=np.zeros((X.shape[0], 1), dtype=np.float32)
    )

    obs_copy = obs.copy()
    obs_copy.index = obs_copy.index.astype(str)

    adata.obs = obs_copy
    adata.obsm["X_common_rna"] = X.astype(np.float32)

    root = choose_root_medoid(X, adata.obs)

    sc.pp.neighbors(
        adata,
        n_neighbors=DPT_NEIGHBORS,
        use_rep="X_common_rna",
        metric="euclidean",
        random_state=RANDOM_SEED,
    )

    sc.tl.diffmap(
        adata,
        n_comps=max(N_DIFFUSION_COMPONENTS + 1, 15),
        random_state=RANDOM_SEED,
    )

    adata.uns["iroot"] = root

    sc.tl.dpt(
        adata,
        n_dcs=N_DIFFUSION_COMPONENTS,
    )

    pt = adata.obs["dpt_pseudotime"].to_numpy(dtype=float)
    pt[~np.isfinite(pt)] = np.nan

    return pt


# =====================================================================
# WINDOWS / GDIS
# =====================================================================

def create_windows(X, pseudotime, window_size, step_size):
    valid = np.isfinite(pseudotime)

    X = X[valid]
    pt = pseudotime[valid]

    order = np.argsort(pt, kind="mergesort")
    X = X[order]
    pt = pt[order]

    trajectories = []
    parameters = []

    start = 0

    while start + window_size <= len(pt):
        stop = start + window_size

        trajectories.append(
            X[start:stop].copy()
        )

        parameters.append(
            float(np.mean(pt[start:stop]))
        )

        start += step_size

    return trajectories, np.asarray(parameters, dtype=float)


def sustained_gdis_curve(
    trajectories,
    parameters,
    critical_value,
):
    model = GDIS(
        transition_weight=REFERENCE_TRANSITION_WEIGHT
    )

    result = model.fit_transform(
        trajectories,
        parameters,
        critical_value=critical_value,
    )

    rescored = transition_weight_sensitivity(
        result,
        weights=(PRIMARY_TRANSITION_WEIGHT,),
    )

    rescored = (
        rescored
        .sort_values("parameter")
        .reset_index(drop=True)
    )

    return (
        rescored["parameter"].to_numpy(dtype=float),
        rescored["gdis"].to_numpy(dtype=float),
    )


# =====================================================================
# STATE MEDIANS
# =====================================================================

def get_state_medians(pseudotime, obs):
    medians = {}

    for state in BETA_LINEAGE:
        mask = (
            obs["cell_type"].astype(str)
            == state
        ).to_numpy()

        values = pseudotime[mask]
        values = values[np.isfinite(values)]

        medians[state] = float(
            np.median(values)
        )

    return medians


# =====================================================================
# PEAK DETECTION / ROBUST FAMILY RE-DERIVATION
# =====================================================================

def detect_local_peaks(
    parameters,
    scores,
    modality,
    design_name,
):
    dynamic_range = float(
        np.max(scores) - np.min(scores)
    )

    if dynamic_range <= 0:
        return pd.DataFrame()

    threshold = (
        MIN_RELATIVE_PROMINENCE
        * dynamic_range
    )

    indices, _ = find_peaks(
        scores,
        prominence=threshold,
        distance=MIN_PEAK_DISTANCE_WINDOWS,
    )

    if len(indices) == 0:
        return pd.DataFrame()

    prominences = peak_prominences(
        scores,
        indices,
    )[0]

    rows = []

    for index, prominence in zip(
        indices,
        prominences,
    ):
        rows.append(
            {
                "modality": modality,
                "window_design": design_name,
                "peak_pt": float(parameters[index]),
                "peak_score": float(scores[index]),
                "relative_prominence": float(
                    prominence / dynamic_range
                ),
            }
        )

    return pd.DataFrame(rows)


def cluster_peak_families(peaks):
    if peaks.empty:
        return []

    peaks = (
        peaks
        .sort_values("peak_pt")
        .reset_index(drop=True)
    )

    clusters = []

    for _, row in peaks.iterrows():
        item = row.to_dict()
        pt = float(item["peak_pt"])

        if not clusters:
            clusters.append([item])
            continue

        centers = np.array(
            [
                np.median(
                    [
                        member["peak_pt"]
                        for member in cluster
                    ]
                )
                for cluster in clusters
            ],
            dtype=float,
        )

        distances = np.abs(
            centers - pt
        )

        closest = int(np.argmin(distances))

        if (
            distances[closest]
            <= FAMILY_MATCH_TOLERANCE
        ):
            clusters[closest].append(item)
        else:
            clusters.append([item])

    return clusters


def summarize_families(modality, clusters):
    rows = []

    for family_id, cluster in enumerate(clusters):
        df = pd.DataFrame(cluster)

        designs = (
            df["window_design"]
            .astype(str)
            .unique()
        )

        median_pt = float(
            df["peak_pt"].median()
        )

        spread = float(
            df["peak_pt"].max()
            - df["peak_pt"].min()
        )

        median_prominence = float(
            df["relative_prominence"].median()
        )

        robust = (
            len(designs)
            == REQUIRED_WINDOW_DESIGNS
            and spread
            <= MAX_ROBUST_FAMILY_SPREAD
            and median_prominence
            >= MIN_RELATIVE_PROMINENCE
        )

        rows.append(
            {
                "modality": modality,
                "family_id": family_id,
                "median_peak_pt": median_pt,
                "peak_pt_spread": spread,
                "median_relative_prominence": median_prominence,
                "robust": robust,
            }
        )

    return pd.DataFrame(rows)


def derive_frozen_centers(
    X_rna_z,
    X_atac_z,
    pseudotime,
    medians,
):
    critical_value = 0.5 * (
        medians["Fev+"]
        + medians["Fev+ Beta"]
    )

    all_peaks = []

    for design_name, window_size, step_size in WINDOW_DESIGNS:
        rna_traj, parameters = create_windows(
            X_rna_z,
            pseudotime,
            window_size,
            step_size,
        )

        atac_traj, atac_parameters = create_windows(
            X_atac_z,
            pseudotime,
            window_size,
            step_size,
        )

        if not np.allclose(
            parameters,
            atac_parameters,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                "RNA and ATAC window centers differ."
            )

        rna_x, rna_y = sustained_gdis_curve(
            rna_traj,
            parameters,
            critical_value,
        )

        atac_x, atac_y = sustained_gdis_curve(
            atac_traj,
            parameters,
            critical_value,
        )

        all_peaks.append(
            detect_local_peaks(
                rna_x,
                rna_y,
                "RNA",
                design_name,
            )
        )

        all_peaks.append(
            detect_local_peaks(
                atac_x,
                atac_y,
                "ATAC",
                design_name,
            )
        )

    peaks = pd.concat(
        all_peaks,
        ignore_index=True,
    )

    rna_families = summarize_families(
        "RNA",
        cluster_peak_families(
            peaks.loc[
                peaks["modality"] == "RNA"
            ]
        ),
    )

    atac_families = summarize_families(
        "ATAC",
        cluster_peak_families(
            peaks.loc[
                peaks["modality"] == "ATAC"
            ]
        ),
    )

    robust_rna = rna_families.loc[
        rna_families["robust"]
    ].copy()

    robust_atac = atac_families.loc[
        atac_families["robust"]
    ].copy()

    # Frozen RNA event must lie in Ngn3 high -> Fev+ interval.
    eligible_rna = robust_rna.loc[
        (
            robust_rna["median_peak_pt"]
            >= medians["Ngn3 high"]
        )
        & (
            robust_rna["median_peak_pt"]
            <= medians["Fev+"]
        )
    ]

    if len(eligible_rna) != 1:
        raise RuntimeError(
            "Frozen RNA family could not be uniquely reproduced."
        )

    rna_row = eligible_rna.iloc[0]
    rna_center = float(
        rna_row["median_peak_pt"]
    )

    # Proximal preceding robust ATAC family from Ngn3-low median onward.
    eligible_atac = robust_atac.loc[
        (
            robust_atac["median_peak_pt"]
            >= medians["Ngn3 low"]
        )
        & (
            robust_atac["median_peak_pt"]
            < rna_center
        )
    ]

    if eligible_atac.empty:
        raise RuntimeError(
            "Frozen ATAC family could not be reproduced."
        )

    atac_row = eligible_atac.loc[
        eligible_atac[
            "median_peak_pt"
        ].idxmax()
    ]

    atac_center = float(
        atac_row["median_peak_pt"]
    )

    return (
        rna_center,
        atac_center,
        critical_value,
    )


# =====================================================================
# DIAGNOSTIC PEAK CHARACTERIZATION
# =====================================================================

def characterize_curve(
    parameters,
    scores,
    frozen_center,
):
    """
    Characterize peak behavior relative to one frozen family center.
    """
    dynamic_range = float(
        np.max(scores) - np.min(scores)
    )

    threshold = (
        MIN_RELATIVE_PROMINENCE
        * dynamic_range
    )

    indices, _ = find_peaks(
        scores,
        prominence=threshold,
        distance=MIN_PEAK_DISTANCE_WINDOWS,
    )

    if len(indices) > 0:
        prominences = peak_prominences(
            scores,
            indices,
        )[0]

        peak_pts = parameters[
            indices
        ]

        distances = np.abs(
            peak_pts - frozen_center
        )

        eligible_mask = (
            distances
            <= FAMILY_MATCH_TOLERANCE
        )

        eligible_indices = np.flatnonzero(
            eligible_mask
        )

        n_all_peaks = len(indices)
        n_eligible = len(
            eligible_indices
        )

        any_peak = True
        family_detected = (
            n_eligible >= 1
        )

        strongest_local = int(
            np.argmax(
                scores[
                    indices
                ]
            )
        )

        strongest_index = indices[
            strongest_local
        ]

        strongest_peak_pt = float(
            parameters[
                strongest_index
            ]
        )

        strongest_peak_distance = float(
            abs(
                strongest_peak_pt
                - frozen_center
            )
        )

        if family_detected:
            best_local = eligible_indices[
                np.argmin(
                    distances[
                        eligible_indices
                    ]
                )
            ]

            best_index = indices[
                best_local
            ]

            tracked_peak_pt = float(
                parameters[
                    best_index
                ]
            )

            tracked_relative_prominence = float(
                prominences[
                    best_local
                ]
                / dynamic_range
            )
        else:
            tracked_peak_pt = np.nan
            tracked_relative_prominence = np.nan

    else:
        any_peak = False
        family_detected = False
        n_all_peaks = 0
        n_eligible = 0
        tracked_peak_pt = np.nan
        tracked_relative_prominence = np.nan
        strongest_peak_pt = np.nan
        strongest_peak_distance = np.nan

    # Characterize the entire frozen family tracking interval, irrespective
    # of whether a qualifying local maximum exists.
    region_mask = (
        np.abs(
            parameters
            - frozen_center
        )
        <= FAMILY_MATCH_TOLERANCE
    )

    local_parameters = parameters[
        region_mask
    ]

    local_scores = scores[
        region_mask
    ]

    if len(local_scores) >= 2:
        local_range = float(
            np.max(local_scores)
            - np.min(local_scores)
        )

        roughness = float(
            np.mean(
                np.abs(
                    np.diff(
                        local_scores
                    )
                )
            )
        )

        local_maximum = float(
            np.max(
                local_scores
            )
        )

        plateau_cutoff = (
            PLATEAU_FRACTION
            * local_maximum
        )

        plateau_mask = (
            local_scores
            >= plateau_cutoff
        )

        if np.any(
            plateau_mask
        ):
            plateau_pts = local_parameters[
                plateau_mask
            ]

            plateau_width = float(
                plateau_pts.max()
                - plateau_pts.min()
            )
        else:
            plateau_width = np.nan

    else:
        local_range = np.nan
        roughness = np.nan
        plateau_width = np.nan

    return {
        "any_peak_detected": any_peak,
        "family_detected": family_detected,
        "n_all_peaks": n_all_peaks,
        "n_eligible_family_peaks": n_eligible,
        "tracked_peak_pt": tracked_peak_pt,
        "tracked_relative_prominence": tracked_relative_prominence,
        "strongest_peak_pt": strongest_peak_pt,
        "strongest_peak_distance_from_family": strongest_peak_distance,
        "local_score_range": local_range,
        "local_roughness": roughness,
        "plateau_width_90pct": plateau_width,
    }


# =====================================================================
# BOOTSTRAP
# =====================================================================

def make_bootstrap_indices(
    ordered_indices,
    rng,
):
    boot = []

    n = len(
        ordered_indices
    )

    for start in range(
        0,
        n,
        BOOTSTRAP_STRATUM_SIZE,
    ):
        stop = min(
            start
            + BOOTSTRAP_STRATUM_SIZE,
            n,
        )

        stratum = ordered_indices[
            start:stop
        ]

        sampled = rng.choice(
            stratum,
            size=len(stratum),
            replace=True,
        )

        sampled = np.sort(
            sampled
        )

        boot.extend(
            sampled.tolist()
        )

    return np.asarray(
        boot,
        dtype=int,
    )


def diagnostic_bootstrap_replicate(
    X_rna_z,
    X_atac_z,
    pseudotime,
    ordered_indices,
    frozen_rna_center,
    frozen_atac_center,
    critical_value,
    rng,
):
    boot_idx = make_bootstrap_indices(
        ordered_indices,
        rng,
    )

    pt = pseudotime[
        boot_idx
    ]

    Xr = X_rna_z[
        boot_idx
    ]

    Xa = X_atac_z[
        boot_idx
    ]

    order = np.argsort(
        pt,
        kind="mergesort",
    )

    pt = pt[
        order
    ]

    Xr = Xr[
        order
    ]

    Xa = Xa[
        order
    ]

    rna_traj, parameters = create_windows(
        Xr,
        pt,
        PRIMARY_WINDOW_SIZE,
        PRIMARY_STEP_SIZE,
    )

    atac_traj, atac_parameters = create_windows(
        Xa,
        pt,
        PRIMARY_WINDOW_SIZE,
        PRIMARY_STEP_SIZE,
    )

    if not np.allclose(
        parameters,
        atac_parameters,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Bootstrap RNA/ATAC centers differ."
        )

    rna_x, rna_y = sustained_gdis_curve(
        rna_traj,
        parameters,
        critical_value,
    )

    atac_x, atac_y = sustained_gdis_curve(
        atac_traj,
        parameters,
        critical_value,
    )

    rna_diag = characterize_curve(
        rna_x,
        rna_y,
        frozen_rna_center,
    )

    atac_diag = characterize_curve(
        atac_x,
        atac_y,
        frozen_atac_center,
    )

    result = {}

    for key, value in rna_diag.items():
        result[
            f"rna_{key}"
        ] = value

    for key, value in atac_diag.items():
        result[
            f"atac_{key}"
        ] = value

    result[
        "paired_family_detected"
    ] = bool(
        rna_diag[
            "family_detected"
        ]
        and atac_diag[
            "family_detected"
        ]
    )

    return result


# =====================================================================
# SUMMARY
# =====================================================================

def binary_rate(series):
    return float(
        np.mean(
            series.astype(bool)
        )
    )


def numeric_summary(df, columns):
    rows = []

    for column in columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        if len(values) == 0:
            continue

        rows.append(
            {
                "metric": column,
                "n": len(values),
                "mean": float(
                    np.mean(values)
                ),
                "median": float(
                    np.median(values)
                ),
                "q25": float(
                    np.quantile(
                        values,
                        0.25,
                    )
                ),
                "q75": float(
                    np.quantile(
                        values,
                        0.75,
                    )
                ),
                "q95": float(
                    np.quantile(
                        values,
                        0.95,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F4b — FROZEN PEAK-ESTIMATOR STABILITY DIAGNOSTICS"
    )

    print(
        f"Dataset: {DATA_FILE.resolve()}"
    )
    print()
    print(
        "Phase F4 primary conclusion remains frozen:"
    )
    print(
        "  event-specific peak CMIL = INCONCLUSIVE under bootstrap."
    )
    print()
    print(
        f"Diagnostic bootstrap replicates: {N_BOOTSTRAPS}"
    )
    print(
        "No thresholds are relaxed."
    )
    print(
        "CMIL is not redefined."
    )
    print(
        "No output files are created."
    )

    if not DATA_FILE.exists():
        print()
        print(
            "ERROR: Dataset file not found."
        )
        sys.exit(1)

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    section("1. LOAD DATASET")

    try:
        mdata = mu.read_h5mu(
            DATA_FILE,
            backed="r",
        )
    except TypeError:
        mdata = mu.read_h5mu(
            DATA_FILE,
        )

    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]
    obs_full = rna.obs.copy()

    beta_mask = (
        obs_full[
            "cell_type"
        ]
        .astype(str)
        .isin(
            BETA_LINEAGE
        )
    ).to_numpy()

    obs = (
        obs_full.loc[
            beta_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    obs.index = pd.Index(
        [
            f"beta_{i}"
            for i in range(
                len(obs)
            )
        ]
    )

    X_rna = as_numpy(
        rna.obsm[
            RNA_REPRESENTATION
        ]
    )[
        beta_mask,
        :N_LATENT_DIMS,
    ]

    X_atac = as_numpy(
        atac.obsm[
            ATAC_REPRESENTATION
        ]
    )[
        beta_mask,
        :N_LATENT_DIMS,
    ]

    X_rna_z = zscore_columns(
        X_rna
    )

    X_atac_z = zscore_columns(
        X_atac
    )

    print(
        f"Cells: {len(obs):,}"
    )

    # -----------------------------------------------------------------
    # Common trajectory / frozen centers
    # -----------------------------------------------------------------

    section(
        "2. REPRODUCE COMMON TRAJECTORY AND FROZEN EVENT CENTERS"
    )

    pseudotime = reconstruct_common_pseudotime(
        X_rna,
        obs,
    )

    medians = get_state_medians(
        pseudotime,
        obs,
    )

    (
        frozen_rna_center,
        frozen_atac_center,
        critical_value,
    ) = derive_frozen_centers(
        X_rna_z,
        X_atac_z,
        pseudotime,
        medians,
    )

    print(
        f"Frozen RNA family center:  "
        f"{frozen_rna_center:.6f}"
    )
    print(
        f"Frozen ATAC family center: "
        f"{frozen_atac_center:.6f}"
    )
    print(
        f"Tracking tolerance:        "
        f"+/- {FAMILY_MATCH_TOLERANCE:.3f}"
    )

    # -----------------------------------------------------------------
    # Original-curve diagnostic
    # -----------------------------------------------------------------

    section(
        "3. ORIGINAL PRIMARY 400/100 CURVE DIAGNOSTICS"
    )

    rna_traj, parameters = create_windows(
        X_rna_z,
        pseudotime,
        PRIMARY_WINDOW_SIZE,
        PRIMARY_STEP_SIZE,
    )

    atac_traj, atac_parameters = create_windows(
        X_atac_z,
        pseudotime,
        PRIMARY_WINDOW_SIZE,
        PRIMARY_STEP_SIZE,
    )

    rna_x, rna_y = sustained_gdis_curve(
        rna_traj,
        parameters,
        critical_value,
    )

    atac_x, atac_y = sustained_gdis_curve(
        atac_traj,
        atac_parameters,
        critical_value,
    )

    original_rna = characterize_curve(
        rna_x,
        rna_y,
        frozen_rna_center,
    )

    original_atac = characterize_curve(
        atac_x,
        atac_y,
        frozen_atac_center,
    )

    original = pd.DataFrame(
        [
            {
                "modality": "RNA",
                **original_rna,
            },
            {
                "modality": "ATAC",
                **original_atac,
            },
        ]
    ).set_index(
        "modality"
    )

    print_df(
        original
    )

    # -----------------------------------------------------------------
    # Diagnostic bootstrap
    # -----------------------------------------------------------------

    section(
        "4. DIAGNOSTIC PAIRED BOOTSTRAP"
    )

    finite_indices = np.flatnonzero(
        np.isfinite(
            pseudotime
        )
    )

    ordered_indices = finite_indices[
        np.argsort(
            pseudotime[
                finite_indices
            ],
            kind="mergesort",
        )
    ]

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    records = []

    for b in range(
        1,
        N_BOOTSTRAPS + 1,
    ):
        row = diagnostic_bootstrap_replicate(
            X_rna_z=X_rna_z,
            X_atac_z=X_atac_z,
            pseudotime=pseudotime,
            ordered_indices=ordered_indices,
            frozen_rna_center=frozen_rna_center,
            frozen_atac_center=frozen_atac_center,
            critical_value=critical_value,
            rng=rng,
        )

        row[
            "bootstrap_id"
        ] = b

        records.append(
            row
        )

        if (
            b % PROGRESS_EVERY == 0
            or b == N_BOOTSTRAPS
        ):
            current = pd.DataFrame(
                records
            )

            paired = int(
                current[
                    "paired_family_detected"
                ].sum()
            )

            print(
                f"Completed {b:3d}/{N_BOOTSTRAPS} | "
                f"paired family detected "
                f"{paired:3d}/{b:3d}"
            )

    results = pd.DataFrame(
        records
    )

    # -----------------------------------------------------------------
    # Detection failure modes
    # -----------------------------------------------------------------

    section(
        "5. EVENT-DETECTION FAILURE MODES"
    )

    rows = []

    for modality in [
        "rna",
        "atac",
    ]:
        any_peak_col = (
            f"{modality}_any_peak_detected"
        )

        family_col = (
            f"{modality}_family_detected"
        )

        n_eligible_col = (
            f"{modality}_n_eligible_family_peaks"
        )

        rows.append(
            {
                "modality": modality.upper(),
                "any_peak_rate": binary_rate(
                    results[
                        any_peak_col
                    ]
                ),
                "frozen_family_detection_rate": binary_rate(
                    results[
                        family_col
                    ]
                ),
                "no_peak_rate": float(
                    np.mean(
                        ~results[
                            any_peak_col
                        ].astype(bool)
                    )
                ),
                "peak_exists_but_family_absent_rate": float(
                    np.mean(
                        results[
                            any_peak_col
                        ].astype(bool)
                        & ~results[
                            family_col
                        ].astype(bool)
                    )
                ),
                "multiple_eligible_family_peaks_rate": float(
                    np.mean(
                        results[
                            n_eligible_col
                        ]
                        > 1
                    )
                ),
            }
        )

    failure_summary = pd.DataFrame(
        rows
    ).set_index(
        "modality"
    )

    print_df(
        failure_summary
    )

    print()
    print(
        "Paired frozen-family detection rate:"
    )
    print(
        f"  {binary_rate(results['paired_family_detected']):.4f}"
    )

    # -----------------------------------------------------------------
    # Shape diagnostics
    # -----------------------------------------------------------------

    section(
        "6. EVENT-REGION SHAPE DIAGNOSTICS"
    )

    metric_columns = [
        "rna_tracked_relative_prominence",
        "atac_tracked_relative_prominence",
        "rna_strongest_peak_distance_from_family",
        "atac_strongest_peak_distance_from_family",
        "rna_local_score_range",
        "atac_local_score_range",
        "rna_local_roughness",
        "atac_local_roughness",
        "rna_plateau_width_90pct",
        "atac_plateau_width_90pct",
    ]

    shape_summary = numeric_summary(
        results,
        metric_columns,
    )

    print_df(
        shape_summary.set_index(
            "metric"
        )
    )

    # -----------------------------------------------------------------
    # Peak location diagnostics
    # -----------------------------------------------------------------

    section(
        "7. TRACKED-PEAK LOCATION DIAGNOSTICS"
    )

    location_summary = numeric_summary(
        results,
        [
            "rna_tracked_peak_pt",
            "atac_tracked_peak_pt",
            "rna_strongest_peak_pt",
            "atac_strongest_peak_pt",
        ],
    )

    print_df(
        location_summary.set_index(
            "metric"
        )
    )

    # -----------------------------------------------------------------
    # Decision
    # -----------------------------------------------------------------

    section(
        "8. PHASE F4b DIAGNOSTIC CONCLUSION"
    )

    rna_family_rate = binary_rate(
        results[
            "rna_family_detected"
        ]
    )

    atac_family_rate = binary_rate(
        results[
            "atac_family_detected"
        ]
    )

    paired_rate = binary_rate(
        results[
            "paired_family_detected"
        ]
    )

    rna_escape = float(
        np.mean(
            results[
                "rna_any_peak_detected"
            ].astype(bool)
            & ~results[
                "rna_family_detected"
            ].astype(bool)
        )
    )

    atac_escape = float(
        np.mean(
            results[
                "atac_any_peak_detected"
            ].astype(bool)
            & ~results[
                "atac_family_detected"
            ].astype(bool)
        )
    )

    print(
        f"RNA frozen-family detection rate:  "
        f"{rna_family_rate:.4f}"
    )
    print(
        f"ATAC frozen-family detection rate: "
        f"{atac_family_rate:.4f}"
    )
    print(
        f"Paired detection rate:             "
        f"{paired_rate:.4f}"
    )
    print()
    print(
        f"RNA peak-exists/family-absent rate:  "
        f"{rna_escape:.4f}"
    )
    print(
        f"ATAC peak-exists/family-absent rate: "
        f"{atac_escape:.4f}"
    )

    print()
    print(
        "PHASE F4b VERDICT: DIAGNOSTIC ONLY"
    )
    print()
    print(
        "The Phase F4 inferential result remains unchanged. "
        "Use these diagnostics to determine whether the instability event "
        "is intrinsically broad/multipeak or whether one modality is mainly "
        "responsible for bootstrap event loss."
    )

    print()
    print(
        "No CMIL threshold was changed."
    )
    print(
        "No new CMIL estimator was introduced."
    )
    print(
        "No priming claim was made."
    )
    print(
        "No files were created."
    )

    line("=")


if __name__ == "__main__":
    main()

