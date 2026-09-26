#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
09_phase_f4_event_specific_cmil_bootstrap.py
=============================================

Project:
Cross-Modal Dynamical Instability Reveals Chromatin Priming
before Transcriptional Cell-Fate Transitions

Phase F4:
Estimate event-specific Cross-Modal Instability Lead (CMIL) for the event pair
that was prospectively frozen in Phase F3c.

FROZEN BIOLOGICAL EVENT
-----------------------
RNA event:
    robust sustained-instability family assigned to
    Ngn3 high -> Fev+

ATAC event:
    proximal robust ATAC sustained-instability family immediately upstream
    of that RNA event.

Phase F3c established that ATAC precedes RNA for this same event pair in ALL
three prespecified window designs.

CMIL DEFINITION
---------------
    CMIL = t_RNA - t_ATAC

Interpretation:
    CMIL > 0  : ATAC instability precedes RNA instability
    CMIL = 0  : approximately synchronous
    CMIL < 0  : RNA instability precedes ATAC instability

IMPORTANT:
This script estimates CMIL ONLY AFTER the event pair has been frozen.

PRIMARY ANALYSIS
----------------
Representation:
    RNA  = X_scVI, 10D
    ATAC = X_poissonvi, 10D

Common ordering:
    fixed RNA-only DPT accepted in Phase F2

GDIS timing profile:
    sustained instability only (lambda_t = 0)

Primary window:
    400 cells / 100-cell step

Observed sensitivity:
    300/75
    400/100
    500/125

BOOTSTRAP
---------
Paired pseudotime-stratified bootstrap:
    - RNA and ATAC use exactly the same resampled cell indices.
    - Cells are first ordered by the accepted common pseudotime.
    - The ordered trajectory is divided into contiguous strata.
    - Cells are resampled WITH replacement within each stratum.
    - Strata remain in developmental order.
    - The accepted common pseudotime itself is NOT recomputed.

Therefore, the bootstrap interval is a CONDITIONAL TRAJECTORY-RESAMPLING
uncertainty interval. It is not a population-level biological confidence
interval and does not create biological replicates from pooled embryos.

EVENT-FAMILY TRACKING
---------------------
To prevent peak hopping during bootstrap:
    - the robust RNA and ATAC family centers are first re-derived from the
      original data across all three window designs;
    - in each bootstrap replicate, a local peak must fall within +/- 0.08
      pseudotime of the frozen family center;
    - the closest eligible local peak is used;
    - if no eligible local peak exists, that replicate is recorded as an
      event-detection failure rather than switching to another peak family.

No files are written.

Run:
    python 09_phase_f4_event_specific_cmil_bootstrap.py

Requirements:
    numpy
    pandas
    scipy
    scanpy
    anndata
    mudata
    pygdis
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

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

FROZEN_TRANSITION = "Ngn3 high -> Fev+"

RNA_REPRESENTATION = "X_scVI"
ATAC_REPRESENTATION = "X_poissonvi"
N_LATENT_DIMS = 10

DPT_NEIGHBORS = 30
N_DIFFUSION_COMPONENTS = 10
RANDOM_SEED = 20260920

WINDOW_DESIGNS = [
    ("300_75", 300, 75),
    ("400_100", 400, 100),   # primary
    ("500_125", 500, 125),
]

PRIMARY_WINDOW_DESIGN = "400_100"
PRIMARY_WINDOW_SIZE = 400
PRIMARY_STEP_SIZE = 100

PRIMARY_TRANSITION_WEIGHT = 0.0
REFERENCE_TRANSITION_WEIGHT = 0.18

# Peak-family rules frozen from F3b/F3c.
MIN_RELATIVE_PROMINENCE = 0.05
MIN_PEAK_DISTANCE_WINDOWS = 3
FAMILY_MATCH_TOLERANCE = 0.08
MAX_ROBUST_FAMILY_SPREAD = 0.10
REQUIRED_WINDOW_DESIGNS = 3

# Bootstrap.
N_BOOTSTRAPS = 500

# Resample within contiguous pseudotime strata of this many cells.
# 100 is one quarter of the primary 400-cell GDIS window.
BOOTSTRAP_STRATUM_SIZE = 100

# Print progress every N replicates.
PROGRESS_EVERY = 25

# GO / review criteria.
MIN_BOOTSTRAP_DETECTION_RATE_GO = 0.80
MIN_BOOTSTRAP_DIRECTIONAL_SUPPORT_GO = 0.95

MIN_BOOTSTRAP_DETECTION_RATE_CAUTION = 0.70
MIN_BOOTSTRAP_DIRECTIONAL_SUPPORT_CAUTION = 0.90


# =====================================================================
# DISPLAY
# =====================================================================

def line(char="=", width=112):
    print(char * width)


def section(title):
    print()
    line("=")
    print(title)
    line("=")


def subsection(title):
    print()
    line("-")
    print(title)
    line("-")


def print_df(df, digits=6):
    if df.empty:
        print("(empty)")
        return

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 320,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string(index=True))


# =====================================================================
# GENERIC HELPERS
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
# WINDOWS + SUSTAINED GDIS
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

    parameters = np.asarray(
        parameters,
        dtype=float,
    )

    if len(trajectories) < 9:
        raise RuntimeError(
            f"Only {len(trajectories)} GDIS windows were produced."
        )

    return trajectories, parameters


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
# BIOLOGICAL TRANSITION STRUCTURE
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

        if len(values) == 0:
            raise RuntimeError(
                f"No finite pseudotime values for {state}."
            )

        medians[state] = float(
            np.median(values)
        )

    return medians


def transition_interval_for_frozen_event(medians):
    """
    Frozen event:
        Ngn3 high -> Fev+

    RNA transition interval:
        [median(Ngn3 high), median(Fev+)]

    ATAC upstream lower boundary:
        median(Ngn3 low)
    """
    return {
        "transition": FROZEN_TRANSITION,
        "rna_lower": medians["Ngn3 high"],
        "rna_upper": medians["Fev+"],
        "atac_upstream_lower": medians["Ngn3 low"],
    }


# =====================================================================
# PEAK DETECTION AND FAMILY CLUSTERING
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

    minimum_prominence = (
        MIN_RELATIVE_PROMINENCE
        * dynamic_range
    )

    indices, _ = find_peaks(
        scores,
        prominence=minimum_prominence,
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
                "peak_index": int(index),
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

        closest = int(
            np.argmin(distances)
        )

        if (
            distances[closest]
            <= FAMILY_MATCH_TOLERANCE
        ):
            clusters[closest].append(
                item
            )
        else:
            clusters.append(
                [item]
            )

    return clusters


def summarize_families(modality, clusters):
    family_rows = []
    member_rows = []

    for family_id, cluster in enumerate(clusters):
        df = pd.DataFrame(cluster)

        designs = sorted(
            df["window_design"]
            .astype(str)
            .unique()
            .tolist()
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

        family_rows.append(
            {
                "modality": modality,
                "family_id": family_id,
                "n_window_designs": len(designs),
                "median_peak_pt": median_pt,
                "peak_pt_spread": spread,
                "median_relative_prominence": median_prominence,
                "robust": robust,
            }
        )

        for _, member in df.iterrows():
            member_rows.append(
                {
                    "modality": modality,
                    "family_id": family_id,
                    "window_design": str(
                        member["window_design"]
                    ),
                    "peak_pt": float(
                        member["peak_pt"]
                    ),
                    "peak_score": float(
                        member["peak_score"]
                    ),
                    "relative_prominence": float(
                        member["relative_prominence"]
                    ),
                }
            )

    return (
        pd.DataFrame(family_rows),
        pd.DataFrame(member_rows),
    )


def derive_frozen_event_pair(
    X_rna_z,
    X_atac_z,
    pseudotime,
    medians,
):
    """
    Re-derive the Phase-F3c frozen event pair from the original data.

    Returns:
        rna_family
        atac_family
        rna_members
        atac_members
    """
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

    rna_families, rna_members = summarize_families(
        "RNA",
        cluster_peak_families(
            peaks.loc[
                peaks["modality"] == "RNA"
            ]
        ),
    )

    atac_families, atac_members = summarize_families(
        "ATAC",
        cluster_peak_families(
            peaks.loc[
                peaks["modality"] == "ATAC"
            ]
        ),
    )

    robust_rna = (
        rna_families.loc[
            rna_families["robust"]
        ]
        .copy()
    )

    robust_atac = (
        atac_families.loc[
            atac_families["robust"]
        ]
        .copy()
    )

    interval = transition_interval_for_frozen_event(
        medians
    )

    # Robust RNA family inside the frozen transition interval.
    eligible_rna = robust_rna.loc[
        (
            robust_rna["median_peak_pt"]
            >= interval["rna_lower"]
        )
        & (
            robust_rna["median_peak_pt"]
            <= interval["rna_upper"]
        )
    ].copy()

    if len(eligible_rna) != 1:
        raise RuntimeError(
            "Could not uniquely reproduce the frozen robust RNA event."
        )

    rna_family = eligible_rna.iloc[0]

    rna_center = float(
        rna_family["median_peak_pt"]
    )

    # Proximal preceding robust ATAC family in frozen upstream interval.
    eligible_atac = robust_atac.loc[
        (
            robust_atac["median_peak_pt"]
            >= interval["atac_upstream_lower"]
        )
        & (
            robust_atac["median_peak_pt"]
            < rna_center
        )
    ].copy()

    if eligible_atac.empty:
        raise RuntimeError(
            "Could not reproduce an eligible upstream robust ATAC family."
        )

    selected_index = (
        eligible_atac["median_peak_pt"]
        .idxmax()
    )

    atac_family = eligible_atac.loc[
        selected_index
    ]

    return (
        rna_family,
        atac_family,
        rna_members,
        atac_members,
    )


# =====================================================================
# TRACK ONE FROZEN FAMILY IN A CURVE
# =====================================================================

def track_family_peak(
    parameters,
    scores,
    frozen_center,
):
    """
    Detect local peaks and select the peak closest to the FROZEN family center
    within +/- FAMILY_MATCH_TOLERANCE.

    Returns:
        peak_pt or None
    """
    dynamic_range = float(
        np.max(scores) - np.min(scores)
    )

    if dynamic_range <= 0:
        return None

    minimum_prominence = (
        MIN_RELATIVE_PROMINENCE
        * dynamic_range
    )

    indices, _ = find_peaks(
        scores,
        prominence=minimum_prominence,
        distance=MIN_PEAK_DISTANCE_WINDOWS,
    )

    if len(indices) == 0:
        return None

    candidate_pts = parameters[indices]

    distances = np.abs(
        candidate_pts - frozen_center
    )

    eligible = np.flatnonzero(
        distances
        <= FAMILY_MATCH_TOLERANCE
    )

    if len(eligible) == 0:
        return None

    # Track the same family by proximity to its frozen center.
    best_local = eligible[
        np.argmin(
            distances[eligible]
        )
    ]

    best_peak_index = indices[
        best_local
    ]

    return float(
        parameters[best_peak_index]
    )


# =====================================================================
# PAIRED PSEUDOTIME-STRATIFIED BOOTSTRAP
# =====================================================================

def make_bootstrap_indices(
    ordered_indices,
    rng,
    stratum_size,
):
    """
    Resample paired cells WITH replacement inside contiguous pseudotime strata.

    Each stratum retains its original position along the trajectory.
    """
    boot = []

    n = len(ordered_indices)

    for start in range(
        0,
        n,
        stratum_size,
    ):
        stop = min(
            start + stratum_size,
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

        # Preserve developmental ordering inside the stratum using original
        # common pseudotime rank, not the random draw order.
        sampled = np.sort(sampled)

        boot.extend(
            sampled.tolist()
        )

    return np.asarray(
        boot,
        dtype=int,
    )


def bootstrap_one_replicate(
    X_rna_z,
    X_atac_z,
    pseudotime,
    ordered_indices,
    frozen_rna_center,
    frozen_atac_center,
    critical_value,
    rng,
):
    """
    One paired conditional-trajectory bootstrap replicate.
    """
    boot_idx = make_bootstrap_indices(
        ordered_indices,
        rng,
        BOOTSTRAP_STRATUM_SIZE,
    )

    X_rna_b = X_rna_z[
        boot_idx
    ]

    X_atac_b = X_atac_z[
        boot_idx
    ]

    pt_b = pseudotime[
        boot_idx
    ]

    # The resampled indices were generated in pseudotime-stratum order,
    # but sort explicitly by pseudotime for safety.
    order = np.argsort(
        pt_b,
        kind="mergesort",
    )

    X_rna_b = X_rna_b[
        order
    ]

    X_atac_b = X_atac_b[
        order
    ]

    pt_b = pt_b[
        order
    ]

    rna_traj, parameters = create_windows(
        X_rna_b,
        pt_b,
        PRIMARY_WINDOW_SIZE,
        PRIMARY_STEP_SIZE,
    )

    atac_traj, atac_parameters = create_windows(
        X_atac_b,
        pt_b,
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
            "Bootstrap RNA/ATAC window centers differ."
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

    t_rna = track_family_peak(
        rna_x,
        rna_y,
        frozen_rna_center,
    )

    t_atac = track_family_peak(
        atac_x,
        atac_y,
        frozen_atac_center,
    )

    if (
        t_rna is None
        or t_atac is None
    ):
        return {
            "detected": False,
            "t_rna": np.nan,
            "t_atac": np.nan,
            "cmil": np.nan,
        }

    cmil = float(
        t_rna - t_atac
    )

    return {
        "detected": True,
        "t_rna": float(t_rna),
        "t_atac": float(t_atac),
        "cmil": cmil,
    }


# =====================================================================
# SUMMARY STATISTICS
# =====================================================================

def percentile_interval(values, alpha=0.05):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    lower = float(
        np.quantile(
            values,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            values,
            1.0 - alpha / 2.0,
        )
    )

    return lower, upper


def summarize_bootstrap(values, name):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    lower, upper = percentile_interval(
        values
    )

    return {
        "quantity": name,
        "n": len(values),
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
        "sd": float(
            np.std(
                values,
                ddof=1,
            )
        ) if len(values) > 1 else np.nan,
        "ci_2.5": lower,
        "ci_97.5": upper,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F4 — EVENT-SPECIFIC CMIL WITH PAIRED TRAJECTORY BOOTSTRAP"
    )

    print(
        f"Dataset: {DATA_FILE.resolve()}"
    )
    print()
    print(
        f"Frozen biological event: {FROZEN_TRANSITION}"
    )
    print(
        "CMIL = t_RNA - t_ATAC"
    )
    print()
    print(
        f"Primary GDIS window: "
        f"{PRIMARY_WINDOW_SIZE}/{PRIMARY_STEP_SIZE}"
    )
    print(
        f"Bootstrap replicates: {N_BOOTSTRAPS}"
    )
    print(
        f"Bootstrap pseudotime stratum size: "
        f"{BOOTSTRAP_STRATUM_SIZE} cells"
    )
    print()
    print(
        "The common RNA-derived pseudotime is held fixed."
    )
    print(
        "RNA and ATAC are resampled with identical paired cell indices."
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
        obs_full["cell_type"]
        .astype(str)
        .isin(BETA_LINEAGE)
    ).to_numpy()

    obs = (
        obs_full.loc[
            beta_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    obs.index = pd.Index(
        [
            f"beta_{i}"
            for i in range(len(obs))
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
    print(
        f"RNA shape:  {X_rna.shape}"
    )
    print(
        f"ATAC shape: {X_atac.shape}"
    )

    # -----------------------------------------------------------------
    # Common pseudotime
    # -----------------------------------------------------------------

    section("2. RECONSTRUCT ACCEPTED COMMON PSEUDOTIME")

    pseudotime = reconstruct_common_pseudotime(
        X_rna,
        obs,
    )

    finite_fraction = float(
        np.isfinite(
            pseudotime
        ).mean()
    )

    print(
        f"Finite pseudotime fraction: "
        f"{finite_fraction:.6f}"
    )

    if finite_fraction < 0.99:
        print(
            "ERROR: Phase-F2 common trajectory was not reproduced."
        )
        sys.exit(1)

    medians = get_state_medians(
        pseudotime,
        obs,
    )

    medians_df = pd.DataFrame(
        [
            {
                "cell_type": state,
                "median_pseudotime": medians[
                    state
                ],
            }
            for state in BETA_LINEAGE
        ]
    ).set_index(
        "cell_type"
    )

    print()
    print_df(
        medians_df
    )

    frozen_interval = (
        transition_interval_for_frozen_event(
            medians
        )
    )

    transition_length = float(
        frozen_interval[
            "rna_upper"
        ]
        - frozen_interval[
            "rna_lower"
        ]
    )

    print()
    print(
        f"Frozen RNA transition interval: "
        f"[{frozen_interval['rna_lower']:.6f}, "
        f"{frozen_interval['rna_upper']:.6f}]"
    )
    print(
        f"Transition interval length: "
        f"{transition_length:.6f}"
    )

    critical_value = 0.5 * (
        medians["Fev+"]
        + medians["Fev+ Beta"]
    )

    # -----------------------------------------------------------------
    # Reproduce frozen family pair
    # -----------------------------------------------------------------

    section("3. REPRODUCE AND FREEZE EVENT FAMILY CENTERS")

    (
        rna_family,
        atac_family,
        rna_members,
        atac_members,
    ) = derive_frozen_event_pair(
        X_rna_z,
        X_atac_z,
        pseudotime,
        medians,
    )

    frozen_rna_center = float(
        rna_family[
            "median_peak_pt"
        ]
    )

    frozen_atac_center = float(
        atac_family[
            "median_peak_pt"
        ]
    )

    print(
        f"Frozen RNA family center:  "
        f"{frozen_rna_center:.6f}"
    )
    print(
        f"Frozen ATAC family center: "
        f"{frozen_atac_center:.6f}"
    )
    print()
    print(
        f"Tracking tolerance for BOTH families: "
        f"+/- {FAMILY_MATCH_TOLERANCE:.3f}"
    )

    # -----------------------------------------------------------------
    # Observed event-specific CMIL across all 3 window designs
    # -----------------------------------------------------------------

    section(
        "4. OBSERVED EVENT-SPECIFIC CMIL ACROSS WINDOW DESIGNS"
    )

    rna_event_members = (
        rna_members.loc[
            rna_members[
                "family_id"
            ]
            == int(
                rna_family[
                    "family_id"
                ]
            )
        ]
        .copy()
    )

    atac_event_members = (
        atac_members.loc[
            atac_members[
                "family_id"
            ]
            == int(
                atac_family[
                    "family_id"
                ]
            )
        ]
        .copy()
    )

    observed = (
        rna_event_members[
            [
                "window_design",
                "peak_pt",
            ]
        ]
        .rename(
            columns={
                "peak_pt": "t_rna"
            }
        )
        .merge(
            atac_event_members[
                [
                    "window_design",
                    "peak_pt",
                ]
            ].rename(
                columns={
                    "peak_pt": "t_atac"
                }
            ),
            on="window_design",
            how="inner",
        )
    )

    observed["cmil"] = (
        observed["t_rna"]
        - observed["t_atac"]
    )

    observed[
        "cmil_fraction_of_transition"
    ] = (
        observed["cmil"]
        / transition_length
    )

    design_order = {
        name: i
        for i, (
            name,
            _,
            _,
        ) in enumerate(
            WINDOW_DESIGNS
        )
    }

    observed[
        "_order"
    ] = (
        observed[
            "window_design"
        ]
        .map(
            design_order
        )
    )

    observed = (
        observed
        .sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(
            drop=True
        )
    )

    print_df(
        observed.set_index(
            "window_design"
        )
    )

    all_observed_positive = bool(
        (
            observed["cmil"]
            > 0
        ).all()
    )

    primary_observed_row = (
        observed.loc[
            observed[
                "window_design"
            ]
            == PRIMARY_WINDOW_DESIGN
        ]
    )

    if len(
        primary_observed_row
    ) != 1:
        raise RuntimeError(
            "Primary observed event pair could not be uniquely recovered."
        )

    primary_observed_row = (
        primary_observed_row
        .iloc[0]
    )

    observed_primary_cmil = float(
        primary_observed_row[
            "cmil"
        ]
    )

    print()
    print(
        f"Observed primary CMIL "
        f"({PRIMARY_WINDOW_DESIGN}): "
        f"{observed_primary_cmil:.6f}"
    )

    # -----------------------------------------------------------------
    # Bootstrap
    # -----------------------------------------------------------------

    section(
        "5. PAIRED PSEUDOTIME-STRATIFIED BOOTSTRAP"
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

    start_clock = time.time()

    for b in range(
        1,
        N_BOOTSTRAPS + 1,
    ):
        result = bootstrap_one_replicate(
            X_rna_z=X_rna_z,
            X_atac_z=X_atac_z,
            pseudotime=pseudotime,
            ordered_indices=ordered_indices,
            frozen_rna_center=frozen_rna_center,
            frozen_atac_center=frozen_atac_center,
            critical_value=critical_value,
            rng=rng,
        )

        result[
            "bootstrap_id"
        ] = b

        records.append(
            result
        )

        if (
            b % PROGRESS_EVERY == 0
            or b == N_BOOTSTRAPS
        ):
            current = pd.DataFrame(
                records
            )

            n_detected = int(
                current[
                    "detected"
                ].sum()
            )

            print(
                f"Completed {b:4d}/{N_BOOTSTRAPS} | "
                f"paired event detected in "
                f"{n_detected:4d}/{b:4d} replicates"
            )

    bootstrap = pd.DataFrame(
        records
    )

    elapsed = time.time() - start_clock

    detected = bootstrap.loc[
        bootstrap["detected"]
    ].copy()

    n_detected = len(
        detected
    )

    detection_rate = (
        n_detected
        / N_BOOTSTRAPS
    )

    print()
    print(
        f"Successful paired-event detections: "
        f"{n_detected}/{N_BOOTSTRAPS}"
    )
    print(
        f"Detection rate: {detection_rate:.4f}"
    )

    # Runtime is descriptive only, useful for reproducibility records.
    print(
        f"Bootstrap computation elapsed seconds: "
        f"{elapsed:.1f}"
    )

    if n_detected < 20:
        print()
        print(
            "ERROR: Too few successful bootstrap detections "
            "for reliable uncertainty estimation."
        )
        sys.exit(1)

    # -----------------------------------------------------------------
    # Bootstrap uncertainty
    # -----------------------------------------------------------------

    section(
        "6. BOOTSTRAP EVENT-LOCATION AND CMIL UNCERTAINTY"
    )

    summary_rows = [
        summarize_bootstrap(
            detected["t_atac"],
            "ATAC event location",
        ),
        summarize_bootstrap(
            detected["t_rna"],
            "RNA event location",
        ),
        summarize_bootstrap(
            detected["cmil"],
            "CMIL = t_RNA - t_ATAC",
        ),
    ]

    bootstrap_summary = pd.DataFrame(
        summary_rows
    ).set_index(
        "quantity"
    )

    print_df(
        bootstrap_summary
    )

    cmil_values = (
        detected["cmil"]
        .to_numpy(
            dtype=float
        )
    )

    directional_support = float(
        np.mean(
            cmil_values > 0
        )
    )

    synchronous_support = float(
        np.mean(
            np.isclose(
                cmil_values,
                0.0,
                atol=1e-12,
            )
        )
    )

    rna_leading_support = float(
        np.mean(
            cmil_values < 0
        )
    )

    cmil_lower, cmil_upper = (
        percentile_interval(
            cmil_values
        )
    )

    normalized_values = (
        cmil_values
        / transition_length
    )

    norm_lower, norm_upper = (
        percentile_interval(
            normalized_values
        )
    )

    print()
    print(
        "Directional bootstrap support:"
    )
    print(
        f"  P(CMIL > 0; ATAC leads RNA) = "
        f"{directional_support:.4f}"
    )
    print(
        f"  P(CMIL = 0)                = "
        f"{synchronous_support:.4f}"
    )
    print(
        f"  P(CMIL < 0; RNA leads ATAC)= "
        f"{rna_leading_support:.4f}"
    )

    print()
    print(
        "Transition-normalized CMIL:"
    )
    print(
        f"  median fraction of "
        f"Ngn3 high -> Fev+ interval = "
        f"{np.median(normalized_values):.4f}"
    )
    print(
        f"  95% percentile interval = "
        f"[{norm_lower:.4f}, {norm_upper:.4f}]"
    )

    # -----------------------------------------------------------------
    # Distribution diagnostics
    # -----------------------------------------------------------------

    section(
        "7. BOOTSTRAP DISTRIBUTION DIAGNOSTICS"
    )

    quantiles = [
        0.01,
        0.025,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.975,
        0.99,
    ]

    quantile_table = pd.DataFrame(
        {
            "quantile": quantiles,
            "t_atac": np.quantile(
                detected["t_atac"],
                quantiles,
            ),
            "t_rna": np.quantile(
                detected["t_rna"],
                quantiles,
            ),
            "cmil": np.quantile(
                detected["cmil"],
                quantiles,
            ),
        }
    ).set_index(
        "quantile"
    )

    print_df(
        quantile_table
    )

    # -----------------------------------------------------------------
    # Decision
    # -----------------------------------------------------------------

    section(
        "8. PHASE F4 DECISION"
    )

    ci_excludes_zero = bool(
        cmil_lower > 0
        or cmil_upper < 0
    )

    positive_ci = bool(
        cmil_lower > 0
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    "Observed CMIL > 0 in all 3 "
                    "prespecified window designs"
                ),
                "pass": all_observed_positive,
            },
            {
                "criterion": (
                    f"Bootstrap paired-event detection rate "
                    f">= {MIN_BOOTSTRAP_DETECTION_RATE_GO:.2f}"
                ),
                "pass": (
                    detection_rate
                    >= MIN_BOOTSTRAP_DETECTION_RATE_GO
                ),
            },
            {
                "criterion": (
                    f"Bootstrap directional support "
                    f"P(CMIL>0) >= "
                    f"{MIN_BOOTSTRAP_DIRECTIONAL_SUPPORT_GO:.2f}"
                ),
                "pass": (
                    directional_support
                    >= MIN_BOOTSTRAP_DIRECTIONAL_SUPPORT_GO
                ),
            },
            {
                "criterion": (
                    "95% percentile CMIL interval entirely > 0"
                ),
                "pass": positive_ci,
            },
        ]
    )

    checks[
        "status"
    ] = np.where(
        checks["pass"],
        "PASS",
        "REVIEW",
    )

    print_df(
        checks.set_index(
            "criterion"
        ),
        digits=0,
    )

    if (
        all_observed_positive
        and detection_rate
        >= MIN_BOOTSTRAP_DETECTION_RATE_GO
        and directional_support
        >= MIN_BOOTSTRAP_DIRECTIONAL_SUPPORT_GO
        and positive_ci
    ):
        verdict = (
            "GO — POSITIVE EVENT-SPECIFIC CMIL SUPPORTED"
        )

        reason = (
            "The frozen ATAC event precedes the frozen RNA event in all "
            "prespecified window designs, the paired event is recovered "
            "reliably under trajectory-stratified resampling, and the "
            "conditional 95% bootstrap interval for CMIL remains above zero."
        )

    elif (
        all_observed_positive
        and detection_rate
        >= MIN_BOOTSTRAP_DETECTION_RATE_CAUTION
        and directional_support
        >= MIN_BOOTSTRAP_DIRECTIONAL_SUPPORT_CAUTION
    ):
        verdict = "GO WITH CAUTION"

        if ci_excludes_zero:
            interval_text = (
                "The bootstrap interval excludes zero, but one or more "
                "other robustness criteria miss the primary GO threshold."
            )
        else:
            interval_text = (
                "The bootstrap interval overlaps zero, so the direction "
                "should be described as suggestive rather than resolved."
            )

        reason = (
            "The frozen pair remains directionally consistent overall. "
            + interval_text
        )

    else:
        verdict = "REVIEW REQUIRED"

        reason = (
            "The frozen event pair is not recovered with sufficient "
            "directional or bootstrap stability to support a CMIL claim."
        )

    print()
    print(
        f"PHASE F4 VERDICT: {verdict}"
    )
    print()
    print(
        reason
    )

    print()
    print(
        "IMPORTANT INTERPRETATION LIMITATION:"
    )
    print(
        "This bootstrap quantifies uncertainty conditional on the accepted "
        "single-cell trajectory and the pooled discovery dataset. It does "
        "not constitute a biological-replicate confidence interval."
    )

    print()
    print(
        "No output files were created."
    )

    if verdict.startswith("GO"):
        print()
        print(
            "Next step should test a null/permutation model and "
            "representation/trajectory sensitivity before any mechanistic "
            "chromatin-priming interpretation."
        )

    line("=")


if __name__ == "__main__":
    main()

