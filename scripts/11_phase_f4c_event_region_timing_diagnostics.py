#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
11_phase_f4c_event_region_timing_diagnostics.py
================================================

Project:
Cross-Modal Dynamical Instability Reveals Chromatin Priming
before Transcriptional Cell-Fate Transitions

Phase F4c:
Exploratory event-region timing diagnostics after the frozen peak-based CMIL
was found to be bootstrap-inconclusive.

STATUS OF THE PRIMARY RESULT
----------------------------
The Phase F4 peak-based CMIL result remains FROZEN and INCONCLUSIVE.

This script does NOT replace that result.

WHY THIS DIAGNOSTIC IS NEEDED
-----------------------------
Phase F4b showed that:
    - both RNA and ATAC retain instability peaks in every bootstrap replicate;
    - the frozen event family is sometimes not represented by a qualifying
      discrete local maximum;
    - both frozen event regions have broad ~90%-of-maximum plateaus;
    - ATAC is strongly multi-event;
    - RNA peak localization is comparatively diffuse.

Therefore a discrete argmax may be an intrinsically noisy timing estimator.

EXPLORATORY REGION-BASED TIMING
-------------------------------
Use the already frozen event-family centers and the already frozen +/-0.08
tracking tolerance to define ONE SHARED analysis interval for both modalities:

    lower = frozen_ATAC_center - 0.08
    upper = frozen_RNA_center  + 0.08

The SAME interval is used for RNA and ATAC.

Within that interval:
1. Interpolate sustained GDIS (lambda_t = 0) onto a dense common grid.
2. Subtract the local minimum as a within-region baseline.
3. Treat the remaining non-negative excess-instability curve as an event
   density over pseudotime.
4. Calculate area-weighted timing quantiles:
       Q25 = early event timing
       Q50 = event center
       Q75 = late event timing
5. Calculate event width:
       Q75 - Q25

Exploratory cross-modal timing descriptors:
    early lead  = Q25_RNA - Q25_ATAC
    center lead = Q50_RNA - Q50_ATAC
    late lead   = Q75_RNA - Q75_ATAC

Positive values mean the ATAC event-region mass occurs earlier than RNA.

IMPORTANT:
These are SECONDARY / EXPLORATORY estimators developed after diagnosing the
limitations of the discrete-peak estimator. They must not be presented as
confirmatory evidence in this discovery dataset. They should be frozen here
and validated in simulations and/or independent datasets before any strong
biological claim.

BOOTSTRAP
---------
Paired pseudotime-stratified bootstrap, same design as Phase F4:
    - fixed accepted RNA-derived pseudotime;
    - identical RNA/ATAC resampled cell indices;
    - 400/100 primary GDIS windows;
    - 500 replicates.

No output files are written.

Run:
    python 11_phase_f4c_event_region_timing_diagnostics.py
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
from scipy.integrate import cumulative_trapezoid

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

PRIMARY_WINDOW_DESIGN = "400_100"
PRIMARY_WINDOW_SIZE = 400
PRIMARY_STEP_SIZE = 100

PRIMARY_TRANSITION_WEIGHT = 0.0
REFERENCE_TRANSITION_WEIGHT = 0.18

MIN_RELATIVE_PROMINENCE = 0.05
MIN_PEAK_DISTANCE_WINDOWS = 3

FAMILY_MATCH_TOLERANCE = 0.08
MAX_ROBUST_FAMILY_SPREAD = 0.10
REQUIRED_WINDOW_DESIGNS = 3

REGION_GRID_POINTS = 500

N_BOOTSTRAPS = 500
BOOTSTRAP_STRATUM_SIZE = 100
PROGRESS_EVERY = 25


# =====================================================================
# DISPLAY
# =====================================================================

def line(char="=", width=116):
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
# COMMON PSEUDOTIME
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

        medians[state] = float(np.median(values))

    return medians


# =====================================================================
# PEAK FAMILY RE-DERIVATION
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

        distances = np.abs(centers - pt)
        closest = int(np.argmin(distances))

        if distances[closest] <= FAMILY_MATCH_TOLERANCE:
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

        median_pt = float(df["peak_pt"].median())
        spread = float(
            df["peak_pt"].max()
            - df["peak_pt"].min()
        )

        median_prominence = float(
            df["relative_prominence"].median()
        )

        robust = (
            len(designs) == REQUIRED_WINDOW_DESIGNS
            and spread <= MAX_ROBUST_FAMILY_SPREAD
            and median_prominence >= MIN_RELATIVE_PROMINENCE
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

    rna_center = float(
        eligible_rna.iloc[0][
            "median_peak_pt"
        ]
    )

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
        eligible_atac["median_peak_pt"].idxmax()
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
# REGION-BASED TIMING
# =====================================================================

def interpolate_region(
    parameters,
    scores,
    region_lower,
    region_upper,
):
    """
    Interpolate one GDIS curve over the same fixed shared region.
    """
    curve_lower = float(np.min(parameters))
    curve_upper = float(np.max(parameters))

    lower = max(region_lower, curve_lower)
    upper = min(region_upper, curve_upper)

    if upper <= lower:
        return None, None

    grid = np.linspace(
        lower,
        upper,
        REGION_GRID_POINTS,
    )

    values = np.interp(
        grid,
        parameters,
        scores,
    )

    return grid, values


def area_quantiles(
    grid,
    values,
    quantiles=(0.25, 0.50, 0.75),
):
    """
    Baseline-correct the region curve and calculate area-weighted pseudotime
    quantiles from the cumulative trapezoidal integral.
    """
    if grid is None or values is None:
        return None

    baseline = float(
        np.min(values)
    )

    excess = (
        values
        - baseline
    )

    excess[
        excess < 0
    ] = 0.0

    cumulative = cumulative_trapezoid(
        excess,
        grid,
        initial=0.0,
    )

    total_area = float(
        cumulative[-1]
    )

    if not np.isfinite(total_area) or total_area <= 0:
        return None

    normalized = (
        cumulative
        / total_area
    )

    result = {
        "baseline": baseline,
        "event_area": total_area,
    }

    for q in quantiles:
        result[
            f"q{int(q * 100):02d}"
        ] = float(
            np.interp(
                q,
                normalized,
                grid,
            )
        )

    result["event_width_q75_q25"] = (
        result["q75"]
        - result["q25"]
    )

    return result


def compute_region_timing(
    parameters,
    scores,
    region_lower,
    region_upper,
):
    grid, values = interpolate_region(
        parameters,
        scores,
        region_lower,
        region_upper,
    )

    return area_quantiles(
        grid,
        values,
    )


# =====================================================================
# BOOTSTRAP
# =====================================================================

def make_bootstrap_indices(
    ordered_indices,
    rng,
):
    boot = []

    n = len(ordered_indices)

    for start in range(
        0,
        n,
        BOOTSTRAP_STRATUM_SIZE,
    ):
        stop = min(
            start + BOOTSTRAP_STRATUM_SIZE,
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

        sampled = np.sort(sampled)

        boot.extend(
            sampled.tolist()
        )

    return np.asarray(
        boot,
        dtype=int,
    )


def bootstrap_region_replicate(
    X_rna_z,
    X_atac_z,
    pseudotime,
    ordered_indices,
    critical_value,
    region_lower,
    region_upper,
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

    rna_timing = compute_region_timing(
        rna_x,
        rna_y,
        region_lower,
        region_upper,
    )

    atac_timing = compute_region_timing(
        atac_x,
        atac_y,
        region_lower,
        region_upper,
    )

    if rna_timing is None or atac_timing is None:
        return {
            "valid": False,
        }

    return {
        "valid": True,

        "rna_q25": rna_timing["q25"],
        "rna_q50": rna_timing["q50"],
        "rna_q75": rna_timing["q75"],
        "rna_width": rna_timing[
            "event_width_q75_q25"
        ],
        "rna_area": rna_timing[
            "event_area"
        ],

        "atac_q25": atac_timing["q25"],
        "atac_q50": atac_timing["q50"],
        "atac_q75": atac_timing["q75"],
        "atac_width": atac_timing[
            "event_width_q75_q25"
        ],
        "atac_area": atac_timing[
            "event_area"
        ],

        "early_lead_q25": (
            rna_timing["q25"]
            - atac_timing["q25"]
        ),

        "center_lead_q50": (
            rna_timing["q50"]
            - atac_timing["q50"]
        ),

        "late_lead_q75": (
            rna_timing["q75"]
            - atac_timing["q75"]
        ),
    }


# =====================================================================
# SUMMARY
# =====================================================================

def percentile_summary(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    return {
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
        "q2.5": float(
            np.quantile(
                values,
                0.025,
            )
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
        "q97.5": float(
            np.quantile(
                values,
                0.975,
            )
        ),
        "p_gt_zero": float(
            np.mean(
                values > 0
            )
        ),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F4c — EXPLORATORY EVENT-REGION TIMING DIAGNOSTICS"
    )

    print(
        f"Dataset: {DATA_FILE.resolve()}"
    )
    print()
    print(
        "PRIMARY PEAK-BASED PHASE F4 RESULT REMAINS:"
    )
    print(
        "  INCONCLUSIVE."
    )
    print()
    print(
        "This script evaluates an exploratory region-based timing "
        "descriptor because Phase F4b demonstrated broad/multipeak events."
    )
    print()
    print(
        "No confirmatory priming claim will be made from this phase."
    )
    print(
        "No files are created."
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
    # Trajectory / frozen centers / shared region
    # -----------------------------------------------------------------

    section(
        "2. REPRODUCE FROZEN EVENT AND DEFINE SHARED REGION"
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

    region_lower = max(
        0.0,
        frozen_atac_center
        - FAMILY_MATCH_TOLERANCE,
    )

    region_upper = min(
        1.0,
        frozen_rna_center
        + FAMILY_MATCH_TOLERANCE,
    )

    print(
        f"Frozen ATAC family center: "
        f"{frozen_atac_center:.6f}"
    )
    print(
        f"Frozen RNA family center:  "
        f"{frozen_rna_center:.6f}"
    )
    print(
        f"Frozen tracking tolerance: "
        f"+/- {FAMILY_MATCH_TOLERANCE:.3f}"
    )
    print()
    print(
        f"SHARED event-region interval for BOTH modalities:"
    )
    print(
        f"  [{region_lower:.6f}, {region_upper:.6f}]"
    )
    print(
        f"  width = {region_upper - region_lower:.6f}"
    )

    # -----------------------------------------------------------------
    # Observed region timing across all three window designs
    # -----------------------------------------------------------------

    section(
        "3. OBSERVED REGION-BASED TIMING ACROSS WINDOW DESIGNS"
    )

    observed_rows = []

    for (
        design_name,
        window_size,
        step_size,
    ) in WINDOW_DESIGNS:

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

        rna_timing = compute_region_timing(
            rna_x,
            rna_y,
            region_lower,
            region_upper,
        )

        atac_timing = compute_region_timing(
            atac_x,
            atac_y,
            region_lower,
            region_upper,
        )

        if (
            rna_timing is None
            or atac_timing is None
        ):
            raise RuntimeError(
                f"Region timing failed for {design_name}."
            )

        observed_rows.append(
            {
                "window_design": design_name,

                "rna_q25": rna_timing["q25"],
                "rna_q50": rna_timing["q50"],
                "rna_q75": rna_timing["q75"],
                "rna_width": rna_timing[
                    "event_width_q75_q25"
                ],

                "atac_q25": atac_timing["q25"],
                "atac_q50": atac_timing["q50"],
                "atac_q75": atac_timing["q75"],
                "atac_width": atac_timing[
                    "event_width_q75_q25"
                ],

                "early_lead_q25": (
                    rna_timing["q25"]
                    - atac_timing["q25"]
                ),

                "center_lead_q50": (
                    rna_timing["q50"]
                    - atac_timing["q50"]
                ),

                "late_lead_q75": (
                    rna_timing["q75"]
                    - atac_timing["q75"]
                ),
            }
        )

    observed = pd.DataFrame(
        observed_rows
    ).set_index(
        "window_design"
    )

    print_df(
        observed
    )

    # -----------------------------------------------------------------
    # Bootstrap
    # -----------------------------------------------------------------

    section(
        "4. PAIRED BOOTSTRAP OF REGION-BASED TIMING"
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
        row = bootstrap_region_replicate(
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

            n_valid = int(
                current["valid"].sum()
            )

            print(
                f"Completed {b:4d}/{N_BOOTSTRAPS} | "
                f"valid region timing "
                f"{n_valid:4d}/{b:4d}"
            )

    bootstrap = pd.DataFrame(
        records
    )

    valid = bootstrap.loc[
        bootstrap["valid"]
    ].copy()

    validity_rate = (
        len(valid)
        / N_BOOTSTRAPS
    )

    print()
    print(
        f"Valid region-based timing replicates: "
        f"{len(valid)}/{N_BOOTSTRAPS}"
    )
    print(
        f"Validity rate: {validity_rate:.4f}"
    )

    if len(valid) < 20:
        print(
            "ERROR: Too few valid region timing replicates."
        )
        sys.exit(1)

    # -----------------------------------------------------------------
    # Timing uncertainty
    # -----------------------------------------------------------------

    section(
        "5. EXPLORATORY REGION-TIMING BOOTSTRAP SUMMARY"
    )

    metrics = [
        "rna_q25",
        "atac_q25",
        "early_lead_q25",
        "rna_q50",
        "atac_q50",
        "center_lead_q50",
        "rna_q75",
        "atac_q75",
        "late_lead_q75",
        "rna_width",
        "atac_width",
    ]

    summary_rows = []

    for metric in metrics:
        stats = percentile_summary(
            valid[metric]
        )

        stats[
            "metric"
        ] = metric

        summary_rows.append(
            stats
        )

    summary = pd.DataFrame(
        summary_rows
    ).set_index(
        "metric"
    )

    print_df(
        summary
    )

    # -----------------------------------------------------------------
    # Interpretation guardrails
    # -----------------------------------------------------------------

    section(
        "6. PHASE F4c INTERPRETATION"
    )

    center = percentile_summary(
        valid[
            "center_lead_q50"
        ]
    )

    early = percentile_summary(
        valid[
            "early_lead_q25"
        ]
    )

    late = percentile_summary(
        valid[
            "late_lead_q75"
        ]
    )

    print(
        "This phase is EXPLORATORY ONLY."
    )
    print()
    print(
        "Observed/bootstrapped region descriptors can determine whether "
        "the broad instability mass tends to be shifted between modalities, "
        "but they do not rescue or replace the inconclusive primary "
        "peak-based CMIL result."
    )
    print()
    print(
        "If the region-based lead is stable, freeze this estimator here "
        "and test it next in simulation/null analyses and external data "
        "before interpreting it as a biological chromatin-priming measure."
    )

    print()
    print(
        "Exploratory directional support:"
    )
    print(
        f"  Q25 early lead:  P(lead>0) = "
        f"{early['p_gt_zero']:.4f}"
    )
    print(
        f"  Q50 center lead: P(lead>0) = "
        f"{center['p_gt_zero']:.4f}"
    )
    print(
        f"  Q75 late lead:   P(lead>0) = "
        f"{late['p_gt_zero']:.4f}"
    )

    print()
    print(
        "PHASE F4c VERDICT: EXPLORATORY DESCRIPTOR ONLY"
    )
    print()
    print(
        "No confirmatory CMIL conclusion is changed."
    )
    print(
        "No priming claim is made."
    )
    print(
        "No files are created."
    )

    line("=")


if __name__ == "__main__":
    main()

