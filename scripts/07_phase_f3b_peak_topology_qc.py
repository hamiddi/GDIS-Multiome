#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
07_phase_f3b_peak_topology_qc.py
=================================

Project:
Cross-Modal Dynamical Instability Reveals Chromatin Priming
before Transcriptional Cell-Fate Transitions

Phase F3b:
Identify reproducible instability-event families before any CMIL calculation.

WHY THIS PHASE IS NECESSARY
---------------------------
Phase F3 showed that the RNA and ATAC sustained-GDIS curves are globally
similar across window designs, but the GLOBAL ARGMAX switches between early
and late local maxima.

Therefore:
    global_peak_RNA - global_peak_ATAC
is not yet a defensible CMIL definition.

This phase instead:
1. reconstructs the accepted common RNA-only pseudotime;
2. recomputes the sustained-GDIS (lambda_t = 0) curves;
3. detects prominent LOCAL peaks within each window design;
4. clusters peaks into persistent within-modality peak families;
5. labels each persistent family by the nearest independently defined
   beta-lineage transition landmark;
6. identifies which biological transition labels are represented in BOTH
   RNA and ATAC.

It does NOT calculate RNA-ATAC timing differences.

Primary representations
-----------------------
RNA  = X_scVI, 10D
ATAC = X_poissonvi, 10D

Window designs
--------------
300/75
400/100
500/125

Peak detection
--------------
A local peak must have prominence >= 5% of that curve's dynamic range.

Peak-family persistence
-----------------------
Peaks from different window designs are grouped when their pseudotime
locations are within 0.08 of the evolving cluster center.

A robust family must:
    - appear in all 3 prespecified window designs;
    - have total pseudotime spread <= 0.10;
    - have median relative prominence >= 0.05.

No GDIS timing difference / CMIL is calculated.
No bootstrap is performed.
No files are written.

Run:
    python 07_phase_f3b_peak_topology_qc.py
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

REFERENCE_TRANSITION_WEIGHT = 0.18
PRIMARY_TRANSITION_WEIGHT = 0.0

# Local-peak detection.
MIN_RELATIVE_PROMINENCE = 0.05
MIN_PEAK_DISTANCE_WINDOWS = 3

# Peak-family clustering/persistence.
FAMILY_MATCH_TOLERANCE = 0.08
MAX_ROBUST_FAMILY_SPREAD = 0.10
REQUIRED_WINDOW_DESIGNS = 3


# =====================================================================
# DISPLAY
# =====================================================================

def line(char="=", width=104):
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


def print_df(df, digits=5):
    if df.empty:
        print("(empty)")
        return

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 280,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string(index=True))


# =====================================================================
# HELPERS
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
# WINDOWS + GDIS
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


def sustained_gdis_curve(trajectories, parameters, critical_value):
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
# BIOLOGICAL LANDMARKS
# =====================================================================

def state_medians(pseudotime, obs):
    result = {}

    for state in BETA_LINEAGE:
        mask = (obs["cell_type"].astype(str) == state).to_numpy()
        values = pseudotime[mask]
        values = values[np.isfinite(values)]

        result[state] = float(np.median(values))

    return result


def transition_landmarks(medians):
    """
    Independent transition landmarks are midpoints between adjacent
    annotated state medians.
    """
    rows = []

    for left, right in zip(
        BETA_LINEAGE[:-1],
        BETA_LINEAGE[1:],
    ):
        value = 0.5 * (
            medians[left] + medians[right]
        )

        rows.append(
            {
                "transition": f"{left} -> {right}",
                "landmark_pt": value,
            }
        )

    return pd.DataFrame(rows)


def nearest_transition(pt, landmarks):
    distances = np.abs(
        landmarks["landmark_pt"].to_numpy(dtype=float)
        - float(pt)
    )

    i = int(np.argmin(distances))

    return (
        str(landmarks.iloc[i]["transition"]),
        float(landmarks.iloc[i]["landmark_pt"]),
        float(distances[i]),
    )


# =====================================================================
# PEAK DETECTION
# =====================================================================

def detect_local_peaks(parameters, scores, modality, design_name):
    dynamic_range = float(
        np.max(scores) - np.min(scores)
    )

    if dynamic_range <= 0:
        return pd.DataFrame()

    absolute_prominence = (
        MIN_RELATIVE_PROMINENCE
        * dynamic_range
    )

    indices, properties = find_peaks(
        scores,
        prominence=absolute_prominence,
        distance=MIN_PEAK_DISTANCE_WINDOWS,
    )

    if len(indices) == 0:
        return pd.DataFrame()

    # Recompute explicitly for clarity/robustness.
    prominences = peak_prominences(
        scores,
        indices,
    )[0]

    rows = []

    global_index = int(np.argmax(scores))

    for i, prominence in zip(
        indices,
        prominences,
    ):
        rows.append(
            {
                "modality": modality,
                "window_design": design_name,
                "peak_index": int(i),
                "peak_pt": float(parameters[i]),
                "peak_score": float(scores[i]),
                "prominence": float(prominence),
                "relative_prominence": float(
                    prominence / dynamic_range
                ),
                "is_global_maximum": bool(i == global_index),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# PEAK-FAMILY CLUSTERING
# =====================================================================

def cluster_peak_families(peaks):
    """
    Greedy 1D clustering of local peaks by pseudotime.

    The cluster center is updated as the median peak location.
    A new peak joins the closest existing cluster if it lies within
    FAMILY_MATCH_TOLERANCE; otherwise a new family is created.
    """
    if peaks.empty:
        return []

    peaks = peaks.sort_values("peak_pt").reset_index(drop=True)

    clusters = []

    for _, row in peaks.iterrows():
        pt = float(row["peak_pt"])

        if not clusters:
            clusters.append([row.to_dict()])
            continue

        centers = np.array(
            [
                np.median(
                    [item["peak_pt"] for item in cluster]
                )
                for cluster in clusters
            ],
            dtype=float,
        )

        distances = np.abs(centers - pt)
        closest = int(np.argmin(distances))

        if distances[closest] <= FAMILY_MATCH_TOLERANCE:
            clusters[closest].append(
                row.to_dict()
            )
        else:
            clusters.append(
                [row.to_dict()]
            )

    return clusters


def summarize_families(
    modality,
    clusters,
    landmarks,
):
    rows = []

    family_id = 0

    for cluster in clusters:
        df = pd.DataFrame(cluster)

        unique_designs = sorted(
            df["window_design"]
            .astype(str)
            .unique()
            .tolist()
        )

        median_pt = float(
            df["peak_pt"].median()
        )

        min_pt = float(
            df["peak_pt"].min()
        )

        max_pt = float(
            df["peak_pt"].max()
        )

        spread = max_pt - min_pt

        median_prominence = float(
            df["relative_prominence"].median()
        )

        transition, landmark_pt, distance = nearest_transition(
            median_pt,
            landmarks,
        )

        robust = (
            len(unique_designs) == REQUIRED_WINDOW_DESIGNS
            and spread <= MAX_ROBUST_FAMILY_SPREAD
            and median_prominence >= MIN_RELATIVE_PROMINENCE
        )

        rows.append(
            {
                "modality": modality,
                "family_id": family_id,
                "n_peaks": len(df),
                "n_window_designs": len(unique_designs),
                "window_designs": ",".join(unique_designs),
                "median_peak_pt": median_pt,
                "min_peak_pt": min_pt,
                "max_peak_pt": max_pt,
                "peak_pt_spread": spread,
                "median_relative_prominence": median_prominence,
                "contains_global_maximum": bool(
                    df["is_global_maximum"].any()
                ),
                "nearest_transition": transition,
                "transition_landmark_pt": landmark_pt,
                "distance_to_transition": distance,
                "robust": robust,
            }
        )

        family_id += 1

    return pd.DataFrame(rows)


# =====================================================================
# MAIN
# =====================================================================

def main():
    section("PHASE F3b — PERSISTENT INSTABILITY-EVENT TOPOLOGY")

    print(f"Dataset: {DATA_FILE.resolve()}")
    print()
    print("Purpose:")
    print("  Resolve global-peak switching before any CMIL calculation.")
    print()
    print("No RNA-ATAC timing difference is calculated.")
    print("No bootstrap is performed.")
    print("No files are created.")

    if not DATA_FILE.exists():
        print("\nERROR: Dataset file not found.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    section("1. LOAD DATASET")

    try:
        mdata = mu.read_h5mu(DATA_FILE, backed="r")
    except TypeError:
        mdata = mu.read_h5mu(DATA_FILE)

    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]
    obs_full = rna.obs.copy()

    beta_mask = (
        obs_full["cell_type"]
        .astype(str)
        .isin(BETA_LINEAGE)
    ).to_numpy()

    obs = (
        obs_full.loc[beta_mask]
        .copy()
        .reset_index(drop=True)
    )

    obs.index = pd.Index(
        [f"beta_{i}" for i in range(len(obs))]
    )

    X_rna = as_numpy(
        rna.obsm[RNA_REPRESENTATION]
    )[beta_mask, :N_LATENT_DIMS]

    X_atac = as_numpy(
        atac.obsm[ATAC_REPRESENTATION]
    )[beta_mask, :N_LATENT_DIMS]

    X_rna_z = zscore_columns(X_rna)
    X_atac_z = zscore_columns(X_atac)

    print(f"Cells: {len(obs):,}")
    print(f"RNA shape:  {X_rna.shape}")
    print(f"ATAC shape: {X_atac.shape}")

    # -----------------------------------------------------------------
    # Common pseudotime
    # -----------------------------------------------------------------

    section("2. COMMON RNA-ONLY PSEUDOTIME")

    pseudotime = reconstruct_common_pseudotime(
        X_rna,
        obs,
    )

    finite = float(
        np.isfinite(pseudotime).mean()
    )

    print(f"Finite pseudotime fraction: {finite:.6f}")

    if finite < 0.99:
        print("ERROR: Phase-F2 trajectory was not reproduced.")
        sys.exit(1)

    medians = state_medians(
        pseudotime,
        obs,
    )

    medians_df = pd.DataFrame(
        [
            {
                "cell_type": state,
                "median_pseudotime": medians[state],
            }
            for state in BETA_LINEAGE
        ]
    ).set_index("cell_type")

    print()
    print_df(medians_df)

    # -----------------------------------------------------------------
    # Independent transition landmarks
    # -----------------------------------------------------------------

    section("3. INDEPENDENT BIOLOGICAL TRANSITION LANDMARKS")

    landmarks = transition_landmarks(
        medians
    )

    print_df(
        landmarks.set_index("transition")
    )

    critical_value = float(
        landmarks.loc[
            landmarks["transition"]
            == "Fev+ -> Fev+ Beta",
            "landmark_pt",
        ].iloc[0]
    )

    print()
    print(
        f"Reference pyGDIS critical value remains: "
        f"{critical_value:.6f}"
    )

    # -----------------------------------------------------------------
    # Recompute sustained curves and detect peaks
    # -----------------------------------------------------------------

    section("4. LOCAL PEAK DETECTION")

    all_peaks = []

    for design_name, window_size, step_size in WINDOW_DESIGNS:
        subsection(
            f"{design_name}: {window_size}/{step_size}"
        )

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

        rna_peaks = detect_local_peaks(
            rna_x,
            rna_y,
            "RNA",
            design_name,
        )

        atac_peaks = detect_local_peaks(
            atac_x,
            atac_y,
            "ATAC",
            design_name,
        )

        all_peaks.append(rna_peaks)
        all_peaks.append(atac_peaks)

        print(
            f"RNA local peaks meeting prominence threshold: "
            f"{len(rna_peaks)}"
        )
        print(
            f"ATAC local peaks meeting prominence threshold: "
            f"{len(atac_peaks)}"
        )

    peaks = pd.concat(
        all_peaks,
        ignore_index=True,
    )

    print()
    print("All detected local peaks:")
    print_df(
        peaks.set_index(
            ["modality", "window_design", "peak_index"]
        )
    )

    # -----------------------------------------------------------------
    # Cluster into within-modality persistent families
    # -----------------------------------------------------------------

    section("5. WITHIN-MODALITY PEAK FAMILIES")

    family_tables = []

    for modality in ["RNA", "ATAC"]:
        subsection(modality)

        modality_peaks = peaks.loc[
            peaks["modality"] == modality
        ].copy()

        clusters = cluster_peak_families(
            modality_peaks
        )

        families = summarize_families(
            modality,
            clusters,
            landmarks,
        )

        family_tables.append(
            families
        )

        print_df(
            families.set_index("family_id")
        )

    families = pd.concat(
        family_tables,
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # Robust families only
    # -----------------------------------------------------------------

    section("6. ROBUST PEAK FAMILIES")

    robust = families.loc[
        families["robust"]
    ].copy()

    if robust.empty:
        print("No robust peak family survived the prespecified criteria.")
    else:
        print_df(
            robust.set_index(
                ["modality", "family_id"]
            )
        )

    # -----------------------------------------------------------------
    # Cross-modal event labels WITHOUT timing subtraction
    # -----------------------------------------------------------------

    section("7. BIOLOGICAL EVENT LABELS REPRESENTED IN BOTH MODALITIES")

    robust_rna = robust.loc[
        robust["modality"] == "RNA"
    ]

    robust_atac = robust.loc[
        robust["modality"] == "ATAC"
    ]

    rna_labels = set(
        robust_rna["nearest_transition"]
        .astype(str)
        .tolist()
    )

    atac_labels = set(
        robust_atac["nearest_transition"]
        .astype(str)
        .tolist()
    )

    shared_labels = sorted(
        rna_labels & atac_labels
    )

    print("RNA robust transition labels:")
    for label in sorted(rna_labels):
        print(f"  {label}")

    print()
    print("ATAC robust transition labels:")
    for label in sorted(atac_labels):
        print(f"  {label}")

    print()
    print("Shared labels eligible for later event-specific CMIL:")
    if shared_labels:
        for label in shared_labels:
            print(f"  {label}")
    else:
        print("  None")

    print()
    print(
        "No RNA-ATAC pseudotime difference is calculated in this phase."
    )

    # -----------------------------------------------------------------
    # Decision
    # -----------------------------------------------------------------

    section("8. PHASE F3b DECISION")

    n_rna_robust = len(robust_rna)
    n_atac_robust = len(robust_atac)
    n_shared = len(shared_labels)

    print(f"Robust RNA peak families:  {n_rna_robust}")
    print(f"Robust ATAC peak families: {n_atac_robust}")
    print(f"Shared biological labels:  {n_shared}")
    print()

    if (
        n_rna_robust >= 1
        and n_atac_robust >= 1
        and n_shared >= 1
    ):
        if n_rna_robust > 1 or n_atac_robust > 1:
            verdict = "GO — EVENT-SPECIFIC CMIL"
            reason = (
                "The instability landscape is multi-peak. "
                "A single global CMIL is not appropriate. "
                "Proceed only with event-specific CMIL for robust "
                "RNA/ATAC peak families sharing the same independent "
                "biological transition label."
            )
        else:
            verdict = "GO — SINGLE EVENT"
            reason = (
                "Each modality has one robust persistent peak family "
                "with a shared biological transition label."
            )

    elif (
        n_rna_robust >= 1
        and n_atac_robust >= 1
        and n_shared == 0
    ):
        verdict = "REVIEW REQUIRED"
        reason = (
            "RNA and ATAC each show persistent events, but they map to "
            "different biological transition labels. Direct CMIL pairing "
            "would not yet be justified."
        )

    else:
        verdict = "REVIEW REQUIRED"
        reason = (
            "At least one modality lacks a persistent peak family across "
            "all three window designs."
        )

    print(f"PHASE F3b VERDICT: {verdict}")
    print()
    print(reason)

    print()
    print("No CMIL was calculated.")
    print("No bootstrap was performed.")
    print("No output files were created.")

    if verdict.startswith("GO"):
        print()
        print(
            "Next step: 08_phase_f4_event_specific_cmil_bootstrap.py"
        )

    line("=")


if __name__ == "__main__":
    main()

