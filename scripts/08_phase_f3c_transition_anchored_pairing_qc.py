#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
08_phase_f3c_transition_anchored_pairing_qc.py
==============================================

Project:
Cross-Modal Dynamical Instability Reveals Chromatin Priming
before Transcriptional Cell-Fate Transitions

Phase F3c:
Prospectively define and test a transition-anchored RNA/ATAC event-pairing rule
BEFORE calculating any cross-modal timing difference.

WHY THIS PHASE EXISTS
---------------------
Phase F3b showed:
- one robust RNA instability family;
- three robust ATAC instability families;
- no shared "nearest midpoint" transition label.

That does NOT automatically invalidate a priming model, because a chromatin
priming event should occur upstream of the transcriptional transition and can
therefore be closest to the preceding biological landmark.

This phase uses a directional, biologically anchored rule:

1. Assign each ROBUST RNA peak family to the adjacent-state interval in which
   its median peak lies:
       [median(state_i), median(state_i+1)]

2. For that RNA event, define an upstream ATAC eligibility interval:
       [median(state_i-1), RNA_peak)
   for internal transitions.

   For the first transition, use:
       [minimum pseudotime, RNA_peak)

3. Among robust ATAC peak families in that interval, choose the PROXIMAL
   preceding family: the ATAC family with the largest median peak pseudotime
   that is still earlier than the RNA family.

4. Directional consistency requirement:
   in every prespecified window design where both families are present,
   ATAC_peak_pt must be < RNA_peak_pt.

This phase identifies eligible event pairs only.
It does NOT subtract RNA and ATAC peak times.
It does NOT calculate CMIL.
It does NOT bootstrap.
It writes no files.

Primary representations:
    RNA  = X_scVI, 10D
    ATAC = X_poissonvi, 10D

Window designs:
    300/75
    400/100
    500/125

Run:
    python 08_phase_f3c_transition_anchored_pairing_qc.py
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

PRIMARY_TRANSITION_WEIGHT = 0.0
REFERENCE_TRANSITION_WEIGHT = 0.18

MIN_RELATIVE_PROMINENCE = 0.05
MIN_PEAK_DISTANCE_WINDOWS = 3

FAMILY_MATCH_TOLERANCE = 0.08
MAX_ROBUST_FAMILY_SPREAD = 0.10
REQUIRED_WINDOW_DESIGNS = 3


# =====================================================================
# DISPLAY
# =====================================================================

def line(char="=", width=108):
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
        "display.width", 300,
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
# BIOLOGICAL STATE MEDIANS AND TRANSITIONS
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


def make_transition_table(medians, pseudotime):
    """
    Define biological intervals from adjacent annotated-state medians.

    RNA transition interval for S_i -> S_i+1:
        [median(S_i), median(S_i+1)]

    Upstream ATAC eligibility lower bound:
        median(S_i-1), for i > 0
        minimum pseudotime, for i = 0

    The ATAC upper bound is set later to the observed robust RNA peak location.
    """
    finite_pt = pseudotime[np.isfinite(pseudotime)]
    pt_min = float(np.min(finite_pt))

    rows = []

    for i in range(len(BETA_LINEAGE) - 1):
        left = BETA_LINEAGE[i]
        right = BETA_LINEAGE[i + 1]

        if i == 0:
            upstream_lower = pt_min
        else:
            upstream_lower = medians[
                BETA_LINEAGE[i - 1]
            ]

        rows.append(
            {
                "transition_id": i,
                "transition": f"{left} -> {right}",
                "left_state": left,
                "right_state": right,
                "rna_interval_lower": medians[left],
                "rna_interval_upper": medians[right],
                "atac_upstream_lower": upstream_lower,
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# LOCAL PEAKS AND ROBUST FAMILIES
# =====================================================================

def detect_local_peaks(parameters, scores, modality, design_name):
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

    global_index = int(np.argmax(scores))

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
                "is_global_maximum": bool(
                    index == global_index
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
                        x["peak_pt"]
                        for x in cluster
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
                "n_peaks": len(df),
                "n_window_designs": len(designs),
                "median_peak_pt": median_pt,
                "peak_pt_spread": spread,
                "median_relative_prominence": median_prominence,
                "contains_global_maximum": bool(
                    df["is_global_maximum"].any()
                ),
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


# =====================================================================
# TRANSITION-ANCHORED PAIRING
# =====================================================================

def assign_rna_family_to_transition(
    family_row,
    transitions,
):
    """
    Assign a robust RNA family only if its median peak lies inside exactly one
    adjacent-state median interval.
    """
    pt = float(
        family_row["median_peak_pt"]
    )

    eligible = transitions.loc[
        (transitions["rna_interval_lower"] <= pt)
        & (pt <= transitions["rna_interval_upper"])
    ]

    if len(eligible) != 1:
        return None

    return eligible.iloc[0]


def find_proximal_atac_family(
    rna_peak_pt,
    transition_row,
    robust_atac,
):
    """
    Find the closest robust ATAC family preceding the RNA family within the
    biologically defined upstream eligibility interval.
    """
    lower = float(
        transition_row["atac_upstream_lower"]
    )

    candidates = robust_atac.loc[
        (robust_atac["median_peak_pt"] >= lower)
        & (robust_atac["median_peak_pt"] < rna_peak_pt)
    ].copy()

    if candidates.empty:
        return None, candidates

    candidates["distance_to_rna_without_subtraction_claim"] = (
        rna_peak_pt
        - candidates["median_peak_pt"]
    )

    # Proximal = latest preceding event.
    selected_index = (
        candidates["median_peak_pt"]
        .idxmax()
    )

    return (
        candidates.loc[selected_index],
        candidates,
    )


def directional_consistency(
    rna_family_id,
    atac_family_id,
    rna_members,
    atac_members,
):
    """
    Check ATAC < RNA separately within every shared window design.

    No time differences are returned.
    """
    rna = rna_members.loc[
        rna_members["family_id"]
        == rna_family_id
    ].copy()

    atac = atac_members.loc[
        atac_members["family_id"]
        == atac_family_id
    ].copy()

    merged = rna.merge(
        atac,
        on="window_design",
        suffixes=("_rna", "_atac"),
    )

    if merged.empty:
        return False, pd.DataFrame()

    merged["atac_precedes_rna"] = (
        merged["peak_pt_atac"]
        < merged["peak_pt_rna"]
    )

    consistent = bool(
        merged["atac_precedes_rna"].all()
        and len(merged) == REQUIRED_WINDOW_DESIGNS
    )

    return consistent, merged[
        [
            "window_design",
            "peak_pt_atac",
            "peak_pt_rna",
            "atac_precedes_rna",
        ]
    ]


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F3c — TRANSITION-ANCHORED EVENT PAIRING QC"
    )

    print(f"Dataset: {DATA_FILE.resolve()}")
    print()
    print(
        "Goal: define defensible RNA/ATAC event pairs "
        "before calculating any timing difference."
    )
    print()
    print("No CMIL is calculated.")
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
    # Common trajectory and transition intervals
    # -----------------------------------------------------------------

    section("2. BIOLOGICAL TRANSITION INTERVALS")

    pseudotime = reconstruct_common_pseudotime(
        X_rna,
        obs,
    )

    if np.isfinite(pseudotime).mean() < 0.99:
        print("ERROR: Phase-F2 trajectory did not reproduce.")
        sys.exit(1)

    medians = get_state_medians(
        pseudotime,
        obs,
    )

    transition_table = make_transition_table(
        medians,
        pseudotime,
    )

    print_df(
        transition_table.set_index(
            "transition"
        )
    )

    critical_value = 0.5 * (
        medians["Fev+"]
        + medians["Fev+ Beta"]
    )

    # -----------------------------------------------------------------
    # Recompute robust families
    # -----------------------------------------------------------------

    section("3. RECOMPUTE ROBUST PEAK FAMILIES")

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

    robust_rna = rna_families.loc[
        rna_families["robust"]
    ].copy()

    robust_atac = atac_families.loc[
        atac_families["robust"]
    ].copy()

    print("Robust RNA families:")
    print_df(
        robust_rna.set_index("family_id")
    )

    print()
    print("Robust ATAC families:")
    print_df(
        robust_atac.set_index("family_id")
    )

    # -----------------------------------------------------------------
    # Transition-anchor robust RNA events
    # -----------------------------------------------------------------

    section("4. ASSIGN ROBUST RNA EVENTS TO BIOLOGICAL TRANSITIONS")

    assignments = []

    for _, rna_family in robust_rna.iterrows():
        transition = assign_rna_family_to_transition(
            rna_family,
            transition_table,
        )

        if transition is None:
            assignments.append(
                {
                    "rna_family_id": int(
                        rna_family["family_id"]
                    ),
                    "rna_median_peak_pt": float(
                        rna_family["median_peak_pt"]
                    ),
                    "transition": "UNASSIGNED",
                }
            )
        else:
            assignments.append(
                {
                    "rna_family_id": int(
                        rna_family["family_id"]
                    ),
                    "rna_median_peak_pt": float(
                        rna_family["median_peak_pt"]
                    ),
                    "transition": str(
                        transition["transition"]
                    ),
                    "rna_interval_lower": float(
                        transition[
                            "rna_interval_lower"
                        ]
                    ),
                    "rna_interval_upper": float(
                        transition[
                            "rna_interval_upper"
                        ]
                    ),
                    "atac_upstream_lower": float(
                        transition[
                            "atac_upstream_lower"
                        ]
                    ),
                }
            )

    assignment_df = pd.DataFrame(
        assignments
    )

    print_df(
        assignment_df.set_index(
            "rna_family_id"
        )
    )

    # -----------------------------------------------------------------
    # Prospectively pair with proximal preceding ATAC event
    # -----------------------------------------------------------------

    section("5. PROXIMAL UPSTREAM ATAC PAIRING")

    pair_rows = []
    consistency_tables = []

    for _, assignment in assignment_df.iterrows():
        if (
            assignment["transition"]
            == "UNASSIGNED"
        ):
            continue

        transition_row = transition_table.loc[
            transition_table["transition"]
            == assignment["transition"]
        ].iloc[0]

        rna_family_id = int(
            assignment["rna_family_id"]
        )

        rna_peak_pt = float(
            assignment["rna_median_peak_pt"]
        )

        selected_atac, candidates = find_proximal_atac_family(
            rna_peak_pt,
            transition_row,
            robust_atac,
        )

        print()
        print(
            f"RNA family {rna_family_id}: "
            f"{assignment['transition']}"
        )
        print(
            f"RNA family median peak: {rna_peak_pt:.6f}"
        )
        print(
            "Eligible robust ATAC families in the prospectively "
            "defined upstream interval:"
        )

        if candidates.empty:
            print("  None")

            pair_rows.append(
                {
                    "transition": assignment["transition"],
                    "rna_family_id": rna_family_id,
                    "atac_family_id": np.nan,
                    "pair_found": False,
                    "directionally_consistent_all_designs": False,
                }
            )

            continue

        print_df(
            candidates[
                [
                    "family_id",
                    "median_peak_pt",
                    "peak_pt_spread",
                    "median_relative_prominence",
                ]
            ].set_index("family_id")
        )

        atac_family_id = int(
            selected_atac["family_id"]
        )

        atac_peak_pt = float(
            selected_atac["median_peak_pt"]
        )

        print()
        print(
            "Selected proximal preceding ATAC family: "
            f"{atac_family_id}"
        )
        print(
            f"ATAC family median peak: {atac_peak_pt:.6f}"
        )

        consistent, detail = directional_consistency(
            rna_family_id,
            atac_family_id,
            rna_members,
            atac_members,
        )

        print()
        print(
            "Window-design directional consistency "
            "(ATAC peak earlier than RNA peak):"
        )

        if detail.empty:
            print("  Not evaluable")
        else:
            print_df(
                detail.set_index(
                    "window_design"
                )
            )

            detail_copy = detail.copy()
            detail_copy["transition"] = (
                assignment["transition"]
            )
            detail_copy["rna_family_id"] = (
                rna_family_id
            )
            detail_copy["atac_family_id"] = (
                atac_family_id
            )

            consistency_tables.append(
                detail_copy
            )

        pair_rows.append(
            {
                "transition": assignment["transition"],
                "rna_family_id": rna_family_id,
                "atac_family_id": atac_family_id,
                "pair_found": True,
                "directionally_consistent_all_designs": consistent,
            }
        )

    pairs = pd.DataFrame(
        pair_rows
    )

    # -----------------------------------------------------------------
    # Decision
    # -----------------------------------------------------------------

    section("6. PHASE F3c DECISION")

    if pairs.empty:
        print("No transition-anchored candidate pair could be evaluated.")
        verdict = "REVIEW REQUIRED"
        reason = (
            "No robust RNA event was eligible for directional "
            "transition-anchored pairing."
        )

    else:
        print_df(
            pairs.set_index(
                "transition"
            ),
            digits=0,
        )

        valid_pairs = pairs.loc[
            pairs["pair_found"]
            & pairs[
                "directionally_consistent_all_designs"
            ]
        ]

        print()
        print(
            f"Directionally consistent candidate pairs: "
            f"{len(valid_pairs)}"
        )

        if len(valid_pairs) >= 1:
            verdict = "GO — FREEZE EVENT PAIR BEFORE BOOTSTRAP"
            reason = (
                "At least one robust RNA instability event has a "
                "prospectively selected proximal upstream ATAC family, "
                "and ATAC precedes RNA in all three prespecified "
                "window designs. Freeze this pairing rule before "
                "estimating timing differences or uncertainty."
            )
        else:
            verdict = "REVIEW REQUIRED"
            reason = (
                "No candidate pair satisfies directional consistency "
                "across all prespecified window designs."
            )

    print()
    print(f"PHASE F3c VERDICT: {verdict}")
    print()
    print(reason)

    print()
    print("No RNA-ATAC timing difference was calculated.")
    print("No CMIL was calculated.")
    print("No bootstrap was performed.")
    print("No output files were created.")

    if verdict.startswith("GO"):
        print()
        print(
            "Next step: "
            "09_phase_f4_event_specific_cmil_bootstrap.py"
        )

    line("=")


if __name__ == "__main__":
    main()

