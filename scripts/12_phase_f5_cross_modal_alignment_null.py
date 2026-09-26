#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
12_phase_f5_cross_modal_alignment_null.py
==========================================

Phase F5 — exploratory cross-modal temporal-alignment null test.

FROZEN PRIOR RESULTS
--------------------
1. Peak-based event-specific CMIL is INCONCLUSIVE.
2. Region-based timing does NOT support a robust ATAC-leading shift.

Therefore this script does NOT test chromatin priming.

QUESTION
--------
Within the already frozen Ngn3-high -> Fev+ event region, are the RNA and
ATAC sustained-GDIS profiles more temporally aligned than expected after
breaking ATAC's global developmental registration while preserving local
ATAC structure?

PRIMARY DATA/REPRESENTATIONS
----------------------------
RNA  = X_scVI, 10D
ATAC = X_poissonvi, 10D
Common ordering = accepted RNA-only DPT from Phase F2
GDIS timing profile = sustained instability only (lambda_t = 0)
Primary window = 400 cells / 100-cell step

NULL MODEL
----------
ATAC block permutation along common pseudotime:
    - cells are ordered by common pseudotime;
    - ATAC is divided into contiguous 100-cell blocks;
    - block order is randomly permuted;
    - within-block order is preserved;
    - RNA and pseudotime remain unchanged.

Observed alignment metrics inside the frozen shared event region:
    1. |Q50_RNA - Q50_ATAC|       smaller = better alignment
    2. Spearman(RNA_GDIS, ATAC_GDIS) larger = better alignment
    3. Jensen-Shannon distance       smaller = better alignment

This analysis is exploratory and must be externally validated.
No output files are written.

Run:
    python 12_phase_f5_cross_modal_alignment_null.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import mudata as mu
import anndata as ad
import scanpy as sc

from scipy.signal import find_peaks, peak_prominences
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

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

REGION_GRID_POINTS = 500

N_PERMUTATIONS = 500
NULL_BLOCK_SIZE = 100
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
    sc.tl.dpt(adata, n_dcs=N_DIFFUSION_COMPONENTS)

    pt = adata.obs["dpt_pseudotime"].to_numpy(dtype=float)
    pt[~np.isfinite(pt)] = np.nan
    return pt


# =====================================================================
# WINDOWS / GDIS
# =====================================================================

def create_windows_from_ordered(X_ordered, pt_ordered, window_size, step_size):
    trajectories = []
    parameters = []

    start = 0
    while start + window_size <= len(pt_ordered):
        stop = start + window_size
        trajectories.append(X_ordered[start:stop].copy())
        parameters.append(float(np.mean(pt_ordered[start:stop])))
        start += step_size

    return trajectories, np.asarray(parameters, dtype=float)


def create_windows(X, pseudotime, window_size, step_size):
    valid = np.isfinite(pseudotime)
    X = X[valid]
    pt = pseudotime[valid]
    order = np.argsort(pt, kind="mergesort")

    return create_windows_from_ordered(
        X[order],
        pt[order],
        window_size,
        step_size,
    )


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
# STATE MEDIANS / FROZEN EVENT CENTERS
# =====================================================================

def get_state_medians(pseudotime, obs):
    medians = {}

    for state in BETA_LINEAGE:
        mask = (obs["cell_type"].astype(str) == state).to_numpy()
        values = pseudotime[mask]
        values = values[np.isfinite(values)]
        medians[state] = float(np.median(values))

    return medians


def detect_local_peaks(parameters, scores, modality, design_name):
    dynamic_range = float(np.max(scores) - np.min(scores))
    if dynamic_range <= 0:
        return pd.DataFrame()

    threshold = MIN_RELATIVE_PROMINENCE * dynamic_range

    indices, _ = find_peaks(
        scores,
        prominence=threshold,
        distance=MIN_PEAK_DISTANCE_WINDOWS,
    )

    if len(indices) == 0:
        return pd.DataFrame()

    prominences = peak_prominences(scores, indices)[0]

    rows = []
    for index, prominence in zip(indices, prominences):
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

    peaks = peaks.sort_values("peak_pt").reset_index(drop=True)
    clusters = []

    for _, row in peaks.iterrows():
        item = row.to_dict()
        pt = float(item["peak_pt"])

        if not clusters:
            clusters.append([item])
            continue

        centers = np.array(
            [
                np.median([member["peak_pt"] for member in cluster])
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
        designs = df["window_design"].astype(str).unique()
        median_pt = float(df["peak_pt"].median())
        spread = float(df["peak_pt"].max() - df["peak_pt"].min())
        median_prominence = float(df["relative_prominence"].median())

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
            raise RuntimeError("RNA and ATAC window centers differ.")

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

    peaks = pd.concat(all_peaks, ignore_index=True)

    rna_families = summarize_families(
        "RNA",
        cluster_peak_families(
            peaks.loc[peaks["modality"] == "RNA"]
        ),
    )

    atac_families = summarize_families(
        "ATAC",
        cluster_peak_families(
            peaks.loc[peaks["modality"] == "ATAC"]
        ),
    )

    robust_rna = rna_families.loc[rna_families["robust"]]
    robust_atac = atac_families.loc[atac_families["robust"]]

    eligible_rna = robust_rna.loc[
        (robust_rna["median_peak_pt"] >= medians["Ngn3 high"])
        & (robust_rna["median_peak_pt"] <= medians["Fev+"])
    ]

    if len(eligible_rna) != 1:
        raise RuntimeError(
            "Frozen RNA family could not be uniquely reproduced."
        )

    rna_center = float(
        eligible_rna.iloc[0]["median_peak_pt"]
    )

    eligible_atac = robust_atac.loc[
        (robust_atac["median_peak_pt"] >= medians["Ngn3 low"])
        & (robust_atac["median_peak_pt"] < rna_center)
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

    return rna_center, atac_center, critical_value


# =====================================================================
# REGION ALIGNMENT METRICS
# =====================================================================

def baseline_density(grid, values):
    excess = values - np.min(values)
    excess[excess < 0] = 0.0

    area = float(
        np.trapezoid(excess, grid)
        if hasattr(np, "trapezoid")
        else np.trapz(excess, grid)
    )

    if area <= 0 or not np.isfinite(area):
        return None

    return excess / area


def area_q50(grid, density):
    cumulative = cumulative_trapezoid(
        density,
        grid,
        initial=0.0,
    )

    total = float(cumulative[-1])
    if total <= 0:
        return np.nan

    cumulative = cumulative / total

    return float(
        np.interp(
            0.5,
            cumulative,
            grid,
        )
    )


def alignment_metrics(
    rna_parameters,
    rna_scores,
    atac_parameters,
    atac_scores,
    region_lower,
    region_upper,
):
    lower = max(
        region_lower,
        float(np.min(rna_parameters)),
        float(np.min(atac_parameters)),
    )

    upper = min(
        region_upper,
        float(np.max(rna_parameters)),
        float(np.max(atac_parameters)),
    )

    if upper <= lower:
        return None

    grid = np.linspace(
        lower,
        upper,
        REGION_GRID_POINTS,
    )

    rna_values = np.interp(
        grid,
        rna_parameters,
        rna_scores,
    )

    atac_values = np.interp(
        grid,
        atac_parameters,
        atac_scores,
    )

    rna_density = baseline_density(
        grid,
        rna_values,
    )

    atac_density = baseline_density(
        grid,
        atac_values,
    )

    if rna_density is None or atac_density is None:
        return None

    q50_rna = area_q50(
        grid,
        rna_density,
    )

    q50_atac = area_q50(
        grid,
        atac_density,
    )

    rho, _ = spearmanr(
        rna_values,
        atac_values,
    )

    # Jensen-Shannon expects discrete probability masses.
    rna_mass = rna_density / np.sum(rna_density)
    atac_mass = atac_density / np.sum(atac_density)

    js_distance = float(
        jensenshannon(
            rna_mass,
            atac_mass,
            base=2.0,
        )
    )

    return {
        "q50_rna": q50_rna,
        "q50_atac": q50_atac,
        "signed_center_difference": q50_rna - q50_atac,
        "absolute_center_separation": abs(q50_rna - q50_atac),
        "curve_spearman_rho": float(rho),
        "jensen_shannon_distance": js_distance,
    }


# =====================================================================
# NULL MODEL
# =====================================================================

def contiguous_blocks(n_cells, block_size):
    return [
        np.arange(start, min(start + block_size, n_cells), dtype=int)
        for start in range(0, n_cells, block_size)
    ]


def permute_atac_blocks(X_atac_ordered, rng):
    blocks = contiguous_blocks(
        len(X_atac_ordered),
        NULL_BLOCK_SIZE,
    )

    order = rng.permutation(
        len(blocks)
    )

    return np.vstack(
        [
            X_atac_ordered[blocks[i]]
            for i in order
        ]
    )


def empirical_p_smaller(observed, null_values):
    null_values = np.asarray(null_values, dtype=float)
    return float(
        (1 + np.sum(null_values <= observed))
        / (len(null_values) + 1)
    )


def empirical_p_larger(observed, null_values):
    null_values = np.asarray(null_values, dtype=float)
    return float(
        (1 + np.sum(null_values >= observed))
        / (len(null_values) + 1)
    )


def null_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q2.5": float(np.quantile(values, 0.025)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q97.5": float(np.quantile(values, 0.975)),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F5 — EXPLORATORY CROSS-MODAL ALIGNMENT NULL TEST"
    )

    print(f"Dataset: {DATA_FILE.resolve()}")
    print()
    print(
        "Directional ATAC-leading priming is NOT supported by "
        "Phase F4/F4c."
    )
    print()
    print(
        "Question: are RNA and ATAC instability profiles more "
        "temporally coordinated than expected after disrupting "
        "ATAC's global developmental registration?"
    )
    print()
    print(f"Null permutations: {N_PERMUTATIONS}")
    print(f"ATAC block size:   {NULL_BLOCK_SIZE} ordered cells")
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

    # -----------------------------------------------------------------
    # Common trajectory / frozen region
    # -----------------------------------------------------------------

    section(
        "2. REPRODUCE COMMON TRAJECTORY AND FROZEN EVENT REGION"
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
        frozen_atac_center - FAMILY_MATCH_TOLERANCE,
    )

    region_upper = min(
        1.0,
        frozen_rna_center + FAMILY_MATCH_TOLERANCE,
    )

    print(f"Frozen ATAC center: {frozen_atac_center:.6f}")
    print(f"Frozen RNA center:  {frozen_rna_center:.6f}")
    print(
        f"Shared event region: "
        f"[{region_lower:.6f}, {region_upper:.6f}]"
    )

    # -----------------------------------------------------------------
    # Observed curves
    # -----------------------------------------------------------------

    section("3. OBSERVED PRIMARY CROSS-MODAL ALIGNMENT")

    valid = np.isfinite(pseudotime)
    order = np.argsort(
        pseudotime[valid],
        kind="mergesort",
    )

    pt_ordered = pseudotime[valid][order]
    X_rna_ordered = X_rna_z[valid][order]
    X_atac_ordered = X_atac_z[valid][order]

    rna_traj, parameters = create_windows_from_ordered(
        X_rna_ordered,
        pt_ordered,
        PRIMARY_WINDOW_SIZE,
        PRIMARY_STEP_SIZE,
    )

    atac_traj, atac_parameters = create_windows_from_ordered(
        X_atac_ordered,
        pt_ordered,
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

    observed = alignment_metrics(
        rna_x,
        rna_y,
        atac_x,
        atac_y,
        region_lower,
        region_upper,
    )

    if observed is None:
        print("ERROR: Observed alignment metrics failed.")
        sys.exit(1)

    print_df(
        pd.DataFrame([observed], index=["observed"])
    )

    # -----------------------------------------------------------------
    # Null permutations
    # -----------------------------------------------------------------

    section("4. ATAC BLOCK-PERMUTATION NULL")

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    records = []

    for i in range(1, N_PERMUTATIONS + 1):
        X_atac_null = permute_atac_blocks(
            X_atac_ordered,
            rng,
        )

        atac_null_traj, atac_null_parameters = (
            create_windows_from_ordered(
                X_atac_null,
                pt_ordered,
                PRIMARY_WINDOW_SIZE,
                PRIMARY_STEP_SIZE,
            )
        )

        atac_null_x, atac_null_y = sustained_gdis_curve(
            atac_null_traj,
            atac_null_parameters,
            critical_value,
        )

        metrics = alignment_metrics(
            rna_x,
            rna_y,
            atac_null_x,
            atac_null_y,
            region_lower,
            region_upper,
        )

        if metrics is not None:
            metrics["permutation_id"] = i
            records.append(metrics)

        if i % PROGRESS_EVERY == 0 or i == N_PERMUTATIONS:
            print(
                f"Completed {i:4d}/{N_PERMUTATIONS} permutations"
            )

    null = pd.DataFrame(records)

    print()
    print(f"Valid null permutations: {len(null)}")

    if len(null) < int(0.95 * N_PERMUTATIONS):
        print("ERROR: Too many null permutations failed.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Null comparison
    # -----------------------------------------------------------------

    section("5. NULL COMPARISON")

    specs = [
        ("absolute_center_separation", "smaller"),
        ("curve_spearman_rho", "larger"),
        ("jensen_shannon_distance", "smaller"),
    ]

    rows = []

    for metric, direction in specs:
        obs_value = float(observed[metric])
        null_values = null[metric].to_numpy(dtype=float)
        summary = null_summary(null_values)

        if direction == "smaller":
            p = empirical_p_smaller(
                obs_value,
                null_values,
            )
        else:
            p = empirical_p_larger(
                obs_value,
                null_values,
            )

        rows.append(
            {
                "metric": metric,
                "observed": obs_value,
                "better_direction": direction,
                "null_mean": summary["mean"],
                "null_median": summary["median"],
                "null_q2.5": summary["q2.5"],
                "null_q25": summary["q25"],
                "null_q75": summary["q75"],
                "null_q97.5": summary["q97.5"],
                "empirical_p": p,
            }
        )

    comparison = pd.DataFrame(rows).set_index("metric")
    print_df(comparison)

    signed_null = null[
        "signed_center_difference"
    ].to_numpy(dtype=float)

    signed_summary = null_summary(
        signed_null
    )

    print()
    print("Signed Q50 difference is descriptive only:")
    print(
        f"  observed RNA - ATAC = "
        f"{observed['signed_center_difference']:.6f}"
    )
    print(
        f"  null median = "
        f"{signed_summary['median']:.6f}"
    )
    print(
        f"  null 95% interval = "
        f"[{signed_summary['q2.5']:.6f}, "
        f"{signed_summary['q97.5']:.6f}]"
    )

    # -----------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------

    section("6. PHASE F5 INTERPRETATION")

    p_center = float(
        comparison.loc[
            "absolute_center_separation",
            "empirical_p",
        ]
    )

    p_rho = float(
        comparison.loc[
            "curve_spearman_rho",
            "empirical_p",
        ]
    )

    p_js = float(
        comparison.loc[
            "jensen_shannon_distance",
            "empirical_p",
        ]
    )

    n_support = sum(
        p < 0.05
        for p in [p_center, p_rho, p_js]
    )

    if n_support >= 2:
        verdict = (
            "EXPLORATORY EVIDENCE OF CROSS-MODAL TEMPORAL COORDINATION"
        )
        reason = (
            "At least two complementary alignment metrics are more "
            "coordinated than expected under the ATAC block-permutation "
            "null. This supports coordination, not directional priming."
        )
    elif n_support == 1:
        verdict = "LIMITED EXPLORATORY ALIGNMENT EVIDENCE"
        reason = (
            "Only one alignment metric departs from the null; evidence "
            "for coordination is incomplete."
        )
    else:
        verdict = "NO ALIGNMENT EVIDENCE BEYOND NULL"
        reason = (
            "The observed RNA/ATAC event-region alignment is not more "
            "coordinated than expected after disrupting ATAC's global "
            "developmental registration."
        )

    print(f"PHASE F5 VERDICT: {verdict}")
    print()
    print(reason)

    print()
    print("Guardrails:")
    print("  - Peak-based CMIL remains inconclusive.")
    print("  - Region timing does not support ATAC-leading priming.")
    print(
        "  - This alignment test is exploratory and requires external "
        "validation before becoming a main biological conclusion."
    )
    print("  - No files were created.")

    line("=")


if __name__ == "__main__":
    main()

