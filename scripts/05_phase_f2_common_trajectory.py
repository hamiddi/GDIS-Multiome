#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
05_phase_f2_common_trajectory.py
================================

Project:
Cross-Modal Dynamical Instability Reveals Chromatin Priming
before Transcriptional Cell-Fate Transitions

Phase F2:
Construct and validate ONE common beta-lineage developmental ordering.

WHY THIS PHASE EXISTS
---------------------
The central cross-modal analysis requires RNA-GDIS and ATAC-GDIS to be
evaluated along the SAME cell ordering. Otherwise an apparent modality lead
could be created by using two different pseudotime coordinate systems.

The common ordering is therefore constructed from RNA ONLY.

Primary trajectory representation
---------------------------------
RNA X_scVI, all 10 stored dimensions.

This choice is based on Phase F1b:
    - very low experiment partial R^2
    - strong biological partial R^2
    - high E15.5 cross-experiment mixing

ATAC X_poissonvi is NOT used to construct pseudotime.
It is used only as an independent cross-modal coherence check.

Primary beta lineage
--------------------
Ngn3 low -> Ngn3 high -> Fev+ -> Fev+ Beta -> Beta

"Ngn3 high cycling" remains excluded from the primary analysis.

Trajectory method
-----------------
Diffusion pseudotime (DPT) on an RNA-scVI kNN graph.

Root selection is objective:
    the medoid of E14.5 Ngn3-low cells in RNA-scVI space.

Primary graph:
    30 nearest neighbors.

Sensitivity graphs:
    20 and 50 nearest neighbors.

Validation
----------
1. Pseudotime must be finite for nearly all cells.
2. Pseudotime should increase with real developmental stage.
3. Median pseudotime should broadly follow the known beta-lineage sequence.
4. The E15.5 trajectory should not be strongly explained by experiment batch
   after controlling for cell type.
5. Pseudotime rankings should be stable across k=20, 30, and 50 graphs.
6. The RNA-derived pseudotime should be locally coherent in BOTH:
       RNA X_scVI
       ATAC X_poissonvi
   even though ATAC was not used to construct it.

No GDIS is calculated in this phase.
No output files are created.

Run
---
    python 05_phase_f2_common_trajectory.py

Requirements
------------
    numpy
    pandas
    scipy
    scikit-learn
    anndata
    scanpy
    mudata
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import mudata as mu
import anndata as ad
import scanpy as sc

from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


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

STATE_ORDER = {
    state: i
    for i, state in enumerate(BETA_LINEAGE)
}

STAGE_ORDER = {
    "E14.5": 0,
    "E15.5": 1,
    "E16.5": 2,
}

TECHNICAL_VARIABLE = "experiment_batch"

RNA_REPRESENTATION = "X_scVI"
ATAC_REPRESENTATION = "X_poissonvi"

PRIMARY_NEIGHBORS = 30
SENSITIVITY_NEIGHBORS = [20, 30, 50]

N_DIFFUSION_COMPONENTS = 10
RANDOM_SEED = 20260920

LOCAL_COHERENCE_K = 30
N_RANDOM_COHERENCE_PERMUTATIONS = 100


# =====================================================================
# DISPLAY
# =====================================================================

def line(char="=", width=94):
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


def print_df(df, digits=4):
    if df.empty:
        print("(empty)")
        return

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 240,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string(index=True))


# =====================================================================
# HELPERS
# =====================================================================

def as_numpy(x):
    """Convert a stored latent matrix to a dense NumPy array."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def zscore_columns(X):
    """Column-standardize a low-dimensional latent representation."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def choose_root_medoid(X, obs):
    """
    Select the root objectively as the medoid of E14.5 Ngn3-low cells.

    The medoid is approximated as the candidate closest to the candidate-group
    centroid in standardized RNA-scVI space.
    """
    candidate_mask = (
        (obs["stage"].astype(str) == "E14.5")
        & (obs["cell_type"].astype(str) == "Ngn3 low")
    ).to_numpy()

    indices = np.flatnonzero(candidate_mask)

    if len(indices) == 0:
        raise RuntimeError(
            "No E14.5 Ngn3-low cells are available for root selection."
        )

    X_candidates = X[indices]
    centroid = X_candidates.mean(axis=0)

    distances = np.linalg.norm(
        X_candidates - centroid,
        axis=1,
    )

    local_best = int(np.argmin(distances))
    global_index = int(indices[local_best])

    return global_index


def construct_dpt(X, obs, n_neighbors):
    """
    Construct diffusion pseudotime using RNA latent coordinates only.

    Returns
    -------
    pseudotime : ndarray
    root_index : int
    connected_fraction : float
    """
    X_std = zscore_columns(X)

    adata = ad.AnnData(
        X=np.zeros((X_std.shape[0], 1), dtype=np.float32)
    )

    adata.obs = obs.copy()
    adata.obsm["X_common_rna"] = X_std.astype(np.float32)

    root_index = choose_root_medoid(
        X_std,
        adata.obs,
    )

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X_common_rna",
        metric="euclidean",
        random_state=RANDOM_SEED,
    )

    sc.tl.diffmap(
        adata,
        n_comps=max(N_DIFFUSION_COMPONENTS + 1, 15),
        random_state=RANDOM_SEED,
    )

    adata.uns["iroot"] = root_index

    sc.tl.dpt(
        adata,
        n_dcs=N_DIFFUSION_COMPONENTS,
    )

    pseudotime = (
        adata.obs["dpt_pseudotime"]
        .to_numpy(dtype=float)
    )

    finite = np.isfinite(pseudotime)
    connected_fraction = float(finite.mean())

    # DPT can use inf for cells outside the root component.
    # We keep them as NaN for downstream diagnostics.
    pseudotime[~finite] = np.nan

    return pseudotime, root_index, connected_fraction


def stage_spearman(pseudotime, obs):
    """Spearman correlation between pseudotime and real developmental stage."""
    stage_numeric = (
        obs["stage"]
        .astype(str)
        .map(STAGE_ORDER)
        .to_numpy(dtype=float)
    )

    valid = (
        np.isfinite(pseudotime)
        & np.isfinite(stage_numeric)
    )

    if valid.sum() < 10:
        return np.nan

    rho, _ = spearmanr(
        pseudotime[valid],
        stage_numeric[valid],
    )

    return float(rho)


def state_spearman(pseudotime, obs):
    """Spearman correlation with prespecified beta-lineage state order."""
    state_numeric = (
        obs["cell_type"]
        .astype(str)
        .map(STATE_ORDER)
        .to_numpy(dtype=float)
    )

    valid = (
        np.isfinite(pseudotime)
        & np.isfinite(state_numeric)
    )

    if valid.sum() < 10:
        return np.nan

    rho, _ = spearmanr(
        pseudotime[valid],
        state_numeric[valid],
    )

    return float(rho)


def group_summary(pseudotime, labels):
    """Summarize pseudotime by a categorical label."""
    df = pd.DataFrame({
        "label": labels.astype(str).to_numpy(),
        "pseudotime": pseudotime,
    })

    df = df.loc[np.isfinite(df["pseudotime"])]

    summary = (
        df.groupby("label")["pseudotime"]
        .agg(["count", "median", "mean", "std"])
    )

    summary["q25"] = (
        df.groupby("label")["pseudotime"]
        .quantile(0.25)
    )

    summary["q75"] = (
        df.groupby("label")["pseudotime"]
        .quantile(0.75)
    )

    return summary[
        ["count", "median", "q25", "q75", "mean", "std"]
    ]


def median_state_order_score(pseudotime, obs):
    """
    Count how many adjacent state medians increase in the expected direction.

    There are four expected adjacent transitions:
        Ngn3 low -> Ngn3 high
        Ngn3 high -> Fev+
        Fev+ -> Fev+ Beta
        Fev+ Beta -> Beta
    """
    medians = {}

    for state in BETA_LINEAGE:
        values = pseudotime[
            obs["cell_type"].astype(str).to_numpy() == state
        ]
        values = values[np.isfinite(values)]

        medians[state] = (
            float(np.median(values))
            if len(values) else np.nan
        )

    increases = []

    for left, right in zip(
        BETA_LINEAGE[:-1],
        BETA_LINEAGE[1:],
    ):
        if np.isfinite(medians[left]) and np.isfinite(medians[right]):
            increases.append(
                medians[right] > medians[left]
            )

    n_correct = int(sum(increases))
    n_total = len(increases)

    return medians, n_correct, n_total


def one_hot(series):
    """Dummy-code a categorical variable, dropping the first level."""
    return pd.get_dummies(
        series.astype(str),
        drop_first=True,
        dtype=float,
    ).to_numpy(dtype=float)


def design_matrix(n, *blocks):
    pieces = [np.ones((n, 1), dtype=float)]

    for block in blocks:
        if block.shape[1] > 0:
            pieces.append(block)

    return np.hstack(pieces)


def sse(y, X):
    """Residual sum of squares for ordinary least squares."""
    beta, _, _, _ = np.linalg.lstsq(
        X,
        y.reshape(-1, 1),
        rcond=None,
    )

    residual = y.reshape(-1, 1) - X @ beta

    return float(np.sum(residual ** 2))


def experiment_partial_r2_on_pseudotime(
    pseudotime,
    obs,
):
    """
    Within E15.5, measure residual experiment effect on common pseudotime
    after controlling for beta-lineage cell type.
    """
    mask = (
        (obs["stage"].astype(str) == "E15.5").to_numpy()
        & np.isfinite(pseudotime)
    )

    y = pseudotime[mask]
    obs_local = obs.loc[mask].reset_index(drop=True)

    if len(y) < 10:
        return np.nan

    cell_block = one_hot(obs_local["cell_type"])
    tech_block = one_hot(obs_local[TECHNICAL_VARIABLE])

    X_cell = design_matrix(
        len(y),
        cell_block,
    )

    X_full = design_matrix(
        len(y),
        cell_block,
        tech_block,
    )

    sse_cell = sse(y, X_cell)
    sse_full = sse(y, X_full)

    if sse_cell <= 0:
        return np.nan

    value = (
        sse_cell - sse_full
    ) / sse_cell

    return max(float(value), 0.0)


def per_state_e155_batch_shift(
    pseudotime,
    obs,
):
    """
    Report median pseudotime difference between Exp_1 and Exp_2
    within each E15.5 beta-lineage state.
    """
    rows = []

    for state in BETA_LINEAGE:
        state_mask = (
            (obs["stage"].astype(str) == "E15.5")
            & (obs["cell_type"].astype(str) == state)
        ).to_numpy()

        local = pd.DataFrame({
            "pt": pseudotime[state_mask],
            "batch": (
                obs.loc[
                    state_mask,
                    TECHNICAL_VARIABLE,
                ]
                .astype(str)
                .to_numpy()
            ),
        })

        local = local.loc[
            np.isfinite(local["pt"])
        ]

        groups = sorted(local["batch"].unique())

        if len(groups) != 2:
            continue

        g1 = local.loc[
            local["batch"] == groups[0],
            "pt",
        ]

        g2 = local.loc[
            local["batch"] == groups[1],
            "pt",
        ]

        rows.append({
            "cell_type": state,
            "group_1": groups[0],
            "group_2": groups[1],
            "n_group_1": len(g1),
            "n_group_2": len(g2),
            "median_group_1": float(g1.median()),
            "median_group_2": float(g2.median()),
            "absolute_median_difference": float(
                abs(g1.median() - g2.median())
            ),
        })

    return pd.DataFrame(rows)


def rank_correlation(pt_a, pt_b):
    """Spearman rank correlation between two pseudotime vectors."""
    valid = (
        np.isfinite(pt_a)
        & np.isfinite(pt_b)
    )

    if valid.sum() < 10:
        return np.nan

    rho, _ = spearmanr(
        pt_a[valid],
        pt_b[valid],
    )

    return float(rho)


def local_pseudotime_coherence(
    X,
    pseudotime,
    k=LOCAL_COHERENCE_K,
    n_permutations=N_RANDOM_COHERENCE_PERMUTATIONS,
):
    """
    Quantify whether cells that are neighbors in a latent space also have
    similar RNA-derived pseudotime.

    observed_mean_delta:
        mean |pt_i - pt_neighbor|

    random_mean_delta:
        mean value after shuffling pseudotime across cells

    coherence_ratio:
        observed / random

        lower is better;
        values far below 1 indicate strong local trajectory coherence.
    """
    valid = np.isfinite(pseudotime)

    X_use = zscore_columns(X[valid])
    pt = pseudotime[valid]

    n = len(pt)

    if n <= k:
        return {
            "observed_mean_delta": np.nan,
            "random_mean_delta": np.nan,
            "coherence_ratio": np.nan,
        }

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
    )

    nn.fit(X_use)

    indices = nn.kneighbors(
        X_use,
        return_distance=False,
    )[:, 1:]

    observed = np.mean(
        np.abs(
            pt[:, None]
            - pt[indices]
        )
    )

    rng = np.random.default_rng(RANDOM_SEED)

    random_values = []

    for _ in range(n_permutations):
        shuffled = rng.permutation(pt)

        random_delta = np.mean(
            np.abs(
                shuffled[:, None]
                - shuffled[indices]
            )
        )

        random_values.append(
            random_delta
        )

    random_mean = float(
        np.mean(random_values)
    )

    ratio = (
        float(observed / random_mean)
        if random_mean > 0
        else np.nan
    )

    return {
        "observed_mean_delta": float(observed),
        "random_mean_delta": random_mean,
        "coherence_ratio": ratio,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():

    section("PHASE F2 — COMMON RNA-DERIVED TRAJECTORY")

    print(f"Dataset: {DATA_FILE.resolve()}")
    print()
    print(f"RNA trajectory representation:  {RNA_REPRESENTATION} (10D)")
    print(f"ATAC validation representation: {ATAC_REPRESENTATION} (10D)")
    print()
    print("ATAC is NOT used to construct pseudotime.")
    print("No GDIS is calculated.")
    print("No output files are created.")

    # -----------------------------------------------------------------
    # Load dataset.
    # -----------------------------------------------------------------

    if not DATA_FILE.exists():
        print("\nERROR: Dataset file not found.")
        sys.exit(1)

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

    required_metadata = [
        "stage",
        "cell_type",
        TECHNICAL_VARIABLE,
    ]

    missing = [
        c for c in required_metadata
        if c not in obs_full.columns
    ]

    if missing:
        print(
            "ERROR: Missing required metadata: "
            + ", ".join(missing)
        )
        sys.exit(1)

    if RNA_REPRESENTATION not in rna.obsm:
        print(
            f"ERROR: RNA representation "
            f"{RNA_REPRESENTATION} not found."
        )
        sys.exit(1)

    if ATAC_REPRESENTATION not in atac.obsm:
        print(
            f"ERROR: ATAC representation "
            f"{ATAC_REPRESENTATION} not found."
        )
        sys.exit(1)

    # -----------------------------------------------------------------
    # Restrict to primary beta lineage.
    # -----------------------------------------------------------------

    section("2. DEFINE PRIMARY BETA LINEAGE")

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

    X_rna = as_numpy(
        rna.obsm[RNA_REPRESENTATION]
    )[beta_mask]

    X_atac_full = as_numpy(
        atac.obsm[ATAC_REPRESENTATION]
    )[beta_mask]

    # Primary matched dimensionality = 10.
    X_rna = X_rna[:, :10]
    X_atac = X_atac_full[:, :10]

    print(f"Beta-lineage cells: {len(obs):,}")
    print(f"RNA latent shape:    {X_rna.shape}")
    print(f"ATAC latent shape:   {X_atac.shape}")

    print()
    print("Stage × cell-state counts:")

    counts = pd.crosstab(
        obs["cell_type"].astype(str),
        obs["stage"].astype(str),
        margins=True,
    )

    print_df(counts, digits=0)

    # -----------------------------------------------------------------
    # Construct primary and sensitivity DPT orderings.
    # -----------------------------------------------------------------

    section("3. RNA-ONLY DIFFUSION PSEUDOTIME")

    trajectories = {}
    trajectory_rows = []

    for k in SENSITIVITY_NEIGHBORS:

        print(f"\nConstructing DPT with k={k}...")

        pt, root_index, connected_fraction = construct_dpt(
            X=X_rna,
            obs=obs,
            n_neighbors=k,
        )

        trajectories[k] = pt

        root_stage = str(
            obs.loc[root_index, "stage"]
        )

        root_state = str(
            obs.loc[root_index, "cell_type"]
        )

        stage_rho = stage_spearman(
            pt,
            obs,
        )

        state_rho = state_spearman(
            pt,
            obs,
        )

        medians, n_correct, n_total = (
            median_state_order_score(
                pt,
                obs,
            )
        )

        batch_r2 = (
            experiment_partial_r2_on_pseudotime(
                pt,
                obs,
            )
        )

        trajectory_rows.append({
            "k_neighbors": k,
            "connected_fraction": connected_fraction,
            "root_stage": root_stage,
            "root_state": root_state,
            "stage_spearman_rho": stage_rho,
            "state_order_spearman_rho": state_rho,
            "adjacent_state_median_increases": (
                f"{n_correct}/{n_total}"
            ),
            "e155_experiment_partial_r2": batch_r2,
        })

    trajectory_summary = pd.DataFrame(
        trajectory_rows
    ).set_index("k_neighbors")

    print()
    print_df(trajectory_summary)

    primary_pt = trajectories[
        PRIMARY_NEIGHBORS
    ]

    # -----------------------------------------------------------------
    # Primary pseudotime biological summaries.
    # -----------------------------------------------------------------

    section(
        f"4. PRIMARY DPT BIOLOGICAL ORDER "
        f"(k={PRIMARY_NEIGHBORS})"
    )

    subsection("Developmental stages")

    stage_summary = group_summary(
        primary_pt,
        obs["stage"],
    )

    # Order stages explicitly.
    stage_summary = stage_summary.reindex(
        ["E14.5", "E15.5", "E16.5"]
    )

    print_df(stage_summary)

    subsection("Beta-lineage states")

    state_summary = group_summary(
        primary_pt,
        obs["cell_type"],
    )

    state_summary = state_summary.reindex(
        BETA_LINEAGE
    )

    print_df(state_summary)

    medians, n_correct, n_total = (
        median_state_order_score(
            primary_pt,
            obs,
        )
    )

    print()
    print(
        "Expected adjacent state-median increases: "
        f"{n_correct}/{n_total}"
    )

    # -----------------------------------------------------------------
    # E15.5 experiment influence on common pseudotime.
    # -----------------------------------------------------------------

    section("5. E15.5 EXPERIMENT EFFECT ON COMMON PSEUDOTIME")

    batch_r2 = (
        experiment_partial_r2_on_pseudotime(
            primary_pt,
            obs,
        )
    )

    print(
        "Experiment partial R² on pseudotime "
        "after controlling cell type:"
    )
    print(f"  {batch_r2:.6f}")

    print()
    print(
        "Per-cell-state absolute median pseudotime differences "
        "between Exp_1 and Exp_2:"
    )

    batch_shift = (
        per_state_e155_batch_shift(
            primary_pt,
            obs,
        )
    )

    if batch_shift.empty:
        print("(not available)")
    else:
        print_df(
            batch_shift.set_index(
                "cell_type"
            )
        )

    # -----------------------------------------------------------------
    # Neighbor-number sensitivity.
    # -----------------------------------------------------------------

    section("6. TRAJECTORY STABILITY ACROSS kNN SETTINGS")

    rows = []

    for k in SENSITIVITY_NEIGHBORS:
        if k == PRIMARY_NEIGHBORS:
            continue

        rho = rank_correlation(
            primary_pt,
            trajectories[k],
        )

        rows.append({
            "comparison": (
                f"k={PRIMARY_NEIGHBORS} vs k={k}"
            ),
            "spearman_rho": rho,
        })

    stability = pd.DataFrame(
        rows
    ).set_index("comparison")

    print_df(stability)

    # -----------------------------------------------------------------
    # Cross-modal local coherence.
    # -----------------------------------------------------------------

    section("7. CROSS-MODAL LOCAL TRAJECTORY COHERENCE")

    print(
        "The pseudotime below was constructed from RNA only."
    )
    print(
        "We now ask whether nearby cells in each modality "
        "also have similar RNA-derived pseudotime."
    )
    print()

    rna_coherence = local_pseudotime_coherence(
        X_rna,
        primary_pt,
    )

    atac_coherence = local_pseudotime_coherence(
        X_atac,
        primary_pt,
    )

    coherence = pd.DataFrame(
        [
            {
                "representation": "RNA_scVI_10D",
                **rna_coherence,
            },
            {
                "representation": "ATAC_poissonVI_10D",
                **atac_coherence,
            },
        ]
    ).set_index("representation")

    print_df(coherence)

    print()
    print(
        "Interpretation: coherence_ratio << 1 means neighboring cells "
        "share similar pseudotime much more than expected after random "
        "pseudotime shuffling."
    )

    # -----------------------------------------------------------------
    # Prespecified decision.
    # -----------------------------------------------------------------

    section("8. PHASE F2 DECISION")

    primary_row = trajectory_summary.loc[
        PRIMARY_NEIGHBORS
    ]

    connected_ok = (
        float(
            primary_row["connected_fraction"]
        )
        >= 0.99
    )

    stage_ok = (
        float(
            primary_row["stage_spearman_rho"]
        )
        >= 0.40
    )

    state_ok = (
        float(
            primary_row[
                "state_order_spearman_rho"
            ]
        )
        >= 0.40
    )

    order_ok = (
        n_correct >= 3
    )

    batch_ok = (
        np.isfinite(batch_r2)
        and batch_r2 < 0.02
    )

    sensitivity_rhos = (
        stability["spearman_rho"]
        .to_numpy(dtype=float)
    )

    stability_ok = (
        np.all(
            np.isfinite(
                sensitivity_rhos
            )
        )
        and np.min(
            sensitivity_rhos
        ) >= 0.90
    )

    rna_coherence_ok = (
        np.isfinite(
            rna_coherence[
                "coherence_ratio"
            ]
        )
        and rna_coherence[
            "coherence_ratio"
        ] < 0.70
    )

    atac_coherence_ok = (
        np.isfinite(
            atac_coherence[
                "coherence_ratio"
            ]
        )
        and atac_coherence[
            "coherence_ratio"
        ] < 0.80
    )

    checks = pd.DataFrame(
        [
            [
                "Connected root component >=99%",
                connected_ok,
            ],
            [
                "Stage Spearman rho >=0.40",
                stage_ok,
            ],
            [
                "State-order Spearman rho >=0.40",
                state_ok,
            ],
            [
                "At least 3/4 adjacent state medians increase",
                order_ok,
            ],
            [
                "E15.5 experiment partial R² <0.02",
                batch_ok,
            ],
            [
                "kNN sensitivity rank rho >=0.90",
                stability_ok,
            ],
            [
                "RNA local coherence ratio <0.70",
                rna_coherence_ok,
            ],
            [
                "ATAC local coherence ratio <0.80",
                atac_coherence_ok,
            ],
        ],
        columns=[
            "criterion",
            "pass",
        ],
    )

    checks["status"] = np.where(
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

    n_pass = int(
        checks["pass"].sum()
    )

    n_total = len(checks)

    print()
    print(
        f"Criteria passed: {n_pass}/{n_total}"
    )

    # Hard requirements.
    hard_ok = (
        connected_ok
        and batch_ok
        and stability_ok
        and rna_coherence_ok
    )

    # Biological ordering can be imperfect because differentiation states
    # overlap; require at least two of the three biological-order checks.
    biological_checks = [
        stage_ok,
        state_ok,
        order_ok,
    ]

    biological_ok = (
        sum(
            bool(x)
            for x in biological_checks
        )
        >= 2
    )

    if (
        hard_ok
        and biological_ok
        and atac_coherence_ok
    ):
        verdict = "GO"

        reason = (
            "The RNA-derived common ordering is technically stable, "
            "biologically ordered, insensitive to reasonable kNN choices, "
            "and independently coherent in ATAC space."
        )

    elif (
        hard_ok
        and biological_ok
    ):
        verdict = "GO WITH CAUTION"

        reason = (
            "The RNA-derived trajectory itself is acceptable, but ATAC "
            "local coherence is weaker than the prespecified target. "
            "Proceed only with explicit ATAC sensitivity analyses."
        )

    else:
        verdict = "REVIEW REQUIRED"

        reason = (
            "The common ordering does not yet satisfy the prespecified "
            "trajectory-quality criteria. Do not calculate cross-modal "
            "GDIS until the failed criterion is resolved."
        )

    print()
    print(f"PHASE F2 VERDICT: {verdict}")
    print()
    print(reason)

    print()
    print("No GDIS was calculated.")
    print("No output files were created.")

    if verdict.startswith("GO"):
        print()
        print(
            "Next step: 06_phase_f3_gdis_rna_atac.py"
        )

    line("=")


if __name__ == "__main__":
    main()

