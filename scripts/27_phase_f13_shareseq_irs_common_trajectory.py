#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
27_phase_f13_shareseq_irs_common_trajectory.py
==============================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F13
---------
Construct and validate the RNA-ONLY common developmental clock for the frozen
TAC-1 -> TAC-2 -> IRS branch.

INPUTS FROM F12
---------------
data/GSE140203/representations_f12/
    f12_irs_branch_cells.tsv.gz
    f12_rna_irs_10d.npy
    f12_atac_irs_10d.npy
    f12_manifest.json

FROZEN F12 STATE SPACES
-----------------------
RNA:
    10D technical-rescued representation
    original PCs 2-11

ATAC:
    10D depth-audited representation

CRITICAL ANTI-CIRCULARITY RULE
------------------------------
The common clock is constructed from RNA ONLY.

ATAC:
    - does NOT enter neighbor construction;
    - does NOT determine the root;
    - does NOT influence diffusion map;
    - does NOT influence DPT;
    - does NOT influence k selection.

ATAC is examined only AFTER the RNA clock is fixed.

BIOLOGICAL DIRECTION
--------------------
The source SHARE-seq study identifies TACs as undifferentiated progenitors
that generate IRS and other hair-follicle fates. TAC-1 was reported as the
RNA-pseudotime-defined root population.

Later multimodal velocity work supports:
    TAC-1 -> TAC-2 -> terminal hair-follicle fates

F13 therefore:

ROOT:
    chooses one medoid-like cell ONLY from TAC-1 using RNA 10D.

PRIMARY:
    k = 30

SENSITIVITY:
    k = 20
    k = 50

TRAJECTORY:
    Scanpy diffusion pseudotime (DPT)

VALIDATION
----------
For k = 20 / 30 / 50:
    - RNA kNN graph must be connected;
    - pseudotime must be finite;
    - root pseudotime must be minimal/near-zero;
    - state medians are reported and tested:
          TAC-1 < TAC-2 < IRS
    - pairwise ordering probabilities are reported:
          P(TAC-2 > TAC-1)
          P(IRS > TAC-2)
          P(IRS > TAC-1)

Across k:
    - k30 vs k20 Spearman >= 0.95
    - k30 vs k50 Spearman >= 0.95

Pseudotime continuity:
    - adjacent-state distribution overlap is reported;
    - local pseudotime smoothness on the RNA graph is reported.

ATAC POST-HOC ONLY:
    Once the primary RNA k=30 clock is frozen, F13 measures how well ATAC
    nearest neighbors preserve that RNA-derived order. This is secondary
    evidence only and does not determine whether the RNA clock passes.

IMPORTANT
---------
No GDIS.
No CMIL.
No peak/event pairing.
No RNA-vs-ATAC lead-lag test.

OUTPUTS
-------
data/GSE140203/representations_f13/
    f13_irs_common_pseudotime.tsv.gz
    f13_trajectory_validation.tsv
    f13_state_quantiles.tsv
    f13_pairwise_ordering.tsv
    f13_atac_posthoc_clock_coherence.tsv
    f13_manifest.json

RUN
---
    python 27_phase_f13_shareseq_irs_common_trajectory.py

DEPENDENCIES
------------
numpy
pandas
scipy
scikit-learn
anndata
scanpy
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from sklearn.neighbors import NearestNeighbors

try:
    import anndata as ad
    import scanpy as sc
except ImportError as exc:
    raise SystemExit(
        "\nERROR: Phase F13 requires Scanpy and AnnData.\n\n"
        "Install in the active environment, for example:\n\n"
        "    conda install -c conda-forge scanpy\n\n"
        "Then rerun Phase F13.\n"
    ) from exc


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE140203"

DATA_DIR = Path("data") / ACCESSION
F12_DIR = DATA_DIR / "representations_f12"
F13_DIR = DATA_DIR / "representations_f13"

F12_CELLS = F12_DIR / "f12_irs_branch_cells.tsv.gz"
F12_RNA = F12_DIR / "f12_rna_irs_10d.npy"
F12_ATAC = F12_DIR / "f12_atac_irs_10d.npy"
F12_MANIFEST = F12_DIR / "f12_manifest.json"

OUTPUT_PSEUDOTIME = F13_DIR / "f13_irs_common_pseudotime.tsv.gz"
OUTPUT_VALIDATION = F13_DIR / "f13_trajectory_validation.tsv"
OUTPUT_QUANTILES = F13_DIR / "f13_state_quantiles.tsv"
OUTPUT_ORDERING = F13_DIR / "f13_pairwise_ordering.tsv"
OUTPUT_ATAC_COHERENCE = F13_DIR / "f13_atac_posthoc_clock_coherence.tsv"
OUTPUT_MANIFEST = F13_DIR / "f13_manifest.json"


# =====================================================================
# FROZEN DESIGN
# =====================================================================

STATE_COL = "celltype"
BARCODE_COL = "rna.bc"

STATES = [
    "TAC-1",
    "TAC-2",
    "IRS",
]

ROOT_STATE = "TAC-1"
INTERMEDIATE_STATE = "TAC-2"
TERMINAL_STATE = "IRS"

EXPECTED_CELLS = 5_050
EXPECTED_RNA_DIMS = 10
EXPECTED_ATAC_DIMS = 10

K_VALUES = [
    20,
    30,
    50,
]

PRIMARY_K = 30

DIFFMAP_COMPONENTS = 10
DPT_COMPONENTS = 8

RANDOM_SEED = 785

MIN_K_SENSITIVITY_SPEARMAN = 0.95

# These are deliberately modest ordering safeguards.
# Median order is the primary categorical requirement.
MIN_ADJACENT_ORDER_PROBABILITY = 0.60
MIN_ENDPOINT_ORDER_PROBABILITY = 0.80

ATAC_KNN_VALUES = [
    20,
    30,
    50,
]


# =====================================================================
# DISPLAY
# =====================================================================

def line(char="=", width=124):
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
    if df is None or df.empty:
        print("(empty)")
        return

    with pd.option_context(
        "display.max_rows", 300,
        "display.max_columns", None,
        "display.width", 500,
        "display.max_colwidth", 140,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string())


def utc_now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


# =====================================================================
# INPUTS
# =====================================================================

def require_inputs():
    required = [
        F12_CELLS,
        F12_RNA,
        F12_ATAC,
        F12_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required F12 input(s):\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )


def load_f12():
    cells = pd.read_csv(
        F12_CELLS,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    rna = np.load(
        F12_RNA
    ).astype(
        np.float64
    )

    atac = np.load(
        F12_ATAC
    ).astype(
        np.float64
    )

    if len(
        cells
    ) != EXPECTED_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_CELLS:,} F12 cells; "
            f"found {len(cells):,}."
        )

    if rna.shape != (
        EXPECTED_CELLS,
        EXPECTED_RNA_DIMS,
    ):
        raise RuntimeError(
            f"Unexpected RNA shape: {rna.shape}"
        )

    if atac.shape != (
        EXPECTED_CELLS,
        EXPECTED_ATAC_DIMS,
    ):
        raise RuntimeError(
            f"Unexpected ATAC shape: {atac.shape}"
        )

    if not np.isfinite(
        rna
    ).all():
        raise RuntimeError(
            "F12 RNA contains NaN/Inf."
        )

    if not np.isfinite(
        atac
    ).all():
        raise RuntimeError(
            "F12 ATAC contains NaN/Inf."
        )

    if STATE_COL not in cells.columns:
        raise RuntimeError(
            f"Missing state column: {STATE_COL}"
        )

    observed = set(
        cells[
            STATE_COL
        ].astype(str).unique()
    )

    if observed != set(
        STATES
    ):
        raise RuntimeError(
            f"Unexpected state labels: {sorted(observed)}"
        )

    return (
        cells,
        rna,
        atac,
    )


# =====================================================================
# ROOT
# =====================================================================

def select_tac1_root(
    cells,
    rna,
):
    """
    Choose a robust medoid-like TAC-1 root:
    the TAC-1 cell nearest the component-wise median of TAC-1 in RNA 10D.
    """
    mask = cells[
        STATE_COL
    ].astype(str).eq(
        ROOT_STATE
    ).to_numpy()

    candidates = np.flatnonzero(
        mask
    )

    if len(
        candidates
    ) < 100:
        raise RuntimeError(
            f"Too few TAC-1 root candidates: {len(candidates)}"
        )

    x = rna[
        candidates,
        :
    ]

    center = np.median(
        x,
        axis=0,
    )

    distances = np.linalg.norm(
        x
        - center,
        axis=1,
    )

    local = int(
        np.argmin(
            distances
        )
    )

    root_index = int(
        candidates[
            local
        ]
    )

    return (
        root_index,
        candidates,
        float(
            distances[
                local
            ]
        ),
    )


# =====================================================================
# DPT
# =====================================================================

def run_dpt(
    rna,
    root_index,
    k,
):
    """
    RNA-only DPT. ATAC is intentionally not accepted as an argument.
    """
    adata = ad.AnnData(
        X=rna.astype(
            np.float32
        )
    )

    sc.pp.neighbors(
        adata,
        n_neighbors=k,
        use_rep="X",
        metric="euclidean",
        method="gauss",
        random_state=RANDOM_SEED,
    )

    graph = adata.obsp[
        "connectivities"
    ]

    n_components, component_labels = connected_components(
        graph,
        directed=False,
    )

    adata.uns[
        "iroot"
    ] = int(
        root_index
    )

    sc.tl.diffmap(
        adata,
        n_comps=DIFFMAP_COMPONENTS,
        random_state=RANDOM_SEED,
    )

    sc.tl.dpt(
        adata,
        n_dcs=DPT_COMPONENTS,
    )

    pt = adata.obs[
        "dpt_pseudotime"
    ].to_numpy(
        dtype=float
    )

    return {
        "pseudotime": pt,
        "n_components": int(
            n_components
        ),
        "component_labels": component_labels,
        "graph": graph,
    }


# =====================================================================
# ORDERING / DISTRIBUTIONS
# =====================================================================

def state_quantiles(
    cells,
    pseudotime,
    k,
):
    rows = []

    for state in STATES:
        values = pseudotime[
            cells[
                STATE_COL
            ].astype(str).eq(
                state
            ).to_numpy()
        ]

        q = np.quantile(
            values,
            [
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
            ],
        )

        rows.append(
            {
                "k": k,
                "state": state,
                "n_cells": len(
                    values
                ),
                "q05": q[
                    0
                ],
                "q25": q[
                    1
                ],
                "median": q[
                    2
                ],
                "q75": q[
                    3
                ],
                "q95": q[
                    4
                ],
            }
        )

    return pd.DataFrame(
        rows
    )


def probability_b_greater_a(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=float,
    )

    b_sorted = np.sort(
        np.asarray(
            b,
            dtype=float,
        )
    )

    n_greater = (
        len(
            b_sorted
        )
        - np.searchsorted(
            b_sorted,
            a,
            side="right",
        )
    )

    return float(
        np.sum(
            n_greater
        )
        / (
            len(
                a
            )
            * len(
                b_sorted
            )
        )
    )


def pairwise_ordering_table(
    cells,
    pseudotime,
    k,
):
    values = {
        state: pseudotime[
            cells[
                STATE_COL
            ].astype(str).eq(
                state
            ).to_numpy()
        ]
        for state in STATES
    }

    pairs = [
        (
            ROOT_STATE,
            INTERMEDIATE_STATE,
            "adjacent",
        ),
        (
            INTERMEDIATE_STATE,
            TERMINAL_STATE,
            "adjacent",
        ),
        (
            ROOT_STATE,
            TERMINAL_STATE,
            "endpoint",
        ),
    ]

    rows = []

    for a, b, kind in pairs:
        probability = probability_b_greater_a(
            values[
                a
            ],
            values[
                b
            ],
        )

        threshold = (
            MIN_ENDPOINT_ORDER_PROBABILITY
            if kind == "endpoint"
            else MIN_ADJACENT_ORDER_PROBABILITY
        )

        rows.append(
            {
                "k": k,
                "comparison": (
                    f"P({b} > {a})"
                ),
                "type": kind,
                "probability": probability,
                "threshold": threshold,
                "passes_threshold": (
                    probability
                    >= threshold
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def median_order_ok(
    quantiles_df,
):
    medians = (
        quantiles_df
        .set_index(
            "state"
        )[
            "median"
        ]
    )

    return bool(
        medians.loc[
            ROOT_STATE
        ]
        <
        medians.loc[
            INTERMEDIATE_STATE
        ]
        <
        medians.loc[
            TERMINAL_STATE
        ]
    )


def interval_overlap_fraction(
    a_low,
    a_high,
    b_low,
    b_high,
):
    left = max(
        a_low,
        b_low,
    )

    right = min(
        a_high,
        b_high,
    )

    overlap = max(
        0.0,
        right
        - left,
    )

    union_left = min(
        a_low,
        b_low,
    )

    union_right = max(
        a_high,
        b_high,
    )

    denominator = (
        union_right
        - union_left
    )

    if denominator <= 0:
        return np.nan

    return float(
        overlap
        / denominator
    )


def adjacent_distribution_overlap(
    quantiles_df,
):
    q = quantiles_df.set_index(
        "state"
    )

    rows = []

    for a, b in [
        (
            ROOT_STATE,
            INTERMEDIATE_STATE,
        ),
        (
            INTERMEDIATE_STATE,
            TERMINAL_STATE,
        ),
    ]:
        iqr_overlap = interval_overlap_fraction(
            q.loc[
                a,
                "q25",
            ],
            q.loc[
                a,
                "q75",
            ],
            q.loc[
                b,
                "q25",
            ],
            q.loc[
                b,
                "q75",
            ],
        )

        broad_overlap = interval_overlap_fraction(
            q.loc[
                a,
                "q05",
            ],
            q.loc[
                a,
                "q95",
            ],
            q.loc[
                b,
                "q05",
            ],
            q.loc[
                b,
                "q95",
            ],
        )

        rows.append(
            {
                "transition": (
                    f"{a} -> {b}"
                ),
                "IQR_overlap_fraction": iqr_overlap,
                "q05_q95_overlap_fraction": broad_overlap,
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# RNA GRAPH LOCAL SMOOTHNESS
# =====================================================================

def graph_local_pseudotime_smoothness(
    graph,
    pseudotime,
):
    """
    Compare absolute pseudotime differences across graph edges to random
    cell-pair differences. Smaller graph/random ratio means smoother ordering.
    """
    coo = graph.tocoo()

    mask = (
        coo.row
        < coo.col
    )

    rows = coo.row[
        mask
    ]

    cols = coo.col[
        mask
    ]

    edge_diff = np.abs(
        pseudotime[
            rows
        ]
        - pseudotime[
            cols
        ]
    )

    edge_median = float(
        np.median(
            edge_diff
        )
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    n_pairs = min(
        200_000,
        max(
            10_000,
            len(
                edge_diff
            ),
        ),
    )

    a = rng.integers(
        0,
        len(
            pseudotime
        ),
        size=n_pairs,
    )

    b = rng.integers(
        0,
        len(
            pseudotime
        ),
        size=n_pairs,
    )

    random_diff = np.abs(
        pseudotime[
            a
        ]
        - pseudotime[
            b
        ]
    )

    random_median = float(
        np.median(
            random_diff
        )
    )

    ratio = (
        edge_median
        / random_median
        if random_median > 0
        else np.nan
    )

    return {
        "graph_edge_median_abs_pt_difference": edge_median,
        "random_pair_median_abs_pt_difference": random_median,
        "graph_to_random_ratio": ratio,
    }


# =====================================================================
# ATAC POST-HOC COHERENCE
# =====================================================================

def atac_posthoc_clock_coherence(
    atac,
    pseudotime,
    k,
):
    """
    Build ATAC kNN only AFTER the RNA clock is fixed.

    Measure how similar RNA-derived pseudotime values are among ATAC neighbors,
    compared with random pairs.

    This is not a clock-construction or pass/fail criterion.
    """
    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        algorithm="auto",
    )

    nn.fit(
        atac
    )

    _, indices = nn.kneighbors(
        atac,
        return_distance=True,
    )

    neighbors = indices[
        :,
        1:
    ]

    row_ids = np.repeat(
        np.arange(
            len(
                pseudotime
            )
        ),
        k,
    )

    neighbor_ids = neighbors.ravel()

    neighbor_diff = np.abs(
        pseudotime[
            row_ids
        ]
        - pseudotime[
            neighbor_ids
        ]
    )

    observed_median = float(
        np.median(
            neighbor_diff
        )
    )

    rng = np.random.default_rng(
        RANDOM_SEED
        + k
    )

    n_pairs = len(
        neighbor_diff
    )

    random_a = rng.integers(
        0,
        len(
            pseudotime
        ),
        size=n_pairs,
    )

    random_b = rng.integers(
        0,
        len(
            pseudotime
        ),
        size=n_pairs,
    )

    random_diff = np.abs(
        pseudotime[
            random_a
        ]
        - pseudotime[
            random_b
        ]
    )

    random_median = float(
        np.median(
            random_diff
        )
    )

    ratio = (
        observed_median
        / random_median
        if random_median > 0
        else np.nan
    )

    return {
        "k": k,
        "ATAC_neighbor_median_abs_RNA_pt_difference": observed_median,
        "random_pair_median_abs_RNA_pt_difference": random_median,
        "ATAC_neighbor_to_random_ratio": ratio,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F13 — SHARE-seq IRS RNA-ONLY COMMON DEVELOPMENTAL CLOCK"
    )

    print(
        "Frozen branch:"
    )

    print(
        "  TAC-1 -> TAC-2 -> IRS"
    )

    print()

    print(
        "Clock modality:"
    )

    print(
        "  RNA 10D ONLY"
    )

    print()

    print(
        "ATAC is excluded from neighbors, root selection, diffusion map, "
        "DPT, and k choice."
    )

    print()

    print(
        "Primary k = 30; sensitivity k = 20 and 50."
    )

    print()

    print(
        "No GDIS."
    )

    print(
        "No CMIL."
    )

    require_inputs()

    F13_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Load.
    # -----------------------------------------------------------------

    section(
        "1. LOAD FROZEN F12 IRS-BRANCH STATE SPACES"
    )

    (
        cells,
        rna,
        atac,
    ) = load_f12()

    print(
        f"Cells: {len(cells):,}"
    )

    print(
        f"RNA shape:  {rna.shape}"
    )

    print(
        f"ATAC shape: {atac.shape}"
    )

    subsection(
        "State counts"
    )

    state_counts = (
        cells[
            STATE_COL
        ]
        .value_counts()
        .reindex(
            STATES
        )
        .rename(
            "n_cells"
        )
        .to_frame()
    )

    print_df(
        state_counts,
        digits=0,
    )

    # -----------------------------------------------------------------
    # Root.
    # -----------------------------------------------------------------

    section(
        "2. FREEZE TAC-1 RNA-ONLY ROOT"
    )

    (
        root_index,
        root_candidates,
        root_distance,
    ) = select_tac1_root(
        cells,
        rna,
    )

    root_row = cells.iloc[
        root_index
    ]

    print(
        f"TAC-1 root candidates: {len(root_candidates):,}"
    )

    print(
        f"Frozen root row: {root_index}"
    )

    if BARCODE_COL in cells.columns:
        print(
            f"Frozen root RNA barcode: "
            f"{root_row[BARCODE_COL]}"
        )

    print(
        f"Distance to TAC-1 multivariate median: "
        f"{root_distance:.6f}"
    )

    # -----------------------------------------------------------------
    # DPT across k.
    # -----------------------------------------------------------------

    section(
        "3. RNA-ONLY DPT k-SENSITIVITY"
    )

    dpt = {}

    validation_rows = []
    quantile_tables = []
    ordering_tables = []
    overlap_tables = []

    for k in K_VALUES:
        subsection(
            f"k = {k}"
        )

        result = run_dpt(
            rna,
            root_index,
            k,
        )

        pt = result[
            "pseudotime"
        ]

        dpt[
            k
        ] = pt

        finite = bool(
            np.isfinite(
                pt
            ).all()
        )

        root_pt = float(
            pt[
                root_index
            ]
        )

        q = state_quantiles(
            cells,
            pt,
            k,
        )

        order = pairwise_ordering_table(
            cells,
            pt,
            k,
        )

        overlap = adjacent_distribution_overlap(
            q
        )

        overlap[
            "k"
        ] = k

        median_order = median_order_ok(
            q
        )

        ordering_pass = bool(
            order[
                "passes_threshold"
            ].all()
        )

        smooth = graph_local_pseudotime_smoothness(
            result[
                "graph"
            ],
            pt,
        )

        print(
            f"Connected components: {result['n_components']}"
        )

        print(
            f"Finite pseudotime: {finite}"
        )

        print(
            f"Root pseudotime: {root_pt:.8f}"
        )

        print(
            f"Median order TAC-1 < TAC-2 < IRS: {median_order}"
        )

        print()

        print(
            "State pseudotime quantiles:"
        )

        print_df(
            q.set_index(
                "state"
            ),
            digits=6,
        )

        print()

        print(
            "Pairwise ordering probabilities:"
        )

        print_df(
            order.set_index(
                "comparison"
            ),
            digits=6,
        )

        print()

        print(
            "Adjacent-state distribution overlap:"
        )

        print_df(
            overlap.set_index(
                "transition"
            ),
            digits=6,
        )

        print()

        print(
            f"RNA graph local smoothness ratio "
            f"(edge median / random-pair median): "
            f"{smooth['graph_to_random_ratio']:.6f}"
        )

        validation_rows.append(
            {
                "k": k,
                "connected_components": result[
                    "n_components"
                ],
                "finite_pseudotime": finite,
                "root_pseudotime": root_pt,
                "median_order_TAC1_TAC2_IRS": median_order,
                "pairwise_ordering_thresholds_pass": ordering_pass,
                **smooth,
            }
        )

        quantile_tables.append(
            q
        )

        ordering_tables.append(
            order
        )

        overlap_tables.append(
            overlap
        )

    validation_df = pd.DataFrame(
        validation_rows
    ).set_index(
        "k"
    )

    quantiles_df = pd.concat(
        quantile_tables,
        ignore_index=True,
    )

    ordering_df = pd.concat(
        ordering_tables,
        ignore_index=True,
    )

    overlap_df = pd.concat(
        overlap_tables,
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # Cross-k stability.
    # -----------------------------------------------------------------

    section(
        "4. CROSS-k PSEUDOTIME STABILITY"
    )

    sensitivity_rows = []

    primary_pt = dpt[
        PRIMARY_K
    ]

    for k in K_VALUES:
        if k == PRIMARY_K:
            continue

        rho, _ = spearmanr(
            primary_pt,
            dpt[
                k
            ],
        )

        sensitivity_rows.append(
            {
                "comparison": (
                    f"k{PRIMARY_K}_vs_k{k}"
                ),
                "spearman_rho": float(
                    rho
                ),
                "threshold": MIN_K_SENSITIVITY_SPEARMAN,
                "pass": (
                    float(
                        rho
                    )
                    >= MIN_K_SENSITIVITY_SPEARMAN
                ),
            }
        )

    sensitivity_df = pd.DataFrame(
        sensitivity_rows
    ).set_index(
        "comparison"
    )

    print_df(
        sensitivity_df,
        digits=6,
    )

    # -----------------------------------------------------------------
    # Freeze primary RNA clock before ATAC audit.
    # -----------------------------------------------------------------

    section(
        "5. FREEZE PRIMARY RNA k=30 CLOCK"
    )

    output = cells.copy()

    for k in K_VALUES:
        output[
            f"dpt_k{k}"
        ] = dpt[
            k
        ]

    output[
        "common_pseudotime"
    ] = primary_pt

    output[
        "is_frozen_root"
    ] = False

    output.loc[
        root_index,
        "is_frozen_root",
    ] = True

    output.to_csv(
        OUTPUT_PSEUDOTIME,
        sep="\t",
        index=False,
        compression="gzip",
    )

    print(
        OUTPUT_PSEUDOTIME
    )

    # -----------------------------------------------------------------
    # ATAC post-hoc.
    # -----------------------------------------------------------------

    section(
        "6. ATAC POST-HOC COHERENCE WITH THE ALREADY-FROZEN RNA CLOCK"
    )

    print(
        "ATAC is evaluated only now, after the RNA k=30 clock has been saved."
    )

    atac_rows = []

    for k in ATAC_KNN_VALUES:
        result = atac_posthoc_clock_coherence(
            atac,
            primary_pt,
            k,
        )

        atac_rows.append(
            result
        )

    atac_df = pd.DataFrame(
        atac_rows
    ).set_index(
        "k"
    )

    print_df(
        atac_df,
        digits=6,
    )

    # -----------------------------------------------------------------
    # Suitability gate.
    # -----------------------------------------------------------------

    section(
        "7. PHASE F13 COMMON-CLOCK SUITABILITY CHECKS"
    )

    all_connected = bool(
        (
            validation_df[
                "connected_components"
            ]
            == 1
        ).all()
    )

    all_finite = bool(
        validation_df[
            "finite_pseudotime"
        ].all()
    )

    all_median_order = bool(
        validation_df[
            "median_order_TAC1_TAC2_IRS"
        ].all()
    )

    all_probability_order = bool(
        validation_df[
            "pairwise_ordering_thresholds_pass"
        ].all()
    )

    k_stable = bool(
        sensitivity_df[
            "pass"
        ].all()
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    f"Frozen IRS branch contains exactly "
                    f"{EXPECTED_CELLS:,} cells"
                ),
                "pass": (
                    len(
                        cells
                    )
                    == EXPECTED_CELLS
                ),
            },
            {
                "criterion": (
                    "RNA kNN graph connected for k=20/30/50"
                ),
                "pass": all_connected,
            },
            {
                "criterion": (
                    "DPT finite for k=20/30/50"
                ),
                "pass": all_finite,
            },
            {
                "criterion": (
                    "State medians ordered TAC-1 < TAC-2 < IRS "
                    "for k=20/30/50"
                ),
                "pass": all_median_order,
            },
            {
                "criterion": (
                    "Pairwise ordering probability thresholds pass "
                    "for k=20/30/50"
                ),
                "pass": all_probability_order,
            },
            {
                "criterion": (
                    f"k30 pseudotime sensitivity Spearman >= "
                    f"{MIN_K_SENSITIVITY_SPEARMAN:.2f}"
                ),
                "pass": k_stable,
            },
            {
                "criterion": (
                    "Frozen root is TAC-1"
                ),
                "pass": (
                    str(
                        root_row[
                            STATE_COL
                        ]
                    )
                    == ROOT_STATE
                ),
            },
        ]
    )

    checks[
        "status"
    ] = np.where(
        checks[
            "pass"
        ],
        "PASS",
        "REVIEW",
    )

    print_df(
        checks.set_index(
            "criterion"
        ),
        digits=0,
    )

    all_pass = bool(
        checks[
            "pass"
        ].all()
    )

    # -----------------------------------------------------------------
    # Save tables.
    # -----------------------------------------------------------------

    section(
        "8. SAVE F13 VALIDATION TABLES / MANIFEST"
    )

    validation_out = validation_df.reset_index()

    sensitivity_out = sensitivity_df.reset_index()

    validation_out.to_csv(
        OUTPUT_VALIDATION,
        sep="\t",
        index=False,
    )

    quantiles_df.to_csv(
        OUTPUT_QUANTILES,
        sep="\t",
        index=False,
    )

    ordering_df.to_csv(
        OUTPUT_ORDERING,
        sep="\t",
        index=False,
    )

    atac_df.reset_index().to_csv(
        OUTPUT_ATAC_COHERENCE,
        sep="\t",
        index=False,
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F13",
        "created_utc": utc_now_iso(),
        "branch": {
            "states": STATES,
            "n_cells": len(
                cells
            ),
        },
        "clock": {
            "modality": "RNA only",
            "representation": (
                "F12 frozen technical-rescued RNA 10D"
            ),
            "primary_k": PRIMARY_K,
            "sensitivity_k": K_VALUES,
            "method": "Scanpy diffusion pseudotime",
            "metric": "euclidean",
            "neighbor_method": "gauss",
            "diffmap_components": DIFFMAP_COMPONENTS,
            "dpt_components": DPT_COMPONENTS,
            "atac_used_in_clock": False,
        },
        "root": {
            "state": ROOT_STATE,
            "rule": (
                "TAC-1 cell nearest component-wise TAC-1 median "
                "in frozen RNA 10D"
            ),
            "row": root_index,
            "rna_barcode": (
                str(
                    root_row[
                        BARCODE_COL
                    ]
                )
                if BARCODE_COL in cells.columns
                else None
            ),
            "distance_to_TAC1_median": root_distance,
        },
        "ordering_thresholds": {
            "adjacent_pair_probability": MIN_ADJACENT_ORDER_PROBABILITY,
            "endpoint_pair_probability": MIN_ENDPOINT_ORDER_PROBABILITY,
            "k_sensitivity_spearman": MIN_K_SENSITIVITY_SPEARMAN,
        },
        "validation_by_k": (
            validation_out.to_dict(
                orient="records"
            )
        ),
        "k_sensitivity": (
            sensitivity_out.to_dict(
                orient="records"
            )
        ),
        "distribution_overlap": (
            overlap_df.to_dict(
                orient="records"
            )
        ),
        "atac_posthoc_coherence": (
            atac_df.reset_index().to_dict(
                orient="records"
            )
        ),
        "checks": checks.to_dict(
            orient="records"
        ),
        "outputs": {
            "pseudotime": {
                "file": str(
                    OUTPUT_PSEUDOTIME
                ),
                "sha256": sha256_file(
                    OUTPUT_PSEUDOTIME
                ),
            },
            "validation": str(
                OUTPUT_VALIDATION
            ),
            "quantiles": str(
                OUTPUT_QUANTILES
            ),
            "ordering": str(
                OUTPUT_ORDERING
            ),
            "atac_posthoc_coherence": str(
                OUTPUT_ATAC_COHERENCE
            ),
        },
        "guardrails": {
            "gdis_calculated": False,
            "cmil_calculated": False,
            "lead_lag_tested": False,
            "atac_used_in_clock": False,
            "branch_reselected": False,
        },
    }

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            default=(
                lambda obj:
                bool(
                    obj
                )
                if isinstance(
                    obj,
                    np.bool_,
                )
                else (
                    int(
                        obj
                    )
                    if isinstance(
                        obj,
                        np.integer,
                    )
                    else (
                        float(
                            obj
                        )
                        if isinstance(
                            obj,
                            np.floating,
                        )
                        else str(
                            obj
                        )
                    )
                )
            ),
        )

        handle.write(
            "\n"
        )

    for path in [
        OUTPUT_PSEUDOTIME,
        OUTPUT_VALIDATION,
        OUTPUT_QUANTILES,
        OUTPUT_ORDERING,
        OUTPUT_ATAC_COHERENCE,
        OUTPUT_MANIFEST,
    ]:
        print(
            path
        )

    # -----------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------

    section(
        "9. PHASE F13 DECISION"
    )

    if all_pass:
        print(
            "PHASE F13 VERDICT: GO — SHARE-seq IRS RNA-DERIVED COMMON "
            "PSEUDOTIME VALIDATED AND FROZEN"
        )

        print()

        print(
            "The primary common clock is the RNA-only k=30 DPT."
        )

        print()

        print(
            "ATAC did not influence construction or validation of the "
            "primary clock."
        )

        print()

        print(
            "Next phase may apply the already-frozen GDIS settings to RNA "
            "and ATAC along this same common pseudotime."
        )

    else:
        print(
            "PHASE F13 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not calculate GDIS until the failed trajectory criterion "
            "is understood."
        )

    print()

    print(
        "No GDIS was calculated."
    )

    print(
        "No CMIL was calculated."
    )

    print(
        "No RNA-vs-ATAC lead-lag test was calculated."
    )

    line("=")


if __name__ == "__main__":
    main()

