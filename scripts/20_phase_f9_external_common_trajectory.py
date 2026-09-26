#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
20_phase_f9_external_common_trajectory.py
========================================

Project:
GDIS-Multiome external validation

Dataset:
GSE205117

PHASE F9
--------
Construct and validate the PRIMARY EXTERNAL COMMON DEVELOPMENTAL CLOCK.

CRITICAL DESIGN PRINCIPLE
-------------------------
The common clock is derived from RNA ONLY.

ATAC:
    - does NOT enter neighbor construction;
    - does NOT determine the root;
    - does NOT influence pseudotime;
    - is evaluated only AFTER the RNA clock is fixed, as a coherence check.

This prevents circularity in later RNA-vs-ATAC GDIS timing comparisons.

PRIMARY NMP-DERIVED ROUTE
-------------------------
The F8c frozen developmental arm contains E7.5-E8.75 ordinary WT cells.
For the NMP-derived posterior mesoderm trajectory, F9 prospectively restricts
the clock to:

    E8.0
    E8.5
    E8.75

and the states:

    NMP
        ->
    Paraxial_mesoderm
        ->
    Somitic_mesoderm

WHY E7.5 / E7.75 ARE NOT PRIMARY CLOCK ANCHORS
----------------------------------------------
Earlier mouse somites can arise through developmental routes that are distinct
from the later NMP-derived trunk-somitogenesis program. Because the external
validation question is specifically NMP -> mesoderm differentiation, E7.5 and
E7.75 cells are retained in the frozen F8c dataset but excluded from the F9
primary NMP-derived clock BEFORE pseudotime is computed.

ROOT
----
The root is selected from E8.0 NMP cells using RNA 10D only.

Root rule:
    choose the E8.0 NMP cell closest to the multivariate median of the
    E8.0 NMP RNA representation.

The same root cell is used for all neighborhood sensitivities.

PRIMARY / SENSITIVITY
---------------------
Primary:
    k = 30 nearest neighbors

Sensitivity:
    k = 20
    k = 50

METHOD
------
Scanpy diffusion pseudotime (DPT):
    - Euclidean neighbors in the frozen RNA 10D space;
    - Gaussian neighbor graph;
    - diffusion map;
    - DPT rooted at the frozen E8.0 NMP root.

VALIDATION
----------
For each k:
    - graph connectedness;
    - finite pseudotime;
    - expected state-median order:
          NMP < Paraxial_mesoderm < Somitic_mesoderm
    - state adjacent-order count;
    - stage association (DESCRIPTIVE ONLY);
    - sample partial R2 after controlling for stage + cell type.

Across k:
    - Spearman correlation of pseudotime profiles;
    - root is fixed by construction.

After the primary RNA clock is fixed:
    - RNA local-order coherence is measured;
    - ATAC local-order coherence is measured using the SAME RNA clock.
      ATAC remains observational and never modifies the clock.

IMPORTANT
---------
Stage is NOT used as an optimization target.
Stage monotonicity is NOT required as a pass criterion because NMPs persist
across developmental stages while differentiated mesoderm accumulates.

OUTPUTS
-------
data/GSE205117/representations_geo_native/
    f9_external_common_pseudotime.tsv.gz
    f9_external_common_trajectory_manifest.json

NO GDIS
-------
No GDIS.
No CMIL.
No peak pairing.
No discovery-dataset modification.

RUN
---
    python 20_phase_f9_external_common_trajectory.py

DEPENDENCIES
------------
    numpy
    pandas
    scipy
    scikit-learn
    anndata
    scanpy

If Scanpy is missing:
    conda install -c conda-forge scanpy
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

try:
    import anndata as ad
    import scanpy as sc
except ImportError as exc:
    raise SystemExit(
        "\nERROR: Phase F9 requires Scanpy and AnnData.\n\n"
        "Install them in the active environment, for example:\n\n"
        "    conda install -c conda-forge scanpy\n\n"
        "Then rerun Phase F9.\n"
    ) from exc


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE205117"

DATA_DIR = Path("data") / ACCESSION
REP_DIR = DATA_DIR / "representations_geo_native"

F8C_CELLS = REP_DIR / "f8c_primary_developmental_cells.tsv.gz"
F8C_RNA = REP_DIR / "f8c_rna_primary_10d.npy"
F8C_ATAC = REP_DIR / "f8c_atac_primary_10d.npy"
F8C_MANIFEST = REP_DIR / "f8c_primary_representation_manifest.json"

OUTPUT_PSEUDOTIME = REP_DIR / "f9_external_common_pseudotime.tsv.gz"
OUTPUT_MANIFEST = REP_DIR / "f9_external_common_trajectory_manifest.json"


# =====================================================================
# FROZEN DESIGN
# =====================================================================

SAMPLE_COL = "sample"
STAGE_COL = "stage"
CELLTYPE_COL = "celltype.mapped"
BARCODE_COL = "barcode"

TRAJECTORY_STAGES = [
    "E8.0",
    "E8.5",
    "E8.75",
]

TRAJECTORY_STATES = [
    "NMP",
    "Paraxial_mesoderm",
    "Somitic_mesoderm",
]

ROOT_STAGE = "E8.0"
ROOT_STATE = "NMP"

K_VALUES = [
    20,
    30,
    50,
]

PRIMARY_K = 30

RANDOM_SEED = 785

DIFFMAP_COMPONENTS = 15
DPT_COMPONENTS = 10

# Prespecified validation thresholds.
MIN_TRAJECTORY_CELLS = 4000
MIN_CELLS_PER_STATE = 200
MIN_ROOT_CANDIDATES = 100

MIN_K_SENSITIVITY_SPEARMAN = 0.95
MAX_SAMPLE_PARTIAL_R2 = 0.05

# Coherence ratio = observed adjacent-pseudotime distance / random-order distance.
# Smaller is better.
MAX_RNA_COHERENCE_RATIO = 0.50
MAX_ATAC_COHERENCE_RATIO = 0.75

N_RANDOM_COHERENCE_PERMUTATIONS = 100


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
        "display.max_rows", 200,
        "display.max_columns", None,
        "display.width", 390,
        "display.max_colwidth", 120,
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
# LOAD FROZEN F8c INPUTS
# =====================================================================

def require_inputs():
    required = [
        F8C_CELLS,
        F8C_RNA,
        F8C_ATAC,
        F8C_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required F8c input(s):\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )


def load_f8c():
    cells = pd.read_csv(
        F8C_CELLS,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    rna = np.load(
        F8C_RNA
    ).astype(
        np.float64
    )

    atac = np.load(
        F8C_ATAC
    ).astype(
        np.float64
    )

    if rna.shape[0] != len(
        cells
    ):
        raise RuntimeError(
            "F8c RNA/cell row mismatch."
        )

    if atac.shape[0] != len(
        cells
    ):
        raise RuntimeError(
            "F8c ATAC/cell row mismatch."
        )

    if rna.shape[1] != 10:
        raise RuntimeError(
            f"Expected frozen RNA 10D, got {rna.shape}."
        )

    if atac.shape[1] != 10:
        raise RuntimeError(
            f"Expected frozen ATAC 10D, got {atac.shape}."
        )

    if not np.isfinite(
        rna
    ).all():
        raise RuntimeError(
            "F8c RNA contains NaN/Inf."
        )

    if not np.isfinite(
        atac
    ).all():
        raise RuntimeError(
            "F8c ATAC contains NaN/Inf."
        )

    return (
        cells,
        rna,
        atac,
    )


# =====================================================================
# FREEZE F9 ROUTE
# =====================================================================

def subset_nmp_route(
    cells,
    rna,
    atac,
):
    mask = (
        cells[
            STAGE_COL
        ].isin(
            TRAJECTORY_STAGES
        )
        & cells[
            CELLTYPE_COL
        ].isin(
            TRAJECTORY_STATES
        )
    ).to_numpy()

    positions = np.flatnonzero(
        mask
    )

    route_cells = cells.iloc[
        positions
    ].copy().reset_index(
        drop=True
    )

    route_rna = rna[
        positions,
        :
    ].copy()

    route_atac = atac[
        positions,
        :
    ].copy()

    route_cells[
        "_f9_row"
    ] = np.arange(
        len(
            route_cells
        ),
        dtype=int,
    )

    return (
        route_cells,
        route_rna,
        route_atac,
    )


# =====================================================================
# ROOT
# =====================================================================

def select_root(
    cells,
    rna,
):
    root_mask = (
        cells[
            STAGE_COL
        ].eq(
            ROOT_STAGE
        )
        & cells[
            CELLTYPE_COL
        ].eq(
            ROOT_STATE
        )
    ).to_numpy()

    candidates = np.flatnonzero(
        root_mask
    )

    if len(
        candidates
    ) < MIN_ROOT_CANDIDATES:
        raise RuntimeError(
            f"Only {len(candidates)} root candidates; "
            f"expected >= {MIN_ROOT_CANDIDATES}."
        )

    candidate_matrix = rna[
        candidates,
        :
    ]

    # Multivariate median rather than mean for robustness.
    center = np.median(
        candidate_matrix,
        axis=0,
    )

    distances = np.linalg.norm(
        candidate_matrix
        - center,
        axis=1,
    )

    local_index = int(
        np.argmin(
            distances
        )
    )

    root_index = int(
        candidates[
            local_index
        ]
    )

    return (
        root_index,
        candidates,
        float(
            distances[
                local_index
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
    Run one RNA-only DPT analysis.

    ATAC is intentionally unavailable inside this function.
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
        method="gauss",
        metric="euclidean",
        random_state=RANDOM_SEED,
    )

    connectivities = adata.obsp[
        "connectivities"
    ]

    n_components, labels = connected_components(
        connectivities,
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

    pseudotime = adata.obs[
        "dpt_pseudotime"
    ].to_numpy(
        dtype=float
    )

    return {
        "pseudotime": pseudotime,
        "n_components": int(
            n_components
        ),
        "component_labels": labels,
    }


# =====================================================================
# STATE / STAGE ORDERING
# =====================================================================

def median_by_category(
    cells,
    pseudotime,
    column,
    order,
):
    df = pd.DataFrame(
        {
            column: cells[
                column
            ].astype(
                str
            ).to_numpy(),
            "pseudotime": pseudotime,
        }
    )

    medians = (
        df.groupby(
            column,
            observed=True,
        )[
            "pseudotime"
        ]
        .median()
        .reindex(
            order
        )
    )

    return medians


def state_order_audit(
    cells,
    pseudotime,
):
    medians = median_by_category(
        cells,
        pseudotime,
        CELLTYPE_COL,
        TRAJECTORY_STATES,
    )

    adjacent = []

    for left, right in zip(
        TRAJECTORY_STATES[
            :-1
        ],
        TRAJECTORY_STATES[
            1:
        ],
    ):
        left_value = medians.loc[
            left
        ]

        right_value = medians.loc[
            right
        ]

        passed = bool(
            np.isfinite(
                left_value
            )
            and np.isfinite(
                right_value
            )
            and left_value
            < right_value
        )

        adjacent.append(
            {
                "transition": (
                    f"{left} -> {right}"
                ),
                "left_median": left_value,
                "right_median": right_value,
                "ordered": passed,
            }
        )

    full_order = bool(
        all(
            row[
                "ordered"
            ]
            for row in adjacent
        )
    )

    return (
        medians,
        pd.DataFrame(
            adjacent
        ),
        full_order,
    )


# =====================================================================
# SAMPLE PARTIAL R2
# =====================================================================

def design_matrix(
    df,
    columns,
):
    encoded = pd.get_dummies(
        df[
            list(
                columns
            )
        ].astype(
            str
        ),
        drop_first=True,
        dtype=float,
    )

    return np.column_stack(
        [
            np.ones(
                len(
                    df
                ),
                dtype=float,
            ),
            encoded.to_numpy(
                dtype=float
            ),
        ]
    )


def scalar_sse(
    y,
    x,
):
    beta, _, _, _ = np.linalg.lstsq(
        x,
        y,
        rcond=None,
    )

    residual = (
        y
        - x @ beta
    )

    return float(
        np.sum(
            residual ** 2
        )
    )


def sample_partial_r2(
    cells,
    pseudotime,
):
    """
    Additional pseudotime variance explained by sample after accounting for
    stage + cell type.

    This tests within-biological-context sample structure.
    """
    reduced = design_matrix(
        cells,
        [
            STAGE_COL,
            CELLTYPE_COL,
        ],
    )

    full = design_matrix(
        cells,
        [
            STAGE_COL,
            CELLTYPE_COL,
            SAMPLE_COL,
        ],
    )

    sse_reduced = scalar_sse(
        pseudotime,
        reduced,
    )

    sse_full = scalar_sse(
        pseudotime,
        full,
    )

    if sse_reduced <= 0:
        return np.nan

    value = (
        sse_reduced
        - sse_full
    ) / sse_reduced

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


# =====================================================================
# LOCAL ORDER COHERENCE
# =====================================================================

def median_adjacent_distance(
    matrix,
    order,
):
    ordered = matrix[
        order,
        :
    ]

    deltas = np.linalg.norm(
        np.diff(
            ordered,
            axis=0,
        ),
        axis=1,
    )

    return float(
        np.median(
            deltas
        )
    )


def local_order_coherence(
    matrix,
    pseudotime,
    seed,
):
    """
    Compare adjacent distances along the RNA-derived pseudotime ordering
    against random cell orderings.

    The same pseudotime order is used for RNA and ATAC.
    """
    observed_order = np.argsort(
        pseudotime,
        kind="mergesort",
    )

    observed = median_adjacent_distance(
        matrix,
        observed_order,
    )

    rng = np.random.default_rng(
        seed
    )

    random_values = []

    base = np.arange(
        matrix.shape[
            0
        ]
    )

    for _ in range(
        N_RANDOM_COHERENCE_PERMUTATIONS
    ):
        permutation = rng.permutation(
            base
        )

        random_values.append(
            median_adjacent_distance(
                matrix,
                permutation,
            )
        )

    random_values = np.asarray(
        random_values,
        dtype=float,
    )

    random_median = float(
        np.median(
            random_values
        )
    )

    ratio = (
        observed
        / random_median
        if random_median > 0
        else np.nan
    )

    return {
        "observed_adjacent_median_distance": observed,
        "random_adjacent_median_distance": random_median,
        "coherence_ratio": ratio,
        "random_q025": float(
            np.quantile(
                random_values,
                0.025,
            )
        ),
        "random_q975": float(
            np.quantile(
                random_values,
                0.975,
            )
        ),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F9 — EXTERNAL RNA-DERIVED COMMON DEVELOPMENTAL TRAJECTORY"
    )

    print(
        "Primary NMP-derived route:"
    )

    print(
        "  E8.0 / E8.5 / E8.75"
    )

    print(
        "  NMP -> Paraxial_mesoderm -> Somitic_mesoderm"
    )

    print()

    print(
        "Clock construction uses RNA 10D ONLY."
    )

    print(
        "ATAC does not influence neighbors, rooting, or pseudotime."
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

    # -----------------------------------------------------------------
    # Load.
    # -----------------------------------------------------------------

    section(
        "1. LOAD FROZEN F8c REPRESENTATIONS"
    )

    require_inputs()

    (
        f8c_cells,
        f8c_rna,
        f8c_atac,
    ) = load_f8c()

    print(
        f"F8c frozen cells: {len(f8c_cells):,}"
    )

    print(
        f"RNA shape:  {f8c_rna.shape}"
    )

    print(
        f"ATAC shape: {f8c_atac.shape}"
    )

    # -----------------------------------------------------------------
    # Freeze route.
    # -----------------------------------------------------------------

    section(
        "2. FREEZE NMP-DERIVED TRAJECTORY SUBSET"
    )

    (
        cells,
        rna,
        atac,
    ) = subset_nmp_route(
        f8c_cells,
        f8c_rna,
        f8c_atac,
    )

    print(
        f"Trajectory cells: {len(cells):,}"
    )

    print(
        f"Excluded F8c cells from primary clock: "
        f"{len(f8c_cells) - len(cells):,}"
    )

    subsection(
        "State counts"
    )

    state_counts = (
        cells[
            CELLTYPE_COL
        ]
        .value_counts()
        .reindex(
            TRAJECTORY_STATES,
            fill_value=0,
        )
        .rename(
            "n_cells"
        )
        .to_frame()
    )

    print_df(
        state_counts
    )

    subsection(
        "State × stage"
    )

    print_df(
        pd.crosstab(
            cells[
                CELLTYPE_COL
            ],
            cells[
                STAGE_COL
            ],
            margins=True,
        )
    )

    subsection(
        "Stage × sample"
    )

    print_df(
        pd.crosstab(
            cells[
                STAGE_COL
            ],
            cells[
                SAMPLE_COL
            ],
            margins=True,
        )
    )

    # -----------------------------------------------------------------
    # Root.
    # -----------------------------------------------------------------

    section(
        "3. FREEZE RNA-ONLY ROOT"
    )

    (
        root_index,
        root_candidates,
        root_distance,
    ) = select_root(
        cells,
        rna,
    )

    root_row = cells.iloc[
        root_index
    ]

    print(
        f"Root candidate count ({ROOT_STAGE} {ROOT_STATE}): "
        f"{len(root_candidates):,}"
    )

    print(
        f"Frozen root row: {root_index}"
    )

    print(
        f"Frozen root sample: "
        f"{root_row[SAMPLE_COL]}"
    )

    print(
        f"Frozen root barcode: "
        f"{root_row[BARCODE_COL]}"
    )

    if "_sample_barcode" in cells.columns:
        print(
            f"Frozen root key: "
            f"{root_row['_sample_barcode']}"
        )

    print(
        f"Distance to E8.0 NMP multivariate median: "
        f"{root_distance:.6f}"
    )

    # -----------------------------------------------------------------
    # DPT.
    # -----------------------------------------------------------------

    section(
        "4. RNA-ONLY DPT SENSITIVITY"
    )

    dpt_results = {}

    summary_rows = []

    state_tables = {}

    for k in K_VALUES:
        subsection(
            f"k = {k}"
        )

        result = run_dpt(
            rna,
            root_index,
            k,
        )

        pseudotime = result[
            "pseudotime"
        ]

        finite = bool(
            np.isfinite(
                pseudotime
            ).all()
        )

        (
            state_medians,
            adjacent_table,
            state_order_ok,
        ) = state_order_audit(
            cells,
            pseudotime,
        )

        state_tables[
            k
        ] = {
            "medians": state_medians,
            "adjacent": adjacent_table,
        }

        sample_r2 = sample_partial_r2(
            cells,
            pseudotime,
        )

        # Stage association is deliberately descriptive, not a pass criterion.
        stage_numeric = (
            cells[
                STAGE_COL
            ]
            .map(
                {
                    "E8.0": 8.0,
                    "E8.5": 8.5,
                    "E8.75": 8.75,
                }
            )
            .to_numpy(
                dtype=float
            )
        )

        stage_rho, _ = spearmanr(
            pseudotime,
            stage_numeric,
        )

        print(
            f"Connected components: {result['n_components']}"
        )

        print(
            f"Finite pseudotime: {finite}"
        )

        print(
            f"Sample partial R2 given stage+state: "
            f"{sample_r2:.6f}"
        )

        print(
            f"Stage Spearman rho (descriptive only): "
            f"{float(stage_rho):.4f}"
        )

        print()

        print(
            "State pseudotime medians:"
        )

        for state in TRAJECTORY_STATES:
            print(
                f"  {state:20s} "
                f"{state_medians.loc[state]:.6f}"
            )

        print()

        print_df(
            adjacent_table.set_index(
                "transition"
            ),
            digits=6,
        )

        dpt_results[
            k
        ] = pseudotime

        summary_rows.append(
            {
                "k": k,
                "connected_components": result[
                    "n_components"
                ],
                "finite_pseudotime": finite,
                "state_order_ok": state_order_ok,
                "adjacent_order_passed": int(
                    adjacent_table[
                        "ordered"
                    ].sum()
                ),
                "adjacent_order_total": len(
                    adjacent_table
                ),
                "sample_partial_R2": sample_r2,
                "stage_spearman_descriptive": float(
                    stage_rho
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    ).set_index(
        "k"
    )

    # -----------------------------------------------------------------
    # k sensitivity.
    # -----------------------------------------------------------------

    section(
        "5. DPT NEIGHBORHOOD SENSITIVITY"
    )

    sensitivity_rows = []

    primary_pt = dpt_results[
        PRIMARY_K
    ]

    for k in K_VALUES:
        if k == PRIMARY_K:
            continue

        rho, _ = spearmanr(
            primary_pt,
            dpt_results[
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
                "pass_ge_0.95": (
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
    # Local coherence after clock fixed.
    # -----------------------------------------------------------------

    section(
        "6. LOCAL STATE-SPACE COHERENCE ALONG THE FIXED RNA CLOCK"
    )

    print(
        "ATAC is evaluated here only AFTER the RNA k=30 clock is fixed."
    )

    print()

    rna_coherence = local_order_coherence(
        rna,
        primary_pt,
        seed=RANDOM_SEED + 1,
    )

    atac_coherence = local_order_coherence(
        atac,
        primary_pt,
        seed=RANDOM_SEED + 2,
    )

    coherence_df = pd.DataFrame(
        [
            {
                "modality": "RNA",
                **rna_coherence,
                "threshold": MAX_RNA_COHERENCE_RATIO,
                "pass": (
                    rna_coherence[
                        "coherence_ratio"
                    ]
                    <= MAX_RNA_COHERENCE_RATIO
                ),
            },
            {
                "modality": "ATAC",
                **atac_coherence,
                "threshold": MAX_ATAC_COHERENCE_RATIO,
                "pass": (
                    atac_coherence[
                        "coherence_ratio"
                    ]
                    <= MAX_ATAC_COHERENCE_RATIO
                ),
            },
        ]
    ).set_index(
        "modality"
    )

    print_df(
        coherence_df,
        digits=6,
    )

    # -----------------------------------------------------------------
    # Save pseudotime.
    # -----------------------------------------------------------------

    section(
        "7. SAVE EXTERNAL COMMON CLOCK"
    )

    output = cells.copy()

    for k in K_VALUES:
        output[
            f"dpt_k{k}"
        ] = dpt_results[
            k
        ]

    output[
        "common_pseudotime"
    ] = dpt_results[
        PRIMARY_K
    ]

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
    # Criteria.
    # -----------------------------------------------------------------

    section(
        "8. PHASE F9 SUITABILITY CHECKS"
    )

    minimum_state_count = int(
        state_counts[
            "n_cells"
        ].min()
    )

    all_connected = bool(
        (
            summary_df[
                "connected_components"
            ]
            == 1
        ).all()
    )

    all_finite = bool(
        summary_df[
            "finite_pseudotime"
        ].all()
    )

    all_state_order = bool(
        summary_df[
            "state_order_ok"
        ].all()
    )

    all_sample_r2 = bool(
        (
            summary_df[
                "sample_partial_R2"
            ]
            <= MAX_SAMPLE_PARTIAL_R2
        ).all()
    )

    all_k_sensitive = bool(
        sensitivity_df[
            "pass_ge_0.95"
        ].all()
    )

    rna_coherence_ok = bool(
        coherence_df.loc[
            "RNA",
            "pass",
        ]
    )

    atac_coherence_ok = bool(
        coherence_df.loc[
            "ATAC",
            "pass",
        ]
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    f"Trajectory contains >= {MIN_TRAJECTORY_CELLS} cells"
                ),
                "pass": (
                    len(
                        cells
                    )
                    >= MIN_TRAJECTORY_CELLS
                ),
            },
            {
                "criterion": (
                    f"Each trajectory state contains >= "
                    f"{MIN_CELLS_PER_STATE} cells"
                ),
                "pass": (
                    minimum_state_count
                    >= MIN_CELLS_PER_STATE
                ),
            },
            {
                "criterion": (
                    f"E8.0 NMP root pool contains >= "
                    f"{MIN_ROOT_CANDIDATES} cells"
                ),
                "pass": (
                    len(
                        root_candidates
                    )
                    >= MIN_ROOT_CANDIDATES
                ),
            },
            {
                "criterion": (
                    "RNA neighbor graph connected for k=20/30/50"
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
                    "State medians ordered NMP < Paraxial < Somitic "
                    "for k=20/30/50"
                ),
                "pass": all_state_order,
            },
            {
                "criterion": (
                    f"k30 pseudotime sensitivity Spearman >= "
                    f"{MIN_K_SENSITIVITY_SPEARMAN:.2f}"
                ),
                "pass": all_k_sensitive,
            },
            {
                "criterion": (
                    f"Sample partial R2 <= {MAX_SAMPLE_PARTIAL_R2:.2f} "
                    "for k=20/30/50"
                ),
                "pass": all_sample_r2,
            },
            {
                "criterion": (
                    f"RNA local-order coherence ratio <= "
                    f"{MAX_RNA_COHERENCE_RATIO:.2f}"
                ),
                "pass": rna_coherence_ok,
            },
            {
                "criterion": (
                    f"ATAC coherence with RNA clock ratio <= "
                    f"{MAX_ATAC_COHERENCE_RATIO:.2f}"
                ),
                "pass": atac_coherence_ok,
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
    # Manifest.
    # -----------------------------------------------------------------

    manifest = {
        "dataset_accession": ACCESSION,
        "phase": "F9",
        "created_utc": utc_now_iso(),
        "clock_policy": {
            "clock_modality": "RNA only",
            "atac_in_clock_construction": False,
            "primary_k": PRIMARY_K,
            "sensitivity_k": K_VALUES,
            "method": "Scanpy diffusion pseudotime",
            "neighbor_method": "gauss",
            "metric": "euclidean",
            "diffmap_components": DIFFMAP_COMPONENTS,
            "dpt_components": DPT_COMPONENTS,
        },
        "trajectory_subset": {
            "stages": TRAJECTORY_STAGES,
            "states": TRAJECTORY_STATES,
            "n_cells": len(
                cells
            ),
            "excluded_early_stages": [
                "E7.5",
                "E7.75",
            ],
            "rationale": (
                "Primary external clock targets the later NMP-derived "
                "posterior mesoderm route; earlier somites may arise through "
                "distinct developmental routes."
            ),
        },
        "root": {
            "rule": (
                "E8.0 NMP cell closest to multivariate median in frozen RNA 10D"
            ),
            "f9_row": root_index,
            "sample": str(
                root_row[
                    SAMPLE_COL
                ]
            ),
            "barcode": str(
                root_row[
                    BARCODE_COL
                ]
            ),
            "sample_barcode": (
                str(
                    root_row[
                        "_sample_barcode"
                    ]
                )
                if "_sample_barcode" in cells.columns
                else None
            ),
        },
        "summary_by_k": (
            summary_df.reset_index()
            .to_dict(
                orient="records"
            )
        ),
        "k_sensitivity": (
            sensitivity_df.reset_index()
            .to_dict(
                orient="records"
            )
        ),
        "coherence": (
            coherence_df.reset_index()
            .to_dict(
                orient="records"
            )
        ),
        "output": {
            "pseudotime_file": str(
                OUTPUT_PSEUDOTIME
            ),
            "sha256": sha256_file(
                OUTPUT_PSEUDOTIME
            ),
        },
        "guardrails": {
            "gdis_calculated": False,
            "cmil_calculated": False,
            "atac_used_to_define_clock": False,
            "discovery_dataset_modified": False,
        },
    }

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    print()

    print(
        OUTPUT_MANIFEST
    )

    # -----------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------

    section(
        "9. PHASE F9 DECISION"
    )

    if all_pass:
        print(
            "PHASE F9 VERDICT: GO — EXTERNAL RNA-DERIVED COMMON "
            "DEVELOPMENTAL CLOCK VALIDATED"
        )

        print()

        print(
            "The primary external common pseudotime is frozen as k=30 DPT."
        )

        print()

        print(
            "ATAC did not influence construction of the clock."
        )

        print()

        print(
            "Next phase may apply the already-frozen GDIS analysis settings "
            "to RNA and ATAC along this same common pseudotime."
        )

    else:
        print(
            "PHASE F9 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not calculate external GDIS until the failed trajectory "
            "criterion is understood."
        )

    print()

    print(
        "Stage association was reported descriptively and was not used "
        "as a pass criterion."
    )

    print()

    print(
        "No GDIS was calculated."
    )

    print(
        "No CMIL was calculated."
    )

    print(
        "No discovery-dataset setting was changed."
    )

    line("=")


if __name__ == "__main__":
    main()

