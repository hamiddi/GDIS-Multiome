#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
21_phase_f9b_external_trajectory_topology_diagnostic.py
========================================================

Project:
GDIS-Multiome external validation

Dataset:
GSE205117

PHASE F9b
---------
Diagnose WHY the Phase F9 three-state RNA DPT failed the prespecified
linear-order criterion.

IMPORTANT
---------
This is a DIAGNOSTIC phase.

It does NOT:
    - change the frozen F8c representations;
    - change the F9 root;
    - change the F9 pseudotime;
    - optimize k;
    - reverse labels;
    - force state ordering;
    - calculate GDIS;
    - calculate CMIL.

SOURCE-SUPPORTED QUESTION
-------------------------
The GSE205117 study explicitly frames the experimentally validated
differentiation as:

    NMP -> somitic mesoderm fate

Phase F9 prospectively tested a stricter three-state linear assumption:

    NMP -> Paraxial_mesoderm -> Somitic_mesoderm

That stricter assumption failed reproducibly because the DPT medians were:

    NMP < Somitic_mesoderm < Paraxial_mesoderm

for k = 20, 30, 50.

F9b therefore asks whether the broad atlas label "Paraxial_mesoderm" behaves
as a mandatory linear intermediate in this RNA manifold, or whether the
three-state population is topologically overlapping/branched.

INPUTS
------
data/GSE205117/representations_geo_native/
    f8c_primary_developmental_cells.tsv.gz
    f8c_rna_primary_10d.npy
    f8c_atac_primary_10d.npy
    f9_external_common_pseudotime.tsv.gz

DIAGNOSTICS
-----------
1. State pseudotime quantiles for k=20/30/50.
2. Stage-stratified state medians for the primary k=30 clock.
3. Pairwise ordering probability:
       P(pseudotime_B > pseudotime_A)
4. RNA 10D kNN state-connectivity matrix and enrichment over state abundance.
5. Pairwise state-centroid distances in RNA and ATAC.
6. Local-order coherence using the EXISTING k=30 RNA clock for:
       all three states
       NMP + Paraxial_mesoderm
       NMP + Somitic_mesoderm
       Paraxial_mesoderm + Somitic_mesoderm
7. Source-aligned NMP -> Somitic ordering across all k.

INTERPRETATION
--------------
The script does not automatically authorize a replacement trajectory.

If the source-aligned NMP -> Somitic route is consistently ordered and
topologically coherent, the next phase may prospectively construct a NEW
RNA-only NMP -> Somitic clock and validate it independently.

No GDIS is permitted in F9b.

RUN
---
    python 21_phase_f9b_external_trajectory_topology_diagnostic.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE205117"

DATA_DIR = Path("data") / ACCESSION
REP_DIR = DATA_DIR / "representations_geo_native"

F8C_CELLS = REP_DIR / "f8c_primary_developmental_cells.tsv.gz"
F8C_RNA = REP_DIR / "f8c_rna_primary_10d.npy"
F8C_ATAC = REP_DIR / "f8c_atac_primary_10d.npy"

F9_PSEUDOTIME = REP_DIR / "f9_external_common_pseudotime.tsv.gz"

OUTPUT_DIAGNOSTIC = REP_DIR / "f9b_external_trajectory_topology_diagnostic.json"


# =====================================================================
# FROZEN F9 DESIGN
# =====================================================================

SAMPLE_COL = "sample"
STAGE_COL = "stage"
CELLTYPE_COL = "celltype.mapped"

TRAJECTORY_STAGES = [
    "E8.0",
    "E8.5",
    "E8.75",
]

STATES = [
    "NMP",
    "Paraxial_mesoderm",
    "Somitic_mesoderm",
]

K_VALUES = [
    20,
    30,
    50,
]

PRIMARY_K = 30

KNN_DIAGNOSTIC_K = 30

RANDOM_SEED = 785
N_RANDOM_COHERENCE = 100

SUBROUTES = {
    "three_state": [
        "NMP",
        "Paraxial_mesoderm",
        "Somitic_mesoderm",
    ],
    "NMP_Paraxial": [
        "NMP",
        "Paraxial_mesoderm",
    ],
    "NMP_Somitic": [
        "NMP",
        "Somitic_mesoderm",
    ],
    "Paraxial_Somitic": [
        "Paraxial_mesoderm",
        "Somitic_mesoderm",
    ],
}


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
        "display.width", 420,
        "display.max_colwidth", 120,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string())


def utc_now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


# =====================================================================
# LOAD / ALIGN
# =====================================================================

def require_inputs():
    required = [
        F8C_CELLS,
        F8C_RNA,
        F8C_ATAC,
        F9_PSEUDOTIME,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required input(s):\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )


def load_aligned_f9():
    f8c_cells = pd.read_csv(
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

    f9 = pd.read_csv(
        F9_PSEUDOTIME,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    if "_sample_barcode" not in f8c_cells.columns:
        raise RuntimeError(
            "F8c cells lack _sample_barcode."
        )

    if "_sample_barcode" not in f9.columns:
        raise RuntimeError(
            "F9 pseudotime file lacks _sample_barcode."
        )

    key_to_f8c_row = {
        key: i
        for i, key in enumerate(
            f8c_cells[
                "_sample_barcode"
            ].astype(str)
        )
    }

    missing = [
        key
        for key in f9[
            "_sample_barcode"
        ].astype(str)
        if key not in key_to_f8c_row
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} F9 cells are not present in F8c."
        )

    positions = np.asarray(
        [
            key_to_f8c_row[
                key
            ]
            for key in f9[
                "_sample_barcode"
            ].astype(str)
        ],
        dtype=int,
    )

    rna_f9 = rna[
        positions,
        :
    ]

    atac_f9 = atac[
        positions,
        :
    ]

    # Confirm the F9 file is exactly the intended stage/state subset.
    if not set(
        f9[
            STAGE_COL
        ].astype(str).unique()
    ).issubset(
        set(
            TRAJECTORY_STAGES
        )
    ):
        raise RuntimeError(
            "F9 contains a stage outside the frozen trajectory stages."
        )

    if not set(
        f9[
            CELLTYPE_COL
        ].astype(str).unique()
    ).issubset(
        set(
            STATES
        )
    ):
        raise RuntimeError(
            "F9 contains a state outside the frozen trajectory states."
        )

    return (
        f9,
        rna_f9,
        atac_f9,
    )


# =====================================================================
# PSEUDOTIME DISTRIBUTIONS
# =====================================================================

def state_quantiles(
    cells,
    pt_column,
):
    quantiles = [
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
    ]

    rows = []

    for state in STATES:
        values = cells.loc[
            cells[
                CELLTYPE_COL
            ].eq(
                state
            ),
            pt_column,
        ].to_numpy(
            dtype=float
        )

        q = np.quantile(
            values,
            quantiles,
        )

        rows.append(
            {
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


def stage_state_medians(
    cells,
    pt_column,
):
    table = pd.pivot_table(
        cells,
        index=CELLTYPE_COL,
        columns=STAGE_COL,
        values=pt_column,
        aggfunc="median",
        observed=True,
    )

    return table.reindex(
        STATES
    )


# =====================================================================
# PAIRWISE ORDERING
# =====================================================================

def pairwise_probability_b_greater_a(
    a,
    b,
):
    """
    Compute P(B > A) exactly enough without forming all nA*nB pairs.

    Sort B and for each A count B values greater than A.
    """
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

    # searchsorted(..., side="right") counts B <= A.
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


def ordering_table(
    cells,
    pt_column,
):
    pairs = [
        (
            "NMP",
            "Paraxial_mesoderm",
        ),
        (
            "NMP",
            "Somitic_mesoderm",
        ),
        (
            "Paraxial_mesoderm",
            "Somitic_mesoderm",
        ),
    ]

    rows = []

    for a_state, b_state in pairs:
        a = cells.loc[
            cells[
                CELLTYPE_COL
            ].eq(
                a_state
            ),
            pt_column,
        ].to_numpy(
            dtype=float
        )

        b = cells.loc[
            cells[
                CELLTYPE_COL
            ].eq(
                b_state
            ),
            pt_column,
        ].to_numpy(
            dtype=float
        )

        probability = pairwise_probability_b_greater_a(
            a,
            b,
        )

        rows.append(
            {
                "comparison": (
                    f"P({b_state} > {a_state})"
                ),
                "probability": probability,
                "a_median": float(
                    np.median(
                        a
                    )
                ),
                "b_median": float(
                    np.median(
                        b
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# KNN STATE CONNECTIVITY
# =====================================================================

def knn_state_connectivity(
    cells,
    rna,
):
    """
    Directed cell-level kNN state mixing in the frozen RNA 10D space.

    The table gives:
        fraction of source-state neighbor slots occupied by target state.

    Enrichment divides this observed fraction by target-state global abundance.
    """
    nn = NearestNeighbors(
        n_neighbors=KNN_DIAGNOSTIC_K + 1,
        metric="euclidean",
        algorithm="auto",
    )

    nn.fit(
        rna
    )

    neighbors = nn.kneighbors(
        rna,
        return_distance=False,
    )[
        :,
        1:
    ]

    labels = cells[
        CELLTYPE_COL
    ].astype(str).to_numpy()

    counts = pd.DataFrame(
        0,
        index=STATES,
        columns=STATES,
        dtype=np.int64,
    )

    source_totals = {
        state: 0
        for state in STATES
    }

    for i in range(
        len(
            cells
        )
    ):
        source = labels[
            i
        ]

        if source not in source_totals:
            continue

        target_labels = labels[
            neighbors[
                i
            ]
        ]

        source_totals[
            source
        ] += len(
            target_labels
        )

        for target in target_labels:
            if target in counts.columns:
                counts.loc[
                    source,
                    target,
                ] += 1

    fractions = counts.astype(
        float
    )

    for state in STATES:
        denominator = source_totals[
            state
        ]

        if denominator > 0:
            fractions.loc[
                state,
                :,
            ] /= denominator

    global_fraction = (
        cells[
            CELLTYPE_COL
        ]
        .value_counts(
            normalize=True
        )
        .reindex(
            STATES,
            fill_value=0.0,
        )
    )

    enrichment = fractions.copy()

    for target in STATES:
        denominator = global_fraction.loc[
            target
        ]

        if denominator > 0:
            enrichment.loc[
                :,
                target,
            ] /= denominator
        else:
            enrichment.loc[
                :,
                target,
            ] = np.nan

    return (
        counts,
        fractions,
        enrichment,
    )


# =====================================================================
# STATE CENTROID DISTANCES
# =====================================================================

def centroid_distance_table(
    cells,
    matrix,
    modality,
):
    centroids = {}

    for state in STATES:
        positions = np.flatnonzero(
            cells[
                CELLTYPE_COL
            ].astype(str).to_numpy()
            == state
        )

        centroids[
            state
        ] = matrix[
            positions,
            :
        ].mean(
            axis=0
        )

    rows = []

    for i in range(
        len(
            STATES
        )
    ):
        for j in range(
            i + 1,
            len(
                STATES
            )
        ):
            a = STATES[
                i
            ]

            b = STATES[
                j
            ]

            rows.append(
                {
                    "modality": modality,
                    "state_a": a,
                    "state_b": b,
                    "euclidean_centroid_distance": float(
                        np.linalg.norm(
                            centroids[
                                a
                            ]
                            - centroids[
                                b
                            ]
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
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


def coherence_for_subset(
    cells,
    matrix,
    pt_column,
    states,
    seed,
):
    mask = cells[
        CELLTYPE_COL
    ].isin(
        states
    ).to_numpy()

    subset_matrix = matrix[
        mask,
        :
    ]

    subset_pt = cells.loc[
        mask,
        pt_column,
    ].to_numpy(
        dtype=float
    )

    order = np.argsort(
        subset_pt,
        kind="mergesort",
    )

    observed = median_adjacent_distance(
        subset_matrix,
        order,
    )

    rng = np.random.default_rng(
        seed
    )

    base = np.arange(
        subset_matrix.shape[
            0
        ]
    )

    random_values = np.empty(
        N_RANDOM_COHERENCE,
        dtype=float,
    )

    for i in range(
        N_RANDOM_COHERENCE
    ):
        permutation = rng.permutation(
            base
        )

        random_values[
            i
        ] = median_adjacent_distance(
            subset_matrix,
            permutation,
        )

    random_median = float(
        np.median(
            random_values
        )
    )

    return {
        "n_cells": int(
            subset_matrix.shape[
                0
            ]
        ),
        "observed_adjacent_distance": observed,
        "random_adjacent_distance": random_median,
        "coherence_ratio": (
            observed
            / random_median
            if random_median > 0
            else np.nan
        ),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F9b — EXTERNAL TRAJECTORY TOPOLOGY DIAGNOSTIC"
    )

    print(
        "This phase diagnoses the failed three-state F9 ordering."
    )

    print()

    print(
        "Existing F9 pseudotime is NOT changed."
    )

    print(
        "Existing root is NOT changed."
    )

    print(
        "No new pseudotime is constructed."
    )

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
        "1. LOAD AND ALIGN F9 CLOCK WITH F8c RNA/ATAC"
    )

    require_inputs()

    (
        cells,
        rna,
        atac,
    ) = load_aligned_f9()

    print(
        f"F9 cells: {len(cells):,}"
    )

    print(
        f"RNA shape:  {rna.shape}"
    )

    print(
        f"ATAC shape: {atac.shape}"
    )

    # -----------------------------------------------------------------
    # Quantiles.
    # -----------------------------------------------------------------

    section(
        "2. STATE PSEUDOTIME DISTRIBUTIONS ACROSS k"
    )

    quantile_tables = {}

    for k in K_VALUES:
        column = (
            f"dpt_k{k}"
        )

        subsection(
            f"k = {k}"
        )

        table = state_quantiles(
            cells,
            column,
        )

        quantile_tables[
            k
        ] = table

        print_df(
            table.set_index(
                "state"
            ),
            digits=6,
        )

    # -----------------------------------------------------------------
    # Stage-stratified medians.
    # -----------------------------------------------------------------

    section(
        "3. STAGE-STRATIFIED STATE MEDIANS — EXISTING k=30 CLOCK"
    )

    stage_medians = stage_state_medians(
        cells,
        "dpt_k30",
    )

    print_df(
        stage_medians,
        digits=6,
    )

    # -----------------------------------------------------------------
    # Pairwise ordering probabilities.
    # -----------------------------------------------------------------

    section(
        "4. PAIRWISE PSEUDOTIME ORDERING PROBABILITIES"
    )

    ordering_tables = {}

    for k in K_VALUES:
        column = (
            f"dpt_k{k}"
        )

        subsection(
            f"k = {k}"
        )

        table = ordering_table(
            cells,
            column,
        )

        ordering_tables[
            k
        ] = table

        print_df(
            table.set_index(
                "comparison"
            ),
            digits=6,
        )

    # -----------------------------------------------------------------
    # RNA graph state topology.
    # -----------------------------------------------------------------

    section(
        "5. RNA 10D kNN STATE CONNECTIVITY — k=30"
    )

    (
        knn_counts,
        knn_fractions,
        knn_enrichment,
    ) = knn_state_connectivity(
        cells,
        rna,
    )

    subsection(
        "Directed neighbor counts"
    )

    print_df(
        knn_counts,
        digits=0,
    )

    subsection(
        "Directed neighbor fractions"
    )

    print_df(
        knn_fractions,
        digits=6,
    )

    subsection(
        "Neighbor enrichment over target-state global abundance"
    )

    print_df(
        knn_enrichment,
        digits=4,
    )

    # -----------------------------------------------------------------
    # Centroid distances.
    # -----------------------------------------------------------------

    section(
        "6. PAIRWISE STATE-CENTROID DISTANCES"
    )

    centroid_rna = centroid_distance_table(
        cells,
        rna,
        "RNA",
    )

    centroid_atac = centroid_distance_table(
        cells,
        atac,
        "ATAC",
    )

    centroid_table = pd.concat(
        [
            centroid_rna,
            centroid_atac,
        ],
        ignore_index=True,
    )

    print_df(
        centroid_table.set_index(
            [
                "modality",
                "state_a",
                "state_b",
            ]
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Subroute coherence using unchanged F9 k30 clock.
    # -----------------------------------------------------------------

    section(
        "7. EXISTING k=30 CLOCK — LOCAL COHERENCE BY SUBROUTE"
    )

    coherence_rows = []

    seed_counter = 0

    for route_name, states in SUBROUTES.items():
        for modality, matrix in [
            (
                "RNA",
                rna,
            ),
            (
                "ATAC",
                atac,
            ),
        ]:
            seed_counter += 1

            result = coherence_for_subset(
                cells,
                matrix,
                "dpt_k30",
                states,
                RANDOM_SEED
                + seed_counter,
            )

            coherence_rows.append(
                {
                    "route": route_name,
                    "states": " -> ".join(
                        states
                    ),
                    "modality": modality,
                    **result,
                }
            )

    coherence_df = pd.DataFrame(
        coherence_rows
    )

    print_df(
        coherence_df.set_index(
            [
                "route",
                "modality",
            ]
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Source-aligned route summary.
    # -----------------------------------------------------------------

    section(
        "8. SOURCE-ALIGNED NMP -> SOMITIC ROUTE SUMMARY"
    )

    source_rows = []

    for k in K_VALUES:
        table = ordering_tables[
            k
        ]

        row = table.loc[
            table[
                "comparison"
            ].eq(
                "P(Somitic_mesoderm > NMP)"
            )
        ].iloc[
            0
        ]

        nmp_median = quantile_tables[
            k
        ].set_index(
            "state"
        ).loc[
            "NMP",
            "median",
        ]

        somitic_median = quantile_tables[
            k
        ].set_index(
            "state"
        ).loc[
            "Somitic_mesoderm",
            "median",
        ]

        source_rows.append(
            {
                "k": k,
                "NMP_median": nmp_median,
                "Somitic_median": somitic_median,
                "NMP_before_Somitic": bool(
                    nmp_median
                    < somitic_median
                ),
                "P_Somitic_gt_NMP": float(
                    row[
                        "probability"
                    ]
                ),
            }
        )

    source_df = pd.DataFrame(
        source_rows
    ).set_index(
        "k"
    )

    print_df(
        source_df,
        digits=6,
    )

    source_order_all_k = bool(
        source_df[
            "NMP_before_Somitic"
        ].all()
    )

    print()

    print(
        f"NMP median precedes Somitic median for all k: "
        f"{source_order_all_k}"
    )

    # -----------------------------------------------------------------
    # Save diagnostic JSON.
    # -----------------------------------------------------------------

    section(
        "9. SAVE DIAGNOSTIC SUMMARY"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F9b",
        "created_utc": utc_now_iso(),
        "purpose": (
            "Diagnose F9 three-state ordering failure without changing "
            "pseudotime or calculating GDIS."
        ),
        "source_supported_transition": (
            "NMP -> Somitic_mesoderm"
        ),
        "existing_clock": {
            "primary_k": PRIMARY_K,
            "changed_in_f9b": False,
        },
        "source_aligned_summary": (
            source_df.reset_index()
            .to_dict(
                orient="records"
            )
        ),
        "state_quantiles": {
            str(
                k
            ): quantile_tables[
                k
            ].to_dict(
                orient="records"
            )
            for k in K_VALUES
        },
        "stage_state_medians_k30": (
            stage_medians.reset_index()
            .to_dict(
                orient="records"
            )
        ),
        "pairwise_ordering": {
            str(
                k
            ): ordering_tables[
                k
            ].to_dict(
                orient="records"
            )
            for k in K_VALUES
        },
        "knn_neighbor_fraction": (
            knn_fractions.reset_index()
            .rename(
                columns={
                    "index": "source_state"
                }
            )
            .to_dict(
                orient="records"
            )
        ),
        "knn_neighbor_enrichment": (
            knn_enrichment.reset_index()
            .rename(
                columns={
                    "index": "source_state"
                }
            )
            .to_dict(
                orient="records"
            )
        ),
        "centroid_distances": (
            centroid_table.to_dict(
                orient="records"
            )
        ),
        "subroute_coherence": (
            coherence_df.to_dict(
                orient="records"
            )
        ),
        "guardrails": {
            "new_pseudotime_constructed": False,
            "root_changed": False,
            "gdis_calculated": False,
            "cmil_calculated": False,
            "discovery_dataset_modified": False,
        },
    }

    with OUTPUT_DIAGNOSTIC.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    print(
        OUTPUT_DIAGNOSTIC
    )

    # -----------------------------------------------------------------
    # Diagnostic verdict.
    # -----------------------------------------------------------------

    section(
        "10. PHASE F9b DIAGNOSTIC VERDICT"
    )

    print(
        "F9b does NOT issue a trajectory GO for GDIS."
    )

    print()

    if source_order_all_k:
        print(
            "The independently source-supported NMP -> Somitic_mesoderm "
            "ordering is present across k=20/30/50 in the existing RNA clock."
        )

        print()

        print(
            "Use the topology/connectivity/coherence results above to decide "
            "whether a prospectively defined two-state/branch-specific F9c "
            "clock is justified."
        )

    else:
        print(
            "Even the source-supported NMP -> Somitic_mesoderm ordering is "
            "not stable across k. External trajectory construction should "
            "remain stopped."
        )

    print()

    print(
        "Do NOT calculate GDIS from the failed F9 three-state clock."
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

