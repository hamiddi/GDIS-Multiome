#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
26_phase_f12_shareseq_irs_freeze_rna_technical_rescue.py
=========================================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F12
---------
Freeze the TAC-1 -> TAC-2 -> IRS branch selected from the F11 RNA-only
continuity screen, objectively remove RNA depth-dominated PCA axes, and verify
that the branch continuity survives the technical rescue.

WHY IRS IS FROZEN BEFORE F12
----------------------------
F11 screened the three published TAC-derived branches using RNA topology only.

At the frozen F11 primary view (RNA 10D, k=30), IRS had:
    - the highest cross-state neighbor fraction;
    - the highest adjacent fraction among cross-state edges;
    - the lowest bypass fraction;
    - the strongest minimum required bridge fraction;
    - valid endpoint geometry.

Across all 15 RNA dimension/k sensitivity runs, IRS also retained:
    - one connected component;
    - valid endpoint geometry in every run;
    - substantially stronger minimum bridge support than Cuticle/Cortex.

Medulla failed endpoint geometry in all F11 sensitivity runs.

Thus the F12 branch choice is:
    TAC-1 -> TAC-2 -> IRS

ATAC did NOT determine the branch choice.

RNA TECHNICAL RESCUE
--------------------
F11 showed RNA PC1 was strongly technical:
    rho(PC1, detected RNA features) ~= -0.95
    rho(PC1, log1p RNA total counts) ~= -0.80

F12 recomputes these technical correlations from the raw RNA table and the
saved F11 RNA PCA.

A PCA axis is flagged when EITHER:
    |rho(axis, log1p total RNA counts)| >= 0.80
OR:
    |rho(axis, detected RNA features)| >= 0.80

This rule is frozen before pseudotime and before GDIS.

The technical rule is computed over ALL 7,197 F11 hair-lineage candidate
cells, not only the IRS branch, so the branch itself does not determine which
RNA axes are removed.

F12 then re-evaluates IRS continuity using the depth-filtered RNA
representation at:
    dimensions: 10, 20, 30, 40, all retained
    k:          20, 30, 50

PRIMARY FREEZE IF F12 PASSES
-----------------------------
Branch:
    TAC-1 -> TAC-2 -> IRS

Final RNA branch state space:
    first 10 retained RNA PCs after technical-axis filtering

Final ATAC branch state space:
    first 10 dimensions of the already depth-audited F11 ATAC LSI
    (F11 flagged no ATAC axis at |rho(depth)| >= 0.80)

These remain modality-specific.
No joint embedding is created.

IMPORTANT
---------
No pseudotime.
No DPT.
No GDIS.
No CMIL.
No RNA/ATAC lead-lag analysis.

INPUTS
------
data/GSE140203/
    GSM4156608_skin.late.anagen.rna.counts.txt.gz

data/GSE140203/representations_f11/
    f11_hair_candidate_cells.tsv.gz
    f11_rna_pca_50.npy
    f11_atac_lsi_depth_filtered.npy
    f11_branch_continuity.tsv.gz
    f11_manifest.json

OUTPUTS
-------
data/GSE140203/representations_f12/
    f12_irs_branch_cells.tsv.gz
    f12_rna_dimension_technical_qc.tsv
    f12_rna_depth_filtered.npy
    f12_rna_irs_10d.npy
    f12_atac_irs_10d.npy
    f12_irs_continuity_sensitivity.tsv.gz
    f12_manifest.json

RUN
---
    python 26_phase_f12_shareseq_irs_freeze_rna_technical_rescue.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE140203"

DATA_DIR = Path("data") / ACCESSION
F11_DIR = DATA_DIR / "representations_f11"
F12_DIR = DATA_DIR / "representations_f12"

RNA_COUNTS_FILE = DATA_DIR / "GSM4156608_skin.late.anagen.rna.counts.txt.gz"

F11_CELLS = F11_DIR / "f11_hair_candidate_cells.tsv.gz"
F11_RNA = F11_DIR / "f11_rna_pca_50.npy"
F11_ATAC = F11_DIR / "f11_atac_lsi_depth_filtered.npy"
F11_CONTINUITY = F11_DIR / "f11_branch_continuity.tsv.gz"
F11_MANIFEST = F11_DIR / "f11_manifest.json"

OUTPUT_CELLS = F12_DIR / "f12_irs_branch_cells.tsv.gz"
OUTPUT_RNA_QC = F12_DIR / "f12_rna_dimension_technical_qc.tsv"
OUTPUT_RNA_FILTERED = F12_DIR / "f12_rna_depth_filtered.npy"
OUTPUT_RNA_10D = F12_DIR / "f12_rna_irs_10d.npy"
OUTPUT_ATAC_10D = F12_DIR / "f12_atac_irs_10d.npy"
OUTPUT_CONTINUITY = F12_DIR / "f12_irs_continuity_sensitivity.tsv.gz"
OUTPUT_MANIFEST = F12_DIR / "f12_manifest.json"


# =====================================================================
# FROZEN DESIGN
# =====================================================================

F11_HAIR_CELLS = 7_197

BRANCH_NAME = "IRS"

BRANCH_STATES = [
    "TAC-1",
    "TAC-2",
    "IRS",
]

ROOT_STATE = "TAC-1"
INTERMEDIATE_STATE = "TAC-2"
TERMINAL_STATE = "IRS"

EXPECTED_BRANCH_CELLS = (
    3_370
    + 1_008
    + 672
)

RNA_TECHNICAL_RHO_THRESHOLD = 0.80

K_VALUES = [
    20,
    30,
    50,
]

PRIMARY_K = 30
FINAL_DIMS = 10

RANDOM_SEED = 785

# Rescue sensitivity evaluates the first N RETAINED dimensions.
BASE_DIMENSION_SENSITIVITY = [
    10,
    20,
    30,
    40,
]

# Predefined continuity safeguards.
MIN_CROSS_STATE_FRACTION = 0.10
MIN_ADJACENT_FRACTION_OF_CROSS = 0.65
MAX_BYPASS_FRACTION_OF_CROSS = 0.35
MIN_REQUIRED_BRIDGE_FRACTION = 0.04
MIN_CENTROID_CHAIN_RATIO = 0.55

# Rescue-vs-F11-primary material-degradation tolerances.
MAX_ABSOLUTE_DROP_CROSS_STATE = 0.05
MAX_ABSOLUTE_DROP_ADJACENT = 0.05
MAX_ABSOLUTE_INCREASE_BYPASS = 0.05
MIN_RETAINED_FRACTION_BRIDGE = 0.75


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
        "display.max_rows", 500,
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
# INPUTS / BARCODE HELPERS
# =====================================================================

def require_inputs():
    required = [
        RNA_COUNTS_FILE,
        F11_CELLS,
        F11_RNA,
        F11_ATAC,
        F11_CONTINUITY,
        F11_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required F11/F10 input(s):\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )


def normalize_rna_matrix_barcode(
    barcode,
):
    return str(
        barcode
    ).strip().replace(
        ",",
        ".",
    )


def load_f11():
    cells = pd.read_csv(
        F11_CELLS,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    rna = np.load(
        F11_RNA
    ).astype(
        np.float64
    )

    atac = np.load(
        F11_ATAC
    ).astype(
        np.float64
    )

    continuity = pd.read_csv(
        F11_CONTINUITY,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    if len(
        cells
    ) != F11_HAIR_CELLS:
        raise RuntimeError(
            f"Expected {F11_HAIR_CELLS:,} F11 cells; "
            f"found {len(cells):,}."
        )

    if rna.shape != (
        F11_HAIR_CELLS,
        50,
    ):
        raise RuntimeError(
            f"Unexpected F11 RNA shape: {rna.shape}"
        )

    if atac.shape[
        0
    ] != F11_HAIR_CELLS:
        raise RuntimeError(
            f"Unexpected F11 ATAC rows: {atac.shape}"
        )

    if atac.shape[
        1
    ] < FINAL_DIMS:
        raise RuntimeError(
            "F11 ATAC representation has fewer than 10 dimensions."
        )

    if not np.isfinite(
        rna
    ).all():
        raise RuntimeError(
            "F11 RNA contains NaN/Inf."
        )

    if not np.isfinite(
        atac
    ).all():
        raise RuntimeError(
            "F11 ATAC contains NaN/Inf."
        )

    if "rna.bc" not in cells.columns:
        raise RuntimeError(
            "F11 cell table lacks rna.bc."
        )

    if "celltype" not in cells.columns:
        raise RuntimeError(
            "F11 cell table lacks celltype."
        )

    return (
        cells,
        rna,
        atac,
        continuity,
    )


# =====================================================================
# RECOMPUTE RNA TECHNICAL COVARIATES OVER ALL 7,197 CELLS
# =====================================================================

def compute_rna_depth_covariates(
    cells,
):
    """
    Stream the RNA count table once and compute total counts + detected genes
    for the exact 7,197 F11 candidate cells.

    This does NOT reconstruct PCA and does NOT use branch labels.
    """
    with gzip.open(
        RNA_COUNTS_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        header = handle.readline().rstrip(
            "\n\r"
        )

        fields = header.split(
            "\t"
        )

        if len(
            fields
        ) < 2:
            raise RuntimeError(
                "RNA header failed TAB parsing."
            )

        matrix_barcodes = [
            normalize_rna_matrix_barcode(
                value
            )
            for value in fields[
                1:
            ]
        ]

        if len(
            matrix_barcodes
        ) != len(
            set(
                matrix_barcodes
            )
        ):
            raise RuntimeError(
                "Normalized RNA matrix barcodes are not unique."
            )

        barcode_to_col = {
            barcode: index
            for index, barcode in enumerate(
                matrix_barcodes
            )
        }

        missing = [
            barcode
            for barcode in cells[
                "rna.bc"
            ].astype(str)
            if barcode not in barcode_to_col
        ]

        if missing:
            raise RuntimeError(
                f"{len(missing)} F11 cells missing from RNA matrix."
            )

        selected_cols = np.asarray(
            [
                barcode_to_col[
                    barcode
                ]
                for barcode in cells[
                    "rna.bc"
                ].astype(str)
            ],
            dtype=np.int64,
        )

        total_counts = np.zeros(
            len(
                cells
            ),
            dtype=np.float64,
        )

        detected_features = np.zeros(
            len(
                cells
            ),
            dtype=np.int32,
        )

        parsed_genes = 0

        for line_text in handle:
            if not line_text.strip():
                continue

            if "\t" not in line_text:
                raise RuntimeError(
                    f"RNA gene row {parsed_genes + 1} has no TAB."
                )

            _, numeric_text = line_text.rstrip(
                "\n\r"
            ).split(
                "\t",
                1,
            )

            values = np.fromstring(
                numeric_text,
                sep="\t",
                dtype=np.float64,
            )

            if len(
                values
            ) != len(
                matrix_barcodes
            ):
                raise RuntimeError(
                    f"RNA row-width mismatch at gene {parsed_genes + 1}: "
                    f"{len(values)} vs {len(matrix_barcodes)}."
                )

            selected = values[
                selected_cols
            ]

            total_counts += selected

            detected_features += (
                selected
                > 0
            ).astype(
                np.int32
            )

            parsed_genes += 1

            if parsed_genes % 4000 == 0:
                print(
                    f"  streamed {parsed_genes:,} genes",
                    flush=True,
                )

    if np.any(
        total_counts <= 0
    ):
        raise RuntimeError(
            "At least one F11 candidate cell has zero RNA depth."
        )

    return (
        total_counts,
        detected_features,
        parsed_genes,
    )


# =====================================================================
# RNA TECHNICAL-AXIS RULE
# =====================================================================

def technical_axis_filter(
    rna,
    total_counts,
    detected_features,
):
    covariates = {
        "log1p_RNA_total_counts": np.log1p(
            total_counts
        ),
        "RNA_detected_features": detected_features.astype(
            np.float64
        ),
    }

    rows = []

    retained = []
    flagged = []

    for j in range(
        rna.shape[
            1
        ]
    ):
        values = rna[
            :,
            j
        ]

        row = {
            "original_dimension": j + 1,
        }

        max_abs = 0.0

        for name, covariate in covariates.items():
            rho, _ = spearmanr(
                values,
                covariate,
            )

            rho = float(
                rho
            )

            row[
                f"rho_{name}"
            ] = rho

            row[
                f"abs_rho_{name}"
            ] = abs(
                rho
            )

            max_abs = max(
                max_abs,
                abs(
                    rho
                ),
            )

        flagged_axis = (
            max_abs
            >= RNA_TECHNICAL_RHO_THRESHOLD
        )

        row[
            "max_abs_technical_rho"
        ] = max_abs

        row[
            "technical_axis"
        ] = flagged_axis

        rows.append(
            row
        )

        if flagged_axis:
            flagged.append(
                j
            )
        else:
            retained.append(
                j
            )

    filtered = rna[
        :,
        retained
    ].copy()

    table = pd.DataFrame(
        rows
    )

    return (
        filtered,
        retained,
        flagged,
        table,
    )


# =====================================================================
# CONTINUITY METRICS
# =====================================================================

def branch_neighbor_metrics(
    cells,
    representation,
    k,
):
    labels = cells[
        "celltype"
    ].astype(
        str
    ).to_numpy()

    if set(
        labels
    ) != set(
        BRANCH_STATES
    ):
        raise RuntimeError(
            "F12 branch contains unexpected state labels."
        )

    if len(
        cells
    ) <= k:
        raise RuntimeError(
            f"Too few branch cells for k={k}."
        )

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        algorithm="auto",
    )

    nn.fit(
        representation
    )

    _, indices = nn.kneighbors(
        representation,
        return_distance=True,
    )

    neighbors = indices[
        :,
        1:
    ]

    state_to_idx = {
        state: index
        for index, state in enumerate(
            BRANCH_STATES
        )
    }

    counts = np.zeros(
        (
            3,
            3,
        ),
        dtype=np.int64,
    )

    for i in range(
        len(
            labels
        )
    ):
        source = state_to_idx[
            labels[
                i
            ]
        ]

        neighbor_labels = labels[
            neighbors[
                i
            ]
        ]

        for target_label in neighbor_labels:
            target = state_to_idx[
                target_label
            ]

            counts[
                source,
                target
            ] += 1

    row_totals = counts.sum(
        axis=1,
        keepdims=True,
    )

    fractions = counts / row_totals

    total_edges = int(
        counts.sum()
    )

    self_edges = int(
        np.trace(
            counts
        )
    )

    cross_edges = (
        total_edges
        - self_edges
    )

    adjacent_cross_edges = int(
        counts[
            0,
            1
        ]
        + counts[
            1,
            0
        ]
        + counts[
            1,
            2
        ]
        + counts[
            2,
            1
        ]
    )

    bypass_edges = int(
        counts[
            0,
            2
        ]
        + counts[
            2,
            0
        ]
    )

    cross_fraction = (
        cross_edges
        / total_edges
        if total_edges
        else np.nan
    )

    adjacent_fraction = (
        adjacent_cross_edges
        / cross_edges
        if cross_edges
        else np.nan
    )

    bypass_fraction = (
        bypass_edges
        / cross_edges
        if cross_edges
        else np.nan
    )

    required_bridge = [
        fractions[
            0,
            1
        ],
        fractions[
            1,
            2
        ],
        fractions[
            2,
            1
        ],
    ]

    min_bridge = float(
        np.min(
            required_bridge
        )
    )

    # Symmetric graph connectedness.
    rows = np.repeat(
        np.arange(
            len(
                cells
            )
        ),
        k,
    )

    cols = neighbors.ravel()

    graph = sp.coo_matrix(
        (
            np.ones(
                len(
                    rows
                ),
                dtype=np.uint8,
            ),
            (
                rows,
                cols,
            ),
        ),
        shape=(
            len(
                cells
            ),
            len(
                cells
            ),
        ),
    ).tocsr()

    graph = (
        graph
        + graph.T
    )

    graph.data[:] = 1

    n_components, _ = connected_components(
        graph,
        directed=False,
    )

    centroids = {
        state: representation[
            labels
            == state
        ].mean(
            axis=0
        )
        for state in BRANCH_STATES
    }

    d12 = float(
        np.linalg.norm(
            centroids[
                ROOT_STATE
            ]
            - centroids[
                INTERMEDIATE_STATE
            ]
        )
    )

    d23 = float(
        np.linalg.norm(
            centroids[
                INTERMEDIATE_STATE
            ]
            - centroids[
                TERMINAL_STATE
            ]
        )
    )

    d13 = float(
        np.linalg.norm(
            centroids[
                ROOT_STATE
            ]
            - centroids[
                TERMINAL_STATE
            ]
        )
    )

    endpoint_largest = bool(
        d13
        >= d12
        and d13
        >= d23
    )

    chain_ratio = (
        d13
        / (
            d12
            + d23
        )
        if (
            d12
            + d23
        )
        > 0
        else np.nan
    )

    sil = float(
        silhouette_score(
            representation,
            labels,
            metric="euclidean",
            sample_size=min(
                3000,
                len(
                    labels
                ),
            ),
            random_state=RANDOM_SEED,
        )
    )

    return {
        "connected_components": int(
            n_components
        ),
        "cross_state_neighbor_fraction": float(
            cross_fraction
        ),
        "adjacent_fraction_of_cross_edges": float(
            adjacent_fraction
        ),
        "bypass_fraction_of_cross_edges": float(
            bypass_fraction
        ),
        "TAC1_to_TAC2_neighbor_fraction": float(
            fractions[
                0,
                1
            ]
        ),
        "TAC2_to_TAC1_neighbor_fraction": float(
            fractions[
                1,
                0
            ]
        ),
        "TAC2_to_IRS_neighbor_fraction": float(
            fractions[
                1,
                2
            ]
        ),
        "IRS_to_TAC2_neighbor_fraction": float(
            fractions[
                2,
                1
            ]
        ),
        "TAC1_to_IRS_bypass_neighbor_fraction": float(
            fractions[
                0,
                2
            ]
        ),
        "IRS_to_TAC1_bypass_neighbor_fraction": float(
            fractions[
                2,
                0
            ]
        ),
        "min_required_bridge_fraction": min_bridge,
        "centroid_d_TAC1_TAC2": d12,
        "centroid_d_TAC2_IRS": d23,
        "centroid_d_TAC1_IRS": d13,
        "endpoint_distance_is_largest": endpoint_largest,
        "centroid_chain_ratio": float(
            chain_ratio
        ),
        "silhouette_state_labels": sil,
    }


def continuity_sensitivity(
    branch_cells,
    branch_rna_filtered,
):
    dimension_counts = list(
        BASE_DIMENSION_SENSITIVITY
    )

    all_retained = branch_rna_filtered.shape[
        1
    ]

    if all_retained not in dimension_counts:
        dimension_counts.append(
            all_retained
        )

    rows = []

    for n_dims in dimension_counts:
        if n_dims > branch_rna_filtered.shape[
            1
        ]:
            continue

        x = branch_rna_filtered[
            :,
            :n_dims
        ]

        for k in K_VALUES:
            metrics = branch_neighbor_metrics(
                branch_cells,
                x,
                k,
            )

            rows.append(
                {
                    "n_dims": n_dims,
                    "k": k,
                    **metrics,
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
        "PHASE F12 — FREEZE TAC-1 -> TAC-2 -> IRS + RNA TECHNICAL RESCUE"
    )

    print(
        "Frozen branch selected from F11 RNA-only continuity:"
    )

    print(
        "  TAC-1 -> TAC-2 -> IRS"
    )

    print()

    print(
        "ATAC did not determine branch choice."
    )

    print()

    print(
        f"RNA technical-axis rule: "
        f"|rho(depth or detected features)| >= "
        f"{RNA_TECHNICAL_RHO_THRESHOLD:.2f}"
    )

    print()

    print(
        "No pseudotime."
    )

    print(
        "No GDIS."
    )

    print(
        "No CMIL."
    )

    require_inputs()

    F12_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Load.
    # -----------------------------------------------------------------

    section(
        "1. LOAD F11 REPRESENTATIONS AND FREEZE IRS BRANCH"
    )

    (
        cells,
        rna,
        atac,
        f11_continuity,
    ) = load_f11()

    branch_mask = cells[
        "celltype"
    ].isin(
        BRANCH_STATES
    ).to_numpy()

    branch_cells = cells.loc[
        branch_mask
    ].copy().reset_index(
        drop=True
    )

    branch_cells[
        "_f12_row"
    ] = np.arange(
        len(
            branch_cells
        ),
        dtype=int,
    )

    branch_rna_raw = rna[
        branch_mask,
        :
    ].copy()

    branch_atac = atac[
        branch_mask,
        :
    ].copy()

    print(
        f"F11 candidate cells: {len(cells):,}"
    )

    print(
        f"Frozen IRS branch cells: {len(branch_cells):,}"
    )

    subsection(
        "Frozen branch state counts"
    )

    branch_counts = (
        branch_cells[
            "celltype"
        ]
        .value_counts()
        .reindex(
            BRANCH_STATES
        )
        .rename(
            "n_cells"
        )
        .to_frame()
    )

    print_df(
        branch_counts,
        digits=0,
    )

    if len(
        branch_cells
    ) != EXPECTED_BRANCH_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_BRANCH_CELLS:,} IRS-branch cells; "
            f"found {len(branch_cells):,}."
        )

    # -----------------------------------------------------------------
    # Recompute RNA technical covariates on all F11 cells.
    # -----------------------------------------------------------------

    section(
        "2. RECOMPUTE RNA TECHNICAL COVARIATES ON ALL 7,197 CANDIDATE CELLS"
    )

    (
        total_counts,
        detected_features,
        parsed_genes,
    ) = compute_rna_depth_covariates(
        cells
    )

    print(
        f"RNA genes streamed: {parsed_genes:,}"
    )

    print(
        f"RNA total counts: "
        f"min={total_counts.min():.0f}, "
        f"median={np.median(total_counts):.0f}, "
        f"max={total_counts.max():.0f}"
    )

    print(
        f"RNA detected genes: "
        f"min={detected_features.min()}, "
        f"median={np.median(detected_features):.0f}, "
        f"max={detected_features.max()}"
    )

    # -----------------------------------------------------------------
    # Objective RNA axis filtering.
    # -----------------------------------------------------------------

    section(
        "3. OBJECTIVE RNA TECHNICAL-AXIS FILTER"
    )

    (
        rna_filtered_all,
        retained_indices,
        flagged_indices,
        rna_qc,
    ) = technical_axis_filter(
        rna,
        total_counts,
        detected_features,
    )

    rna_qc[
        "retained"
    ] = ~rna_qc[
        "technical_axis"
    ]

    rna_qc.to_csv(
        OUTPUT_RNA_QC,
        sep="\t",
        index=False,
    )

    print_df(
        rna_qc.set_index(
            "original_dimension"
        ),
        digits=4,
    )

    print()

    print(
        "Flagged RNA dimensions: "
        + (
            ", ".join(
                f"D{i + 1}"
                for i in flagged_indices
            )
            if flagged_indices
            else "<none>"
        )
    )

    print(
        f"Retained RNA dimensions: {len(retained_indices)}"
    )

    print(
        f"Filtered RNA shape: {rna_filtered_all.shape}"
    )

    np.save(
        OUTPUT_RNA_FILTERED,
        rna_filtered_all.astype(
            np.float32
        ),
    )

    # -----------------------------------------------------------------
    # Branch continuity after technical rescue.
    # -----------------------------------------------------------------

    section(
        "4. IRS BRANCH CONTINUITY AFTER RNA TECHNICAL RESCUE"
    )

    branch_rna_filtered = rna_filtered_all[
        branch_mask,
        :
    ].copy()

    continuity = continuity_sensitivity(
        branch_cells,
        branch_rna_filtered,
    )

    continuity.to_csv(
        OUTPUT_CONTINUITY,
        sep="\t",
        index=False,
        compression="gzip",
    )

    subsection(
        "All rescue sensitivity runs"
    )

    print_df(
        continuity.set_index(
            [
                "n_dims",
                "k",
            ]
        ),
        digits=6,
    )

    primary = continuity.loc[
        (
            continuity[
                "n_dims"
            ]
            == FINAL_DIMS
        )
        & (
            continuity[
                "k"
            ]
            == PRIMARY_K
        )
    ]

    if len(
        primary
    ) != 1:
        raise RuntimeError(
            "Could not identify unique F12 primary continuity row."
        )

    primary = primary.iloc[
        0
    ]

    # -----------------------------------------------------------------
    # Frozen F11 raw IRS primary reference.
    # -----------------------------------------------------------------

    section(
        "5. COMPARE RESCUED PRIMARY CONTINUITY WITH F11 RAW PRIMARY"
    )

    f11_ref = f11_continuity.loc[
        (
            f11_continuity[
                "branch"
            ]
            == BRANCH_NAME
        )
        & (
            f11_continuity[
                "modality"
            ]
            == "RNA_PCA"
        )
        & (
            f11_continuity[
                "n_dims"
            ]
            == FINAL_DIMS
        )
        & (
            f11_continuity[
                "k"
            ]
            == PRIMARY_K
        )
    ]

    if len(
        f11_ref
    ) != 1:
        raise RuntimeError(
            "Could not identify unique F11 IRS raw primary reference."
        )

    f11_ref = f11_ref.iloc[
        0
    ]

    comparison = pd.DataFrame(
        [
            {
                "metric": "cross_state_neighbor_fraction",
                "F11_raw": float(
                    f11_ref[
                        "cross_state_neighbor_fraction"
                    ]
                ),
                "F12_rescued": float(
                    primary[
                        "cross_state_neighbor_fraction"
                    ]
                ),
            },
            {
                "metric": "adjacent_fraction_of_cross_edges",
                "F11_raw": float(
                    f11_ref[
                        "adjacent_fraction_of_cross_edges"
                    ]
                ),
                "F12_rescued": float(
                    primary[
                        "adjacent_fraction_of_cross_edges"
                    ]
                ),
            },
            {
                "metric": "bypass_fraction_of_cross_edges",
                "F11_raw": float(
                    f11_ref[
                        "bypass_fraction_of_cross_edges"
                    ]
                ),
                "F12_rescued": float(
                    primary[
                        "bypass_fraction_of_cross_edges"
                    ]
                ),
            },
            {
                "metric": "min_required_bridge_fraction",
                "F11_raw": float(
                    f11_ref[
                        "min_required_bridge_fraction"
                    ]
                ),
                "F12_rescued": float(
                    primary[
                        "min_required_bridge_fraction"
                    ]
                ),
            },
            {
                "metric": "centroid_chain_ratio",
                "F11_raw": float(
                    f11_ref[
                        "centroid_chain_ratio"
                    ]
                ),
                "F12_rescued": float(
                    primary[
                        "centroid_chain_ratio"
                    ]
                ),
            },
        ]
    )

    comparison[
        "difference_rescued_minus_raw"
    ] = (
        comparison[
            "F12_rescued"
        ]
        - comparison[
            "F11_raw"
        ]
    )

    print_df(
        comparison.set_index(
            "metric"
        ),
        digits=6,
    )

    print()

    print(
        f"F11 raw endpoint distance largest: "
        f"{bool(f11_ref['endpoint_distance_is_largest'])}"
    )

    print(
        f"F12 rescued endpoint distance largest: "
        f"{bool(primary['endpoint_distance_is_largest'])}"
    )

    # -----------------------------------------------------------------
    # Rescue safeguards.
    # -----------------------------------------------------------------

    section(
        "6. PHASE F12 RESCUE / FREEZE SAFEGUARDS"
    )

    all_connected = bool(
        (
            continuity[
                "connected_components"
            ]
            == 1
        ).all()
    )

    all_cross = bool(
        (
            continuity[
                "cross_state_neighbor_fraction"
            ]
            >= MIN_CROSS_STATE_FRACTION
        ).all()
    )

    all_adjacent = bool(
        (
            continuity[
                "adjacent_fraction_of_cross_edges"
            ]
            >= MIN_ADJACENT_FRACTION_OF_CROSS
        ).all()
    )

    all_bypass = bool(
        (
            continuity[
                "bypass_fraction_of_cross_edges"
            ]
            <= MAX_BYPASS_FRACTION_OF_CROSS
        ).all()
    )

    all_bridge = bool(
        (
            continuity[
                "min_required_bridge_fraction"
            ]
            >= MIN_REQUIRED_BRIDGE_FRACTION
        ).all()
    )

    all_endpoint = bool(
        continuity[
            "endpoint_distance_is_largest"
        ].all()
    )

    all_chain = bool(
        (
            continuity[
                "centroid_chain_ratio"
            ]
            >= MIN_CENTROID_CHAIN_RATIO
        ).all()
    )

    f11_cross = float(
        f11_ref[
            "cross_state_neighbor_fraction"
        ]
    )

    f11_adjacent = float(
        f11_ref[
            "adjacent_fraction_of_cross_edges"
        ]
    )

    f11_bypass = float(
        f11_ref[
            "bypass_fraction_of_cross_edges"
        ]
    )

    f11_bridge = float(
        f11_ref[
            "min_required_bridge_fraction"
        ]
    )

    no_material_cross_drop = bool(
        primary[
            "cross_state_neighbor_fraction"
        ]
        >= (
            f11_cross
            - MAX_ABSOLUTE_DROP_CROSS_STATE
        )
    )

    no_material_adjacent_drop = bool(
        primary[
            "adjacent_fraction_of_cross_edges"
        ]
        >= (
            f11_adjacent
            - MAX_ABSOLUTE_DROP_ADJACENT
        )
    )

    no_material_bypass_increase = bool(
        primary[
            "bypass_fraction_of_cross_edges"
        ]
        <= (
            f11_bypass
            + MAX_ABSOLUTE_INCREASE_BYPASS
        )
    )

    retained_bridge = bool(
        primary[
            "min_required_bridge_fraction"
        ]
        >= (
            MIN_RETAINED_FRACTION_BRIDGE
            * f11_bridge
        )
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    f"Frozen IRS branch has exactly "
                    f"{EXPECTED_BRANCH_CELLS:,} cells"
                ),
                "pass": (
                    len(
                        branch_cells
                    )
                    == EXPECTED_BRANCH_CELLS
                ),
            },
            {
                "criterion": (
                    "RNA technical rule flags >=1 and <=5 axes"
                ),
                "pass": (
                    1
                    <= len(
                        flagged_indices
                    )
                    <= 5
                ),
            },
            {
                "criterion": (
                    "RNA technical filter retains >=40 axes"
                ),
                "pass": (
                    len(
                        retained_indices
                    )
                    >= 40
                ),
            },
            {
                "criterion": (
                    "All rescued continuity graphs connected"
                ),
                "pass": all_connected,
            },
            {
                "criterion": (
                    f"All rescued cross-state fractions >= "
                    f"{MIN_CROSS_STATE_FRACTION:.2f}"
                ),
                "pass": all_cross,
            },
            {
                "criterion": (
                    f"All rescued adjacent-cross fractions >= "
                    f"{MIN_ADJACENT_FRACTION_OF_CROSS:.2f}"
                ),
                "pass": all_adjacent,
            },
            {
                "criterion": (
                    f"All rescued bypass fractions <= "
                    f"{MAX_BYPASS_FRACTION_OF_CROSS:.2f}"
                ),
                "pass": all_bypass,
            },
            {
                "criterion": (
                    f"All rescued minimum bridge fractions >= "
                    f"{MIN_REQUIRED_BRIDGE_FRACTION:.2f}"
                ),
                "pass": all_bridge,
            },
            {
                "criterion": (
                    "Endpoint distance largest in every rescue sensitivity run"
                ),
                "pass": all_endpoint,
            },
            {
                "criterion": (
                    f"Centroid chain ratio >= "
                    f"{MIN_CENTROID_CHAIN_RATIO:.2f} in every run"
                ),
                "pass": all_chain,
            },
            {
                "criterion": (
                    "Primary rescued cross-state fraction not degraded "
                    "by >0.05 from F11 raw"
                ),
                "pass": no_material_cross_drop,
            },
            {
                "criterion": (
                    "Primary rescued adjacent fraction not degraded "
                    "by >0.05 from F11 raw"
                ),
                "pass": no_material_adjacent_drop,
            },
            {
                "criterion": (
                    "Primary rescued bypass fraction not increased "
                    "by >0.05 from F11 raw"
                ),
                "pass": no_material_bypass_increase,
            },
            {
                "criterion": (
                    "Primary rescued minimum bridge retains >=75% "
                    "of F11 raw support"
                ),
                "pass": retained_bridge,
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
    # Freeze matched 10D branch state spaces only if rescue passes.
    # -----------------------------------------------------------------

    section(
        "7. FREEZE MATCHED IRS-BRANCH STATE SPACES"
    )

    if all_pass:
        branch_rna_10d = branch_rna_filtered[
            :,
            :FINAL_DIMS
        ].astype(
            np.float32
        )

        branch_atac_10d = branch_atac[
            :,
            :FINAL_DIMS
        ].astype(
            np.float32
        )

        if branch_rna_10d.shape != (
            EXPECTED_BRANCH_CELLS,
            FINAL_DIMS,
        ):
            raise RuntimeError(
                f"Unexpected final RNA shape: {branch_rna_10d.shape}"
            )

        if branch_atac_10d.shape != (
            EXPECTED_BRANCH_CELLS,
            FINAL_DIMS,
        ):
            raise RuntimeError(
                f"Unexpected final ATAC shape: {branch_atac_10d.shape}"
            )

        if not np.isfinite(
            branch_rna_10d
        ).all():
            raise RuntimeError(
                "Final branch RNA contains NaN/Inf."
            )

        if not np.isfinite(
            branch_atac_10d
        ).all():
            raise RuntimeError(
                "Final branch ATAC contains NaN/Inf."
            )

        branch_cells.to_csv(
            OUTPUT_CELLS,
            sep="\t",
            index=False,
            compression="gzip",
        )

        np.save(
            OUTPUT_RNA_10D,
            branch_rna_10d,
        )

        np.save(
            OUTPUT_ATAC_10D,
            branch_atac_10d,
        )

        print(
            OUTPUT_CELLS
        )

        print(
            OUTPUT_RNA_10D
        )

        print(
            OUTPUT_ATAC_10D
        )

        print()

        print(
            "Final RNA 10D corresponds to retained original PCs:"
        )

        print(
            "  "
            + ", ".join(
                f"PC{i + 1}"
                for i in retained_indices[
                    :FINAL_DIMS
                ]
            )
        )

        print()

        print(
            "Final ATAC 10D corresponds to first 10 F11 depth-audited LSI axes."
        )

    else:
        print(
            "F12 safeguards did not all pass; branch-specific 10D "
            "representations were NOT frozen."
        )

    # -----------------------------------------------------------------
    # Manifest.
    # -----------------------------------------------------------------

    section(
        "8. SAVE F12 MANIFEST"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F12",
        "created_utc": utc_now_iso(),
        "branch": {
            "name": BRANCH_NAME,
            "states": BRANCH_STATES,
            "n_cells": len(
                branch_cells
            ),
            "selection_basis": (
                "F11 RNA-only continuity screen; ATAC not used for branch choice"
            ),
        },
        "rna_technical_filter": {
            "threshold_abs_spearman": RNA_TECHNICAL_RHO_THRESHOLD,
            "covariates": [
                "log1p total RNA counts",
                "RNA detected features",
            ],
            "computed_on_all_f11_candidate_cells": True,
            "flagged_original_dimensions": [
                i + 1
                for i in flagged_indices
            ],
            "retained_original_dimensions": [
                i + 1
                for i in retained_indices
            ],
            "filtered_output": str(
                OUTPUT_RNA_FILTERED
            ),
            "filtered_output_sha256": sha256_file(
                OUTPUT_RNA_FILTERED
            ),
        },
        "continuity": {
            "dimension_sensitivity": (
                sorted(
                    continuity[
                        "n_dims"
                    ].unique().tolist()
                )
            ),
            "k_sensitivity": K_VALUES,
            "primary_dims": FINAL_DIMS,
            "primary_k": PRIMARY_K,
            "output": str(
                OUTPUT_CONTINUITY
            ),
            "sha256": sha256_file(
                OUTPUT_CONTINUITY
            ),
        },
        "checks": checks.to_dict(
            orient="records"
        ),
        "frozen_outputs_created": bool(
            all_pass
        ),
        "guardrails": {
            "pseudotime_calculated": False,
            "gdis_calculated": False,
            "cmil_calculated": False,
            "lead_lag_tested": False,
            "joint_embedding_created": False,
        },
    }

    if all_pass:
        payload[
            "frozen_representations"
        ] = {
            "rna": {
                "dimensions": FINAL_DIMS,
                "original_pcs_used": [
                    i + 1
                    for i in retained_indices[
                        :FINAL_DIMS
                    ]
                ],
                "file": str(
                    OUTPUT_RNA_10D
                ),
                "sha256": sha256_file(
                    OUTPUT_RNA_10D
                ),
            },
            "atac": {
                "dimensions": FINAL_DIMS,
                "file": str(
                    OUTPUT_ATAC_10D
                ),
                "sha256": sha256_file(
                    OUTPUT_ATAC_10D
                ),
            },
            "cells": {
                "file": str(
                    OUTPUT_CELLS
                ),
                "sha256": sha256_file(
                    OUTPUT_CELLS
                ),
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

    print(
        OUTPUT_RNA_QC
    )

    print(
        OUTPUT_RNA_FILTERED
    )

    print(
        OUTPUT_CONTINUITY
    )

    print(
        OUTPUT_MANIFEST
    )

    # -----------------------------------------------------------------
    # Decision.
    # -----------------------------------------------------------------

    section(
        "9. PHASE F12 DECISION"
    )

    if all_pass:
        print(
            "PHASE F12 VERDICT: GO — IRS BRANCH AND MATCHED 10D "
            "RNA/ATAC STATE SPACES FROZEN"
        )

        print()

        print(
            "The RNA technical-axis rescue preserved IRS branch continuity."
        )

        print()

        print(
            "Next phase may construct and validate an RNA-only common "
            "pseudotime for TAC-1 -> TAC-2 -> IRS."
        )

        print()

        print(
            "ATAC must not influence the common clock."
        )

    else:
        print(
            "PHASE F12 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not construct pseudotime until the failed rescue criterion "
            "is understood."
        )

    print()

    print(
        "No pseudotime was calculated."
    )

    print(
        "No GDIS was calculated."
    )

    print(
        "No CMIL was calculated."
    )

    line("=")


if __name__ == "__main__":
    main()

