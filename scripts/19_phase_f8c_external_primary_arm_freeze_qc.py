#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
19_phase_f8c_external_primary_arm_freeze_qc.py
===============================================

Project:
GDIS-Multiome external validation

Dataset:
GSE205117

PHASE F8c
---------
Freeze the PRIMARY developmental-control arm and the final modality-specific
10-dimensional representations BEFORE external trajectory construction.

RATIONALE
---------
Phase F8b established that technical rescue is valid:

RNA:
    stage-protected within-stage sample centering greatly reduced residual
    sample structure while preserving coarse biological geometry.

ATAC:
    LSI dimensions with extreme fragment-depth association were identified
    objectively using:
        |Spearman rho(LSI_j, log1p(nFrags_atac))| >= 0.80

However, Phase F8/F8b still included:
    E8.5_CRISPR_T_WT

Although this sample is genotype WT, it belongs to the special CRISPR
perturbation/control arm paired with:
    E8.5_CRISPR_T_KO

For the PRIMARY developmental time-course trajectory, F8c therefore uses only
ordinary developmental samples and holds BOTH CRISPR samples outside the
primary trajectory.

PRIMARY DEVELOPMENTAL SAMPLES
-----------------------------
E7.5_rep1
E7.5_rep2
E7.75_rep1
E8.0_rep1
E8.0_rep2
E8.5_rep1
E8.5_rep2
E8.75_rep1
E8.75_rep2

PRIMARY BIOLOGICAL STATES
-------------------------
NMP
Paraxial_mesoderm
Somitic_mesoderm

SECONDARY ARM
-------------
E8.5_CRISPR_T_WT
E8.5_CRISPR_T_KO

FROZEN REPRESENTATION DIMENSION
-------------------------------
10 dimensions per modality.

This choice is prespecified from F8b because:
    1. 10D gives the strongest biology R2 for rescued RNA;
    2. 10D gives the strongest biology R2 for depth-filtered ATAC;
    3. sample partial R2 is already very small at 10D;
    4. adding dimensions dilutes biology R2 without materially improving
       same-biology replicate mixing;
    5. matched dimensionality avoids giving either modality greater state-space
       dimension;
    6. the discovery analysis also used matched 10D modality representations.

IMPORTANT
---------
F8c re-computes the RNA stage-protected sample correction and the ATAC
technical-axis filter AFTER removing E8.5_CRISPR_T_WT. This prevents the
special CRISPR-control arm from influencing the primary developmental
representation.

INPUTS
------
data/GSE205117/representations_geo_native/
    f8_primary_cells.tsv.gz
    f8_rna_pca_50.npy
    f8_atac_lsi_50.npy

data/GSE205117/
    GSE205117_cell_metadata.txt.gz

OUTPUTS
-------
data/GSE205117/representations_geo_native/
    f8c_primary_developmental_cells.tsv.gz
    f8c_rna_primary_10d.npy
    f8c_atac_primary_10d.npy
    f8c_atac_retained_original_dimensions.tsv
    f8c_primary_representation_manifest.json

NO TRAJECTORY ANALYSIS
----------------------
No pseudotime.
No DPT.
No GDIS.
No CMIL.

RUN
---
    python 19_phase_f8c_external_primary_arm_freeze_qc.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE205117"

DATA_DIR = Path("data") / ACCESSION
REP_DIR = DATA_DIR / "representations_geo_native"

METADATA_FILE = DATA_DIR / "GSE205117_cell_metadata.txt.gz"

F8_CELLS_FILE = REP_DIR / "f8_primary_cells.tsv.gz"
RNA_RAW_FILE = REP_DIR / "f8_rna_pca_50.npy"
ATAC_RAW_FILE = REP_DIR / "f8_atac_lsi_50.npy"

OUTPUT_CELLS = REP_DIR / "f8c_primary_developmental_cells.tsv.gz"
OUTPUT_RNA = REP_DIR / "f8c_rna_primary_10d.npy"
OUTPUT_ATAC = REP_DIR / "f8c_atac_primary_10d.npy"
OUTPUT_ATAC_DIMS = REP_DIR / "f8c_atac_retained_original_dimensions.tsv"
OUTPUT_MANIFEST = REP_DIR / "f8c_primary_representation_manifest.json"


# =====================================================================
# FROZEN DESIGN
# =====================================================================

CELL_COL = "cell"
BARCODE_COL = "barcode"
SAMPLE_COL = "sample"
STAGE_COL = "stage"
GENOTYPE_COL = "genotype"
CELLTYPE_COL = "celltype.mapped"

ATAC_DEPTH_COL = "nFrags_atac"

PRIMARY_SAMPLES = [
    "E7.5_rep1",
    "E7.5_rep2",
    "E7.75_rep1",
    "E8.0_rep1",
    "E8.0_rep2",
    "E8.5_rep1",
    "E8.5_rep2",
    "E8.75_rep1",
    "E8.75_rep2",
]

SECONDARY_CRISPR_SAMPLES = [
    "E8.5_CRISPR_T_WT",
    "E8.5_CRISPR_T_KO",
]

PRIMARY_STATES = [
    "NMP",
    "Paraxial_mesoderm",
    "Somitic_mesoderm",
]

FINAL_DIMENSIONS = 10

EXPECTED_RAW_DIMS = 50

# Same frozen technical threshold used in F8b.
ATAC_EXTREME_DEPTH_RHO = 0.80

# QC.
KNN_K = 30
MIN_GROUP_SIZE_FOR_MIXING = KNN_K + 5

MIN_CELLS_PER_STATE = 200
MIN_TOTAL_PRIMARY_CELLS = 4000

# Final representation safeguards.
MAX_SAMPLE_PARTIAL_R2 = 0.05
MIN_BIOLOGY_R2 = 0.20
MIN_RNA_GEOMETRY_RHO = 0.90


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
# LOAD / ALIGN
# =====================================================================

def require_inputs():
    required = [
        METADATA_FILE,
        F8_CELLS_FILE,
        RNA_RAW_FILE,
        ATAC_RAW_FILE,
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


def load_f8_inputs():
    cells = pd.read_csv(
        F8_CELLS_FILE,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    metadata = pd.read_csv(
        METADATA_FILE,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    metadata[SAMPLE_COL] = metadata[SAMPLE_COL].astype(
        str
    )

    metadata[BARCODE_COL] = metadata[BARCODE_COL].astype(
        str
    )

    metadata["_sample_barcode"] = (
        metadata[SAMPLE_COL]
        + "::"
        + metadata[BARCODE_COL]
    )

    if "_sample_barcode" not in cells.columns:
        raise RuntimeError(
            "F8 cell file lacks _sample_barcode."
        )

    if cells["_sample_barcode"].duplicated().any():
        raise RuntimeError(
            "Duplicate F8 cell keys."
        )

    meta_indexed = metadata.set_index(
        "_sample_barcode",
        drop=False,
    )

    missing = [
        key
        for key in cells["_sample_barcode"]
        if key not in meta_indexed.index
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} F8 cells are missing from metadata."
        )

    aligned_meta = meta_indexed.loc[
        cells["_sample_barcode"]
    ].reset_index(
        drop=True
    )

    rna = np.load(
        RNA_RAW_FILE
    ).astype(
        np.float64
    )

    atac = np.load(
        ATAC_RAW_FILE
    ).astype(
        np.float64
    )

    if rna.shape != (
        len(cells),
        EXPECTED_RAW_DIMS,
    ):
        raise RuntimeError(
            f"Unexpected RNA matrix shape: {rna.shape}"
        )

    if atac.shape != (
        len(cells),
        EXPECTED_RAW_DIMS,
    ):
        raise RuntimeError(
            f"Unexpected ATAC matrix shape: {atac.shape}"
        )

    if not np.isfinite(
        rna
    ).all():
        raise RuntimeError(
            "RNA representation contains NaN/Inf."
        )

    if not np.isfinite(
        atac
    ).all():
        raise RuntimeError(
            "ATAC representation contains NaN/Inf."
        )

    return (
        cells,
        aligned_meta,
        rna,
        atac,
    )


def subset_primary_developmental_arm(
    cells,
    metadata,
    rna,
    atac,
):
    """
    Keep only ordinary developmental WT samples.
    """
    mask = (
        metadata[SAMPLE_COL].isin(
            PRIMARY_SAMPLES
        )
        & metadata[GENOTYPE_COL].eq(
            "WT"
        )
        & metadata[CELLTYPE_COL].isin(
            PRIMARY_STATES
        )
    ).to_numpy()

    positions = np.flatnonzero(
        mask
    )

    cells_sub = cells.iloc[
        positions
    ].copy().reset_index(
        drop=True
    )

    meta_sub = metadata.iloc[
        positions
    ].copy().reset_index(
        drop=True
    )

    rna_sub = rna[
        positions,
        :
    ].copy()

    atac_sub = atac[
        positions,
        :
    ].copy()

    cells_sub[
        "_f8c_row"
    ] = np.arange(
        len(
            cells_sub
        ),
        dtype=int,
    )

    return (
        cells_sub,
        meta_sub,
        rna_sub,
        atac_sub,
    )


# =====================================================================
# RNA TECHNICAL RESCUE
# =====================================================================

def stage_protected_sample_centering(
    metadata,
    matrix,
):
    """
    For stages represented by >=2 ordinary developmental samples, shift each
    sample centroid to the stage centroid.

    No cell-type label is used in the correction.
    """
    rescued = matrix.copy()

    rows = []

    for stage, stage_indices in metadata.groupby(
        STAGE_COL,
        sort=False,
    ).groups.items():

        stage_positions = np.asarray(
            list(
                stage_indices
            ),
            dtype=int,
        )

        sample_labels = metadata.iloc[
            stage_positions
        ][SAMPLE_COL].astype(
            str
        ).to_numpy()

        unique_samples = list(
            pd.unique(
                sample_labels
            )
        )

        stage_centroid = matrix[
            stage_positions,
            :
        ].mean(
            axis=0
        )

        if len(
            unique_samples
        ) < 2:
            rows.append(
                {
                    "stage": stage,
                    "sample": unique_samples[0],
                    "n_cells": len(
                        stage_positions
                    ),
                    "correction_applied": False,
                    "shift_norm": 0.0,
                }
            )

            continue

        for sample in unique_samples:
            sample_positions = stage_positions[
                sample_labels
                == sample
            ]

            sample_centroid = matrix[
                sample_positions,
                :
            ].mean(
                axis=0
            )

            shift = (
                sample_centroid
                - stage_centroid
            )

            rescued[
                sample_positions,
                :
            ] = (
                matrix[
                    sample_positions,
                    :
                ]
                - shift
            )

            rows.append(
                {
                    "stage": stage,
                    "sample": sample,
                    "n_cells": len(
                        sample_positions
                    ),
                    "correction_applied": True,
                    "shift_norm": float(
                        np.linalg.norm(
                            shift
                        )
                    ),
                }
            )

    return (
        rescued,
        pd.DataFrame(
            rows
        ),
    )


# =====================================================================
# ATAC DEPTH FILTER
# =====================================================================

def filter_atac_depth_axes(
    metadata,
    matrix,
):
    depth = pd.to_numeric(
        metadata[
            ATAC_DEPTH_COL
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    log_depth = np.log1p(
        depth
    )

    retained = []

    rows = []

    for j in range(
        matrix.shape[
            1
        ]
    ):
        values = matrix[
            :,
            j,
        ]

        valid = (
            np.isfinite(
                values
            )
            & np.isfinite(
                log_depth
            )
        )

        rho, _ = spearmanr(
            values[
                valid
            ],
            log_depth[
                valid
            ],
        )

        rho = float(
            rho
        )

        flagged = (
            abs(
                rho
            )
            >= ATAC_EXTREME_DEPTH_RHO
        )

        rows.append(
            {
                "original_dimension": j + 1,
                "spearman_rho_log1p_nFrags": rho,
                "abs_rho": abs(
                    rho
                ),
                "depth_dominant": flagged,
            }
        )

        if not flagged:
            retained.append(
                j
            )

    filtered = matrix[
        :,
        retained,
    ].copy()

    return (
        filtered,
        retained,
        pd.DataFrame(
            rows
        ),
    )


# =====================================================================
# QC METRICS
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
                )
            ),
            encoded.to_numpy(
                dtype=float
            ),
        ]
    )


def multivariate_sse(
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


def multivariate_r2(
    y,
    x,
):
    centered = (
        y
        - y.mean(
            axis=0,
            keepdims=True,
        )
    )

    sst = float(
        np.sum(
            centered ** 2
        )
    )

    if sst <= 0:
        return np.nan

    sse = multivariate_sse(
        y,
        x,
    )

    return float(
        np.clip(
            1.0
            - sse
            / sst,
            0.0,
            1.0,
        )
    )


def partial_r2(
    y,
    reduced_x,
    full_x,
):
    reduced = multivariate_sse(
        y,
        reduced_x,
    )

    full = multivariate_sse(
        y,
        full_x,
    )

    if reduced <= 0:
        return np.nan

    return float(
        np.clip(
            (
                reduced
                - full
            )
            / reduced,
            0.0,
            1.0,
        )
    )


def same_biology_mixing(
    metadata,
    matrix,
):
    """
    Same-stage + same-cell-type neighbor mixing across ordinary replicate
    samples. Used for QC evaluation only.
    """
    labels = (
        metadata[STAGE_COL].astype(
            str
        )
        + "||"
        + metadata[CELLTYPE_COL].astype(
            str
        )
    )

    scores = []

    groups_used = 0
    cells_used = 0

    for label in pd.unique(
        labels
    ):
        positions = np.flatnonzero(
            labels.to_numpy()
            == label
        )

        if len(
            positions
        ) < MIN_GROUP_SIZE_FOR_MIXING:
            continue

        samples = metadata.iloc[
            positions
        ][SAMPLE_COL].astype(
            str
        ).to_numpy()

        if len(
            np.unique(
                samples
            )
        ) < 2:
            continue

        local = matrix[
            positions,
            :
        ].copy()

        std = local.std(
            axis=0
        )

        usable = std > 0

        if not np.any(
            usable
        ):
            continue

        local = local[
            :,
            usable
        ]

        local = (
            local
            - local.mean(
                axis=0
            )
        ) / local.std(
            axis=0
        )

        k = min(
            KNN_K,
            len(
                positions
            )
            - 1,
        )

        tree = cKDTree(
            local
        )

        _, neighbors = tree.query(
            local,
            k=k + 1,
        )

        neighbors = neighbors[
            :,
            1:
        ]

        for i in range(
            len(
                positions
            )
        ):
            neighbor_samples = samples[
                neighbors[
                    i
                ]
            ]

            scores.append(
                float(
                    np.mean(
                        neighbor_samples
                        != samples[
                            i
                        ]
                    )
                )
            )

        groups_used += 1
        cells_used += len(
            positions
        )

    return (
        float(
            np.mean(
                scores
            )
        ) if scores else np.nan,
        groups_used,
        cells_used,
    )


def representation_metrics(
    metadata,
    matrix,
):
    biology_x = design_matrix(
        metadata,
        [
            STAGE_COL,
            CELLTYPE_COL,
        ],
    )

    full_x = design_matrix(
        metadata,
        [
            STAGE_COL,
            CELLTYPE_COL,
            SAMPLE_COL,
        ],
    )

    biological_r2 = multivariate_r2(
        matrix,
        biology_x,
    )

    sample_partial = partial_r2(
        matrix,
        biology_x,
        full_x,
    )

    mixing, groups, cells = same_biology_mixing(
        metadata,
        matrix,
    )

    return {
        "biology_R2_stage_celltype": biological_r2,
        "sample_partial_R2_given_biology": sample_partial,
        "same_biology_knn_mixing": mixing,
        "mixing_groups_used": groups,
        "mixing_cells_used": cells,
    }


def centroid_geometry_rho(
    metadata,
    before,
    after,
):
    labels = (
        metadata[STAGE_COL].astype(
            str
        )
        + "||"
        + metadata[CELLTYPE_COL].astype(
            str
        )
    )

    def distance_vector(matrix):
        centroids = {}

        for label in pd.unique(
            labels
        ):
            positions = np.flatnonzero(
                labels.to_numpy()
                == label
            )

            # Ignore the two-cell E7.75 group in this preservation check.
            if len(
                positions
            ) < 20:
                continue

            centroids[
                label
            ] = matrix[
                positions,
                :
            ].mean(
                axis=0
            )

        keys = sorted(
            centroids
        )

        distances = []

        for i in range(
            len(
                keys
            )
        ):
            for j in range(
                i + 1,
                len(
                    keys
                )
            ):
                distances.append(
                    float(
                        np.linalg.norm(
                            centroids[
                                keys[
                                    i
                                ]
                            ]
                            - centroids[
                                keys[
                                    j
                                ]
                            ]
                        )
                    )
                )

        return np.asarray(
            distances,
            dtype=float,
        )

    a = distance_vector(
        before
    )

    b = distance_vector(
        after
    )

    if len(
        a
    ) != len(
        b
    ) or len(
        a
    ) < 3:
        return np.nan

    rho, _ = spearmanr(
        a,
        b,
    )

    return float(
        rho
    )


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F8c — PRIMARY DEVELOPMENTAL ARM + FINAL 10D REPRESENTATION FREEZE"
    )

    print(
        "Primary arm:"
    )

    for sample in PRIMARY_SAMPLES:
        print(
            f"  {sample}"
        )

    print()

    print(
        "Held outside primary trajectory:"
    )

    for sample in SECONDARY_CRISPR_SAMPLES:
        print(
            f"  {sample}"
        )

    print()

    print(
        "Frozen final dimensionality: 10D RNA + 10D ATAC"
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

    # -----------------------------------------------------------------
    # Load / subset.
    # -----------------------------------------------------------------

    section(
        "1. LOAD F8 MATRICES AND ISOLATE ORDINARY DEVELOPMENTAL WT ARM"
    )

    require_inputs()

    (
        f8_cells,
        f8_metadata,
        rna_raw,
        atac_raw,
    ) = load_f8_inputs()

    (
        cells,
        metadata,
        rna_primary_raw,
        atac_primary_raw,
    ) = subset_primary_developmental_arm(
        f8_cells,
        f8_metadata,
        rna_raw,
        atac_raw,
    )

    print(
        f"F8 cells before arm restriction: {len(f8_cells):,}"
    )

    print(
        f"Primary ordinary-development WT cells: {len(cells):,}"
    )

    print(
        f"Removed from primary representation: "
        f"{len(f8_cells) - len(cells):,}"
    )

    subsection(
        "Primary state counts"
    )

    state_counts = (
        metadata[
            CELLTYPE_COL
        ]
        .value_counts()
        .reindex(
            PRIMARY_STATES,
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
        "Primary state × stage"
    )

    print_df(
        pd.crosstab(
            metadata[
                CELLTYPE_COL
            ],
            metadata[
                STAGE_COL
            ],
            margins=True,
        )
    )

    subsection(
        "Primary stage × sample"
    )

    print_df(
        pd.crosstab(
            metadata[
                STAGE_COL
            ],
            metadata[
                SAMPLE_COL
            ],
            margins=True,
        )
    )

    # -----------------------------------------------------------------
    # Re-run modality-specific technical rescue on clean primary arm.
    # -----------------------------------------------------------------

    section(
        "2. RECOMPUTE RNA TECHNICAL RESCUE ON PRIMARY ARM ONLY"
    )

    (
        rna_corrected_50,
        rna_centering,
    ) = stage_protected_sample_centering(
        metadata,
        rna_primary_raw,
    )

    print_df(
        rna_centering.set_index(
            [
                "stage",
                "sample",
            ]
        ),
        digits=4,
    )

    section(
        "3. RECOMPUTE ATAC DEPTH-AXIS FILTER ON PRIMARY ARM ONLY"
    )

    (
        atac_filtered,
        retained_indices,
        atac_dimension_table,
    ) = filter_atac_depth_axes(
        metadata,
        atac_primary_raw,
    )

    print_df(
        atac_dimension_table.set_index(
            "original_dimension"
        ),
        digits=4,
    )

    flagged_original = (
        atac_dimension_table.loc[
            atac_dimension_table[
                "depth_dominant"
            ],
            "original_dimension",
        ]
        .astype(
            int
        )
        .tolist()
    )

    print()

    print(
        "Flagged depth-dominant original ATAC dimensions: "
        + (
            ", ".join(
                f"D{i}"
                for i in flagged_original
            )
            if flagged_original
            else "<none>"
        )
    )

    print(
        f"Retained ATAC dimensions: {atac_filtered.shape[1]}"
    )

    if atac_filtered.shape[
        1
    ] < FINAL_DIMENSIONS:
        raise RuntimeError(
            "Fewer than 10 ATAC dimensions remain after frozen depth filter."
        )

    # -----------------------------------------------------------------
    # Freeze matched 10D state spaces.
    # -----------------------------------------------------------------

    section(
        "4. FREEZE MATCHED 10D STATE SPACES"
    )

    rna_final = rna_corrected_50[
        :,
        :FINAL_DIMENSIONS,
    ].copy()

    atac_final = atac_filtered[
        :,
        :FINAL_DIMENSIONS,
    ].copy()

    retained_original_dimensions = [
        index + 1
        for index in retained_indices[
            :FINAL_DIMENSIONS
        ]
    ]

    print(
        f"RNA final shape:  {rna_final.shape}"
    )

    print(
        f"ATAC final shape: {atac_final.shape}"
    )

    print(
        "ATAC final 10 dimensions correspond to original LSI axes:"
    )

    print(
        "  "
        + ", ".join(
            f"D{i}"
            for i in retained_original_dimensions
        )
    )

    # -----------------------------------------------------------------
    # Final QC.
    # -----------------------------------------------------------------

    section(
        "5. FINAL 10D REPRESENTATION QC"
    )

    rna_metrics = representation_metrics(
        metadata,
        rna_final,
    )

    atac_metrics = representation_metrics(
        metadata,
        atac_final,
    )

    final_metrics = pd.DataFrame(
        [
            {
                "modality": "RNA_PRIMARY_10D",
                **rna_metrics,
            },
            {
                "modality": "ATAC_PRIMARY_10D",
                **atac_metrics,
            },
        ]
    ).set_index(
        "modality"
    )

    print_df(
        final_metrics,
        digits=4,
    )

    rna_geometry = centroid_geometry_rho(
        metadata,
        rna_primary_raw[
            :,
            :FINAL_DIMENSIONS,
        ],
        rna_final,
    )

    print()

    print(
        f"RNA raw-vs-corrected 10D centroid-distance Spearman: "
        f"{rna_geometry:.4f}"
    )

    # -----------------------------------------------------------------
    # Safeguards.
    # -----------------------------------------------------------------

    section(
        "6. F8c FREEZE SAFEGUARDS"
    )

    minimum_state = int(
        state_counts[
            "n_cells"
        ].min()
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    f"Primary ordinary-development arm has "
                    f">={MIN_TOTAL_PRIMARY_CELLS} cells"
                ),
                "pass": (
                    len(
                        cells
                    )
                    >= MIN_TOTAL_PRIMARY_CELLS
                ),
            },
            {
                "criterion": (
                    f"Each primary state has >={MIN_CELLS_PER_STATE} cells"
                ),
                "pass": (
                    minimum_state
                    >= MIN_CELLS_PER_STATE
                ),
            },
            {
                "criterion": "RNA final matrix is 10D and finite",
                "pass": (
                    rna_final.shape[
                        1
                    ]
                    == FINAL_DIMENSIONS
                    and np.isfinite(
                        rna_final
                    ).all()
                ),
            },
            {
                "criterion": "ATAC final matrix is 10D and finite",
                "pass": (
                    atac_final.shape[
                        1
                    ]
                    == FINAL_DIMENSIONS
                    and np.isfinite(
                        atac_final
                    ).all()
                ),
            },
            {
                "criterion": (
                    f"RNA sample partial R2 <= {MAX_SAMPLE_PARTIAL_R2:.2f}"
                ),
                "pass": (
                    rna_metrics[
                        "sample_partial_R2_given_biology"
                    ]
                    <= MAX_SAMPLE_PARTIAL_R2
                ),
            },
            {
                "criterion": (
                    f"ATAC sample partial R2 <= {MAX_SAMPLE_PARTIAL_R2:.2f}"
                ),
                "pass": (
                    atac_metrics[
                        "sample_partial_R2_given_biology"
                    ]
                    <= MAX_SAMPLE_PARTIAL_R2
                ),
            },
            {
                "criterion": (
                    f"RNA biology R2 >= {MIN_BIOLOGY_R2:.2f}"
                ),
                "pass": (
                    rna_metrics[
                        "biology_R2_stage_celltype"
                    ]
                    >= MIN_BIOLOGY_R2
                ),
            },
            {
                "criterion": (
                    f"ATAC biology R2 >= {MIN_BIOLOGY_R2:.2f}"
                ),
                "pass": (
                    atac_metrics[
                        "biology_R2_stage_celltype"
                    ]
                    >= MIN_BIOLOGY_R2
                ),
            },
            {
                "criterion": (
                    f"RNA coarse geometry preservation >= "
                    f"{MIN_RNA_GEOMETRY_RHO:.2f}"
                ),
                "pass": (
                    np.isfinite(
                        rna_geometry
                    )
                    and rna_geometry
                    >= MIN_RNA_GEOMETRY_RHO
                ),
            },
            {
                "criterion": (
                    "ATAC frozen depth filter retains >=10 dimensions"
                ),
                "pass": (
                    atac_filtered.shape[
                        1
                    ]
                    >= FINAL_DIMENSIONS
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
    # Save only if all safeguards pass.
    # -----------------------------------------------------------------

    section(
        "7. SAVE FROZEN PRIMARY REPRESENTATIONS"
    )

    if all_pass:
        cells_to_save = cells.copy()

        cells_to_save[
            "_f8c_row"
        ] = np.arange(
            len(
                cells_to_save
            ),
            dtype=int,
        )

        cells_to_save.to_csv(
            OUTPUT_CELLS,
            sep="\t",
            index=False,
            compression="gzip",
        )

        np.save(
            OUTPUT_RNA,
            rna_final.astype(
                np.float32
            ),
        )

        np.save(
            OUTPUT_ATAC,
            atac_final.astype(
                np.float32
            ),
        )

        dim_table = atac_dimension_table.copy()

        dim_table[
            "retained_after_depth_filter"
        ] = ~dim_table[
            "depth_dominant"
        ]

        dim_table[
            "used_in_final_10d"
        ] = dim_table[
            "original_dimension"
        ].isin(
            retained_original_dimensions
        )

        dim_table.to_csv(
            OUTPUT_ATAC_DIMS,
            sep="\t",
            index=False,
        )

        manifest = {
            "dataset_accession": ACCESSION,
            "phase": "F8c",
            "created_utc": utc_now_iso(),
            "primary_design": {
                "ordinary_developmental_samples": PRIMARY_SAMPLES,
                "held_out_secondary_crispr_samples": (
                    SECONDARY_CRISPR_SAMPLES
                ),
                "states": PRIMARY_STATES,
                "genotype": "WT",
                "n_cells": len(
                    cells
                ),
            },
            "final_representations": {
                "rna": {
                    "method": (
                        "F8 GEO-native PCA followed by stage-protected "
                        "within-stage sample centering"
                    ),
                    "dimensions": FINAL_DIMENSIONS,
                    "file": str(
                        OUTPUT_RNA
                    ),
                    "sha256": sha256_file(
                        OUTPUT_RNA
                    ),
                },
                "atac": {
                    "method": (
                        "F8 GEO-native LSI followed by removal of axes with "
                        "|rho(log1p(nFrags_atac))| >= 0.80"
                    ),
                    "dimensions": FINAL_DIMENSIONS,
                    "original_lsi_dimensions_used": (
                        retained_original_dimensions
                    ),
                    "file": str(
                        OUTPUT_ATAC
                    ),
                    "sha256": sha256_file(
                        OUTPUT_ATAC
                    ),
                },
            },
            "cell_file": {
                "file": str(
                    OUTPUT_CELLS
                ),
                "sha256": sha256_file(
                    OUTPUT_CELLS
                ),
            },
            "guardrails": {
                "pseudotime_calculated": False,
                "gdis_calculated": False,
                "cmil_calculated": False,
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

        for path in [
            OUTPUT_CELLS,
            OUTPUT_RNA,
            OUTPUT_ATAC,
            OUTPUT_ATAC_DIMS,
            OUTPUT_MANIFEST,
        ]:
            print(
                path
            )

    else:
        print(
            "Safeguards did not all pass; final files were not frozen."
        )

    # -----------------------------------------------------------------
    # Decision.
    # -----------------------------------------------------------------

    section(
        "8. PHASE F8c DECISION"
    )

    if all_pass:
        print(
            "PHASE F8c VERDICT: GO — PRIMARY DEVELOPMENTAL ARM AND "
            "MATCHED 10D REPRESENTATIONS FROZEN"
        )

        print()

        print(
            "Primary external analysis is now restricted to ordinary WT "
            "developmental samples."
        )

        print()

        print(
            "CRISPR_T_WT and CRISPR_T_KO remain outside the primary "
            "trajectory and are reserved for secondary perturbation/control "
            "analysis."
        )

        print()

        print(
            "Next phase may construct and validate the RNA-derived common "
            "developmental trajectory using these frozen cells and RNA 10D "
            "representation. ATAC must not influence the primary clock."
        )

    else:
        print(
            "PHASE F8c VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not construct pseudotime."
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

    print(
        "No discovery-dataset setting was changed."
    )

    line("=")


if __name__ == "__main__":
    main()

