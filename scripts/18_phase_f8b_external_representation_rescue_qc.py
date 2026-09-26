#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
18_phase_f8b_external_representation_rescue_qc.py
==================================================

Project:
GDIS-Multiome external validation

Dataset:
GSE205117

PHASE F8b
---------
Technical rescue / final representation QC BEFORE any trajectory construction.

INPUTS FROM F8
--------------
data/GSE205117/representations_geo_native/
    f8_primary_cells.tsv.gz
    f8_rna_pca_50.npy
    f8_atac_lsi_50.npy
    f8_reconstruction_manifest.json

FROZEN CELL UNIVERSE
--------------------
Exactly the same 6,579 paired WT cells from Phase F8:
    NMP
    Paraxial_mesoderm
    Somitic_mesoderm

No cells are added or removed in this phase.

RNA RESCUE
----------
Stage-protected within-stage sample centering in the existing 50-PC space.

For each developmental stage represented by >=2 samples:
    sample centroid -> stage centroid

Important:
    - cell-type labels are NOT used in the correction;
    - stage means are preserved;
    - E7.75 has one sample, so no correction is applied there.

ATAC RESCUE
-----------
No sample batch correction is applied to ATAC because Phase F8 showed very
small sample partial R^2.

Instead, a frozen technical-only rule is used:
    exclude any LSI dimension with
    |Spearman rho(dimension, log1p(nFrags_atac))| >= 0.80

The rule is frozen before pseudotime and before GDIS.

BIOLOGICAL PRESERVATION AUDIT
-----------------------------
Cell-type labels are used only for QC evaluation, never for correction.

The script reports:
    - biology multivariate R^2 from stage + cell type;
    - sample partial R^2 beyond stage + cell type;
    - same-stage/same-cell-type kNN sample mixing;
    - stage×cell-type centroid-distance correlation before vs after RNA rescue.

DIMENSION SENSITIVITY
---------------------
RNA:
    first 10, 20, 30, 40, 50 corrected PCs

ATAC:
    first 10, 20, 30, 40 retained LSI axes after depth filtering
    (50 only if at least 50 axes remain)

No trajectory.
No pseudotime.
No GDIS.
No CMIL.

RUN
---
    python 18_phase_f8b_external_representation_rescue_qc.py
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


ACCESSION = "GSE205117"

DATA_DIR = Path("data") / ACCESSION
REP_DIR = DATA_DIR / "representations_geo_native"

METADATA_FILE = DATA_DIR / "GSE205117_cell_metadata.txt.gz"

CELLS_FILE = REP_DIR / "f8_primary_cells.tsv.gz"
RNA_RAW_FILE = REP_DIR / "f8_rna_pca_50.npy"
ATAC_RAW_FILE = REP_DIR / "f8_atac_lsi_50.npy"
F8_MANIFEST = REP_DIR / "f8_reconstruction_manifest.json"

RNA_RESCUED_FILE = REP_DIR / "f8b_rna_pca_sample_centered_50.npy"
ATAC_RESCUED_FILE = REP_DIR / "f8b_atac_lsi_depth_filtered.npy"
ATAC_DIMENSION_FILE = REP_DIR / "f8b_atac_retained_dimensions.tsv"
F8B_MANIFEST = REP_DIR / "f8b_representation_rescue_manifest.json"

CELL_COL = "cell"
BARCODE_COL = "barcode"
SAMPLE_COL = "sample"
STAGE_COL = "stage"
CELLTYPE_COL = "celltype.mapped"

RNA_DEPTH_COL = "nCount_RNA"
RNA_FEATURE_COL = "nFeature_RNA"
ATAC_DEPTH_COL = "nFrags_atac"
ATAC_TSS_COL = "TSSEnrichment_atac"

EXPECTED_CELLS = 6579
EXPECTED_DIMS = 50

DIMENSION_SENSITIVITY = [10, 20, 30, 40, 50]

KNN_K = 30
MIN_GROUP_SIZE_FOR_MIXING = KNN_K + 5

ATAC_EXTREME_DEPTH_RHO = 0.80

STRONG_SAMPLE_PARTIAL_R2_MAX = 0.05
ACCEPTABLE_SAMPLE_PARTIAL_R2_MAX = 0.07
CAUTION_SAMPLE_PARTIAL_R2_MAX = 0.10

STRONG_MIXING_MIN = 0.70
ACCEPTABLE_MIXING_MIN = 0.60
CAUTION_MIXING_MIN = 0.40

MIN_CENTROID_GEOMETRY_SPEARMAN = 0.90


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
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def require_inputs():
    required = [
        METADATA_FILE,
        CELLS_FILE,
        RNA_RAW_FILE,
        ATAC_RAW_FILE,
        F8_MANIFEST,
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        raise RuntimeError(
            "Missing required Phase F8 input(s):\n"
            + "\n".join(f"  {path}" for path in missing)
        )


def load_aligned_data():
    cells = pd.read_csv(
        CELLS_FILE,
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

    metadata[SAMPLE_COL] = metadata[SAMPLE_COL].astype(str)
    metadata[BARCODE_COL] = metadata[BARCODE_COL].astype(str)
    metadata["_sample_barcode"] = (
        metadata[SAMPLE_COL]
        + "::"
        + metadata[BARCODE_COL]
    )

    if "_sample_barcode" not in cells.columns:
        raise RuntimeError(
            "F8 primary cell file is missing _sample_barcode."
        )

    if cells["_sample_barcode"].duplicated().any():
        raise RuntimeError(
            "Duplicate F8 primary sample+barcode keys detected."
        )

    meta_indexed = metadata.set_index(
        "_sample_barcode",
        drop=False,
    )

    missing_meta = [
        key
        for key in cells["_sample_barcode"]
        if key not in meta_indexed.index
    ]

    if missing_meta:
        raise RuntimeError(
            f"{len(missing_meta)} F8 cells are missing from metadata."
        )

    aligned_meta = meta_indexed.loc[
        cells["_sample_barcode"]
    ].reset_index(
        drop=True
    )

    for column in [
        SAMPLE_COL,
        BARCODE_COL,
        STAGE_COL,
        CELLTYPE_COL,
    ]:
        if column in cells.columns:
            left = cells[column].astype(str).to_numpy()
            right = aligned_meta[column].astype(str).to_numpy()

            if not np.array_equal(left, right):
                raise RuntimeError(
                    f"F8 cell file and metadata disagree for column: {column}"
                )

    rna = np.load(RNA_RAW_FILE)
    atac = np.load(ATAC_RAW_FILE)

    if rna.shape != (len(cells), EXPECTED_DIMS):
        raise RuntimeError(
            f"Unexpected RNA shape: {rna.shape}"
        )

    if atac.shape != (len(cells), EXPECTED_DIMS):
        raise RuntimeError(
            f"Unexpected ATAC shape: {atac.shape}"
        )

    if not np.isfinite(rna).all():
        raise RuntimeError(
            "Raw F8 RNA PCA contains NaN/Inf."
        )

    if not np.isfinite(atac).all():
        raise RuntimeError(
            "Raw F8 ATAC LSI contains NaN/Inf."
        )

    return (
        cells,
        aligned_meta,
        rna.astype(np.float64),
        atac.astype(np.float64),
    )


def stage_protected_sample_centering(
    metadata,
    matrix,
):
    rescued = matrix.copy()
    diagnostics = []

    for stage, stage_idx in metadata.groupby(
        STAGE_COL,
        sort=False,
    ).groups.items():

        stage_positions = np.asarray(
            list(stage_idx),
            dtype=int,
        )

        stage_samples = metadata.iloc[
            stage_positions
        ][SAMPLE_COL].astype(str)

        unique_samples = list(
            pd.unique(stage_samples)
        )

        stage_centroid = matrix[
            stage_positions,
            :
        ].mean(axis=0)

        if len(unique_samples) < 2:
            diagnostics.append(
                {
                    "stage": stage,
                    "sample": unique_samples[0],
                    "n_cells": len(stage_positions),
                    "correction_applied": False,
                    "centroid_shift_norm": 0.0,
                }
            )
            continue

        for sample in unique_samples:
            local_mask = (
                stage_samples.to_numpy()
                == sample
            )

            sample_positions = stage_positions[
                local_mask
            ]

            sample_centroid = matrix[
                sample_positions,
                :
            ].mean(axis=0)

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

            diagnostics.append(
                {
                    "stage": stage,
                    "sample": sample,
                    "n_cells": len(sample_positions),
                    "correction_applied": True,
                    "centroid_shift_norm": float(
                        np.linalg.norm(shift)
                    ),
                }
            )

    return rescued, pd.DataFrame(diagnostics)


def atac_depth_axis_filter(
    metadata,
    matrix,
):
    depth = pd.to_numeric(
        metadata[ATAC_DEPTH_COL],
        errors="coerce",
    ).to_numpy(dtype=float)

    log_depth = np.log1p(depth)

    rows = []
    retained = []
    flagged = []

    for j in range(matrix.shape[1]):
        values = matrix[:, j]

        valid = (
            np.isfinite(values)
            & np.isfinite(log_depth)
        )

        rho, _ = spearmanr(
            values[valid],
            log_depth[valid],
        )

        rho = float(rho)

        is_flagged = (
            abs(rho)
            >= ATAC_EXTREME_DEPTH_RHO
        )

        rows.append(
            {
                "original_dimension": j + 1,
                "spearman_rho_log1p_nFrags": rho,
                "abs_rho": abs(rho),
                "depth_dominant": is_flagged,
            }
        )

        if is_flagged:
            flagged.append(j)
        else:
            retained.append(j)

    rescued = matrix[:, retained].copy()

    return (
        rescued,
        retained,
        flagged,
        pd.DataFrame(rows),
    )


def design_matrix(
    df,
    categorical_columns,
):
    encoded = pd.get_dummies(
        df[list(categorical_columns)].astype(str),
        drop_first=True,
        dtype=float,
    )

    return np.column_stack(
        [
            np.ones(len(df)),
            encoded.to_numpy(dtype=float),
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

    residual = y - x @ beta

    return float(
        np.sum(residual ** 2)
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
        np.sum(centered ** 2)
    )

    if sst <= 0:
        return np.nan

    sse = multivariate_sse(
        y,
        x,
    )

    return float(
        np.clip(
            1.0 - sse / sst,
            0.0,
            1.0,
        )
    )


def partial_r2(
    y,
    reduced_x,
    full_x,
):
    sse_reduced = multivariate_sse(
        y,
        reduced_x,
    )

    sse_full = multivariate_sse(
        y,
        full_x,
    )

    if sse_reduced <= 0:
        return np.nan

    return float(
        np.clip(
            (
                sse_reduced
                - sse_full
            )
            / sse_reduced,
            0.0,
            1.0,
        )
    )


def same_biology_knn_mixing(
    df,
    matrix,
):
    scores = []

    group_key = (
        df[STAGE_COL].astype(str)
        + "||"
        + df[CELLTYPE_COL].astype(str)
    )

    groups_used = 0
    cells_used = 0

    for group_value in pd.unique(group_key):
        positions = np.flatnonzero(
            group_key.to_numpy()
            == group_value
        )

        if len(positions) < MIN_GROUP_SIZE_FOR_MIXING:
            continue

        sample_labels = df.iloc[
            positions
        ][SAMPLE_COL].astype(str).to_numpy()

        if len(np.unique(sample_labels)) < 2:
            continue

        local = matrix[
            positions,
            :
        ].copy()

        std = local.std(axis=0)
        usable = std > 0

        if not np.any(usable):
            continue

        local = local[:, usable]

        local = (
            local
            - local.mean(axis=0)
        ) / local.std(axis=0)

        local_k = min(
            KNN_K,
            len(positions) - 1,
        )

        tree = cKDTree(local)

        _, neighbors = tree.query(
            local,
            k=local_k + 1,
        )

        neighbors = neighbors[:, 1:]

        for i in range(len(positions)):
            neighbor_samples = sample_labels[
                neighbors[i]
            ]

            scores.append(
                float(
                    np.mean(
                        neighbor_samples
                        != sample_labels[i]
                    )
                )
            )

        groups_used += 1
        cells_used += len(positions)

    return (
        float(np.mean(scores))
        if scores
        else np.nan,
        groups_used,
        cells_used,
    )


def rating(
    sample_partial_r2,
    mixing,
):
    if (
        sample_partial_r2
        <= STRONG_SAMPLE_PARTIAL_R2_MAX
        and mixing
        >= STRONG_MIXING_MIN
    ):
        return "STRONG"

    if (
        sample_partial_r2
        <= ACCEPTABLE_SAMPLE_PARTIAL_R2_MAX
        and mixing
        >= ACCEPTABLE_MIXING_MIN
    ):
        return "ACCEPTABLE"

    if (
        sample_partial_r2
        <= CAUTION_SAMPLE_PARTIAL_R2_MAX
        and mixing
        >= CAUTION_MIXING_MIN
    ):
        return "CAUTION"

    return "REVIEW"


def structure_table(
    metadata,
    matrix,
    modality,
    dimension_counts,
):
    rows = []

    biology_x = design_matrix(
        metadata,
        [
            STAGE_COL,
            CELLTYPE_COL,
        ],
    )

    biology_sample_x = design_matrix(
        metadata,
        [
            STAGE_COL,
            CELLTYPE_COL,
            SAMPLE_COL,
        ],
    )

    for n_dims in dimension_counts:
        if n_dims > matrix.shape[1]:
            continue

        y = matrix[:, :n_dims]

        biological_r2 = multivariate_r2(
            y,
            biology_x,
        )

        sample_r2 = partial_r2(
            y,
            biology_x,
            biology_sample_x,
        )

        mixing, n_groups, n_cells = same_biology_knn_mixing(
            metadata,
            y,
        )

        rows.append(
            {
                "modality": modality,
                "n_dims": n_dims,
                "biology_R2_stage_celltype": biological_r2,
                "sample_partial_R2_given_biology": sample_r2,
                "bio_to_sample_ratio": (
                    biological_r2 / sample_r2
                    if sample_r2 > 0
                    else np.inf
                ),
                "same_biology_knn_mixing": mixing,
                "mixing_groups_used": n_groups,
                "mixing_cells_used": n_cells,
                "rating": rating(
                    sample_r2,
                    mixing,
                ),
            }
        )

    return pd.DataFrame(rows)


def centroid_distance_vector(
    metadata,
    matrix,
):
    labels = (
        metadata[STAGE_COL].astype(str)
        + "||"
        + metadata[CELLTYPE_COL].astype(str)
    )

    centroids = {}

    for label_value in pd.unique(labels):
        positions = np.flatnonzero(
            labels.to_numpy()
            == label_value
        )

        # E7.75 contains only two primary cells; do not define a QC centroid
        # from such a tiny group.
        if len(positions) < 20:
            continue

        centroids[label_value] = matrix[
            positions,
            :
        ].mean(axis=0)

    keys = sorted(centroids)

    distances = []
    pair_labels = []

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            distances.append(
                float(
                    np.linalg.norm(
                        centroids[keys[i]]
                        - centroids[keys[j]]
                    )
                )
            )

            pair_labels.append(
                (
                    keys[i],
                    keys[j],
                )
            )

    return (
        np.asarray(
            distances,
            dtype=float,
        ),
        pair_labels,
    )


def geometry_preservation_rho(
    metadata,
    raw_matrix,
    rescued_matrix,
    n_dims,
):
    raw_vector, raw_pairs = centroid_distance_vector(
        metadata,
        raw_matrix[:, :n_dims],
    )

    rescued_vector, rescued_pairs = centroid_distance_vector(
        metadata,
        rescued_matrix[:, :n_dims],
    )

    if raw_pairs != rescued_pairs:
        raise RuntimeError(
            "Centroid pair definitions changed unexpectedly."
        )

    if len(raw_vector) < 3:
        return np.nan

    rho, _ = spearmanr(
        raw_vector,
        rescued_vector,
    )

    return float(rho)


def top_depth_correlations(
    metadata,
    matrix,
    modality,
    covariates,
):
    rows = []

    for covariate in covariates:
        values = pd.to_numeric(
            metadata[covariate],
            errors="coerce",
        ).to_numpy(dtype=float)

        if covariate in {
            RNA_DEPTH_COL,
            ATAC_DEPTH_COL,
        }:
            values = np.log1p(values)

        for j in range(matrix.shape[1]):
            dim = matrix[:, j]

            valid = (
                np.isfinite(dim)
                & np.isfinite(values)
            )

            rho, _ = spearmanr(
                dim[valid],
                values[valid],
            )

            rows.append(
                {
                    "modality": modality,
                    "dimension": j + 1,
                    "covariate": covariate,
                    "rho": float(rho),
                    "abs_rho": abs(float(rho)),
                }
            )

    return pd.DataFrame(rows)


def main():
    section(
        "PHASE F8b — EXTERNAL REPRESENTATION TECHNICAL RESCUE + QC"
    )

    print(
        "Frozen cells: exactly the Phase F8 6,579 WT paired cells."
    )

    print()

    print(
        "RNA rescue:"
    )

    print(
        "  stage-protected within-stage sample centering in PCA space"
    )

    print(
        "  (cell-type labels are NOT used for correction)"
    )

    print()

    print(
        "ATAC rescue:"
    )

    print(
        f"  exclude LSI axes with |rho(log1p(nFrags_atac))| >= "
        f"{ATAC_EXTREME_DEPTH_RHO:.2f}"
    )

    print(
        "  no sample batch correction"
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

    section(
        "1. LOAD FROZEN F8 REPRESENTATIONS"
    )

    require_inputs()

    (
        cells,
        metadata,
        rna_raw,
        atac_raw,
    ) = load_aligned_data()

    print(
        f"Cells: {len(cells):,}"
    )

    print(
        f"RNA raw shape:  {rna_raw.shape}"
    )

    print(
        f"ATAC raw shape: {atac_raw.shape}"
    )

    print(
        f"Cell count equals frozen expectation: "
        f"{len(cells) == EXPECTED_CELLS}"
    )

    section(
        "2. RNA STAGE-PROTECTED SAMPLE-CENTERING RESCUE"
    )

    (
        rna_rescued,
        centering_table,
    ) = stage_protected_sample_centering(
        metadata,
        rna_raw,
    )

    print_df(
        centering_table.set_index(
            [
                "stage",
                "sample",
            ]
        ),
        digits=4,
    )

    print()

    print(
        f"RNA rescued finite: "
        f"{np.isfinite(rna_rescued).all()}"
    )

    section(
        "3. ATAC EXTREME DEPTH-AXIS FILTER"
    )

    (
        atac_rescued,
        retained_indices,
        flagged_indices,
        atac_dimension_table,
    ) = atac_depth_axis_filter(
        metadata,
        atac_raw,
    )

    print_df(
        atac_dimension_table.set_index(
            "original_dimension"
        ),
        digits=4,
    )

    print()

    print(
        "Flagged ATAC dimensions: "
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
        "Retained ATAC dimensions: "
        f"{len(retained_indices)}"
    )

    print(
        f"ATAC rescued shape: {atac_rescued.shape}"
    )

    section(
        "4. RNA STRUCTURE QC — RAW VS RESCUED"
    )

    rna_raw_structure = structure_table(
        metadata,
        rna_raw,
        "RNA_RAW",
        DIMENSION_SENSITIVITY,
    )

    rna_rescued_structure = structure_table(
        metadata,
        rna_rescued,
        "RNA_SAMPLE_CENTERED",
        DIMENSION_SENSITIVITY,
    )

    subsection(
        "RNA raw"
    )

    print_df(
        rna_raw_structure.set_index(
            "n_dims"
        ),
        digits=4,
    )

    subsection(
        "RNA rescued"
    )

    print_df(
        rna_rescued_structure.set_index(
            "n_dims"
        ),
        digits=4,
    )

    section(
        "5. RNA BIOLOGICAL-GEOMETRY PRESERVATION"
    )

    geometry_rows = []

    for n_dims in DIMENSION_SENSITIVITY:
        rho = geometry_preservation_rho(
            metadata,
            rna_raw,
            rna_rescued,
            n_dims,
        )

        geometry_rows.append(
            {
                "n_dims": n_dims,
                "centroid_distance_spearman_raw_vs_rescued": rho,
                "preserves_geometry_ge_0.90": (
                    np.isfinite(rho)
                    and rho
                    >= MIN_CENTROID_GEOMETRY_SPEARMAN
                ),
            }
        )

    geometry_df = pd.DataFrame(
        geometry_rows
    )

    print_df(
        geometry_df.set_index(
            "n_dims"
        ),
        digits=4,
    )

    section(
        "6. ATAC STRUCTURE QC — RAW VS DEPTH-FILTERED"
    )

    atac_raw_structure = structure_table(
        metadata,
        atac_raw,
        "ATAC_RAW",
        DIMENSION_SENSITIVITY,
    )

    atac_dimension_counts = [
        n
        for n in DIMENSION_SENSITIVITY
        if n <= atac_rescued.shape[1]
    ]

    atac_rescued_structure = structure_table(
        metadata,
        atac_rescued,
        "ATAC_DEPTH_FILTERED",
        atac_dimension_counts,
    )

    subsection(
        "ATAC raw"
    )

    print_df(
        atac_raw_structure.set_index(
            "n_dims"
        ),
        digits=4,
    )

    subsection(
        "ATAC depth-filtered"
    )

    print_df(
        atac_rescued_structure.set_index(
            "n_dims"
        ),
        digits=4,
    )

    section(
        "7. POST-RESCUE DEPTH/QC CORRELATIONS"
    )

    rna_corr = top_depth_correlations(
        metadata,
        rna_rescued,
        "RNA_SAMPLE_CENTERED",
        [
            RNA_DEPTH_COL,
            RNA_FEATURE_COL,
        ],
    )

    atac_corr = top_depth_correlations(
        metadata,
        atac_rescued,
        "ATAC_DEPTH_FILTERED",
        [
            ATAC_DEPTH_COL,
            ATAC_TSS_COL,
        ],
    )

    subsection(
        "RNA rescued — top 15 absolute correlations"
    )

    print_df(
        rna_corr.sort_values(
            "abs_rho",
            ascending=False,
        ).head(
            15
        ).set_index(
            [
                "covariate",
                "dimension",
            ]
        ),
        digits=4,
    )

    subsection(
        "ATAC rescued — top 15 absolute correlations"
    )

    print_df(
        atac_corr.sort_values(
            "abs_rho",
            ascending=False,
        ).head(
            15
        ).set_index(
            [
                "covariate",
                "dimension",
            ]
        ),
        digits=4,
    )

    section(
        "8. SAVE F8b REPRESENTATION ARTIFACTS"
    )

    np.save(
        RNA_RESCUED_FILE,
        rna_rescued.astype(
            np.float32
        ),
    )

    np.save(
        ATAC_RESCUED_FILE,
        atac_rescued.astype(
            np.float32
        ),
    )

    atac_dimension_table[
        "retained"
    ] = ~atac_dimension_table[
        "depth_dominant"
    ]

    atac_dimension_table.to_csv(
        ATAC_DIMENSION_FILE,
        sep="\t",
        index=False,
    )

    manifest = {
        "dataset_accession": ACCESSION,
        "phase": "F8b",
        "created_utc": utc_now_iso(),
        "frozen_cell_count": len(cells),
        "rna_rescue": {
            "method": (
                "stage-protected within-stage sample-centroid correction "
                "in the existing 50-PC space"
            ),
            "uses_celltype_for_correction": False,
            "output_file": str(
                RNA_RESCUED_FILE
            ),
            "sha256": sha256_file(
                RNA_RESCUED_FILE
            ),
        },
        "atac_rescue": {
            "method": (
                "exclude LSI axes with absolute Spearman correlation "
                "with log1p(nFrags_atac) >= 0.80"
            ),
            "threshold": ATAC_EXTREME_DEPTH_RHO,
            "flagged_original_dimensions": [
                i + 1
                for i in flagged_indices
            ],
            "retained_original_dimensions": [
                i + 1
                for i in retained_indices
            ],
            "n_retained": len(retained_indices),
            "output_file": str(
                ATAC_RESCUED_FILE
            ),
            "sha256": sha256_file(
                ATAC_RESCUED_FILE
            ),
        },
        "guardrails": {
            "pseudotime_calculated": False,
            "gdis_calculated": False,
            "cmil_calculated": False,
            "discovery_dataset_modified": False,
        },
    }

    with F8B_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    for path in [
        RNA_RESCUED_FILE,
        ATAC_RESCUED_FILE,
        ATAC_DIMENSION_FILE,
        F8B_MANIFEST,
    ]:
        print(path)

    section(
        "9. PHASE F8b DECISION"
    )

    geometry_ok = bool(
        geometry_df[
            "preserves_geometry_ge_0.90"
        ].all()
    )

    rna_improved = bool(
        (
            rna_rescued_structure[
                "sample_partial_R2_given_biology"
            ].to_numpy()
            <
            rna_raw_structure[
                "sample_partial_R2_given_biology"
            ].to_numpy()
        ).all()
    )

    atac_depth_rule_worked = bool(
        len(flagged_indices) >= 1
        and atac_rescued.shape[1] >= 40
    )

    print(
        f"RNA sample partial R2 improved at all tested dimensions: "
        f"{rna_improved}"
    )

    print(
        f"RNA coarse centroid geometry preserved >=0.90 at all tested dimensions: "
        f"{geometry_ok}"
    )

    print(
        f"ATAC technical-depth rule flagged >=1 axis and retained >=40 axes: "
        f"{atac_depth_rule_worked}"
    )

    print()

    if (
        rna_improved
        and geometry_ok
        and atac_depth_rule_worked
    ):
        print(
            "PHASE F8b VERDICT: GO — TECHNICAL RESCUE CANDIDATES VALID"
        )

        print()

        print(
            "Do not construct pseudotime yet."
        )

        print(
            "Review the dimension-sensitivity tables and then freeze one "
            "RNA and one ATAC dimensionality for the external trajectory."
        )

    else:
        print(
            "PHASE F8b VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not construct pseudotime. At least one rescue safeguard "
            "did not pass."
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

