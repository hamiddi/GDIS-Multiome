#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
25_phase_f11_shareseq_representations_branch_continuity.py
==========================================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F11
---------
Construct modality-specific representations for the PREDEFINED broad
hair-follicle candidate population, then screen the three PUBLISHED
TAC-derived branches for manifold continuity BEFORE freezing any branch.

SOURCE-SUPPORTED BRANCHES
-------------------------
The SHARE-seq study recovered three TAC-derived hair-follicle trajectories:

    TAC -> IRS
    TAC -> Medulla
    TAC -> Hair Shaft-cuticle/cortex

Later multiomic velocity work describes lineage directionality as:

    TAC-1 -> TAC-2 -> terminal lineage

Accordingly, F11 prospectively screens:

    TAC-1 -> TAC-2 -> IRS
    TAC-1 -> TAC-2 -> Medulla
    TAC-1 -> TAC-2 -> Hair Shaft-cuticle.cortex

IMPORTANT
---------
Branch choice is based on representation/topology suitability only.

F11 does NOT:
    - calculate pseudotime;
    - calculate GDIS;
    - calculate CMIL;
    - inspect any RNA-vs-ATAC lead/lag;
    - choose a branch based on a desired priming result.

INPUTS
------
data/GSE140203/
    GSM4156608_skin.late.anagen.rna.counts.txt.gz
    GSM4156597_skin.late.anagen.counts.txt.gz
    GSM4156597_skin.late.anagen.barcodes.txt.gz
    GSM4156597_skin.late.anagen.peaks.bed.gz
    f10c_paired_cell_mapping.tsv.gz

REPRESENTATION DESIGN
---------------------

RNA:
    paired hair-lineage cells only
    library-size normalization to 10,000
    log1p
    2,500 high-variance genes
    gene-wise standardization, clip [-10, 10]
    PCA, 50 dimensions

ATAC:
    paired hair-lineage cells only
    processed peak x cell MatrixMarket matrix
    top 25,000 peaks by total count in candidate cells
    binary accessibility
    TF-IDF
    truncated SVD / LSI, 50 dimensions

ATAC depth audit:
    dimensions with
        |Spearman rho(LSI_j, log1p(ATAC depth))| >= 0.80
    are FLAGGED as depth-dominant.

The flag is diagnostic in F11.  F11 also creates a depth-filtered ATAC
representation for continuity sensitivity, but does not yet freeze its
final dimension count.

CONTINUITY SCREEN
-----------------
For each published branch, F11 evaluates RNA and depth-filtered ATAC at
10/20/30/40/50 dimensions where available and k=20/30/50.

Metrics include:
    - connected components;
    - directed kNN state-neighbor fractions;
    - overall cross-state neighbor fraction;
    - expected-adjacent fraction among cross-state edges;
    - direct TAC-1 <-> terminal bypass fraction;
    - minimum required bridge fraction across:
          TAC-1 -> TAC-2
          TAC-2 -> terminal
          terminal -> TAC-2
    - state-centroid distances;
    - whether TAC-1-to-terminal is the largest centroid distance;
    - centroid-chain ratio:
          d(TAC1, terminal) / [d(TAC1,TAC2) + d(TAC2,terminal)]
    - silhouette score of the three labels.

F11 intentionally does NOT collapse these metrics into a single optimized
score.  The output must be reviewed before a branch is frozen.

OUTPUTS
-------
data/GSE140203/representations_f11/
    f11_hair_candidate_cells.tsv.gz
    f11_rna_hvg_2500.tsv.gz
    f11_atac_peaks_25000.tsv.gz
    f11_rna_pca_50.npy
    f11_atac_lsi_50.npy
    f11_atac_lsi_depth_filtered.npy
    f11_atac_dimension_depth_qc.tsv
    f11_branch_continuity.tsv.gz
    f11_manifest.json

TEMPORARY DISK
--------------
RNA selected-count memmap:
    roughly 0.7 GB

ATAC selected coordinate-pair binary file:
    size depends on candidate-cell nnz; usually a few hundred MB

Temporary files are deleted after successful completion unless:
    --keep-temp

RUN
---
    python 25_phase_f11_shareseq_representations_branch_continuity.py

DEPENDENCIES
------------
numpy
pandas
scipy
scikit-learn
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE140203"

DATA_DIR = Path("data") / ACCESSION
OUT_DIR = DATA_DIR / "representations_f11"
TEMP_DIR = OUT_DIR / "_temp"

MAPPING_FILE = DATA_DIR / "f10c_paired_cell_mapping.tsv.gz"

RNA_COUNTS_FILE = DATA_DIR / "GSM4156608_skin.late.anagen.rna.counts.txt.gz"

ATAC_COUNTS_FILE = DATA_DIR / "GSM4156597_skin.late.anagen.counts.txt.gz"
ATAC_BARCODES_FILE = DATA_DIR / "GSM4156597_skin.late.anagen.barcodes.txt.gz"
ATAC_PEAKS_FILE = DATA_DIR / "GSM4156597_skin.late.anagen.peaks.bed.gz"

OUTPUT_CELLS = OUT_DIR / "f11_hair_candidate_cells.tsv.gz"
OUTPUT_HVG = OUT_DIR / "f11_rna_hvg_2500.tsv.gz"
OUTPUT_ATAC_PEAKS = OUT_DIR / "f11_atac_peaks_25000.tsv.gz"

OUTPUT_RNA = OUT_DIR / "f11_rna_pca_50.npy"
OUTPUT_ATAC = OUT_DIR / "f11_atac_lsi_50.npy"
OUTPUT_ATAC_FILTERED = OUT_DIR / "f11_atac_lsi_depth_filtered.npy"

OUTPUT_ATAC_DIM_QC = OUT_DIR / "f11_atac_dimension_depth_qc.tsv"
OUTPUT_CONTINUITY = OUT_DIR / "f11_branch_continuity.tsv.gz"
OUTPUT_MANIFEST = OUT_DIR / "f11_manifest.json"

RNA_MEMMAP_FILE = TEMP_DIR / "rna_selected_counts.uint32.dat"
ATAC_PAIR_FILE = TEMP_DIR / "atac_selected_pairs.int32.dat"


# =====================================================================
# FROZEN CANDIDATE POPULATION / BRANCHES
# =====================================================================

HAIR_LABELS = [
    "TAC-1",
    "TAC-2",
    "IRS",
    "Medulla",
    "Hair Shaft-cuticle.cortex",
]

BRANCHES = {
    "IRS": [
        "TAC-1",
        "TAC-2",
        "IRS",
    ],
    "Medulla": [
        "TAC-1",
        "TAC-2",
        "Medulla",
    ],
    "CuticleCortex": [
        "TAC-1",
        "TAC-2",
        "Hair Shaft-cuticle.cortex",
    ],
}

ROOT_STATE = "TAC-1"
INTERMEDIATE_STATE = "TAC-2"

EXPECTED_HAIR_CELLS = 7_197

RNA_HVG_COUNT = 2_500
RNA_PCA_DIMS = 50
RNA_NORMALIZATION_TARGET = 10_000.0
RNA_SCALE_CLIP = 10.0

ATAC_FEATURE_COUNT = 25_000
ATAC_LSI_DIMS = 50
ATAC_EXTREME_DEPTH_RHO = 0.80

DIMENSION_SENSITIVITY = [
    10,
    20,
    30,
    40,
    50,
]

K_VALUES = [
    20,
    30,
    50,
]

PRIMARY_K = 30
PRIMARY_DIMS = 10

RANDOM_SEED = 785

RNA_GENE_CHUNK = 256
ATAC_PAIR_BUFFER = 1_000_000
ATAC_BINARY_READ_ROWS = 2_000_000


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


def human_size(n_bytes):
    value = float(
        n_bytes
    )

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"

        value /= 1024.0

    return str(
        n_bytes
    )


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
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
# CLI / INPUTS
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Construct SHARE-seq hair-lineage RNA/ATAC representations "
            "and screen published branches for continuity."
        )
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Retain temporary RNA memmap and ATAC pair files.",
    )

    return parser.parse_args()


def require_inputs():
    required = [
        MAPPING_FILE,
        RNA_COUNTS_FILE,
        ATAC_COUNTS_FILE,
        ATAC_BARCODES_FILE,
        ATAC_PEAKS_FILE,
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


# =====================================================================
# BARCODE HELPERS
# =====================================================================

def normalize_rna_matrix_barcode(
    barcode,
):
    return str(
        barcode
    ).strip().replace(
        ",",
        ".",
    )


def read_single_column_gzip(
    path,
):
    values = []

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line_text in handle:
            value = line_text.strip()

            if not value:
                continue

            values.append(
                value.split(
                    "\t"
                )[
                    0
                ]
            )

    return values


# =====================================================================
# FREEZE BROAD HAIR CANDIDATE CELLS
# =====================================================================

def load_hair_candidate_mapping():
    mapping = pd.read_csv(
        MAPPING_FILE,
        sep="\t",
        compression="gzip",
        dtype={
            "rna.bc": str,
            "atac.bc": str,
            "fragment.bc": str,
            "celltype": str,
        },
    )

    required = [
        "rna.bc",
        "atac.bc",
        "fragment.bc",
        "celltype",
    ]

    missing = [
        column
        for column in required
        if column not in mapping.columns
    ]

    if missing:
        raise RuntimeError(
            "F10c mapping missing columns: "
            + ", ".join(
                missing
            )
        )

    hair = mapping.loc[
        mapping[
            "celltype"
        ].isin(
            HAIR_LABELS
        )
    ].copy().reset_index(
        drop=True
    )

    hair[
        "_f11_row"
    ] = np.arange(
        len(
            hair
        ),
        dtype=int,
    )

    if len(
        hair
    ) != EXPECTED_HAIR_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_HAIR_CELLS:,} candidate hair cells, "
            f"found {len(hair):,}."
        )

    if hair[
        "rna.bc"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate rna.bc in F11 candidate cells."
        )

    return hair


# =====================================================================
# RNA REPRESENTATION
# =====================================================================

def count_rna_gene_rows():
    count = 0

    with gzip.open(
        RNA_COUNTS_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        # Header.
        handle.readline()

        for line_text in handle:
            if line_text.strip():
                count += 1

    return count


def parse_rna_header():
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

    normalized_barcodes = [
        normalize_rna_matrix_barcode(
            value
        )
        for value in fields[
            1:
        ]
    ]

    if len(
        normalized_barcodes
    ) != len(
        set(
            normalized_barcodes
        )
    ):
        raise RuntimeError(
            "Normalized RNA matrix barcodes are not unique."
        )

    return (
        fields[
            0
        ],
        normalized_barcodes,
    )


def reconstruct_rna_pca(
    hair,
):
    subsection(
        "RNA: identify candidate-cell columns"
    )

    (
        first_field,
        rna_barcodes,
    ) = parse_rna_header()

    print(
        f"RNA row-name field: {first_field}"
    )

    print(
        f"RNA matrix cell columns: {len(rna_barcodes):,}"
    )

    barcode_to_column = {
        barcode: index
        for index, barcode in enumerate(
            rna_barcodes
        )
    }

    missing = [
        barcode
        for barcode in hair[
            "rna.bc"
        ]
        if barcode not in barcode_to_column
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} F11 hair cells are missing from RNA matrix."
        )

    selected_columns = np.asarray(
        [
            barcode_to_column[
                barcode
            ]
            for barcode in hair[
                "rna.bc"
            ]
        ],
        dtype=np.int64,
    )

    n_genes = count_rna_gene_rows()

    print(
        f"RNA gene rows: {n_genes:,}"
    )

    print(
        f"Selected paired hair cells: {len(hair):,}"
    )

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RNA_MEMMAP_FILE.unlink(
        missing_ok=True
    )

    counts = np.memmap(
        RNA_MEMMAP_FILE,
        dtype=np.uint32,
        mode="w+",
        shape=(
            n_genes,
            len(
                hair
            ),
        ),
    )

    gene_names = []

    subsection(
        "RNA: stream candidate counts to temporary memmap"
    )

    with gzip.open(
        RNA_COUNTS_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        handle.readline()

        row = 0

        for line_text in handle:
            if not line_text.strip():
                continue

            if "\t" not in line_text:
                raise RuntimeError(
                    f"RNA row {row + 1} has no TAB separator."
                )

            gene, numeric_text = line_text.rstrip(
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
                rna_barcodes
            ):
                raise RuntimeError(
                    f"RNA row {row + 1} width mismatch: "
                    f"{len(values)} vs {len(rna_barcodes)}."
                )

            selected = values[
                selected_columns
            ]

            if np.any(
                selected < 0
            ):
                raise RuntimeError(
                    f"Negative RNA count encountered at gene row {row + 1}."
                )

            counts[
                row,
                :,
            ] = selected.astype(
                np.uint32
            )

            gene_names.append(
                gene
            )

            row += 1

            if row % 2000 == 0:
                print(
                    f"  parsed {row:,} / {n_genes:,} genes",
                    flush=True,
                )

    counts.flush()

    if row != n_genes:
        raise RuntimeError(
            f"RNA parsed-row mismatch: {row} vs {n_genes}."
        )

    print(
        f"RNA temp memmap: {human_size(RNA_MEMMAP_FILE.stat().st_size)}"
    )

    subsection(
        "RNA: depth / detected-feature statistics"
    )

    library_size = np.zeros(
        len(
            hair
        ),
        dtype=np.float64,
    )

    detected_features = np.zeros(
        len(
            hair
        ),
        dtype=np.int32,
    )

    for start in range(
        0,
        n_genes,
        RNA_GENE_CHUNK,
    ):
        stop = min(
            start + RNA_GENE_CHUNK,
            n_genes,
        )

        block = np.asarray(
            counts[
                start:
                stop,
                :
            ],
            dtype=np.float64,
        )

        library_size += block.sum(
            axis=0
        )

        detected_features += (
            block
            > 0
        ).sum(
            axis=0
        ).astype(
            np.int32
        )

    if np.any(
        library_size <= 0
    ):
        raise RuntimeError(
            "At least one F11 candidate cell has zero RNA library size."
        )

    print(
        f"RNA total counts: "
        f"min={library_size.min():.0f}, "
        f"median={np.median(library_size):.0f}, "
        f"max={library_size.max():.0f}"
    )

    print(
        f"RNA detected genes: "
        f"min={detected_features.min()}, "
        f"median={np.median(detected_features):.0f}, "
        f"max={detected_features.max()}"
    )

    subsection(
        "RNA: calculate log-normalized gene variance"
    )

    variances = np.zeros(
        n_genes,
        dtype=np.float64,
    )

    means = np.zeros(
        n_genes,
        dtype=np.float64,
    )

    for start in range(
        0,
        n_genes,
        RNA_GENE_CHUNK,
    ):
        stop = min(
            start + RNA_GENE_CHUNK,
            n_genes,
        )

        block = np.asarray(
            counts[
                start:
                stop,
                :
            ],
            dtype=np.float64,
        )

        normalized = np.log1p(
            (
                block
                / library_size[
                    None,
                    :
                ]
            )
            * RNA_NORMALIZATION_TARGET
        )

        means[
            start:
            stop
        ] = normalized.mean(
            axis=1
        )

        variances[
            start:
            stop
        ] = normalized.var(
            axis=1,
            ddof=1,
        )

    n_hvg = min(
        RNA_HVG_COUNT,
        n_genes,
    )

    top_idx = np.argpartition(
        variances,
        -n_hvg,
    )[
        -n_hvg:
    ]

    top_idx = top_idx[
        np.argsort(
            variances[
                top_idx
            ]
        )[
            ::-1
        ]
    ]

    hvg_table = pd.DataFrame(
        {
            "rank": np.arange(
                1,
                n_hvg + 1,
            ),
            "gene_index": top_idx,
            "gene": [
                gene_names[
                    i
                ]
                for i in top_idx
            ],
            "lognorm_mean": means[
                top_idx
            ],
            "lognorm_variance": variances[
                top_idx
            ],
        }
    )

    hvg_table.to_csv(
        OUTPUT_HVG,
        sep="\t",
        index=False,
        compression="gzip",
    )

    subsection(
        "RNA: construct standardized 2,500-HVG matrix"
    )

    x = np.empty(
        (
            len(
                hair
            ),
            n_hvg,
        ),
        dtype=np.float32,
    )

    for feature_position, gene_index in enumerate(
        top_idx
    ):
        values = np.asarray(
            counts[
                gene_index,
                :
            ],
            dtype=np.float64,
        )

        normalized = np.log1p(
            (
                values
                / library_size
            )
            * RNA_NORMALIZATION_TARGET
        )

        mean = normalized.mean()
        std = normalized.std(
            ddof=0
        )

        if std <= 0:
            scaled = np.zeros_like(
                normalized
            )
        else:
            scaled = (
                normalized
                - mean
            ) / std

        scaled = np.clip(
            scaled,
            -RNA_SCALE_CLIP,
            RNA_SCALE_CLIP,
        )

        x[
            :,
            feature_position,
        ] = scaled.astype(
            np.float32
        )

    subsection(
        "RNA: PCA 50D"
    )

    pca = PCA(
        n_components=RNA_PCA_DIMS,
        svd_solver="randomized",
        random_state=RANDOM_SEED,
    )

    representation = pca.fit_transform(
        x
    ).astype(
        np.float32
    )

    print(
        f"RNA PCA shape: {representation.shape}"
    )

    print(
        f"RNA PCA cumulative variance first 10 PCs: "
        f"{pca.explained_variance_ratio_[:10].sum():.4f}"
    )

    print(
        f"RNA PCA cumulative variance first 50 PCs: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    return {
        "representation": representation,
        "library_size": library_size,
        "detected_features": detected_features,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "n_genes": n_genes,
        "hvg_table": hvg_table,
    }


# =====================================================================
# ATAC REPRESENTATION
# =====================================================================

def matrixmarket_dimensions_and_stream_start(
    handle,
):
    banner = handle.readline().strip()

    if not banner.startswith(
        "%%MatrixMarket"
    ):
        raise RuntimeError(
            "ATAC counts file is not MatrixMarket."
        )

    for line_text in handle:
        stripped = line_text.strip()

        if not stripped:
            continue

        if stripped.startswith(
            "%"
        ):
            continue

        fields = re.split(
            r"\s+",
            stripped,
        )

        if len(
            fields
        ) != 3:
            raise RuntimeError(
                f"Unexpected MatrixMarket dimension line: {stripped}"
            )

        return (
            banner,
            int(
                fields[
                    0
                ]
            ),
            int(
                fields[
                    1
                ]
            ),
            int(
                fields[
                    2
                ]
            ),
        )

    raise RuntimeError(
        "Could not parse MatrixMarket dimensions."
    )


def read_peak_table():
    rows = []

    with gzip.open(
        ATAC_PEAKS_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for index, line_text in enumerate(
            handle
        ):
            if not line_text.strip():
                continue

            fields = line_text.rstrip(
                "\n\r"
            ).split(
                "\t"
            )

            if len(
                fields
            ) < 3:
                fields = re.split(
                    r"\s+",
                    line_text.strip(),
                )

            if len(
                fields
            ) < 3:
                raise RuntimeError(
                    f"Malformed ATAC peak row {index + 1}."
                )

            rows.append(
                (
                    fields[
                        0
                    ],
                    int(
                        fields[
                            1
                        ]
                    ),
                    int(
                        fields[
                            2
                        ]
                    ),
                )
            )

    return rows


def flush_atac_pairs(
    path,
    cell_buffer,
    peak_buffer,
):
    if not cell_buffer:
        return

    block = np.empty(
        (
            len(
                cell_buffer
            ),
            2,
        ),
        dtype=np.int32,
    )

    block[
        :,
        0
    ] = cell_buffer

    block[
        :,
        1
    ] = peak_buffer

    with path.open(
        "ab"
    ) as handle:
        block.tofile(
            handle
        )

    cell_buffer.clear()
    peak_buffer.clear()


def scan_atac_selected_coordinates(
    hair,
):
    atac_barcodes = read_single_column_gzip(
        ATAC_BARCODES_FILE
    )

    if len(
        atac_barcodes
    ) != len(
        set(
            atac_barcodes
        )
    ):
        raise RuntimeError(
            "ATAC barcode file contains duplicates."
        )

    barcode_to_column = {
        barcode: index
        for index, barcode in enumerate(
            atac_barcodes
        )
    }

    missing = [
        barcode
        for barcode in hair[
            "rna.bc"
        ]
        if barcode not in barcode_to_column
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} F11 hair cells missing from ATAC matrix."
        )

    selected_matrix_columns = np.asarray(
        [
            barcode_to_column[
                barcode
            ]
            for barcode in hair[
                "rna.bc"
            ]
        ],
        dtype=np.int64,
    )

    # MatrixMarket columns are one-based.
    col_to_local = np.full(
        len(
            atac_barcodes
        )
        + 1,
        -1,
        dtype=np.int32,
    )

    for local_index, zero_based_col in enumerate(
        selected_matrix_columns
    ):
        col_to_local[
            zero_based_col
            + 1
        ] = local_index

    ATAC_PAIR_FILE.unlink(
        missing_ok=True
    )

    with gzip.open(
        ATAC_COUNTS_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        (
            banner,
            n_peaks,
            n_cells,
            n_nnz,
        ) = matrixmarket_dimensions_and_stream_start(
            handle
        )

        if n_cells != len(
            atac_barcodes
        ):
            raise RuntimeError(
                "ATAC MatrixMarket/barcode-column mismatch."
            )

        peak_total = np.zeros(
            n_peaks,
            dtype=np.float64,
        )

        cell_depth = np.zeros(
            len(
                hair
            ),
            dtype=np.float64,
        )

        cell_nonzero_peaks = np.zeros(
            len(
                hair
            ),
            dtype=np.int32,
        )

        cell_buffer = []
        peak_buffer = []

        parsed_nnz = 0
        selected_nnz = 0
        malformed = 0

        print(
            f"ATAC MatrixMarket: {n_peaks:,} peaks x "
            f"{n_cells:,} cells; declared nnz={n_nnz:,}"
        )

        for line_text in handle:
            if not line_text.strip():
                continue

            fields = re.split(
                r"\s+",
                line_text.strip(),
            )

            if len(
                fields
            ) < 3:
                malformed += 1
                continue

            row_index = int(
                fields[
                    0
                ]
            )

            col_index = int(
                fields[
                    1
                ]
            )

            value = float(
                fields[
                    2
                ]
            )

            parsed_nnz += 1

            local_cell = col_to_local[
                col_index
            ]

            if local_cell < 0:
                continue

            peak_zero = (
                row_index
                - 1
            )

            peak_total[
                peak_zero
            ] += value

            cell_depth[
                local_cell
            ] += value

            cell_nonzero_peaks[
                local_cell
            ] += 1

            cell_buffer.append(
                local_cell
            )

            peak_buffer.append(
                peak_zero
            )

            selected_nnz += 1

            if len(
                cell_buffer
            ) >= ATAC_PAIR_BUFFER:
                flush_atac_pairs(
                    ATAC_PAIR_FILE,
                    cell_buffer,
                    peak_buffer,
                )

            if parsed_nnz % 20_000_000 == 0:
                print(
                    f"  parsed {parsed_nnz:,} / {n_nnz:,} MatrixMarket entries; "
                    f"selected={selected_nnz:,}",
                    flush=True,
                )

        flush_atac_pairs(
            ATAC_PAIR_FILE,
            cell_buffer,
            peak_buffer,
        )

    if malformed != 0:
        raise RuntimeError(
            f"ATAC MatrixMarket had {malformed} malformed coordinate rows."
        )

    if parsed_nnz != n_nnz:
        raise RuntimeError(
            f"ATAC parsed nnz mismatch: {parsed_nnz:,} vs {n_nnz:,}."
        )

    if np.any(
        cell_depth <= 0
    ):
        raise RuntimeError(
            "At least one F11 hair cell has zero ATAC depth."
        )

    print(
        f"Selected ATAC coordinate nnz: {selected_nnz:,}"
    )

    print(
        f"ATAC temp pair file: {human_size(ATAC_PAIR_FILE.stat().st_size)}"
    )

    print(
        f"ATAC total counts: "
        f"min={cell_depth.min():.0f}, "
        f"median={np.median(cell_depth):.0f}, "
        f"max={cell_depth.max():.0f}"
    )

    print(
        f"ATAC nonzero peaks: "
        f"min={cell_nonzero_peaks.min()}, "
        f"median={np.median(cell_nonzero_peaks):.0f}, "
        f"max={cell_nonzero_peaks.max()}"
    )

    return {
        "n_peaks": n_peaks,
        "n_matrix_cells": n_cells,
        "n_matrix_nnz": n_nnz,
        "selected_nnz": selected_nnz,
        "peak_total": peak_total,
        "cell_depth": cell_depth,
        "cell_nonzero_peaks": cell_nonzero_peaks,
    }


def build_atac_binary_matrix(
    n_cells,
    selected_peaks_sorted,
):
    file_size = ATAC_PAIR_FILE.stat().st_size

    itemsize = np.dtype(
        np.int32
    ).itemsize

    if file_size % (
        2
        * itemsize
    ) != 0:
        raise RuntimeError(
            "Corrupt ATAC temporary pair file."
        )

    n_pairs = file_size // (
        2
        * itemsize
    )

    mmap = np.memmap(
        ATAC_PAIR_FILE,
        dtype=np.int32,
        mode="r",
        shape=(
            n_pairs,
            2,
        ),
    )

    matrix = sp.csr_matrix(
        (
            n_cells,
            len(
                selected_peaks_sorted
            ),
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        n_pairs,
        ATAC_BINARY_READ_ROWS,
    ):
        stop = min(
            start
            + ATAC_BINARY_READ_ROWS,
            n_pairs,
        )

        block = np.asarray(
            mmap[
                start:
                stop
            ]
        )

        cells = block[
            :,
            0
        ]

        peaks = block[
            :,
            1
        ]

        positions = np.searchsorted(
            selected_peaks_sorted,
            peaks,
        )

        valid = (
            positions
            < len(
                selected_peaks_sorted
            )
        )

        valid_idx = np.flatnonzero(
            valid
        )

        if len(
            valid_idx
        ) == 0:
            continue

        exact = (
            selected_peaks_sorted[
                positions[
                    valid_idx
                ]
            ]
            == peaks[
                valid_idx
            ]
        )

        keep = valid_idx[
            exact
        ]

        if len(
            keep
        ) == 0:
            continue

        chunk = sp.coo_matrix(
            (
                np.ones(
                    len(
                        keep
                    ),
                    dtype=np.float32,
                ),
                (
                    cells[
                        keep
                    ],
                    positions[
                        keep
                    ],
                ),
            ),
            shape=matrix.shape,
        ).tocsr()

        matrix = matrix + chunk

    del mmap

    matrix.sum_duplicates()

    # Binary accessibility.
    matrix.data[:] = 1.0

    matrix.eliminate_zeros()

    return matrix


def tfidf_lsi(
    binary_matrix,
):
    row_sum = np.asarray(
        binary_matrix.sum(
            axis=1
        )
    ).ravel()

    if np.any(
        row_sum <= 0
    ):
        raise RuntimeError(
            "ATAC selected-feature matrix has a zero row."
        )

    tf = sp.diags(
        1.0
        / row_sum
    ) @ binary_matrix

    df = np.asarray(
        (
            binary_matrix
            > 0
        ).sum(
            axis=0
        )
    ).ravel()

    idf = np.log1p(
        binary_matrix.shape[
            0
        ]
        / (
            1.0
            + df
        )
    )

    tfidf = tf.multiply(
        idf
    ).tocsr()

    svd = TruncatedSVD(
        n_components=ATAC_LSI_DIMS,
        algorithm="randomized",
        n_iter=7,
        random_state=RANDOM_SEED,
    )

    lsi = svd.fit_transform(
        tfidf
    ).astype(
        np.float32
    )

    return (
        lsi,
        svd.explained_variance_ratio_,
    )


def reconstruct_atac_lsi(
    hair,
):
    subsection(
        "ATAC: stream selected candidate-cell coordinates"
    )

    scan = scan_atac_selected_coordinates(
        hair
    )

    peak_total = scan[
        "peak_total"
    ]

    n_features = min(
        ATAC_FEATURE_COUNT,
        len(
            peak_total
        ),
    )

    top_idx = np.argpartition(
        peak_total,
        -n_features,
    )[
        -n_features:
    ]

    top_by_support = top_idx[
        np.argsort(
            peak_total[
                top_idx
            ]
        )[
            ::-1
        ]
    ]

    peaks = read_peak_table()

    if len(
        peaks
    ) != scan[
        "n_peaks"
    ]:
        raise RuntimeError(
            "Peak BED row count does not match MatrixMarket rows."
        )

    peak_rows = []

    for rank, peak_index in enumerate(
        top_by_support,
        start=1,
    ):
        chrom, start_coord, end_coord = peaks[
            peak_index
        ]

        peak_rows.append(
            {
                "rank": rank,
                "matrix_peak_index_zero_based": int(
                    peak_index
                ),
                "chrom": chrom,
                "start": start_coord,
                "end": end_coord,
                "candidate_total_count": float(
                    peak_total[
                        peak_index
                    ]
                ),
            }
        )

    peak_table = pd.DataFrame(
        peak_rows
    )

    peak_table.to_csv(
        OUTPUT_ATAC_PEAKS,
        sep="\t",
        index=False,
        compression="gzip",
    )

    selected_sorted = np.sort(
        top_by_support.astype(
            np.int32
        )
    )

    subsection(
        "ATAC: construct binary candidate-cell x top-25k-peak matrix"
    )

    binary_matrix = build_atac_binary_matrix(
        len(
            hair
        ),
        selected_sorted,
    )

    print(
        f"ATAC binary matrix: "
        f"{binary_matrix.shape[0]:,} cells x "
        f"{binary_matrix.shape[1]:,} peaks"
    )

    print(
        f"ATAC binary nnz: {binary_matrix.nnz:,}"
    )

    selected_accessible = np.asarray(
        binary_matrix.sum(
            axis=1
        )
    ).ravel()

    print(
        f"Selected accessible peaks: "
        f"min={selected_accessible.min():.0f}, "
        f"median={np.median(selected_accessible):.0f}, "
        f"max={selected_accessible.max():.0f}"
    )

    subsection(
        "ATAC: TF-IDF + LSI 50D"
    )

    lsi, explained = tfidf_lsi(
        binary_matrix
    )

    print(
        f"ATAC LSI shape: {lsi.shape}"
    )

    print(
        f"ATAC LSI cumulative variance first 10 dims: "
        f"{explained[:10].sum():.4f}"
    )

    print(
        f"ATAC LSI cumulative variance first 50 dims: "
        f"{explained.sum():.4f}"
    )

    return {
        "representation": lsi,
        "explained_variance_ratio": explained,
        "cell_depth": scan[
            "cell_depth"
        ],
        "cell_nonzero_peaks": scan[
            "cell_nonzero_peaks"
        ],
        "selected_accessible_peaks": selected_accessible,
        "peak_table": peak_table,
        "scan": scan,
    }


# =====================================================================
# DEPTH / TECHNICAL QC
# =====================================================================

def dimension_correlation_table(
    matrix,
    covariates,
    modality,
):
    rows = []

    for covariate_name, values in covariates.items():
        values = np.asarray(
            values,
            dtype=float,
        )

        for j in range(
            matrix.shape[
                1
            ]
        ):
            dim = matrix[
                :,
                j
            ]

            valid = (
                np.isfinite(
                    dim
                )
                & np.isfinite(
                    values
                )
            )

            rho, _ = spearmanr(
                dim[
                    valid
                ],
                values[
                    valid
                ],
            )

            rows.append(
                {
                    "modality": modality,
                    "dimension": j + 1,
                    "covariate": covariate_name,
                    "spearman_rho": float(
                        rho
                    ),
                    "abs_rho": abs(
                        float(
                            rho
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def filter_atac_depth_axes(
    lsi,
    atac_depth,
):
    log_depth = np.log1p(
        np.asarray(
            atac_depth,
            dtype=float,
        )
    )

    rows = []
    retained = []
    flagged = []

    for j in range(
        lsi.shape[
            1
        ]
    ):
        rho, _ = spearmanr(
            lsi[
                :,
                j
            ],
            log_depth,
        )

        rho = float(
            rho
        )

        is_flagged = (
            abs(
                rho
            )
            >= ATAC_EXTREME_DEPTH_RHO
        )

        rows.append(
            {
                "original_dimension": j + 1,
                "spearman_rho_log1p_atac_depth": rho,
                "abs_rho": abs(
                    rho
                ),
                "depth_dominant": is_flagged,
            }
        )

        if is_flagged:
            flagged.append(
                j
            )
        else:
            retained.append(
                j
            )

    filtered = lsi[
        :,
        retained
    ].copy()

    return (
        filtered,
        retained,
        flagged,
        pd.DataFrame(
            rows
        ),
    )


# =====================================================================
# CONTINUITY METRICS
# =====================================================================

def branch_neighbor_metrics(
    hair,
    representation,
    states,
    k,
):
    terminal = states[
        2
    ]

    mask = hair[
        "celltype"
    ].isin(
        states
    ).to_numpy()

    local_cells = hair.loc[
        mask
    ].reset_index(
        drop=True
    )

    x = representation[
        mask,
        :
    ]

    labels = local_cells[
        "celltype"
    ].astype(
        str
    ).to_numpy()

    if len(
        local_cells
    ) <= k:
        raise RuntimeError(
            f"Branch {states} has too few cells for k={k}."
        )

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        algorithm="auto",
    )

    nn.fit(
        x
    )

    distances, indices = nn.kneighbors(
        x,
        return_distance=True,
    )

    # Remove self.
    neighbors = indices[
        :,
        1:
    ]

    state_order = [
        ROOT_STATE,
        INTERMEDIATE_STATE,
        terminal,
    ]

    state_to_idx = {
        state: index
        for index, state in enumerate(
            state_order
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

    self_edges = int(
        np.trace(
            counts
        )
    )

    total_edges = int(
        counts.sum()
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

    adjacent_fraction_of_cross = (
        adjacent_cross_edges
        / cross_edges
        if cross_edges
        else np.nan
    )

    bypass_fraction_of_cross = (
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

    min_required_bridge = float(
        np.min(
            required_bridge
        )
    )

    # Connected components in the symmetric kNN adjacency.
    rows = np.repeat(
        np.arange(
            len(
                local_cells
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
                local_cells
            ),
            len(
                local_cells
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

    # Centroid geometry.
    centroids = {
        state: x[
            labels
            == state
        ].mean(
            axis=0
        )
        for state in state_order
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
                terminal
            ]
        )
    )

    d13 = float(
        np.linalg.norm(
            centroids[
                ROOT_STATE
            ]
            - centroids[
                terminal
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

    # Silhouette can be expensive but branch sizes are only ~5k.
    sil = float(
        silhouette_score(
            x,
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
        "n_cells": len(
            local_cells
        ),
        "n_TAC1": int(
            np.sum(
                labels
                == ROOT_STATE
            )
        ),
        "n_TAC2": int(
            np.sum(
                labels
                == INTERMEDIATE_STATE
            )
        ),
        "n_terminal": int(
            np.sum(
                labels
                == terminal
            )
        ),
        "connected_components": int(
            n_components
        ),
        "cross_state_neighbor_fraction": float(
            cross_fraction
        ),
        "adjacent_fraction_of_cross_edges": float(
            adjacent_fraction_of_cross
        ),
        "bypass_fraction_of_cross_edges": float(
            bypass_fraction_of_cross
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
        "TAC2_to_terminal_neighbor_fraction": float(
            fractions[
                1,
                2
            ]
        ),
        "terminal_to_TAC2_neighbor_fraction": float(
            fractions[
                2,
                1
            ]
        ),
        "TAC1_to_terminal_bypass_neighbor_fraction": float(
            fractions[
                0,
                2
            ]
        ),
        "terminal_to_TAC1_bypass_neighbor_fraction": float(
            fractions[
                2,
                0
            ]
        ),
        "min_required_bridge_fraction": min_required_bridge,
        "centroid_d_TAC1_TAC2": d12,
        "centroid_d_TAC2_terminal": d23,
        "centroid_d_TAC1_terminal": d13,
        "endpoint_distance_is_largest": endpoint_largest,
        "centroid_chain_ratio": float(
            chain_ratio
        ),
        "silhouette_state_labels": sil,
    }


def continuity_screen(
    hair,
    rna,
    atac_filtered,
):
    rows = []

    modalities = [
        (
            "RNA_PCA",
            rna,
        ),
        (
            "ATAC_LSI_DEPTH_FILTERED",
            atac_filtered,
        ),
    ]

    for branch_name, states in BRANCHES.items():
        terminal = states[
            2
        ]

        for modality, representation in modalities:
            for n_dims in DIMENSION_SENSITIVITY:
                if n_dims > representation.shape[
                    1
                ]:
                    continue

                x = representation[
                    :,
                    :n_dims
                ]

                for k in K_VALUES:
                    metrics = branch_neighbor_metrics(
                        hair,
                        x,
                        states,
                        k,
                    )

                    rows.append(
                        {
                            "branch": branch_name,
                            "terminal_state": terminal,
                            "modality": modality,
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
    args = parse_args()

    section(
        "PHASE F11 — SHARE-seq MODALITY REPRESENTATIONS + PUBLISHED-BRANCH CONTINUITY SCREEN"
    )

    print(
        "Candidate population:"
    )

    print(
        "  TAC-1, TAC-2, IRS, Medulla, Hair Shaft-cuticle.cortex"
    )

    print()

    print(
        "Published branches screened:"
    )

    for name, states in BRANCHES.items():
        print(
            "  "
            + " -> ".join(
                states
            )
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

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Candidate cells.
    # -----------------------------------------------------------------

    section(
        "1. FREEZE BROAD HAIR-FOLLICLE CANDIDATE POPULATION"
    )

    hair = load_hair_candidate_mapping()

    counts = (
        hair[
            "celltype"
        ]
        .value_counts()
        .reindex(
            HAIR_LABELS,
        )
        .rename(
            "n_cells"
        )
        .to_frame()
    )

    print_df(
        counts,
        digits=0,
    )

    print()

    print(
        f"Total candidate cells: {len(hair):,}"
    )

    hair.to_csv(
        OUTPUT_CELLS,
        sep="\t",
        index=False,
        compression="gzip",
    )

    # -----------------------------------------------------------------
    # RNA.
    # -----------------------------------------------------------------

    section(
        "2. CONSTRUCT RNA REPRESENTATION"
    )

    rna_result = reconstruct_rna_pca(
        hair
    )

    rna = rna_result[
        "representation"
    ]

    np.save(
        OUTPUT_RNA,
        rna
    )

    # -----------------------------------------------------------------
    # ATAC.
    # -----------------------------------------------------------------

    section(
        "3. CONSTRUCT ATAC REPRESENTATION"
    )

    atac_result = reconstruct_atac_lsi(
        hair
    )

    atac = atac_result[
        "representation"
    ]

    np.save(
        OUTPUT_ATAC,
        atac
    )

    # -----------------------------------------------------------------
    # Technical correlations.
    # -----------------------------------------------------------------

    section(
        "4. REPRESENTATION TECHNICAL-COVARIATE AUDIT"
    )

    rna_corr = dimension_correlation_table(
        rna,
        {
            "log1p_RNA_total_counts": np.log1p(
                rna_result[
                    "library_size"
                ]
            ),
            "RNA_detected_features": rna_result[
                "detected_features"
            ],
        },
        "RNA_PCA",
    )

    (
        atac_filtered,
        retained_atac,
        flagged_atac,
        atac_dim_qc,
    ) = filter_atac_depth_axes(
        atac,
        atac_result[
            "cell_depth"
        ],
    )

    np.save(
        OUTPUT_ATAC_FILTERED,
        atac_filtered.astype(
            np.float32
        ),
    )

    atac_dim_qc[
        "retained"
    ] = ~atac_dim_qc[
        "depth_dominant"
    ]

    atac_dim_qc.to_csv(
        OUTPUT_ATAC_DIM_QC,
        sep="\t",
        index=False,
    )

    subsection(
        "RNA PCA — strongest absolute technical correlations"
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
        "ATAC LSI — depth-dominant axis rule"
    )

    print_df(
        atac_dim_qc.set_index(
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
                for i in flagged_atac
            )
            if flagged_atac
            else "<none>"
        )
    )

    print(
        f"Retained ATAC dimensions: {len(retained_atac)}"
    )

    print(
        f"Depth-filtered ATAC shape: {atac_filtered.shape}"
    )

    # -----------------------------------------------------------------
    # Branch continuity.
    # -----------------------------------------------------------------

    section(
        "5. PUBLISHED-BRANCH CONTINUITY SCREEN"
    )

    continuity = continuity_screen(
        hair,
        rna,
        atac_filtered,
    )

    continuity.to_csv(
        OUTPUT_CONTINUITY,
        sep="\t",
        index=False,
        compression="gzip",
    )

    primary_view = continuity.loc[
        (
            continuity[
                "modality"
            ]
            == "RNA_PCA"
        )
        & (
            continuity[
                "n_dims"
            ]
            == PRIMARY_DIMS
        )
        & (
            continuity[
                "k"
            ]
            == PRIMARY_K
        )
    ].copy()

    primary_columns = [
        "branch",
        "terminal_state",
        "n_cells",
        "connected_components",
        "cross_state_neighbor_fraction",
        "adjacent_fraction_of_cross_edges",
        "bypass_fraction_of_cross_edges",
        "TAC1_to_TAC2_neighbor_fraction",
        "TAC2_to_terminal_neighbor_fraction",
        "terminal_to_TAC2_neighbor_fraction",
        "min_required_bridge_fraction",
        "endpoint_distance_is_largest",
        "centroid_chain_ratio",
        "silhouette_state_labels",
    ]

    subsection(
        "RNA primary view — 10D, k=30"
    )

    print_df(
        primary_view[
            primary_columns
        ].set_index(
            "branch"
        ),
        digits=6,
    )

    atac_primary_view = continuity.loc[
        (
            continuity[
                "modality"
            ]
            == "ATAC_LSI_DEPTH_FILTERED"
        )
        & (
            continuity[
                "n_dims"
            ]
            == PRIMARY_DIMS
        )
        & (
            continuity[
                "k"
            ]
            == PRIMARY_K
        )
    ].copy()

    subsection(
        "ATAC depth-filtered sensitivity view — 10D, k=30"
    )

    print_df(
        atac_primary_view[
            primary_columns
        ].set_index(
            "branch"
        ),
        digits=6,
    )

    # Stability summary across RNA dimension/k sensitivity.
    subsection(
        "RNA continuity sensitivity ranges across 10/20/30/40/50D and k=20/30/50"
    )

    rna_all = continuity.loc[
        continuity[
            "modality"
        ]
        == "RNA_PCA"
    ]

    stability = (
        rna_all.groupby(
            "branch"
        )
        .agg(
            n_runs=(
                "branch",
                "size",
            ),
            max_connected_components=(
                "connected_components",
                "max",
            ),
            cross_state_fraction_min=(
                "cross_state_neighbor_fraction",
                "min",
            ),
            cross_state_fraction_median=(
                "cross_state_neighbor_fraction",
                "median",
            ),
            adjacent_cross_fraction_min=(
                "adjacent_fraction_of_cross_edges",
                "min",
            ),
            adjacent_cross_fraction_median=(
                "adjacent_fraction_of_cross_edges",
                "median",
            ),
            min_required_bridge_min=(
                "min_required_bridge_fraction",
                "min",
            ),
            min_required_bridge_median=(
                "min_required_bridge_fraction",
                "median",
            ),
            endpoint_largest_fraction=(
                "endpoint_distance_is_largest",
                "mean",
            ),
            chain_ratio_min=(
                "centroid_chain_ratio",
                "min",
            ),
            chain_ratio_median=(
                "centroid_chain_ratio",
                "median",
            ),
            silhouette_min=(
                "silhouette_state_labels",
                "min",
            ),
            silhouette_median=(
                "silhouette_state_labels",
                "median",
            ),
            silhouette_max=(
                "silhouette_state_labels",
                "max",
            ),
        )
    )

    print_df(
        stability,
        digits=6,
    )

    # -----------------------------------------------------------------
    # Structural safeguards.
    # -----------------------------------------------------------------

    section(
        "6. PHASE F11 REPRESENTATION SAFEGUARDS"
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    f"Candidate population has exactly "
                    f"{EXPECTED_HAIR_CELLS:,} paired cells"
                ),
                "pass": (
                    len(
                        hair
                    )
                    == EXPECTED_HAIR_CELLS
                ),
            },
            {
                "criterion": (
                    "All five prespecified candidate labels are present"
                ),
                "pass": (
                    set(
                        HAIR_LABELS
                    )
                    == set(
                        hair[
                            "celltype"
                        ].unique()
                    )
                ),
            },
            {
                "criterion": (
                    "RNA PCA is 7,197 x 50 and finite"
                ),
                "pass": (
                    rna.shape
                    == (
                        EXPECTED_HAIR_CELLS,
                        RNA_PCA_DIMS,
                    )
                    and np.isfinite(
                        rna
                    ).all()
                ),
            },
            {
                "criterion": (
                    "ATAC LSI is 7,197 x 50 and finite"
                ),
                "pass": (
                    atac.shape
                    == (
                        EXPECTED_HAIR_CELLS,
                        ATAC_LSI_DIMS,
                    )
                    and np.isfinite(
                        atac
                    ).all()
                ),
            },
            {
                "criterion": (
                    "ATAC depth filter retains >=40 dimensions"
                ),
                "pass": (
                    atac_filtered.shape[
                        1
                    ]
                    >= 40
                ),
            },
            {
                "criterion": (
                    "Every candidate cell has nonzero RNA depth"
                ),
                "pass": (
                    rna_result[
                        "library_size"
                    ].min()
                    > 0
                ),
            },
            {
                "criterion": (
                    "Every candidate cell has nonzero ATAC depth"
                ),
                "pass": (
                    atac_result[
                        "cell_depth"
                    ].min()
                    > 0
                ),
            },
            {
                "criterion": (
                    "All three published branches have >=1,000 cells"
                ),
                "pass": bool(
                    (
                        primary_view[
                            "n_cells"
                        ]
                        >= 1000
                    ).all()
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
    # Manifest.
    # -----------------------------------------------------------------

    section(
        "7. SAVE F11 REPRODUCIBILITY MANIFEST"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F11",
        "created_utc": utc_now_iso(),
        "candidate_population": {
            "labels": HAIR_LABELS,
            "n_cells": len(
                hair
            ),
            "label_counts": {
                str(
                    key
                ): int(
                    value
                )
                for key, value in hair[
                    "celltype"
                ].value_counts().items()
            },
        },
        "published_branches_screened": BRANCHES,
        "rna": {
            "normalization_target": RNA_NORMALIZATION_TARGET,
            "hvg_count": RNA_HVG_COUNT,
            "pca_dimensions": RNA_PCA_DIMS,
            "scale_clip": RNA_SCALE_CLIP,
            "output": str(
                OUTPUT_RNA
            ),
            "sha256": sha256_file(
                OUTPUT_RNA
            ),
        },
        "atac": {
            "input_features": int(
                atac_result[
                    "scan"
                ][
                    "n_peaks"
                ]
            ),
            "selected_features": ATAC_FEATURE_COUNT,
            "representation": "TF-IDF + TruncatedSVD",
            "lsi_dimensions": ATAC_LSI_DIMS,
            "depth_axis_threshold_abs_rho": ATAC_EXTREME_DEPTH_RHO,
            "flagged_dimensions": [
                i + 1
                for i in flagged_atac
            ],
            "retained_dimensions": [
                i + 1
                for i in retained_atac
            ],
            "output_raw": str(
                OUTPUT_ATAC
            ),
            "output_depth_filtered": str(
                OUTPUT_ATAC_FILTERED
            ),
            "sha256_raw": sha256_file(
                OUTPUT_ATAC
            ),
            "sha256_depth_filtered": sha256_file(
                OUTPUT_ATAC_FILTERED
            ),
        },
        "continuity_screen": {
            "dimension_sensitivity": DIMENSION_SENSITIVITY,
            "k_sensitivity": K_VALUES,
            "primary_view_dimensions": PRIMARY_DIMS,
            "primary_view_k": PRIMARY_K,
            "branch_frozen": False,
            "output": str(
                OUTPUT_CONTINUITY
            ),
        },
        "checks": checks.to_dict(
            orient="records"
        ),
        "guardrails": {
            "pseudotime_calculated": False,
            "branch_frozen": False,
            "gdis_calculated": False,
            "cmil_calculated": False,
            "lead_lag_tested": False,
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
        OUTPUT_CELLS,
        OUTPUT_HVG,
        OUTPUT_ATAC_PEAKS,
        OUTPUT_RNA,
        OUTPUT_ATAC,
        OUTPUT_ATAC_FILTERED,
        OUTPUT_ATAC_DIM_QC,
        OUTPUT_CONTINUITY,
        OUTPUT_MANIFEST,
    ]:
        print(
            path
        )

    # -----------------------------------------------------------------
    # Cleanup.
    # -----------------------------------------------------------------

    if not args.keep_temp:
        RNA_MEMMAP_FILE.unlink(
            missing_ok=True
        )

        ATAC_PAIR_FILE.unlink(
            missing_ok=True
        )

        try:
            TEMP_DIR.rmdir()
        except OSError:
            pass

    # -----------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------

    section(
        "8. PHASE F11 DECISION"
    )

    if all_pass:
        print(
            "PHASE F11 VERDICT: GO — MODALITY-SPECIFIC REPRESENTATIONS "
            "CONSTRUCTED; BRANCH CONTINUITY SCREEN COMPLETE"
        )

        print()

        print(
            "No branch is frozen automatically."
        )

        print()

        print(
            "Review the RNA continuity sensitivity table before selecting "
            "one published branch for an independent RNA-only common clock."
        )

        print()

        print(
            "ATAC continuity is secondary evidence only and did not influence "
            "the branch choice."
        )

    else:
        print(
            "PHASE F11 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not construct pseudotime until the failed representation "
            "criterion is understood."
        )

    print()

    print(
        "No pseudotime was calculated."
    )

    print(
        "No branch was frozen."
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

