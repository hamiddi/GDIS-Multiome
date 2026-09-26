#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
17_phase_f8_geo_native_reconstruction_qc_dynamicbins.py
=========================================================

Project:
GDIS-Multiome external validation

Dataset:
GSE205117

PHASE F8 — GEO-NATIVE REPRESENTATION RECONSTRUCTION + QC (CONTIG FIX)
--------------------------------------------------------

WHY THIS VERSION EXISTS
-----------------------
The publication authors distribute processed RNA PCA and ATAC LSI files through
an FTP server. On the target HPC system, outbound FTP port 21 times out.

To avoid making the validation pipeline depend on a network path that is not
accessible from the analysis environment, this version reconstructs
modality-specific low-dimensional representations directly from the already
downloaded and checksum-verified GEO raw archive:

    data/GSE205117/GSE205117_RAW.tar

No FTP access is required.

ATAC IMPLEMENTATION NOTE
------------------------
GSE205117 Cell Ranger ARC fragments contain valid reference metadata but
do not provide contig-length header records.  Fixed 10-kb ATAC bins are
therefore encoded directly from each observed fragment's chromosome and
midpoint; no chromosome-size table or second genome scan is required.

SCIENTIFIC DESIGN
-----------------
The representations remain completely modality-specific:

RNA:
    raw GEO GEX matrices
        -> select paired primary WT lineage cells
        -> library-size normalization
        -> log1p
        -> top 2,500 variable genes
        -> standardized PCA
        -> 50 dimensions

ATAC:
    raw GEO ATAC fragments
        -> same paired primary WT lineage cells
        -> 10-kb fixed genomic bins
        -> select top 25,000 accessible bins
        -> binary accessibility
        -> TF-IDF
        -> LSI / TruncatedSVD
        -> 50 dimensions

A fixed-bin ATAC representation is used deliberately so external validation
does not depend on an inaccessible author-specific peak object or on a new,
data-dependent peak-calling step.

FROZEN PRIMARY EXTERNAL LINEAGE
-------------------------------
genotype == "WT"

celltype.mapped in:
    NMP
    Paraxial_mesoderm
    Somitic_mesoderm

and:
    pass_rnaQC == True
    pass_atacQC == True
    celltype.mapped not missing
    present in the GEX matrix

F7b already demonstrated 100% ATAC fragment presence for the joint-QC and
mapped+joint-QC metadata universes.

IMPORTANT
---------
The T_KO arm is NOT included in the primary WT representation construction.

THIS PHASE DOES NOT
-------------------
- calculate pseudotime;
- build DPT;
- calculate GDIS;
- calculate CMIL;
- pair GDIS peaks;
- change the frozen GSE275562 discovery analysis.

OUTPUTS
-------
Created under:

    data/GSE205117/representations_geo_native/

Necessary reproducibility artifacts:
    f8_primary_cells.tsv.gz
    f8_rna_hvg_2500.tsv.gz
    f8_atac_bins_25000.tsv.gz
    f8_rna_pca_50.npy
    f8_atac_lsi_50.npy
    f8_reconstruction_manifest.json

Temporary ATAC row/bin files are deleted by default.

RUN
---
    python 17_phase_f8_geo_native_reconstruction_qc_dynamicbins.py

OPTIONAL
--------
    --keep-temp
        Retain intermediate ATAC row/bin binary files.

    --force
        Reconstruct representations even if final F8 artifacts already exist.

DEPENDENCIES
------------
    numpy
    pandas
    scipy
    scikit-learn

No R, ArchR, Scanpy, or FTP access is required.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import tarfile
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

try:
    from sklearn.decomposition import PCA, TruncatedSVD
except ImportError as exc:
    raise SystemExit(
        "\nERROR: scikit-learn is required for Phase F8.\n"
        "Install it in the active environment, for example:\n\n"
        "    conda install -c conda-forge scikit-learn\n"
    ) from exc


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE205117"

DATA_DIR = Path("data") / ACCESSION
RAW_TAR = DATA_DIR / "GSE205117_RAW.tar"
METADATA_FILE = DATA_DIR / "GSE205117_cell_metadata.txt.gz"
ACQUISITION_MANIFEST = DATA_DIR / "acquisition_manifest.json"

OUTPUT_DIR = DATA_DIR / "representations_geo_native"
TEMP_DIR = OUTPUT_DIR / "atac_temp"

CELLS_FILE = OUTPUT_DIR / "f8_primary_cells.tsv.gz"
RNA_HVG_FILE = OUTPUT_DIR / "f8_rna_hvg_2500.tsv.gz"
ATAC_BIN_FILE = OUTPUT_DIR / "f8_atac_bins_25000.tsv.gz"

RNA_PCA_FILE = OUTPUT_DIR / "f8_rna_pca_50.npy"
ATAC_LSI_FILE = OUTPUT_DIR / "f8_atac_lsi_50.npy"

F8_MANIFEST = OUTPUT_DIR / "f8_reconstruction_manifest.json"


# =====================================================================
# FROZEN BIOLOGICAL / REPRESENTATION SETTINGS
# =====================================================================

CELL_COL = "cell"
BARCODE_COL = "barcode"
SAMPLE_COL = "sample"
STAGE_COL = "stage"
GENOTYPE_COL = "genotype"
CELLTYPE_COL = "celltype.mapped"

RNA_QC_COL = "pass_rnaQC"
ATAC_QC_COL = "pass_atacQC"

RNA_DEPTH_COL = "nCount_RNA"
RNA_FEATURE_COL = "nFeature_RNA"

ATAC_DEPTH_COL = "nFrags_atac"
ATAC_TSS_COL = "TSSEnrichment_atac"

PRIMARY_GENOTYPE = "WT"

PRIMARY_LINEAGE_STATES = [
    "NMP",
    "Paraxial_mesoderm",
    "Somitic_mesoderm",
]

RNA_TARGET_LIBRARY_SIZE = 10_000.0
RNA_HVG_COUNT = 2_500
RNA_PCA_DIMS = 50

ATAC_BIN_SIZE = 10_000
ATAC_FEATURE_COUNT = 25_000

# Stable fixed-bin encoding.  Each canonical chromosome receives a block
# of 100,000 possible 10-kb bins (= 1 Gb of coordinate space), which is
# comfortably larger than any mouse chromosome.  This removes any need
# for chromosome lengths in the fragment-file header.
ATAC_CHROM_BIN_BLOCK = 100_000
ATAC_LSI_DIMS = 50

# Only standard mouse chromosomes are used for the fixed-bin ATAC state space.
CANONICAL_CHROMS = [
    *(f"chr{i}" for i in range(1, 20)),
    "chrX",
    "chrY",
]

RANDOM_SEED = 785

DIMENSION_SENSITIVITY = [
    10,
    20,
    30,
    40,
    50,
]

KNN_K = 30
MIN_GROUP_SIZE_FOR_MIXING = KNN_K + 5

MIN_PRIMARY_TOTAL = 1_000
MIN_STATE_CELLS = 200
MIN_REPRESENTATION_COVERAGE = 0.95

# ATAC streaming.
ATAC_PAIR_BUFFER_SIZE = 500_000
ATAC_BINARY_READ_ROWS = 2_000_000


# =====================================================================
# RAW TAR MEMBER PATTERNS
# =====================================================================

GEX_PATTERN = re.compile(
    r"^(?:.*/)?"
    r"(GSM\d+)_"
    r"(.+?)_"
    r"GEX_"
    r"(barcodes|features|matrix)"
    r"\."
    r"(tsv|mtx)"
    r"\.gz$"
)

ATAC_PATTERN = re.compile(
    r"^(?:.*/)?"
    r"(GSM\d+)_"
    r"(.+?)_"
    r"ATAC_fragments"
    r"\.tsv"
    r"\.gz$"
)

CONTIG_HEADER_PATTERN = re.compile(
    r"^#contig=<ID=([^,>]+),length=(\d+)>"
)


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
        "display.max_colwidth", 100,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string())


def human_size(n_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n_bytes)

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"

        value /= 1024.0

    return f"{n_bytes} B"


def utc_now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct independent RNA PCA and ATAC LSI representations "
            "for GSE205117 directly from the checksum-verified GEO raw data."
        )
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help=(
            "Keep intermediate ATAC row/bin binary files after the final "
            "ATAC matrix has been constructed."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reconstruct final F8 artifacts even when they already exist."
        ),
    )

    return parser.parse_args()


# =====================================================================
# GENERAL HELPERS
# =====================================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                16 * 1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def normalize_bool(series):
    mapping = {
        "true": True,
        "t": True,
        "1": True,
        "yes": True,
        "pass": True,
        "passed": True,
        "false": False,
        "f": False,
        "0": False,
        "no": False,
        "fail": False,
        "failed": False,
    }

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


def require_inputs():
    missing = [
        path
        for path in [
            RAW_TAR,
            METADATA_FILE,
            ACQUISITION_MANIFEST,
        ]
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Required prior-phase input(s) are missing:\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )


def verify_raw_size_against_manifest():
    with ACQUISITION_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    raw_record = (
        manifest
        .get("files", {})
        .get("raw_archive")
    )

    if raw_record is None:
        raise RuntimeError(
            "Phase F6a acquisition manifest has no raw_archive record."
        )

    expected_size = int(
        raw_record[
            "size_bytes"
        ]
    )

    observed_size = RAW_TAR.stat().st_size

    if expected_size != observed_size:
        raise RuntimeError(
            "Raw TAR size no longer agrees with the Phase F6a manifest."
        )

    return {
        "recorded_sha256": raw_record[
            "sha256"
        ],
        "expected_size": expected_size,
        "observed_size": observed_size,
        "size_match": True,
    }


# =====================================================================
# METADATA
# =====================================================================

def load_metadata():
    df = pd.read_csv(
        METADATA_FILE,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    required = [
        CELL_COL,
        BARCODE_COL,
        SAMPLE_COL,
        STAGE_COL,
        GENOTYPE_COL,
        CELLTYPE_COL,
        RNA_QC_COL,
        ATAC_QC_COL,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Metadata is missing required field(s): "
            + ", ".join(
                missing
            )
        )

    df[
        SAMPLE_COL
    ] = df[
        SAMPLE_COL
    ].astype(str)

    df[
        BARCODE_COL
    ] = df[
        BARCODE_COL
    ].astype(str)

    df[
        "_sample_barcode"
    ] = (
        df[
            SAMPLE_COL
        ]
        + "::"
        + df[
            BARCODE_COL
        ]
    )

    if df[
        "_sample_barcode"
    ].duplicated().any():
        raise RuntimeError(
            "sample+barcode is not unique. "
            "This contradicts the frozen F7/F7b pairing result."
        )

    df[
        "_pass_rna"
    ] = normalize_bool(
        df[
            RNA_QC_COL
        ]
    ).fillna(
        False
    )

    df[
        "_pass_atac"
    ] = normalize_bool(
        df[
            ATAC_QC_COL
        ]
    ).fillna(
        False
    )

    df[
        "_primary_candidate"
    ] = (
        df[
            "_pass_rna"
        ]
        & df[
            "_pass_atac"
        ]
        & df[
            CELLTYPE_COL
        ].notna()
        & df[
            GENOTYPE_COL
        ].eq(
            PRIMARY_GENOTYPE
        )
        & df[
            CELLTYPE_COL
        ].isin(
            PRIMARY_LINEAGE_STATES
        )
    )

    return df


# =====================================================================
# TAR INVENTORY / NESTED GZIP
# =====================================================================

def inventory_tar():
    inventory = defaultdict(
        lambda: {
            "gex_barcodes": [],
            "gex_features": [],
            "gex_matrix": [],
            "atac_fragments": [],
        }
    )

    with tarfile.open(
        RAW_TAR,
        "r",
    ) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue

            name = member.name

            gex = GEX_PATTERN.match(
                name
            )

            if gex:
                _, sample, component, _ = gex.groups()

                inventory[
                    sample
                ][
                    f"gex_{component}"
                ].append(
                    name
                )

                continue

            atac = ATAC_PATTERN.match(
                name
            )

            if atac:
                _, sample = atac.groups()

                inventory[
                    sample
                ][
                    "atac_fragments"
                ].append(
                    name
                )

    return inventory


def open_gzip_member_binary(
    tar,
    member_name,
):
    member = tar.getmember(
        member_name
    )

    raw = tar.extractfile(
        member
    )

    if raw is None:
        raise RuntimeError(
            f"Could not open TAR member: {member_name}"
        )

    return gzip.GzipFile(
        fileobj=raw,
        mode="rb",
    )


def open_gzip_member_text(
    tar,
    member_name,
):
    gz = open_gzip_member_binary(
        tar,
        member_name,
    )

    return io.TextIOWrapper(
        gz,
        encoding="utf-8",
        errors="replace",
        newline="",
    )


def read_barcode_member(
    tar,
    member_name,
):
    values = []

    with open_gzip_member_text(
        tar,
        member_name,
    ) as handle:
        for line_text in handle:
            value = line_text.strip()

            if value:
                values.append(
                    value
                )

    return values


def read_feature_member(
    tar,
    member_name,
):
    rows = []

    with open_gzip_member_text(
        tar,
        member_name,
    ) as handle:
        for line_text in handle:
            fields = line_text.rstrip(
                "\n"
            ).split(
                "\t"
            )

            if len(
                fields
            ) < 2:
                continue

            feature_id = fields[
                0
            ]

            feature_name = fields[
                1
            ]

            feature_type = (
                fields[
                    2
                ]
                if len(
                    fields
                ) >= 3
                else "Gene Expression"
            )

            rows.append(
                (
                    feature_id,
                    feature_name,
                    feature_type,
                )
            )

    return rows


# =====================================================================
# IDENTIFY PRIMARY CELLS PRESENT IN GEX
# =====================================================================

def freeze_primary_gex_cells(
    metadata,
    inventory,
):
    """
    Intersect the frozen metadata-defined primary WT lineage with raw GEX
    barcode presence, sample by sample.

    F7b already established complete ATAC presence for mapped+joint-QC cells.
    """
    candidate = metadata.loc[
        metadata[
            "_primary_candidate"
        ]
    ].copy()

    candidate[
        "_in_gex"
    ] = False

    gex_presence_rows = []

    with tarfile.open(
        RAW_TAR,
        "r",
    ) as tar:

        for sample in sorted(
            candidate[
                SAMPLE_COL
            ].unique()
        ):
            member_list = inventory[
                sample
            ][
                "gex_barcodes"
            ]

            if len(
                member_list
            ) != 1:
                raise RuntimeError(
                    f"Expected exactly one GEX barcode member for {sample}; "
                    f"found {len(member_list)}"
                )

            gex_barcodes = set(
                read_barcode_member(
                    tar,
                    member_list[
                        0
                    ],
                )
            )

            mask = (
                candidate[
                    SAMPLE_COL
                ].eq(
                    sample
                )
            )

            sample_barcodes = candidate.loc[
                mask,
                BARCODE_COL,
            ].astype(
                str
            )

            present = sample_barcodes.isin(
                gex_barcodes
            )

            candidate.loc[
                mask,
                "_in_gex",
            ] = present.to_numpy()

            gex_presence_rows.append(
                {
                    "sample": sample,
                    "primary_candidate": int(
                        mask.sum()
                    ),
                    "present_in_gex": int(
                        present.sum()
                    ),
                    "missing_from_gex": int(
                        (
                            ~present
                        ).sum()
                    ),
                    "coverage": float(
                        present.mean()
                    ) if len(
                        present
                    ) else np.nan,
                }
            )

    paired_primary = candidate.loc[
        candidate[
            "_in_gex"
        ]
    ].copy()

    paired_primary = paired_primary.sort_values(
        [
            STAGE_COL,
            CELLTYPE_COL,
            SAMPLE_COL,
            BARCODE_COL,
        ]
    ).reset_index(
        drop=True
    )

    paired_primary[
        "_f8_row"
    ] = np.arange(
        len(
            paired_primary
        ),
        dtype=int,
    )

    return (
        paired_primary,
        pd.DataFrame(
            gex_presence_rows
        ).set_index(
            "sample"
        ),
    )


# =====================================================================
# RNA RECONSTRUCTION
# =====================================================================

def common_gene_definition(
    tar,
    inventory,
    samples,
):
    """
    Determine the common Gene Expression feature universe across the samples
    contributing primary-lineage cells.
    """
    feature_tables = {}

    for sample in samples:
        member_list = inventory[
            sample
        ][
            "gex_features"
        ]

        if len(
            member_list
        ) != 1:
            raise RuntimeError(
                f"Expected exactly one GEX feature member for {sample}."
            )

        features = read_feature_member(
            tar,
            member_list[
                0
            ],
        )

        genes = [
            (
                feature_id,
                feature_name,
            )
            for (
                feature_id,
                feature_name,
                feature_type,
            ) in features
            if feature_type == "Gene Expression"
        ]

        if not genes:
            # Conservative fallback for legacy 2-column feature files.
            genes = [
                (
                    feature_id,
                    feature_name,
                )
                for (
                    feature_id,
                    feature_name,
                    _
                ) in features
            ]

        feature_tables[
            sample
        ] = genes

    first_sample = samples[
        0
    ]

    first_ids = [
        feature_id
        for feature_id, _ in feature_tables[
            first_sample
        ]
    ]

    common_ids = set(
        first_ids
    )

    for sample in samples[
        1:
    ]:
        common_ids &= {
            feature_id
            for feature_id, _ in feature_tables[
                sample
            ]
        }

    ordered_common_ids = [
        feature_id
        for feature_id in first_ids
        if feature_id in common_ids
    ]

    first_name_map = {
        feature_id: feature_name
        for feature_id, feature_name in feature_tables[
            first_sample
        ]
    }

    common_names = [
        first_name_map.get(
            feature_id,
            feature_id,
        )
        for feature_id in ordered_common_ids
    ]

    return (
        feature_tables,
        ordered_common_ids,
        common_names,
    )


def read_sample_gex_matrix(
    tar,
    inventory,
    sample,
    sample_primary,
    feature_table,
    common_gene_ids,
):
    """
    Read one sample's raw GEX MatrixMarket matrix and subset it immediately to:
      - common Gene Expression rows;
      - F8 primary cells.

    Returns cells x common_genes CSR.
    """
    barcode_member = inventory[
        sample
    ][
        "gex_barcodes"
    ][0]

    matrix_member = inventory[
        sample
    ][
        "gex_matrix"
    ][0]

    barcodes = read_barcode_member(
        tar,
        barcode_member,
    )

    barcode_to_col = {
        barcode: i
        for i, barcode in enumerate(
            barcodes
        )
    }

    selected_barcodes = sample_primary[
        BARCODE_COL
    ].astype(
        str
    ).tolist()

    missing = [
        barcode
        for barcode in selected_barcodes
        if barcode not in barcode_to_col
    ]

    if missing:
        raise RuntimeError(
            f"{sample}: {len(missing)} frozen F8 cells are unexpectedly "
            "missing from the GEX barcode file."
        )

    selected_cols = np.array(
        [
            barcode_to_col[
                barcode
            ]
            for barcode in selected_barcodes
        ],
        dtype=int,
    )

    # Map common gene IDs to rows in this sample's full feature table.
    full_gene_rows = {}

    for row_index, (
        feature_id,
        _feature_name,
        feature_type,
    ) in enumerate(
        feature_table
    ):
        if feature_type == "Gene Expression":
            full_gene_rows[
                feature_id
            ] = row_index

    if not full_gene_rows:
        # Legacy 2-column fallback.
        full_gene_rows = {
            feature_id: row_index
            for row_index, (
                feature_id,
                _feature_name,
                _feature_type,
            ) in enumerate(
                feature_table
            )
        }

    selected_rows = np.array(
        [
            full_gene_rows[
                feature_id
            ]
            for feature_id in common_gene_ids
        ],
        dtype=int,
    )

    # scipy.io.mmread can read a decompressed binary file-like stream.
    with open_gzip_member_binary(
        tar,
        matrix_member,
    ) as matrix_stream:
        matrix = mmread(
            matrix_stream
        )

    if not sp.issparse(
        matrix
    ):
        matrix = sp.coo_matrix(
            matrix
        )

    matrix = matrix.tocsr()

    if matrix.shape[
        1
    ] != len(
        barcodes
    ):
        raise RuntimeError(
            f"{sample}: GEX matrix/barcode dimension mismatch. "
            f"matrix={matrix.shape}, barcodes={len(barcodes)}"
        )

    if matrix.shape[
        0
    ] < (
        int(
            selected_rows.max()
        )
        + 1
    ):
        raise RuntimeError(
            f"{sample}: feature row index exceeds matrix dimensions."
        )

    subset = matrix[
        selected_rows,
        :,
    ][
        :,
        selected_cols,
    ]

    # Return cell x gene.
    return subset.T.tocsr()


def reconstruct_rna_pca(
    primary,
    inventory,
):
    subsection(
        "RNA: reconstruct GEX matrix"
    )

    samples = [
        sample
        for sample in sorted(
            primary[
                SAMPLE_COL
            ].unique()
        )
    ]

    with tarfile.open(
        RAW_TAR,
        "r",
    ) as tar:

        feature_tables_raw = {}

        # Read complete feature tables first.
        for sample in samples:
            member = inventory[
                sample
            ][
                "gex_features"
            ][0]

            feature_tables_raw[
                sample
            ] = read_feature_member(
                tar,
                member,
            )

        # Derive common Gene Expression IDs.
        first_sample = samples[
            0
        ]

        def gene_pairs(sample):
            values = [
                (
                    fid,
                    fname,
                )
                for fid, fname, ftype in feature_tables_raw[
                    sample
                ]
                if ftype == "Gene Expression"
            ]

            if not values:
                values = [
                    (
                        fid,
                        fname,
                    )
                    for fid, fname, _ in feature_tables_raw[
                        sample
                    ]
                ]

            return values

        first_pairs = gene_pairs(
            first_sample
        )

        first_ids = [
            fid
            for fid, _ in first_pairs
        ]

        common = set(
            first_ids
        )

        for sample in samples[
            1:
        ]:
            common &= {
                fid
                for fid, _ in gene_pairs(
                    sample
                )
            }

        common_ids = [
            fid
            for fid in first_ids
            if fid in common
        ]

        first_names = {
            fid: fname
            for fid, fname in first_pairs
        }

        common_names = [
            first_names.get(
                fid,
                fid,
            )
            for fid in common_ids
        ]

        print(
            f"Samples contributing primary WT lineage: "
            f"{len(samples)}"
        )

        print(
            f"Common Gene Expression features: "
            f"{len(common_ids):,}"
        )

        sample_blocks = []

        sample_key_order = []

        for sample in samples:
            sample_primary = primary.loc[
                primary[
                    SAMPLE_COL
                ].eq(
                    sample
                )
            ].copy()

            print(
                f"Reading GEX: {sample} "
                f"({len(sample_primary):,} selected cells)"
            )

            block = read_sample_gex_matrix(
                tar,
                inventory,
                sample,
                sample_primary,
                feature_tables_raw[
                    sample
                ],
                common_ids,
            )

            sample_blocks.append(
                block
            )

            sample_key_order.extend(
                sample_primary[
                    "_sample_barcode"
                ].tolist()
            )

    x_counts = sp.vstack(
        sample_blocks,
        format="csr",
    ).astype(
        np.float32
    )

    # Reorder from sample-block order to the global frozen F8 row order.
    key_to_block_row = {
        key: i
        for i, key in enumerate(
            sample_key_order
        )
    }

    reorder = np.array(
        [
            key_to_block_row[
                key
            ]
            for key in primary[
                "_sample_barcode"
            ]
        ],
        dtype=int,
    )

    x_counts = x_counts[
        reorder,
        :,
    ]

    print(
        f"RNA count matrix: "
        f"{x_counts.shape[0]:,} cells x {x_counts.shape[1]:,} genes"
    )

    # Library normalization.
    library_size = np.asarray(
        x_counts.sum(
            axis=1
        )
    ).ravel()

    if np.any(
        library_size <= 0
    ):
        raise RuntimeError(
            "At least one primary cell has zero RNA library size."
        )

    scale = (
        RNA_TARGET_LIBRARY_SIZE
        / library_size
    )

    x_norm = sp.diags(
        scale.astype(
            np.float32
        )
    ) @ x_counts

    x_norm = x_norm.tocsr()

    x_norm.data = np.log1p(
        x_norm.data
    )

    # Sparse mean / variance after log-normalization.
    mean = np.asarray(
        x_norm.mean(
            axis=0
        )
    ).ravel()

    mean_sq = np.asarray(
        x_norm.multiply(
            x_norm
        ).mean(
            axis=0
        )
    ).ravel()

    variance = np.maximum(
        mean_sq
        - mean ** 2,
        0.0,
    )

    n_hvg = min(
        RNA_HVG_COUNT,
        len(
            variance
        ),
    )

    hvg_idx = np.argpartition(
        variance,
        -n_hvg,
    )[
        -n_hvg:
    ]

    # Stable descending variance order.
    hvg_idx = hvg_idx[
        np.argsort(
            variance[
                hvg_idx
            ]
        )[
            ::-1
        ]
    ]

    hvg_table = pd.DataFrame(
        {
            "feature_id": np.asarray(
                common_ids,
                dtype=object,
            )[
                hvg_idx
            ],
            "feature_name": np.asarray(
                common_names,
                dtype=object,
            )[
                hvg_idx
            ],
            "log_normalized_variance": variance[
                hvg_idx
            ],
        }
    )

    x_hvg = x_norm[
        :,
        hvg_idx,
    ].toarray().astype(
        np.float32
    )

    gene_mean = x_hvg.mean(
        axis=0
    )

    gene_std = x_hvg.std(
        axis=0,
        ddof=0,
    )

    usable = gene_std > 0

    if int(
        usable.sum()
    ) < RNA_PCA_DIMS:
        raise RuntimeError(
            "Too few non-zero-variance RNA HVGs for 50-dimensional PCA."
        )

    x_hvg = x_hvg[
        :,
        usable,
    ]

    hvg_table = hvg_table.loc[
        usable
    ].reset_index(
        drop=True
    )

    gene_mean = gene_mean[
        usable
    ]

    gene_std = gene_std[
        usable
    ]

    x_scaled = (
        x_hvg
        - gene_mean
    ) / gene_std

    # Clip extreme standardized values for numerical stability.
    x_scaled = np.clip(
        x_scaled,
        -10.0,
        10.0,
    )

    print(
        f"RNA PCA input: "
        f"{x_scaled.shape[0]:,} cells x {x_scaled.shape[1]:,} HVGs"
    )

    pca = PCA(
        n_components=RNA_PCA_DIMS,
        svd_solver="randomized",
        random_state=RANDOM_SEED,
    )

    rna_pca = pca.fit_transform(
        x_scaled
    ).astype(
        np.float32
    )

    explained = pca.explained_variance_ratio_

    print(
        f"RNA PCA shape: {rna_pca.shape}"
    )

    print(
        f"RNA PCA cumulative variance, first 10 PCs: "
        f"{explained[:10].sum():.4f}"
    )

    print(
        f"RNA PCA cumulative variance, first 50 PCs: "
        f"{explained.sum():.4f}"
    )

    return (
        rna_pca,
        hvg_table,
        library_size,
        explained,
    )


# =====================================================================
# ATAC RECONSTRUCTION
# =====================================================================

def _canonical_chromosome_token(chrom):
    """
    Normalize common mouse chromosome naming conventions for selection/order.

    Examples:
        chr1 -> 1
        1    -> 1
        chrX -> X
        X    -> X

    The raw chromosome label is NOT changed in the fragment parser.  This
    helper is only used to recognize canonical chromosomes and determine a
    stable order.
    """
    value = str(chrom).strip()

    if value.lower().startswith("chr"):
        value = value[3:]

    value_upper = value.upper()

    if value_upper in {"X", "Y"}:
        return value_upper

    if value.isdigit():
        number = int(value)

        if 1 <= number <= 19:
            return str(number)

    return None


def _canonical_chromosome_order(token):
    if token is None:
        return 10_000

    if token == "X":
        return 20

    if token == "Y":
        return 21

    return int(token)


def inspect_atac_reference_header(
    tar,
    member_name,
):
    """
    Inspect the 10x Cell Ranger ARC fragment header for provenance only.

    GSE205117 fragment files provide reference metadata such as:
        reference_path
        reference_fasta_hash
        reference_version

    but do NOT provide '#contig=<ID=...,length=...>' records.

    Therefore chromosome lengths are intentionally NOT required by F8.
    """
    header_lines = []
    reference_fields = {}

    with open_gzip_member_text(
        tar,
        member_name,
    ) as handle:
        for line_text in handle:
            if not line_text.startswith(
                "#"
            ):
                break

            stripped = line_text.strip()

            if len(
                header_lines
            ) < 20:
                header_lines.append(
                    stripped
                )

            if "=" in stripped:
                content = stripped.lstrip(
                    "#"
                ).strip()

                if "=" in content:
                    key, value = content.split(
                        "=",
                        1,
                    )

                    key = key.strip()
                    value = value.strip()

                    if key in {
                        "reference_path",
                        "reference_fasta_hash",
                        "reference_gtf_hash",
                        "reference_version",
                        "pipeline_name",
                        "pipeline_version",
                    }:
                        reference_fields[
                            key
                        ] = value

    print(
        f"ATAC header preview lines retained: "
        f"{len(header_lines)}"
    )

    for value in header_lines:
        print(
            f"  {value[:180]}"
        )

    if reference_fields:
        print()
        print(
            "ATAC reference metadata detected:"
        )

        for key in sorted(
            reference_fields
        ):
            print(
                f"  {key}: "
                f"{reference_fields[key]}"
            )

    print()
    print(
        "No chromosome-length header is required. "
        "Observed fragments are assigned directly to fixed 10-kb bins."
    )

    return reference_fields


def _chromosome_code_from_token(
    token,
):
    """
    Convert normalized canonical chromosome token to a compact integer code.

        1..19 -> 1..19
        X     -> 20
        Y     -> 21
    """
    if token is None:
        return None

    if token == "X":
        return 20

    if token == "Y":
        return 21

    return int(
        token
    )


def _chromosome_label_from_code(
    code,
):
    if 1 <= code <= 19:
        return f"chr{code}"

    if code == 20:
        return "chrX"

    if code == 21:
        return "chrY"

    raise ValueError(
        f"Unexpected canonical chromosome code: {code}"
    )


def encode_fixed_atac_bin(
    chrom,
    midpoint,
):
    """
    Encode a canonical chromosome + 10-kb local-bin index as one int32-safe
    global key.

    No chromosome length is needed.  Only bins actually observed in fragments
    are ever retained.
    """
    token = _canonical_chromosome_token(
        chrom
    )

    code = _chromosome_code_from_token(
        token
    )

    if code is None:
        return None

    if midpoint < 0:
        return None

    local_bin = midpoint // ATAC_BIN_SIZE

    if local_bin >= ATAC_CHROM_BIN_BLOCK:
        raise RuntimeError(
            f"ATAC coordinate exceeds the reserved chromosome block: "
            f"chrom={chrom}, midpoint={midpoint:,}, "
            f"local_bin={local_bin:,}"
        )

    return int(
        code
        * ATAC_CHROM_BIN_BLOCK
        + local_bin
    )


def decode_fixed_atac_bin(
    encoded_bin,
):
    """
    Decode the stable integer key to canonical chromosome and coordinates.
    """
    code = int(
        encoded_bin
        // ATAC_CHROM_BIN_BLOCK
    )

    local_bin = int(
        encoded_bin
        % ATAC_CHROM_BIN_BLOCK
    )

    chrom = _chromosome_label_from_code(
        code
    )

    start = (
        local_bin
        * ATAC_BIN_SIZE
    )

    end = (
        start
        + ATAC_BIN_SIZE
    )

    return (
        chrom,
        start,
        end,
        code,
        local_bin,
    )


def flush_pair_buffer(
    path,
    row_buffer,
    col_buffer,
):
    if not row_buffer:
        return

    array = np.empty(
        (
            len(
                row_buffer
            ),
            2,
        ),
        dtype=np.int32,
    )

    array[
        :,
        0,
    ] = np.asarray(
        row_buffer,
        dtype=np.int32,
    )

    array[
        :,
        1,
    ] = np.asarray(
        col_buffer,
        dtype=np.int32,
    )

    with path.open(
        "ab"
    ) as handle:
        array.tofile(
            handle
        )

    row_buffer.clear()
    col_buffer.clear()


def scan_atac_to_temp_pairs(
    tar,
    inventory,
    primary,
):
    """
    Stream each ATAC fragment file once.

    For fragments belonging to the frozen F8 cells:
      - recognize canonical mouse chromosomes directly from the fragment;
      - encode midpoint // 10,000 as a stable chromosome/bin key;
      - count fragment support per observed fixed bin;
      - record (cell_row, encoded_bin) pairs to compact temporary int32 files.

    This is a single-pass procedure and requires no chromosome-length table.
    """
    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bin_totals = defaultdict(
        int
    )

    per_cell_fragments = np.zeros(
        len(
            primary
        ),
        dtype=np.int64,
    )

    primary_by_sample = {
        sample: group.copy()
        for sample, group in primary.groupby(
            SAMPLE_COL,
            sort=True,
        )
    }

    temp_files = []
    scan_rows = []

    canonical_chrom_records = defaultdict(
        int
    )

    for sample, sample_primary in primary_by_sample.items():
        member_list = inventory[
            sample
        ][
            "atac_fragments"
        ]

        if len(
            member_list
        ) != 1:
            raise RuntimeError(
                f"Expected one ATAC fragments member for {sample}."
            )

        member_name = member_list[
            0
        ]

        barcode_to_row = {
            barcode: int(
                row
            )
            for barcode, row in zip(
                sample_primary[
                    BARCODE_COL
                ].astype(
                    str
                ),
                sample_primary[
                    "_f8_row"
                ],
            )
        }

        temp_path = TEMP_DIR / (
            re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                sample,
            )
            + ".row_bin.int32"
        )

        temp_path.unlink(
            missing_ok=True
        )

        row_buffer = []
        col_buffer = []

        header_lines = 0
        fragment_records = 0
        selected_fragment_records = 0
        malformed_records = 0
        noncanonical_records = 0

        print(
            f"Scanning ATAC fragments: {sample} "
            f"({len(sample_primary):,} selected cells)"
        )

        with open_gzip_member_text(
            tar,
            member_name,
        ) as handle:
            for line_text in handle:
                if line_text.startswith(
                    "#"
                ):
                    header_lines += 1
                    continue

                line_text = line_text.rstrip(
                    "\n"
                )

                if not line_text:
                    continue

                fields = line_text.split(
                    "\t"
                )

                if len(
                    fields
                ) < 5:
                    malformed_records += 1
                    continue

                fragment_records += 1

                chrom = fields[
                    0
                ]

                token = _canonical_chromosome_token(
                    chrom
                )

                if token is None:
                    noncanonical_records += 1
                    continue

                barcode = fields[
                    3
                ]

                row = barcode_to_row.get(
                    barcode
                )

                if row is None:
                    continue

                try:
                    start_coord = int(
                        fields[
                            1
                        ]
                    )

                    end_coord = int(
                        fields[
                            2
                        ]
                    )

                except ValueError:
                    malformed_records += 1
                    continue

                midpoint = (
                    start_coord
                    + end_coord
                ) // 2

                encoded_bin = encode_fixed_atac_bin(
                    chrom,
                    midpoint,
                )

                if encoded_bin is None:
                    noncanonical_records += 1
                    continue

                selected_fragment_records += 1

                bin_totals[
                    encoded_bin
                ] += 1

                per_cell_fragments[
                    row
                ] += 1

                canonical_chrom_records[
                    token
                ] += 1

                row_buffer.append(
                    row
                )

                col_buffer.append(
                    encoded_bin
                )

                if len(
                    row_buffer
                ) >= ATAC_PAIR_BUFFER_SIZE:
                    flush_pair_buffer(
                        temp_path,
                        row_buffer,
                        col_buffer,
                    )

        flush_pair_buffer(
            temp_path,
            row_buffer,
            col_buffer,
        )

        temp_files.append(
            temp_path
        )

        scan_rows.append(
            {
                "sample": sample,
                "header_lines": header_lines,
                "fragment_records": fragment_records,
                "selected_fragment_records": selected_fragment_records,
                "malformed_records": malformed_records,
                "noncanonical_records": noncanonical_records,
                "observed_fixed_bins": len(
                    bin_totals
                ),
                "temp_size_bytes": temp_path.stat().st_size,
            }
        )

    print()
    print(
        "Canonical chromosomes represented among selected-cell fragments:"
    )

    ordered_tokens = sorted(
        canonical_chrom_records,
        key=_canonical_chromosome_order,
    )

    for token in ordered_tokens:
        print(
            f"  chr{token}: "
            f"{canonical_chrom_records[token]:,} selected fragment records"
        )

    return (
        dict(
            bin_totals
        ),
        per_cell_fragments,
        temp_files,
        pd.DataFrame(
            scan_rows
        ).set_index(
            "sample"
        ),
    )


def build_selected_atac_matrix(
    temp_files,
    n_cells,
    selected_encoded_bins,
):
    """
    Construct cell x selected-bin sparse matrix from temporary encoded pairs.

    A dictionary is used rather than an array indexed by genome length,
    because fixed-bin keys are sparse chromosome-block encodings.
    """
    feature_map = {
        int(
            encoded_bin
        ): int(
            feature_index
        )
        for feature_index, encoded_bin in enumerate(
            selected_encoded_bins
        )
    }

    x = sp.csr_matrix(
        (
            n_cells,
            len(
                selected_encoded_bins
            ),
        ),
        dtype=np.float32,
    )

    for path in temp_files:
        file_size = path.stat().st_size

        if file_size == 0:
            continue

        n_int32 = (
            file_size
            // np.dtype(
                np.int32
            ).itemsize
        )

        if n_int32 % 2 != 0:
            raise RuntimeError(
                f"Corrupt temporary ATAC pair file: {path}"
            )

        n_rows = n_int32 // 2

        mmap = np.memmap(
            path,
            dtype=np.int32,
            mode="r",
            shape=(
                n_rows,
                2,
            ),
        )

        for start in range(
            0,
            n_rows,
            ATAC_BINARY_READ_ROWS,
        ):
            stop = min(
                start + ATAC_BINARY_READ_ROWS,
                n_rows,
            )

            block = np.asarray(
                mmap[
                    start:
                    stop
                ]
            )

            rows = block[
                :,
                0,
            ]

            encoded_cols = block[
                :,
                1,
            ]

            # Vectorized mapping through sorted selected encoded keys.
            # selected_encoded_bins is kept sorted for deterministic lookup.
            positions = np.searchsorted(
                selected_encoded_bins,
                encoded_cols,
            )

            valid = (
                positions
                < len(
                    selected_encoded_bins
                )
            )

            if np.any(
                valid
            ):
                valid_indices = np.flatnonzero(
                    valid
                )

                exact = (
                    selected_encoded_bins[
                        positions[
                            valid
                        ]
                    ]
                    == encoded_cols[
                        valid
                    ]
                )

                keep_indices = valid_indices[
                    exact
                ]

            else:
                keep_indices = np.empty(
                    0,
                    dtype=int,
                )

            if len(
                keep_indices
            ) == 0:
                continue

            cols = positions[
                keep_indices
            ]

            chunk = sp.coo_matrix(
                (
                    np.ones(
                        len(
                            keep_indices
                        ),
                        dtype=np.float32,
                    ),
                    (
                        rows[
                            keep_indices
                        ],
                        cols,
                    ),
                ),
                shape=x.shape,
            ).tocsr()

            x = x + chunk

        del mmap

    x.sum_duplicates()

    # Binary accessibility is used for TF-IDF / LSI.
    x.data[:] = 1.0

    x.eliminate_zeros()

    return x


def tfidf_lsi(
    x_binary,
):
    row_sum = np.asarray(
        x_binary.sum(
            axis=1
        )
    ).ravel()

    if np.any(
        row_sum <= 0
    ):
        raise RuntimeError(
            "At least one F8 primary cell has zero selected-bin "
            "ATAC accessibility."
        )

    document_frequency = np.asarray(
        (
            x_binary > 0
        ).sum(
            axis=0
        )
    ).ravel()

    inverse_document_frequency = np.log1p(
        x_binary.shape[
            0
        ]
        / (
            1.0
            + document_frequency
        )
    ).astype(
        np.float32
    )

    tf = sp.diags(
        (
            1.0
            / row_sum
        ).astype(
            np.float32
        )
    ) @ x_binary

    tfidf = tf @ sp.diags(
        inverse_document_frequency
    )

    tfidf = tfidf.tocsr()

    svd = TruncatedSVD(
        n_components=ATAC_LSI_DIMS,
        algorithm="randomized",
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
    primary,
    inventory,
    keep_temp=False,
):
    subsection(
        "ATAC: inspect reference header"
    )

    sample0 = sorted(
        primary[
            SAMPLE_COL
        ].unique()
    )[
        0
    ]

    member0 = inventory[
        sample0
    ][
        "atac_fragments"
    ][0]

    with tarfile.open(
        RAW_TAR,
        "r",
    ) as tar:
        reference_fields = inspect_atac_reference_header(
            tar,
            member0,
        )

        subsection(
            "ATAC: one-pass fragment scan and dynamic 10-kb bin capture"
        )

        (
            bin_totals,
            per_cell_fragments,
            temp_files,
            scan_table,
        ) = scan_atac_to_temp_pairs(
            tar,
            inventory,
            primary,
        )

    print_df(
        scan_table,
        digits=0,
    )

    if int(
        scan_table[
            "malformed_records"
        ].sum()
    ) != 0:
        raise RuntimeError(
            "True malformed non-header ATAC records were observed in F8."
        )

    if np.any(
        per_cell_fragments <= 0
    ):
        missing = int(
            (
                per_cell_fragments
                <= 0
            ).sum()
        )

        raise RuntimeError(
            f"{missing} frozen F8 cells had no ATAC fragments."
        )

    if not bin_totals:
        raise RuntimeError(
            "No canonical fixed ATAC bins were observed for the frozen "
            "primary cells."
        )

    observed_bins = np.fromiter(
        bin_totals.keys(),
        dtype=np.int32,
    )

    observed_counts = np.fromiter(
        (
            bin_totals[
                int(
                    key
                )
            ]
            for key in observed_bins
        ),
        dtype=np.int64,
    )

    n_features = min(
        ATAC_FEATURE_COUNT,
        len(
            observed_bins
        ),
    )

    if len(
        observed_bins
    ) > n_features:
        top_idx = np.argpartition(
            observed_counts,
            -n_features,
        )[
            -n_features:
        ]

    else:
        top_idx = np.arange(
            len(
                observed_bins
            )
        )

    # Sort selected bins first by descending support for reporting.
    selected_by_support = top_idx[
        np.argsort(
            observed_counts[
                top_idx
            ]
        )[
            ::-1
        ]
    ]

    selected_encoded_report = observed_bins[
        selected_by_support
    ]

    selected_counts_report = observed_counts[
        selected_by_support
    ]

    print(
        f"Observed canonical 10-kb bins: "
        f"{len(observed_bins):,}"
    )

    print(
        f"Selected ATAC bins for LSI: "
        f"{len(selected_encoded_report):,}"
    )

    selected_rows = []

    for rank, (
        encoded_bin,
        fragment_count,
    ) in enumerate(
        zip(
            selected_encoded_report,
            selected_counts_report,
        ),
        start=1,
    ):
        (
            chrom,
            start_coord,
            end_coord,
            chrom_code,
            local_bin,
        ) = decode_fixed_atac_bin(
            int(
                encoded_bin
            )
        )

        selected_rows.append(
            {
                "feature_rank_by_fragment_support": rank,
                "encoded_bin": int(
                    encoded_bin
                ),
                "chrom": chrom,
                "start": start_coord,
                "end": end_coord,
                "chromosome_code": chrom_code,
                "local_bin": local_bin,
                "selected_fragment_records": int(
                    fragment_count
                ),
            }
        )

    selected_bins = pd.DataFrame(
        selected_rows
    )

    # Matrix construction uses ascending encoded keys so np.searchsorted can
    # map millions of temporary pairs efficiently and deterministically.
    selected_encoded_matrix = np.sort(
        selected_encoded_report.astype(
            np.int32
        )
    )

    subsection(
        "ATAC: construct selected-bin accessibility matrix"
    )

    x_binary = build_selected_atac_matrix(
        temp_files,
        len(
            primary
        ),
        selected_encoded_matrix,
    )

    print(
        f"ATAC binary matrix: "
        f"{x_binary.shape[0]:,} cells x "
        f"{x_binary.shape[1]:,} selected bins"
    )

    print(
        f"ATAC binary matrix nnz: "
        f"{x_binary.nnz:,}"
    )

    cell_accessible_bins = np.asarray(
        x_binary.sum(
            axis=1
        )
    ).ravel()

    print(
        f"Per-cell selected accessible bins: "
        f"min={int(cell_accessible_bins.min()):,}, "
        f"median={float(np.median(cell_accessible_bins)):.1f}, "
        f"max={int(cell_accessible_bins.max()):,}"
    )

    subsection(
        "ATAC: TF-IDF + 50-dimensional LSI"
    )

    lsi, explained = tfidf_lsi(
        x_binary
    )

    print(
        f"ATAC LSI shape: {lsi.shape}"
    )

    print(
        f"ATAC LSI cumulative explained variance, dims 1-10: "
        f"{explained[:10].sum():.4f}"
    )

    print(
        f"ATAC LSI cumulative explained variance, dims 1-50: "
        f"{explained.sum():.4f}"
    )

    if not keep_temp:
        for path in temp_files:
            path.unlink(
                missing_ok=True
            )

        try:
            TEMP_DIR.rmdir()
        except OSError:
            pass

    return (
        lsi,
        selected_bins,
        per_cell_fragments,
        explained,
        scan_table,
    )


# =====================================================================
# REPRESENTATION QC
# =====================================================================

def design_matrix(
    df,
    categorical_columns,
):
    if not categorical_columns:
        return np.ones(
            (
                len(
                    df
                ),
                1,
            ),
            dtype=float,
        )

    encoded = pd.get_dummies(
        df[
            list(
                categorical_columns
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

    residual = y - x @ beta

    return float(
        np.sum(
            residual ** 2
        )
    )


def multivariate_r2(
    y,
    x,
):
    centered = y - y.mean(
        axis=0,
        keepdims=True,
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
    reduced_sse = multivariate_sse(
        y,
        reduced_x,
    )

    full_sse = multivariate_sse(
        y,
        full_x,
    )

    if reduced_sse <= 0:
        return np.nan

    return float(
        np.clip(
            (
                reduced_sse
                - full_sse
            )
            / reduced_sse,
            0.0,
            1.0,
        )
    )


def same_biology_knn_mixing(
    df,
    matrix,
    k=KNN_K,
):
    scores = []

    groups_used = 0
    cells_used = 0

    group_key = (
        df[
            STAGE_COL
        ].astype(
            str
        )
        + "||"
        + df[
            CELLTYPE_COL
        ].astype(
            str
        )
    )

    positions = np.arange(
        len(
            df
        )
    )

    temp = pd.DataFrame(
        {
            "group": group_key.to_numpy(),
            "position": positions,
        }
    )

    for _, group in temp.groupby(
        "group"
    ):
        pos = group[
            "position"
        ].to_numpy(
            dtype=int
        )

        samples = df.iloc[
            pos
        ][
            SAMPLE_COL
        ].astype(
            str
        ).to_numpy()

        if len(
            np.unique(
                samples
            )
        ) < 2:
            continue

        if len(
            pos
        ) < MIN_GROUP_SIZE_FOR_MIXING:
            continue

        local = matrix[
            pos,
            :
        ].astype(
            float,
            copy=True,
        )

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

        local_k = min(
            k,
            len(
                pos
            )
            - 1,
        )

        tree = cKDTree(
            local
        )

        _, neighbors = tree.query(
            local,
            k=local_k + 1,
        )

        neighbors = neighbors[
            :,
            1:
        ]

        for i in range(
            len(
                pos
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
            pos
        )

    if not scores:
        return (
            np.nan,
            groups_used,
            cells_used,
        )

    return (
        float(
            np.mean(
                scores
            )
        ),
        groups_used,
        cells_used,
    )


def structure_qc(
    primary,
    matrix,
    modality,
    exclude_first_dimension=False,
):
    rows = []

    for n_dims in DIMENSION_SENSITIVITY:
        if exclude_first_dimension:
            start = 1
            stop = 1 + n_dims

        else:
            start = 0
            stop = n_dims

        if stop > matrix.shape[
            1
        ]:
            continue

        y = matrix[
            :,
            start:
            stop,
        ]

        biology_x = design_matrix(
            primary,
            [
                STAGE_COL,
                CELLTYPE_COL,
            ],
        )

        biology_sample_x = design_matrix(
            primary,
            [
                STAGE_COL,
                CELLTYPE_COL,
                SAMPLE_COL,
            ],
        )

        biology_r2 = multivariate_r2(
            y,
            biology_x,
        )

        sample_partial = partial_r2(
            y,
            biology_x,
            biology_sample_x,
        )

        mixing, groups_used, cells_used = same_biology_knn_mixing(
            primary,
            y,
        )

        rows.append(
            {
                "modality": modality,
                "dimension_set": (
                    f"D2-D{n_dims + 1}"
                    if exclude_first_dimension
                    else f"D1-D{n_dims}"
                ),
                "n_dims": n_dims,
                "biology_R2_stage_celltype": biology_r2,
                "sample_partial_R2_given_biology": sample_partial,
                "bio_to_sample_ratio": (
                    biology_r2
                    / sample_partial
                    if (
                        np.isfinite(
                            sample_partial
                        )
                        and sample_partial > 0
                    )
                    else np.inf
                ),
                "same_biology_knn_mixing": mixing,
                "mixing_groups_used": groups_used,
                "mixing_cells_used": cells_used,
            }
        )

    return pd.DataFrame(
        rows
    )


def qc_correlations(
    primary,
    matrix,
    covariates,
    modality,
):
    rows = []

    for covariate in covariates:
        if covariate not in primary.columns:
            continue

        cov = pd.to_numeric(
            primary[
                covariate
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if covariate in {
            RNA_DEPTH_COL,
            ATAC_DEPTH_COL,
        }:
            cov = np.log1p(
                cov
            )

        for j in range(
            matrix.shape[
                1
            ]
        ):
            values = matrix[
                :,
                j,
            ].astype(
                float
            )

            valid = (
                np.isfinite(
                    values
                )
                & np.isfinite(
                    cov
                )
            )

            if int(
                valid.sum()
            ) < 10:
                rho = np.nan

            else:
                rho, _ = spearmanr(
                    values[
                        valid
                    ],
                    cov[
                        valid
                    ],
                )

            rows.append(
                {
                    "modality": modality,
                    "dimension": j + 1,
                    "covariate": covariate,
                    "spearman_rho": (
                        float(
                            rho
                        )
                        if np.isfinite(
                            rho
                        )
                        else np.nan
                    ),
                    "abs_rho": (
                        abs(
                            float(
                                rho
                            )
                        )
                        if np.isfinite(
                            rho
                        )
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# OUTPUT / MANIFEST
# =====================================================================

def save_outputs(
    primary,
    rna_pca,
    atac_lsi,
    hvg_table,
    atac_bin_table,
    rna_explained,
    atac_explained,
    raw_manifest_info,
    gex_presence,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_to_save = primary[
        [
            CELL_COL,
            BARCODE_COL,
            SAMPLE_COL,
            STAGE_COL,
            GENOTYPE_COL,
            CELLTYPE_COL,
            "_sample_barcode",
            "_f8_row",
        ]
    ].copy()

    primary_to_save.to_csv(
        CELLS_FILE,
        sep="\t",
        index=False,
        compression="gzip",
    )

    hvg_table.to_csv(
        RNA_HVG_FILE,
        sep="\t",
        index=False,
        compression="gzip",
    )

    atac_bin_table.to_csv(
        ATAC_BIN_FILE,
        sep="\t",
        index=False,
        compression="gzip",
    )

    np.save(
        RNA_PCA_FILE,
        rna_pca,
    )

    np.save(
        ATAC_LSI_FILE,
        atac_lsi,
    )

    manifest = {
        "dataset_accession": ACCESSION,
        "pipeline_phase": "F8",
        "created_utc": utc_now_iso(),
        "source_policy": (
            "GEO-native reconstruction from checksum-verified GSE205117_RAW.tar; "
            "no FTP-dependent author processed representation was used."
        ),
        "raw_archive": {
            "path": str(
                RAW_TAR
            ),
            **raw_manifest_info,
        },
        "frozen_primary_lineage": {
            "genotype": PRIMARY_GENOTYPE,
            "celltypes": PRIMARY_LINEAGE_STATES,
            "requires_pass_rnaQC": True,
            "requires_pass_atacQC": True,
            "requires_GEX_presence": True,
            "ATAC_presence_prior_phase": (
                "F7b showed 100% mapped+joint-QC target presence in ATAC fragments."
            ),
            "n_cells": len(
                primary
            ),
        },
        "rna_representation": {
            "method": (
                "library normalization -> log1p -> top-variance HVG -> "
                "standardized randomized PCA"
            ),
            "target_library_size": RNA_TARGET_LIBRARY_SIZE,
            "hvg_count_target": RNA_HVG_COUNT,
            "pca_dimensions": RNA_PCA_DIMS,
            "batch_correction": None,
            "explained_variance_ratio": [
                float(
                    value
                )
                for value in rna_explained
            ],
            "file": str(
                RNA_PCA_FILE
            ),
            "sha256": sha256_file(
                RNA_PCA_FILE
            ),
        },
        "atac_representation": {
            "method": (
                "10-kb fixed genomic bins -> top 25,000 accessible bins -> "
                "binary accessibility -> TF-IDF -> randomized TruncatedSVD/LSI"
            ),
            "bin_size_bp": ATAC_BIN_SIZE,
            "selected_bin_count_target": ATAC_FEATURE_COUNT,
            "lsi_dimensions": ATAC_LSI_DIMS,
            "batch_correction": None,
            "explained_variance_ratio": [
                float(
                    value
                )
                for value in atac_explained
            ],
            "file": str(
                ATAC_LSI_FILE
            ),
            "sha256": sha256_file(
                ATAC_LSI_FILE
            ),
        },
        "cell_file": {
            "path": str(
                CELLS_FILE
            ),
            "sha256": sha256_file(
                CELLS_FILE
            ),
        },
        "rna_hvg_file": {
            "path": str(
                RNA_HVG_FILE
            ),
            "sha256": sha256_file(
                RNA_HVG_FILE
            ),
        },
        "atac_bin_file": {
            "path": str(
                ATAC_BIN_FILE
            ),
            "sha256": sha256_file(
                ATAC_BIN_FILE
            ),
        },
        "gex_primary_coverage_by_sample": (
            gex_presence
            .reset_index()
            .to_dict(
                orient="records"
            )
        ),
        "analysis_guardrails": {
            "pseudotime_calculated": False,
            "gdis_calculated": False,
            "cmil_calculated": False,
            "discovery_dataset_modified": False,
        },
    }

    with F8_MANIFEST.open(
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


# =====================================================================
# MAIN
# =====================================================================

def main():
    args = parse_args()

    section(
        "PHASE F8 — GSE205117 GEO-NATIVE MODALITY-SPECIFIC REPRESENTATION RECONSTRUCTION + QC"
    )

    print(
        "Network policy:"
    )

    print(
        "  No FTP access is used."
    )

    print(
        "  Representations are reconstructed from the checksum-verified GEO raw archive."
    )

    print()

    print(
        "RNA:"
    )

    print(
        "  raw GEX -> normalization -> log1p -> 2500 variable genes -> 50D PCA"
    )

    print()

    print(
        "ATAC:"
    )

    print(
        "  raw fragments -> 10-kb bins -> top 25000 bins -> TF-IDF -> 50D LSI"
    )

    print()

    print(
        "No joint RNA+ATAC embedding."
    )

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
    # Inputs.
    # -----------------------------------------------------------------

    section(
        "1. VERIFY FROZEN GEO INPUTS"
    )

    require_inputs()

    raw_manifest_info = verify_raw_size_against_manifest()

    print(
        f"Raw archive: {RAW_TAR.resolve()}"
    )

    print(
        f"Raw size:    {human_size(RAW_TAR.stat().st_size)}"
    )

    print(
        f"Recorded SHA-256: "
        f"{raw_manifest_info['recorded_sha256']}"
    )

    print(
        "Raw size matches acquisition manifest: True"
    )

    print()

    print(
        "Note: F7 already recomputed the full 32.5-GB SHA-256. "
        "F8 does not hash the raw TAR again."
    )

    # -----------------------------------------------------------------
    # Metadata / archive.
    # -----------------------------------------------------------------

    section(
        "2. FREEZE PRIMARY WT LINEAGE AND GEX INTERSECTION"
    )

    metadata = load_metadata()

    inventory = inventory_tar()

    (
        primary,
        gex_presence,
    ) = freeze_primary_gex_cells(
        metadata,
        inventory,
    )

    print(
        f"Metadata primary WT joint-QC candidates: "
        f"{int(metadata['_primary_candidate'].sum()):,}"
    )

    print(
        f"Frozen primary cells after GEX presence intersection: "
        f"{len(primary):,}"
    )

    subsection(
        "Primary GEX coverage by sample"
    )

    print_df(
        gex_presence,
        digits=4,
    )

    subsection(
        "Frozen primary state counts"
    )

    state_counts = (
        primary[
            CELLTYPE_COL
        ]
        .value_counts()
        .reindex(
            PRIMARY_LINEAGE_STATES,
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
        "Frozen primary state × stage"
    )

    print_df(
        pd.crosstab(
            primary[
                CELLTYPE_COL
            ],
            primary[
                STAGE_COL
            ],
            margins=True,
        )
    )

    subsection(
        "Frozen primary stage × sample"
    )

    print_df(
        pd.crosstab(
            primary[
                STAGE_COL
            ],
            primary[
                SAMPLE_COL
            ],
            margins=True,
        )
    )

    candidate_count = int(
        metadata[
            "_primary_candidate"
        ].sum()
    )

    primary_coverage = (
        len(
            primary
        )
        / candidate_count
        if candidate_count
        else np.nan
    )

    print()

    print(
        f"Primary representation coverage after GEX intersection: "
        f"{primary_coverage:.6f}"
    )

    # -----------------------------------------------------------------
    # Check minimum lineage support before expensive reconstruction.
    # -----------------------------------------------------------------

    if len(
        primary
    ) < MIN_PRIMARY_TOTAL:
        raise RuntimeError(
            "Too few primary WT lineage cells for F8."
        )

    if int(
        state_counts[
            "n_cells"
        ].min()
    ) < MIN_STATE_CELLS:
        raise RuntimeError(
            "At least one primary lineage state has too few cells."
        )

    # -----------------------------------------------------------------
    # Reconstruct representations.
    # -----------------------------------------------------------------

    section(
        "3. RECONSTRUCT RNA PCA FROM GEO GEX"
    )

    (
        rna_pca,
        hvg_table,
        rna_library_size,
        rna_explained,
    ) = reconstruct_rna_pca(
        primary,
        inventory,
    )

    section(
        "4. RECONSTRUCT ATAC LSI FROM GEO FRAGMENTS"
    )

    (
        atac_lsi,
        atac_bin_table,
        atac_selected_fragment_count,
        atac_explained,
        atac_scan_table,
    ) = reconstruct_atac_lsi(
        primary,
        inventory,
        keep_temp=args.keep_temp,
    )

    # -----------------------------------------------------------------
    # Integrity.
    # -----------------------------------------------------------------

    section(
        "5. REPRESENTATION INTEGRITY"
    )

    print(
        f"RNA PCA shape:  {rna_pca.shape}"
    )

    print(
        f"ATAC LSI shape: {atac_lsi.shape}"
    )

    rna_finite = bool(
        np.isfinite(
            rna_pca
        ).all()
    )

    atac_finite = bool(
        np.isfinite(
            atac_lsi
        ).all()
    )

    print(
        f"RNA PCA all finite:  {rna_finite}"
    )

    print(
        f"ATAC LSI all finite: {atac_finite}"
    )

    same_rows = (
        rna_pca.shape[
            0
        ]
        == atac_lsi.shape[
            0
        ]
        == len(
            primary
        )
    )

    print(
        f"Identical RNA/ATAC cell rows: {same_rows}"
    )

    # -----------------------------------------------------------------
    # Depth correlations.
    # -----------------------------------------------------------------

    section(
        "6. LATENT DIMENSION ↔ DEPTH/QC CORRELATIONS"
    )

    rna_corr = qc_correlations(
        primary,
        rna_pca,
        [
            RNA_DEPTH_COL,
            RNA_FEATURE_COL,
        ],
        "RNA_PCA",
    )

    atac_corr = qc_correlations(
        primary,
        atac_lsi,
        [
            ATAC_DEPTH_COL,
            ATAC_TSS_COL,
        ],
        "ATAC_LSI",
    )

    subsection(
        "RNA PCA: strongest absolute QC correlations"
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
        "ATAC LSI: strongest absolute QC correlations"
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

    # -----------------------------------------------------------------
    # Structure QC.
    # -----------------------------------------------------------------

    section(
        "7. DIMENSION-SENSITIVITY REPRESENTATION STRUCTURE QC"
    )

    rna_structure = structure_qc(
        primary,
        rna_pca,
        "RNA_PCA",
        exclude_first_dimension=False,
    )

    atac_structure_with_d1 = structure_qc(
        primary,
        atac_lsi,
        "ATAC_LSI_D1_INCLUDED",
        exclude_first_dimension=False,
    )

    # Prespecified ATAC sensitivity because LSI1 is often depth-associated.
    # This is diagnostic only; it does not automatically authorize dropping D1.
    atac_structure_without_d1 = structure_qc(
        primary,
        atac_lsi,
        "ATAC_LSI_D1_EXCLUDED",
        exclude_first_dimension=True,
    )

    subsection(
        "RNA PCA"
    )

    print_df(
        rna_structure.set_index(
            "dimension_set"
        ),
        digits=4,
    )

    subsection(
        "ATAC LSI — D1 included"
    )

    print_df(
        atac_structure_with_d1.set_index(
            "dimension_set"
        ),
        digits=4,
    )

    subsection(
        "ATAC LSI — prespecified D1-excluded sensitivity"
    )

    print_df(
        atac_structure_without_d1.set_index(
            "dimension_set"
        ),
        digits=4,
    )

    # -----------------------------------------------------------------
    # Save reproducibility artifacts.
    # -----------------------------------------------------------------

    section(
        "8. SAVE F8 REPRODUCIBILITY ARTIFACTS"
    )

    save_outputs(
        primary,
        rna_pca,
        atac_lsi,
        hvg_table,
        atac_bin_table,
        rna_explained,
        atac_explained,
        raw_manifest_info,
        gex_presence,
    )

    for path in [
        CELLS_FILE,
        RNA_HVG_FILE,
        ATAC_BIN_FILE,
        RNA_PCA_FILE,
        ATAC_LSI_FILE,
        F8_MANIFEST,
    ]:
        print(
            f"{path}  [{human_size(path.stat().st_size)}]"
        )

    # -----------------------------------------------------------------
    # Structural decision.
    # -----------------------------------------------------------------

    section(
        "9. PHASE F8 STRUCTURAL CHECKS"
    )

    minimum_state_cells = int(
        state_counts[
            "n_cells"
        ].min()
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    "Primary WT joint-QC cells represented after GEX "
                    "intersection >=95%"
                ),
                "pass": (
                    primary_coverage
                    >= MIN_REPRESENTATION_COVERAGE
                ),
            },
            {
                "criterion": (
                    f"Primary paired lineage has >={MIN_PRIMARY_TOTAL} cells"
                ),
                "pass": (
                    len(
                        primary
                    )
                    >= MIN_PRIMARY_TOTAL
                ),
            },
            {
                "criterion": (
                    f"Each primary lineage state has >={MIN_STATE_CELLS} cells"
                ),
                "pass": (
                    minimum_state_cells
                    >= MIN_STATE_CELLS
                ),
            },
            {
                "criterion": "RNA PCA has 50 dimensions",
                "pass": (
                    rna_pca.shape[
                        1
                    ]
                    == RNA_PCA_DIMS
                ),
            },
            {
                "criterion": "ATAC LSI has 50 dimensions",
                "pass": (
                    atac_lsi.shape[
                        1
                    ]
                    == ATAC_LSI_DIMS
                ),
            },
            {
                "criterion": "RNA PCA finite",
                "pass": rna_finite,
            },
            {
                "criterion": "ATAC LSI finite",
                "pass": atac_finite,
            },
            {
                "criterion": "RNA/ATAC use identical ordered cells",
                "pass": same_rows,
            },
            {
                "criterion": "Every F8 cell has selected ATAC fragments",
                "pass": bool(
                    np.all(
                        atac_selected_fragment_count
                        > 0
                    )
                ),
            },
            {
                "criterion": "No malformed ATAC fragment records",
                "pass": (
                    int(
                        atac_scan_table[
                            "malformed_records"
                        ].sum()
                    )
                    == 0
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

    structural_go = bool(
        checks[
            "pass"
        ].all()
    )

    # -----------------------------------------------------------------
    # Final interpretation.
    # -----------------------------------------------------------------

    section(
        "10. PHASE F8 DECISION"
    )

    if structural_go:
        print(
            "PHASE F8 VERDICT: GO — GEO-NATIVE MODALITY-SPECIFIC "
            "REPRESENTATIONS RECONSTRUCTED"
        )

        print()

        print(
            "RNA PCA and ATAC LSI were reconstructed from the durable GEO "
            "raw archive using the same frozen primary WT cells."
        )

        print()

        print(
            "Do NOT construct pseudotime yet."
        )

        print(
            "First review the dimension-sensitivity and depth-correlation "
            "tables above and freeze the external representation dimensions."
        )

    else:
        print(
            "PHASE F8 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "At least one structural representation criterion failed. "
            "Do not proceed to external pseudotime."
        )

    print()

    print(
        "FROZEN EXTERNAL PRIMARY BIOLOGICAL LINEAGE:"
    )

    print(
        "  WT: NMP -> Paraxial_mesoderm -> Somitic_mesoderm"
    )

    print()

    print(
        "T_KO remains a separate perturbational arm."
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

