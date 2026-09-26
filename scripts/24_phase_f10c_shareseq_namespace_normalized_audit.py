#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
24_phase_f10c_shareseq_namespace_normalized_audit.py
=====================================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F10c
----------
Final namespace-normalized structural audit.

WHY F10c IS REQUIRED
---------------------
F10b correctly identified the physical file formats but still compared
barcodes before applying the namespace conversions used by published
SHARE-seq preprocessing code.

Published preprocessing for this dataset performs:

RNA matrix barcode normalization:
    rna.obs_names = rna.obs_names.str.replace(",", ".")

ATAC raw-fragment barcode normalization:
    annotation atac.bc is stored as eight dot-separated parts:
        A.B.C.D.E.F.G.H

    raw fragments use paired components:
        A.B,C.D,E.F,G.H

Thus the correct comparisons are:

1. RNA:
       normalize RNA header barcode "," -> "."
       then compare to annotation rna.bc

2. Processed ATAC MatrixMarket:
       separate barcode file is already in annotation rna.bc namespace

3. Raw ATAC fragments:
       transform annotation atac.bc from 8-part dot form into fragment form
       before comparison with fragment column 4

NO DATA RE-DOWNLOAD
-------------------
This phase uses the six files already downloaded in F10.

WHAT F10c DOES
--------------
1. Parses RNA as TAB-delimited.
2. Normalizes RNA cell names by comma -> dot.
3. Confirms all 34,774 paired annotation rna.bc cells occur in RNA.
4. Confirms ATAC MatrixMarket has 34,774 columns.
5. Confirms the ATAC barcode file exactly equals annotation rna.bc.
6. Confirms RNA paired subset and processed ATAC use the same 34,774-cell
   matched RNA-barcode universe.
7. Reproduces the published atac.bc -> fragment barcode conversion.
8. Confirms transformed atac.bc values are unique.
9. Scans the first 1,000,000 fragments against transformed annotation atac.bc.
10. Optionally scans the complete fragment file:
       --full-fragment-scan
11. Freezes the exact paired 34,774-cell mapping table for later phases.

OUTPUTS
-------
data/GSE140203/
    f10c_paired_cell_mapping.tsv.gz
    f10c_shareseq_namespace_normalized_audit.json

NO REPRESENTATION
-----------------
No PCA.
No LSI.
No pseudotime.
No GDIS.
No CMIL.
No lineage frozen.

RUN
---
    python 24_phase_f10c_shareseq_namespace_normalized_audit.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE140203"

DATA_DIR = Path("data") / ACCESSION

CELLTYPE_FILE = DATA_DIR / "GSM4156597_skin_celltype.txt.gz"
ATAC_COUNTS_FILE = DATA_DIR / "GSM4156597_skin.late.anagen.counts.txt.gz"
ATAC_BARCODES_FILE = DATA_DIR / "GSM4156597_skin.late.anagen.barcodes.txt.gz"
ATAC_PEAKS_FILE = DATA_DIR / "GSM4156597_skin.late.anagen.peaks.bed.gz"
ATAC_FRAGMENTS_FILE = DATA_DIR / "GSM4156597_skin.late.anagen.atac.fragments.bed.gz"
RNA_COUNTS_FILE = DATA_DIR / "GSM4156608_skin.late.anagen.rna.counts.txt.gz"

OUTPUT_MAPPING = DATA_DIR / "f10c_paired_cell_mapping.tsv.gz"
OUTPUT_MANIFEST = DATA_DIR / "f10c_shareseq_namespace_normalized_audit.json"

EXPECTED_PAIRED_CELLS = 34_774
EXPECTED_ATAC_PEAKS = 344_592

BOUNDED_FRAGMENT_RECORDS = 1_000_000

HAIR_LINEAGE_PATTERN = re.compile(
    r"(?:TAC|IRS|MEDULLA|CUTICLE|CORTEX|HAIR[\s_\-]*SHAFT)",
    flags=re.IGNORECASE,
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
        "display.max_rows", 300,
        "display.max_columns", None,
        "display.width", 420,
        "display.max_colwidth", 140,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string())


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
            "Namespace-normalized structural audit for "
            "GSE140203 SHARE-seq mouse skin."
        )
    )

    parser.add_argument(
        "--full-fragment-scan",
        action="store_true",
        help=(
            "Scan the complete raw fragment file and verify presence of "
            "all transformed paired ATAC barcodes."
        ),
    )

    return parser.parse_args()


# =====================================================================
# INPUTS
# =====================================================================

def require_inputs():
    required = [
        CELLTYPE_FILE,
        ATAC_COUNTS_FILE,
        ATAC_BARCODES_FILE,
        ATAC_PEAKS_FILE,
        ATAC_FRAGMENTS_FILE,
        RNA_COUNTS_FILE,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required file(s):\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )


# =====================================================================
# BARCODE NORMALIZATION
# =====================================================================

def normalize_rna_matrix_barcode(
    barcode,
):
    """
    Published SHARE-seq preprocessing converts RNA matrix barcodes from:
        R1.01,R2.01,R3.06,P1.55
    to:
        R1.01.R2.01.R3.06.P1.55
    """
    return str(
        barcode
    ).strip().replace(
        ",",
        ".",
    )


def annotation_atac_to_fragment_barcode(
    barcode,
):
    """
    Reproduce the published preprocessing conversion.

    Annotation atac.bc:
        part0.part1.part2.part3.part4.part5.part6.part7

    Raw fragment barcode:
        part0.part1,part2.part3,part4.part5,part6.part7
    """
    parts = str(
        barcode
    ).strip().split(
        "."
    )

    if len(
        parts
    ) != 8:
        return None

    return (
        f"{parts[0]}.{parts[1]},"
        f"{parts[2]}.{parts[3]},"
        f"{parts[4]}.{parts[5]},"
        f"{parts[6]}.{parts[7]}"
    )


# =====================================================================
# FILE AUDITS
# =====================================================================

def read_rna_header():
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
            "RNA file failed TAB-delimited header parsing."
        )

    raw_barcodes = [
        value.strip()
        for value in fields[
            1:
        ]
    ]

    normalized = [
        normalize_rna_matrix_barcode(
            value
        )
        for value in raw_barcodes
    ]

    return (
        fields[
            0
        ],
        raw_barcodes,
        normalized,
    )


def matrixmarket_dimensions():
    with gzip.open(
        ATAC_COUNTS_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        banner = handle.readline().strip()

        if not banner.startswith(
            "%%MatrixMarket"
        ):
            raise RuntimeError(
                "ATAC count file is not MatrixMarket."
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
        "Could not locate MatrixMarket dimensions."
    )


def read_single_column(
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

            if value:
                values.append(
                    value.split(
                        "\t"
                    )[
                        0
                    ]
                )

    return values


def count_nonempty_lines(
    path,
):
    count = 0

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line_text in handle:
            if line_text.strip():
                count += 1

    return count


def load_annotation():
    annotation = pd.read_csv(
        CELLTYPE_FILE,
        sep="\t",
        compression="gzip",
        dtype=str,
    )

    required = [
        "atac.bc",
        "rna.bc",
        "celltype",
    ]

    missing = [
        column
        for column in required
        if column not in annotation.columns
    ]

    if missing:
        raise RuntimeError(
            "Annotation missing required columns: "
            + ", ".join(
                missing
            )
        )

    for column in required:
        annotation[
            column
        ] = (
            annotation[
                column
            ]
            .astype(str)
            .str.strip()
        )

    annotation[
        "fragment.bc"
    ] = annotation[
        "atac.bc"
    ].map(
        annotation_atac_to_fragment_barcode
    )

    return annotation


# =====================================================================
# FRAGMENT AUDIT
# =====================================================================

def bounded_fragment_scan(
    target_fragment_barcodes,
):
    records = 0
    malformed = 0

    unique_fragment_barcodes = set()
    hits = set()
    chromosomes = Counter()

    with gzip.open(
        ATAC_FRAGMENTS_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line_text in handle:
            if not line_text.strip():
                continue

            if line_text.startswith(
                "#"
            ):
                continue

            fields = line_text.rstrip(
                "\n\r"
            ).split(
                "\t"
            )

            if len(
                fields
            ) < 4:
                fields = re.split(
                    r"\s+",
                    line_text.strip(),
                )

            if len(
                fields
            ) < 4:
                malformed += 1
                continue

            records += 1

            chrom = fields[
                0
            ]

            barcode = fields[
                3
            ]

            chromosomes[
                chrom
            ] += 1

            unique_fragment_barcodes.add(
                barcode
            )

            if barcode in target_fragment_barcodes:
                hits.add(
                    barcode
                )

            if records >= BOUNDED_FRAGMENT_RECORDS:
                break

    return {
        "records_scanned": records,
        "malformed_records": malformed,
        "unique_fragment_barcodes_seen": len(
            unique_fragment_barcodes
        ),
        "paired_fragment_barcodes_seen": len(
            hits
        ),
        "paired_fragment_fraction_seen": (
            len(
                hits
            )
            / len(
                target_fragment_barcodes
            )
            if target_fragment_barcodes
            else np.nan
        ),
        "top_chromosomes": dict(
            chromosomes.most_common(
                25
            )
        ),
    }


def full_fragment_scan(
    target_fragment_barcodes,
):
    found = set()

    records = 0
    malformed = 0

    with gzip.open(
        ATAC_FRAGMENTS_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line_text in handle:
            if not line_text.strip():
                continue

            if line_text.startswith(
                "#"
            ):
                continue

            fields = line_text.rstrip(
                "\n\r"
            ).split(
                "\t"
            )

            if len(
                fields
            ) < 4:
                fields = re.split(
                    r"\s+",
                    line_text.strip(),
                )

            if len(
                fields
            ) < 4:
                malformed += 1
                continue

            records += 1

            barcode = fields[
                3
            ]

            if barcode in target_fragment_barcodes:
                found.add(
                    barcode
                )

                if len(
                    found
                ) == len(
                    target_fragment_barcodes
                ):
                    break

    return {
        "records_scanned": records,
        "malformed_records": malformed,
        "target_n": len(
            target_fragment_barcodes
        ),
        "found_n": len(
            found
        ),
        "coverage": (
            len(
                found
            )
            / len(
                target_fragment_barcodes
            )
            if target_fragment_barcodes
            else np.nan
        ),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    args = parse_args()

    section(
        "PHASE F10c — GSE140203 SHARE-seq NAMESPACE-NORMALIZED PAIRED-CELL AUDIT"
    )

    print(
        "Published barcode transformations are reproduced explicitly."
    )

    print()

    print(
        "No representation."
    )

    print(
        "No lineage freeze."
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

    require_inputs()

    # -----------------------------------------------------------------
    # Annotation.
    # -----------------------------------------------------------------

    section(
        "1. LOAD PAIRED ANNOTATION AND CONSTRUCT PUBLISHED BARCODE NAMESPACES"
    )

    annotation = load_annotation()

    annotation_rna = set(
        annotation[
            "rna.bc"
        ]
    )

    annotation_atac = set(
        annotation[
            "atac.bc"
        ]
    )

    invalid_fragment_conversion = int(
        annotation[
            "fragment.bc"
        ].isna().sum()
    )

    valid_fragment_barcodes = set(
        annotation[
            "fragment.bc"
        ].dropna()
    )

    print(
        f"Annotation rows: {len(annotation):,}"
    )

    print(
        f"Unique rna.bc: {len(annotation_rna):,}"
    )

    print(
        f"Unique atac.bc: {len(annotation_atac):,}"
    )

    print(
        f"Invalid atac.bc -> fragment conversions: "
        f"{invalid_fragment_conversion:,}"
    )

    print(
        f"Unique transformed fragment barcodes: "
        f"{len(valid_fragment_barcodes):,}"
    )

    print()

    print(
        "Example barcode transformation:"
    )

    example = annotation.iloc[
        0
    ]

    print(
        f"  annotation atac.bc : {example['atac.bc']}"
    )

    print(
        f"  fragment barcode   : {example['fragment.bc']}"
    )

    print(
        f"  matched rna.bc     : {example['rna.bc']}"
    )

    # -----------------------------------------------------------------
    # RNA normalization.
    # -----------------------------------------------------------------

    section(
        "2. RNA BARCODE NORMALIZATION AND PAIRED-SUBSET AUDIT"
    )

    (
        first_field,
        rna_raw_barcodes,
        rna_normalized_barcodes,
    ) = read_rna_header()

    rna_raw_set = set(
        rna_raw_barcodes
    )

    rna_normalized_set = set(
        rna_normalized_barcodes
    )

    paired_rna_overlap = (
        rna_normalized_set
        & annotation_rna
    )

    print(
        f"RNA first field: {first_field}"
    )

    print(
        f"RNA total columns: {len(rna_raw_barcodes):,}"
    )

    print(
        f"RNA raw unique columns: {len(rna_raw_set):,}"
    )

    print(
        f"RNA normalized unique columns: "
        f"{len(rna_normalized_set):,}"
    )

    print(
        f"Annotation paired rna.bc represented in RNA: "
        f"{len(paired_rna_overlap):,} / {len(annotation_rna):,}"
    )

    print(
        f"All paired annotation rna.bc are present in RNA: "
        f"{annotation_rna.issubset(rna_normalized_set)}"
    )

    print(
        f"RNA cells outside paired annotation: "
        f"{len(rna_normalized_set - annotation_rna):,}"
    )

    # -----------------------------------------------------------------
    # ATAC processed matrix.
    # -----------------------------------------------------------------

    section(
        "3. PROCESSED ATAC MATRIX / MATCHED-RNA BARCODE AUDIT"
    )

    (
        mm_banner,
        mm_rows,
        mm_cols,
        mm_nnz,
    ) = matrixmarket_dimensions()

    atac_barcodes = read_single_column(
        ATAC_BARCODES_FILE
    )

    atac_barcode_set = set(
        atac_barcodes
    )

    peak_count = count_nonempty_lines(
        ATAC_PEAKS_FILE
    )

    print(
        f"MatrixMarket banner: {mm_banner}"
    )

    print(
        f"ATAC rows: {mm_rows:,}"
    )

    print(
        f"ATAC columns: {mm_cols:,}"
    )

    print(
        f"ATAC nnz: {mm_nnz:,}"
    )

    print(
        f"ATAC barcode rows: {len(atac_barcodes):,}"
    )

    print(
        f"ATAC barcode unique: {len(atac_barcode_set):,}"
    )

    print(
        f"ATAC peak rows: {peak_count:,}"
    )

    print()

    print(
        f"ATAC barcode file exactly equals annotation rna.bc: "
        f"{atac_barcode_set == annotation_rna}"
    )

    print(
        f"Paired RNA subset exactly equals ATAC matched cell universe: "
        f"{paired_rna_overlap == atac_barcode_set}"
    )

    # -----------------------------------------------------------------
    # Cell labels.
    # -----------------------------------------------------------------

    section(
        "4. PAIRED CELL-TYPE INVENTORY"
    )

    celltype_counts = (
        annotation[
            "celltype"
        ]
        .value_counts()
        .rename(
            "n_cells"
        )
        .to_frame()
    )

    print_df(
        celltype_counts,
        digits=0,
    )

    hair_mask = annotation[
        "celltype"
    ].str.contains(
        HAIR_LINEAGE_PATTERN,
        regex=True,
        na=False,
    )

    hair_counts = (
        annotation.loc[
            hair_mask,
            "celltype",
        ]
        .value_counts()
        .rename(
            "n_cells"
        )
        .to_frame()
    )

    subsection(
        "Candidate published hair-follicle branch states"
    )

    print_df(
        hair_counts,
        digits=0,
    )

    hair_total = int(
        hair_mask.sum()
    )

    print()

    print(
        f"Candidate hair-lineage paired cells: {hair_total:,}"
    )

    # -----------------------------------------------------------------
    # Raw fragments after exact published conversion.
    # -----------------------------------------------------------------

    section(
        "5. RAW ATAC FRAGMENTS AFTER PUBLISHED atac.bc CONVERSION"
    )

    bounded = bounded_fragment_scan(
        valid_fragment_barcodes
    )

    print(
        f"Fragment records scanned: "
        f"{bounded['records_scanned']:,}"
    )

    print(
        f"Malformed records: "
        f"{bounded['malformed_records']:,}"
    )

    print(
        f"Unique raw fragment barcodes seen: "
        f"{bounded['unique_fragment_barcodes_seen']:,}"
    )

    print(
        f"Paired transformed fragment barcodes seen: "
        f"{bounded['paired_fragment_barcodes_seen']:,}"
    )

    print(
        f"Fraction of paired transformed barcodes seen: "
        f"{bounded['paired_fragment_fraction_seen']:.6f}"
    )

    full = None

    if args.full_fragment_scan:
        subsection(
            "FULL TARGETED FRAGMENT PRESENCE SCAN"
        )

        full = full_fragment_scan(
            valid_fragment_barcodes
        )

        print(
            f"Fragment records scanned: {full['records_scanned']:,}"
        )

        print(
            f"Malformed records: {full['malformed_records']:,}"
        )

        print(
            f"Target paired barcodes: {full['target_n']:,}"
        )

        print(
            f"Found paired barcodes: {full['found_n']:,}"
        )

        print(
            f"Coverage: {full['coverage']:.6f}"
        )

    # -----------------------------------------------------------------
    # Structural checks.
    # -----------------------------------------------------------------

    section(
        "6. PHASE F10c STRUCTURAL CHECKS"
    )

    one_to_one = (
        annotation[
            "atac.bc"
        ].nunique()
        == len(
            annotation
        )
        and annotation[
            "rna.bc"
        ].nunique()
        == len(
            annotation
        )
        and annotation[
            "fragment.bc"
        ].nunique()
        == len(
            annotation
        )
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    f"Annotation contains exactly "
                    f"{EXPECTED_PAIRED_CELLS:,} paired cells"
                ),
                "pass": (
                    len(
                        annotation
                    )
                    == EXPECTED_PAIRED_CELLS
                ),
            },
            {
                "criterion": (
                    "RNA/ATAC/fragment barcode mapping is one-to-one"
                ),
                "pass": bool(
                    one_to_one
                    and invalid_fragment_conversion
                    == 0
                ),
            },
            {
                "criterion": (
                    "All 34,774 paired rna.bc occur in normalized RNA matrix"
                ),
                "pass": (
                    len(
                        paired_rna_overlap
                    )
                    == EXPECTED_PAIRED_CELLS
                ),
            },
            {
                "criterion": (
                    "ATAC MatrixMarket has 34,774 columns"
                ),
                "pass": (
                    mm_cols
                    == EXPECTED_PAIRED_CELLS
                ),
            },
            {
                "criterion": (
                    "ATAC barcode rows equal MatrixMarket columns"
                ),
                "pass": (
                    len(
                        atac_barcodes
                    )
                    == mm_cols
                ),
            },
            {
                "criterion": (
                    "ATAC barcode file exactly equals annotation rna.bc"
                ),
                "pass": (
                    atac_barcode_set
                    == annotation_rna
                ),
            },
            {
                "criterion": (
                    "Paired normalized RNA subset equals processed ATAC "
                    "cell universe"
                ),
                "pass": (
                    paired_rna_overlap
                    == atac_barcode_set
                    and len(
                        paired_rna_overlap
                    )
                    == EXPECTED_PAIRED_CELLS
                ),
            },
            {
                "criterion": (
                    "ATAC MatrixMarket rows equal peak BED rows"
                ),
                "pass": (
                    mm_rows
                    == peak_count
                    == EXPECTED_ATAC_PEAKS
                ),
            },
            {
                "criterion": (
                    "Bounded raw-fragment scan has zero malformed records"
                ),
                "pass": (
                    bounded[
                        "malformed_records"
                    ]
                    == 0
                ),
            },
            {
                "criterion": (
                    "Bounded raw-fragment scan overlaps transformed paired "
                    "atac.bc namespace"
                ),
                "pass": (
                    bounded[
                        "paired_fragment_barcodes_seen"
                    ]
                    > 0
                ),
            },
            {
                "criterion": (
                    "Candidate TAC/hair lineage contains >=5,000 paired cells"
                ),
                "pass": (
                    hair_total
                    >= 5000
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
    # Save exact paired mapping.
    # -----------------------------------------------------------------

    section(
        "7. SAVE EXACT PAIRED CELL MAPPING"
    )

    mapping = annotation[
        [
            "rna.bc",
            "atac.bc",
            "fragment.bc",
            "celltype",
        ]
    ].copy()

    mapping[
        "_paired_row"
    ] = np.arange(
        len(
            mapping
        ),
        dtype=int,
    )

    mapping.to_csv(
        OUTPUT_MAPPING,
        sep="\t",
        index=False,
        compression="gzip",
    )

    print(
        OUTPUT_MAPPING
    )

    # -----------------------------------------------------------------
    # Manifest.
    # -----------------------------------------------------------------

    section(
        "8. SAVE F10c AUDIT MANIFEST"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F10c",
        "created_utc": utc_now_iso(),
        "published_namespace_rules": {
            "rna_matrix_to_annotation_rna_bc": (
                "replace comma with dot"
            ),
            "annotation_atac_bc_to_raw_fragment_bc": (
                "8 dot-separated parts -> four comma-separated paired "
                "components"
            ),
            "processed_atac_barcode_namespace": (
                "annotation rna.bc"
            ),
        },
        "paired_cells": {
            "annotation_rows": len(
                annotation
            ),
            "paired_rna_overlap": len(
                paired_rna_overlap
            ),
            "processed_atac_cells": len(
                atac_barcode_set
            ),
            "mapping_one_to_one": bool(
                one_to_one
            ),
        },
        "rna": {
            "total_raw_columns": len(
                rna_raw_barcodes
            ),
            "normalized_unique_columns": len(
                rna_normalized_set
            ),
            "paired_cells_present": len(
                paired_rna_overlap
            ),
            "unpaired_rna_cells": len(
                rna_normalized_set
                - annotation_rna
            ),
        },
        "atac": {
            "matrix_rows": mm_rows,
            "matrix_columns": mm_cols,
            "matrix_nnz": mm_nnz,
            "peak_rows": peak_count,
            "barcode_rows": len(
                atac_barcodes
            ),
        },
        "hair_candidate_cells": hair_total,
        "bounded_fragment_scan": bounded,
        "full_fragment_scan": full,
        "checks": checks.to_dict(
            orient="records"
        ),
        "outputs": {
            "paired_mapping": str(
                OUTPUT_MAPPING
            ),
        },
        "guardrails": {
            "representation_constructed": False,
            "lineage_frozen": False,
            "pseudotime_calculated": False,
            "gdis_calculated": False,
            "cmil_calculated": False,
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
        OUTPUT_MANIFEST
    )

    # -----------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------

    section(
        "9. PHASE F10c DECISION"
    )

    if all_pass:
        print(
            "PHASE F10c VERDICT: GO — EXACT 34,774-CELL SHARE-seq "
            "RNA/ATAC PAIRING VERIFIED"
        )

        print()

        print(
            "The paired RNA subset and processed ATAC matrix use the same "
            "matched RNA-barcode universe."
        )

        print()

        print(
            "The original ATAC barcode namespace is also correctly linked "
            "to the raw fragment barcode namespace."
        )

        print()

        print(
            "Next phase may construct modality-specific representations and "
            "screen TAC-derived trajectory branches for continuity."
        )

    else:
        print(
            "PHASE F10c VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not construct representations until any failed namespace "
            "criterion is understood."
        )

    print()

    print(
        "No representation was constructed."
    )

    print(
        "No lineage was frozen."
    )

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

