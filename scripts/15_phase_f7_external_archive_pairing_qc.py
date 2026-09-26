#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
15_phase_f7_external_archive_pairing_qc.py
===========================================

Project:
GDIS-Multiome external validation

Dataset:
GSE205117

PHASE F7
--------
Reproducible archive-structure and RNA/ATAC pairing audit.

This phase is deliberately BEFORE representation selection and BEFORE
trajectory construction.

GOALS
-----
1. Verify that the downloaded GSE205117_RAW.tar is the same file recorded in
   the Phase F6a acquisition manifest.
2. Inventory the TAR without extracting the 32.5-GB archive.
3. Verify the expected 11 paired biological samples.
4. Verify that each sample has:
       - one GEX barcodes.tsv.gz
       - one GEX features.tsv.gz
       - one GEX matrix.mtx.gz
       - one ATAC fragments.tsv.gz
5. Verify GEX cell barcodes against the canonical GEO metadata using the
   compound key:
       sample + barcode
   IMPORTANT: barcode alone is NOT globally unique in this dataset.
6. Inspect a bounded number of ATAC fragment records per sample and verify
   that observed ATAC barcodes map to metadata barcodes for the same sample.
7. Report a strict structural GO/REVIEW decision.

THIS PHASE DOES NOT
-------------------
- extract the entire archive;
- build an ATAC peak-by-cell matrix;
- normalize RNA;
- compute PCA/LSI/scVI/PoissonVI;
- construct pseudotime;
- calculate GDIS;
- calculate CMIL;
- alter any discovery-dataset parameter.

DEFAULT ATAC AUDIT
------------------
ATAC fragment files can be several GB each. By default this script reads only
the first 100,000 fragment records per sample. This is a structural barcode
correspondence audit, not an exhaustive fragment analysis.

An optional exhaustive ATAC barcode scan is available:

    python 15_phase_f7_external_archive_pairing_qc.py \
        --full-atac-barcode-scan

Use that only if desired; it can take substantially longer because all
compressed ATAC fragment files must be decompressed.

RUN
---
    python 15_phase_f7_external_archive_pairing_qc.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd


# =====================================================================
# FROZEN PATHS / SETTINGS
# =====================================================================

ACCESSION = "GSE205117"

DATA_DIR = Path("data") / ACCESSION

METADATA_FILE = DATA_DIR / "GSE205117_cell_metadata.txt.gz"
MANIFEST_FILE = DATA_DIR / "acquisition_manifest.json"
RAW_TAR = DATA_DIR / "GSE205117_RAW.tar"

SAMPLE_COL = "sample"
BARCODE_COL = "barcode"
CELL_COL = "cell"

RNA_QC_COL = "pass_rnaQC"
ATAC_QC_COL = "pass_atacQC"

EXPECTED_SAMPLES = [
    "E7.5_rep1",
    "E7.5_rep2",
    "E7.75_rep1",
    "E8.0_rep1",
    "E8.0_rep2",
    "E8.5_CRISPR_T_KO",
    "E8.5_CRISPR_T_WT",
    "E8.5_rep1",
    "E8.5_rep2",
    "E8.75_rep1",
    "E8.75_rep2",
]

DEFAULT_ATAC_FRAGMENT_LINES = 100_000

# Expected archive member patterns.
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


# =====================================================================
# DISPLAY
# =====================================================================

def line(char="=", width=118):
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
        "display.max_rows", 150,
        "display.max_columns", None,
        "display.width", 360,
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


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit GSE205117 raw archive structure and sample/barcode pairing "
            "without extracting the full archive."
        )
    )

    parser.add_argument(
        "--full-atac-barcode-scan",
        action="store_true",
        help=(
            "Read every ATAC fragment record to enumerate all ATAC barcodes. "
            "This can be slow because the archive contains very large "
            "compressed fragment files."
        ),
    )

    parser.add_argument(
        "--atac-fragment-lines",
        type=int,
        default=DEFAULT_ATAC_FRAGMENT_LINES,
        help=(
            "Number of ATAC fragment records to inspect per sample in the "
            "default bounded audit. Ignored with --full-atac-barcode-scan."
        ),
    )

    return parser.parse_args()


# =====================================================================
# PROVENANCE
# =====================================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_manifest():
    with MANIFEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def verify_raw_provenance():
    """
    Verify local raw TAR against Phase F6a manifest.

    The full SHA-256 is recomputed here because this phase is the first
    analysis step to consume the large archive.
    """
    if not RAW_TAR.exists():
        return False, {
            "reason": f"Raw archive not found: {RAW_TAR}"
        }

    if not MANIFEST_FILE.exists():
        return False, {
            "reason": f"Manifest not found: {MANIFEST_FILE}"
        }

    manifest = load_manifest()

    raw_record = (
        manifest
        .get("files", {})
        .get("raw_archive")
    )

    if not raw_record:
        return False, {
            "reason": (
                "Manifest does not contain a raw_archive record. Re-run "
                "13_phase_f6a_external_metadata_acquisition.py --include-raw"
            )
        }

    expected_size = int(
        raw_record["size_bytes"]
    )

    expected_sha = str(
        raw_record["sha256"]
    )

    observed_size = RAW_TAR.stat().st_size

    print(
        "Computing SHA-256 of the 32.5-GB raw archive once for Phase F7 "
        "consumption verification..."
    )

    observed_sha = sha256_file(
        RAW_TAR
    )

    details = {
        "source_url": raw_record.get(
            "source_url",
            "UNKNOWN",
        ),
        "expected_size": expected_size,
        "observed_size": observed_size,
        "expected_sha256": expected_sha,
        "observed_sha256": observed_sha,
        "size_match": expected_size == observed_size,
        "checksum_match": expected_sha == observed_sha,
    }

    ok = bool(
        details["size_match"]
        and details["checksum_match"]
    )

    return ok, details


# =====================================================================
# METADATA
# =====================================================================

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


def load_metadata():
    metadata = pd.read_csv(
        METADATA_FILE,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    required = [
        SAMPLE_COL,
        BARCODE_COL,
        CELL_COL,
    ]

    missing = [
        column
        for column in required
        if column not in metadata.columns
    ]

    if missing:
        raise RuntimeError(
            "Metadata is missing required columns: "
            + ", ".join(missing)
        )

    metadata[SAMPLE_COL] = (
        metadata[SAMPLE_COL]
        .astype(str)
    )

    metadata[BARCODE_COL] = (
        metadata[BARCODE_COL]
        .astype(str)
    )

    # Frozen compound key: barcode alone is not globally unique.
    metadata["_sample_barcode"] = (
        metadata[SAMPLE_COL]
        + "::"
        + metadata[BARCODE_COL]
    )

    return metadata


# =====================================================================
# TAR INVENTORY
# =====================================================================

def inventory_tar():
    """
    Inventory file members without extracting them.
    """
    with tarfile.open(
        RAW_TAR,
        mode="r",
    ) as tar:
        members = [
            member
            for member in tar.getmembers()
            if member.isfile()
        ]

    return members


def classify_members(members):
    """
    Build per-sample GEX/ATAC archive inventory.
    """
    inventory = defaultdict(
        lambda: {
            "gex_barcodes": [],
            "gex_features": [],
            "gex_matrix": [],
            "atac_fragments": [],
            "unclassified": [],
        }
    )

    parsed_members = []

    unclassified = []

    for member in members:
        name = member.name

        gex = GEX_PATTERN.match(
            name
        )

        if gex:
            gsm, sample, component, extension = gex.groups()

            key = f"gex_{component}"

            inventory[sample][key].append(
                name
            )

            parsed_members.append(
                {
                    "member": name,
                    "GSM": gsm,
                    "sample": sample,
                    "modality": "RNA/GEX",
                    "component": component,
                    "size_bytes": member.size,
                }
            )

            continue

        atac = ATAC_PATTERN.match(
            name
        )

        if atac:
            gsm, sample = atac.groups()

            inventory[sample][
                "atac_fragments"
            ].append(
                name
            )

            parsed_members.append(
                {
                    "member": name,
                    "GSM": gsm,
                    "sample": sample,
                    "modality": "ATAC",
                    "component": "fragments",
                    "size_bytes": member.size,
                }
            )

            continue

        unclassified.append(
            name
        )

    return inventory, pd.DataFrame(
        parsed_members
    ), unclassified


# =====================================================================
# NESTED GZIP READERS
# =====================================================================

def open_gzip_member(tar, member_name):
    """
    Return a text stream for a gzip-compressed TAR member.

    The archive is not extracted to disk.
    """
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

    gz = gzip.GzipFile(
        fileobj=raw,
        mode="rb",
    )

    return io.TextIOWrapper(
        gz,
        encoding="utf-8",
        errors="replace",
        newline="",
    )


def read_gex_barcodes(tar, member_name):
    """
    Read the complete GEX barcode list. Barcode files are small.
    """
    barcodes = []

    with open_gzip_member(
        tar,
        member_name,
    ) as handle:
        for line_text in handle:
            barcode = line_text.strip()

            if barcode:
                barcodes.append(
                    barcode
                )

    return barcodes


def inspect_atac_barcodes(
    tar,
    member_name,
    max_lines,
    full_scan=False,
):
    """
    Read ATAC fragment TSV records directly from the compressed TAR member.

    Expected columns:
        chromosome, start, end, barcode, reads

    Returns:
        unique_barcodes
        n_lines_read
        malformed_lines
        n_columns_seen
    """
    barcodes = set()
    n_lines = 0
    malformed = 0
    column_counts = defaultdict(
        int
    )

    with open_gzip_member(
        tar,
        member_name,
    ) as handle:
        for line_text in handle:
            if (
                not full_scan
                and n_lines >= max_lines
            ):
                break

            line_text = line_text.rstrip(
                "\n"
            )

            if not line_text:
                continue

            fields = line_text.split(
                "\t"
            )

            column_counts[
                len(fields)
            ] += 1

            if len(fields) < 4:
                malformed += 1
                n_lines += 1
                continue

            barcode = fields[3]

            if barcode:
                barcodes.add(
                    barcode
                )

            n_lines += 1

    return {
        "barcodes": barcodes,
        "n_lines": n_lines,
        "malformed": malformed,
        "column_counts": dict(
            column_counts
        ),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    args = parse_args()

    section(
        "PHASE F7 — GSE205117 ARCHIVE STRUCTURE AND RNA/ATAC PAIRING QC"
    )

    print(
        f"Raw archive: {RAW_TAR.resolve()}"
    )

    print(
        f"Metadata:    {METADATA_FILE.resolve()}"
    )

    print()

    print(
        "Frozen cell pairing key: sample + barcode"
    )

    print(
        "Barcode alone is NOT treated as globally unique."
    )

    print()

    print(
        "No representation is constructed."
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
    # Provenance.
    # -----------------------------------------------------------------

    section(
        "1. RAW ARCHIVE PROVENANCE"
    )

    verified, provenance = verify_raw_provenance()

    for key, value in provenance.items():
        print(
            f"{key}: {value}"
        )

    if not verified:
        print()

        print(
            "PHASE F7 VERDICT: RAW ARCHIVE PROVENANCE FAILED"
        )

        return

    print()

    print(
        "Raw archive provenance: PASS"
    )

    # -----------------------------------------------------------------
    # Metadata key audit.
    # -----------------------------------------------------------------

    section(
        "2. METADATA CELL-PAIRING KEY AUDIT"
    )

    metadata = load_metadata()

    n_rows = len(
        metadata
    )

    n_barcodes = metadata[
        BARCODE_COL
    ].nunique(
        dropna=False
    )

    n_compound = metadata[
        "_sample_barcode"
    ].nunique(
        dropna=False
    )

    n_cells = metadata[
        CELL_COL
    ].nunique(
        dropna=False
    )

    print(
        f"Metadata rows:                 {n_rows:,}"
    )

    print(
        f"Unique barcode values:         {n_barcodes:,}"
    )

    print(
        f"Unique sample+barcode keys:    {n_compound:,}"
    )

    print(
        f"Unique cell identifiers:       {n_cells:,}"
    )

    print()

    print(
        f"Barcode alone globally unique: "
        f"{n_barcodes == n_rows}"
    )

    print(
        f"sample+barcode globally unique: "
        f"{n_compound == n_rows}"
    )

    print(
        f"cell globally unique: "
        f"{n_cells == n_rows}"
    )

    duplicate_key_count = int(
        metadata[
            "_sample_barcode"
        ].duplicated(
            keep=False
        ).sum()
    )

    print(
        f"Rows involved in duplicated sample+barcode keys: "
        f"{duplicate_key_count:,}"
    )

    # -----------------------------------------------------------------
    # Archive inventory.
    # -----------------------------------------------------------------

    section(
        "3. TAR MEMBER INVENTORY"
    )

    members = inventory_tar()

    inventory, parsed_df, unclassified = classify_members(
        members
    )

    print(
        f"Total file members: {len(members):,}"
    )

    print(
        f"Parsed GEX/ATAC members: {len(parsed_df):,}"
    )

    print(
        f"Unclassified members: {len(unclassified):,}"
    )

    if not parsed_df.empty:
        print()

        display = parsed_df.copy()

        display[
            "size"
        ] = display[
            "size_bytes"
        ].map(
            human_size
        )

        print_df(
            display[
                [
                    "GSM",
                    "sample",
                    "modality",
                    "component",
                    "size",
                    "member",
                ]
            ].set_index(
                [
                    "sample",
                    "modality",
                    "component",
                ]
            )
        )

    if unclassified:
        print()

        print(
            "Unclassified archive members:"
        )

        for name in unclassified[:100]:
            print(
                f"  {name}"
            )

        if len(unclassified) > 100:
            print(
                f"  ... {len(unclassified) - 100} more"
            )

    # -----------------------------------------------------------------
    # Sample completeness.
    # -----------------------------------------------------------------

    section(
        "4. PER-SAMPLE MODALITY COMPLETENESS"
    )

    metadata_samples = set(
        metadata[
            SAMPLE_COL
        ].astype(str)
    )

    archive_samples = set(
        inventory.keys()
    )

    expected_samples = set(
        EXPECTED_SAMPLES
    )

    all_samples = sorted(
        metadata_samples
        | archive_samples
        | expected_samples
    )

    rows = []

    for sample in all_samples:
        record = inventory[
            sample
        ]

        rows.append(
            {
                "sample": sample,
                "in_expected_set": sample in expected_samples,
                "in_metadata": sample in metadata_samples,
                "GEX_barcodes": len(
                    record["gex_barcodes"]
                ),
                "GEX_features": len(
                    record["gex_features"]
                ),
                "GEX_matrix": len(
                    record["gex_matrix"]
                ),
                "ATAC_fragments": len(
                    record["atac_fragments"]
                ),
            }
        )

    sample_table = pd.DataFrame(
        rows
    ).set_index(
        "sample"
    )

    sample_table[
        "complete_1x_each"
    ] = (
        sample_table[
            [
                "GEX_barcodes",
                "GEX_features",
                "GEX_matrix",
                "ATAC_fragments",
            ]
        ]
        .eq(1)
        .all(
            axis=1
        )
    )

    print_df(
        sample_table,
        digits=0,
    )

    expected_exact = bool(
        metadata_samples
        == expected_samples
        == archive_samples
    )

    all_complete = bool(
        sample_table.loc[
            sorted(expected_samples),
            "complete_1x_each",
        ].all()
    )

    print()

    print(
        f"Metadata/archive/expected sample sets identical: "
        f"{expected_exact}"
    )

    print(
        f"All 11 samples have exactly one of each required member: "
        f"{all_complete}"
    )

    # -----------------------------------------------------------------
    # GEX barcode exact audit.
    # -----------------------------------------------------------------

    section(
        "5. GEX BARCODE ↔ METADATA EXACT CORRESPONDENCE"
    )

    gex_rows = []

    with tarfile.open(
        RAW_TAR,
        mode="r",
    ) as tar:

        for sample in EXPECTED_SAMPLES:
            record = inventory[
                sample
            ]

            if len(
                record[
                    "gex_barcodes"
                ]
            ) != 1:
                gex_rows.append(
                    {
                        "sample": sample,
                        "status": "MISSING/AMBIGUOUS",
                    }
                )

                continue

            member_name = record[
                "gex_barcodes"
            ][0]

            raw_barcodes = read_gex_barcodes(
                tar,
                member_name,
            )

            raw_set = set(
                raw_barcodes
            )

            meta_sample = metadata.loc[
                metadata[
                    SAMPLE_COL
                ].eq(
                    sample
                )
            ]

            meta_set = set(
                meta_sample[
                    BARCODE_COL
                ].astype(str)
            )

            intersection = (
                raw_set
                & meta_set
            )

            missing_from_gex = (
                meta_set
                - raw_set
            )

            extra_in_gex = (
                raw_set
                - meta_set
            )

            gex_rows.append(
                {
                    "sample": sample,
                    "gex_barcode_rows": len(
                        raw_barcodes
                    ),
                    "gex_unique_barcodes": len(
                        raw_set
                    ),
                    "metadata_unique_barcodes": len(
                        meta_set
                    ),
                    "intersection": len(
                        intersection
                    ),
                    "metadata_not_in_gex": len(
                        missing_from_gex
                    ),
                    "gex_not_in_metadata": len(
                        extra_in_gex
                    ),
                    "metadata_coverage": (
                        len(
                            intersection
                        )
                        / len(
                            meta_set
                        )
                        if meta_set
                        else np.nan
                    ),
                    "exact_set_match": (
                        raw_set
                        == meta_set
                    ),
                    "status": "OK",
                }
            )

    gex_df = pd.DataFrame(
        gex_rows
    ).set_index(
        "sample"
    )

    print_df(
        gex_df,
        digits=4,
    )

    valid_gex = gex_df.loc[
        gex_df[
            "status"
        ].eq(
            "OK"
        )
    ]

    gex_all_read = (
        len(
            valid_gex
        )
        == len(
            EXPECTED_SAMPLES
        )
    )

    # We do NOT require exact equality to pass because metadata may include
    # cells failing RNA QC and/or the archive barcode list may represent a
    # different Cell Ranger filtering stage. We require complete metadata
    # coverage or explain the discrepancy quantitatively.
    if gex_all_read:
        min_gex_coverage = float(
            valid_gex[
                "metadata_coverage"
            ].min()
        )
    else:
        min_gex_coverage = 0.0

    print()

    print(
        f"All 11 GEX barcode files readable: "
        f"{gex_all_read}"
    )

    print(
        f"Minimum per-sample metadata barcode coverage by GEX: "
        f"{min_gex_coverage:.4f}"
    )

    # -----------------------------------------------------------------
    # ATAC bounded or full barcode audit.
    # -----------------------------------------------------------------

    section(
        "6. ATAC FRAGMENT BARCODE ↔ METADATA CORRESPONDENCE"
    )

    if args.full_atac_barcode_scan:
        print(
            "Mode: FULL ATAC barcode scan"
        )
    else:
        print(
            f"Mode: bounded structural scan "
            f"({args.atac_fragment_lines:,} fragment records/sample)"
        )

    atac_rows = []

    with tarfile.open(
        RAW_TAR,
        mode="r",
    ) as tar:

        for sample in EXPECTED_SAMPLES:
            record = inventory[
                sample
            ]

            if len(
                record[
                    "atac_fragments"
                ]
            ) != 1:
                atac_rows.append(
                    {
                        "sample": sample,
                        "status": "MISSING/AMBIGUOUS",
                    }
                )

                continue

            member_name = record[
                "atac_fragments"
            ][0]

            inspected = inspect_atac_barcodes(
                tar,
                member_name,
                max_lines=args.atac_fragment_lines,
                full_scan=args.full_atac_barcode_scan,
            )

            atac_set = inspected[
                "barcodes"
            ]

            meta_sample = metadata.loc[
                metadata[
                    SAMPLE_COL
                ].eq(
                    sample
                )
            ]

            meta_set = set(
                meta_sample[
                    BARCODE_COL
                ].astype(str)
            )

            intersection = (
                atac_set
                & meta_set
            )

            unknown_atac = (
                atac_set
                - meta_set
            )

            # For a bounded fragment scan, we expect only a subset of all
            # cell barcodes to appear in the first N records. Therefore the
            # key structural criterion is whether sampled ATAC barcodes map
            # back to metadata for the correct sample.
            sampled_mapping_fraction = (
                len(
                    intersection
                )
                / len(
                    atac_set
                )
                if atac_set
                else np.nan
            )

            atac_rows.append(
                {
                    "sample": sample,
                    "fragment_lines_read": inspected[
                        "n_lines"
                    ],
                    "unique_atac_barcodes_seen": len(
                        atac_set
                    ),
                    "mapped_to_metadata": len(
                        intersection
                    ),
                    "unknown_atac_barcodes": len(
                        unknown_atac
                    ),
                    "sampled_barcode_mapping_fraction": (
                        sampled_mapping_fraction
                    ),
                    "malformed_lines": inspected[
                        "malformed"
                    ],
                    "column_counts": str(
                        inspected[
                            "column_counts"
                        ]
                    ),
                    "status": "OK",
                }
            )

    atac_df = pd.DataFrame(
        atac_rows
    ).set_index(
        "sample"
    )

    print_df(
        atac_df,
        digits=4,
    )

    valid_atac = atac_df.loc[
        atac_df[
            "status"
        ].eq(
            "OK"
        )
    ]

    atac_all_read = (
        len(
            valid_atac
        )
        == len(
            EXPECTED_SAMPLES
        )
    )

    if atac_all_read:
        min_atac_mapping = float(
            valid_atac[
                "sampled_barcode_mapping_fraction"
            ].min()
        )
    else:
        min_atac_mapping = 0.0

    total_malformed = (
        int(
            valid_atac[
                "malformed_lines"
            ].sum()
        )
        if atac_all_read
        else -1
    )

    print()

    print(
        f"All 11 ATAC fragment files readable: "
        f"{atac_all_read}"
    )

    print(
        f"Minimum sampled ATAC barcode mapping fraction: "
        f"{min_atac_mapping:.4f}"
    )

    print(
        f"Total malformed fragment records encountered: "
        f"{total_malformed:,}"
    )

    # -----------------------------------------------------------------
    # QC-filtered metadata key counts.
    # -----------------------------------------------------------------

    section(
        "7. JOINT-QC PAIRING COUNTS BY SAMPLE"
    )

    joint_mask = pd.Series(
        True,
        index=metadata.index,
    )

    if RNA_QC_COL in metadata.columns:
        joint_mask &= (
            normalize_bool(
                metadata[
                    RNA_QC_COL
                ]
            )
            .fillna(
                False
            )
        )

    if ATAC_QC_COL in metadata.columns:
        joint_mask &= (
            normalize_bool(
                metadata[
                    ATAC_QC_COL
                ]
            )
            .fillna(
                False
            )
        )

    joint = metadata.loc[
        joint_mask
    ].copy()

    joint_counts = (
        joint[
            SAMPLE_COL
        ]
        .value_counts()
        .reindex(
            EXPECTED_SAMPLES,
            fill_value=0,
        )
        .rename(
            "joint_QC_cells"
        )
        .to_frame()
    )

    print_df(
        joint_counts
    )

    print()

    print(
        f"Total cells passing both RNA and ATAC QC: "
        f"{len(joint):,}"
    )

    # -----------------------------------------------------------------
    # Decision.
    # -----------------------------------------------------------------

    section(
        "8. PHASE F7 SUITABILITY CHECKS"
    )

    # Structural thresholds are intentionally strict for archive/sample
    # organization. Barcode overlap thresholds are conservative because raw
    # Cell Ranger outputs and final metadata can differ by QC stage.
    checks = pd.DataFrame(
        [
            {
                "criterion": "Raw TAR provenance verified",
                "pass": verified,
            },
            {
                "criterion": "sample+barcode is globally unique",
                "pass": n_compound == n_rows,
            },
            {
                "criterion": "cell identifier is globally unique",
                "pass": n_cells == n_rows,
            },
            {
                "criterion": "Expected/metadata/archive sample sets identical",
                "pass": expected_exact,
            },
            {
                "criterion": (
                    "Each sample has exactly one GEX barcode/features/matrix "
                    "and one ATAC fragments file"
                ),
                "pass": all_complete,
            },
            {
                "criterion": "All 11 GEX barcode files readable",
                "pass": gex_all_read,
            },
            {
                "criterion": (
                    "GEX covers >=95% of metadata barcodes in every sample"
                ),
                "pass": min_gex_coverage >= 0.95,
            },
            {
                "criterion": "All 11 ATAC fragment files readable",
                "pass": atac_all_read,
            },
            {
                "criterion": (
                    ">=95% of sampled ATAC barcodes map to metadata "
                    "within the same sample"
                ),
                "pass": min_atac_mapping >= 0.95,
            },
            {
                "criterion": "No malformed sampled ATAC fragment records",
                "pass": total_malformed == 0,
            },
            {
                "criterion": "Joint RNA+ATAC QC retains cells",
                "pass": len(joint) > 0,
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

    n_pass = int(
        checks[
            "pass"
        ].sum()
    )

    print()

    print(
        f"Criteria passed: {n_pass}/{len(checks)}"
    )

    section(
        "9. PHASE F7 DECISION"
    )

    all_pass = bool(
        checks[
            "pass"
        ].all()
    )

    if all_pass:
        print(
            "PHASE F7 VERDICT: GO — PAIRED ARCHIVE STRUCTURE VERIFIED"
        )

        print()

        print(
            "The 11 biological samples have matched RNA/GEX and ATAC "
            "archive components, sample+barcode is the correct cell key, "
            "and sampled ATAC barcodes map back to metadata within sample."
        )

        print()

        print(
            "Next phase should acquire/audit independent modality-specific "
            "representations (RNA PCA and ATAC LSI) before selecting the "
            "external common trajectory."
        )

        print()

        print(
            "Suggested next script:"
        )

        print(
            "  16_phase_f8_external_representation_acquisition_qc.py"
        )

    else:
        print(
            "PHASE F7 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not construct representations or pseudotime until the "
            "failed pairing/archive criterion is understood."
        )

    print()

    print(
        "No representation was constructed."
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

