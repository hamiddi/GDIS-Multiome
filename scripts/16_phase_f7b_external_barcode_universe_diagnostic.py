#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
16_phase_f7b_external_barcode_universe_diagnostic.py
====================================================

Project:
GDIS-Multiome external validation

Dataset:
GSE205117

PHASE F7b
---------
Diagnostic follow-up to Phase F7.

WHY THIS PHASE EXISTS
---------------------
Phase F7 correctly verified:
    - raw archive provenance;
    - sample+barcode as the globally unique pairing key;
    - 11/11 expected samples;
    - one GEX barcode/features/matrix set per sample;
    - one ATAC fragment file per sample.

However, three Phase F7 criteria were not valid as originally written:

1. GEX barcode files were compared against ALL metadata rows.
   The metadata table contains multiple QC/annotation strata, so the relevant
   GEX universe must first be identified empirically.

2. ATAC raw fragments contain many barcodes beyond the final annotated/QC cell
   set. Therefore ">=95% of sampled ATAC barcodes must map to metadata" is not
   an appropriate bounded-scan pass/fail criterion.

3. 10x fragment files contain '#'-prefixed header lines. These must be counted
   as headers, not malformed records.

F7b DOES NOT relax a failed biological threshold. It corrects the structural
audit definition.

GOALS
-----
A. Identify which metadata stratum most closely corresponds to each sample's
   complete GEX barcode set.

B. Explicitly distinguish:
       all metadata rows
       genotype-known rows
       celltype-mapped rows
       pass_rnaQC rows
       pass_atacQC rows
       joint RNA+ATAC QC rows
       mapped + joint QC rows

C. Parse 10x ATAC '#' header lines correctly.

D. Treat the bounded ATAC scan as descriptive only.

E. Optionally perform a FULL targeted ATAC scan to ask the scientifically
   relevant pairing question:

       Are the metadata-defined cells we plan to analyze actually present
       in the ATAC fragment files for the same sample?

The optional full scan can be run with:

    python 16_phase_f7b_external_barcode_universe_diagnostic.py \
        --full-atac-target-scan

The full scan does not store all fragment barcodes. It keeps only the target
metadata barcode sets and marks them as they are encountered.

NO ANALYSIS
-----------
No representation.
No pseudotime.
No GDIS.
No CMIL.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import io
import json
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd


# =====================================================================
# PATHS / SETTINGS
# =====================================================================

ACCESSION = "GSE205117"
DATA_DIR = Path("data") / ACCESSION

METADATA_FILE = DATA_DIR / "GSE205117_cell_metadata.txt.gz"
MANIFEST_FILE = DATA_DIR / "acquisition_manifest.json"
RAW_TAR = DATA_DIR / "GSE205117_RAW.tar"

SAMPLE_COL = "sample"
BARCODE_COL = "barcode"
GENOTYPE_COL = "genotype"
CELLTYPE_COL = "celltype.mapped"
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

def line(char="=", width=120):
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
        "display.width", 380,
        "display.max_colwidth", 100,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string())


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the GEX/ATAC barcode universe for GSE205117 without "
            "constructing any representation or trajectory."
        )
    )

    parser.add_argument(
        "--atac-fragment-lines",
        type=int,
        default=DEFAULT_ATAC_FRAGMENT_LINES,
        help=(
            "Number of non-header ATAC fragment records to inspect per sample "
            "during the bounded descriptive scan."
        ),
    )

    parser.add_argument(
        "--full-atac-target-scan",
        action="store_true",
        help=(
            "Scan every ATAC fragment file and test whether metadata target "
            "barcodes are present. This can take substantial time."
        ),
    )

    return parser.parse_args()


# =====================================================================
# HELPERS
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
    df = pd.read_csv(
        METADATA_FILE,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    df[SAMPLE_COL] = df[SAMPLE_COL].astype(str)
    df[BARCODE_COL] = df[BARCODE_COL].astype(str)

    return df


def load_manifest():
    with MANIFEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def verify_manifest_presence():
    if not METADATA_FILE.exists():
        raise RuntimeError(
            f"Missing metadata file: {METADATA_FILE}"
        )

    if not RAW_TAR.exists():
        raise RuntimeError(
            f"Missing raw archive: {RAW_TAR}"
        )

    if not MANIFEST_FILE.exists():
        raise RuntimeError(
            f"Missing acquisition manifest: {MANIFEST_FILE}"
        )

    manifest = load_manifest()

    raw_record = (
        manifest
        .get("files", {})
        .get("raw_archive")
    )

    if raw_record is None:
        raise RuntimeError(
            "Acquisition manifest has no raw_archive record."
        )

    recorded_size = int(
        raw_record["size_bytes"]
    )

    observed_size = RAW_TAR.stat().st_size

    if recorded_size != observed_size:
        raise RuntimeError(
            "Raw archive size no longer matches acquisition manifest."
        )

    return manifest


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

            match = GEX_PATTERN.match(
                name
            )

            if match:
                _, sample, component, _ = match.groups()

                inventory[
                    sample
                ][
                    f"gex_{component}"
                ].append(
                    name
                )

                continue

            match = ATAC_PATTERN.match(
                name
            )

            if match:
                _, sample = match.groups()

                inventory[
                    sample
                ][
                    "atac_fragments"
                ].append(
                    name
                )

    return inventory


def open_gzip_member(tar, member_name):
    member = tar.getmember(
        member_name
    )

    raw = tar.extractfile(
        member
    )

    if raw is None:
        raise RuntimeError(
            f"Could not extract stream for: {member_name}"
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
    values = []

    with open_gzip_member(
        tar,
        member_name,
    ) as handle:
        for line_text in handle:
            barcode = line_text.strip()

            if barcode:
                values.append(
                    barcode
                )

    return values


def build_metadata_strata(df):
    """
    Return named boolean masks defining plausible cell universes.
    """
    masks = {}

    masks[
        "all_metadata"
    ] = pd.Series(
        True,
        index=df.index,
    )

    masks[
        "genotype_known"
    ] = df[
        GENOTYPE_COL
    ].notna()

    masks[
        "celltype_mapped"
    ] = df[
        CELLTYPE_COL
    ].notna()

    rna_pass = normalize_bool(
        df[
            RNA_QC_COL
        ]
    ).fillna(
        False
    )

    atac_pass = normalize_bool(
        df[
            ATAC_QC_COL
        ]
    ).fillna(
        False
    )

    masks[
        "pass_rnaQC"
    ] = rna_pass

    masks[
        "pass_atacQC"
    ] = atac_pass

    masks[
        "joint_RNA_ATAC_QC"
    ] = (
        rna_pass
        & atac_pass
    )

    masks[
        "mapped_and_pass_rnaQC"
    ] = (
        df[
            CELLTYPE_COL
        ].notna()
        & rna_pass
    )

    masks[
        "mapped_and_joint_QC"
    ] = (
        df[
            CELLTYPE_COL
        ].notna()
        & rna_pass
        & atac_pass
    )

    return masks


def set_metrics(reference_set, observed_set):
    intersection = (
        reference_set
        & observed_set
    )

    reference_only = (
        reference_set
        - observed_set
    )

    observed_only = (
        observed_set
        - reference_set
    )

    recall = (
        len(
            intersection
        )
        / len(
            reference_set
        )
        if reference_set
        else np.nan
    )

    precision = (
        len(
            intersection
        )
        / len(
            observed_set
        )
        if observed_set
        else np.nan
    )

    union = (
        reference_set
        | observed_set
    )

    jaccard = (
        len(
            intersection
        )
        / len(
            union
        )
        if union
        else np.nan
    )

    return {
        "reference_n": len(
            reference_set
        ),
        "observed_n": len(
            observed_set
        ),
        "intersection": len(
            intersection
        ),
        "reference_not_observed": len(
            reference_only
        ),
        "observed_not_reference": len(
            observed_only
        ),
        "reference_recall": recall,
        "observed_precision": precision,
        "jaccard": jaccard,
        "exact_match": (
            reference_set
            == observed_set
        ),
    }


def bounded_atac_scan(
    tar,
    member_name,
    metadata_set,
    max_fragment_records,
):
    """
    Inspect initial ATAC fragment records.

    '#' lines are 10x header lines and are NOT malformed records.
    The bounded scan is descriptive only.
    """
    header_lines = 0
    fragment_records = 0
    malformed_records = 0

    unique_barcodes = set()
    metadata_barcodes_seen = set()

    with open_gzip_member(
        tar,
        member_name,
    ) as handle:
        for line_text in handle:
            if line_text.startswith(
                "#"
            ):
                header_lines += 1
                continue

            if (
                fragment_records
                >= max_fragment_records
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

            if len(fields) < 5:
                malformed_records += 1
                continue

            barcode = fields[3]

            unique_barcodes.add(
                barcode
            )

            if barcode in metadata_set:
                metadata_barcodes_seen.add(
                    barcode
                )

            fragment_records += 1

    return {
        "header_lines": header_lines,
        "fragment_records": fragment_records,
        "malformed_records": malformed_records,
        "unique_raw_atac_barcodes_seen": len(
            unique_barcodes
        ),
        "unique_metadata_barcodes_seen": len(
            metadata_barcodes_seen
        ),
        "raw_barcodes_in_metadata_fraction": (
            len(
                unique_barcodes
                & metadata_set
            )
            / len(
                unique_barcodes
            )
            if unique_barcodes
            else np.nan
        ),
    }


def full_target_atac_scan(
    tar,
    member_name,
    target_sets,
):
    """
    Exhaustively scan an ATAC fragments file, but retain only target barcodes.

    target_sets:
        dict[str, set[str]]

    Returns coverage of each target set.

    This is more meaningful than asking whether all raw ATAC barcodes belong
    to the filtered metadata universe.
    """
    remaining = {
        name: set(
            values
        )
        for name, values in target_sets.items()
    }

    found = {
        name: set()
        for name in target_sets
    }

    header_lines = 0
    fragment_records = 0
    malformed_records = 0

    with open_gzip_member(
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

            if len(fields) < 5:
                malformed_records += 1
                continue

            barcode = fields[3]

            for name in target_sets:
                if barcode in remaining[
                    name
                ]:
                    found[
                        name
                    ].add(
                        barcode
                    )

                    remaining[
                        name
                    ].discard(
                        barcode
                    )

            fragment_records += 1

            # If every target barcode has already been found, we can stop.
            if all(
                len(
                    values
                )
                == 0
                for values in remaining.values()
            ):
                break

    results = {
        "header_lines": header_lines,
        "fragment_records_scanned": fragment_records,
        "malformed_records": malformed_records,
    }

    for name, target in target_sets.items():
        n_target = len(
            target
        )

        n_found = len(
            found[
                name
            ]
        )

        results[
            f"{name}_target_n"
        ] = n_target

        results[
            f"{name}_found"
        ] = n_found

        results[
            f"{name}_coverage"
        ] = (
            n_found
            / n_target
            if n_target
            else np.nan
        )

    return results


# =====================================================================
# MAIN
# =====================================================================

def main():
    args = parse_args()

    section(
        "PHASE F7b — GSE205117 BARCODE-UNIVERSE DIAGNOSTIC"
    )

    print(
        "Purpose: resolve F7 audit-definition mismatches without changing "
        "the biological analysis."
    )

    print()

    print(
        "No representation."
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
        "1. INPUT / MANIFEST CONSISTENCY"
    )

    manifest = verify_manifest_presence()

    print(
        f"Metadata: {METADATA_FILE.resolve()}"
    )

    print(
        f"Raw TAR:  {RAW_TAR.resolve()}"
    )

    print(
        f"Raw TAR size matches manifest: True"
    )

    raw_record = manifest[
        "files"
    ][
        "raw_archive"
    ]

    print(
        f"Recorded raw SHA-256: {raw_record['sha256']}"
    )

    print()

    print(
        "Note: F7 already recomputed and verified the full raw SHA-256; "
        "F7b does not hash 32.5 GB a second time."
    )

    # -----------------------------------------------------------------
    # Metadata strata.
    # -----------------------------------------------------------------

    section(
        "2. METADATA CELL UNIVERSES"
    )

    metadata = load_metadata()
    strata = build_metadata_strata(
        metadata
    )

    stratum_rows = []

    for name, mask in strata.items():
        subset = metadata.loc[
            mask
        ]

        stratum_rows.append(
            {
                "stratum": name,
                "rows": len(
                    subset
                ),
                "unique_sample_barcode": (
                    subset[
                        SAMPLE_COL
                    ]
                    + "::"
                    + subset[
                        BARCODE_COL
                    ]
                ).nunique(),
                "samples": subset[
                    SAMPLE_COL
                ].nunique(),
            }
        )

    print_df(
        pd.DataFrame(
            stratum_rows
        ).set_index(
            "stratum"
        ),
        digits=0,
    )

    # -----------------------------------------------------------------
    # Inventory.
    # -----------------------------------------------------------------

    section(
        "3. RAW ARCHIVE MEMBER RESOLUTION"
    )

    inventory = inventory_tar()

    rows = []

    for sample in EXPECTED_SAMPLES:
        record = inventory[
            sample
        ]

        rows.append(
            {
                "sample": sample,
                "GEX_barcodes": len(
                    record[
                        "gex_barcodes"
                    ]
                ),
                "GEX_features": len(
                    record[
                        "gex_features"
                    ]
                ),
                "GEX_matrix": len(
                    record[
                        "gex_matrix"
                    ]
                ),
                "ATAC_fragments": len(
                    record[
                        "atac_fragments"
                    ]
                ),
            }
        )

    archive_table = pd.DataFrame(
        rows
    ).set_index(
        "sample"
    )

    archive_table[
        "complete"
    ] = (
        archive_table.eq(
            1
        ).all(
            axis=1
        )
    )

    print_df(
        archive_table,
        digits=0,
    )

    # -----------------------------------------------------------------
    # GEX vs metadata strata.
    # -----------------------------------------------------------------

    section(
        "4. COMPLETE GEX BARCODE SET VS METADATA STRATA"
    )

    gex_comparison_rows = []

    best_rows = []

    with tarfile.open(
        RAW_TAR,
        "r",
    ) as tar:

        for sample in EXPECTED_SAMPLES:
            record = inventory[
                sample
            ]

            member_name = record[
                "gex_barcodes"
            ][0]

            gex_barcodes = set(
                read_gex_barcodes(
                    tar,
                    member_name,
                )
            )

            sample_metadata = metadata.loc[
                metadata[
                    SAMPLE_COL
                ].eq(
                    sample
                )
            ]

            sample_strata = build_metadata_strata(
                sample_metadata
            )

            sample_rows = []

            for stratum_name, mask in sample_strata.items():
                reference_set = set(
                    sample_metadata.loc[
                        mask,
                        BARCODE_COL,
                    ].astype(str)
                )

                metrics = set_metrics(
                    reference_set,
                    gex_barcodes,
                )

                row = {
                    "sample": sample,
                    "stratum": stratum_name,
                    **metrics,
                }

                gex_comparison_rows.append(
                    row
                )

                sample_rows.append(
                    row
                )

            best = sorted(
                sample_rows,
                key=lambda row: (
                    -row[
                        "jaccard"
                    ],
                    -row[
                        "reference_recall"
                    ],
                    row[
                        "reference_not_observed"
                    ],
                ),
            )[0]

            best_rows.append(
                best
            )

    gex_compare = pd.DataFrame(
        gex_comparison_rows
    )

    for sample in EXPECTED_SAMPLES:
        subsection(
            sample
        )

        sample_df = (
            gex_compare.loc[
                gex_compare[
                    "sample"
                ].eq(
                    sample
                )
            ]
            .drop(
                columns=[
                    "sample"
                ]
            )
            .set_index(
                "stratum"
            )
        )

        print_df(
            sample_df,
            digits=4,
        )

    subsection(
        "Best-matching metadata stratum per sample"
    )

    best_df = pd.DataFrame(
        best_rows
    ).set_index(
        "sample"
    )

    print_df(
        best_df[
            [
                "stratum",
                "reference_n",
                "observed_n",
                "intersection",
                "reference_not_observed",
                "observed_not_reference",
                "reference_recall",
                "observed_precision",
                "jaccard",
                "exact_match",
            ]
        ],
        digits=4,
    )

    # -----------------------------------------------------------------
    # Bounded ATAC scan, correctly parsing headers.
    # -----------------------------------------------------------------

    section(
        "5. BOUNDED ATAC STRUCTURAL SCAN — DESCRIPTIVE ONLY"
    )

    print(
        f"Inspecting first {args.atac_fragment_lines:,} NON-HEADER "
        f"fragment records per sample."
    )

    print()

    print(
        "'#' lines are recorded as 10x headers, not malformed fragments."
    )

    atac_rows = []

    with tarfile.open(
        RAW_TAR,
        "r",
    ) as tar:

        for sample in EXPECTED_SAMPLES:
            record = inventory[
                sample
            ]

            member_name = record[
                "atac_fragments"
            ][0]

            sample_metadata = metadata.loc[
                metadata[
                    SAMPLE_COL
                ].eq(
                    sample
                )
            ]

            metadata_set = set(
                sample_metadata[
                    BARCODE_COL
                ].astype(str)
            )

            result = bounded_atac_scan(
                tar,
                member_name,
                metadata_set,
                args.atac_fragment_lines,
            )

            atac_rows.append(
                {
                    "sample": sample,
                    **result,
                }
            )

    bounded_df = pd.DataFrame(
        atac_rows
    ).set_index(
        "sample"
    )

    print_df(
        bounded_df,
        digits=4,
    )

    total_headers = int(
        bounded_df[
            "header_lines"
        ].sum()
    )

    total_malformed = int(
        bounded_df[
            "malformed_records"
        ].sum()
    )

    print()

    print(
        f"Total '#' 10x header lines across samples: "
        f"{total_headers:,}"
    )

    print(
        f"True malformed non-header records observed: "
        f"{total_malformed:,}"
    )

    print()

    print(
        "The fraction of raw ATAC barcodes that occur in filtered metadata "
        "is DESCRIPTIVE ONLY and is not used as a suitability threshold."
    )

    # -----------------------------------------------------------------
    # Optional full targeted scan.
    # -----------------------------------------------------------------

    full_scan_complete = False
    full_scan_min_joint = np.nan
    full_scan_min_mapped_joint = np.nan

    if args.full_atac_target_scan:

        section(
            "6. FULL TARGETED ATAC PRESENCE SCAN"
        )

        print(
            "Scanning all ATAC fragments for metadata-defined target cells."
        )

        print()

        full_rows = []

        with tarfile.open(
            RAW_TAR,
            "r",
        ) as tar:

            for sample in EXPECTED_SAMPLES:

                sample_metadata = metadata.loc[
                    metadata[
                        SAMPLE_COL
                    ].eq(
                        sample
                    )
                ].copy()

                sample_strata = build_metadata_strata(
                    sample_metadata
                )

                target_sets = {
                    "pass_atacQC": set(
                        sample_metadata.loc[
                            sample_strata[
                                "pass_atacQC"
                            ],
                            BARCODE_COL,
                        ].astype(str)
                    ),
                    "joint_QC": set(
                        sample_metadata.loc[
                            sample_strata[
                                "joint_RNA_ATAC_QC"
                            ],
                            BARCODE_COL,
                        ].astype(str)
                    ),
                    "mapped_joint_QC": set(
                        sample_metadata.loc[
                            sample_strata[
                                "mapped_and_joint_QC"
                            ],
                            BARCODE_COL,
                        ].astype(str)
                    ),
                }

                member_name = inventory[
                    sample
                ][
                    "atac_fragments"
                ][0]

                result = full_target_atac_scan(
                    tar,
                    member_name,
                    target_sets,
                )

                full_rows.append(
                    {
                        "sample": sample,
                        **result,
                    }
                )

                print(
                    f"Completed ATAC target scan: {sample}"
                )

        full_df = pd.DataFrame(
            full_rows
        ).set_index(
            "sample"
        )

        print()

        print_df(
            full_df,
            digits=4,
        )

        full_scan_complete = (
            len(
                full_df
            )
            == len(
                EXPECTED_SAMPLES
            )
        )

        if full_scan_complete:
            full_scan_min_joint = float(
                full_df[
                    "joint_QC_coverage"
                ].min()
            )

            full_scan_min_mapped_joint = float(
                full_df[
                    "mapped_joint_QC_coverage"
                ].min()
            )

        print()

        print(
            f"Minimum joint-QC ATAC target coverage: "
            f"{full_scan_min_joint:.4f}"
        )

        print(
            f"Minimum mapped+joint-QC ATAC target coverage: "
            f"{full_scan_min_mapped_joint:.4f}"
        )

    else:

        section(
            "6. FULL TARGETED ATAC PRESENCE SCAN"
        )

        print(
            "Not requested."
        )

        print()

        print(
            "If needed after reviewing this diagnostic, run:"
        )

        print(
            "  python "
            "16_phase_f7b_external_barcode_universe_diagnostic.py "
            "--full-atac-target-scan"
        )

    # -----------------------------------------------------------------
    # Diagnostic conclusion.
    # -----------------------------------------------------------------

    section(
        "7. F7b DIAGNOSTIC CONCLUSION"
    )

    archive_complete = bool(
        archive_table[
            "complete"
        ].all()
    )

    all_gex_read = (
        best_df.shape[
            0
        ]
        == len(
            EXPECTED_SAMPLES
        )
    )

    no_true_malformed = (
        total_malformed
        == 0
    )

    print(
        f"Archive modality structure complete: {archive_complete}"
    )

    print(
        f"All GEX barcode files compared across metadata strata: "
        f"{all_gex_read}"
    )

    print(
        f"No true malformed ATAC records in bounded scan: "
        f"{no_true_malformed}"
    )

    print()

    if (
        archive_complete
        and all_gex_read
        and no_true_malformed
    ):
        print(
            "PHASE F7b VERDICT: ORIGINAL F7 REVIEW FLAGS WERE "
            "AUDIT-DEFINITION ISSUES"
        )

        print()

        print(
            "The archive/sample pairing remains structurally valid."
        )

        print()

        print(
            "Use the best-matching GEX metadata strata reported above "
            "to document the RNA barcode universe."
        )

        if args.full_atac_target_scan:
            print()

            if (
                full_scan_complete
                and full_scan_min_joint >= 0.95
                and full_scan_min_mapped_joint >= 0.95
            ):
                print(
                    "FULL ATAC TARGET VERDICT: PASS — metadata-defined "
                    "joint-QC cells are represented in ATAC fragments."
                )
            else:
                print(
                    "FULL ATAC TARGET VERDICT: REVIEW — at least one sample "
                    "has <95% coverage of the metadata-defined target set."
                )
        else:
            print()

            print(
                "No exhaustive ATAC target-presence claim is made from the "
                "bounded scan."
            )

    else:
        print(
            "PHASE F7b VERDICT: REVIEW REQUIRED"
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

