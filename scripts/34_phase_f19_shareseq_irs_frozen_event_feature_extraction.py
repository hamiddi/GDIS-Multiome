#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
34_phase_f19_shareseq_irs_frozen_event_feature_extraction.py
=============================================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F19
---------
Begin mechanistic follow-up WITHOUT reopening timing inference.

F18 froze the inferential interpretation:
    - localized frozen ATAC-earlier event: bootstrap-supported;
    - broad RNA/ATAC transition-energy alignment: limited under the primary
      circular-shift null.

F19 therefore does NOT perform another timing/null test.

Instead, it extracts RNA and ATAC features from the ALREADY-FROZEN primary
event windows so that later biological annotation can ask:

    Which chromatin-accessibility features change at the frozen ATAC event?

    Which RNA features rise later at the frozen RNA event?

NO NEW EVENT SELECTION
----------------------
Primary windows are taken directly from F16:

    ATAC event:
        primary_w400_s100 ATAC selected-family peak

    RNA event:
        primary_w400_s100 RNA selected-family peak

A PRE-ATAC baseline is defined mechanically from the frozen F14 primary
window family:

    latest complete 400-cell primary window whose stop rank is <=
    the ATAC-event window start rank.

This creates a nonoverlapping predecessor window whenever available.

FROZEN FEATURE CONTRASTS
------------------------
RNA descriptive contrasts:

    A. ATAC-event vs PRE-ATAC baseline
    B. RNA-event  vs ATAC-event

ATAC descriptive contrasts:

    A. ATAC-event vs PRE-ATAC baseline
    B. RNA-event  vs ATAC-event

These are descriptive pseudobulk / accessibility effect sizes.

IMPORTANT PSEUDOREPLICATION SAFEGUARD
-------------------------------------
The SHARE-seq late-anagen cells are not independent biological replicates.

F19 therefore:
    - does NOT calculate cell-level P values;
    - does NOT claim differential-expression significance;
    - does NOT claim differential-accessibility significance;
    - reports effect sizes and feature ranks only.

RNA METRICS
-----------
For every gene:
    - pseudobulk raw count per frozen window;
    - pseudobulk CPM;
    - detected-cell fraction;
    - log2 CPM fold change:
          ATAC event / pre-ATAC
          RNA event / ATAC event

ATAC METRICS
------------
For every peak:
    - accessible-cell count;
    - accessibility fraction;
    - summed matrix count;
    - mean matrix count per cell;
    - accessibility-fraction change:
          ATAC event - pre-ATAC
          RNA event - ATAC event
    - log2 accessibility-ratio effect sizes.

TOP CANDIDATES
--------------
F19 saves complete effect-size tables and descriptive top-200 lists.

RNA top lists:
    - strongest RNA-event increases relative to ATAC-event;
    - strongest ATAC-event RNA increases relative to baseline.

ATAC top lists:
    - strongest accessibility gains at ATAC event vs baseline;
    - strongest later accessibility changes at RNA event vs ATAC event.

No motif enrichment.
No peak-to-gene linking.
No TF-target inference.
Those require a later dedicated annotation phase.

INPUTS
------
data/GSE140203/
    GSM4156608_skin.late.anagen.rna.counts.txt.gz
    GSM4156597_skin.late.anagen.counts.txt.gz
    GSM4156597_skin.late.anagen.barcodes.txt.gz
    GSM4156597_skin.late.anagen.peaks.bed.gz

data/GSE140203/representations_f12/
    f12_irs_branch_cells.tsv.gz

data/GSE140203/representations_f13/
    f13_irs_common_pseudotime.tsv.gz

data/GSE140203/representations_f14/
    f14_common_window_metadata.tsv.gz

data/GSE140203/representations_f16/
    f16_window_specific_pairing.tsv
    f16_manifest.json

data/GSE140203/representations_f17/
    f17_manifest.json

data/GSE140203/representations_f18/
    f18_alignment_summary.tsv
    f18_manifest.json

OUTPUTS
-------
data/GSE140203/representations_f19/
    f19_frozen_event_windows.tsv
    f19_window_cell_membership.tsv.gz
    f19_rna_event_feature_effects.tsv.gz
    f19_rna_top_event_genes.tsv
    f19_atac_event_peak_effects.tsv.gz
    f19_atac_top_event_peaks.tsv
    f19_manifest.json

RUN
---
    python 34_phase_f19_shareseq_irs_frozen_event_feature_extraction.py

DEPENDENCIES
------------
numpy
pandas
"""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
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
F12_DIR = DATA_DIR / "representations_f12"
F13_DIR = DATA_DIR / "representations_f13"
F14_DIR = DATA_DIR / "representations_f14"
F16_DIR = DATA_DIR / "representations_f16"
F17_DIR = DATA_DIR / "representations_f17"
F18_DIR = DATA_DIR / "representations_f18"
F19_DIR = DATA_DIR / "representations_f19"

RNA_COUNTS_FILE = DATA_DIR / "GSM4156608_skin.late.anagen.rna.counts.txt.gz"

ATAC_COUNTS_FILE = DATA_DIR / "GSM4156597_skin.late.anagen.counts.txt.gz"
ATAC_BARCODES_FILE = DATA_DIR / "GSM4156597_skin.late.anagen.barcodes.txt.gz"
ATAC_PEAKS_FILE = DATA_DIR / "GSM4156597_skin.late.anagen.peaks.bed.gz"

F12_CELLS = F12_DIR / "f12_irs_branch_cells.tsv.gz"
F13_CLOCK = F13_DIR / "f13_irs_common_pseudotime.tsv.gz"
F14_WINDOWS = F14_DIR / "f14_common_window_metadata.tsv.gz"

F16_WINDOWS = F16_DIR / "f16_window_specific_pairing.tsv"
F16_MANIFEST = F16_DIR / "f16_manifest.json"

F17_MANIFEST = F17_DIR / "f17_manifest.json"

F18_SUMMARY = F18_DIR / "f18_alignment_summary.tsv"
F18_MANIFEST = F18_DIR / "f18_manifest.json"

OUTPUT_WINDOWS = F19_DIR / "f19_frozen_event_windows.tsv"
OUTPUT_MEMBERSHIP = F19_DIR / "f19_window_cell_membership.tsv.gz"

OUTPUT_RNA_EFFECTS = F19_DIR / "f19_rna_event_feature_effects.tsv.gz"
OUTPUT_RNA_TOP = F19_DIR / "f19_rna_top_event_genes.tsv"

OUTPUT_ATAC_EFFECTS = F19_DIR / "f19_atac_event_peak_effects.tsv.gz"
OUTPUT_ATAC_TOP = F19_DIR / "f19_atac_top_event_peaks.tsv"

OUTPUT_MANIFEST = F19_DIR / "f19_manifest.json"


# =====================================================================
# FROZEN DESIGN
# =====================================================================

BARCODE_COL = "rna.bc"
STATE_COL = "celltype"
PSEUDOTIME_COL = "common_pseudotime"

PRIMARY_WINDOW_CONFIG = "primary_w400_s100"

EXPECTED_WINDOW_SIZE = 400

GROUP_PRE = "pre_ATAC_baseline"
GROUP_ATAC = "ATAC_event"
GROUP_RNA = "RNA_event"

GROUPS = [
    GROUP_PRE,
    GROUP_ATAC,
    GROUP_RNA,
]

TOP_N = 200

RNA_PSEUDOCOUNT_CPM = 0.5
ATAC_PSEUDOCOUNT_FRACTION = 1e-4


# =====================================================================
# DISPLAY / UTILITIES
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


def print_df(df, digits=6):
    if df is None or df.empty:
        print("(empty)")
        return

    with pd.option_context(
        "display.max_rows", 500,
        "display.max_columns", None,
        "display.width", 520,
        "display.max_colwidth", 180,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string())


def utc_now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


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

            if value:
                values.append(
                    value.split(
                        "\t"
                    )[
                        0
                    ]
                )

    return values


# =====================================================================
# INPUTS
# =====================================================================

def require_inputs():
    required = [
        RNA_COUNTS_FILE,
        ATAC_COUNTS_FILE,
        ATAC_BARCODES_FILE,
        ATAC_PEAKS_FILE,
        F12_CELLS,
        F13_CLOCK,
        F14_WINDOWS,
        F16_WINDOWS,
        F16_MANIFEST,
        F17_MANIFEST,
        F18_SUMMARY,
        F18_MANIFEST,
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


def load_frozen_inputs():
    cells = pd.read_csv(
        F12_CELLS,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    clock = pd.read_csv(
        F13_CLOCK,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    windows = pd.read_csv(
        F14_WINDOWS,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    pair_windows = pd.read_csv(
        F16_WINDOWS,
        sep="\t",
        low_memory=False,
    )

    f18_summary = pd.read_csv(
        F18_SUMMARY,
        sep="\t",
        low_memory=False,
    )

    if cells[
        BARCODE_COL
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate F12 barcodes."
        )

    if clock[
        BARCODE_COL
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate F13 barcodes."
        )

    clock_indexed = clock.set_index(
        BARCODE_COL,
        drop=False,
    )

    missing = [
        barcode
        for barcode in cells[
            BARCODE_COL
        ].astype(str)
        if barcode not in clock_indexed.index
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} F12 cells missing from F13 clock."
        )

    aligned_clock = clock_indexed.loc[
        cells[
            BARCODE_COL
        ].astype(str)
    ].reset_index(
        drop=True
    )

    pseudotime = pd.to_numeric(
        aligned_clock[
            PSEUDOTIME_COL
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if not np.isfinite(
        pseudotime
    ).all():
        raise RuntimeError(
            "Frozen pseudotime contains NaN/Inf."
        )

    primary_windows = windows.loc[
        windows[
            "window_config"
        ].astype(str)
        == PRIMARY_WINDOW_CONFIG
    ].sort_values(
        "window_index"
    ).reset_index(
        drop=True
    )

    if primary_windows.empty:
        raise RuntimeError(
            "No primary F14 windows found."
        )

    primary_pair = pair_windows.loc[
        pair_windows[
            "window_config"
        ].astype(str)
        == PRIMARY_WINDOW_CONFIG
    ]

    if len(
        primary_pair
    ) != 1:
        raise RuntimeError(
            "Could not identify unique F16 primary pairing row."
        )

    if len(
        f18_summary
    ) != 1:
        raise RuntimeError(
            "F18 summary must contain exactly one row."
        )

    return {
        "cells": cells,
        "pseudotime": pseudotime,
        "primary_windows": primary_windows,
        "primary_pair": primary_pair.iloc[
            0
        ],
        "f18_summary": f18_summary.iloc[
            0
        ],
    }


# =====================================================================
# FREEZE EVENT WINDOWS
# =====================================================================

def locate_window_by_parameter(
    windows,
    parameter,
    label,
):
    distance = np.abs(
        windows[
            "parameter"
        ].to_numpy(
            dtype=float
        )
        - float(
            parameter
        )
    )

    index = int(
        np.argmin(
            distance
        )
    )

    if distance[
        index
    ] > 1e-10:
        raise RuntimeError(
            f"{label}: no exact F14 primary window matches "
            f"parameter {parameter:.12f}; nearest distance "
            f"{distance[index]:.3e}."
        )

    return windows.iloc[
        index
    ].copy()


def define_frozen_windows(
    primary_windows,
    pair_row,
):
    atac_event = locate_window_by_parameter(
        primary_windows,
        float(
            pair_row[
                "ATAC_peak_parameter"
            ]
        ),
        "ATAC event",
    )

    rna_event = locate_window_by_parameter(
        primary_windows,
        float(
            pair_row[
                "RNA_peak_parameter"
            ]
        ),
        "RNA event",
    )

    eligible_pre = primary_windows.loc[
        primary_windows[
            "stop_rank_exclusive"
        ].astype(int)
        <= int(
            atac_event[
                "start_rank_zero_based"
            ]
        )
    ].copy()

    if eligible_pre.empty:
        raise RuntimeError(
            "No nonoverlapping primary window exists before ATAC event."
        )

    pre_event = eligible_pre.sort_values(
        "stop_rank_exclusive"
    ).iloc[
        -1
    ].copy()

    rows = []

    for group_name, row in [
        (
            GROUP_PRE,
            pre_event,
        ),
        (
            GROUP_ATAC,
            atac_event,
        ),
        (
            GROUP_RNA,
            rna_event,
        ),
    ]:
        rows.append(
            {
                "group": group_name,
                "window_index": int(
                    row[
                        "window_index"
                    ]
                ),
                "start_rank_zero_based": int(
                    row[
                        "start_rank_zero_based"
                    ]
                ),
                "stop_rank_exclusive": int(
                    row[
                        "stop_rank_exclusive"
                    ]
                ),
                "n_cells": int(
                    row[
                        "n_cells"
                    ]
                ),
                "parameter": float(
                    row[
                        "parameter"
                    ]
                ),
                "pseudotime_min": float(
                    row[
                        "pseudotime_min"
                    ]
                ),
                "pseudotime_max": float(
                    row[
                        "pseudotime_max"
                    ]
                ),
                "TAC1_fraction": float(
                    row[
                        "TAC1_fraction"
                    ]
                ),
                "TAC2_fraction": float(
                    row[
                        "TAC2_fraction"
                    ]
                ),
                "IRS_fraction": float(
                    row[
                        "IRS_fraction"
                    ]
                ),
                "dominant_state": str(
                    row[
                        "dominant_state"
                    ]
                ),
            }
        )

    frozen = pd.DataFrame(
        rows
    )

    return frozen


def construct_membership(
    cells,
    pseudotime,
    frozen_windows,
):
    order = np.argsort(
        pseudotime,
        kind="mergesort",
    )

    cells_sorted = cells.iloc[
        order
    ].copy().reset_index(
        drop=True
    )

    pt_sorted = pseudotime[
        order
    ]

    membership_rows = []

    group_sets = {}

    for row in frozen_windows.itertuples(
        index=False
    ):
        start = int(
            row.start_rank_zero_based
        )

        stop = int(
            row.stop_rank_exclusive
        )

        subset = cells_sorted.iloc[
            start:
            stop
        ].copy()

        if len(
            subset
        ) != EXPECTED_WINDOW_SIZE:
            raise RuntimeError(
                f"{row.group}: expected {EXPECTED_WINDOW_SIZE} cells, "
                f"found {len(subset)}."
            )

        subset[
            "group"
        ] = row.group

        subset[
            "rank_within_common_order"
        ] = np.arange(
            start,
            stop,
            dtype=int,
        )

        subset[
            PSEUDOTIME_COL
        ] = pt_sorted[
            start:
            stop
        ]

        membership_rows.append(
            subset
        )

        group_sets[
            row.group
        ] = set(
            subset[
                BARCODE_COL
            ].astype(str)
        )

    membership = pd.concat(
        membership_rows,
        ignore_index=True,
    )

    return (
        membership,
        group_sets,
    )


# =====================================================================
# RNA FEATURE EXTRACTION
# =====================================================================

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
            "RNA file header failed TAB parsing."
        )

    barcodes = [
        normalize_rna_matrix_barcode(
            value
        )
        for value in fields[
            1:
        ]
    ]

    if len(
        barcodes
    ) != len(
        set(
            barcodes
        )
    ):
        raise RuntimeError(
            "Normalized RNA matrix barcodes are not unique."
        )

    return (
        fields[
            0
        ],
        barcodes,
    )


def extract_rna_features(
    membership,
):
    (
        first_field,
        matrix_barcodes,
    ) = parse_rna_header()

    barcode_to_col = {
        barcode: index
        for index, barcode in enumerate(
            matrix_barcodes
        )
    }

    group_columns = {}

    for group in GROUPS:
        barcodes = membership.loc[
            membership[
                "group"
            ]
            == group,
            BARCODE_COL,
        ].astype(str).tolist()

        missing = [
            barcode
            for barcode in barcodes
            if barcode not in barcode_to_col
        ]

        if missing:
            raise RuntimeError(
                f"{group}: {len(missing)} cells missing from RNA matrix."
            )

        group_columns[
            group
        ] = np.asarray(
            [
                barcode_to_col[
                    barcode
                ]
                for barcode in barcodes
            ],
            dtype=np.int64,
        )

    genes = []

    sums = {
        group: []
        for group in GROUPS
    }

    detected = {
        group: []
        for group in GROUPS
    }

    group_library_totals = {
        group: 0.0
        for group in GROUPS
    }

    parsed_genes = 0

    with gzip.open(
        RNA_COUNTS_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        handle.readline()

        for line_text in handle:
            if not line_text.strip():
                continue

            if "\t" not in line_text:
                raise RuntimeError(
                    f"RNA row {parsed_genes + 1} has no TAB."
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
                matrix_barcodes
            ):
                raise RuntimeError(
                    f"RNA row-width mismatch at gene {gene}: "
                    f"{len(values)} vs {len(matrix_barcodes)}."
                )

            genes.append(
                gene
            )

            for group in GROUPS:
                selected = values[
                    group_columns[
                        group
                    ]
                ]

                total = float(
                    np.sum(
                        selected
                    )
                )

                detected_n = int(
                    np.sum(
                        selected
                        > 0
                    )
                )

                sums[
                    group
                ].append(
                    total
                )

                detected[
                    group
                ].append(
                    detected_n
                )

                group_library_totals[
                    group
                ] += total

            parsed_genes += 1

            if parsed_genes % 4000 == 0:
                print(
                    f"  RNA streamed {parsed_genes:,} genes",
                    flush=True,
                )

    table = pd.DataFrame(
        {
            "gene": genes,
        }
    )

    for group in GROUPS:
        raw = np.asarray(
            sums[
                group
            ],
            dtype=float,
        )

        det = np.asarray(
            detected[
                group
            ],
            dtype=float,
        )

        library_total = float(
            group_library_totals[
                group
            ]
        )

        if library_total <= 0:
            raise RuntimeError(
                f"{group}: zero total RNA library."
            )

        cpm = (
            raw
            / library_total
            * 1_000_000.0
        )

        table[
            f"{group}_raw_count"
        ] = raw

        table[
            f"{group}_CPM"
        ] = cpm

        table[
            f"{group}_detected_fraction"
        ] = (
            det
            / EXPECTED_WINDOW_SIZE
        )

    table[
        "log2FC_ATACevent_vs_pre_CPM"
    ] = np.log2(
        (
            table[
                f"{GROUP_ATAC}_CPM"
            ]
            + RNA_PSEUDOCOUNT_CPM
        )
        / (
            table[
                f"{GROUP_PRE}_CPM"
            ]
            + RNA_PSEUDOCOUNT_CPM
        )
    )

    table[
        "log2FC_RNAevent_vs_ATACevent_CPM"
    ] = np.log2(
        (
            table[
                f"{GROUP_RNA}_CPM"
            ]
            + RNA_PSEUDOCOUNT_CPM
        )
        / (
            table[
                f"{GROUP_ATAC}_CPM"
            ]
            + RNA_PSEUDOCOUNT_CPM
        )
    )

    table[
        "delta_detected_fraction_ATACevent_vs_pre"
    ] = (
        table[
            f"{GROUP_ATAC}_detected_fraction"
        ]
        - table[
            f"{GROUP_PRE}_detected_fraction"
        ]
    )

    table[
        "delta_detected_fraction_RNAevent_vs_ATACevent"
    ] = (
        table[
            f"{GROUP_RNA}_detected_fraction"
        ]
        - table[
            f"{GROUP_ATAC}_detected_fraction"
        ]
    )

    # Descriptive top lists.
    top_later = (
        table.loc[
            table[
                f"{GROUP_RNA}_CPM"
            ]
            >= 1.0
        ]
        .sort_values(
            [
                "log2FC_RNAevent_vs_ATACevent_CPM",
                f"{GROUP_RNA}_CPM",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            TOP_N
        )
        .copy()
    )

    top_later.insert(
        0,
        "candidate_class",
        "RNA_event_increase_vs_ATAC_event",
    )

    top_atac_stage = (
        table.loc[
            table[
                f"{GROUP_ATAC}_CPM"
            ]
            >= 1.0
        ]
        .sort_values(
            [
                "log2FC_ATACevent_vs_pre_CPM",
                f"{GROUP_ATAC}_CPM",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            TOP_N
        )
        .copy()
    )

    top_atac_stage.insert(
        0,
        "candidate_class",
        "ATAC_event_RNA_increase_vs_pre",
    )

    top = pd.concat(
        [
            top_later,
            top_atac_stage,
        ],
        ignore_index=True,
    )

    return (
        table,
        top,
        {
            group: float(
                group_library_totals[
                    group
                ]
            )
            for group in GROUPS
        },
    )


# =====================================================================
# ATAC FEATURE EXTRACTION
# =====================================================================

def parse_matrixmarket_dimensions(
    handle,
):
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
        for line_text in handle:
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
                    "Malformed ATAC peak row."
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


def extract_atac_features(
    membership,
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

    barcode_to_col = {
        barcode: index + 1
        for index, barcode in enumerate(
            atac_barcodes
        )
    }

    # Each frozen window is nonoverlapping by construction in F19.
    # Map MatrixMarket one-based column -> group index.
    col_to_group = np.full(
        len(
            atac_barcodes
        )
        + 1,
        -1,
        dtype=np.int8,
    )

    for group_index, group in enumerate(
        GROUPS
    ):
        group_barcodes = membership.loc[
            membership[
                "group"
            ]
            == group,
            BARCODE_COL,
        ].astype(str).tolist()

        missing = [
            barcode
            for barcode in group_barcodes
            if barcode not in barcode_to_col
        ]

        if missing:
            raise RuntimeError(
                f"{group}: {len(missing)} cells missing from ATAC matrix."
            )

        for barcode in group_barcodes:
            column = barcode_to_col[
                barcode
            ]

            if col_to_group[
                column
            ] != -1:
                raise RuntimeError(
                    "F19 frozen windows overlap in ATAC cell membership."
                )

            col_to_group[
                column
            ] = group_index

    with gzip.open(
        ATAC_COUNTS_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        (
            n_peaks,
            n_cells,
            n_nnz,
        ) = parse_matrixmarket_dimensions(
            handle
        )

        if n_cells != len(
            atac_barcodes
        ):
            raise RuntimeError(
                "ATAC MatrixMarket/barcode count mismatch."
            )

        accessible = np.zeros(
            (
                len(
                    GROUPS
                ),
                n_peaks,
            ),
            dtype=np.int32,
        )

        summed_counts = np.zeros(
            (
                len(
                    GROUPS
                ),
                n_peaks,
            ),
            dtype=np.float64,
        )

        parsed = 0

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
                raise RuntimeError(
                    f"Malformed MatrixMarket row at nnz {parsed + 1}."
                )

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

            parsed += 1

            group_index = col_to_group[
                col_index
            ]

            if group_index < 0:
                continue

            peak_index = (
                row_index
                - 1
            )

            accessible[
                group_index,
                peak_index
            ] += 1

            summed_counts[
                group_index,
                peak_index
            ] += value

            if parsed % 20_000_000 == 0:
                print(
                    f"  ATAC parsed {parsed:,} / {n_nnz:,} entries",
                    flush=True,
                )

    if parsed != n_nnz:
        raise RuntimeError(
            f"ATAC parsed nnz mismatch: {parsed:,} vs {n_nnz:,}."
        )

    peaks = read_peak_table()

    if len(
        peaks
    ) != n_peaks:
        raise RuntimeError(
            "ATAC peak BED row count does not match matrix rows."
        )

    table = pd.DataFrame(
        {
            "peak_index_zero_based": np.arange(
                n_peaks,
                dtype=int,
            ),
            "chrom": [
                value[
                    0
                ]
                for value in peaks
            ],
            "start": [
                value[
                    1
                ]
                for value in peaks
            ],
            "end": [
                value[
                    2
                ]
                for value in peaks
            ],
        }
    )

    for group_index, group in enumerate(
        GROUPS
    ):
        access_fraction = (
            accessible[
                group_index,
                :
            ].astype(
                float
            )
            / EXPECTED_WINDOW_SIZE
        )

        mean_count = (
            summed_counts[
                group_index,
                :
            ]
            / EXPECTED_WINDOW_SIZE
        )

        table[
            f"{group}_accessible_cells"
        ] = accessible[
            group_index,
            :
        ]

        table[
            f"{group}_accessibility_fraction"
        ] = access_fraction

        table[
            f"{group}_summed_count"
        ] = summed_counts[
            group_index,
            :
        ]

        table[
            f"{group}_mean_count_per_cell"
        ] = mean_count

    table[
        "delta_accessibility_ATACevent_vs_pre"
    ] = (
        table[
            f"{GROUP_ATAC}_accessibility_fraction"
        ]
        - table[
            f"{GROUP_PRE}_accessibility_fraction"
        ]
    )

    table[
        "delta_accessibility_RNAevent_vs_ATACevent"
    ] = (
        table[
            f"{GROUP_RNA}_accessibility_fraction"
        ]
        - table[
            f"{GROUP_ATAC}_accessibility_fraction"
        ]
    )

    table[
        "log2ratio_accessibility_ATACevent_vs_pre"
    ] = np.log2(
        (
            table[
                f"{GROUP_ATAC}_accessibility_fraction"
            ]
            + ATAC_PSEUDOCOUNT_FRACTION
        )
        / (
            table[
                f"{GROUP_PRE}_accessibility_fraction"
            ]
            + ATAC_PSEUDOCOUNT_FRACTION
        )
    )

    table[
        "log2ratio_accessibility_RNAevent_vs_ATACevent"
    ] = np.log2(
        (
            table[
                f"{GROUP_RNA}_accessibility_fraction"
            ]
            + ATAC_PSEUDOCOUNT_FRACTION
        )
        / (
            table[
                f"{GROUP_ATAC}_accessibility_fraction"
            ]
            + ATAC_PSEUDOCOUNT_FRACTION
        )
    )

    top_opening = (
        table.sort_values(
            [
                "delta_accessibility_ATACevent_vs_pre",
                f"{GROUP_ATAC}_accessibility_fraction",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            TOP_N
        )
        .copy()
    )

    top_opening.insert(
        0,
        "candidate_class",
        "ATAC_event_accessibility_gain_vs_pre",
    )

    top_later = (
        table.sort_values(
            [
                "delta_accessibility_RNAevent_vs_ATACevent",
                f"{GROUP_RNA}_accessibility_fraction",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            TOP_N
        )
        .copy()
    )

    top_later.insert(
        0,
        "candidate_class",
        "RNA_event_later_accessibility_change",
    )

    top = pd.concat(
        [
            top_opening,
            top_later,
        ],
        ignore_index=True,
    )

    return (
        table,
        top,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F19 — FROZEN EVENT-WINDOW RNA/ATAC FEATURE EXTRACTION"
    )

    print(
        "Timing inference is CLOSED."
    )

    print()

    print(
        "F19 performs descriptive mechanistic feature extraction only."
    )

    print()

    print(
        "No new event selection."
    )

    print(
        "No P values."
    )

    print(
        "No motif enrichment."
    )

    print(
        "No peak-to-gene linking."
    )

    require_inputs()

    F19_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_frozen_inputs()

    cells = data[
        "cells"
    ]

    pseudotime = data[
        "pseudotime"
    ]

    primary_windows = data[
        "primary_windows"
    ]

    pair_row = data[
        "primary_pair"
    ]

    f18_summary = data[
        "f18_summary"
    ]

    # -----------------------------------------------------------------
    # Freeze event windows.
    # -----------------------------------------------------------------

    section(
        "1. FREEZE MECHANISTIC WINDOWS FROM THE ALREADY-FROZEN F16 EVENTS"
    )

    frozen_windows = define_frozen_windows(
        primary_windows,
        pair_row,
    )

    print_df(
        frozen_windows.set_index(
            "group"
        ),
        digits=6,
    )

    frozen_windows.to_csv(
        OUTPUT_WINDOWS,
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Membership.
    # -----------------------------------------------------------------

    section(
        "2. FREEZE EXACT CELL MEMBERSHIP OF THE THREE NONOVERLAPPING WINDOWS"
    )

    (
        membership,
        group_sets,
    ) = construct_membership(
        cells,
        pseudotime,
        frozen_windows,
    )

    membership.to_csv(
        OUTPUT_MEMBERSHIP,
        sep="\t",
        index=False,
        compression="gzip",
    )

    overlap_rows = []

    for i, group_a in enumerate(
        GROUPS
    ):
        for group_b in GROUPS[
            i
            + 1:
        ]:
            overlap_rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "shared_cells": len(
                        group_sets[
                            group_a
                        ]
                        & group_sets[
                            group_b
                        ]
                    ),
                }
            )

    overlap_df = pd.DataFrame(
        overlap_rows
    )

    subsection(
        "Pairwise cell overlap"
    )

    print_df(
        overlap_df.set_index(
            [
                "group_a",
                "group_b",
            ]
        ),
        digits=0,
    )

    subsection(
        "State composition by frozen mechanistic window"
    )

    state_composition = (
        membership.groupby(
            [
                "group",
                STATE_COL,
            ]
        )
        .size()
        .rename(
            "n_cells"
        )
        .reset_index()
    )

    state_composition[
        "fraction"
    ] = (
        state_composition[
            "n_cells"
        ]
        / EXPECTED_WINDOW_SIZE
    )

    print_df(
        state_composition.set_index(
            [
                "group",
                STATE_COL,
            ]
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # RNA.
    # -----------------------------------------------------------------

    section(
        "3. STREAM RAW RNA COUNTS AND CALCULATE DESCRIPTIVE EVENT EFFECT SIZES"
    )

    (
        rna_effects,
        rna_top,
        rna_library_totals,
    ) = extract_rna_features(
        membership
    )

    rna_effects.to_csv(
        OUTPUT_RNA_EFFECTS,
        sep="\t",
        index=False,
        compression="gzip",
    )

    rna_top.to_csv(
        OUTPUT_RNA_TOP,
        sep="\t",
        index=False,
    )

    subsection(
        "Top 20 RNA-event increases relative to frozen ATAC-event window"
    )

    top_rna_display = rna_top.loc[
        rna_top[
            "candidate_class"
        ]
        == "RNA_event_increase_vs_ATAC_event"
    ].head(
        20
    )

    print_df(
        top_rna_display[
            [
                "gene",
                f"{GROUP_ATAC}_CPM",
                f"{GROUP_RNA}_CPM",
                "log2FC_RNAevent_vs_ATACevent_CPM",
                f"{GROUP_ATAC}_detected_fraction",
                f"{GROUP_RNA}_detected_fraction",
                "delta_detected_fraction_RNAevent_vs_ATACevent",
            ]
        ].set_index(
            "gene"
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # ATAC.
    # -----------------------------------------------------------------

    section(
        "4. STREAM RAW ATAC MATRIX AND CALCULATE DESCRIPTIVE EVENT EFFECT SIZES"
    )

    (
        atac_effects,
        atac_top,
    ) = extract_atac_features(
        membership
    )

    atac_effects.to_csv(
        OUTPUT_ATAC_EFFECTS,
        sep="\t",
        index=False,
        compression="gzip",
    )

    atac_top.to_csv(
        OUTPUT_ATAC_TOP,
        sep="\t",
        index=False,
    )

    subsection(
        "Top 20 accessibility gains at frozen ATAC event vs pre-ATAC baseline"
    )

    top_atac_display = atac_top.loc[
        atac_top[
            "candidate_class"
        ]
        == "ATAC_event_accessibility_gain_vs_pre"
    ].head(
        20
    )

    print_df(
        top_atac_display[
            [
                "chrom",
                "start",
                "end",
                f"{GROUP_PRE}_accessibility_fraction",
                f"{GROUP_ATAC}_accessibility_fraction",
                "delta_accessibility_ATACevent_vs_pre",
                "log2ratio_accessibility_ATACevent_vs_pre",
            ]
        ].set_index(
            [
                "chrom",
                "start",
                "end",
            ]
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Safeguards.
    # -----------------------------------------------------------------

    section(
        "5. PHASE F19 MECHANISTIC-EXTRACTION SAFEGUARDS"
    )

    all_windows_400 = bool(
        (
            frozen_windows[
                "n_cells"
            ]
            == EXPECTED_WINDOW_SIZE
        ).all()
    )

    nonoverlapping = bool(
        (
            overlap_df[
                "shared_cells"
            ]
            == 0
        ).all()
    )

    atac_before_rna = bool(
        frozen_windows.set_index(
            "group"
        ).loc[
            GROUP_ATAC,
            "parameter",
        ]
        <
        frozen_windows.set_index(
            "group"
        ).loc[
            GROUP_RNA,
            "parameter",
        ]
    )

    baseline_before_atac = bool(
        frozen_windows.set_index(
            "group"
        ).loc[
            GROUP_PRE,
            "stop_rank_exclusive",
        ]
        <=
        frozen_windows.set_index(
            "group"
        ).loc[
            GROUP_ATAC,
            "start_rank_zero_based",
        ]
    )

    f18_class = str(
        f18_summary[
            "circular_alignment_evidence_class"
        ]
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    "Frozen pre/ATAC/RNA mechanistic windows all contain "
                    "exactly 400 cells"
                ),
                "pass": all_windows_400,
            },
            {
                "criterion": (
                    "Frozen mechanistic windows are cell-nonoverlapping"
                ),
                "pass": nonoverlapping,
            },
            {
                "criterion": (
                    "Pre-ATAC baseline ends before/equal ATAC-event start"
                ),
                "pass": baseline_before_atac,
            },
            {
                "criterion": (
                    "Frozen ATAC event precedes frozen RNA event"
                ),
                "pass": atac_before_rna,
            },
            {
                "criterion": (
                    "RNA complete effect-size table created"
                ),
                "pass": (
                    len(
                        rna_effects
                    )
                    > 0
                ),
            },
            {
                "criterion": (
                    "ATAC complete effect-size table has 344,592 peaks"
                ),
                "pass": (
                    len(
                        atac_effects
                    )
                    == 344_592
                ),
            },
            {
                "criterion": (
                    "F18 primary alignment result retained without reinterpretation"
                ),
                "pass": (
                    f18_class
                    == "LIMITED_OR_NONE"
                ),
            },
            {
                "criterion": (
                    "No statistical significance tests performed on cell-level "
                    "feature contrasts"
                ),
                "pass": True,
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
        "6. SAVE F19 MANIFEST"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F19",
        "created_utc": utc_now_iso(),
        "purpose": (
            "Descriptive feature extraction from frozen event windows after "
            "timing inference was closed."
        ),
        "frozen_windows": (
            frozen_windows.to_dict(
                orient="records"
            )
        ),
        "group_state_composition": (
            state_composition.to_dict(
                orient="records"
            )
        ),
        "RNA": {
            "pseudobulk_normalization": "CPM",
            "CPM_pseudocount_for_log2FC": RNA_PSEUDOCOUNT_CPM,
            "library_totals": rna_library_totals,
            "cell_level_p_values": False,
            "output_effects": str(
                OUTPUT_RNA_EFFECTS
            ),
            "output_top": str(
                OUTPUT_RNA_TOP
            ),
        },
        "ATAC": {
            "primary_metric": "binary accessibility fraction",
            "fraction_pseudocount_for_log2ratio": (
                ATAC_PSEUDOCOUNT_FRACTION
            ),
            "cell_level_p_values": False,
            "output_effects": str(
                OUTPUT_ATAC_EFFECTS
            ),
            "output_top": str(
                OUTPUT_ATAC_TOP
            ),
        },
        "F18_primary_alignment_evidence_class": f18_class,
        "checks": checks.to_dict(
            orient="records"
        ),
        "outputs": {
            "windows": {
                "file": str(
                    OUTPUT_WINDOWS
                ),
                "sha256": sha256_file(
                    OUTPUT_WINDOWS
                ),
            },
            "membership": {
                "file": str(
                    OUTPUT_MEMBERSHIP
                ),
                "sha256": sha256_file(
                    OUTPUT_MEMBERSHIP
                ),
            },
            "rna_effects": {
                "file": str(
                    OUTPUT_RNA_EFFECTS
                ),
                "sha256": sha256_file(
                    OUTPUT_RNA_EFFECTS
                ),
            },
            "rna_top": {
                "file": str(
                    OUTPUT_RNA_TOP
                ),
                "sha256": sha256_file(
                    OUTPUT_RNA_TOP
                ),
            },
            "atac_effects": {
                "file": str(
                    OUTPUT_ATAC_EFFECTS
                ),
                "sha256": sha256_file(
                    OUTPUT_ATAC_EFFECTS
                ),
            },
            "atac_top": {
                "file": str(
                    OUTPUT_ATAC_TOP
                ),
                "sha256": sha256_file(
                    OUTPUT_ATAC_TOP
                ),
            },
        },
        "guardrails": {
            "timing_inference_reopened": False,
            "new_event_selected": False,
            "new_null_test_performed": False,
            "gdis_recomputed": False,
            "pseudotime_recomputed": False,
            "cell_level_differential_test_p_values": False,
            "motif_analysis_performed": False,
            "peak_to_gene_linking_performed": False,
            "priming_claim_made": False,
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
        OUTPUT_WINDOWS,
        OUTPUT_MEMBERSHIP,
        OUTPUT_RNA_EFFECTS,
        OUTPUT_RNA_TOP,
        OUTPUT_ATAC_EFFECTS,
        OUTPUT_ATAC_TOP,
        OUTPUT_MANIFEST,
    ]:
        print(
            path
        )

    # -----------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------

    section(
        "7. PHASE F19 DECISION"
    )

    if all_pass:
        print(
            "PHASE F19 VERDICT: GO — FROZEN EVENT-WINDOW RNA/ATAC "
            "MECHANISTIC CANDIDATES EXTRACTED"
        )

        print()

        print(
            "Timing inference remains frozen."
        )

        print()

        print(
            "Next phase may annotate the frozen ATAC-opening candidates "
            "with nearby genes / regulatory context and test whether those "
            "annotations connect biologically to later RNA-event genes."
        )

    else:
        print(
            "PHASE F19 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not proceed to motif or peak-to-gene annotation until "
            "failed extraction safeguards are understood."
        )

    print()

    print(
        "No new timing analysis was performed."
    )

    print(
        "No cell-level significance P values were calculated."
    )

    print(
        "No priming claim was made."
    )

    line("=")


if __name__ == "__main__":
    main()

