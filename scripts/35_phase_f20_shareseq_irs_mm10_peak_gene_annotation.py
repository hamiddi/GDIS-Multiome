#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
35_phase_f20_shareseq_irs_mm10_peak_gene_annotation.py
=======================================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F20
---------
Annotate the already-frozen ATAC-opening candidates to source-compatible mm10
RefSeq transcription start sites (TSSs), and construct prevalence-aware RNA
candidate lists.

F19 TIMING / WINDOWS REMAIN FROZEN
----------------------------------
F20 does NOT:
    - reopen timing inference;
    - change the ATAC or RNA event windows;
    - calculate GDIS;
    - calculate CMIL;
    - perform another null test;
    - perform cell-level DE/DA significance testing.

REFERENCE GENOME / ANNOTATION
-----------------------------
GSE140203 mouse samples were processed against mm10.

For source compatibility, F20 uses the RefSeq BED distributed by the original
SHARE-seq alignment repository:

    https://raw.githubusercontent.com/masai1116/
        SHARE-seq-alignment/main/mm10.UCSC_RefSeq.bed

To translate RefSeq transcript accessions to gene symbols when needed, F20
also uses UCSC mm10 refFlat:

    https://hgdownload.soe.ucsc.edu/
        goldenPath/mm10/database/refFlat.txt.gz

Both files are cached under:

    data/GSE140203/reference_f20/

If your HPC blocks outbound HTTPS, download these two files manually to that
directory and rerun F20.  Existing files are never overwritten.

NEAREST-TSS ANNOTATION
----------------------
For every F19 ATAC peak:
    - peak midpoint;
    - nearest mm10 RefSeq TSS;
    - nearest RefSeq label/transcript;
    - gene symbol when resolvable;
    - strand;
    - signed midpoint-to-TSS distance;
    - absolute distance;
    - promoter/proximal/distal category:

        promoter_2kb : abs(distance) <= 2,000 bp
        proximal_10kb: abs(distance) <= 10,000 bp
        proximal_50kb: abs(distance) <= 50,000 bp
        distal       : > 50,000 bp

This is PROXIMITY ANNOTATION ONLY.
Nearest-TSS assignment is not treated as proof of regulation.

PREVALENCE-AWARE RNA CANDIDATES
-------------------------------
F19 raw fold-change ranking included some genes detected in only a handful of
cells.  F20 therefore adds a descriptive prevalence gate.

Later RNA-event candidate:
    RNA_event detected fraction >= 0.05
    RNA_event CPM >= 2
    log2FC(RNA event / ATAC event) >= 0.5
    delta detected fraction >= 0.02

Early/ATAC-stage RNA candidate:
    ATAC_event detected fraction >= 0.05
    ATAC_event CPM >= 2
    log2FC(ATAC event / pre-ATAC) >= 0.5
    delta detected fraction >= 0.02

These are effect-size / prevalence filters, NOT statistical significance tests.

ATAC-OPENING CANDIDATES
-----------------------
A descriptive opening candidate must satisfy:

    delta accessibility (ATAC event - pre-ATAC) >= 0.05
    ATAC-event accessibility fraction >= 0.05

Candidates are ranked by:
    1. larger accessibility gain
    2. larger ATAC-event accessibility fraction
    3. larger log2 accessibility ratio

CROSS-MODAL PROXIMITY CANDIDATES
--------------------------------
F20 reports genes for which:
    - at least one ATAC-opening candidate has that gene as nearest TSS; AND
    - the same gene passes the prevalence-aware later RNA-event filter.

This is called:
    proximity-supported cross-modal candidate

It is NOT called:
    regulatory target
    causal link
    chromatin priming mechanism

Those require later motif / TF / peak-to-gene analyses.

INPUTS
------
data/GSE140203/representations_f19/
    f19_rna_event_feature_effects.tsv.gz
    f19_atac_event_peak_effects.tsv.gz
    f19_frozen_event_windows.tsv
    f19_manifest.json

OUTPUTS
-------
data/GSE140203/reference_f20/
    mm10.UCSC_RefSeq.bed
    refFlat.txt.gz

data/GSE140203/representations_f20/
    f20_reference_annotation_audit.tsv
    f20_all_atac_peaks_nearest_tss.tsv.gz
    f20_atac_opening_candidates_annotated.tsv
    f20_rna_prevalence_filtered_candidates.tsv
    f20_cross_modal_proximity_candidates.tsv
    f20_manifest.json

RUN
---
    python 35_phase_f20_shareseq_irs_mm10_peak_gene_annotation.py

DEPENDENCIES
------------
numpy
pandas
Python standard library only otherwise.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE140203"

DATA_DIR = Path("data") / ACCESSION
F19_DIR = DATA_DIR / "representations_f19"
REF_DIR = DATA_DIR / "reference_f20"
F20_DIR = DATA_DIR / "representations_f20"

F19_RNA = F19_DIR / "f19_rna_event_feature_effects.tsv.gz"
F19_ATAC = F19_DIR / "f19_atac_event_peak_effects.tsv.gz"
F19_WINDOWS = F19_DIR / "f19_frozen_event_windows.tsv"
F19_MANIFEST = F19_DIR / "f19_manifest.json"

REFSEQ_BED = REF_DIR / "mm10.UCSC_RefSeq.bed"
REFFLAT_GZ = REF_DIR / "refFlat.txt.gz"

OUTPUT_REFERENCE_AUDIT = F20_DIR / "f20_reference_annotation_audit.tsv"
OUTPUT_ALL_PEAKS = F20_DIR / "f20_all_atac_peaks_nearest_tss.tsv.gz"
OUTPUT_ATAC_CANDIDATES = F20_DIR / "f20_atac_opening_candidates_annotated.tsv"
OUTPUT_RNA_CANDIDATES = F20_DIR / "f20_rna_prevalence_filtered_candidates.tsv"
OUTPUT_CROSS_MODAL = F20_DIR / "f20_cross_modal_proximity_candidates.tsv"
OUTPUT_MANIFEST = F20_DIR / "f20_manifest.json"


# =====================================================================
# FROZEN REFERENCE URLS
# =====================================================================

REFSEQ_BED_URL = (
    "https://raw.githubusercontent.com/"
    "masai1116/SHARE-seq-alignment/main/mm10.UCSC_RefSeq.bed"
)

REFFLAT_URL = (
    "https://hgdownload.soe.ucsc.edu/"
    "goldenPath/mm10/database/refFlat.txt.gz"
)


# =====================================================================
# FROZEN DESCRIPTIVE FILTERS
# =====================================================================

RNA_MIN_DETECTED_FRACTION = 0.05
RNA_MIN_CPM = 2.0
RNA_MIN_LOG2FC = 0.50
RNA_MIN_DELTA_DETECTED = 0.02

ATAC_MIN_ACCESSIBILITY_GAIN = 0.05
ATAC_MIN_EVENT_ACCESSIBILITY = 0.05

TOP_ATAC_DISPLAY = 200

PROMOTER_BP = 2_000
PROXIMAL_10KB = 10_000
PROXIMAL_50KB = 50_000


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

    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def download_if_missing(
    url,
    path,
):
    if path.exists():
        print(
            f"Using existing reference: {path}"
        )
        return False

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".part"
    )

    temporary.unlink(
        missing_ok=True
    )

    print(
        f"Downloading:\n  {url}\n  -> {path}"
    )

    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "GDIS-Multiome-F20/1.0 "
                    "(source-compatible annotation acquisition)"
                )
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(
                response,
                output,
                length=1024 * 1024,
            )

        if temporary.stat().st_size < 10_000:
            raise RuntimeError(
                f"Downloaded reference is unexpectedly small: "
                f"{temporary.stat().st_size} bytes"
            )

        temporary.replace(
            path
        )

    except Exception as exc:
        temporary.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "\nCould not download required reference annotation.\n\n"
            f"URL:\n  {url}\n\n"
            f"Expected local path:\n  {path}\n\n"
            "If outbound HTTPS is blocked on the HPC, download the file "
            "manually to the expected path and rerun F20.\n\n"
            f"Original error: {exc}"
        ) from exc

    return True


# =====================================================================
# INPUT VALIDATION
# =====================================================================

def require_inputs():
    required = [
        F19_RNA,
        F19_ATAC,
        F19_WINDOWS,
        F19_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required F19 input(s):\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )


# =====================================================================
# REFERENCE PARSING
# =====================================================================

def normalize_chromosome(
    chrom,
):
    chrom = str(
        chrom
    ).strip()

    if not chrom:
        return chrom

    if chrom.startswith(
        "chr"
    ):
        return chrom

    return (
        "chr"
        + chrom
    )


def read_refflat():
    """
    UCSC refFlat columns:
        geneName
        name (transcript accession)
        chrom
        strand
        txStart
        txEnd
        cdsStart
        cdsEnd
        exonCount
        exonStarts
        exonEnds
    """
    columns = [
        "gene_symbol",
        "transcript_id",
        "chrom",
        "strand",
        "tx_start",
        "tx_end",
        "cds_start",
        "cds_end",
        "exon_count",
        "exon_starts",
        "exon_ends",
    ]

    table = pd.read_csv(
        REFFLAT_GZ,
        sep="\t",
        compression="gzip",
        header=None,
        names=columns,
        dtype={
            "gene_symbol": str,
            "transcript_id": str,
            "chrom": str,
            "strand": str,
        },
        low_memory=False,
    )

    table[
        "chrom"
    ] = table[
        "chrom"
    ].map(
        normalize_chromosome
    )

    for column in [
        "tx_start",
        "tx_end",
    ]:
        table[
            column
        ] = pd.to_numeric(
            table[
                column
            ],
            errors="coerce",
        )

    table = table.dropna(
        subset=[
            "gene_symbol",
            "transcript_id",
            "chrom",
            "strand",
            "tx_start",
            "tx_end",
        ]
    ).copy()

    table[
        "tx_start"
    ] = table[
        "tx_start"
    ].astype(
        np.int64
    )

    table[
        "tx_end"
    ] = table[
        "tx_end"
    ].astype(
        np.int64
    )

    table = table.loc[
        table[
            "strand"
        ].isin(
            [
                "+",
                "-",
            ]
        )
    ].copy()

    table[
        "tss"
    ] = np.where(
        table[
            "strand"
        ]
        == "+",
        table[
            "tx_start"
        ],
        table[
            "tx_end"
        ]
        - 1,
    ).astype(
        np.int64
    )

    return table


def parse_source_refseq_bed(
    refflat,
):
    """
    Parse the original SHARE-seq repository mm10.UCSC_RefSeq.bed.

    We intentionally use the first six BED fields:
        chrom, start, end, name, score, strand

    If name is a RefSeq transcript accession, translate to gene symbol through
    UCSC refFlat. If name is already a gene symbol, preserve it.
    """
    transcript_to_symbol = (
        refflat[
            [
                "transcript_id",
                "gene_symbol",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "transcript_id"
        )[
            "gene_symbol"
        ]
        .to_dict()
    )

    known_symbols = set(
        refflat[
            "gene_symbol"
        ].astype(str)
    )

    rows = []

    malformed = 0

    with REFSEQ_BED.open(
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line_number, line_text in enumerate(
            handle,
            start=1,
        ):
            stripped = line_text.strip()

            if not stripped:
                continue

            if stripped.startswith(
                "#"
            ) or stripped.startswith(
                "track"
            ) or stripped.startswith(
                "browser"
            ):
                continue

            fields = stripped.split(
                "\t"
            )

            if len(
                fields
            ) < 6:
                fields = re.split(
                    r"\s+",
                    stripped,
                )

            if len(
                fields
            ) < 6:
                malformed += 1
                continue

            chrom = normalize_chromosome(
                fields[
                    0
                ]
            )

            try:
                start = int(
                    fields[
                        1
                    ]
                )

                end = int(
                    fields[
                        2
                    ]
                )
            except ValueError:
                malformed += 1
                continue

            name = str(
                fields[
                    3
                ]
            ).strip()

            strand = str(
                fields[
                    5
                ]
            ).strip()

            if strand not in {
                "+",
                "-",
            }:
                malformed += 1
                continue

            tss = (
                start
                if strand
                == "+"
                else end
                - 1
            )

            mapped_symbol = transcript_to_symbol.get(
                name
            )

            if mapped_symbol:
                gene_symbol = mapped_symbol
                symbol_source = "refFlat_transcript_map"

            elif name in known_symbols:
                gene_symbol = name
                symbol_source = "BED_name_is_gene_symbol"

            else:
                gene_symbol = name
                symbol_source = "BED_name_unresolved"

            rows.append(
                {
                    "chrom": chrom,
                    "bed_start": start,
                    "bed_end": end,
                    "refseq_label": name,
                    "strand": strand,
                    "tss": int(
                        tss
                    ),
                    "gene_symbol": gene_symbol,
                    "gene_symbol_source": symbol_source,
                    "source_bed_line": line_number,
                }
            )

    if not rows:
        raise RuntimeError(
            "No valid entries parsed from mm10.UCSC_RefSeq.bed."
        )

    table = pd.DataFrame(
        rows
    )

    return (
        table,
        malformed,
    )


def build_annotation_tss(
    source_bed,
    refflat,
):
    """
    Primary TSS set comes from the source-compatible SHARE-seq BED.

    For unresolved BED labels, retain the source label.
    """
    tss = source_bed.copy()

    # Remove exact duplicate transcript/TSS rows.
    tss = tss.drop_duplicates(
        subset=[
            "chrom",
            "tss",
            "strand",
            "refseq_label",
            "gene_symbol",
        ]
    ).reset_index(
        drop=True
    )

    return tss


# =====================================================================
# PEAK -> NEAREST TSS
# =====================================================================

def classify_distance(
    absolute_distance,
):
    if absolute_distance <= PROMOTER_BP:
        return "promoter_2kb"

    if absolute_distance <= PROXIMAL_10KB:
        return "proximal_10kb"

    if absolute_distance <= PROXIMAL_50KB:
        return "proximal_50kb"

    return "distal_gt50kb"


def annotate_peaks_nearest_tss(
    peaks,
    tss_table,
):
    by_chrom = {}

    for chrom, subset in tss_table.groupby(
        "chrom",
        sort=False,
    ):
        subset = subset.sort_values(
            "tss"
        ).reset_index(
            drop=True
        )

        by_chrom[
            chrom
        ] = {
            "positions": subset[
                "tss"
            ].to_numpy(
                dtype=np.int64
            ),
            "table": subset,
        }

    annotations = []

    for chrom, peak_subset in peaks.groupby(
        "chrom",
        sort=False,
    ):
        peak_subset = peak_subset.copy()

        reference = by_chrom.get(
            chrom
        )

        if reference is None:
            for row in peak_subset.itertuples(
                index=False
            ):
                annotations.append(
                    {
                        "peak_index_zero_based": int(
                            row.peak_index_zero_based
                        ),
                        "peak_midpoint": int(
                            (
                                int(
                                    row.start
                                )
                                + int(
                                    row.end
                                )
                            )
                            // 2
                        ),
                        "nearest_refseq_label": "",
                        "nearest_gene_symbol": "",
                        "nearest_gene_symbol_source": "",
                        "nearest_tss": np.nan,
                        "nearest_strand": "",
                        "signed_peak_mid_minus_tss": np.nan,
                        "absolute_distance_to_tss": np.nan,
                        "proximity_class": "no_reference_chromosome",
                    }
                )

            continue

        tss_positions = reference[
            "positions"
        ]

        ref_table = reference[
            "table"
        ]

        mids = (
            (
                peak_subset[
                    "start"
                ].to_numpy(
                    dtype=np.int64
                )
                + peak_subset[
                    "end"
                ].to_numpy(
                    dtype=np.int64
                )
            )
            // 2
        )

        insertions = np.searchsorted(
            tss_positions,
            mids,
            side="left",
        )

        for local_index, (
            peak_row,
            midpoint,
            insertion,
        ) in enumerate(
            zip(
                peak_subset.itertuples(
                    index=False
                ),
                mids,
                insertions,
            )
        ):
            candidates = []

            if insertion < len(
                tss_positions
            ):
                candidates.append(
                    int(
                        insertion
                    )
                )

            if insertion > 0:
                candidates.append(
                    int(
                        insertion
                        - 1
                    )
                )

            if not candidates:
                raise RuntimeError(
                    f"No TSS candidate on chromosome {chrom}."
                )

            best_index = min(
                candidates,
                key=lambda index: (
                    abs(
                        int(
                            midpoint
                        )
                        - int(
                            tss_positions[
                                index
                            ]
                        )
                    ),
                    int(
                        tss_positions[
                            index
                        ]
                    ),
                ),
            )

            ref = ref_table.iloc[
                best_index
            ]

            signed_distance = (
                int(
                    midpoint
                )
                - int(
                    ref[
                        "tss"
                    ]
                )
            )

            absolute_distance = abs(
                signed_distance
            )

            annotations.append(
                {
                    "peak_index_zero_based": int(
                        peak_row.peak_index_zero_based
                    ),
                    "peak_midpoint": int(
                        midpoint
                    ),
                    "nearest_refseq_label": str(
                        ref[
                            "refseq_label"
                        ]
                    ),
                    "nearest_gene_symbol": str(
                        ref[
                            "gene_symbol"
                        ]
                    ),
                    "nearest_gene_symbol_source": str(
                        ref[
                            "gene_symbol_source"
                        ]
                    ),
                    "nearest_tss": int(
                        ref[
                            "tss"
                        ]
                    ),
                    "nearest_strand": str(
                        ref[
                            "strand"
                        ]
                    ),
                    "signed_peak_mid_minus_tss": int(
                        signed_distance
                    ),
                    "absolute_distance_to_tss": int(
                        absolute_distance
                    ),
                    "proximity_class": classify_distance(
                        absolute_distance
                    ),
                }
            )

    annotation = pd.DataFrame(
        annotations
    )

    if annotation[
        "peak_index_zero_based"
    ].duplicated().any():
        raise RuntimeError(
            "Nearest-TSS annotation produced duplicate peak indices."
        )

    return annotation


# =====================================================================
# RNA CANDIDATE FILTERS
# =====================================================================

def prevalence_filtered_rna_candidates(
    rna,
):
    required = [
        "gene",
        "pre_ATAC_baseline_CPM",
        "ATAC_event_CPM",
        "RNA_event_CPM",
        "pre_ATAC_baseline_detected_fraction",
        "ATAC_event_detected_fraction",
        "RNA_event_detected_fraction",
        "log2FC_ATACevent_vs_pre_CPM",
        "log2FC_RNAevent_vs_ATACevent_CPM",
        "delta_detected_fraction_ATACevent_vs_pre",
        "delta_detected_fraction_RNAevent_vs_ATACevent",
    ]

    missing = [
        column
        for column in required
        if column not in rna.columns
    ]

    if missing:
        raise RuntimeError(
            "F19 RNA effect table missing column(s): "
            + ", ".join(
                missing
            )
        )

    later_mask = (
        (
            rna[
                "RNA_event_detected_fraction"
            ]
            >= RNA_MIN_DETECTED_FRACTION
        )
        & (
            rna[
                "RNA_event_CPM"
            ]
            >= RNA_MIN_CPM
        )
        & (
            rna[
                "log2FC_RNAevent_vs_ATACevent_CPM"
            ]
            >= RNA_MIN_LOG2FC
        )
        & (
            rna[
                "delta_detected_fraction_RNAevent_vs_ATACevent"
            ]
            >= RNA_MIN_DELTA_DETECTED
        )
    )

    early_mask = (
        (
            rna[
                "ATAC_event_detected_fraction"
            ]
            >= RNA_MIN_DETECTED_FRACTION
        )
        & (
            rna[
                "ATAC_event_CPM"
            ]
            >= RNA_MIN_CPM
        )
        & (
            rna[
                "log2FC_ATACevent_vs_pre_CPM"
            ]
            >= RNA_MIN_LOG2FC
        )
        & (
            rna[
                "delta_detected_fraction_ATACevent_vs_pre"
            ]
            >= RNA_MIN_DELTA_DETECTED
        )
    )

    later = rna.loc[
        later_mask
    ].copy()

    later.insert(
        0,
        "candidate_class",
        "later_RNA_event_increase",
    )

    later = later.sort_values(
        by=[
            "log2FC_RNAevent_vs_ATACevent_CPM",
            "delta_detected_fraction_RNAevent_vs_ATACevent",
            "RNA_event_CPM",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    early = rna.loc[
        early_mask
    ].copy()

    early.insert(
        0,
        "candidate_class",
        "ATAC_stage_RNA_increase",
    )

    early = early.sort_values(
        by=[
            "log2FC_ATACevent_vs_pre_CPM",
            "delta_detected_fraction_ATACevent_vs_pre",
            "ATAC_event_CPM",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    return pd.concat(
        [
            later,
            early,
        ],
        ignore_index=True,
    )


# =====================================================================
# CROSS-MODAL PROXIMITY OVERLAP
# =====================================================================

def build_cross_modal_candidates(
    atac_candidates,
    rna_candidates,
):
    later_rna = rna_candidates.loc[
        rna_candidates[
            "candidate_class"
        ]
        == "later_RNA_event_increase"
    ].copy()

    later_rna = later_rna.drop_duplicates(
        subset=[
            "gene",
        ]
    )

    gene_summary = (
        atac_candidates.loc[
            atac_candidates[
                "nearest_gene_symbol"
            ].astype(str)
            .ne("")
        ]
        .groupby(
            "nearest_gene_symbol",
            as_index=False,
        )
        .agg(
            n_ATAC_opening_peaks=(
                "peak_index_zero_based",
                "size",
            ),
            strongest_delta_accessibility=(
                "delta_accessibility_ATACevent_vs_pre",
                "max",
            ),
            strongest_ATAC_event_accessibility_fraction=(
                "ATAC_event_accessibility_fraction",
                "max",
            ),
            minimum_abs_distance_to_TSS=(
                "absolute_distance_to_tss",
                "min",
            ),
        )
        .rename(
            columns={
                "nearest_gene_symbol": "gene",
            }
        )
    )

    merged = later_rna.merge(
        gene_summary,
        on="gene",
        how="inner",
    )

    if merged.empty:
        return merged

    merged[
        "proximity_supported_cross_modal_candidate"
    ] = True

    merged = merged.sort_values(
        by=[
            "minimum_abs_distance_to_TSS",
            "strongest_delta_accessibility",
            "log2FC_RNAevent_vs_ATACevent_CPM",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return merged


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F20 — mm10 PEAK-TO-TSS ANNOTATION + PREVALENCE-AWARE RNA CANDIDATES"
    )

    print(
        "Timing inference remains CLOSED."
    )

    print()

    print(
        "Nearest-TSS mapping is proximity annotation only."
    )

    print(
        "No motif enrichment."
    )

    print(
        "No causal peak-to-gene claim."
    )

    print(
        "No cell-level P values."
    )

    require_inputs()

    REF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    F20_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Reference acquisition.
    # -----------------------------------------------------------------

    section(
        "1. ACQUIRE / VERIFY SOURCE-COMPATIBLE mm10 REFERENCE ANNOTATION"
    )

    downloaded_bed = download_if_missing(
        REFSEQ_BED_URL,
        REFSEQ_BED,
    )

    downloaded_refflat = download_if_missing(
        REFFLAT_URL,
        REFFLAT_GZ,
    )

    print(
        f"RefSeq BED bytes: {REFSEQ_BED.stat().st_size:,}"
    )

    print(
        f"RefSeq BED SHA256: {sha256_file(REFSEQ_BED)}"
    )

    print(
        f"refFlat bytes: {REFFLAT_GZ.stat().st_size:,}"
    )

    print(
        f"refFlat SHA256: {sha256_file(REFFLAT_GZ)}"
    )

    # -----------------------------------------------------------------
    # Parse references.
    # -----------------------------------------------------------------

    section(
        "2. PARSE mm10 RefSeq / GENE-SYMBOL ANNOTATION"
    )

    refflat = read_refflat()

    (
        source_bed,
        malformed_bed_rows,
    ) = parse_source_refseq_bed(
        refflat
    )

    tss = build_annotation_tss(
        source_bed,
        refflat,
    )

    print(
        f"refFlat rows: {len(refflat):,}"
    )

    print(
        f"Source RefSeq BED valid rows: {len(source_bed):,}"
    )

    print(
        f"Source RefSeq BED malformed rows skipped: "
        f"{malformed_bed_rows:,}"
    )

    print(
        f"Unique TSS annotation rows: {len(tss):,}"
    )

    symbol_source_counts = (
        tss[
            "gene_symbol_source"
        ]
        .value_counts()
        .rename(
            "n_rows"
        )
        .to_frame()
    )

    subsection(
        "Gene-symbol resolution source"
    )

    print_df(
        symbol_source_counts,
        digits=0,
    )

    reference_audit = pd.DataFrame(
        [
            {
                "resource": "SHARE-seq_mm10_UCSC_RefSeq_BED",
                "path": str(
                    REFSEQ_BED
                ),
                "url": REFSEQ_BED_URL,
                "downloaded_this_run": downloaded_bed,
                "bytes": REFSEQ_BED.stat().st_size,
                "sha256": sha256_file(
                    REFSEQ_BED
                ),
                "parsed_rows": len(
                    source_bed
                ),
                "malformed_rows": malformed_bed_rows,
            },
            {
                "resource": "UCSC_mm10_refFlat",
                "path": str(
                    REFFLAT_GZ
                ),
                "url": REFFLAT_URL,
                "downloaded_this_run": downloaded_refflat,
                "bytes": REFFLAT_GZ.stat().st_size,
                "sha256": sha256_file(
                    REFFLAT_GZ
                ),
                "parsed_rows": len(
                    refflat
                ),
                "malformed_rows": 0,
            },
        ]
    )

    reference_audit.to_csv(
        OUTPUT_REFERENCE_AUDIT,
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Load F19 effects.
    # -----------------------------------------------------------------

    section(
        "3. LOAD F19 FROZEN-WINDOW FEATURE EFFECTS"
    )

    rna = pd.read_csv(
        F19_RNA,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    atac = pd.read_csv(
        F19_ATAC,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    windows = pd.read_csv(
        F19_WINDOWS,
        sep="\t",
        low_memory=False,
    )

    print(
        f"RNA features: {len(rna):,}"
    )

    print(
        f"ATAC peaks: {len(atac):,}"
    )

    print(
        f"Frozen mechanistic windows: {len(windows):,}"
    )

    if len(
        atac
    ) != 344_592:
        raise RuntimeError(
            f"Expected 344,592 F19 ATAC peaks; found {len(atac):,}."
        )

    atac[
        "chrom"
    ] = atac[
        "chrom"
    ].map(
        normalize_chromosome
    )

    # -----------------------------------------------------------------
    # Annotate all ATAC peaks.
    # -----------------------------------------------------------------

    section(
        "4. ANNOTATE ALL 344,592 ATAC PEAKS TO NEAREST mm10 RefSeq TSS"
    )

    nearest = annotate_peaks_nearest_tss(
        atac[
            [
                "peak_index_zero_based",
                "chrom",
                "start",
                "end",
            ]
        ],
        tss,
    )

    annotated = atac.merge(
        nearest,
        on="peak_index_zero_based",
        how="left",
        validate="one_to_one",
    )

    if len(
        annotated
    ) != len(
        atac
    ):
        raise RuntimeError(
            "Peak annotation changed ATAC row count."
        )

    annotated.to_csv(
        OUTPUT_ALL_PEAKS,
        sep="\t",
        index=False,
        compression="gzip",
    )

    proximity_counts = (
        annotated[
            "proximity_class"
        ]
        .value_counts()
        .rename(
            "n_peaks"
        )
        .to_frame()
    )

    subsection(
        "All-peak nearest-TSS proximity classes"
    )

    print_df(
        proximity_counts,
        digits=0,
    )

    # -----------------------------------------------------------------
    # ATAC opening candidates.
    # -----------------------------------------------------------------

    section(
        "5. PREVALENCE / EFFECT-SIZE FILTER THE FROZEN ATAC-OPENING CANDIDATES"
    )

    atac_candidates = annotated.loc[
        (
            annotated[
                "delta_accessibility_ATACevent_vs_pre"
            ]
            >= ATAC_MIN_ACCESSIBILITY_GAIN
        )
        & (
            annotated[
                "ATAC_event_accessibility_fraction"
            ]
            >= ATAC_MIN_EVENT_ACCESSIBILITY
        )
    ].copy()

    atac_candidates = atac_candidates.sort_values(
        by=[
            "delta_accessibility_ATACevent_vs_pre",
            "ATAC_event_accessibility_fraction",
            "log2ratio_accessibility_ATACevent_vs_pre",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    atac_candidates.insert(
        0,
        "descriptive_rank",
        np.arange(
            1,
            len(
                atac_candidates
            )
            + 1,
            dtype=int,
        ),
    )

    atac_candidates.to_csv(
        OUTPUT_ATAC_CANDIDATES,
        sep="\t",
        index=False,
    )

    print(
        f"ATAC opening candidates passing frozen descriptive filters: "
        f"{len(atac_candidates):,}"
    )

    subsection(
        "Top 30 annotated ATAC-opening candidates"
    )

    print_df(
        atac_candidates.head(
            30
        )[
            [
                "descriptive_rank",
                "chrom",
                "start",
                "end",
                "delta_accessibility_ATACevent_vs_pre",
                "ATAC_event_accessibility_fraction",
                "log2ratio_accessibility_ATACevent_vs_pre",
                "nearest_gene_symbol",
                "nearest_refseq_label",
                "absolute_distance_to_tss",
                "proximity_class",
            ]
        ].set_index(
            "descriptive_rank"
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # RNA prevalence-aware candidates.
    # -----------------------------------------------------------------

    section(
        "6. BUILD PREVALENCE-AWARE RNA CANDIDATE LISTS"
    )

    rna_candidates = prevalence_filtered_rna_candidates(
        rna
    )

    rna_candidates.to_csv(
        OUTPUT_RNA_CANDIDATES,
        sep="\t",
        index=False,
    )

    rna_counts = (
        rna_candidates[
            "candidate_class"
        ]
        .value_counts()
        .rename(
            "n_genes"
        )
        .to_frame()
    )

    print_df(
        rna_counts,
        digits=0,
    )

    subsection(
        "Top 30 prevalence-aware later RNA-event candidates"
    )

    later_display = rna_candidates.loc[
        rna_candidates[
            "candidate_class"
        ]
        == "later_RNA_event_increase"
    ].head(
        30
    )

    print_df(
        later_display[
            [
                "gene",
                "ATAC_event_CPM",
                "RNA_event_CPM",
                "log2FC_RNAevent_vs_ATACevent_CPM",
                "ATAC_event_detected_fraction",
                "RNA_event_detected_fraction",
                "delta_detected_fraction_RNAevent_vs_ATACevent",
            ]
        ].set_index(
            "gene"
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Descriptive cross-modal proximity overlap.
    # -----------------------------------------------------------------

    section(
        "7. BUILD DESCRIPTIVE CROSS-MODAL PROXIMITY CANDIDATES"
    )

    cross_modal = build_cross_modal_candidates(
        atac_candidates,
        rna_candidates,
    )

    cross_modal.to_csv(
        OUTPUT_CROSS_MODAL,
        sep="\t",
        index=False,
    )

    print(
        f"Genes with >=1 ATAC-opening candidate as nearest TSS "
        f"AND later RNA-event prevalence/effect support: "
        f"{len(cross_modal):,}"
    )

    if not cross_modal.empty:
        print_df(
            cross_modal.head(
                50
            )[
                [
                    "gene",
                    "minimum_abs_distance_to_TSS",
                    "n_ATAC_opening_peaks",
                    "strongest_delta_accessibility",
                    "RNA_event_CPM",
                    "RNA_event_detected_fraction",
                    "log2FC_RNAevent_vs_ATACevent_CPM",
                    "delta_detected_fraction_RNAevent_vs_ATACevent",
                    "proximity_supported_cross_modal_candidate",
                ]
            ].set_index(
                "gene"
            ),
            digits=6,
        )

    # -----------------------------------------------------------------
    # Safeguards.
    # -----------------------------------------------------------------

    section(
        "8. PHASE F20 ANNOTATION SAFEGUARDS"
    )

    resolved_symbol_fraction = float(
        np.mean(
            tss[
                "gene_symbol_source"
            ]
            != "BED_name_unresolved"
        )
    )

    annotated_fraction = float(
        np.mean(
            annotated[
                "nearest_tss"
            ].notna()
        )
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    "Source-compatible SHARE-seq mm10 RefSeq BED available"
                ),
                "pass": REFSEQ_BED.exists(),
            },
            {
                "criterion": (
                    "UCSC mm10 refFlat transcript-to-symbol resource available"
                ),
                "pass": REFFLAT_GZ.exists(),
            },
            {
                "criterion": (
                    "Source RefSeq BED parsed with >10,000 valid rows"
                ),
                "pass": (
                    len(
                        source_bed
                    )
                    > 10_000
                ),
            },
            {
                "criterion": (
                    "All F19 ATAC peaks retained after nearest-TSS annotation"
                ),
                "pass": (
                    len(
                        annotated
                    )
                    == 344_592
                ),
            },
            {
                "criterion": (
                    ">=99% of ATAC peaks have a nearest RefSeq TSS"
                ),
                "pass": (
                    annotated_fraction
                    >= 0.99
                ),
            },
            {
                "criterion": (
                    "RNA prevalence-aware candidate table created"
                ),
                "pass": (
                    len(
                        rna_candidates
                    )
                    > 0
                ),
            },
            {
                "criterion": (
                    "No cell-level statistical significance tests performed"
                ),
                "pass": True,
            },
            {
                "criterion": (
                    "Nearest-TSS overlap labeled proximity only, not causal"
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

    print(
        f"Peak nearest-TSS annotation fraction: "
        f"{annotated_fraction:.6f}"
    )

    print(
        f"Source-BED gene-symbol resolution fraction: "
        f"{resolved_symbol_fraction:.6f}"
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
        "9. SAVE F20 MANIFEST"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F20",
        "created_utc": utc_now_iso(),
        "reference_build": "mm10",
        "references": {
            "SHARE_seq_mm10_UCSC_RefSeq_BED": {
                "url": REFSEQ_BED_URL,
                "path": str(
                    REFSEQ_BED
                ),
                "sha256": sha256_file(
                    REFSEQ_BED
                ),
                "parsed_rows": len(
                    source_bed
                ),
            },
            "UCSC_mm10_refFlat": {
                "url": REFFLAT_URL,
                "path": str(
                    REFFLAT_GZ
                ),
                "sha256": sha256_file(
                    REFFLAT_GZ
                ),
                "parsed_rows": len(
                    refflat
                ),
            },
        },
        "nearest_TSS_annotation": {
            "distance_origin": "ATAC peak midpoint",
            "promoter_bp": PROMOTER_BP,
            "proximal_10kb_bp": PROXIMAL_10KB,
            "proximal_50kb_bp": PROXIMAL_50KB,
            "annotation_fraction": annotated_fraction,
            "causal_interpretation": False,
        },
        "RNA_candidate_filter": {
            "minimum_detected_fraction": RNA_MIN_DETECTED_FRACTION,
            "minimum_CPM": RNA_MIN_CPM,
            "minimum_log2FC": RNA_MIN_LOG2FC,
            "minimum_delta_detected_fraction": RNA_MIN_DELTA_DETECTED,
            "p_values_calculated": False,
        },
        "ATAC_candidate_filter": {
            "minimum_accessibility_gain": ATAC_MIN_ACCESSIBILITY_GAIN,
            "minimum_event_accessibility_fraction": (
                ATAC_MIN_EVENT_ACCESSIBILITY
            ),
            "p_values_calculated": False,
        },
        "counts": {
            "all_ATAC_peaks": len(
                annotated
            ),
            "ATAC_opening_candidates": len(
                atac_candidates
            ),
            "RNA_prevalence_filtered_candidate_rows": len(
                rna_candidates
            ),
            "cross_modal_proximity_candidate_genes": len(
                cross_modal
            ),
        },
        "checks": checks.to_dict(
            orient="records"
        ),
        "outputs": {
            "reference_audit": {
                "file": str(
                    OUTPUT_REFERENCE_AUDIT
                ),
                "sha256": sha256_file(
                    OUTPUT_REFERENCE_AUDIT
                ),
            },
            "all_ATAC_nearest_TSS": {
                "file": str(
                    OUTPUT_ALL_PEAKS
                ),
                "sha256": sha256_file(
                    OUTPUT_ALL_PEAKS
                ),
            },
            "ATAC_opening_candidates": {
                "file": str(
                    OUTPUT_ATAC_CANDIDATES
                ),
                "sha256": sha256_file(
                    OUTPUT_ATAC_CANDIDATES
                ),
            },
            "RNA_prevalence_candidates": {
                "file": str(
                    OUTPUT_RNA_CANDIDATES
                ),
                "sha256": sha256_file(
                    OUTPUT_RNA_CANDIDATES
                ),
            },
            "cross_modal_proximity_candidates": {
                "file": str(
                    OUTPUT_CROSS_MODAL
                ),
                "sha256": sha256_file(
                    OUTPUT_CROSS_MODAL
                ),
            },
        },
        "guardrails": {
            "timing_inference_reopened": False,
            "event_windows_changed": False,
            "gdis_recomputed": False,
            "cmil_recomputed": False,
            "new_null_test_performed": False,
            "cell_level_DE_DA_test_performed": False,
            "motif_analysis_performed": False,
            "causal_peak_gene_link_claimed": False,
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
        OUTPUT_REFERENCE_AUDIT,
        OUTPUT_ALL_PEAKS,
        OUTPUT_ATAC_CANDIDATES,
        OUTPUT_RNA_CANDIDATES,
        OUTPUT_CROSS_MODAL,
        OUTPUT_MANIFEST,
    ]:
        print(
            path
        )

    # -----------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------

    section(
        "10. PHASE F20 DECISION"
    )

    if all_pass:
        print(
            "PHASE F20 VERDICT: GO — mm10 PEAK-TO-TSS ANNOTATION AND "
            "PREVALENCE-AWARE CROSS-MODAL CANDIDATES CONSTRUCTED"
        )

        print()

        print(
            "Nearest-TSS relationships remain descriptive proximity "
            "annotations only."
        )

        print()

        print(
            "Next phase may test regulatory plausibility using motif/TF "
            "annotation and the frozen candidate sets without reopening "
            "timing inference."
        )

    else:
        print(
            "PHASE F20 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not proceed to motif / TF analysis until failed annotation "
            "safeguards are understood."
        )

    print()

    print(
        "No timing analysis was performed."
    )

    print(
        "No cell-level P values were calculated."
    )

    print(
        "No causal peak-to-gene claim was made."
    )

    print(
        "No priming claim was made."
    )

    line("=")


if __name__ == "__main__":
    main()

