#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
36_phase_f21_shareseq_irs_motif_tf_regulatory_plausibility.py
================================================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F21
---------
Test regulatory plausibility of the ALREADY-FROZEN ATAC-opening candidate set
using JASPAR2026 CORE vertebrate motifs and TF expression in the frozen
TAC->IRS trajectory.

TIMING INFERENCE REMAINS CLOSED
-------------------------------
F21 does NOT:
    - reopen event timing;
    - change the ATAC or RNA event windows;
    - recompute GDIS;
    - recompute CMIL;
    - run another timing null;
    - reselect the 310 F20 ATAC-opening peaks.

PRIMARY BIOLOGICAL QUESTION
---------------------------
Do the frozen ATAC-opening peaks show enrichment for transcription-factor
motifs whose TF genes are themselves expressed at or before the frozen ATAC
event?

REFERENCE RESOURCES
-------------------
Mouse genome:
    UCSC GRCm38/mm10 2bit
    https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.2bit

Motifs:
    JASPAR2026 CORE vertebrates, non-redundant PFM batch
    https://jaspar.elixir.no/download/data/2026/CORE/
        JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt

JASPAR CORE is used rather than UNVALIDATED motifs.

DEPENDENCIES
------------
numpy
pandas
scipy
py2bit
MOODS-python

Install missing sequence/motif dependencies with:

    python -m pip install py2bit MOODS-python

MOTIF UNIVERSE
--------------
Primary motif analysis is deliberately restricted to SINGLE-TF motifs.

A JASPAR motif enters the primary universe only when:

    1. motif TF name maps case-insensitively to one RNA gene symbol;
    2. motif name is not a compound "::" motif;
    3. TF RNA expression at or before the ATAC event satisfies:

           max(pre_ATAC CPM, ATAC_event CPM) >= 2

       AND

           max(pre_ATAC detected fraction,
               ATAC_event detected fraction) >= 0.02

This avoids interpreting enrichment for TFs with no detectable support in the
trajectory.

MOTIF SCANNING
--------------
MOODS scans both strands.

Primary motif-match threshold:

    per-position motif-match p-value <= 1e-4

PFMs are converted to log-odds PWMs with:
    flat DNA background = [0.25, 0.25, 0.25, 0.25]
    pseudocount = 0.8

A peak is counted as motif-positive when >=1 hit occurs on either strand.

MATCHED BACKGROUND
------------------
Candidate set:
    the 310 F20 frozen ATAC-opening peaks.

Background set:
    non-candidate F20 peaks.

Matching is performed without replacement using:
    - nearest-TSS proximity class;
    - pre-ATAC baseline accessibility bin (width = 0.05);
    - GC-content bin after mm10 sequence extraction.

Procedure:
    1. For each candidate proximity/accessibility stratum, draw a large
       deterministic candidate background pool from non-opening peaks.
    2. Extract mm10 sequence and calculate GC.
    3. Sub-stratify by fixed GC bins.
    4. Sample up to 10 background peaks per candidate within exact
       proximity/accessibility/GC strata.

No motif information is used to select background peaks.

Background QC reports:
    - final candidate/background ratio;
    - mean pre-accessibility;
    - mean GC;
    - standardized mean differences (SMD).

MOTIF ENRICHMENT
----------------
For each eligible motif:

    candidate motif-positive peaks
    candidate motif-negative peaks
    background motif-positive peaks
    background motif-negative peaks

One-sided Fisher exact test:
    H1 = motif enriched in ATAC-opening candidates.

Multiple testing:
    Benjamini-Hochberg FDR across the primary motif universe.

Descriptive enriched-motif flag:
    FDR <= 0.05
    odds ratio >= 1.5
    candidate motif-positive peaks >= 5

This flag is not used to alter prior GDIS/timing conclusions.

TF / RNA INTEGRATION
--------------------
For each motif, F21 also reports:
    - TF pre-ATAC CPM/detection;
    - TF ATAC-event CPM/detection;
    - TF RNA-event CPM/detection;
    - TF RNA event vs ATAC-event log2FC;
    - whether TF belongs to F20 later-RNA candidate set;
    - whether TF belongs to F20 ATAC-stage RNA candidate set.

CROSS-MODAL CANDIDATE ANNOTATION
--------------------------------
For F20 proximity-supported genes, F21 reports enriched motif hits within the
specific ATAC-opening peaks assigned by nearest-TSS proximity.

This remains:
    regulatory plausibility

It is NOT:
    causal peak->gene proof
    TF-target proof
    universal chromatin priming

INPUTS
------
data/GSE140203/representations_f19/
    f19_rna_event_feature_effects.tsv.gz

data/GSE140203/representations_f20/
    f20_all_atac_peaks_nearest_tss.tsv.gz
    f20_atac_opening_candidates_annotated.tsv
    f20_rna_prevalence_filtered_candidates.tsv
    f20_cross_modal_proximity_candidates.tsv
    f20_manifest.json

OUTPUTS
-------
data/GSE140203/reference_f21/
    mm10.2bit
    JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt

data/GSE140203/representations_f21/
    f21_reference_audit.tsv
    f21_background_matching_qc.tsv
    f21_primary_motif_universe.tsv
    f21_motif_enrichment.tsv
    f21_enriched_tf_candidates.tsv
    f21_atac_opening_peak_enriched_motif_hits.tsv.gz
    f21_cross_modal_regulatory_plausibility.tsv
    f21_manifest.json

RUN
---
    python 36_phase_f21_shareseq_irs_motif_tf_regulatory_plausibility.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

try:
    import py2bit
except ImportError as exc:
    raise SystemExit(
        "\nERROR: F21 requires py2bit.\n\n"
        "Install with:\n"
        "    python -m pip install py2bit\n"
    ) from exc

try:
    import MOODS.scan
    import MOODS.tools
except ImportError as exc:
    raise SystemExit(
        "\nERROR: F21 requires MOODS-python.\n\n"
        "Install with:\n"
        "    python -m pip install MOODS-python\n"
    ) from exc


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE140203"

DATA_DIR = Path("data") / ACCESSION
F19_DIR = DATA_DIR / "representations_f19"
F20_DIR = DATA_DIR / "representations_f20"
REF_DIR = DATA_DIR / "reference_f21"
F21_DIR = DATA_DIR / "representations_f21"

F19_RNA = F19_DIR / "f19_rna_event_feature_effects.tsv.gz"

F20_ALL_ATAC = F20_DIR / "f20_all_atac_peaks_nearest_tss.tsv.gz"
F20_ATAC_CANDIDATES = F20_DIR / "f20_atac_opening_candidates_annotated.tsv"
F20_RNA_CANDIDATES = F20_DIR / "f20_rna_prevalence_filtered_candidates.tsv"
F20_CROSS_MODAL = F20_DIR / "f20_cross_modal_proximity_candidates.tsv"
F20_MANIFEST = F20_DIR / "f20_manifest.json"

MM10_2BIT = REF_DIR / "mm10.2bit"
JASPAR_PFM = (
    REF_DIR
    / "JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
)

OUTPUT_REFERENCE_AUDIT = F21_DIR / "f21_reference_audit.tsv"
OUTPUT_BACKGROUND_QC = F21_DIR / "f21_background_matching_qc.tsv"
OUTPUT_MOTIF_UNIVERSE = F21_DIR / "f21_primary_motif_universe.tsv"
OUTPUT_ENRICHMENT = F21_DIR / "f21_motif_enrichment.tsv"
OUTPUT_ENRICHED_TF = F21_DIR / "f21_enriched_tf_candidates.tsv"
OUTPUT_PEAK_HITS = F21_DIR / "f21_atac_opening_peak_enriched_motif_hits.tsv.gz"
OUTPUT_CROSS_MODAL = F21_DIR / "f21_cross_modal_regulatory_plausibility.tsv"
OUTPUT_MANIFEST = F21_DIR / "f21_manifest.json"


# =====================================================================
# FROZEN REFERENCE URLS
# =====================================================================

MM10_2BIT_URL = (
    "https://hgdownload.soe.ucsc.edu/"
    "goldenPath/mm10/bigZips/mm10.2bit"
)

JASPAR_PFM_URL = (
    "https://jaspar.elixir.no/download/data/2026/CORE/"
    "JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
)


# =====================================================================
# FROZEN ANALYSIS SETTINGS
# =====================================================================

RANDOM_SEED = 785

EXPECTED_ATAC_CANDIDATES = 310

PRE_ACCESS_BIN_WIDTH = 0.05

GC_BIN_EDGES = np.asarray(
    [
        0.00,
        0.35,
        0.45,
        0.55,
        0.65,
        1.000001,
    ],
    dtype=float,
)

INITIAL_BACKGROUND_POOL_MULTIPLIER = 40
FINAL_BACKGROUND_PER_CANDIDATE = 10

MIN_FINAL_BACKGROUND_RATIO = 5.0
MAX_ABS_SMD_PRE_ACCESS = 0.25
MAX_ABS_SMD_GC = 0.25

TF_MIN_CPM = 2.0
TF_MIN_DETECTED_FRACTION = 0.02

MOTIF_MATCH_PVALUE = 1e-4
MOTIF_PSEUDOCOUNT = 0.8
MOODS_WINDOW_SIZE = 7

ENRICHMENT_FDR = 0.05
ENRICHMENT_MIN_OR = 1.5
ENRICHMENT_MIN_CANDIDATE_HITS = 5


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
        "display.width", 560,
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
    minimum_bytes,
):
    if path.exists():
        if path.stat().st_size < minimum_bytes:
            raise RuntimeError(
                f"Existing file is unexpectedly small: {path} "
                f"({path.stat().st_size:,} bytes)"
            )

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
        f"Downloading:\n  {url}\n  -> {path}",
        flush=True,
    )

    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "GDIS-Multiome-F21/1.0 "
                    "(motif-regulatory-plausibility)"
                )
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=180,
        ) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(
                response,
                output,
                length=4 * 1024 * 1024,
            )

        if temporary.stat().st_size < minimum_bytes:
            raise RuntimeError(
                f"Downloaded file is unexpectedly small: "
                f"{temporary.stat().st_size:,} bytes"
            )

        temporary.replace(
            path
        )

    except Exception as exc:
        temporary.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "\nCould not download required F21 reference.\n\n"
            f"URL:\n  {url}\n\n"
            f"Expected path:\n  {path}\n\n"
            "If outbound HTTPS is restricted, download this file manually "
            "to the expected path and rerun F21.\n\n"
            f"Original error: {exc}"
        ) from exc

    return True


def normalize_symbol(
    value,
):
    return str(
        value
    ).strip().upper()


def bh_fdr(
    pvalues,
):
    p = np.asarray(
        pvalues,
        dtype=float,
    )

    n = len(
        p
    )

    order = np.argsort(
        p
    )

    ranked = p[
        order
    ]

    adjusted = (
        ranked
        * n
        / np.arange(
            1,
            n + 1,
            dtype=float,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[
            ::-1
        ]
    )[
        ::-1
    ]

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    output = np.empty(
        n,
        dtype=float,
    )

    output[
        order
    ] = adjusted

    return output


def standardized_mean_difference(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    pooled = math.sqrt(
        (
            np.var(
                a,
                ddof=1,
            )
            + np.var(
                b,
                ddof=1,
            )
        )
        / 2.0
    )

    if pooled <= 0:
        return 0.0

    return float(
        (
            np.mean(
                a
            )
            - np.mean(
                b
            )
        )
        / pooled
    )


# =====================================================================
# INPUTS
# =====================================================================

def require_inputs():
    required = [
        F19_RNA,
        F20_ALL_ATAC,
        F20_ATAC_CANDIDATES,
        F20_RNA_CANDIDATES,
        F20_CROSS_MODAL,
        F20_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required F19/F20 input(s):\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )


# =====================================================================
# JASPAR PARSER
# =====================================================================

def parse_jaspar_pfm(
    path,
):
    motifs = []

    motif_id = None
    tf_name = None
    rows = {}

    def flush():
        nonlocal motif_id, tf_name, rows

        if motif_id is None:
            return

        if set(
            rows
        ) != {
            "A",
            "C",
            "G",
            "T",
        }:
            raise RuntimeError(
                f"Incomplete PFM for {motif_id} {tf_name}"
            )

        lengths = {
            len(
                rows[
                    base
                ]
            )
            for base in [
                "A",
                "C",
                "G",
                "T",
            ]
        }

        if len(
            lengths
        ) != 1:
            raise RuntimeError(
                f"Inconsistent PFM row lengths for {motif_id}"
            )

        motifs.append(
            {
                "motif_id": motif_id,
                "tf_name": tf_name,
                "pfm": [
                    rows[
                        "A"
                    ],
                    rows[
                        "C"
                    ],
                    rows[
                        "G"
                    ],
                    rows[
                        "T"
                    ],
                ],
                "motif_length": next(
                    iter(
                        lengths
                    )
                ),
            }
        )

        motif_id = None
        tf_name = None
        rows = {}

    with path.open(
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line_text in handle:
            stripped = line_text.strip()

            if not stripped:
                continue

            if stripped.startswith(
                ">"
            ):
                flush()

                header = stripped[
                    1:
                ].split(
                    "\t",
                    1,
                )

                motif_id = header[
                    0
                ].strip()

                tf_name = (
                    header[
                        1
                    ].strip()
                    if len(
                        header
                    )
                    > 1
                    else motif_id
                )

                continue

            base = stripped[
                0
            ].upper()

            if base not in {
                "A",
                "C",
                "G",
                "T",
            }:
                continue

            if "[" not in stripped or "]" not in stripped:
                raise RuntimeError(
                    f"Unexpected JASPAR row: {stripped}"
                )

            values_text = stripped.split(
                "[",
                1,
            )[
                1
            ].split(
                "]",
                1,
            )[
                0
            ]

            values = [
                float(
                    value
                )
                for value in values_text.split()
            ]

            rows[
                base
            ] = values

    flush()

    return motifs


# =====================================================================
# SEQUENCE / GC
# =====================================================================

def gc_fraction(
    sequence,
):
    seq = str(
        sequence
    ).upper()

    valid = sum(
        base in {
            "A",
            "C",
            "G",
            "T",
        }
        for base in seq
    )

    if valid == 0:
        return np.nan

    gc = (
        seq.count(
            "G"
        )
        + seq.count(
            "C"
        )
    )

    return float(
        gc
        / valid
    )


def extract_sequences(
    peaks,
    genome,
):
    chrom_sizes = genome.chroms()

    rows = []

    sequences = {}

    for row in peaks.itertuples(
        index=False
    ):
        peak_id = int(
            row.peak_index_zero_based
        )

        chrom = str(
            row.chrom
        )

        start = int(
            row.start
        )

        end = int(
            row.end
        )

        valid = True
        reason = ""

        if chrom not in chrom_sizes:
            valid = False
            reason = "chromosome_not_in_mm10"

        elif start < 0 or end <= start or end > int(
            chrom_sizes[
                chrom
            ]
        ):
            valid = False
            reason = "coordinate_out_of_range"

        if valid:
            seq = genome.sequence(
                chrom,
                start,
                end,
            )

            if seq is None:
                valid = False
                reason = "sequence_none"

        if valid:
            seq = str(
                seq
            ).upper()

            if len(
                seq
            ) != (
                end
                - start
            ):
                valid = False
                reason = "sequence_length_mismatch"

        if valid:
            gc = gc_fraction(
                seq
            )

            if not np.isfinite(
                gc
            ):
                valid = False
                reason = "no_acgt_sequence"

        if valid:
            sequences[
                peak_id
            ] = seq
        else:
            gc = np.nan

        rows.append(
            {
                "peak_index_zero_based": peak_id,
                "sequence_valid": valid,
                "sequence_issue": reason,
                "sequence_length": (
                    len(
                        seq
                    )
                    if valid
                    else np.nan
                ),
                "GC_fraction": gc,
            }
        )

    return (
        pd.DataFrame(
            rows
        ),
        sequences,
    )


# =====================================================================
# BACKGROUND MATCHING
# =====================================================================

def pre_access_bin(
    values,
):
    return np.floor(
        np.asarray(
            values,
            dtype=float,
        )
        / PRE_ACCESS_BIN_WIDTH
    ).astype(
        int
    )


def gc_bin(
    values,
):
    return np.digitize(
        np.asarray(
            values,
            dtype=float,
        ),
        GC_BIN_EDGES[
            1:
            -1
        ],
        right=False,
    ).astype(
        int
    )


def build_initial_background_pool(
    all_peaks,
    candidates,
):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    candidate_ids = set(
        candidates[
            "peak_index_zero_based"
        ].astype(int)
    )

    background = all_peaks.loc[
        ~all_peaks[
            "peak_index_zero_based"
        ].astype(int).isin(
            candidate_ids
        )
    ].copy()

    candidates = candidates.copy()

    candidates[
        "pre_access_bin"
    ] = pre_access_bin(
        candidates[
            "pre_ATAC_baseline_accessibility_fraction"
        ]
    )

    background[
        "pre_access_bin"
    ] = pre_access_bin(
        background[
            "pre_ATAC_baseline_accessibility_fraction"
        ]
    )

    selected_pool_indices = []

    candidate_strata = (
        candidates.groupby(
            [
                "proximity_class",
                "pre_access_bin",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "candidate_n"
        )
        .reset_index()
    )

    audit_rows = []

    for stratum in candidate_strata.itertuples(
        index=False
    ):
        exact = background.loc[
            (
                background[
                    "proximity_class"
                ].astype(str)
                == str(
                    stratum.proximity_class
                )
            )
            & (
                background[
                    "pre_access_bin"
                ]
                == int(
                    stratum.pre_access_bin
                )
            )
        ]

        requested = int(
            stratum.candidate_n
            * INITIAL_BACKGROUND_POOL_MULTIPLIER
        )

        if len(
            exact
        ) <= requested:
            chosen = exact
        else:
            chosen_positions = rng.choice(
                len(
                    exact
                ),
                size=requested,
                replace=False,
            )

            chosen = exact.iloc[
                chosen_positions
            ]

        selected_pool_indices.extend(
            chosen.index.tolist()
        )

        audit_rows.append(
            {
                "proximity_class": str(
                    stratum.proximity_class
                ),
                "pre_access_bin": int(
                    stratum.pre_access_bin
                ),
                "candidate_n": int(
                    stratum.candidate_n
                ),
                "available_background_n": int(
                    len(
                        exact
                    )
                ),
                "requested_initial_pool_n": requested,
                "selected_initial_pool_n": int(
                    len(
                        chosen
                    )
                ),
            }
        )

    pool = (
        background.loc[
            sorted(
                set(
                    selected_pool_indices
                )
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        candidates,
        pool,
        pd.DataFrame(
            audit_rows
        ),
    )


def final_gc_matched_background(
    candidates,
    pool,
):
    rng = np.random.default_rng(
        RANDOM_SEED
        + 1
    )

    candidates = candidates.copy()
    pool = pool.copy()

    candidates[
        "gc_bin"
    ] = gc_bin(
        candidates[
            "GC_fraction"
        ]
    )

    pool[
        "gc_bin"
    ] = gc_bin(
        pool[
            "GC_fraction"
        ]
    )

    selected_indices = []

    rows = []

    candidate_strata = (
        candidates.groupby(
            [
                "proximity_class",
                "pre_access_bin",
                "gc_bin",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "candidate_n"
        )
        .reset_index()
    )

    for stratum in candidate_strata.itertuples(
        index=False
    ):
        available = pool.loc[
            (
                pool[
                    "proximity_class"
                ].astype(str)
                == str(
                    stratum.proximity_class
                )
            )
            & (
                pool[
                    "pre_access_bin"
                ]
                == int(
                    stratum.pre_access_bin
                )
            )
            & (
                pool[
                    "gc_bin"
                ]
                == int(
                    stratum.gc_bin
                )
            )
        ]

        requested = int(
            stratum.candidate_n
            * FINAL_BACKGROUND_PER_CANDIDATE
        )

        available = available.loc[
            ~available.index.isin(
                selected_indices
            )
        ]

        if len(
            available
        ) <= requested:
            chosen = available
        else:
            chosen_positions = rng.choice(
                len(
                    available
                ),
                size=requested,
                replace=False,
            )

            chosen = available.iloc[
                chosen_positions
            ]

        selected_indices.extend(
            chosen.index.tolist()
        )

        rows.append(
            {
                "proximity_class": str(
                    stratum.proximity_class
                ),
                "pre_access_bin": int(
                    stratum.pre_access_bin
                ),
                "gc_bin": int(
                    stratum.gc_bin
                ),
                "candidate_n": int(
                    stratum.candidate_n
                ),
                "requested_background_n": requested,
                "available_background_n": int(
                    len(
                        available
                    )
                ),
                "selected_background_n": int(
                    len(
                        chosen
                    )
                ),
            }
        )

    final_background = (
        pool.loc[
            sorted(
                set(
                    selected_indices
                )
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        final_background,
        pd.DataFrame(
            rows
        ),
    )


# =====================================================================
# MOTIF UNIVERSE
# =====================================================================

def build_primary_motif_universe(
    motifs,
    rna,
    rna_candidates,
):
    rna = rna.copy()

    rna[
        "_symbol_norm"
    ] = rna[
        "gene"
    ].map(
        normalize_symbol
    )

    # Keep the first exact gene-symbol record if duplicates exist.
    rna_lookup = (
        rna.drop_duplicates(
            subset=[
                "_symbol_norm",
            ],
            keep="first",
        )
        .set_index(
            "_symbol_norm"
        )
    )

    later_symbols = set(
        rna_candidates.loc[
            rna_candidates[
                "candidate_class"
            ].astype(str)
            == "later_RNA_event_increase",
            "gene",
        ].map(
            normalize_symbol
        )
    )

    early_symbols = set(
        rna_candidates.loc[
            rna_candidates[
                "candidate_class"
            ].astype(str)
            == "ATAC_stage_RNA_increase",
            "gene",
        ].map(
            normalize_symbol
        )
    )

    rows = []
    eligible_motifs = []

    for motif in motifs:
        tf_name = str(
            motif[
                "tf_name"
            ]
        ).strip()

        single_tf = (
            "::"
            not in tf_name
        )

        tf_norm = normalize_symbol(
            tf_name
        )

        mapped = (
            tf_norm
            in rna_lookup.index
        )

        pre_cpm = np.nan
        atac_cpm = np.nan
        rna_cpm = np.nan

        pre_detect = np.nan
        atac_detect = np.nan
        rna_detect = np.nan

        later_log2fc = np.nan

        expressed_before_or_at_event = False

        mapped_gene = ""

        if mapped:
            row = rna_lookup.loc[
                tf_norm
            ]

            mapped_gene = str(
                row[
                    "gene"
                ]
            )

            pre_cpm = float(
                row[
                    "pre_ATAC_baseline_CPM"
                ]
            )

            atac_cpm = float(
                row[
                    "ATAC_event_CPM"
                ]
            )

            rna_cpm = float(
                row[
                    "RNA_event_CPM"
                ]
            )

            pre_detect = float(
                row[
                    "pre_ATAC_baseline_detected_fraction"
                ]
            )

            atac_detect = float(
                row[
                    "ATAC_event_detected_fraction"
                ]
            )

            rna_detect = float(
                row[
                    "RNA_event_detected_fraction"
                ]
            )

            later_log2fc = float(
                row[
                    "log2FC_RNAevent_vs_ATACevent_CPM"
                ]
            )

            expressed_before_or_at_event = bool(
                max(
                    pre_cpm,
                    atac_cpm,
                )
                >= TF_MIN_CPM
                and max(
                    pre_detect,
                    atac_detect,
                )
                >= TF_MIN_DETECTED_FRACTION
            )

        eligible = bool(
            single_tf
            and mapped
            and expressed_before_or_at_event
        )

        rows.append(
            {
                "motif_id": motif[
                    "motif_id"
                ],
                "jaspar_tf_name": tf_name,
                "single_tf_motif": single_tf,
                "mapped_to_rna_gene": mapped,
                "mapped_gene_symbol": mapped_gene,
                "motif_length": motif[
                    "motif_length"
                ],
                "TF_pre_ATAC_CPM": pre_cpm,
                "TF_ATAC_event_CPM": atac_cpm,
                "TF_RNA_event_CPM": rna_cpm,
                "TF_pre_ATAC_detected_fraction": pre_detect,
                "TF_ATAC_event_detected_fraction": atac_detect,
                "TF_RNA_event_detected_fraction": rna_detect,
                "TF_RNAevent_vs_ATACevent_log2FC": later_log2fc,
                "TF_expressed_pre_or_ATAC_event": (
                    expressed_before_or_at_event
                ),
                "TF_is_F20_later_RNA_candidate": (
                    tf_norm
                    in later_symbols
                ),
                "TF_is_F20_ATAC_stage_RNA_candidate": (
                    tf_norm
                    in early_symbols
                ),
                "eligible_primary_motif": eligible,
            }
        )

        if eligible:
            motif_copy = dict(
                motif
            )

            motif_copy[
                "mapped_gene_symbol"
            ] = mapped_gene

            eligible_motifs.append(
                motif_copy
            )

    return (
        pd.DataFrame(
            rows
        ),
        eligible_motifs,
    )


# =====================================================================
# MOODS SCANNER
# =====================================================================

def prepare_moods_scanner(
    motifs,
):
    background = [
        0.25,
        0.25,
        0.25,
        0.25,
    ]

    forward_matrices = []

    reverse_matrices = []

    forward_thresholds = []

    reverse_thresholds = []

    for motif in motifs:
        pfm = [
            [
                float(
                    value
                )
                for value in row
            ]
            for row in motif[
                "pfm"
            ]
        ]

        pwm = MOODS.tools.log_odds(
            pfm,
            background,
            MOTIF_PSEUDOCOUNT,
        )

        reverse_pwm = MOODS.tools.reverse_complement(
            pwm
        )

        threshold_forward = MOODS.tools.threshold_from_p(
            pwm,
            background,
            MOTIF_MATCH_PVALUE,
        )

        threshold_reverse = MOODS.tools.threshold_from_p(
            reverse_pwm,
            background,
            MOTIF_MATCH_PVALUE,
        )

        forward_matrices.append(
            pwm
        )

        reverse_matrices.append(
            reverse_pwm
        )

        forward_thresholds.append(
            threshold_forward
        )

        reverse_thresholds.append(
            threshold_reverse
        )

    matrices = (
        forward_matrices
        + reverse_matrices
    )

    thresholds = (
        forward_thresholds
        + reverse_thresholds
    )

    scanner = MOODS.scan.Scanner(
        MOODS_WINDOW_SIZE
    )

    scanner.set_motifs(
        matrices,
        background,
        thresholds,
    )

    return scanner


def scan_peak_sequences(
    peaks,
    sequences,
    motifs,
    scanner,
):
    n_motifs = len(
        motifs
    )

    presence = np.zeros(
        (
            len(
                peaks
            ),
            n_motifs,
        ),
        dtype=np.uint8,
    )

    hit_rows = []

    for row_index, peak in enumerate(
        peaks.itertuples(
            index=False
        )
    ):
        peak_id = int(
            peak.peak_index_zero_based
        )

        seq = sequences[
            peak_id
        ]

        results = scanner.scan(
            seq
        )

        if len(
            results
        ) != (
            2
            * n_motifs
        ):
            raise RuntimeError(
                "Unexpected MOODS result length."
            )

        for motif_index, motif in enumerate(
            motifs
        ):
            forward_hits = results[
                motif_index
            ]

            reverse_hits = results[
                motif_index
                + n_motifs
            ]

            if (
                len(
                    forward_hits
                )
                > 0
                or len(
                    reverse_hits
                )
                > 0
            ):
                presence[
                    row_index,
                    motif_index
                ] = 1

                hit_rows.append(
                    {
                        "peak_index_zero_based": peak_id,
                        "motif_id": motif[
                            "motif_id"
                        ],
                        "jaspar_tf_name": motif[
                            "tf_name"
                        ],
                        "mapped_gene_symbol": motif[
                            "mapped_gene_symbol"
                        ],
                        "forward_hit_count": int(
                            len(
                                forward_hits
                            )
                        ),
                        "reverse_hit_count": int(
                            len(
                                reverse_hits
                            )
                        ),
                        "total_hit_count": int(
                            len(
                                forward_hits
                            )
                            + len(
                                reverse_hits
                            )
                        ),
                    }
                )

        if (
            (
                row_index
                + 1
            )
            % 500
            == 0
            or row_index
            == 0
        ):
            print(
                f"  motif-scanned {row_index + 1:,} / "
                f"{len(peaks):,} peaks",
                flush=True,
            )

    return (
        presence,
        pd.DataFrame(
            hit_rows
        ),
    )


# =====================================================================
# ENRICHMENT
# =====================================================================

def motif_enrichment(
    candidate_presence,
    background_presence,
    motifs,
    motif_universe_table,
):
    universe_lookup = (
        motif_universe_table.loc[
            motif_universe_table[
                "eligible_primary_motif"
            ]
        ]
        .set_index(
            "motif_id"
        )
    )

    rows = []

    n_candidate = candidate_presence.shape[
        0
    ]

    n_background = background_presence.shape[
        0
    ]

    for motif_index, motif in enumerate(
        motifs
    ):
        candidate_hits = int(
            np.sum(
                candidate_presence[
                    :,
                    motif_index
                ]
            )
        )

        background_hits = int(
            np.sum(
                background_presence[
                    :,
                    motif_index
                ]
            )
        )

        candidate_nonhits = (
            n_candidate
            - candidate_hits
        )

        background_nonhits = (
            n_background
            - background_hits
        )

        odds_ratio, pvalue = fisher_exact(
            [
                [
                    candidate_hits,
                    candidate_nonhits,
                ],
                [
                    background_hits,
                    background_nonhits,
                ],
            ],
            alternative="greater",
        )

        meta = universe_lookup.loc[
            motif[
                "motif_id"
            ]
        ]

        rows.append(
            {
                "motif_id": motif[
                    "motif_id"
                ],
                "jaspar_tf_name": motif[
                    "tf_name"
                ],
                "mapped_gene_symbol": motif[
                    "mapped_gene_symbol"
                ],
                "motif_length": motif[
                    "motif_length"
                ],
                "candidate_n": n_candidate,
                "candidate_hit_n": candidate_hits,
                "candidate_hit_fraction": (
                    candidate_hits
                    / n_candidate
                ),
                "background_n": n_background,
                "background_hit_n": background_hits,
                "background_hit_fraction": (
                    background_hits
                    / n_background
                ),
                "odds_ratio": float(
                    odds_ratio
                ),
                "fisher_p": float(
                    pvalue
                ),
                "TF_pre_ATAC_CPM": float(
                    meta[
                        "TF_pre_ATAC_CPM"
                    ]
                ),
                "TF_ATAC_event_CPM": float(
                    meta[
                        "TF_ATAC_event_CPM"
                    ]
                ),
                "TF_RNA_event_CPM": float(
                    meta[
                        "TF_RNA_event_CPM"
                    ]
                ),
                "TF_pre_ATAC_detected_fraction": float(
                    meta[
                        "TF_pre_ATAC_detected_fraction"
                    ]
                ),
                "TF_ATAC_event_detected_fraction": float(
                    meta[
                        "TF_ATAC_event_detected_fraction"
                    ]
                ),
                "TF_RNA_event_detected_fraction": float(
                    meta[
                        "TF_RNA_event_detected_fraction"
                    ]
                ),
                "TF_RNAevent_vs_ATACevent_log2FC": float(
                    meta[
                        "TF_RNAevent_vs_ATACevent_log2FC"
                    ]
                ),
                "TF_is_F20_later_RNA_candidate": bool(
                    meta[
                        "TF_is_F20_later_RNA_candidate"
                    ]
                ),
                "TF_is_F20_ATAC_stage_RNA_candidate": bool(
                    meta[
                        "TF_is_F20_ATAC_stage_RNA_candidate"
                    ]
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "BH_FDR"
    ] = bh_fdr(
        result[
            "fisher_p"
        ].to_numpy(
            dtype=float
        )
    )

    result[
        "enriched_motif"
    ] = (
        (
            result[
                "BH_FDR"
            ]
            <= ENRICHMENT_FDR
        )
        & (
            result[
                "odds_ratio"
            ]
            >= ENRICHMENT_MIN_OR
        )
        & (
            result[
                "candidate_hit_n"
            ]
            >= ENRICHMENT_MIN_CANDIDATE_HITS
        )
    )

    result = result.sort_values(
        by=[
            "BH_FDR",
            "fisher_p",
            "odds_ratio",
            "candidate_hit_fraction",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    result.insert(
        0,
        "motif_rank",
        np.arange(
            1,
            len(
                result
            )
            + 1,
            dtype=int,
        ),
    )

    return result


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F21 — JASPAR2026 MOTIF / TF REGULATORY PLAUSIBILITY"
    )

    print(
        "Timing inference remains CLOSED."
    )

    print()

    print(
        "Primary motif collection:"
    )

    print(
        "  JASPAR2026 CORE vertebrates, non-redundant"
    )

    print()

    print(
        "Primary motif universe:"
    )

    print(
        "  single-TF motifs mapped to TF genes expressed before/at ATAC event"
    )

    print()

    print(
        f"Motif match threshold: p <= {MOTIF_MATCH_PVALUE:g}"
    )

    print()

    print(
        "No causal TF-target claim."
    )

    require_inputs()

    REF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    F21_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Acquire references.
    # -----------------------------------------------------------------

    section(
        "1. ACQUIRE / VERIFY mm10 GENOME AND JASPAR2026 CORE MOTIFS"
    )

    downloaded_mm10 = download_if_missing(
        MM10_2BIT_URL,
        MM10_2BIT,
        minimum_bytes=600_000_000,
    )

    downloaded_jaspar = download_if_missing(
        JASPAR_PFM_URL,
        JASPAR_PFM,
        minimum_bytes=100_000,
    )

    reference_audit = pd.DataFrame(
        [
            {
                "resource": "UCSC_mm10_2bit",
                "path": str(
                    MM10_2BIT
                ),
                "url": MM10_2BIT_URL,
                "downloaded_this_run": downloaded_mm10,
                "bytes": MM10_2BIT.stat().st_size,
                "sha256": sha256_file(
                    MM10_2BIT
                ),
            },
            {
                "resource": "JASPAR2026_CORE_vertebrates_nonredundant_PFM",
                "path": str(
                    JASPAR_PFM
                ),
                "url": JASPAR_PFM_URL,
                "downloaded_this_run": downloaded_jaspar,
                "bytes": JASPAR_PFM.stat().st_size,
                "sha256": sha256_file(
                    JASPAR_PFM
                ),
            },
        ]
    )

    reference_audit.to_csv(
        OUTPUT_REFERENCE_AUDIT,
        sep="\t",
        index=False,
    )

    print_df(
        reference_audit.set_index(
            "resource"
        ),
        digits=0,
    )

    # -----------------------------------------------------------------
    # Load frozen F19/F20 data.
    # -----------------------------------------------------------------

    section(
        "2. LOAD FROZEN F20 CANDIDATES / RNA SUPPORT"
    )

    all_atac = pd.read_csv(
        F20_ALL_ATAC,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    candidates = pd.read_csv(
        F20_ATAC_CANDIDATES,
        sep="\t",
        low_memory=False,
    )

    rna = pd.read_csv(
        F19_RNA,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    rna_candidates = pd.read_csv(
        F20_RNA_CANDIDATES,
        sep="\t",
        low_memory=False,
    )

    cross_modal = pd.read_csv(
        F20_CROSS_MODAL,
        sep="\t",
        low_memory=False,
    )

    print(
        f"All annotated ATAC peaks: {len(all_atac):,}"
    )

    print(
        f"Frozen ATAC-opening candidates: {len(candidates):,}"
    )

    print(
        f"RNA features: {len(rna):,}"
    )

    print(
        f"F20 cross-modal proximity candidate genes: {len(cross_modal):,}"
    )

    if len(
        candidates
    ) != EXPECTED_ATAC_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_ATAC_CANDIDATES} frozen ATAC candidates; "
            f"found {len(candidates)}."
        )

    # -----------------------------------------------------------------
    # Build initial background pool.
    # -----------------------------------------------------------------

    section(
        "3. BUILD MOTIF-AGNOSTIC ACCESSIBILITY / PROXIMITY BACKGROUND POOL"
    )

    (
        candidates,
        initial_pool,
        initial_pool_audit,
    ) = build_initial_background_pool(
        all_atac,
        candidates,
    )

    print(
        f"Initial candidate-matched background pool: "
        f"{len(initial_pool):,} peaks"
    )

    # -----------------------------------------------------------------
    # Extract sequences / GC.
    # -----------------------------------------------------------------

    section(
        "4. EXTRACT mm10 SEQUENCES AND CALCULATE GC CONTENT"
    )

    genome = py2bit.open(
        str(
            MM10_2BIT
        )
    )

    candidate_seq_qc, candidate_sequences = extract_sequences(
        candidates[
            [
                "peak_index_zero_based",
                "chrom",
                "start",
                "end",
            ]
        ],
        genome,
    )

    pool_seq_qc, pool_sequences = extract_sequences(
        initial_pool[
            [
                "peak_index_zero_based",
                "chrom",
                "start",
                "end",
            ]
        ],
        genome,
    )

    genome.close()

    candidates = candidates.merge(
        candidate_seq_qc,
        on="peak_index_zero_based",
        how="left",
        validate="one_to_one",
    )

    initial_pool = initial_pool.merge(
        pool_seq_qc,
        on="peak_index_zero_based",
        how="left",
        validate="one_to_one",
    )

    candidates = candidates.loc[
        candidates[
            "sequence_valid"
        ]
    ].copy()

    initial_pool = initial_pool.loc[
        initial_pool[
            "sequence_valid"
        ]
    ].copy()

    print(
        f"Candidate sequences valid: "
        f"{len(candidates):,}/{EXPECTED_ATAC_CANDIDATES:,}"
    )

    print(
        f"Initial background sequences valid: "
        f"{len(initial_pool):,}"
    )

    # -----------------------------------------------------------------
    # Final GC matching.
    # -----------------------------------------------------------------

    section(
        "5. FINAL GC-MATCHED BACKGROUND"
    )

    final_background, gc_match_audit = final_gc_matched_background(
        candidates,
        initial_pool,
    )

    final_background_ratio = (
        len(
            final_background
        )
        / len(
            candidates
        )
    )

    smd_pre = standardized_mean_difference(
        candidates[
            "pre_ATAC_baseline_accessibility_fraction"
        ],
        final_background[
            "pre_ATAC_baseline_accessibility_fraction"
        ],
    )

    smd_gc = standardized_mean_difference(
        candidates[
            "GC_fraction"
        ],
        final_background[
            "GC_fraction"
        ],
    )

    qc_summary = pd.DataFrame(
        [
            {
                "candidate_n": len(
                    candidates
                ),
                "background_n": len(
                    final_background
                ),
                "background_to_candidate_ratio": final_background_ratio,
                "candidate_mean_pre_accessibility": float(
                    candidates[
                        "pre_ATAC_baseline_accessibility_fraction"
                    ].mean()
                ),
                "background_mean_pre_accessibility": float(
                    final_background[
                        "pre_ATAC_baseline_accessibility_fraction"
                    ].mean()
                ),
                "SMD_pre_accessibility": smd_pre,
                "candidate_mean_GC": float(
                    candidates[
                        "GC_fraction"
                    ].mean()
                ),
                "background_mean_GC": float(
                    final_background[
                        "GC_fraction"
                    ].mean()
                ),
                "SMD_GC": smd_gc,
            }
        ]
    )

    print_df(
        qc_summary.T,
        digits=6,
    )

    background_qc = pd.concat(
        [
            initial_pool_audit.assign(
                audit_stage="initial_pool"
            ),
            gc_match_audit.assign(
                audit_stage="final_gc_match"
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    background_qc.to_csv(
        OUTPUT_BACKGROUND_QC,
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Parse JASPAR / expressed-TF universe.
    # -----------------------------------------------------------------

    section(
        "6. BUILD EXPRESSED SINGLE-TF JASPAR2026 PRIMARY MOTIF UNIVERSE"
    )

    all_motifs = parse_jaspar_pfm(
        JASPAR_PFM
    )

    motif_universe, eligible_motifs = build_primary_motif_universe(
        all_motifs,
        rna,
        rna_candidates,
    )

    motif_universe.to_csv(
        OUTPUT_MOTIF_UNIVERSE,
        sep="\t",
        index=False,
    )

    print(
        f"JASPAR2026 CORE vertebrate motifs parsed: "
        f"{len(all_motifs):,}"
    )

    print(
        f"Eligible expressed single-TF motifs: "
        f"{len(eligible_motifs):,}"
    )

    subsection(
        "Primary motif-universe summary"
    )

    universe_summary = pd.DataFrame(
        [
            {
                "all_JASPAR_motifs": len(
                    all_motifs
                ),
                "single_TF_motifs": int(
                    motif_universe[
                        "single_tf_motif"
                    ].sum()
                ),
                "mapped_to_RNA_gene": int(
                    motif_universe[
                        "mapped_to_rna_gene"
                    ].sum()
                ),
                "TF_expressed_pre_or_ATAC_event": int(
                    motif_universe[
                        "TF_expressed_pre_or_ATAC_event"
                    ].sum()
                ),
                "eligible_primary_motifs": len(
                    eligible_motifs
                ),
            }
        ]
    )

    print_df(
        universe_summary.T,
        digits=0,
    )

    if len(
        eligible_motifs
    ) < 25:
        raise RuntimeError(
            f"Primary motif universe unexpectedly small: "
            f"{len(eligible_motifs)} motifs."
        )

    # -----------------------------------------------------------------
    # Scan motifs.
    # -----------------------------------------------------------------

    section(
        "7. SCAN CANDIDATE AND MATCHED-BACKGROUND PEAKS WITH MOODS"
    )

    scanner = prepare_moods_scanner(
        eligible_motifs
    )

    candidate_presence, candidate_hits = scan_peak_sequences(
        candidates,
        candidate_sequences,
        eligible_motifs,
        scanner,
    )

    background_presence, _ = scan_peak_sequences(
        final_background,
        pool_sequences,
        eligible_motifs,
        scanner,
    )

    # -----------------------------------------------------------------
    # Enrichment.
    # -----------------------------------------------------------------

    section(
        "8. PRIMARY MOTIF ENRICHMENT"
    )

    enrichment = motif_enrichment(
        candidate_presence,
        background_presence,
        eligible_motifs,
        motif_universe,
    )

    enrichment.to_csv(
        OUTPUT_ENRICHMENT,
        sep="\t",
        index=False,
    )

    enriched = enrichment.loc[
        enrichment[
            "enriched_motif"
        ]
    ].copy()

    enriched.to_csv(
        OUTPUT_ENRICHED_TF,
        sep="\t",
        index=False,
    )

    print(
        f"Motifs tested: {len(enrichment):,}"
    )

    print(
        f"Enriched motifs "
        f"(FDR<=0.05, OR>=1.5, candidate hits>=5): "
        f"{len(enriched):,}"
    )

    subsection(
        "Top 40 motif-enrichment results"
    )

    display_columns = [
        "motif_rank",
        "motif_id",
        "jaspar_tf_name",
        "mapped_gene_symbol",
        "candidate_hit_n",
        "candidate_hit_fraction",
        "background_hit_n",
        "background_hit_fraction",
        "odds_ratio",
        "fisher_p",
        "BH_FDR",
        "enriched_motif",
        "TF_pre_ATAC_CPM",
        "TF_ATAC_event_CPM",
        "TF_RNA_event_CPM",
        "TF_ATAC_event_detected_fraction",
        "TF_is_F20_later_RNA_candidate",
        "TF_is_F20_ATAC_stage_RNA_candidate",
    ]

    print_df(
        enrichment.head(
            40
        )[
            display_columns
        ].set_index(
            "motif_rank"
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Keep opening-peak hits for enriched motifs only.
    # -----------------------------------------------------------------

    section(
        "9. MAP ENRICHED MOTIFS BACK TO FROZEN ATAC-OPENING PEAKS"
    )

    enriched_ids = set(
        enriched[
            "motif_id"
        ].astype(str)
    )

    if candidate_hits.empty or not enriched_ids:
        enriched_peak_hits = pd.DataFrame(
            columns=[
                "peak_index_zero_based",
                "motif_id",
                "jaspar_tf_name",
                "mapped_gene_symbol",
                "forward_hit_count",
                "reverse_hit_count",
                "total_hit_count",
                "chrom",
                "start",
                "end",
                "nearest_gene_symbol",
                "proximity_class",
                "delta_accessibility_ATACevent_vs_pre",
                "ATAC_event_accessibility_fraction",
            ]
        )

    else:
        enriched_peak_hits = candidate_hits.loc[
            candidate_hits[
                "motif_id"
            ].astype(str).isin(
                enriched_ids
            )
        ].copy()

        enriched_peak_hits = enriched_peak_hits.merge(
            candidates[
                [
                    "peak_index_zero_based",
                    "chrom",
                    "start",
                    "end",
                    "nearest_gene_symbol",
                    "proximity_class",
                    "delta_accessibility_ATACevent_vs_pre",
                    "ATAC_event_accessibility_fraction",
                ]
            ],
            on="peak_index_zero_based",
            how="left",
            validate="many_to_one",
        )

    enriched_peak_hits.to_csv(
        OUTPUT_PEAK_HITS,
        sep="\t",
        index=False,
        compression="gzip",
    )

    print(
        f"Opening-peak/enriched-motif hit rows: "
        f"{len(enriched_peak_hits):,}"
    )

    # -----------------------------------------------------------------
    # Cross-modal regulatory plausibility.
    # -----------------------------------------------------------------

    section(
        "10. CROSS-MODAL REGULATORY PLAUSIBILITY FOR F20 PROXIMITY GENES"
    )

    cross_genes = set(
        cross_modal[
            "gene"
        ].astype(str)
    )

    if enriched_peak_hits.empty:
        plausibility = pd.DataFrame(
            columns=[
                "gene",
                "peak_index_zero_based",
                "chrom",
                "start",
                "end",
                "motif_id",
                "motif_TF",
                "motif_TF_gene",
                "total_hit_count",
                "delta_accessibility_ATACevent_vs_pre",
                "ATAC_event_accessibility_fraction",
            ]
        )

    else:
        plausibility = enriched_peak_hits.loc[
            enriched_peak_hits[
                "nearest_gene_symbol"
            ].astype(str).isin(
                cross_genes
            )
        ].copy()

        plausibility = plausibility.rename(
            columns={
                "nearest_gene_symbol": "gene",
                "jaspar_tf_name": "motif_TF",
                "mapped_gene_symbol": "motif_TF_gene",
            }
        )

        plausibility = plausibility.merge(
            cross_modal,
            on="gene",
            how="left",
            validate="many_to_one",
            suffixes=(
                "",
                "_F20",
            ),
        )

        plausibility = plausibility.sort_values(
            by=[
                "gene",
                "delta_accessibility_ATACevent_vs_pre",
                "total_hit_count",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        ).reset_index(
            drop=True
        )

    plausibility.to_csv(
        OUTPUT_CROSS_MODAL,
        sep="\t",
        index=False,
    )

    print(
        f"F20 cross-modal proximity genes with >=1 enriched-TF motif "
        f"in their assigned opening peak(s): "
        f"{plausibility['gene'].nunique() if not plausibility.empty else 0}"
    )

    if not plausibility.empty:
        print_df(
            plausibility.head(
                80
            )[
                [
                    "gene",
                    "chrom",
                    "start",
                    "end",
                    "motif_id",
                    "motif_TF",
                    "motif_TF_gene",
                    "total_hit_count",
                    "delta_accessibility_ATACevent_vs_pre",
                    "ATAC_event_accessibility_fraction",
                    "RNA_event_CPM",
                    "RNA_event_detected_fraction",
                    "log2FC_RNAevent_vs_ATACevent_CPM",
                ]
            ].set_index(
                [
                    "gene",
                    "motif_TF",
                ]
            ),
            digits=6,
        )

    # -----------------------------------------------------------------
    # Safeguards.
    # -----------------------------------------------------------------

    section(
        "11. PHASE F21 REGULATORY-PLAUSIBILITY SAFEGUARDS"
    )

    candidate_sequence_fraction = (
        len(
            candidates
        )
        / EXPECTED_ATAC_CANDIDATES
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    "JASPAR2026 CORE vertebrate motif reference available"
                ),
                "pass": JASPAR_PFM.exists(),
            },
            {
                "criterion": (
                    "UCSC mm10 2bit reference available"
                ),
                "pass": MM10_2BIT.exists(),
            },
            {
                "criterion": (
                    ">=99% frozen ATAC-opening candidates have valid sequence"
                ),
                "pass": (
                    candidate_sequence_fraction
                    >= 0.99
                ),
            },
            {
                "criterion": (
                    f"Matched background ratio >= "
                    f"{MIN_FINAL_BACKGROUND_RATIO:.1f}:1"
                ),
                "pass": (
                    final_background_ratio
                    >= MIN_FINAL_BACKGROUND_RATIO
                ),
            },
            {
                "criterion": (
                    f"|SMD| baseline accessibility <= "
                    f"{MAX_ABS_SMD_PRE_ACCESS:.2f}"
                ),
                "pass": (
                    abs(
                        smd_pre
                    )
                    <= MAX_ABS_SMD_PRE_ACCESS
                ),
            },
            {
                "criterion": (
                    f"|SMD| GC <= {MAX_ABS_SMD_GC:.2f}"
                ),
                "pass": (
                    abs(
                        smd_gc
                    )
                    <= MAX_ABS_SMD_GC
                ),
            },
            {
                "criterion": (
                    "Primary motif universe contains >=25 expressed "
                    "single-TF motifs"
                ),
                "pass": (
                    len(
                        eligible_motifs
                    )
                    >= 25
                ),
            },
            {
                "criterion": (
                    "BH-FDR applied across complete primary motif universe"
                ),
                "pass": (
                    len(
                        enrichment
                    )
                    == len(
                        eligible_motifs
                    )
                ),
            },
            {
                "criterion": (
                    "No motif information used for background matching"
                ),
                "pass": True,
            },
            {
                "criterion": (
                    "No timing/null/GDIS analysis reopened"
                ),
                "pass": True,
            },
            {
                "criterion": (
                    "Enriched motifs interpreted as regulatory plausibility, "
                    "not causal TF-target proof"
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
        "12. SAVE F21 MANIFEST"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F21",
        "created_utc": utc_now_iso(),
        "purpose": (
            "Motif/TF regulatory plausibility analysis of the frozen F20 "
            "ATAC-opening candidate set."
        ),
        "references": {
            "mm10_2bit": {
                "url": MM10_2BIT_URL,
                "path": str(
                    MM10_2BIT
                ),
                "sha256": sha256_file(
                    MM10_2BIT
                ),
            },
            "JASPAR2026_CORE_vertebrates_nonredundant": {
                "url": JASPAR_PFM_URL,
                "path": str(
                    JASPAR_PFM
                ),
                "sha256": sha256_file(
                    JASPAR_PFM
                ),
            },
        },
        "background_matching": {
            "pre_access_bin_width": PRE_ACCESS_BIN_WIDTH,
            "gc_bin_edges": GC_BIN_EDGES.tolist(),
            "initial_pool_multiplier": (
                INITIAL_BACKGROUND_POOL_MULTIPLIER
            ),
            "final_background_per_candidate_target": (
                FINAL_BACKGROUND_PER_CANDIDATE
            ),
            "candidate_n": len(
                candidates
            ),
            "background_n": len(
                final_background
            ),
            "background_ratio": final_background_ratio,
            "SMD_pre_accessibility": smd_pre,
            "SMD_GC": smd_gc,
            "motif_information_used_for_matching": False,
        },
        "motif_universe": {
            "JASPAR_collection": "CORE",
            "taxonomic_group": "vertebrates",
            "non_redundant": True,
            "single_TF_only": True,
            "TF_min_pre_or_ATAC_CPM": TF_MIN_CPM,
            "TF_min_pre_or_ATAC_detected_fraction": (
                TF_MIN_DETECTED_FRACTION
            ),
            "eligible_motif_n": len(
                eligible_motifs
            ),
        },
        "motif_scanning": {
            "engine": "MOODS-python",
            "both_strands": True,
            "match_pvalue": MOTIF_MATCH_PVALUE,
            "flat_background": [
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            "pseudocount": MOTIF_PSEUDOCOUNT,
        },
        "enrichment": {
            "test": "one-sided Fisher exact",
            "alternative": "greater",
            "multiple_testing": "Benjamini-Hochberg",
            "enriched_FDR_threshold": ENRICHMENT_FDR,
            "enriched_min_odds_ratio": ENRICHMENT_MIN_OR,
            "enriched_min_candidate_hits": (
                ENRICHMENT_MIN_CANDIDATE_HITS
            ),
            "tested_motifs": len(
                enrichment
            ),
            "enriched_motifs": len(
                enriched
            ),
        },
        "counts": {
            "F20_cross_modal_proximity_genes": len(
                cross_modal
            ),
            "cross_modal_genes_with_enriched_motif_support": (
                int(
                    plausibility[
                        "gene"
                    ].nunique()
                )
                if not plausibility.empty
                else 0
            ),
            "opening_peak_enriched_motif_hit_rows": len(
                enriched_peak_hits
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
            "background_qc": {
                "file": str(
                    OUTPUT_BACKGROUND_QC
                ),
                "sha256": sha256_file(
                    OUTPUT_BACKGROUND_QC
                ),
            },
            "motif_universe": {
                "file": str(
                    OUTPUT_MOTIF_UNIVERSE
                ),
                "sha256": sha256_file(
                    OUTPUT_MOTIF_UNIVERSE
                ),
            },
            "motif_enrichment": {
                "file": str(
                    OUTPUT_ENRICHMENT
                ),
                "sha256": sha256_file(
                    OUTPUT_ENRICHMENT
                ),
            },
            "enriched_TF_candidates": {
                "file": str(
                    OUTPUT_ENRICHED_TF
                ),
                "sha256": sha256_file(
                    OUTPUT_ENRICHED_TF
                ),
            },
            "opening_peak_enriched_motif_hits": {
                "file": str(
                    OUTPUT_PEAK_HITS
                ),
                "sha256": sha256_file(
                    OUTPUT_PEAK_HITS
                ),
            },
            "cross_modal_regulatory_plausibility": {
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
            "ATAC_candidate_set_changed": False,
            "gdis_recomputed": False,
            "cmil_recomputed": False,
            "new_timing_null_performed": False,
            "cell_level_DE_DA_test_performed": False,
            "causal_TF_target_claimed": False,
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
        OUTPUT_BACKGROUND_QC,
        OUTPUT_MOTIF_UNIVERSE,
        OUTPUT_ENRICHMENT,
        OUTPUT_ENRICHED_TF,
        OUTPUT_PEAK_HITS,
        OUTPUT_CROSS_MODAL,
        OUTPUT_MANIFEST,
    ]:
        print(
            path
        )

    # -----------------------------------------------------------------
    # Final verdict.
    # -----------------------------------------------------------------

    section(
        "13. PHASE F21 DECISION"
    )

    if all_pass:
        print(
            "PHASE F21 VERDICT: GO — SOURCE-COMPATIBLE MOTIF/TF "
            "REGULATORY-PLAUSIBILITY ANALYSIS COMPLETED"
        )

        print()

        if len(
            enriched
        ) > 0:
            print(
                f"{len(enriched)} motif(s) satisfy the prespecified "
                "enrichment criteria."
            )

            print()

            print(
                "These motifs may be interpreted as candidate regulatory "
                "programs associated with the frozen ATAC-opening event."
            )
        else:
            print(
                "No motif satisfies the prespecified enrichment criteria."
            )

            print()

            print(
                "This is a valid negative mechanistic result and should not "
                "be rescued by changing thresholds."
            )

        print()

        print(
            "Timing inference remains frozen."
        )

        print()

        print(
            "Next phase should review biological identities of any enriched "
            "TFs and cross-modal genes before deciding whether a further "
            "TF-target network analysis is justified."
        )

    else:
        print(
            "PHASE F21 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not interpret motif enrichment until failed reference, "
            "background-matching, or motif-universe safeguards are resolved."
        )

    print()

    print(
        "No timing analysis was performed."
    )

    print(
        "No causal TF-target claim was made."
    )

    print(
        "No universal chromatin-priming claim was made."
    )

    line("=")


if __name__ == "__main__":
    main()

