#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
32_phase_f17_shareseq_irs_frozen_pair_bootstrap.py
==================================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F17
---------
Bootstrap the ALREADY-FROZEN localized RNA/ATAC transition-energy event pair.

FROZEN EVENT PAIR FROM F16
--------------------------
The pair identity was selected BEFORE bootstrap and independently by modality.

RNA:
    selected transition-energy family from F16

ATAC:
    selected transition-energy family from F16

Family selection is NOT repeated inside bootstrap replicates.

INFERENTIAL TARGET
------------------
The PRIMARY 400-cell / 100-cell-step window analysis is the inferential
target.

The observed statistic is therefore:

    CMIL_primary =
        RNA primary-window selected-family peak parameter
        -
        ATAC primary-window selected-family peak parameter

Positive:
    ATAC event occurs earlier on the frozen common pseudotime.

The three-window FAMILY-CENTER separation remains descriptive and is not mixed
with the primary bootstrap estimator.

BOOTSTRAP DESIGN
----------------
Number of bootstrap replicates:
    500

Paired-cell bootstrap:
    RNA and ATAC are always resampled using the SAME cell indices.

Pseudotime-stratified:
    The 5,050 frozen IRS-branch cells are ordered by the frozen F13 common
    pseudotime and divided into 20 contiguous equal-count strata.

Within each stratum:
    sample the same number of paired cells WITH replacement.

Then:
    concatenate sampled paired cells;
    sort them by their ORIGINAL frozen common pseudotime;
    construct the primary 400/100 common windows;
    run pyGDIS using the frozen lambda_t=0 configuration.

This preserves:
    - paired RNA/ATAC identity;
    - broad pseudotime coverage;
    - the original RNA-derived common clock;
    - the primary window design.

It does NOT re-estimate pseudotime.

FROZEN FAMILY RECOVERY RULE
---------------------------
For each modality independently, a bootstrap replicate must recover a
transition-energy local maximum that is:

    1. MATERIAL under the exact F15 rule:
           prominence >= max(0.005, 0.05 * profile range)

    2. INTERIOR under the exact F15 edge rule:
           not within first/last 3 profile points
           AND not within 2 * median parameter spacing of profile boundary

    3. within the frozen F15/F16 family tolerance:
           |peak parameter - frozen family center| <= 0.02

If multiple eligible peaks occur:
    choose the one closest to that modality's own frozen family center;
    tie-break by larger prominence;
    then smaller parameter.

RNA recovery never uses ATAC position.
ATAC recovery never uses RNA position.

If one modality is not recovered:
    the replicate is a paired NONDETECTION.
    No alternative family is substituted.

BOOTSTRAP INFERENCE
-------------------
Among paired detections, F17 reports:

    mean and median CMIL
    95% percentile CI
    P_boot(CMIL > 0)
    P_boot(CMIL < 0)

Detection rates:
    RNA detection rate
    ATAC detection rate
    paired detection rate

PRE-SPECIFIED SUPPORT CRITERIA
------------------------------
These criteria are frozen before viewing F17 results:

    paired detection rate >= 0.80
    95% percentile CI lower bound > 0
    P_boot(CMIL > 0) >= 0.95

All three are required for:

    BOOTSTRAP SUPPORT FOR ATAC-EARLIER DIRECTION

Failure of any criterion means the timing lead remains suggestive/unsupported.

IMPORTANT
---------
F17 does NOT:
    - change the event pair;
    - change the branch;
    - change the trajectory;
    - change the representations;
    - run a cross-modal alignment/permutation null;
    - claim chromatin priming.

A separate null/alignment phase is required even if F17 passes.

INPUTS
------
data/GSE140203/representations_f12/
    f12_irs_branch_cells.tsv.gz
    f12_rna_irs_10d.npy
    f12_atac_irs_10d.npy

data/GSE140203/representations_f13/
    f13_irs_common_pseudotime.tsv.gz

data/GSE140203/representations_f15b/
    f15b_manifest.json

data/GSE140203/representations_f16/
    f16_frozen_transition_energy_pair.tsv
    f16_window_specific_pairing.tsv
    f16_manifest.json

OUTPUTS
-------
data/GSE140203/representations_f17/
    f17_bootstrap_replicates.tsv.gz
    f17_bootstrap_summary.tsv
    f17_manifest.json

RUN
---
    python 32_phase_f17_shareseq_irs_frozen_pair_bootstrap.py

DEPENDENCIES
------------
numpy
pandas
scipy
pygdis==1.0.0
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scipy.signal import find_peaks, peak_prominences

try:
    import gdis
    from gdis import GDIS, GDISConfig
except ImportError as exc:
    raise SystemExit(
        "\nERROR: Phase F17 requires pyGDIS 1.0.0.\n\n"
        'Install with:\n\n'
        '    python -m pip install "pygdis==1.0.0"\n'
    ) from exc


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE140203"

DATA_DIR = Path("data") / ACCESSION
F12_DIR = DATA_DIR / "representations_f12"
F13_DIR = DATA_DIR / "representations_f13"
F15B_DIR = DATA_DIR / "representations_f15b"
F16_DIR = DATA_DIR / "representations_f16"
F17_DIR = DATA_DIR / "representations_f17"

F12_CELLS = F12_DIR / "f12_irs_branch_cells.tsv.gz"
F12_RNA = F12_DIR / "f12_rna_irs_10d.npy"
F12_ATAC = F12_DIR / "f12_atac_irs_10d.npy"

F13_CLOCK = F13_DIR / "f13_irs_common_pseudotime.tsv.gz"

F15B_MANIFEST = F15B_DIR / "f15b_manifest.json"

F16_PAIR = F16_DIR / "f16_frozen_transition_energy_pair.tsv"
F16_WINDOWS = F16_DIR / "f16_window_specific_pairing.tsv"
F16_MANIFEST = F16_DIR / "f16_manifest.json"

OUTPUT_REPLICATES = F17_DIR / "f17_bootstrap_replicates.tsv.gz"
OUTPUT_SUMMARY = F17_DIR / "f17_bootstrap_summary.tsv"
OUTPUT_MANIFEST = F17_DIR / "f17_manifest.json"


# =====================================================================
# FROZEN DESIGN
# =====================================================================

EXPECTED_PYGDIS_VERSION = "1.0.0"

EXPECTED_CELLS = 5_050
EXPECTED_DIMS = 10

BARCODE_COL = "rna.bc"
STATE_COL = "celltype"
PSEUDOTIME_COL = "common_pseudotime"

PRIMARY_WINDOW_CONFIG = "primary_w400_s100"
WINDOW_SIZE = 400
STEP_SIZE = 100

N_BOOTSTRAP = 500
N_STRATA = 20
RANDOM_SEED = 785

FAMILY_MATCH_TOLERANCE = 0.02

MATERIAL_PROMINENCE_ABSOLUTE_FLOOR = 0.005
MATERIAL_PROMINENCE_RANGE_FRACTION = 0.05

EDGE_POINT_COUNT = 3
EDGE_SPACING_MULTIPLIER = 2.0

MIN_PAIRED_DETECTION_RATE = 0.80
MIN_DIRECTIONAL_BOOTSTRAP_SUPPORT = 0.95

PRIMARY_TRANSITION_WEIGHT = 0.00

GDIS_FIXED = {
    "alpha_j": 0.42,
    "alpha_s": 0.33,
    "alpha_a": 0.25,
    "k_j": 3.0,
    "k_s": 2.6,
    "k_a": 2.4,
    "hill_gamma": 0.72,
    "hill_c": 0.22,
    "complexity_gain": 0.055,
    "temporal_gain": 0.055,
    "temporal_threshold": 0.60,
    "transition_width_fraction": 0.09,
    "critical_value": None,
    "smoothing_window": 9,
    "smoothing_polynomial_order": 3,
    "max_sustained": 0.985,
}

MAX_RESAMPLE_ATTEMPTS_PER_REPLICATE = 25


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
            block = handle.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def bool_value(value):
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(
            value
        )

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


# =====================================================================
# INPUTS / ALIGNMENT
# =====================================================================

def require_inputs():
    required = [
        F12_CELLS,
        F12_RNA,
        F12_ATAC,
        F13_CLOCK,
        F15B_MANIFEST,
        F16_PAIR,
        F16_WINDOWS,
        F16_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required F12/F13/F15b/F16 input(s):\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )


def load_and_align():
    cells = pd.read_csv(
        F12_CELLS,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    rna = np.load(
        F12_RNA
    ).astype(
        np.float64
    )

    atac = np.load(
        F12_ATAC
    ).astype(
        np.float64
    )

    clock = pd.read_csv(
        F13_CLOCK,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    pair = pd.read_csv(
        F16_PAIR,
        sep="\t",
        low_memory=False,
    )

    window_pairing = pd.read_csv(
        F16_WINDOWS,
        sep="\t",
        low_memory=False,
    )

    with F15B_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        f15b_manifest = json.load(
            handle
        )

    with F16_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        f16_manifest = json.load(
            handle
        )

    if len(
        cells
    ) != EXPECTED_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_CELLS:,} cells; found {len(cells):,}."
        )

    if rna.shape != (
        EXPECTED_CELLS,
        EXPECTED_DIMS,
    ):
        raise RuntimeError(
            f"Unexpected RNA shape: {rna.shape}"
        )

    if atac.shape != (
        EXPECTED_CELLS,
        EXPECTED_DIMS,
    ):
        raise RuntimeError(
            f"Unexpected ATAC shape: {atac.shape}"
        )

    if pair.shape[
        0
    ] != 1:
        raise RuntimeError(
            "F16 frozen-pair table must contain exactly one row."
        )

    if cells[
        BARCODE_COL
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate barcodes in F12 cell table."
        )

    if clock[
        BARCODE_COL
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate barcodes in F13 clock table."
        )

    clock_indexed = clock.set_index(
        BARCODE_COL,
        drop=False,
    )

    missing_clock = [
        barcode
        for barcode in cells[
            BARCODE_COL
        ].astype(str)
        if barcode not in clock_indexed.index
    ]

    if missing_clock:
        raise RuntimeError(
            f"{len(missing_clock)} F12 cells missing from F13 clock."
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
            "Common pseudotime contains NaN/Inf."
        )

    if not np.isfinite(
        rna
    ).all() or not np.isfinite(
        atac
    ).all():
        raise RuntimeError(
            "Frozen representation contains NaN/Inf."
        )

    pair_row = pair.iloc[
        0
    ]

    rna_center = float(
        pair_row[
            "RNA_family_center_parameter"
        ]
    )

    atac_center = float(
        pair_row[
            "ATAC_family_center_parameter"
        ]
    )

    # Verify F15b tolerance, rather than silently introducing a new one.
    manifest_tolerance = float(
        f15b_manifest[
            "transition_energy_families"
        ][
            "family_tolerance"
        ]
    )

    if not np.isclose(
        manifest_tolerance,
        FAMILY_MATCH_TOLERANCE,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"F17 tolerance {FAMILY_MATCH_TOLERANCE} differs from "
            f"F15b frozen tolerance {manifest_tolerance}."
        )

    # Verify F16 did not bootstrap already.
    guardrails = f16_manifest.get(
        "guardrails",
        {},
    )

    if bool(
        guardrails.get(
            "bootstrap_performed",
            False,
        )
    ):
        raise RuntimeError(
            "F16 manifest unexpectedly reports bootstrap already performed."
        )

    primary_row = window_pairing.loc[
        window_pairing[
            "window_config"
        ].astype(str)
        == PRIMARY_WINDOW_CONFIG
    ]

    if len(
        primary_row
    ) != 1:
        raise RuntimeError(
            "Could not identify unique F16 primary-window pairing row."
        )

    primary_row = primary_row.iloc[
        0
    ]

    if not bool_value(
        primary_row[
            "paired_observation_available"
        ]
    ):
        raise RuntimeError(
            "F16 primary-window frozen pair was not jointly observed."
        )

    observed_primary = float(
        primary_row[
            "RNA_minus_ATAC_parameter"
        ]
    )

    observed_rna_parameter = float(
        primary_row[
            "RNA_peak_parameter"
        ]
    )

    observed_atac_parameter = float(
        primary_row[
            "ATAC_peak_parameter"
        ]
    )

    return {
        "cells": cells,
        "rna": rna,
        "atac": atac,
        "pseudotime": pseudotime,
        "rna_family_center": rna_center,
        "atac_family_center": atac_center,
        "observed_primary_separation": observed_primary,
        "observed_primary_rna_parameter": observed_rna_parameter,
        "observed_primary_atac_parameter": observed_atac_parameter,
        "f15b_manifest": f15b_manifest,
        "f16_manifest": f16_manifest,
    }


# =====================================================================
# GDIS CONFIG / WINDOWS
# =====================================================================

def make_gdis_config():
    config = GDISConfig(
        alpha_j=GDIS_FIXED[
            "alpha_j"
        ],
        alpha_s=GDIS_FIXED[
            "alpha_s"
        ],
        alpha_a=GDIS_FIXED[
            "alpha_a"
        ],
        k_j=GDIS_FIXED[
            "k_j"
        ],
        k_s=GDIS_FIXED[
            "k_s"
        ],
        k_a=GDIS_FIXED[
            "k_a"
        ],
        hill_gamma=GDIS_FIXED[
            "hill_gamma"
        ],
        hill_c=GDIS_FIXED[
            "hill_c"
        ],
        complexity_gain=GDIS_FIXED[
            "complexity_gain"
        ],
        temporal_gain=GDIS_FIXED[
            "temporal_gain"
        ],
        temporal_threshold=GDIS_FIXED[
            "temporal_threshold"
        ],
        transition_weight=PRIMARY_TRANSITION_WEIGHT,
        transition_width_fraction=GDIS_FIXED[
            "transition_width_fraction"
        ],
        critical_value=None,
        smoothing_window=GDIS_FIXED[
            "smoothing_window"
        ],
        smoothing_polynomial_order=GDIS_FIXED[
            "smoothing_polynomial_order"
        ],
        max_sustained=GDIS_FIXED[
            "max_sustained"
        ],
    )

    config.validate()

    return config


def construct_primary_windows(
    pseudotime_sorted,
):
    n_cells = len(
        pseudotime_sorted
    )

    starts = list(
        range(
            0,
            n_cells
            - WINDOW_SIZE
            + 1,
            STEP_SIZE,
        )
    )

    parameters = []

    slices = []

    for start in starts:
        stop = (
            start
            + WINDOW_SIZE
        )

        parameters.append(
            float(
                np.median(
                    pseudotime_sorted[
                        start:
                        stop
                    ]
                )
            )
        )

        slices.append(
            (
                start,
                stop,
            )
        )

    parameters = np.asarray(
        parameters,
        dtype=float,
    )

    if np.any(
        np.diff(
            parameters
        )
        <= 0
    ):
        return None

    return {
        "parameters": parameters,
        "slices": slices,
    }


def run_transition_energy_profile(
    representation_sorted,
    windows,
):
    trajectories = [
        representation_sorted[
            start:
            stop,
            :
        ]
        for start, stop in windows[
            "slices"
        ]
    ]

    config = make_gdis_config()

    model = GDIS(
        config=config
    )

    result = model.fit_transform(
        trajectories,
        windows[
            "parameters"
        ],
        jacobian_function=None,
        critical_value=None,
    )

    transition_energy = np.asarray(
        result.components[
            "transition_energy"
        ],
        dtype=float,
    )

    parameters = np.asarray(
        result.parameters,
        dtype=float,
    )

    if not np.isfinite(
        transition_energy
    ).all():
        raise RuntimeError(
            "Bootstrap transition-energy profile contains NaN/Inf."
        )

    if not np.allclose(
        parameters,
        windows[
            "parameters"
        ],
        atol=1e-14,
        rtol=0.0,
    ):
        raise RuntimeError(
            "pyGDIS parameters changed inside bootstrap."
        )

    return (
        parameters,
        transition_energy,
    )


# =====================================================================
# FIXED PEAK RECOVERY
# =====================================================================

def recover_frozen_family_peak(
    parameters,
    values,
    frozen_center,
):
    """
    Recover only the prespecified family neighborhood.

    Returns a dict with detected=False when the frozen family is not recovered.
    """
    parameters = np.asarray(
        parameters,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    peak_indices, _ = find_peaks(
        values
    )

    profile_range = float(
        np.max(
            values
        )
        - np.min(
            values
        )
    )

    material_threshold = max(
        MATERIAL_PROMINENCE_ABSOLUTE_FLOOR,
        MATERIAL_PROMINENCE_RANGE_FRACTION
        * profile_range,
    )

    median_spacing = float(
        np.median(
            np.diff(
                parameters
            )
        )
    )

    edge_parameter_threshold = (
        EDGE_SPACING_MULTIPLIER
        * median_spacing
    )

    if len(
        peak_indices
    ) == 0:
        return {
            "detected": False,
            "parameter": np.nan,
            "prominence": np.nan,
            "distance_to_frozen_center": np.nan,
            "n_eligible_peaks": 0,
        }

    prominences, _, _ = peak_prominences(
        values,
        peak_indices,
    )

    candidates = []

    for peak_index, prominence in zip(
        peak_indices,
        prominences,
    ):
        parameter = float(
            parameters[
                peak_index
            ]
        )

        material = bool(
            prominence
            >= material_threshold
        )

        edge_by_index = bool(
            peak_index
            < EDGE_POINT_COUNT
            or peak_index
            >= (
                len(
                    values
                )
                - EDGE_POINT_COUNT
            )
        )

        distance_left = (
            parameter
            - float(
                parameters[
                    0
                ]
            )
        )

        distance_right = (
            float(
                parameters[
                    -1
                ]
            )
            - parameter
        )

        edge_by_parameter = bool(
            min(
                distance_left,
                distance_right,
            )
            <= edge_parameter_threshold
        )

        interior = bool(
            not edge_by_index
            and not edge_by_parameter
        )

        family_match = bool(
            abs(
                parameter
                - frozen_center
            )
            <= FAMILY_MATCH_TOLERANCE
        )

        if (
            material
            and interior
            and family_match
        ):
            candidates.append(
                {
                    "peak_index": int(
                        peak_index
                    ),
                    "parameter": parameter,
                    "prominence": float(
                        prominence
                    ),
                    "distance_to_frozen_center": abs(
                        parameter
                        - frozen_center
                    ),
                }
            )

    if not candidates:
        return {
            "detected": False,
            "parameter": np.nan,
            "prominence": np.nan,
            "distance_to_frozen_center": np.nan,
            "n_eligible_peaks": 0,
        }

    candidate_df = pd.DataFrame(
        candidates
    ).sort_values(
        by=[
            "distance_to_frozen_center",
            "prominence",
            "parameter",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="mergesort",
    )

    selected = candidate_df.iloc[
        0
    ]

    return {
        "detected": True,
        "parameter": float(
            selected[
                "parameter"
            ]
        ),
        "prominence": float(
            selected[
                "prominence"
            ]
        ),
        "distance_to_frozen_center": float(
            selected[
                "distance_to_frozen_center"
            ]
        ),
        "n_eligible_peaks": int(
            len(
                candidate_df
            )
        ),
    }


# =====================================================================
# STRATIFIED PAIRED BOOTSTRAP
# =====================================================================

def make_contiguous_strata(
    sorted_indices,
):
    """
    Split pseudotime-ordered cells into N_STRATA contiguous nearly equal strata.
    """
    chunks = np.array_split(
        sorted_indices,
        N_STRATA,
    )

    if any(
        len(
            chunk
        )
        < WINDOW_SIZE
        / N_STRATA
        for chunk in chunks
    ):
        # Mostly defensive; with 5050 / 20 this is never close to problematic.
        pass

    return chunks


def sample_paired_indices(
    rng,
    strata,
):
    sampled = []

    for stratum in strata:
        draw = rng.choice(
            stratum,
            size=len(
                stratum
            ),
            replace=True,
        )

        sampled.append(
            draw
        )

    return np.concatenate(
        sampled
    )


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F17 — BOOTSTRAP THE FROZEN SHARE-seq IRS TRANSITION-ENERGY PAIR"
    )

    print(
        f"Bootstrap replicates: {N_BOOTSTRAP}"
    )

    print(
        f"Pseudotime strata: {N_STRATA}"
    )

    print(
        "Paired RNA/ATAC resampling: YES"
    )

    print(
        "Pseudotime re-estimation: NO"
    )

    print(
        "Family reselection: NO"
    )

    print()

    print(
        f"Frozen family-match tolerance: ±{FAMILY_MATCH_TOLERANCE:.3f}"
    )

    print()

    print(
        "Primary inferential window design:"
    )

    print(
        f"  {WINDOW_SIZE} cells / {STEP_SIZE} step"
    )

    print()

    print(
        "No alignment/permutation null."
    )

    print(
        "No priming claim."
    )

    require_inputs()

    installed_version = str(
        getattr(
            gdis,
            "__version__",
            "unknown",
        )
    )

    if installed_version != EXPECTED_PYGDIS_VERSION:
        raise RuntimeError(
            f"F17 requires pyGDIS {EXPECTED_PYGDIS_VERSION}; "
            f"found {installed_version}."
        )

    F17_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_and_align()

    cells = data[
        "cells"
    ]

    rna = data[
        "rna"
    ]

    atac = data[
        "atac"
    ]

    pseudotime = data[
        "pseudotime"
    ]

    rna_center = data[
        "rna_family_center"
    ]

    atac_center = data[
        "atac_family_center"
    ]

    observed_primary = data[
        "observed_primary_separation"
    ]

    observed_rna = data[
        "observed_primary_rna_parameter"
    ]

    observed_atac = data[
        "observed_primary_atac_parameter"
    ]

    section(
        "1. VERIFY FROZEN EVENT / PRIMARY ESTIMATOR"
    )

    print(
        f"Frozen RNA family center:  {rna_center:.6f}"
    )

    print(
        f"Frozen ATAC family center: {atac_center:.6f}"
    )

    print()

    print(
        f"Observed primary RNA peak:  {observed_rna:.6f}"
    )

    print(
        f"Observed primary ATAC peak: {observed_atac:.6f}"
    )

    print(
        f"Observed primary separation (RNA - ATAC): "
        f"{observed_primary:.6f}"
    )

    # -----------------------------------------------------------------
    # Sort / strata.
    # -----------------------------------------------------------------

    section(
        "2. FREEZE PSEUDOTIME STRATA FOR PAIRED BOOTSTRAP"
    )

    sorted_original_indices = np.argsort(
        pseudotime,
        kind="mergesort",
    )

    strata = make_contiguous_strata(
        sorted_original_indices
    )

    stratum_rows = []

    for index, stratum in enumerate(
        strata
    ):
        pt = pseudotime[
            stratum
        ]

        stratum_rows.append(
            {
                "stratum": index + 1,
                "n_cells": len(
                    stratum
                ),
                "pseudotime_min": float(
                    np.min(
                        pt
                    )
                ),
                "pseudotime_median": float(
                    np.median(
                        pt
                    )
                ),
                "pseudotime_max": float(
                    np.max(
                        pt
                    )
                ),
            }
        )

    print_df(
        pd.DataFrame(
            stratum_rows
        ).set_index(
            "stratum"
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Bootstrap.
    # -----------------------------------------------------------------

    section(
        "3. RUN 500 PAIRED PSEUDOTIME-STRATIFIED BOOTSTRAPS"
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    replicate_rows = []

    invalid_window_parameter_retries = 0

    for replicate in range(
        1,
        N_BOOTSTRAP
        + 1,
    ):
        successful_draw = False

        for attempt in range(
            1,
            MAX_RESAMPLE_ATTEMPTS_PER_REPLICATE
            + 1,
        ):
            sampled_indices = sample_paired_indices(
                rng,
                strata,
            )

            sampled_pt = pseudotime[
                sampled_indices
            ]

            reorder = np.argsort(
                sampled_pt,
                kind="mergesort",
            )

            sampled_indices_sorted = sampled_indices[
                reorder
            ]

            sampled_pt_sorted = sampled_pt[
                reorder
            ]

            windows = construct_primary_windows(
                sampled_pt_sorted
            )

            if windows is None:
                invalid_window_parameter_retries += 1
                continue

            successful_draw = True
            break

        if not successful_draw:
            raise RuntimeError(
                f"Replicate {replicate}: could not obtain strictly "
                f"increasing bootstrap window parameters after "
                f"{MAX_RESAMPLE_ATTEMPTS_PER_REPLICATE} attempts."
            )

        rna_boot = rna[
            sampled_indices_sorted,
            :
        ]

        atac_boot = atac[
            sampled_indices_sorted,
            :
        ]

        (
            rna_parameters,
            rna_energy,
        ) = run_transition_energy_profile(
            rna_boot,
            windows,
        )

        (
            atac_parameters,
            atac_energy,
        ) = run_transition_energy_profile(
            atac_boot,
            windows,
        )

        if not np.allclose(
            rna_parameters,
            atac_parameters,
            atol=1e-14,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Replicate {replicate}: RNA/ATAC common window "
                f"parameters differ."
            )

        rna_match = recover_frozen_family_peak(
            rna_parameters,
            rna_energy,
            rna_center,
        )

        atac_match = recover_frozen_family_peak(
            atac_parameters,
            atac_energy,
            atac_center,
        )

        paired_detected = bool(
            rna_match[
                "detected"
            ]
            and atac_match[
                "detected"
            ]
        )

        if paired_detected:
            separation = (
                rna_match[
                    "parameter"
                ]
                - atac_match[
                    "parameter"
                ]
            )
        else:
            separation = np.nan

        replicate_rows.append(
            {
                "bootstrap_replicate": replicate,
                "resample_attempt": attempt,
                "RNA_detected": bool(
                    rna_match[
                        "detected"
                    ]
                ),
                "RNA_peak_parameter": rna_match[
                    "parameter"
                ],
                "RNA_peak_prominence": rna_match[
                    "prominence"
                ],
                "RNA_distance_to_frozen_center": rna_match[
                    "distance_to_frozen_center"
                ],
                "RNA_n_eligible_family_peaks": rna_match[
                    "n_eligible_peaks"
                ],
                "ATAC_detected": bool(
                    atac_match[
                        "detected"
                    ]
                ),
                "ATAC_peak_parameter": atac_match[
                    "parameter"
                ],
                "ATAC_peak_prominence": atac_match[
                    "prominence"
                ],
                "ATAC_distance_to_frozen_center": atac_match[
                    "distance_to_frozen_center"
                ],
                "ATAC_n_eligible_family_peaks": atac_match[
                    "n_eligible_peaks"
                ],
                "paired_detected": paired_detected,
                "CMIL_RNA_minus_ATAC": separation,
                "ATAC_earlier": bool(
                    paired_detected
                    and separation
                    > 0
                ),
                "RNA_earlier": bool(
                    paired_detected
                    and separation
                    < 0
                ),
                "synchronous": bool(
                    paired_detected
                    and np.isclose(
                        separation,
                        0.0,
                        atol=1e-12,
                        rtol=0.0,
                    )
                ),
                "bootstrap_parameter_min": float(
                    windows[
                        "parameters"
                    ][
                        0
                    ]
                ),
                "bootstrap_parameter_max": float(
                    windows[
                        "parameters"
                    ][
                        -1
                    ]
                ),
            }
        )

        if (
            replicate % 25
            == 0
            or replicate
            == 1
        ):
            partial = pd.DataFrame(
                replicate_rows
            )

            paired_rate = float(
                partial[
                    "paired_detected"
                ].mean()
            )

            n_paired = int(
                partial[
                    "paired_detected"
                ].sum()
            )

            print(
                f"  replicate {replicate:3d}/{N_BOOTSTRAP} | "
                f"paired detections={n_paired:3d} "
                f"({paired_rate:.3f})",
                flush=True,
            )

    replicates = pd.DataFrame(
        replicate_rows
    )

    replicates.to_csv(
        OUTPUT_REPLICATES,
        sep="\t",
        index=False,
        compression="gzip",
    )

    # -----------------------------------------------------------------
    # Summary.
    # -----------------------------------------------------------------

    section(
        "4. BOOTSTRAP DETECTION / TIMING SUMMARY"
    )

    rna_detection_rate = float(
        replicates[
            "RNA_detected"
        ].mean()
    )

    atac_detection_rate = float(
        replicates[
            "ATAC_detected"
        ].mean()
    )

    paired_detection_rate = float(
        replicates[
            "paired_detected"
        ].mean()
    )

    detected = replicates.loc[
        replicates[
            "paired_detected"
        ]
    ].copy()

    n_detected = len(
        detected
    )

    if n_detected < 10:
        raise RuntimeError(
            f"Too few paired bootstrap detections ({n_detected}) "
            f"for percentile summary."
        )

    values = detected[
        "CMIL_RNA_minus_ATAC"
    ].to_numpy(
        dtype=float
    )

    ci_lower, ci_upper = np.quantile(
        values,
        [
            0.025,
            0.975,
        ],
    )

    p_positive = float(
        np.mean(
            values
            > 0
        )
    )

    p_negative = float(
        np.mean(
            values
            < 0
        )
    )

    p_zero = float(
        np.mean(
            np.isclose(
                values,
                0.0,
                atol=1e-12,
                rtol=0.0,
            )
        )
    )

    bootstrap_mean = float(
        np.mean(
            values
        )
    )

    bootstrap_median = float(
        np.median(
            values
        )
    )

    support_detection = bool(
        paired_detection_rate
        >= MIN_PAIRED_DETECTION_RATE
    )

    support_ci = bool(
        ci_lower
        > 0
    )

    support_direction = bool(
        p_positive
        >= MIN_DIRECTIONAL_BOOTSTRAP_SUPPORT
    )

    all_support = bool(
        support_detection
        and support_ci
        and support_direction
    )

    summary = pd.DataFrame(
        [
            {
                "observed_primary_RNA_peak_parameter": observed_rna,
                "observed_primary_ATAC_peak_parameter": observed_atac,
                "observed_primary_CMIL_RNA_minus_ATAC": observed_primary,
                "n_bootstrap": N_BOOTSTRAP,
                "RNA_detection_n": int(
                    replicates[
                        "RNA_detected"
                    ].sum()
                ),
                "RNA_detection_rate": rna_detection_rate,
                "ATAC_detection_n": int(
                    replicates[
                        "ATAC_detected"
                    ].sum()
                ),
                "ATAC_detection_rate": atac_detection_rate,
                "paired_detection_n": int(
                    n_detected
                ),
                "paired_detection_rate": paired_detection_rate,
                "bootstrap_CMIL_mean": bootstrap_mean,
                "bootstrap_CMIL_median": bootstrap_median,
                "bootstrap_CMIL_ci025": float(
                    ci_lower
                ),
                "bootstrap_CMIL_ci975": float(
                    ci_upper
                ),
                "P_boot_CMIL_gt_0": p_positive,
                "P_boot_CMIL_lt_0": p_negative,
                "P_boot_CMIL_eq_0": p_zero,
                "paired_detection_threshold": MIN_PAIRED_DETECTION_RATE,
                "directional_support_threshold": (
                    MIN_DIRECTIONAL_BOOTSTRAP_SUPPORT
                ),
                "paired_detection_pass": support_detection,
                "CI_lower_gt_0_pass": support_ci,
                "directional_support_pass": support_direction,
                "bootstrap_ATAC_earlier_support_pass": all_support,
            }
        ]
    )

    print_df(
        summary.T,
        digits=6,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Decision.
    # -----------------------------------------------------------------

    section(
        "5. PHASE F17 SUPPORT CHECKS"
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    f"Paired family detection rate >= "
                    f"{MIN_PAIRED_DETECTION_RATE:.2f}"
                ),
                "pass": support_detection,
            },
            {
                "criterion": (
                    "95% bootstrap percentile CI lower bound > 0"
                ),
                "pass": support_ci,
            },
            {
                "criterion": (
                    f"P_boot(CMIL > 0) >= "
                    f"{MIN_DIRECTIONAL_BOOTSTRAP_SUPPORT:.2f}"
                ),
                "pass": support_direction,
            },
            {
                "criterion": (
                    "Frozen family centers unchanged during bootstrap"
                ),
                "pass": True,
            },
            {
                "criterion": (
                    "Same sampled cell indices used for RNA and ATAC"
                ),
                "pass": True,
            },
            {
                "criterion": (
                    "Frozen F13 pseudotime reused; pseudotime not re-estimated"
                ),
                "pass": True,
            },
            {
                "criterion": (
                    "No alternate family substituted after nondetection"
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

    # -----------------------------------------------------------------
    # Manifest.
    # -----------------------------------------------------------------

    section(
        "6. SAVE F17 MANIFEST"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F17",
        "created_utc": utc_now_iso(),
        "software": {
            "python": sys.version,
            "pygdis_version": installed_version,
        },
        "bootstrap_design": {
            "n_bootstrap": N_BOOTSTRAP,
            "n_pseudotime_strata": N_STRATA,
            "paired_cell_resampling": True,
            "sampling_with_replacement_within_strata": True,
            "random_seed": RANDOM_SEED,
            "common_pseudotime_reestimated": False,
            "primary_window_size": WINDOW_SIZE,
            "primary_step_size": STEP_SIZE,
        },
        "frozen_pair": {
            "RNA_family_center": rna_center,
            "ATAC_family_center": atac_center,
            "family_match_tolerance": FAMILY_MATCH_TOLERANCE,
            "observed_primary_RNA_peak": observed_rna,
            "observed_primary_ATAC_peak": observed_atac,
            "observed_primary_CMIL": observed_primary,
            "family_reselection_inside_bootstrap": False,
        },
        "peak_recovery": {
            "material_prominence_absolute_floor": (
                MATERIAL_PROMINENCE_ABSOLUTE_FLOOR
            ),
            "material_prominence_range_fraction": (
                MATERIAL_PROMINENCE_RANGE_FRACTION
            ),
            "edge_point_count": EDGE_POINT_COUNT,
            "edge_spacing_multiplier": EDGE_SPACING_MULTIPLIER,
            "multiple_peak_rule": (
                "closest to own frozen family center; "
                "tie larger prominence; then smaller parameter"
            ),
        },
        "support_thresholds": {
            "minimum_paired_detection_rate": MIN_PAIRED_DETECTION_RATE,
            "CI_lower_must_exceed_zero": True,
            "minimum_P_boot_CMIL_gt_0": (
                MIN_DIRECTIONAL_BOOTSTRAP_SUPPORT
            ),
        },
        "summary": (
            summary.iloc[
                0
            ].to_dict()
        ),
        "checks": checks.to_dict(
            orient="records"
        ),
        "invalid_window_parameter_retries": (
            invalid_window_parameter_retries
        ),
        "outputs": {
            "replicates": {
                "file": str(
                    OUTPUT_REPLICATES
                ),
                "sha256": sha256_file(
                    OUTPUT_REPLICATES
                ),
            },
            "summary": {
                "file": str(
                    OUTPUT_SUMMARY
                ),
                "sha256": sha256_file(
                    OUTPUT_SUMMARY
                ),
            },
        },
        "guardrails": {
            "family_pair_changed": False,
            "branch_changed": False,
            "trajectory_changed": False,
            "representations_changed": False,
            "alignment_null_calculated": False,
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
        OUTPUT_REPLICATES,
        OUTPUT_SUMMARY,
        OUTPUT_MANIFEST,
    ]:
        print(
            path
        )

    # -----------------------------------------------------------------
    # Final verdict.
    # -----------------------------------------------------------------

    section(
        "7. PHASE F17 DECISION"
    )

    if all_support:
        print(
            "PHASE F17 VERDICT: GO — BOOTSTRAP SUPPORTS THE FROZEN "
            "ATAC-EARLIER TRANSITION-ENERGY EVENT PAIR"
        )

        print()

        print(
            "This is bootstrap support for the already-frozen timing "
            "relationship."
        )

        print()

        print(
            "It is NOT yet a chromatin-priming conclusion."
        )

        print()

        print(
            "Next phase must test whether the observed cross-modal alignment "
            "is stronger than an appropriate null."
        )

    else:
        print(
            "PHASE F17 VERDICT: REVIEW REQUIRED — BOOTSTRAP DOES NOT "
            "FULLY SUPPORT THE FROZEN ATAC-EARLIER EVENT PAIR"
        )

        print()

        print(
            "Do not reinterpret or reselect event families."
        )

    print()

    print(
        "No alignment/permutation null was calculated."
    )

    print(
        "No priming claim was made."
    )

    line("=")


if __name__ == "__main__":
    main()

