#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
28_phase_f14_shareseq_irs_gdis_primary_profiles.py
==================================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F14
---------
FIRST GDIS CALCULATION on the independently validated SHARE-seq IRS branch.

FROZEN INPUTS
-------------
Branch:
    TAC-1 -> TAC-2 -> IRS

Cells:
    5,050 paired RNA+ATAC cells

Common clock:
    F13 RNA-only k=30 diffusion pseudotime

RNA state space:
    F12 technical-rescued 10D
    original PCA axes PC2-PC11

ATAC state space:
    F12 depth-audited 10D
    first 10 F11 LSI axes

ANTI-CIRCULARITY
----------------
RNA and ATAC are evaluated:
    - on EXACTLY the same cells;
    - in EXACTLY the same pseudotime order;
    - with EXACTLY the same sliding-window membership;
    - with EXACTLY the same window pseudotime parameters.

ATAC did not participate in construction of the common clock.

INDEPENDENT BIOLOGICAL LANDMARK
-------------------------------
Before GDIS is calculated, F14 freezes the TAC-2 distribution from the F13
common clock:

    transition center:
        median TAC-2 pseudotime

    transition region:
        TAC-2 interquartile pseudotime interval [q25, q75]

This landmark is descriptive/validation metadata only.
It is NEVER supplied to pyGDIS as critical_value.

WINDOWING
---------
Frozen discovery settings:

Primary:
    400 cells/window
    100-cell step
    75% overlap

Sensitivity:
    300 cells/window, 75-cell step
    500 cells/window, 125-cell step

Each window is assigned the median common pseudotime of its cells.

GDIS SETTINGS
-------------
pyGDIS 1.0.0
data-only mode
critical_value = None

Frozen parameters:
    alpha_j                     = 0.42
    alpha_s                     = 0.33
    alpha_a                     = 0.25
    k_j                         = 3.0
    k_s                         = 2.6
    k_a                         = 2.4
    hill_gamma                  = 0.72
    hill_c                      = 0.22
    complexity_gain             = 0.055
    temporal_gain               = 0.055
    temporal_threshold          = 0.60
    transition_width_fraction   = 0.09
    smoothing_window            = 9
    smoothing_polynomial_order  = 3
    max_sustained               = 0.985

Two PRE-SPECIFIED transition weights are calculated:

PRIMARY MULTIOME PROFILE:
    lambda_t = 0.00

This removes localized transition amplification and evaluates the sustained
instability architecture directly.

REFERENCE pyGDIS PROFILE:
    lambda_t = 0.18

This is the frozen reference pyGDIS configuration.

The descriptors, transition-energy estimate, critical-window center, and
window family are otherwise identical.

IMPORTANT
---------
F14 does NOT:
    - pair RNA and ATAC peaks;
    - compute ATAC-minus-RNA lead/lag;
    - choose a cross-modal event;
    - calculate CMIL;
    - bootstrap timing differences;
    - run a cross-modal alignment null;
    - claim chromatin priming.

Those steps are explicitly deferred until profile topology and window
sensitivity are reviewed.

OUTPUTS
-------
data/GSE140203/representations_f14/
    f14_independent_state_landmarks.tsv
    f14_common_window_metadata.tsv.gz
    f14_gdis_profiles.tsv.gz
    f14_gdis_summary.tsv
    f14_window_sensitivity.tsv
    f14_manifest.json

RUN
---
    python 28_phase_f14_shareseq_irs_gdis_primary_profiles.py

DEPENDENCIES
------------
numpy
pandas
scipy
pygdis==1.0.0

Install if needed:
    python -m pip install "pygdis==1.0.0"
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
from scipy.stats import pearsonr, spearmanr

try:
    import gdis
    from gdis import GDIS, GDISConfig
except ImportError as exc:
    raise SystemExit(
        "\nERROR: Phase F14 requires pyGDIS 1.0.0.\n\n"
        "Install it with:\n\n"
        '    python -m pip install "pygdis==1.0.0"\n'
    ) from exc


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE140203"

DATA_DIR = Path("data") / ACCESSION
F12_DIR = DATA_DIR / "representations_f12"
F13_DIR = DATA_DIR / "representations_f13"
F14_DIR = DATA_DIR / "representations_f14"

F12_CELLS = F12_DIR / "f12_irs_branch_cells.tsv.gz"
F12_RNA = F12_DIR / "f12_rna_irs_10d.npy"
F12_ATAC = F12_DIR / "f12_atac_irs_10d.npy"
F12_MANIFEST = F12_DIR / "f12_manifest.json"

F13_PSEUDOTIME = F13_DIR / "f13_irs_common_pseudotime.tsv.gz"
F13_MANIFEST = F13_DIR / "f13_manifest.json"

OUTPUT_LANDMARKS = F14_DIR / "f14_independent_state_landmarks.tsv"
OUTPUT_WINDOWS = F14_DIR / "f14_common_window_metadata.tsv.gz"
OUTPUT_PROFILES = F14_DIR / "f14_gdis_profiles.tsv.gz"
OUTPUT_SUMMARY = F14_DIR / "f14_gdis_summary.tsv"
OUTPUT_SENSITIVITY = F14_DIR / "f14_window_sensitivity.tsv"
OUTPUT_MANIFEST = F14_DIR / "f14_manifest.json"


# =====================================================================
# FROZEN DESIGN
# =====================================================================

EXPECTED_PYGDIS_VERSION = "1.0.0"

EXPECTED_CELLS = 5_050
EXPECTED_DIMS = 10

BARCODE_COL = "rna.bc"
STATE_COL = "celltype"
PSEUDOTIME_COL = "common_pseudotime"

STATES = [
    "TAC-1",
    "TAC-2",
    "IRS",
]

TRANSITION_STATE = "TAC-2"

WINDOW_CONFIGS = {
    "sensitivity_w300_s75": {
        "window_size": 300,
        "step_size": 75,
    },
    "primary_w400_s100": {
        "window_size": 400,
        "step_size": 100,
    },
    "sensitivity_w500_s125": {
        "window_size": 500,
        "step_size": 125,
    },
}

PRIMARY_WINDOW_CONFIG = "primary_w400_s100"

PRIMARY_TRANSITION_WEIGHT = 0.00
REFERENCE_TRANSITION_WEIGHT = 0.18

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

MODALITIES = [
    "RNA",
    "ATAC",
]


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
        "display.width", 500,
        "display.max_colwidth", 140,
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


def safe_spearman(x, y):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if len(
        x
    ) < 3:
        return np.nan

    if np.allclose(
        x,
        x[
            0
        ],
    ) or np.allclose(
        y,
        y[
            0
        ],
    ):
        return np.nan

    rho, _ = spearmanr(
        x,
        y,
    )

    return float(
        rho
    )


def safe_pearson(x, y):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if len(
        x
    ) < 3:
        return np.nan

    if np.allclose(
        x,
        x[
            0
        ],
    ) or np.allclose(
        y,
        y[
            0
        ],
    ):
        return np.nan

    r, _ = pearsonr(
        x,
        y,
    )

    return float(
        r
    )


# =====================================================================
# INPUT VALIDATION / ALIGNMENT
# =====================================================================

def require_inputs():
    required = [
        F12_CELLS,
        F12_RNA,
        F12_ATAC,
        F12_MANIFEST,
        F13_PSEUDOTIME,
        F13_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required F12/F13 input(s):\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )


def load_and_align():
    """
    Load F12 row-ordered representations and align F13 pseudotime by rna.bc.

    We do NOT assume that the pseudotime table happens to retain the exact same
    physical row order.
    """
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
        F13_PSEUDOTIME,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    if len(
        cells
    ) != EXPECTED_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_CELLS:,} F12 cells; "
            f"found {len(cells):,}."
        )

    if rna.shape != (
        EXPECTED_CELLS,
        EXPECTED_DIMS,
    ):
        raise RuntimeError(
            f"Unexpected F12 RNA shape: {rna.shape}"
        )

    if atac.shape != (
        EXPECTED_CELLS,
        EXPECTED_DIMS,
    ):
        raise RuntimeError(
            f"Unexpected F12 ATAC shape: {atac.shape}"
        )

    if not np.isfinite(
        rna
    ).all():
        raise RuntimeError(
            "F12 RNA contains NaN/Inf."
        )

    if not np.isfinite(
        atac
    ).all():
        raise RuntimeError(
            "F12 ATAC contains NaN/Inf."
        )

    required_cell_cols = [
        BARCODE_COL,
        STATE_COL,
    ]

    for column in required_cell_cols:
        if column not in cells.columns:
            raise RuntimeError(
                f"F12 cell table lacks required column: {column}"
            )

    required_clock_cols = [
        BARCODE_COL,
        STATE_COL,
        PSEUDOTIME_COL,
    ]

    for column in required_clock_cols:
        if column not in clock.columns:
            raise RuntimeError(
                f"F13 clock table lacks required column: {column}"
            )

    if cells[
        BARCODE_COL
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate RNA barcodes in F12 cell table."
        )

    if clock[
        BARCODE_COL
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate RNA barcodes in F13 clock table."
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
            f"{len(missing_clock)} F12 cells are missing from F13 clock."
        )

    aligned_clock = clock_indexed.loc[
        cells[
            BARCODE_COL
        ].astype(str)
    ].reset_index(
        drop=True
    )

    if not np.array_equal(
        cells[
            STATE_COL
        ].astype(str).to_numpy(),
        aligned_clock[
            STATE_COL
        ].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "F12/F13 cell-state labels disagree after barcode alignment."
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
            "F13 common pseudotime contains NaN/Inf."
        )

    if np.min(
        pseudotime
    ) < 0:
        raise RuntimeError(
            "F13 common pseudotime contains negative values."
        )

    observed_states = set(
        cells[
            STATE_COL
        ].astype(str).unique()
    )

    if observed_states != set(
        STATES
    ):
        raise RuntimeError(
            f"Unexpected F12/F13 states: {sorted(observed_states)}"
        )

    return (
        cells,
        rna,
        atac,
        aligned_clock,
        pseudotime,
    )


# =====================================================================
# INDEPENDENT BIOLOGICAL LANDMARKS
# =====================================================================

def freeze_state_landmarks(
    cells,
    pseudotime,
):
    """
    Freeze state-distribution landmarks BEFORE any GDIS call.

    TAC-2 median and IQR are the prespecified transition center/region.
    """
    rows = []

    for state in STATES:
        mask = cells[
            STATE_COL
        ].astype(str).eq(
            state
        ).to_numpy()

        values = pseudotime[
            mask
        ]

        q = np.quantile(
            values,
            [
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
            ],
        )

        rows.append(
            {
                "state": state,
                "n_cells": int(
                    len(
                        values
                    )
                ),
                "q05": float(
                    q[
                        0
                    ]
                ),
                "q25": float(
                    q[
                        1
                    ]
                ),
                "median": float(
                    q[
                        2
                    ]
                ),
                "q75": float(
                    q[
                        3
                    ]
                ),
                "q95": float(
                    q[
                        4
                    ]
                ),
                "is_transition_state": (
                    state
                    == TRANSITION_STATE
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# COMMON WINDOW FAMILY
# =====================================================================

def construct_common_windows(
    cells_sorted,
    pseudotime_sorted,
    config_name,
    window_size,
    step_size,
):
    """
    Create complete overlapping windows only.

    The returned metadata defines the common window family used identically by
    RNA and ATAC.
    """
    n_cells = len(
        cells_sorted
    )

    starts = list(
        range(
            0,
            n_cells
            - window_size
            + 1,
            step_size,
        )
    )

    if len(
        starts
    ) < 3:
        raise RuntimeError(
            f"{config_name}: fewer than three complete windows."
        )

    rows = []

    for window_index, start in enumerate(
        starts
    ):
        stop = (
            start
            + window_size
        )

        pt = pseudotime_sorted[
            start:
            stop
        ]

        states = cells_sorted.iloc[
            start:
            stop
        ][
            STATE_COL
        ].astype(str)

        counts = states.value_counts()

        state_counts = {
            state: int(
                counts.get(
                    state,
                    0,
                )
            )
            for state in STATES
        }

        dominant_state = max(
            STATES,
            key=lambda state: state_counts[
                state
            ],
        )

        rows.append(
            {
                "window_config": config_name,
                "window_size": int(
                    window_size
                ),
                "step_size": int(
                    step_size
                ),
                "window_index": int(
                    window_index
                ),
                "start_rank_zero_based": int(
                    start
                ),
                "stop_rank_exclusive": int(
                    stop
                ),
                "n_cells": int(
                    window_size
                ),
                "parameter": float(
                    np.median(
                        pt
                    )
                ),
                "pseudotime_min": float(
                    np.min(
                        pt
                    )
                ),
                "pseudotime_max": float(
                    np.max(
                        pt
                    )
                ),
                "TAC1_n": state_counts[
                    "TAC-1"
                ],
                "TAC2_n": state_counts[
                    "TAC-2"
                ],
                "IRS_n": state_counts[
                    "IRS"
                ],
                "TAC1_fraction": (
                    state_counts[
                        "TAC-1"
                    ]
                    / window_size
                ),
                "TAC2_fraction": (
                    state_counts[
                        "TAC-2"
                    ]
                    / window_size
                ),
                "IRS_fraction": (
                    state_counts[
                        "IRS"
                    ]
                    / window_size
                ),
                "dominant_state": dominant_state,
            }
        )

    metadata = pd.DataFrame(
        rows
    )

    parameters = metadata[
        "parameter"
    ].to_numpy(
        dtype=float
    )

    if np.any(
        np.diff(
            parameters
        )
        <= 0
    ):
        duplicates = metadata.loc[
            metadata[
                "parameter"
            ].duplicated(
                keep=False
            ),
            [
                "window_index",
                "parameter",
            ],
        ]

        raise RuntimeError(
            f"{config_name}: window median pseudotimes are not strictly "
            f"increasing/unique.\n{duplicates.to_string(index=False)}"
        )

    return metadata


def trajectories_from_windows(
    representation_sorted,
    window_metadata,
):
    trajectories = []

    for row in window_metadata.itertuples(
        index=False
    ):
        start = int(
            row.start_rank_zero_based
        )

        stop = int(
            row.stop_rank_exclusive
        )

        trajectory = representation_sorted[
            start:
            stop,
            :
        ]

        if trajectory.shape[
            0
        ] < 4:
            raise RuntimeError(
                "pyGDIS requires at least four rows per trajectory."
            )

        if not np.isfinite(
            trajectory
        ).all():
            raise RuntimeError(
                "Window trajectory contains NaN/Inf."
            )

        trajectories.append(
            trajectory
        )

    return trajectories


# =====================================================================
# pyGDIS CONFIGURATION / EXECUTION
# =====================================================================

def make_gdis_config(
    transition_weight,
):
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
        transition_weight=float(
            transition_weight
        ),
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


def verify_config(
    config,
    transition_weight,
):
    observed = asdict(
        config
    )

    expected = {
        **GDIS_FIXED,
        "transition_weight": float(
            transition_weight
        ),
    }

    rows = []

    all_match = True

    for key, expected_value in expected.items():
        observed_value = observed.get(
            key
        )

        if expected_value is None:
            match = (
                observed_value
                is None
            )
        elif isinstance(
            expected_value,
            float,
        ):
            match = bool(
                np.isclose(
                    float(
                        observed_value
                    ),
                    expected_value,
                    atol=1e-12,
                    rtol=0.0,
                )
            )
        else:
            match = (
                observed_value
                == expected_value
            )

        all_match = (
            all_match
            and match
        )

        rows.append(
            {
                "parameter": key,
                "expected": expected_value,
                "observed": observed_value,
                "match": match,
            }
        )

    return (
        all_match,
        pd.DataFrame(
            rows
        ),
    )


def run_one_gdis(
    trajectories,
    parameters,
    transition_weight,
):
    """
    Run one completely independent pyGDIS calculation.

    No critical_value is provided.
    """
    config = make_gdis_config(
        transition_weight
    )

    config_ok, config_table = verify_config(
        config,
        transition_weight,
    )

    if not config_ok:
        raise RuntimeError(
            "Frozen pyGDIS configuration verification failed."
        )

    model = GDIS(
        config=config
    )

    result = model.fit_transform(
        trajectories,
        parameters,
        jacobian_function=None,
        critical_value=None,
    )

    if (
        result.metadata.get(
            "critical_value_source"
        )
        != "data_driven_transition_energy_peak"
    ):
        raise RuntimeError(
            "pyGDIS unexpectedly used a provided biological critical value."
        )

    return (
        result,
        config_table,
    )


def validate_result(
    result,
    expected_parameters,
):
    arrays = {
        "parameters": result.parameters,
        "gdis": result.gdis,
        "potential": result.potential,
        "sustained_instability": result.sustained_instability,
        "transition_instability": result.transition_instability,
    }

    for name, values in arrays.items():
        values = np.asarray(
            values,
            dtype=float,
        )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"pyGDIS result {name} contains NaN/Inf."
            )

    if not np.allclose(
        result.parameters,
        expected_parameters,
        atol=1e-14,
        rtol=0.0,
    ):
        raise RuntimeError(
            "pyGDIS parameters differ from frozen common-window parameters."
        )

    if np.any(
        result.gdis
        < 0
    ) or np.any(
        result.gdis
        >= 1
    ):
        raise RuntimeError(
            "GDIS score is outside [0,1)."
        )

    if np.any(
        result.sustained_instability
        < 0
    ) or np.any(
        result.sustained_instability
        >= 1
    ):
        raise RuntimeError(
            "Sustained instability is outside [0,1)."
        )

    return True


def descriptors_identical_between_weights(
    primary_result,
    reference_result,
):
    """
    Only transition weighting is allowed to differ.

    Descriptor and sustained-instability arrays must be identical.
    """
    names = [
        "jacobian_raw",
        "stretching_raw",
        "expansion_raw",
        "entropy_raw",
        "temporal_raw",
        "temporal_mean",
        "temporal_persistence",
        "jacobian_scaled",
        "stretching_scaled",
        "expansion_scaled",
        "jacobian_saturated",
        "stretching_saturated",
        "expansion_saturated",
        "core",
        "complexity_factor",
        "temporal_factor",
        "transition_energy",
        "critical_window",
        "transition_base",
    ]

    checks = []

    for name in names:
        a = np.asarray(
            primary_result.components[
                name
            ],
            dtype=float,
        )

        b = np.asarray(
            reference_result.components[
                name
            ],
            dtype=float,
        )

        checks.append(
            np.allclose(
                a,
                b,
                atol=1e-12,
                rtol=1e-12,
                equal_nan=True,
            )
        )

    checks.append(
        np.allclose(
            primary_result.sustained_instability,
            reference_result.sustained_instability,
            atol=1e-12,
            rtol=1e-12,
            equal_nan=True,
        )
    )

    return bool(
        all(
            checks
        )
    )


# =====================================================================
# PROFILE ASSEMBLY / SUMMARIES
# =====================================================================

def result_to_profile(
    result,
    window_metadata,
    modality,
    config_name,
    transition_weight,
    profile_role,
):
    result_df = result.to_dataframe().copy()

    if len(
        result_df
    ) != len(
        window_metadata
    ):
        raise RuntimeError(
            "pyGDIS result/window metadata row-count mismatch."
        )

    if not np.allclose(
        result_df[
            "parameter"
        ].to_numpy(
            dtype=float
        ),
        window_metadata[
            "parameter"
        ].to_numpy(
            dtype=float
        ),
        atol=1e-14,
        rtol=0.0,
    ):
        raise RuntimeError(
            "pyGDIS result/window parameter mismatch."
        )

    window_columns = [
        column
        for column in window_metadata.columns
        if column != "parameter"
    ]

    output = pd.concat(
        [
            window_metadata[
                [
                    "parameter",
                    *window_columns,
                ]
            ].reset_index(
                drop=True
            ),
            result_df.drop(
                columns=[
                    "parameter",
                ]
            ).reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    output.insert(
        0,
        "profile_role",
        profile_role,
    )

    output.insert(
        0,
        "transition_weight",
        float(
            transition_weight
        ),
    )

    output.insert(
        0,
        "modality",
        modality,
    )

    output[
        "resolved_critical_value"
    ] = float(
        result.metadata[
            "critical_value"
        ]
    )

    output[
        "critical_value_source"
    ] = str(
        result.metadata[
            "critical_value_source"
        ]
    )

    return output


def summarize_profile(
    profile,
):
    idx_peak = int(
        np.argmax(
            profile[
                "gdis"
            ].to_numpy(
                dtype=float
            )
        )
    )

    idx_sustained = int(
        np.argmax(
            profile[
                "sustained_instability"
            ].to_numpy(
                dtype=float
            )
        )
    )

    idx_energy = int(
        np.argmax(
            profile[
                "transition_energy"
            ].to_numpy(
                dtype=float
            )
        )
    )

    idx_transition = int(
        np.argmax(
            profile[
                "transition_instability"
            ].to_numpy(
                dtype=float
            )
        )
    )

    peak_row = profile.iloc[
        idx_peak
    ]

    sustained_row = profile.iloc[
        idx_sustained
    ]

    energy_row = profile.iloc[
        idx_energy
    ]

    transition_row = profile.iloc[
        idx_transition
    ]

    return {
        "modality": str(
            profile[
                "modality"
            ].iloc[
                0
            ]
        ),
        "window_config": str(
            profile[
                "window_config"
            ].iloc[
                0
            ]
        ),
        "profile_role": str(
            profile[
                "profile_role"
            ].iloc[
                0
            ]
        ),
        "transition_weight": float(
            profile[
                "transition_weight"
            ].iloc[
                0
            ]
        ),
        "n_windows": int(
            len(
                profile
            )
        ),
        "gdis_min": float(
            profile[
                "gdis"
            ].min()
        ),
        "gdis_median": float(
            profile[
                "gdis"
            ].median()
        ),
        "gdis_mean": float(
            profile[
                "gdis"
            ].mean()
        ),
        "gdis_max": float(
            peak_row[
                "gdis"
            ]
        ),
        "gdis_peak_window_index": int(
            peak_row[
                "window_index"
            ]
        ),
        "gdis_peak_parameter": float(
            peak_row[
                "parameter"
            ]
        ),
        "gdis_peak_dominant_state": str(
            peak_row[
                "dominant_state"
            ]
        ),
        "gdis_peak_TAC1_fraction": float(
            peak_row[
                "TAC1_fraction"
            ]
        ),
        "gdis_peak_TAC2_fraction": float(
            peak_row[
                "TAC2_fraction"
            ]
        ),
        "gdis_peak_IRS_fraction": float(
            peak_row[
                "IRS_fraction"
            ]
        ),
        "sustained_peak_parameter": float(
            sustained_row[
                "parameter"
            ]
        ),
        "transition_energy_peak_parameter": float(
            energy_row[
                "parameter"
            ]
        ),
        "transition_instability_peak_parameter": float(
            transition_row[
                "parameter"
            ]
        ),
        "resolved_critical_value": float(
            profile[
                "resolved_critical_value"
            ].iloc[
                0
            ]
        ),
        "critical_value_source": str(
            profile[
                "critical_value_source"
            ].iloc[
                0
            ]
        ),
        "spearman_gdis_vs_pseudotime": safe_spearman(
            profile[
                "parameter"
            ],
            profile[
                "gdis"
            ],
        ),
    }


# =====================================================================
# WINDOW-SIZE SENSITIVITY
# =====================================================================

def compare_profiles(
    reference_profile,
    alternative_profile,
):
    """
    Compare GDIS shape after interpolating the alternative profile onto
    reference-profile parameter coordinates within their shared range.

    This is WITHIN-MODALITY window-size sensitivity only.
    It is not a cross-modal alignment test.
    """
    ref_x = reference_profile[
        "parameter"
    ].to_numpy(
        dtype=float
    )

    ref_y = reference_profile[
        "gdis"
    ].to_numpy(
        dtype=float
    )

    alt_x = alternative_profile[
        "parameter"
    ].to_numpy(
        dtype=float
    )

    alt_y = alternative_profile[
        "gdis"
    ].to_numpy(
        dtype=float
    )

    shared_min = max(
        float(
            np.min(
                ref_x
            )
        ),
        float(
            np.min(
                alt_x
            )
        ),
    )

    shared_max = min(
        float(
            np.max(
                ref_x
            )
        ),
        float(
            np.max(
                alt_x
            )
        ),
    )

    mask = (
        ref_x
        >= shared_min
    ) & (
        ref_x
        <= shared_max
    )

    grid = ref_x[
        mask
    ]

    ref_shared = ref_y[
        mask
    ]

    if len(
        grid
    ) < 3:
        raise RuntimeError(
            "Insufficient shared parameter range for window sensitivity."
        )

    alt_interp = np.interp(
        grid,
        alt_x,
        alt_y,
    )

    ref_peak_parameter = float(
        ref_x[
            np.argmax(
                ref_y
            )
        ]
    )

    alt_peak_parameter = float(
        alt_x[
            np.argmax(
                alt_y
            )
        ]
    )

    return {
        "shared_pseudotime_min": shared_min,
        "shared_pseudotime_max": shared_max,
        "n_reference_points_compared": int(
            len(
                grid
            )
        ),
        "profile_pearson_vs_primary": safe_pearson(
            ref_shared,
            alt_interp,
        ),
        "profile_spearman_vs_primary": safe_spearman(
            ref_shared,
            alt_interp,
        ),
        "profile_mae_vs_primary": float(
            np.mean(
                np.abs(
                    ref_shared
                    - alt_interp
                )
            )
        ),
        "primary_peak_parameter": ref_peak_parameter,
        "alternative_peak_parameter": alt_peak_parameter,
        "absolute_peak_parameter_shift": abs(
            alt_peak_parameter
            - ref_peak_parameter
        ),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F14 — FIRST SHARE-seq IRS GDIS PROFILES ON THE FROZEN COMMON CLOCK"
    )

    print(
        "This is the FIRST GDIS calculation on GSE140203 in this workflow."
    )

    print()

    print(
        "Branch:"
    )

    print(
        "  TAC-1 -> TAC-2 -> IRS"
    )

    print()

    print(
        "Common clock:"
    )

    print(
        "  F13 RNA-only k=30 DPT"
    )

    print()

    print(
        "Primary GDIS transition weight:"
    )

    print(
        f"  lambda_t = {PRIMARY_TRANSITION_WEIGHT:.2f}"
    )

    print()

    print(
        "Reference pyGDIS transition weight:"
    )

    print(
        f"  lambda_t = {REFERENCE_TRANSITION_WEIGHT:.2f}"
    )

    print()

    print(
        "No cross-modal peak pairing."
    )

    print(
        "No lead/lag calculation."
    )

    print(
        "No CMIL."
    )

    require_inputs()

    installed_version = str(
        getattr(
            gdis,
            "__version__",
            "unknown",
        )
    )

    print()

    print(
        f"pyGDIS version: {installed_version}"
    )

    if installed_version != EXPECTED_PYGDIS_VERSION:
        raise RuntimeError(
            f"Phase F14 requires pyGDIS {EXPECTED_PYGDIS_VERSION}; "
            f"found {installed_version}."
        )

    F14_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Load and align.
    # -----------------------------------------------------------------

    section(
        "1. LOAD AND ALIGN FROZEN F12 STATE SPACES WITH F13 COMMON CLOCK"
    )

    (
        cells,
        rna,
        atac,
        aligned_clock,
        pseudotime,
    ) = load_and_align()

    print(
        f"Cells: {len(cells):,}"
    )

    print(
        f"RNA shape:  {rna.shape}"
    )

    print(
        f"ATAC shape: {atac.shape}"
    )

    print(
        f"Common pseudotime range: "
        f"{pseudotime.min():.6f} - {pseudotime.max():.6f}"
    )

    subsection(
        "State counts"
    )

    state_counts = (
        cells[
            STATE_COL
        ]
        .value_counts()
        .reindex(
            STATES
        )
        .rename(
            "n_cells"
        )
        .to_frame()
    )

    print_df(
        state_counts,
        digits=0,
    )

    # -----------------------------------------------------------------
    # Freeze independent biological landmark before any GDIS call.
    # -----------------------------------------------------------------

    section(
        "2. FREEZE INDEPENDENT BIOLOGICAL STATE LANDMARKS BEFORE GDIS"
    )

    landmarks = freeze_state_landmarks(
        cells,
        pseudotime,
    )

    landmarks.to_csv(
        OUTPUT_LANDMARKS,
        sep="\t",
        index=False,
    )

    print_df(
        landmarks.set_index(
            "state"
        ),
        digits=6,
    )

    tac2_row = landmarks.loc[
        landmarks[
            "state"
        ]
        == TRANSITION_STATE
    ].iloc[
        0
    ]

    tac2_center = float(
        tac2_row[
            "median"
        ]
    )

    tac2_q25 = float(
        tac2_row[
            "q25"
        ]
    )

    tac2_q75 = float(
        tac2_row[
            "q75"
        ]
    )

    print()

    print(
        f"Frozen TAC-2 transition center (median): "
        f"{tac2_center:.6f}"
    )

    print(
        f"Frozen TAC-2 transition region (IQR): "
        f"[{tac2_q25:.6f}, {tac2_q75:.6f}]"
    )

    print()

    print(
        "These values will NOT be supplied to pyGDIS."
    )

    # -----------------------------------------------------------------
    # Sort once by common pseudotime.
    # -----------------------------------------------------------------

    section(
        "3. FREEZE COMMON PSEUDOTIME ORDER AND COMMON WINDOW FAMILIES"
    )

    order = np.argsort(
        pseudotime,
        kind="mergesort",
    )

    cells_sorted = cells.iloc[
        order
    ].copy().reset_index(
        drop=True
    )

    pseudotime_sorted = pseudotime[
        order
    ]

    rna_sorted = rna[
        order,
        :
    ]

    atac_sorted = atac[
        order,
        :
    ]

    window_metadata_by_config = {}

    window_tables = []

    for config_name, cfg in WINDOW_CONFIGS.items():
        metadata = construct_common_windows(
            cells_sorted,
            pseudotime_sorted,
            config_name,
            cfg[
                "window_size"
            ],
            cfg[
                "step_size"
            ],
        )

        window_metadata_by_config[
            config_name
        ] = metadata

        window_tables.append(
            metadata
        )

        print(
            f"{config_name}: "
            f"{len(metadata):,} windows | "
            f"size={cfg['window_size']} | "
            f"step={cfg['step_size']} | "
            f"parameter range "
            f"{metadata['parameter'].min():.6f}-"
            f"{metadata['parameter'].max():.6f}"
        )

    all_windows = pd.concat(
        window_tables,
        ignore_index=True,
    )

    all_windows.to_csv(
        OUTPUT_WINDOWS,
        sep="\t",
        index=False,
        compression="gzip",
    )

    # -----------------------------------------------------------------
    # Verify frozen pyGDIS configurations before profile values.
    # -----------------------------------------------------------------

    section(
        "4. VERIFY FROZEN pyGDIS CONFIGURATION"
    )

    for weight, role in [
        (
            PRIMARY_TRANSITION_WEIGHT,
            "primary_multiome",
        ),
        (
            REFERENCE_TRANSITION_WEIGHT,
            "reference_pygdis",
        ),
    ]:
        config = make_gdis_config(
            weight
        )

        ok, table = verify_config(
            config,
            weight
        )

        subsection(
            f"{role} — lambda_t={weight:.2f}"
        )

        print_df(
            table.set_index(
                "parameter"
            ),
            digits=6,
        )

        if not ok:
            raise RuntimeError(
                f"Frozen configuration mismatch for lambda_t={weight}."
            )

    # -----------------------------------------------------------------
    # Calculate profiles.
    # -----------------------------------------------------------------

    section(
        "5. CALCULATE RNA AND ATAC GDIS PROFILES"
    )

    representations = {
        "RNA": rna_sorted,
        "ATAC": atac_sorted,
    }

    profile_tables = []
    summary_rows = []

    technical_checks = []

    for config_name, metadata in window_metadata_by_config.items():
        parameters = metadata[
            "parameter"
        ].to_numpy(
            dtype=float
        )

        for modality in MODALITIES:
            subsection(
                f"{modality} | {config_name}"
            )

            representation = representations[
                modality
            ]

            trajectories = trajectories_from_windows(
                representation,
                metadata,
            )

            (
                primary_result,
                _,
            ) = run_one_gdis(
                trajectories,
                parameters,
                PRIMARY_TRANSITION_WEIGHT,
            )

            (
                reference_result,
                _,
            ) = run_one_gdis(
                trajectories,
                parameters,
                REFERENCE_TRANSITION_WEIGHT,
            )

            validate_result(
                primary_result,
                parameters,
            )

            validate_result(
                reference_result,
                parameters,
            )

            descriptors_match = descriptors_identical_between_weights(
                primary_result,
                reference_result,
            )

            if not descriptors_match:
                raise RuntimeError(
                    f"{modality}/{config_name}: descriptor arrays differ "
                    f"between lambda_t=0 and lambda_t=0.18."
                )

            if not np.isclose(
                float(
                    primary_result.metadata[
                        "critical_value"
                    ]
                ),
                float(
                    reference_result.metadata[
                        "critical_value"
                    ]
                ),
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"{modality}/{config_name}: data-driven critical center "
                    f"changed across transition weights."
                )

            primary_profile = result_to_profile(
                primary_result,
                metadata,
                modality,
                config_name,
                PRIMARY_TRANSITION_WEIGHT,
                "primary_multiome",
            )

            reference_profile = result_to_profile(
                reference_result,
                metadata,
                modality,
                config_name,
                REFERENCE_TRANSITION_WEIGHT,
                "reference_pygdis",
            )

            profile_tables.extend(
                [
                    primary_profile,
                    reference_profile,
                ]
            )

            primary_summary = summarize_profile(
                primary_profile
            )

            reference_summary = summarize_profile(
                reference_profile
            )

            summary_rows.extend(
                [
                    primary_summary,
                    reference_summary,
                ]
            )

            technical_checks.append(
                {
                    "window_config": config_name,
                    "modality": modality,
                    "n_windows": len(
                        metadata
                    ),
                    "parameters_strictly_increasing": bool(
                        np.all(
                            np.diff(
                                parameters
                            )
                            > 0
                        )
                    ),
                    "descriptors_identical_across_weights": descriptors_match,
                    "primary_critical_source_data_driven": (
                        primary_result.metadata[
                            "critical_value_source"
                        ]
                        == "data_driven_transition_energy_peak"
                    ),
                    "reference_critical_source_data_driven": (
                        reference_result.metadata[
                            "critical_value_source"
                        ]
                        == "data_driven_transition_energy_peak"
                    ),
                }
            )

            print(
                f"PRIMARY lambda=0.00 | "
                f"windows={len(metadata):3d} | "
                f"GDIS range "
                f"{primary_profile['gdis'].min():.6f}-"
                f"{primary_profile['gdis'].max():.6f} | "
                f"peak p="
                f"{primary_summary['gdis_peak_parameter']:.6f} | "
                f"energy peak p="
                f"{primary_summary['transition_energy_peak_parameter']:.6f}"
            )

            print(
                f"REFERENCE lambda=0.18 | "
                f"windows={len(metadata):3d} | "
                f"GDIS range "
                f"{reference_profile['gdis'].min():.6f}-"
                f"{reference_profile['gdis'].max():.6f} | "
                f"peak p="
                f"{reference_summary['gdis_peak_parameter']:.6f} | "
                f"energy peak p="
                f"{reference_summary['transition_energy_peak_parameter']:.6f}"
            )

    profiles = pd.concat(
        profile_tables,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    technical_df = pd.DataFrame(
        technical_checks
    )

    profiles.to_csv(
        OUTPUT_PROFILES,
        sep="\t",
        index=False,
        compression="gzip",
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Display primary summaries without cross-modal deltas.
    # -----------------------------------------------------------------

    section(
        "6. PROFILE DESCRIPTIVE SUMMARY — NO CROSS-MODAL PAIRING"
    )

    summary_display_columns = [
        "modality",
        "window_config",
        "profile_role",
        "transition_weight",
        "n_windows",
        "gdis_min",
        "gdis_median",
        "gdis_max",
        "gdis_peak_parameter",
        "gdis_peak_dominant_state",
        "sustained_peak_parameter",
        "transition_energy_peak_parameter",
        "resolved_critical_value",
        "spearman_gdis_vs_pseudotime",
    ]

    print_df(
        summary[
            summary_display_columns
        ].set_index(
            [
                "modality",
                "window_config",
                "profile_role",
            ]
        ),
        digits=6,
    )

    print()

    print(
        "No RNA-vs-ATAC peak difference is calculated in F14."
    )

    # -----------------------------------------------------------------
    # Within-modality window sensitivity.
    # -----------------------------------------------------------------

    section(
        "7. WITHIN-MODALITY WINDOW-SIZE SENSITIVITY"
    )

    sensitivity_rows = []

    for modality in MODALITIES:
        for profile_role, weight in [
            (
                "primary_multiome",
                PRIMARY_TRANSITION_WEIGHT,
            ),
            (
                "reference_pygdis",
                REFERENCE_TRANSITION_WEIGHT,
            ),
        ]:
            reference = profiles.loc[
                (
                    profiles[
                        "modality"
                    ]
                    == modality
                )
                & (
                    profiles[
                        "window_config"
                    ]
                    == PRIMARY_WINDOW_CONFIG
                )
                & (
                    profiles[
                        "profile_role"
                    ]
                    == profile_role
                )
            ].sort_values(
                "parameter"
            )

            for config_name in WINDOW_CONFIGS:
                alternative = profiles.loc[
                    (
                        profiles[
                            "modality"
                        ]
                        == modality
                    )
                    & (
                        profiles[
                            "window_config"
                        ]
                        == config_name
                    )
                    & (
                        profiles[
                            "profile_role"
                        ]
                        == profile_role
                    )
                ].sort_values(
                    "parameter"
                )

                metrics = compare_profiles(
                    reference,
                    alternative,
                )

                sensitivity_rows.append(
                    {
                        "modality": modality,
                        "profile_role": profile_role,
                        "transition_weight": weight,
                        "config": config_name,
                        **metrics,
                    }
                )

    sensitivity = pd.DataFrame(
        sensitivity_rows
    )

    sensitivity.to_csv(
        OUTPUT_SENSITIVITY,
        sep="\t",
        index=False,
    )

    print_df(
        sensitivity.set_index(
            [
                "modality",
                "profile_role",
                "config",
            ]
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Technical calculation gate.
    # -----------------------------------------------------------------

    section(
        "8. PHASE F14 TECHNICAL CALCULATION CHECKS"
    )

    expected_window_counts = {
        name: int(
            len(
                metadata
            )
        )
        for name, metadata in window_metadata_by_config.items()
    }

    profile_finite = bool(
        np.isfinite(
            profiles.select_dtypes(
                include=[
                    np.number,
                ]
            ).to_numpy(
                dtype=float
            )
        ).all()
    )

    gdis_bounded = bool(
        (
            profiles[
                "gdis"
            ]
            >= 0
        ).all()
        and (
            profiles[
                "gdis"
            ]
            < 1
        ).all()
    )

    common_window_rows_expected = int(
        sum(
            expected_window_counts.values()
        )
    )

    # For each common window there are:
    # 2 modalities x 2 transition-weight profiles.
    expected_profile_rows = int(
        common_window_rows_expected
        * len(
            MODALITIES
        )
        * 2
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    f"pyGDIS version exactly {EXPECTED_PYGDIS_VERSION}"
                ),
                "pass": (
                    installed_version
                    == EXPECTED_PYGDIS_VERSION
                ),
            },
            {
                "criterion": (
                    f"Frozen paired branch has exactly "
                    f"{EXPECTED_CELLS:,} cells"
                ),
                "pass": (
                    len(
                        cells
                    )
                    == EXPECTED_CELLS
                ),
            },
            {
                "criterion": (
                    "RNA and ATAC frozen state spaces are both 10D"
                ),
                "pass": (
                    rna.shape[
                        1
                    ]
                    == EXPECTED_DIMS
                    and atac.shape[
                        1
                    ]
                    == EXPECTED_DIMS
                ),
            },
            {
                "criterion": (
                    "All common-window parameter sequences strictly increase"
                ),
                "pass": bool(
                    technical_df[
                        "parameters_strictly_increasing"
                    ].all()
                ),
            },
            {
                "criterion": (
                    "Descriptors identical between lambda=0 and lambda=0.18"
                ),
                "pass": bool(
                    technical_df[
                        "descriptors_identical_across_weights"
                    ].all()
                ),
            },
            {
                "criterion": (
                    "All pyGDIS critical centers are data-driven"
                ),
                "pass": bool(
                    technical_df[
                        "primary_critical_source_data_driven"
                    ].all()
                    and technical_df[
                        "reference_critical_source_data_driven"
                    ].all()
                ),
            },
            {
                "criterion": (
                    "All numeric profile values finite"
                ),
                "pass": profile_finite,
            },
            {
                "criterion": (
                    "All GDIS values lie in [0,1)"
                ),
                "pass": gdis_bounded,
            },
            {
                "criterion": (
                    f"Profile table has expected "
                    f"{expected_profile_rows:,} rows"
                ),
                "pass": (
                    len(
                        profiles
                    )
                    == expected_profile_rows
                ),
            },
            {
                "criterion": (
                    "Independent TAC-2 landmark frozen before GDIS and "
                    "not supplied as critical_value"
                ),
                "pass": bool(
                    landmarks.loc[
                        landmarks[
                            "state"
                        ]
                        == TRANSITION_STATE,
                        "is_transition_state",
                    ].all()
                    and technical_df[
                        "primary_critical_source_data_driven"
                    ].all()
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
        "9. SAVE F14 MANIFEST"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F14",
        "created_utc": utc_now_iso(),
        "software": {
            "python": sys.version,
            "pygdis_version": installed_version,
        },
        "branch": {
            "states": STATES,
            "n_cells": len(
                cells
            ),
        },
        "common_clock": {
            "source_phase": "F13",
            "modality": "RNA only",
            "column": PSEUDOTIME_COL,
            "atac_used_to_construct_clock": False,
        },
        "independent_transition_landmark": {
            "state": TRANSITION_STATE,
            "definition": (
                "TAC-2 median common pseudotime; TAC-2 IQR defines "
                "transition region"
            ),
            "center": tac2_center,
            "region_q25": tac2_q25,
            "region_q75": tac2_q75,
            "supplied_to_pygdis": False,
        },
        "window_configs": WINDOW_CONFIGS,
        "primary_window_config": PRIMARY_WINDOW_CONFIG,
        "gdis_fixed_parameters": GDIS_FIXED,
        "transition_weights": {
            "primary_multiome": PRIMARY_TRANSITION_WEIGHT,
            "reference_pygdis": REFERENCE_TRANSITION_WEIGHT,
        },
        "profiles": {
            "n_rows": len(
                profiles
            ),
            "output": str(
                OUTPUT_PROFILES
            ),
            "sha256": sha256_file(
                OUTPUT_PROFILES
            ),
        },
        "summary": {
            "output": str(
                OUTPUT_SUMMARY
            ),
            "sha256": sha256_file(
                OUTPUT_SUMMARY
            ),
        },
        "window_sensitivity": {
            "output": str(
                OUTPUT_SENSITIVITY
            ),
            "sha256": sha256_file(
                OUTPUT_SENSITIVITY
            ),
        },
        "landmarks": {
            "output": str(
                OUTPUT_LANDMARKS
            ),
            "sha256": sha256_file(
                OUTPUT_LANDMARKS
            ),
        },
        "common_windows": {
            "output": str(
                OUTPUT_WINDOWS
            ),
            "sha256": sha256_file(
                OUTPUT_WINDOWS
            ),
        },
        "technical_checks": checks.to_dict(
            orient="records"
        ),
        "guardrails": {
            "critical_value_supplied_to_pygdis": False,
            "cross_modal_peak_pairing_performed": False,
            "lead_lag_calculated": False,
            "cmil_calculated": False,
            "bootstrap_timing_calculated": False,
            "cross_modal_alignment_null_calculated": False,
            "priming_claim_made": False,
            "trajectory_changed": False,
            "representation_changed": False,
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
        OUTPUT_LANDMARKS,
        OUTPUT_WINDOWS,
        OUTPUT_PROFILES,
        OUTPUT_SUMMARY,
        OUTPUT_SENSITIVITY,
        OUTPUT_MANIFEST,
    ]:
        print(
            path
        )

    # -----------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------

    section(
        "10. PHASE F14 DECISION"
    )

    if all_pass:
        print(
            "PHASE F14 VERDICT: GO — FIRST INDEPENDENT SHARE-seq RNA/ATAC "
            "GDIS PROFILES CALCULATED ON THE FROZEN COMMON CLOCK"
        )

        print()

        print(
            "The calculation is technically valid."
        )

        print()

        print(
            "Do NOT infer chromatin priming from peak locations yet."
        )

        print()

        print(
            "Next phase should diagnose modality-specific peak topology and "
            "window-size stability BEFORE any cross-modal event is paired."
        )

    else:
        print(
            "PHASE F14 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not perform cross-modal timing analysis until failed "
            "calculation criteria are understood."
        )

    print()

    print(
        "No cross-modal peak was paired."
    )

    print(
        "No lead/lag was calculated."
    )

    print(
        "No CMIL was calculated."
    )

    print(
        "No priming claim was made."
    )

    line("=")


if __name__ == "__main__":
    main()

