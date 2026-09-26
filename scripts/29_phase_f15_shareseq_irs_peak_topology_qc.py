#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
29_phase_f15_shareseq_irs_peak_topology_qc.py
=============================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F15
---------
Diagnose modality-specific GDIS peak topology and window-size persistence
BEFORE any RNA/ATAC event pairing or lead-lag calculation.

WHY F15 IS NECESSARY
--------------------
F14 produced technically valid and highly window-stable profiles.

However, a global maximum alone is not sufficient for cross-modal timing:
    - a modality may contain multiple local instability events;
    - a broad plateau may make the argmax unstable;
    - an apparent peak may lie too close to the observed pseudotime boundary;
    - different window sizes may resolve the same event differently.

F15 therefore treats RNA and ATAC INDEPENDENTLY.

NO CROSS-MODAL TIMING
---------------------
F15 does NOT:
    - subtract an ATAC peak from an RNA peak;
    - calculate lead/lag;
    - pair an ATAC family with an RNA family;
    - calculate CMIL;
    - bootstrap a cross-modal timing difference;
    - claim chromatin priming.

INPUTS
------
data/GSE140203/representations_f14/
    f14_independent_state_landmarks.tsv
    f14_common_window_metadata.tsv.gz
    f14_gdis_profiles.tsv.gz
    f14_gdis_summary.tsv
    f14_manifest.json

PROFILE PRIORITY
----------------
PRIMARY topology:
    profile_role = primary_multiome
    transition_weight = 0.00

SECONDARY sensitivity:
    profile_role = reference_pygdis
    transition_weight = 0.18

The PRIMARY topology is used for the F15 decision.

LOCAL-PEAK DETECTION
--------------------
All mathematical local maxima are first detected with scipy.signal.find_peaks.

A peak is called MATERIAL when its prominence satisfies:

    prominence >= max(
        0.005,
        0.05 * [profile max - profile min]
    )

This rule is fixed before examining cross-modal peak relationships.

No peak count is forced.

EDGE-PROXIMITY RULE
-------------------
A local peak is called edge-proximal if EITHER:

    peak index is within the first/last 3 profile points

OR:

    parameter distance to the nearest profile boundary
        <= 2 * median adjacent parameter spacing

This is deliberately conservative because an event too close to the complete-
window support boundary may be a truncation artifact.

PEAK FAMILIES ACROSS WINDOW SIZES
---------------------------------
Material peaks are clustered WITHIN ONE MODALITY and ONE PROFILE ROLE only.

Family-clustering tolerance is derived from the PRIMARY 400/100 window
resolution, not from peak locations:

    family_tolerance =
        max(0.02, 3 * median primary parameter spacing)

A PRIMARY peak family is called ROBUST when:
    - it appears in >= 2 of the 3 window configurations; AND
    - it includes primary_w400_s100 support.

A robust family is called ROBUST INTERIOR when:
    - it is ROBUST; AND
    - it has interior support in >= 2 window configurations.

A family supported by all three window configurations is additionally labeled
THREE-WINDOW.

INDEPENDENT BIOLOGICAL LANDMARK
-------------------------------
The already-frozen TAC-2 IQR and median are used only to annotate each family:
    - family inside TAC-2 IQR?
    - distance to TAC-2 median

The TAC-2 landmark is not used to select or cluster peaks.

ADDITIONAL TOPOLOGY
-------------------
F15 also reports:
    - raw global maximum;
    - all local maxima;
    - material local maxima;
    - local prominence;
    - full width at half prominence when available;
    - profile value at the peak;
    - state composition of the corresponding window;
    - whether the peak is edge-proximal;
    - number of robust / robust-interior families;
    - transition-energy local maxima as a separate diagnostic.

DECISION
--------
F15 issues:

GO — MODALITY-SPECIFIC INTERIOR EVENT FAMILIES AVAILABLE

only if BOTH RNA and ATAC each contain >=1 robust-interior PRIMARY GDIS family.

Otherwise:
REVIEW REQUIRED

This does NOT authorize a priming claim.
It only determines whether prospective cross-modal event pairing is
scientifically defensible.

OUTPUTS
-------
data/GSE140203/representations_f15/
    f15_all_local_peaks.tsv.gz
    f15_material_gdis_peaks.tsv
    f15_gdis_peak_families.tsv
    f15_transition_energy_peaks.tsv
    f15_topology_summary.tsv
    f15_manifest.json

RUN
---
    python 29_phase_f15_shareseq_irs_peak_topology_qc.py

DEPENDENCIES
------------
numpy
pandas
scipy
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import find_peaks, peak_prominences, peak_widths


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE140203"

DATA_DIR = Path("data") / ACCESSION
F14_DIR = DATA_DIR / "representations_f14"
F15_DIR = DATA_DIR / "representations_f15"

F14_LANDMARKS = F14_DIR / "f14_independent_state_landmarks.tsv"
F14_WINDOWS = F14_DIR / "f14_common_window_metadata.tsv.gz"
F14_PROFILES = F14_DIR / "f14_gdis_profiles.tsv.gz"
F14_SUMMARY = F14_DIR / "f14_gdis_summary.tsv"
F14_MANIFEST = F14_DIR / "f14_manifest.json"

OUTPUT_ALL_PEAKS = F15_DIR / "f15_all_local_peaks.tsv.gz"
OUTPUT_MATERIAL_GDIS = F15_DIR / "f15_material_gdis_peaks.tsv"
OUTPUT_FAMILIES = F15_DIR / "f15_gdis_peak_families.tsv"
OUTPUT_ENERGY = F15_DIR / "f15_transition_energy_peaks.tsv"
OUTPUT_SUMMARY = F15_DIR / "f15_topology_summary.tsv"
OUTPUT_MANIFEST = F15_DIR / "f15_manifest.json"


# =====================================================================
# FROZEN DESIGN
# =====================================================================

MODALITIES = [
    "RNA",
    "ATAC",
]

PRIMARY_PROFILE_ROLE = "primary_multiome"
REFERENCE_PROFILE_ROLE = "reference_pygdis"

PRIMARY_TRANSITION_WEIGHT = 0.00
REFERENCE_TRANSITION_WEIGHT = 0.18

WINDOW_CONFIGS = [
    "sensitivity_w300_s75",
    "primary_w400_s100",
    "sensitivity_w500_s125",
]

PRIMARY_WINDOW_CONFIG = "primary_w400_s100"

MATERIAL_PROMINENCE_ABSOLUTE_FLOOR = 0.005
MATERIAL_PROMINENCE_RANGE_FRACTION = 0.05

EDGE_POINT_COUNT = 3
EDGE_SPACING_MULTIPLIER = 2.0

FAMILY_TOLERANCE_MIN = 0.02
FAMILY_TOLERANCE_SPACING_MULTIPLIER = 3.0

ROBUST_MIN_WINDOW_CONFIGS = 2
ROBUST_INTERIOR_MIN_WINDOW_CONFIGS = 2

TRANSITION_STATE = "TAC-2"


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
        "display.max_colwidth", 160,
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


# =====================================================================
# INPUTS
# =====================================================================

def require_inputs():
    required = [
        F14_LANDMARKS,
        F14_WINDOWS,
        F14_PROFILES,
        F14_SUMMARY,
        F14_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required F14 input(s):\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )


def load_inputs():
    landmarks = pd.read_csv(
        F14_LANDMARKS,
        sep="\t",
        low_memory=False,
    )

    windows = pd.read_csv(
        F14_WINDOWS,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    profiles = pd.read_csv(
        F14_PROFILES,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    summary = pd.read_csv(
        F14_SUMMARY,
        sep="\t",
        low_memory=False,
    )

    required_profile_cols = [
        "modality",
        "window_config",
        "profile_role",
        "transition_weight",
        "window_index",
        "parameter",
        "gdis",
        "transition_energy",
        "sustained_instability",
        "transition_instability",
        "TAC1_fraction",
        "TAC2_fraction",
        "IRS_fraction",
        "dominant_state",
    ]

    missing_profile = [
        column
        for column in required_profile_cols
        if column not in profiles.columns
    ]

    if missing_profile:
        raise RuntimeError(
            "F14 profile table missing column(s): "
            + ", ".join(
                missing_profile
            )
        )

    tac2 = landmarks.loc[
        landmarks[
            "state"
        ].astype(str)
        == TRANSITION_STATE
    ]

    if len(
        tac2
    ) != 1:
        raise RuntimeError(
            "Could not identify unique TAC-2 landmark."
        )

    tac2 = tac2.iloc[
        0
    ]

    tac2_landmark = {
        "median": float(
            tac2[
                "median"
            ]
        ),
        "q25": float(
            tac2[
                "q25"
            ]
        ),
        "q75": float(
            tac2[
                "q75"
            ]
        ),
    }

    return (
        landmarks,
        windows,
        profiles,
        summary,
        tac2_landmark,
    )


# =====================================================================
# FAMILY RESOLUTION TOLERANCE
# =====================================================================

def calculate_family_tolerance(
    profiles,
):
    primary = profiles.loc[
        (
            profiles[
                "modality"
            ]
            == "RNA"
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
            == PRIMARY_PROFILE_ROLE
        )
    ].sort_values(
        "parameter"
    )

    if len(
        primary
    ) < 4:
        raise RuntimeError(
            "Insufficient primary profile points for resolution tolerance."
        )

    diffs = np.diff(
        primary[
            "parameter"
        ].to_numpy(
            dtype=float
        )
    )

    median_spacing = float(
        np.median(
            diffs
        )
    )

    tolerance = max(
        FAMILY_TOLERANCE_MIN,
        FAMILY_TOLERANCE_SPACING_MULTIPLIER
        * median_spacing,
    )

    return (
        median_spacing,
        float(
            tolerance
        ),
    )


# =====================================================================
# LOCAL PEAK DETECTION
# =====================================================================

def interpolate_parameter_at_index(
    parameters,
    fractional_index,
):
    indices = np.arange(
        len(
            parameters
        ),
        dtype=float,
    )

    return float(
        np.interp(
            fractional_index,
            indices,
            parameters,
        )
    )


def detect_local_peaks(
    profile,
    value_column,
    modality,
    profile_role,
    window_config,
):
    profile = profile.sort_values(
        "parameter"
    ).reset_index(
        drop=True
    )

    x = profile[
        "parameter"
    ].to_numpy(
        dtype=float
    )

    y = profile[
        value_column
    ].to_numpy(
        dtype=float
    )

    if len(
        y
    ) < 5:
        raise RuntimeError(
            f"{modality}/{profile_role}/{window_config}: "
            f"too few profile points."
        )

    if not np.isfinite(
        x
    ).all() or not np.isfinite(
        y
    ).all():
        raise RuntimeError(
            "Profile contains NaN/Inf."
        )

    if np.any(
        np.diff(
            x
        )
        <= 0
    ):
        raise RuntimeError(
            "Profile parameters are not strictly increasing."
        )

    peak_indices, _ = find_peaks(
        y
    )

    profile_range = float(
        np.max(
            y
        )
        - np.min(
            y
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
                x
            )
        )
    )

    edge_parameter_threshold = (
        EDGE_SPACING_MULTIPLIER
        * median_spacing
    )

    rows = []

    if len(
        peak_indices
    ) == 0:
        return (
            pd.DataFrame(),
            {
                "profile_range": profile_range,
                "material_prominence_threshold": material_threshold,
                "median_parameter_spacing": median_spacing,
                "edge_parameter_threshold": edge_parameter_threshold,
                "global_max_index": int(
                    np.argmax(
                        y
                    )
                ),
                "global_max_parameter": float(
                    x[
                        np.argmax(
                            y
                        )
                    ]
                ),
            },
        )

    prominences, left_bases, right_bases = peak_prominences(
        y,
        peak_indices,
    )

    widths, width_heights, left_ips, right_ips = peak_widths(
        y,
        peak_indices,
        rel_height=0.5,
        prominence_data=(
            prominences,
            left_bases,
            right_bases,
        ),
    )

    global_max_index = int(
        np.argmax(
            y
        )
    )

    for local_number, peak_index in enumerate(
        peak_indices,
        start=1,
    ):
        prominence = float(
            prominences[
                local_number
                - 1
            ]
        )

        parameter = float(
            x[
                peak_index
            ]
        )

        distance_left = (
            parameter
            - float(
                x[
                    0
                ]
            )
        )

        distance_right = (
            float(
                x[
                    -1
                ]
            )
            - parameter
        )

        point_edge = bool(
            peak_index
            < EDGE_POINT_COUNT
            or peak_index
            >= (
                len(
                    y
                )
                - EDGE_POINT_COUNT
            )
        )

        parameter_edge = bool(
            min(
                distance_left,
                distance_right,
            )
            <= edge_parameter_threshold
        )

        edge_proximal = bool(
            point_edge
            or parameter_edge
        )

        left_parameter = interpolate_parameter_at_index(
            x,
            float(
                left_ips[
                    local_number
                    - 1
                ]
            ),
        )

        right_parameter = interpolate_parameter_at_index(
            x,
            float(
                right_ips[
                    local_number
                    - 1
                ]
            ),
        )

        row = profile.iloc[
            peak_index
        ]

        rows.append(
            {
                "modality": modality,
                "profile_role": profile_role,
                "transition_weight": float(
                    row[
                        "transition_weight"
                    ]
                ),
                "window_config": window_config,
                "value_column": value_column,
                "local_peak_number": local_number,
                "peak_index": int(
                    peak_index
                ),
                "n_profile_points": int(
                    len(
                        y
                    )
                ),
                "parameter": parameter,
                "value": float(
                    y[
                        peak_index
                    ]
                ),
                "prominence": prominence,
                "profile_range": profile_range,
                "prominence_fraction_of_range": (
                    prominence
                    / profile_range
                    if profile_range
                    > 0
                    else np.nan
                ),
                "material_prominence_threshold": material_threshold,
                "is_material": bool(
                    prominence
                    >= material_threshold
                ),
                "is_global_maximum": bool(
                    peak_index
                    == global_max_index
                ),
                "distance_to_left_boundary": distance_left,
                "distance_to_right_boundary": distance_right,
                "median_parameter_spacing": median_spacing,
                "edge_parameter_threshold": edge_parameter_threshold,
                "edge_by_point_index": point_edge,
                "edge_by_parameter_distance": parameter_edge,
                "edge_proximal": edge_proximal,
                "interior_peak": bool(
                    not edge_proximal
                ),
                "prominence_left_base_index": int(
                    left_bases[
                        local_number
                        - 1
                    ]
                ),
                "prominence_right_base_index": int(
                    right_bases[
                        local_number
                        - 1
                    ]
                ),
                "half_prominence_width_points": float(
                    widths[
                        local_number
                        - 1
                    ]
                ),
                "half_prominence_left_parameter": left_parameter,
                "half_prominence_right_parameter": right_parameter,
                "half_prominence_width_parameter": (
                    right_parameter
                    - left_parameter
                ),
                "dominant_state": str(
                    row[
                        "dominant_state"
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
            }
        )

    metadata = {
        "profile_range": profile_range,
        "material_prominence_threshold": material_threshold,
        "median_parameter_spacing": median_spacing,
        "edge_parameter_threshold": edge_parameter_threshold,
        "global_max_index": global_max_index,
        "global_max_parameter": float(
            x[
                global_max_index
            ]
        ),
    }

    return (
        pd.DataFrame(
            rows
        ),
        metadata,
    )


# =====================================================================
# FAMILY CLUSTERING
# =====================================================================

def cluster_peak_families(
    material_peaks,
    modality,
    profile_role,
    family_tolerance,
    tac2_landmark,
):
    subset = material_peaks.loc[
        (
            material_peaks[
                "modality"
            ]
            == modality
        )
        & (
            material_peaks[
                "profile_role"
            ]
            == profile_role
        )
        & (
            material_peaks[
                "value_column"
            ]
            == "gdis"
        )
        & (
            material_peaks[
                "is_material"
            ]
        )
    ].copy()

    if subset.empty:
        return (
            pd.DataFrame(),
            subset,
        )

    positions = subset[
        "parameter"
    ].to_numpy(
        dtype=float
    )

    if len(
        subset
    ) == 1:
        cluster_ids = np.ones(
            1,
            dtype=int,
        )
    else:
        z = linkage(
            positions.reshape(
                -1,
                1,
            ),
            method="complete",
            metric="euclidean",
        )

        cluster_ids = fcluster(
            z,
            t=family_tolerance,
            criterion="distance",
        )

    subset[
        "family_cluster_id"
    ] = cluster_ids

    family_rows = []

    ordered_cluster_ids = (
        subset.groupby(
            "family_cluster_id"
        )[
            "parameter"
        ]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    cluster_to_family = {
        cluster_id: index + 1
        for index, cluster_id in enumerate(
            ordered_cluster_ids
        )
    }

    subset[
        "family_id"
    ] = subset[
        "family_cluster_id"
    ].map(
        cluster_to_family
    )

    for family_id, group in subset.groupby(
        "family_id"
    ):
        configs = sorted(
            group[
                "window_config"
            ].astype(str).unique().tolist()
        )

        support_count = len(
            configs
        )

        primary_support = bool(
            PRIMARY_WINDOW_CONFIG
            in configs
        )

        interior_configs = sorted(
            group.loc[
                group[
                    "interior_peak"
                ],
                "window_config",
            ].astype(str).unique().tolist()
        )

        interior_support_count = len(
            interior_configs
        )

        robust = bool(
            support_count
            >= ROBUST_MIN_WINDOW_CONFIGS
            and primary_support
        )

        robust_interior = bool(
            robust
            and interior_support_count
            >= ROBUST_INTERIOR_MIN_WINDOW_CONFIGS
        )

        three_window = bool(
            support_count
            == len(
                WINDOW_CONFIGS
            )
        )

        center = float(
            np.median(
                group[
                    "parameter"
                ].to_numpy(
                    dtype=float
                )
            )
        )

        minimum = float(
            group[
                "parameter"
            ].min()
        )

        maximum = float(
            group[
                "parameter"
            ].max()
        )

        family_rows.append(
            {
                "modality": modality,
                "profile_role": profile_role,
                "family_id": int(
                    family_id
                ),
                "family_center_parameter": center,
                "family_min_parameter": minimum,
                "family_max_parameter": maximum,
                "family_span": (
                    maximum
                    - minimum
                ),
                "n_peak_observations": int(
                    len(
                        group
                    )
                ),
                "window_config_support_count": int(
                    support_count
                ),
                "window_configs": ",".join(
                    configs
                ),
                "includes_primary_window_config": primary_support,
                "interior_window_support_count": int(
                    interior_support_count
                ),
                "interior_window_configs": ",".join(
                    interior_configs
                ),
                "robust_family": robust,
                "robust_interior_family": robust_interior,
                "three_window_family": three_window,
                "median_prominence": float(
                    group[
                        "prominence"
                    ].median()
                ),
                "max_prominence": float(
                    group[
                        "prominence"
                    ].max()
                ),
                "median_prominence_fraction_of_range": float(
                    group[
                        "prominence_fraction_of_range"
                    ].median()
                ),
                "edge_proximal_fraction": float(
                    group[
                        "edge_proximal"
                    ].mean()
                ),
                "median_peak_value": float(
                    group[
                        "value"
                    ].median()
                ),
                "median_TAC1_fraction": float(
                    group[
                        "TAC1_fraction"
                    ].median()
                ),
                "median_TAC2_fraction": float(
                    group[
                        "TAC2_fraction"
                    ].median()
                ),
                "median_IRS_fraction": float(
                    group[
                        "IRS_fraction"
                    ].median()
                ),
                "inside_TAC2_IQR": bool(
                    tac2_landmark[
                        "q25"
                    ]
                    <= center
                    <= tac2_landmark[
                        "q75"
                    ]
                ),
                "distance_to_TAC2_median": (
                    center
                    - tac2_landmark[
                        "median"
                    ]
                ),
                "absolute_distance_to_TAC2_median": abs(
                    center
                    - tac2_landmark[
                        "median"
                    ]
                ),
                "family_tolerance": family_tolerance,
            }
        )

    families = pd.DataFrame(
        family_rows
    ).sort_values(
        "family_center_parameter"
    ).reset_index(
        drop=True
    )

    return (
        families,
        subset,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F15 — SHARE-seq IRS MODALITY-SPECIFIC GDIS PEAK TOPOLOGY QC"
    )

    print(
        "RNA and ATAC are analyzed independently."
    )

    print()

    print(
        "Primary topology: lambda_t = 0.00."
    )

    print(
        "Reference sensitivity: lambda_t = 0.18."
    )

    print()

    print(
        "No cross-modal event pairing."
    )

    print(
        "No lead/lag."
    )

    print(
        "No CMIL."
    )

    require_inputs()

    F15_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        landmarks,
        windows,
        profiles,
        summary,
        tac2_landmark,
    ) = load_inputs()

    # -----------------------------------------------------------------
    # Resolution.
    # -----------------------------------------------------------------

    section(
        "1. FREEZE TOPOLOGY RESOLUTION / EDGE RULES"
    )

    (
        primary_median_spacing,
        family_tolerance,
    ) = calculate_family_tolerance(
        profiles
    )

    print(
        f"Primary 400/100 median parameter spacing: "
        f"{primary_median_spacing:.6f}"
    )

    print(
        f"Peak-family clustering tolerance: "
        f"{family_tolerance:.6f}"
    )

    print()

    print(
        f"Material prominence rule: max("
        f"{MATERIAL_PROMINENCE_ABSOLUTE_FLOOR:.3f}, "
        f"{MATERIAL_PROMINENCE_RANGE_FRACTION:.2f} x profile range)"
    )

    print(
        f"Edge rule: first/last {EDGE_POINT_COUNT} points OR "
        f"within {EDGE_SPACING_MULTIPLIER:.1f} x median spacing "
        f"of profile boundary."
    )

    print()

    print(
        f"Frozen TAC-2 median: {tac2_landmark['median']:.6f}"
    )

    print(
        f"Frozen TAC-2 IQR: "
        f"[{tac2_landmark['q25']:.6f}, "
        f"{tac2_landmark['q75']:.6f}]"
    )

    # -----------------------------------------------------------------
    # Detect all GDIS and transition-energy peaks.
    # -----------------------------------------------------------------

    section(
        "2. DETECT ALL WITHIN-MODALITY LOCAL MAXIMA"
    )

    all_peak_tables = []
    profile_metadata_rows = []

    for modality in MODALITIES:
        for profile_role in [
            PRIMARY_PROFILE_ROLE,
            REFERENCE_PROFILE_ROLE,
        ]:
            for window_config in WINDOW_CONFIGS:
                subset = profiles.loc[
                    (
                        profiles[
                            "modality"
                        ]
                        == modality
                    )
                    & (
                        profiles[
                            "profile_role"
                        ]
                        == profile_role
                    )
                    & (
                        profiles[
                            "window_config"
                        ]
                        == window_config
                    )
                ].copy()

                if subset.empty:
                    raise RuntimeError(
                        f"Missing profile: {modality}/"
                        f"{profile_role}/{window_config}"
                    )

                for value_column in [
                    "gdis",
                    "transition_energy",
                ]:
                    (
                        peak_table,
                        metadata,
                    ) = detect_local_peaks(
                        subset,
                        value_column,
                        modality,
                        profile_role,
                        window_config,
                    )

                    if not peak_table.empty:
                        all_peak_tables.append(
                            peak_table
                        )

                    profile_metadata_rows.append(
                        {
                            "modality": modality,
                            "profile_role": profile_role,
                            "window_config": window_config,
                            "value_column": value_column,
                            **metadata,
                        }
                    )

    if all_peak_tables:
        all_peaks = pd.concat(
            all_peak_tables,
            ignore_index=True,
        )
    else:
        all_peaks = pd.DataFrame()

    profile_metadata = pd.DataFrame(
        profile_metadata_rows
    )

    all_peaks.to_csv(
        OUTPUT_ALL_PEAKS,
        sep="\t",
        index=False,
        compression="gzip",
    )

    # -----------------------------------------------------------------
    # GDIS material peaks.
    # -----------------------------------------------------------------

    section(
        "3. MATERIAL GDIS LOCAL MAXIMA"
    )

    material_gdis = all_peaks.loc[
        (
            all_peaks[
                "value_column"
            ]
            == "gdis"
        )
        & (
            all_peaks[
                "is_material"
            ]
        )
    ].copy()

    material_gdis.to_csv(
        OUTPUT_MATERIAL_GDIS,
        sep="\t",
        index=False,
    )

    display_columns = [
        "modality",
        "profile_role",
        "window_config",
        "peak_index",
        "parameter",
        "value",
        "prominence",
        "prominence_fraction_of_range",
        "is_global_maximum",
        "edge_proximal",
        "interior_peak",
        "dominant_state",
        "TAC1_fraction",
        "TAC2_fraction",
        "IRS_fraction",
    ]

    print_df(
        material_gdis[
            display_columns
        ].set_index(
            [
                "modality",
                "profile_role",
                "window_config",
                "peak_index",
            ]
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Peak families.
    # -----------------------------------------------------------------

    section(
        "4. CLUSTER GDIS PEAKS INTO WITHIN-MODALITY WINDOW-STABLE FAMILIES"
    )

    family_tables = []
    assigned_material_tables = []

    for modality in MODALITIES:
        for profile_role in [
            PRIMARY_PROFILE_ROLE,
            REFERENCE_PROFILE_ROLE,
        ]:
            families, assigned = cluster_peak_families(
                material_gdis,
                modality,
                profile_role,
                family_tolerance,
                tac2_landmark,
            )

            if not families.empty:
                family_tables.append(
                    families
                )

            if not assigned.empty:
                assigned_material_tables.append(
                    assigned
                )

    if family_tables:
        families = pd.concat(
            family_tables,
            ignore_index=True,
        )
    else:
        families = pd.DataFrame()

    families.to_csv(
        OUTPUT_FAMILIES,
        sep="\t",
        index=False,
    )

    if families.empty:
        print(
            "No material GDIS peak families detected."
        )
    else:
        family_display = [
            "modality",
            "profile_role",
            "family_id",
            "family_center_parameter",
            "family_min_parameter",
            "family_max_parameter",
            "window_config_support_count",
            "includes_primary_window_config",
            "interior_window_support_count",
            "robust_family",
            "robust_interior_family",
            "three_window_family",
            "median_prominence",
            "edge_proximal_fraction",
            "inside_TAC2_IQR",
            "distance_to_TAC2_median",
            "median_TAC1_fraction",
            "median_TAC2_fraction",
            "median_IRS_fraction",
        ]

        print_df(
            families[
                family_display
            ].set_index(
                [
                    "modality",
                    "profile_role",
                    "family_id",
                ]
            ),
            digits=6,
        )

    # -----------------------------------------------------------------
    # Transition energy topology diagnostic.
    # -----------------------------------------------------------------

    section(
        "5. TRANSITION-ENERGY LOCAL MAXIMA — SEPARATE DIAGNOSTIC"
    )

    energy = all_peaks.loc[
        (
            all_peaks[
                "value_column"
            ]
            == "transition_energy"
        )
        & (
            all_peaks[
                "is_material"
            ]
        )
    ].copy()

    energy.to_csv(
        OUTPUT_ENERGY,
        sep="\t",
        index=False,
    )

    if energy.empty:
        print(
            "No material transition-energy local maxima detected."
        )
    else:
        energy_display = [
            "modality",
            "profile_role",
            "window_config",
            "peak_index",
            "parameter",
            "value",
            "prominence",
            "edge_proximal",
            "interior_peak",
            "dominant_state",
            "TAC1_fraction",
            "TAC2_fraction",
            "IRS_fraction",
        ]

        print_df(
            energy[
                energy_display
            ].set_index(
                [
                    "modality",
                    "profile_role",
                    "window_config",
                    "peak_index",
                ]
            ),
            digits=6,
        )

    # -----------------------------------------------------------------
    # Primary topology summary.
    # -----------------------------------------------------------------

    section(
        "6. PRIMARY lambda=0 TOPOLOGY SUMMARY"
    )

    summary_rows = []

    for modality in MODALITIES:
        primary_families = families.loc[
            (
                families[
                    "modality"
                ]
                == modality
            )
            & (
                families[
                    "profile_role"
                ]
                == PRIMARY_PROFILE_ROLE
            )
        ].copy()

        robust = primary_families.loc[
            primary_families[
                "robust_family"
            ]
        ]

        robust_interior = primary_families.loc[
            primary_families[
                "robust_interior_family"
            ]
        ]

        three_window = primary_families.loc[
            primary_families[
                "three_window_family"
            ]
        ]

        primary_profile_peaks = material_gdis.loc[
            (
                material_gdis[
                    "modality"
                ]
                == modality
            )
            & (
                material_gdis[
                    "profile_role"
                ]
                == PRIMARY_PROFILE_ROLE
            )
        ]

        global_peaks = primary_profile_peaks.loc[
            primary_profile_peaks[
                "is_global_maximum"
            ]
        ]

        global_edge_fraction = (
            float(
                global_peaks[
                    "edge_proximal"
                ].mean()
            )
            if not global_peaks.empty
            else np.nan
        )

        summary_rows.append(
            {
                "modality": modality,
                "n_material_peak_observations": int(
                    len(
                        primary_profile_peaks
                    )
                ),
                "n_peak_families": int(
                    len(
                        primary_families
                    )
                ),
                "n_robust_families": int(
                    len(
                        robust
                    )
                ),
                "n_robust_interior_families": int(
                    len(
                        robust_interior
                    )
                ),
                "n_three_window_families": int(
                    len(
                        three_window
                    )
                ),
                "global_max_edge_proximal_fraction_across_windows": (
                    global_edge_fraction
                ),
                "has_robust_interior_family": bool(
                    len(
                        robust_interior
                    )
                    >= 1
                ),
                "robust_interior_family_centers": (
                    ",".join(
                        f"{value:.6f}"
                        for value in robust_interior[
                            "family_center_parameter"
                        ].sort_values()
                    )
                    if not robust_interior.empty
                    else ""
                ),
                "robust_family_centers_all": (
                    ",".join(
                        f"{value:.6f}"
                        for value in robust[
                            "family_center_parameter"
                        ].sort_values()
                    )
                    if not robust.empty
                    else ""
                ),
            }
        )

    topology_summary = pd.DataFrame(
        summary_rows
    ).set_index(
        "modality"
    )

    print_df(
        topology_summary,
        digits=6,
    )

    topology_summary.reset_index().to_csv(
        OUTPUT_SUMMARY,
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Safeguards / decision.
    # -----------------------------------------------------------------

    section(
        "7. PHASE F15 EVENT-FAMILY SUITABILITY CHECKS"
    )

    primary_role_present = bool(
        set(
            profiles.loc[
                profiles[
                    "profile_role"
                ]
                == PRIMARY_PROFILE_ROLE,
                "modality",
            ].unique()
        )
        == set(
            MODALITIES
        )
    )

    all_primary_configs_present = True

    for modality in MODALITIES:
        observed_configs = set(
            profiles.loc[
                (
                    profiles[
                        "profile_role"
                    ]
                    == PRIMARY_PROFILE_ROLE
                )
                & (
                    profiles[
                        "modality"
                    ]
                    == modality
                ),
                "window_config",
            ].unique()
        )

        if observed_configs != set(
            WINDOW_CONFIGS
        ):
            all_primary_configs_present = False

    rna_has_interior = bool(
        topology_summary.loc[
            "RNA",
            "has_robust_interior_family",
        ]
    )

    atac_has_interior = bool(
        topology_summary.loc[
            "ATAC",
            "has_robust_interior_family",
        ]
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    "Primary lambda=0 profiles present for both modalities"
                ),
                "pass": primary_role_present,
            },
            {
                "criterion": (
                    "All 3 window configurations present for both "
                    "primary modality profiles"
                ),
                "pass": all_primary_configs_present,
            },
            {
                "criterion": (
                    "RNA contains >=1 robust-interior primary GDIS family"
                ),
                "pass": rna_has_interior,
            },
            {
                "criterion": (
                    "ATAC contains >=1 robust-interior primary GDIS family"
                ),
                "pass": atac_has_interior,
            },
            {
                "criterion": (
                    "Peak-family tolerance is derived from window resolution"
                ),
                "pass": (
                    family_tolerance
                    >= FAMILY_TOLERANCE_MIN
                ),
            },
            {
                "criterion": (
                    "No biological landmark used to detect or cluster peaks"
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
        "8. SAVE F15 MANIFEST"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F15",
        "created_utc": utc_now_iso(),
        "purpose": (
            "Within-modality GDIS peak-topology and edge-effect QC before "
            "cross-modal event pairing."
        ),
        "primary_profile": {
            "profile_role": PRIMARY_PROFILE_ROLE,
            "transition_weight": PRIMARY_TRANSITION_WEIGHT,
        },
        "reference_profile": {
            "profile_role": REFERENCE_PROFILE_ROLE,
            "transition_weight": REFERENCE_TRANSITION_WEIGHT,
        },
        "material_peak_rule": {
            "absolute_prominence_floor": (
                MATERIAL_PROMINENCE_ABSOLUTE_FLOOR
            ),
            "profile_range_fraction": (
                MATERIAL_PROMINENCE_RANGE_FRACTION
            ),
        },
        "edge_rule": {
            "edge_point_count": EDGE_POINT_COUNT,
            "spacing_multiplier": EDGE_SPACING_MULTIPLIER,
        },
        "family_clustering": {
            "primary_median_parameter_spacing": primary_median_spacing,
            "family_tolerance_min": FAMILY_TOLERANCE_MIN,
            "spacing_multiplier": FAMILY_TOLERANCE_SPACING_MULTIPLIER,
            "resolved_family_tolerance": family_tolerance,
            "linkage": "complete",
            "robust_min_window_configs": ROBUST_MIN_WINDOW_CONFIGS,
            "robust_requires_primary_window_config": True,
            "robust_interior_min_window_configs": (
                ROBUST_INTERIOR_MIN_WINDOW_CONFIGS
            ),
        },
        "independent_landmark_annotation": {
            "state": TRANSITION_STATE,
            "median": tac2_landmark[
                "median"
            ],
            "q25": tac2_landmark[
                "q25"
            ],
            "q75": tac2_landmark[
                "q75"
            ],
            "used_for_peak_detection": False,
            "used_for_family_clustering": False,
        },
        "topology_summary": (
            topology_summary.reset_index().to_dict(
                orient="records"
            )
        ),
        "checks": checks.to_dict(
            orient="records"
        ),
        "outputs": {
            "all_local_peaks": {
                "file": str(
                    OUTPUT_ALL_PEAKS
                ),
                "sha256": sha256_file(
                    OUTPUT_ALL_PEAKS
                ),
            },
            "material_gdis_peaks": {
                "file": str(
                    OUTPUT_MATERIAL_GDIS
                ),
                "sha256": sha256_file(
                    OUTPUT_MATERIAL_GDIS
                ),
            },
            "gdis_peak_families": {
                "file": str(
                    OUTPUT_FAMILIES
                ),
                "sha256": sha256_file(
                    OUTPUT_FAMILIES
                ),
            },
            "transition_energy_peaks": {
                "file": str(
                    OUTPUT_ENERGY
                ),
                "sha256": sha256_file(
                    OUTPUT_ENERGY
                ),
            },
            "topology_summary": {
                "file": str(
                    OUTPUT_SUMMARY
                ),
                "sha256": sha256_file(
                    OUTPUT_SUMMARY
                ),
            },
        },
        "guardrails": {
            "cross_modal_peak_pairing_performed": False,
            "lead_lag_calculated": False,
            "cmil_calculated": False,
            "bootstrap_timing_calculated": False,
            "cross_modal_alignment_null_calculated": False,
            "priming_claim_made": False,
            "trajectory_changed": False,
            "representation_changed": False,
            "gdis_recomputed": False,
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
        OUTPUT_ALL_PEAKS,
        OUTPUT_MATERIAL_GDIS,
        OUTPUT_FAMILIES,
        OUTPUT_ENERGY,
        OUTPUT_SUMMARY,
        OUTPUT_MANIFEST,
    ]:
        print(
            path
        )

    # -----------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------

    section(
        "9. PHASE F15 DECISION"
    )

    if all_pass:
        print(
            "PHASE F15 VERDICT: GO — MODALITY-SPECIFIC ROBUST INTERIOR "
            "GDIS EVENT FAMILIES ARE AVAILABLE"
        )

        print()

        print(
            "A prospective cross-modal event-pairing rule may now be defined "
            "using the independently frozen biological transition context."
        )

        print()

        print(
            "No event pair has been selected yet."
        )

    else:
        print(
            "PHASE F15 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "At least one modality lacks a robust-interior primary GDIS "
            "event family under the prespecified topology rules."
        )

        print()

        print(
            "Do not calculate cross-modal lead/lag from global maxima."
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

