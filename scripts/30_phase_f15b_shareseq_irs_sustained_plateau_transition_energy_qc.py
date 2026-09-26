#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
30_phase_f15b_shareseq_irs_sustained_plateau_transition_energy_qc.py
===================================================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F15b
----------
Resolve the F15 topology finding without forcing an RNA GDIS peak.

F15 found:
    - ATAC primary lambda_t=0 GDIS contains a robust interior local-peak family.
    - RNA primary lambda_t=0 GDIS contains NO material local maximum under the
      prespecified prominence rule.

This means a cross-modal comparison of GDIS argmax locations is not valid.

F15b therefore separates two questions:

1. SUSTAINED GDIS ARCHITECTURE
   Is the RNA lambda_t=0 profile a broad near-maximum plateau / shoulder rather
   than a localized instability event?

2. TRANSITION-ENERGY TOPOLOGY
   Does each modality contain reproducible, interior localized transition-
   energy families across the three frozen window designs?

NO CROSS-MODAL TIMING
---------------------
F15b does NOT:
    - pair an RNA family with an ATAC family;
    - subtract family centers;
    - calculate lead/lag;
    - calculate CMIL;
    - bootstrap cross-modal timing;
    - claim chromatin priming.

INPUTS
------
data/GSE140203/representations_f14/
    f14_gdis_profiles.tsv.gz
    f14_independent_state_landmarks.tsv
    f14_manifest.json

data/GSE140203/representations_f15/
    f15_transition_energy_peaks.tsv
    f15_topology_summary.tsv
    f15_manifest.json

PRIMARY PROFILE
---------------
Only:
    profile_role = primary_multiome
    transition_weight = 0.00

is used for sustained-GDIS architecture.

Transition energy is descriptor-derived and identical between lambda_t=0 and
lambda_t=0.18, so F15b de-duplicates it and uses one copy only.

SUSTAINED-GDIS NEAR-MAX ENVELOPES
---------------------------------
For each modality/window profile, F15b reports the contiguous region
containing the global maximum where:

    GDIS >= min + q * (max - min)

for q =:
    0.90
    0.95
    0.99

This avoids inventing a local peak where none exists.

The following are reported:
    - first/last pseudotime in the near-max component;
    - pseudotime width;
    - number/fraction of profile points in the component;
    - global-max position;
    - whether the component intersects TAC-2 IQR;
    - derivative sign balance and total variation.

No single plateau threshold is optimized.

TRANSITION-ENERGY FAMILY RULE
-----------------------------
F15 already detected material transition-energy local maxima using the frozen
prominence rule.

F15b clusters those material peaks WITHIN EACH MODALITY using the SAME
resolution-derived tolerance frozen in F15:

    family_tolerance = max(0.02, 3 * primary median parameter spacing)

Robust family:
    support in >=2 of 3 window configurations
    AND includes primary_w400_s100

Robust-interior family:
    robust
    AND interior support in >=2 configurations

Three-window family:
    support in all 3 configurations

The independent TAC-2 median/IQR is used only for annotation:
    - family center inside TAC-2 IQR?
    - distance to TAC-2 median

It is NOT used to form families.

DECISION
--------
F15b issues:

GO — LOCALIZED TRANSITION-ENERGY FAMILIES AVAILABLE

only if BOTH RNA and ATAC contain >=1 robust-interior transition-energy family.

This does NOT authorize cross-modal pairing by itself.
It only establishes that localized event families exist in both modalities
even though the sustained-GDIS architectures differ.

OUTPUTS
-------
data/GSE140203/representations_f15b/
    f15b_sustained_gdis_envelopes.tsv
    f15b_sustained_gdis_shape_summary.tsv
    f15b_transition_energy_families.tsv
    f15b_transition_energy_peak_assignments.tsv.gz
    f15b_topology_summary.tsv
    f15b_manifest.json

RUN
---
    python 30_phase_f15b_shareseq_irs_sustained_plateau_transition_energy_qc.py

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


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE140203"

DATA_DIR = Path("data") / ACCESSION
F14_DIR = DATA_DIR / "representations_f14"
F15_DIR = DATA_DIR / "representations_f15"
F15B_DIR = DATA_DIR / "representations_f15b"

F14_PROFILES = F14_DIR / "f14_gdis_profiles.tsv.gz"
F14_LANDMARKS = F14_DIR / "f14_independent_state_landmarks.tsv"
F14_MANIFEST = F14_DIR / "f14_manifest.json"

F15_ENERGY = F15_DIR / "f15_transition_energy_peaks.tsv"
F15_SUMMARY = F15_DIR / "f15_topology_summary.tsv"
F15_MANIFEST = F15_DIR / "f15_manifest.json"

OUTPUT_ENVELOPES = F15B_DIR / "f15b_sustained_gdis_envelopes.tsv"
OUTPUT_SHAPE = F15B_DIR / "f15b_sustained_gdis_shape_summary.tsv"
OUTPUT_ENERGY_FAMILIES = F15B_DIR / "f15b_transition_energy_families.tsv"
OUTPUT_ENERGY_ASSIGNMENTS = F15B_DIR / "f15b_transition_energy_peak_assignments.tsv.gz"
OUTPUT_SUMMARY = F15B_DIR / "f15b_topology_summary.tsv"
OUTPUT_MANIFEST = F15B_DIR / "f15b_manifest.json"


# =====================================================================
# FROZEN DESIGN
# =====================================================================

MODALITIES = [
    "RNA",
    "ATAC",
]

PRIMARY_PROFILE_ROLE = "primary_multiome"
PRIMARY_TRANSITION_WEIGHT = 0.00

WINDOW_CONFIGS = [
    "sensitivity_w300_s75",
    "primary_w400_s100",
    "sensitivity_w500_s125",
]

PRIMARY_WINDOW_CONFIG = "primary_w400_s100"

NEAR_MAX_LEVELS = [
    0.90,
    0.95,
    0.99,
]

FAMILY_TOLERANCE_MIN = 0.02
FAMILY_TOLERANCE_SPACING_MULTIPLIER = 3.0

ROBUST_MIN_WINDOW_CONFIGS = 2
ROBUST_INTERIOR_MIN_WINDOW_CONFIGS = 2

TRANSITION_STATE = "TAC-2"


# =====================================================================
# DISPLAY / HELPERS
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
        F14_PROFILES,
        F14_LANDMARKS,
        F14_MANIFEST,
        F15_ENERGY,
        F15_SUMMARY,
        F15_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required F14/F15 input(s):\n"
            + "\n".join(
                f"  {path}"
                for path in missing
            )
        )


def load_inputs():
    profiles = pd.read_csv(
        F14_PROFILES,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    landmarks = pd.read_csv(
        F14_LANDMARKS,
        sep="\t",
        low_memory=False,
    )

    energy = pd.read_csv(
        F15_ENERGY,
        sep="\t",
        low_memory=False,
    )

    topology = pd.read_csv(
        F15_SUMMARY,
        sep="\t",
        low_memory=False,
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
        profiles,
        energy,
        topology,
        tac2_landmark,
    )


# =====================================================================
# FROZEN RESOLUTION
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

    diffs = np.diff(
        primary[
            "parameter"
        ].to_numpy(
            dtype=float
        )
    )

    if len(
        diffs
    ) < 2:
        raise RuntimeError(
            "Insufficient primary profile resolution."
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
# SUSTAINED-GDIS ENVELOPE / SHAPE
# =====================================================================

def contiguous_component_containing_index(
    mask,
    target_index,
):
    """
    Return inclusive start/end indices of the contiguous True component
    containing target_index.
    """
    if not mask[
        target_index
    ]:
        raise RuntimeError(
            "Target index is not inside requested mask."
        )

    start = target_index

    while (
        start
        > 0
        and mask[
            start
            - 1
        ]
    ):
        start -= 1

    end = target_index

    while (
        end
        < len(
            mask
        )
        - 1
        and mask[
            end
            + 1
        ]
    ):
        end += 1

    return (
        int(
            start
        ),
        int(
            end
        ),
    )


def shape_diagnostics(
    profile,
    modality,
    window_config,
    tac2_landmark,
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
        "gdis"
    ].to_numpy(
        dtype=float
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

    minimum = float(
        np.min(
            y
        )
    )

    maximum = float(
        np.max(
            y
        )
    )

    dynamic_range = (
        maximum
        - minimum
    )

    global_index = int(
        np.argmax(
            y
        )
    )

    global_parameter = float(
        x[
            global_index
        ]
    )

    slopes = np.diff(
        y
    ) / np.diff(
        x
    )

    slope_epsilon = max(
        1e-12,
        0.01
        * (
            np.nanmedian(
                np.abs(
                    slopes
                )
            )
            if len(
                slopes
            )
            else 0.0
        ),
    )

    positive_fraction = float(
        np.mean(
            slopes
            > slope_epsilon
        )
    )

    negative_fraction = float(
        np.mean(
            slopes
            < -slope_epsilon
        )
    )

    near_zero_fraction = float(
        np.mean(
            np.abs(
                slopes
            )
            <= slope_epsilon
        )
    )

    total_variation = float(
        np.sum(
            np.abs(
                np.diff(
                    y
                )
            )
        )
    )

    net_change = float(
        y[
            -1
        ]
        - y[
            0
        ]
    )

    variation_to_net = (
        total_variation
        / abs(
            net_change
        )
        if abs(
            net_change
        )
        > 1e-12
        else np.nan
    )

    envelope_rows = []

    for level in NEAR_MAX_LEVELS:
        threshold = (
            minimum
            + level
            * dynamic_range
        )

        mask = (
            y
            >= threshold
        )

        (
            start,
            end,
        ) = contiguous_component_containing_index(
            mask,
            global_index,
        )

        start_parameter = float(
            x[
                start
            ]
        )

        end_parameter = float(
            x[
                end
            ]
        )

        width = (
            end_parameter
            - start_parameter
        )

        n_points = (
            end
            - start
            + 1
        )

        intersects_tac2_iqr = bool(
            (
                end_parameter
                >= tac2_landmark[
                    "q25"
                ]
            )
            and (
                start_parameter
                <= tac2_landmark[
                    "q75"
                ]
            )
        )

        envelope_rows.append(
            {
                "modality": modality,
                "window_config": window_config,
                "near_max_level": level,
                "threshold_value": threshold,
                "global_max_parameter": global_parameter,
                "global_max_value": maximum,
                "component_start_index": start,
                "component_end_index": end,
                "component_start_parameter": start_parameter,
                "component_end_parameter": end_parameter,
                "component_width_parameter": width,
                "component_n_points": int(
                    n_points
                ),
                "component_fraction_profile_points": (
                    n_points
                    / len(
                        y
                    )
                ),
                "component_intersects_TAC2_IQR": (
                    intersects_tac2_iqr
                ),
                "global_max_inside_TAC2_IQR": bool(
                    tac2_landmark[
                        "q25"
                    ]
                    <= global_parameter
                    <= tac2_landmark[
                        "q75"
                    ]
                ),
                "distance_global_max_to_TAC2_median": (
                    global_parameter
                    - tac2_landmark[
                        "median"
                    ]
                ),
            }
        )

    shape_row = {
        "modality": modality,
        "window_config": window_config,
        "n_profile_points": int(
            len(
                y
            )
        ),
        "parameter_min": float(
            x[
                0
            ]
        ),
        "parameter_max": float(
            x[
                -1
            ]
        ),
        "gdis_min": minimum,
        "gdis_max": maximum,
        "gdis_dynamic_range": dynamic_range,
        "global_max_index": global_index,
        "global_max_parameter": global_parameter,
        "fraction_positive_slopes": positive_fraction,
        "fraction_negative_slopes": negative_fraction,
        "fraction_near_zero_slopes": near_zero_fraction,
        "total_variation": total_variation,
        "net_change_first_to_last": net_change,
        "total_variation_to_abs_net_change": variation_to_net,
        "median_abs_slope": float(
            np.median(
                np.abs(
                    slopes
                )
            )
        ),
        "max_abs_slope": float(
            np.max(
                np.abs(
                    slopes
                )
            )
        ),
    }

    return (
        pd.DataFrame(
            envelope_rows
        ),
        shape_row,
    )


# =====================================================================
# TRANSITION-ENERGY FAMILIES
# =====================================================================

def deduplicate_energy_peaks(
    energy,
):
    """
    Transition energy is identical between lambda=0 and lambda=0.18.
    Keep one unique peak observation per modality/window/parameter.
    """
    required = [
        "modality",
        "window_config",
        "parameter",
        "prominence",
        "edge_proximal",
        "interior_peak",
        "dominant_state",
        "TAC1_fraction",
        "TAC2_fraction",
        "IRS_fraction",
    ]

    missing = [
        column
        for column in required
        if column not in energy.columns
    ]

    if missing:
        raise RuntimeError(
            "F15 transition-energy table missing: "
            + ", ".join(
                missing
            )
        )

    dedup = (
        energy.sort_values(
            [
                "modality",
                "window_config",
                "parameter",
                "profile_role",
            ]
        )
        .drop_duplicates(
            subset=[
                "modality",
                "window_config",
                "parameter",
            ],
            keep="first",
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    return dedup


def cluster_energy_families(
    energy_peaks,
    modality,
    family_tolerance,
    tac2_landmark,
):
    subset = energy_peaks.loc[
        energy_peaks[
            "modality"
        ]
        == modality
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
        cluster_id: i + 1
        for i, cluster_id in enumerate(
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

    family_rows = []

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

        interior_count = len(
            interior_configs
        )

        robust = bool(
            support_count
            >= ROBUST_MIN_WINDOW_CONFIGS
            and primary_support
        )

        robust_interior = bool(
            robust
            and interior_count
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
                ]
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
                    interior_count
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
                "edge_proximal_fraction": float(
                    group[
                        "edge_proximal"
                    ].mean()
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
        "PHASE F15b — SUSTAINED-GDIS SHAPE + TRANSITION-ENERGY FAMILY QC"
    )

    print(
        "F15b does not force an RNA GDIS local maximum."
    )

    print()

    print(
        "Sustained GDIS architecture and localized transition-energy "
        "topology are analyzed separately."
    )

    print()

    print(
        "No RNA/ATAC event pairing."
    )

    print(
        "No lead/lag."
    )

    print(
        "No CMIL."
    )

    require_inputs()

    F15B_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        profiles,
        energy,
        f15_topology,
        tac2_landmark,
    ) = load_inputs()

    # -----------------------------------------------------------------
    # Resolution.
    # -----------------------------------------------------------------

    section(
        "1. FREEZE RESOLUTION-DERIVED FAMILY TOLERANCE"
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
        f"Transition-energy family tolerance: "
        f"{family_tolerance:.6f}"
    )

    print()

    print(
        f"TAC-2 median: {tac2_landmark['median']:.6f}"
    )

    print(
        f"TAC-2 IQR: "
        f"[{tac2_landmark['q25']:.6f}, "
        f"{tac2_landmark['q75']:.6f}]"
    )

    # -----------------------------------------------------------------
    # Sustained profile shape.
    # -----------------------------------------------------------------

    section(
        "2. PRIMARY lambda=0 SUSTAINED-GDIS SHAPE / NEAR-MAX ENVELOPES"
    )

    envelope_tables = []
    shape_rows = []

    for modality in MODALITIES:
        for window_config in WINDOW_CONFIGS:
            profile = profiles.loc[
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
                    == PRIMARY_PROFILE_ROLE
                )
                & (
                    profiles[
                        "window_config"
                    ]
                    == window_config
                )
            ].copy()

            if profile.empty:
                raise RuntimeError(
                    f"Missing primary profile: "
                    f"{modality}/{window_config}"
                )

            envelopes, shape = shape_diagnostics(
                profile,
                modality,
                window_config,
                tac2_landmark,
            )

            envelope_tables.append(
                envelopes
            )

            shape_rows.append(
                shape
            )

    envelopes = pd.concat(
        envelope_tables,
        ignore_index=True,
    )

    shape_summary = pd.DataFrame(
        shape_rows
    )

    envelopes.to_csv(
        OUTPUT_ENVELOPES,
        sep="\t",
        index=False,
    )

    shape_summary.to_csv(
        OUTPUT_SHAPE,
        sep="\t",
        index=False,
    )

    subsection(
        "Near-maximum envelopes"
    )

    print_df(
        envelopes.set_index(
            [
                "modality",
                "window_config",
                "near_max_level",
            ]
        ),
        digits=6,
    )

    subsection(
        "Shape diagnostics"
    )

    print_df(
        shape_summary.set_index(
            [
                "modality",
                "window_config",
            ]
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Transition-energy families.
    # -----------------------------------------------------------------

    section(
        "3. DE-DUPLICATE MATERIAL TRANSITION-ENERGY PEAKS"
    )

    dedup_energy = deduplicate_energy_peaks(
        energy
    )

    print(
        f"Material transition-energy observations after "
        f"lambda de-duplication: {len(dedup_energy)}"
    )

    print_df(
        dedup_energy[
            [
                "modality",
                "window_config",
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
        ].set_index(
            [
                "modality",
                "window_config",
                "parameter",
            ]
        ),
        digits=6,
    )

    section(
        "4. CLUSTER TRANSITION-ENERGY PEAKS INTO WITHIN-MODALITY FAMILIES"
    )

    family_tables = []
    assignment_tables = []

    for modality in MODALITIES:
        families, assignments = cluster_energy_families(
            dedup_energy,
            modality,
            family_tolerance,
            tac2_landmark,
        )

        if not families.empty:
            family_tables.append(
                families
            )

        if not assignments.empty:
            assignment_tables.append(
                assignments
            )

    if family_tables:
        families = pd.concat(
            family_tables,
            ignore_index=True,
        )
    else:
        families = pd.DataFrame()

    if assignment_tables:
        assignments = pd.concat(
            assignment_tables,
            ignore_index=True,
        )
    else:
        assignments = pd.DataFrame()

    families.to_csv(
        OUTPUT_ENERGY_FAMILIES,
        sep="\t",
        index=False,
    )

    assignments.to_csv(
        OUTPUT_ENERGY_ASSIGNMENTS,
        sep="\t",
        index=False,
        compression="gzip",
    )

    family_display = [
        "modality",
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
                "family_id",
            ]
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Summary.
    # -----------------------------------------------------------------

    section(
        "5. F15b TOPOLOGY SUMMARY"
    )

    summary_rows = []

    for modality in MODALITIES:
        modality_families = families.loc[
            families[
                "modality"
            ]
            == modality
        ]

        robust = modality_families.loc[
            modality_families[
                "robust_family"
            ]
        ]

        robust_interior = modality_families.loc[
            modality_families[
                "robust_interior_family"
            ]
        ]

        three_window = modality_families.loc[
            modality_families[
                "three_window_family"
            ]
        ]

        inside_tac2 = robust_interior.loc[
            robust_interior[
                "inside_TAC2_IQR"
            ]
        ]

        modality_envelopes = envelopes.loc[
            envelopes[
                "modality"
            ]
            == modality
        ]

        e95 = modality_envelopes.loc[
            modality_envelopes[
                "near_max_level"
            ]
            == 0.95
        ]

        summary_rows.append(
            {
                "modality": modality,
                "n_transition_energy_families": int(
                    len(
                        modality_families
                    )
                ),
                "n_robust_energy_families": int(
                    len(
                        robust
                    )
                ),
                "n_robust_interior_energy_families": int(
                    len(
                        robust_interior
                    )
                ),
                "n_three_window_energy_families": int(
                    len(
                        three_window
                    )
                ),
                "n_robust_interior_families_inside_TAC2_IQR": int(
                    len(
                        inside_tac2
                    )
                ),
                "robust_interior_energy_family_centers": (
                    ",".join(
                        f"{value:.6f}"
                        for value in robust_interior[
                            "family_center_parameter"
                        ].sort_values()
                    )
                    if not robust_interior.empty
                    else ""
                ),
                "robust_interior_TAC2_region_centers": (
                    ",".join(
                        f"{value:.6f}"
                        for value in inside_tac2[
                            "family_center_parameter"
                        ].sort_values()
                    )
                    if not inside_tac2.empty
                    else ""
                ),
                "median_95pct_nearmax_width": float(
                    e95[
                        "component_width_parameter"
                    ].median()
                ),
                "median_95pct_nearmax_fraction_points": float(
                    e95[
                        "component_fraction_profile_points"
                    ].median()
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
    # Decision.
    # -----------------------------------------------------------------

    section(
        "6. PHASE F15b SUITABILITY CHECKS"
    )

    rna_has_energy = bool(
        topology_summary.loc[
            "RNA",
            "n_robust_interior_energy_families",
        ]
        >= 1
    )

    atac_has_energy = bool(
        topology_summary.loc[
            "ATAC",
            "n_robust_interior_energy_families",
        ]
        >= 1
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    "RNA contains >=1 robust-interior transition-energy family"
                ),
                "pass": rna_has_energy,
            },
            {
                "criterion": (
                    "ATAC contains >=1 robust-interior transition-energy family"
                ),
                "pass": atac_has_energy,
            },
            {
                "criterion": (
                    "Transition-energy family tolerance remains "
                    "resolution-derived"
                ),
                "pass": (
                    family_tolerance
                    >= FAMILY_TOLERANCE_MIN
                ),
            },
            {
                "criterion": (
                    "No TAC-2 landmark used to form transition-energy families"
                ),
                "pass": True,
            },
            {
                "criterion": (
                    "No cross-modal family pairing performed"
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
        "7. SAVE F15b MANIFEST"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F15b",
        "created_utc": utc_now_iso(),
        "purpose": (
            "Separate sustained-GDIS plateau/shoulder architecture from "
            "localized transition-energy families."
        ),
        "sustained_gdis": {
            "profile_role": PRIMARY_PROFILE_ROLE,
            "transition_weight": PRIMARY_TRANSITION_WEIGHT,
            "near_max_levels": NEAR_MAX_LEVELS,
            "local_peak_forced": False,
        },
        "transition_energy_families": {
            "lambda_profiles_deduplicated": True,
            "family_tolerance": family_tolerance,
            "primary_median_parameter_spacing": primary_median_spacing,
            "robust_min_window_configs": ROBUST_MIN_WINDOW_CONFIGS,
            "robust_requires_primary_window_config": True,
            "robust_interior_min_window_configs": (
                ROBUST_INTERIOR_MIN_WINDOW_CONFIGS
            ),
        },
        "independent_TAC2_landmark": {
            **tac2_landmark,
            "used_for_family_clustering": False,
            "used_for_family_annotation_only": True,
        },
        "summary": (
            topology_summary.reset_index().to_dict(
                orient="records"
            )
        ),
        "checks": checks.to_dict(
            orient="records"
        ),
        "outputs": {
            "sustained_envelopes": {
                "file": str(
                    OUTPUT_ENVELOPES
                ),
                "sha256": sha256_file(
                    OUTPUT_ENVELOPES
                ),
            },
            "sustained_shape_summary": {
                "file": str(
                    OUTPUT_SHAPE
                ),
                "sha256": sha256_file(
                    OUTPUT_SHAPE
                ),
            },
            "transition_energy_families": {
                "file": str(
                    OUTPUT_ENERGY_FAMILIES
                ),
                "sha256": sha256_file(
                    OUTPUT_ENERGY_FAMILIES
                ),
            },
            "transition_energy_peak_assignments": {
                "file": str(
                    OUTPUT_ENERGY_ASSIGNMENTS
                ),
                "sha256": sha256_file(
                    OUTPUT_ENERGY_ASSIGNMENTS
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
            "cross_modal_event_pairing_performed": False,
            "lead_lag_calculated": False,
            "cmil_calculated": False,
            "bootstrap_timing_calculated": False,
            "cross_modal_alignment_null_calculated": False,
            "priming_claim_made": False,
            "gdis_recomputed": False,
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
        OUTPUT_ENVELOPES,
        OUTPUT_SHAPE,
        OUTPUT_ENERGY_FAMILIES,
        OUTPUT_ENERGY_ASSIGNMENTS,
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
        "8. PHASE F15b DECISION"
    )

    if all_pass:
        print(
            "PHASE F15b VERDICT: GO — ROBUST INTERIOR LOCALIZED "
            "TRANSITION-ENERGY FAMILIES EXIST IN BOTH MODALITIES"
        )

        print()

        print(
            "The sustained-GDIS and localized transition-energy architectures "
            "must remain conceptually separate."
        )

        print()

        print(
            "A later phase may prospectively define an event-pairing rule "
            "using the already-frozen biological transition context."
        )

        print()

        print(
            "No cross-modal pair has been selected yet."
        )

    else:
        print(
            "PHASE F15b VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "At least one modality lacks a robust-interior localized "
            "transition-energy family."
        )

        print()

        print(
            "Do not calculate cross-modal timing."
        )

    print()

    print(
        "No cross-modal event was paired."
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

