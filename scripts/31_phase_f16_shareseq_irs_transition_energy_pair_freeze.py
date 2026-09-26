#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
31_phase_f16_shareseq_irs_transition_energy_pair_freeze.py
===========================================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F16
---------
Prospectively freeze ONE localized RNA/ATAC transition-energy event pair
using the modality-specific families already identified in F15b.

WHY TRANSITION ENERGY, NOT GDIS ARGMAX
--------------------------------------
F15/F15b established that:
    - RNA primary lambda_t=0 sustained GDIS is a broad near-maximum plateau;
    - ATAC primary sustained GDIS contains a localized interior peak family;
    - both modalities nevertheless contain reproducible localized
      transition-energy families.

Therefore:
    sustained-GDIS argmaxes are NOT paired.

Only transition-energy families may be paired.

PAIR-SELECTION RULE — FROZEN BEFORE BOOTSTRAP
---------------------------------------------
Selection is performed INDEPENDENTLY within RNA and ATAC.

Eligible family:
    1. robust_interior_family == True
    2. inside_TAC2_IQR == True

Hierarchical selection within each modality:

    Priority 1:
        Prefer THREE-WINDOW families
        (support in all 300/75, 400/100, and 500/125 designs).

    Priority 2:
        Among equally reproducible eligible families, choose the family whose
        center is closest to the independently frozen TAC-2 median.

    Tie breakers, in order:
        a. larger window_config_support_count
        b. larger median_prominence
        c. smaller family_center_parameter

CRITICAL ANTI-CHERRY-PICKING RULE
---------------------------------
RNA selection never uses ATAC family positions.
ATAC selection never uses RNA family positions.

The sign or magnitude of RNA-minus-ATAC timing is NOT used for selection.

Only after both modality-specific families are frozen independently does F16
calculate:
    observed CMIL-family separation =
        RNA family center - ATAC family center

Sign convention:
    > 0  : ATAC family earlier than RNA family
    = 0  : synchronous family centers
    < 0  : RNA family earlier than ATAC family

WINDOW-SPECIFIC CONSISTENCY
---------------------------
F16 maps each selected family back to its observed material peak in each
window configuration.

If both selected families occur in a given window configuration, F16 reports:

    RNA peak parameter - ATAC peak parameter

This is descriptive only.

F16 requires:
    - both independently selected families are robust-interior;
    - both are inside TAC-2 IQR;
    - both are three-window families;
    - each has exactly one assigned material peak per window configuration;
    - all three window-specific paired observations are available.

The sign need NOT be positive for F16 to pass.
Direction is reported, not imposed.

IMPORTANT
---------
No bootstrap.
No confidence interval.
No P(CMIL > 0).
No cross-modal permutation/alignment null.
No priming claim.

Those belong to later phases after the event pair is frozen.

INPUTS
------
data/GSE140203/representations_f14/
    f14_independent_state_landmarks.tsv

data/GSE140203/representations_f15b/
    f15b_transition_energy_families.tsv
    f15b_transition_energy_peak_assignments.tsv.gz
    f15b_manifest.json

OUTPUTS
-------
data/GSE140203/representations_f16/
    f16_candidate_families.tsv
    f16_frozen_transition_energy_pair.tsv
    f16_window_specific_pairing.tsv
    f16_manifest.json

RUN
---
    python 31_phase_f16_shareseq_irs_transition_energy_pair_freeze.py

DEPENDENCIES
------------
numpy
pandas
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE140203"

DATA_DIR = Path("data") / ACCESSION
F14_DIR = DATA_DIR / "representations_f14"
F15B_DIR = DATA_DIR / "representations_f15b"
F16_DIR = DATA_DIR / "representations_f16"

F14_LANDMARKS = F14_DIR / "f14_independent_state_landmarks.tsv"

F15B_FAMILIES = F15B_DIR / "f15b_transition_energy_families.tsv"
F15B_ASSIGNMENTS = F15B_DIR / "f15b_transition_energy_peak_assignments.tsv.gz"
F15B_MANIFEST = F15B_DIR / "f15b_manifest.json"

OUTPUT_CANDIDATES = F16_DIR / "f16_candidate_families.tsv"
OUTPUT_PAIR = F16_DIR / "f16_frozen_transition_energy_pair.tsv"
OUTPUT_WINDOWS = F16_DIR / "f16_window_specific_pairing.tsv"
OUTPUT_MANIFEST = F16_DIR / "f16_manifest.json"


# =====================================================================
# FROZEN DESIGN
# =====================================================================

MODALITIES = [
    "RNA",
    "ATAC",
]

TRANSITION_STATE = "TAC-2"

WINDOW_CONFIGS = [
    "sensitivity_w300_s75",
    "primary_w400_s100",
    "sensitivity_w500_s125",
]

PRIMARY_WINDOW_CONFIG = "primary_w400_s100"

REQUIRE_ROBUST_INTERIOR = True
REQUIRE_INSIDE_TAC2_IQR = True

PREFER_THREE_WINDOW = True

SIGN_CONVENTION = (
    "RNA_parameter_minus_ATAC_parameter; "
    "positive means ATAC occurs earlier"
)


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


def bool_series(series):
    """
    Robust conversion of TSV booleans.
    """
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.astype(
            bool
        )

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        [
            "true",
            "1",
            "yes",
            "y",
        ]
    )


# =====================================================================
# INPUTS
# =====================================================================

def require_inputs():
    required = [
        F14_LANDMARKS,
        F15B_FAMILIES,
        F15B_ASSIGNMENTS,
        F15B_MANIFEST,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required F14/F15b input(s):\n"
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

    families = pd.read_csv(
        F15B_FAMILIES,
        sep="\t",
        low_memory=False,
    )

    assignments = pd.read_csv(
        F15B_ASSIGNMENTS,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    required_family_cols = [
        "modality",
        "family_id",
        "family_center_parameter",
        "window_config_support_count",
        "robust_interior_family",
        "three_window_family",
        "median_prominence",
        "inside_TAC2_IQR",
        "absolute_distance_to_TAC2_median",
    ]

    missing_family = [
        column
        for column in required_family_cols
        if column not in families.columns
    ]

    if missing_family:
        raise RuntimeError(
            "F15b family table missing column(s): "
            + ", ".join(
                missing_family
            )
        )

    required_assignment_cols = [
        "modality",
        "family_id",
        "window_config",
        "parameter",
        "value",
        "prominence",
        "interior_peak",
    ]

    missing_assignment = [
        column
        for column in required_assignment_cols
        if column not in assignments.columns
    ]

    if missing_assignment:
        raise RuntimeError(
            "F15b assignment table missing column(s): "
            + ", ".join(
                missing_assignment
            )
        )

    # Normalize booleans after TSV round trip.
    for column in [
        "robust_interior_family",
        "three_window_family",
        "inside_TAC2_IQR",
    ]:
        families[
            column
        ] = bool_series(
            families[
                column
            ]
        )

    assignments[
        "interior_peak"
    ] = bool_series(
        assignments[
            "interior_peak"
        ]
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
        families,
        assignments,
        tac2_landmark,
    )


# =====================================================================
# INDEPENDENT FAMILY SELECTION
# =====================================================================

def eligible_candidates(
    families,
    modality,
):
    subset = families.loc[
        families[
            "modality"
        ].astype(str)
        == modality
    ].copy()

    if REQUIRE_ROBUST_INTERIOR:
        subset = subset.loc[
            subset[
                "robust_interior_family"
            ]
        ].copy()

    if REQUIRE_INSIDE_TAC2_IQR:
        subset = subset.loc[
            subset[
                "inside_TAC2_IQR"
            ]
        ].copy()

    if subset.empty:
        raise RuntimeError(
            f"{modality}: no eligible robust-interior TAC-2-region family."
        )

    return subset


def select_family_independently(
    candidates,
):
    """
    Hierarchical prespecified selection.

    Priority 1:
        three-window family, if any.

    Priority 2:
        smallest absolute distance to TAC-2 median.

    Tie breakers:
        more window support;
        greater median prominence;
        earlier center.
    """
    working = candidates.copy()

    if PREFER_THREE_WINDOW:
        three_window = working.loc[
            working[
                "three_window_family"
            ]
        ].copy()

        if not three_window.empty:
            working = three_window

    working = working.sort_values(
        by=[
            "absolute_distance_to_TAC2_median",
            "window_config_support_count",
            "median_prominence",
            "family_center_parameter",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    return working.iloc[
        0
    ].copy()


# =====================================================================
# ASSIGN SELECTED FAMILY PEAKS
# =====================================================================

def selected_family_assignments(
    assignments,
    modality,
    family_id,
):
    subset = assignments.loc[
        (
            assignments[
                "modality"
            ].astype(str)
            == modality
        )
        & (
            pd.to_numeric(
                assignments[
                    "family_id"
                ],
                errors="coerce",
            )
            == int(
                family_id
            )
        )
    ].copy()

    if subset.empty:
        raise RuntimeError(
            f"{modality} family {family_id}: no peak assignments found."
        )

    return subset


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F16 — FREEZE SHARE-seq IRS LOCALIZED TRANSITION-ENERGY EVENT PAIR"
    )

    print(
        "Pairing object:"
    )

    print(
        "  transition-energy families only"
    )

    print()

    print(
        "Sustained-GDIS argmaxes are NOT paired."
    )

    print()

    print(
        "Selection is independent within each modality."
    )

    print(
        "Three-window reproducibility is prioritized before TAC-2-median proximity."
    )

    print()

    print(
        "No bootstrap."
    )

    print(
        "No confidence interval."
    )

    print(
        "No permutation/alignment null."
    )

    print(
        "No priming claim."
    )

    require_inputs()

    F16_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        families,
        assignments,
        tac2_landmark,
    ) = load_inputs()

    # -----------------------------------------------------------------
    # Candidate families.
    # -----------------------------------------------------------------

    section(
        "1. IDENTIFY ELIGIBLE FAMILIES INDEPENDENTLY WITHIN EACH MODALITY"
    )

    candidate_tables = []
    selected_rows = {}

    for modality in MODALITIES:
        candidates = eligible_candidates(
            families,
            modality,
        )

        candidates = candidates.copy()

        candidates[
            "eligible_for_F16"
        ] = True

        candidate_tables.append(
            candidates
        )

        subsection(
            f"{modality} eligible families"
        )

        display_cols = [
            "family_id",
            "family_center_parameter",
            "window_config_support_count",
            "three_window_family",
            "median_prominence",
            "inside_TAC2_IQR",
            "absolute_distance_to_TAC2_median",
            "median_TAC1_fraction",
            "median_TAC2_fraction",
            "median_IRS_fraction",
        ]

        print_df(
            candidates[
                display_cols
            ].set_index(
                "family_id"
            ),
            digits=6,
        )

        selected = select_family_independently(
            candidates
        )

        selected_rows[
            modality
        ] = selected

        print()

        print(
            f"Selected {modality} family: "
            f"family_id={int(selected['family_id'])}, "
            f"center={float(selected['family_center_parameter']):.6f}, "
            f"three_window={bool(selected['three_window_family'])}, "
            f"|center-TAC2 median|="
            f"{float(selected['absolute_distance_to_TAC2_median']):.6f}"
        )

    candidates_all = pd.concat(
        candidate_tables,
        ignore_index=True,
    )

    candidates_all.to_csv(
        OUTPUT_CANDIDATES,
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Freeze independently selected pair.
    # -----------------------------------------------------------------

    section(
        "2. FREEZE THE TWO INDEPENDENTLY SELECTED FAMILY IDENTITIES"
    )

    rna_selected = selected_rows[
        "RNA"
    ]

    atac_selected = selected_rows[
        "ATAC"
    ]

    rna_center = float(
        rna_selected[
            "family_center_parameter"
        ]
    )

    atac_center = float(
        atac_selected[
            "family_center_parameter"
        ]
    )

    observed_separation = (
        rna_center
        - atac_center
    )

    if observed_separation > 0:
        direction = "ATAC earlier than RNA"
    elif observed_separation < 0:
        direction = "RNA earlier than ATAC"
    else:
        direction = "synchronous family centers"

    pair_table = pd.DataFrame(
        [
            {
                "pair_name": "TAC2_transition_energy_family_pair",
                "RNA_family_id": int(
                    rna_selected[
                        "family_id"
                    ]
                ),
                "RNA_family_center_parameter": rna_center,
                "RNA_three_window_family": bool(
                    rna_selected[
                        "three_window_family"
                    ]
                ),
                "RNA_distance_to_TAC2_median": float(
                    rna_selected[
                        "absolute_distance_to_TAC2_median"
                    ]
                ),
                "ATAC_family_id": int(
                    atac_selected[
                        "family_id"
                    ]
                ),
                "ATAC_family_center_parameter": atac_center,
                "ATAC_three_window_family": bool(
                    atac_selected[
                        "three_window_family"
                    ]
                ),
                "ATAC_distance_to_TAC2_median": float(
                    atac_selected[
                        "absolute_distance_to_TAC2_median"
                    ]
                ),
                "observed_RNA_minus_ATAC_family_center": observed_separation,
                "observed_direction": direction,
                "sign_convention": SIGN_CONVENTION,
                "bootstrap_performed": False,
                "confidence_interval_calculated": False,
                "priming_inference_made": False,
            }
        ]
    )

    print_df(
        pair_table.set_index(
            "pair_name"
        ),
        digits=6,
    )

    pair_table.to_csv(
        OUTPUT_PAIR,
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Map selected families to each window configuration.
    # -----------------------------------------------------------------

    section(
        "3. WINDOW-SPECIFIC OBSERVATIONS OF THE FROZEN FAMILY PAIR"
    )

    rna_assign = selected_family_assignments(
        assignments,
        "RNA",
        int(
            rna_selected[
                "family_id"
            ]
        ),
    )

    atac_assign = selected_family_assignments(
        assignments,
        "ATAC",
        int(
            atac_selected[
                "family_id"
            ]
        ),
    )

    window_rows = []

    for window_config in WINDOW_CONFIGS:
        rna_window = rna_assign.loc[
            rna_assign[
                "window_config"
            ].astype(str)
            == window_config
        ]

        atac_window = atac_assign.loc[
            atac_assign[
                "window_config"
            ].astype(str)
            == window_config
        ]

        rna_available = (
            len(
                rna_window
            )
            == 1
        )

        atac_available = (
            len(
                atac_window
            )
            == 1
        )

        if len(
            rna_window
        ) > 1:
            raise RuntimeError(
                f"RNA selected family has >1 assignment in {window_config}."
            )

        if len(
            atac_window
        ) > 1:
            raise RuntimeError(
                f"ATAC selected family has >1 assignment in {window_config}."
            )

        if rna_available:
            rna_parameter = float(
                rna_window.iloc[
                    0
                ][
                    "parameter"
                ]
            )

            rna_prominence = float(
                rna_window.iloc[
                    0
                ][
                    "prominence"
                ]
            )
        else:
            rna_parameter = np.nan
            rna_prominence = np.nan

        if atac_available:
            atac_parameter = float(
                atac_window.iloc[
                    0
                ][
                    "parameter"
                ]
            )

            atac_prominence = float(
                atac_window.iloc[
                    0
                ][
                    "prominence"
                ]
            )
        else:
            atac_parameter = np.nan
            atac_prominence = np.nan

        paired_available = bool(
            rna_available
            and atac_available
        )

        separation = (
            rna_parameter
            - atac_parameter
            if paired_available
            else np.nan
        )

        if paired_available:
            if separation > 0:
                window_direction = "ATAC earlier than RNA"
            elif separation < 0:
                window_direction = "RNA earlier than ATAC"
            else:
                window_direction = "synchronous"
        else:
            window_direction = "not paired"

        window_rows.append(
            {
                "window_config": window_config,
                "RNA_family_id": int(
                    rna_selected[
                        "family_id"
                    ]
                ),
                "RNA_peak_available": rna_available,
                "RNA_peak_parameter": rna_parameter,
                "RNA_peak_prominence": rna_prominence,
                "ATAC_family_id": int(
                    atac_selected[
                        "family_id"
                    ]
                ),
                "ATAC_peak_available": atac_available,
                "ATAC_peak_parameter": atac_parameter,
                "ATAC_peak_prominence": atac_prominence,
                "paired_observation_available": paired_available,
                "RNA_minus_ATAC_parameter": separation,
                "direction": window_direction,
            }
        )

    window_pairing = pd.DataFrame(
        window_rows
    )

    print_df(
        window_pairing.set_index(
            "window_config"
        ),
        digits=6,
    )

    window_pairing.to_csv(
        OUTPUT_WINDOWS,
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------------------
    # Suitability checks.
    # -----------------------------------------------------------------

    section(
        "4. PHASE F16 PAIR-FREEZE SAFEGUARDS"
    )

    all_three_paired = bool(
        window_pairing[
            "paired_observation_available"
        ].all()
    )

    if all_three_paired:
        separations = window_pairing[
            "RNA_minus_ATAC_parameter"
        ].to_numpy(
            dtype=float
        )

        all_positive = bool(
            np.all(
                separations
                > 0
            )
        )

        all_negative = bool(
            np.all(
                separations
                < 0
            )
        )

        sign_consistent = bool(
            all_positive
            or all_negative
            or np.allclose(
                separations,
                0.0,
                atol=1e-12,
                rtol=0.0,
            )
        )
    else:
        sign_consistent = False

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    "RNA selected family is robust-interior"
                ),
                "pass": bool(
                    rna_selected[
                        "robust_interior_family"
                    ]
                ),
            },
            {
                "criterion": (
                    "ATAC selected family is robust-interior"
                ),
                "pass": bool(
                    atac_selected[
                        "robust_interior_family"
                    ]
                ),
            },
            {
                "criterion": (
                    "RNA selected family is inside frozen TAC-2 IQR"
                ),
                "pass": bool(
                    rna_selected[
                        "inside_TAC2_IQR"
                    ]
                ),
            },
            {
                "criterion": (
                    "ATAC selected family is inside frozen TAC-2 IQR"
                ),
                "pass": bool(
                    atac_selected[
                        "inside_TAC2_IQR"
                    ]
                ),
            },
            {
                "criterion": (
                    "RNA selected family has three-window support"
                ),
                "pass": bool(
                    rna_selected[
                        "three_window_family"
                    ]
                ),
            },
            {
                "criterion": (
                    "ATAC selected family has three-window support"
                ),
                "pass": bool(
                    atac_selected[
                        "three_window_family"
                    ]
                ),
            },
            {
                "criterion": (
                    "Frozen family pair has one paired observation "
                    "in all 3 window configurations"
                ),
                "pass": all_three_paired,
            },
            {
                "criterion": (
                    "Family selection performed independently by modality"
                ),
                "pass": True,
            },
            {
                "criterion": (
                    "Observed direction was not used for pair selection"
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

    print()

    print(
        f"Window-specific direction sign consistent: {sign_consistent}"
    )

    if all_three_paired:
        print(
            "Window-specific observed separations:"
        )

        for row in window_pairing.itertuples(
            index=False
        ):
            print(
                f"  {row.window_config}: "
                f"{row.RNA_minus_ATAC_parameter:.6f} "
                f"({row.direction})"
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
        "5. SAVE F16 MANIFEST"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F16",
        "created_utc": utc_now_iso(),
        "pairing_object": "localized transition-energy family",
        "sustained_gdis_argmax_paired": False,
        "selection_rule": {
            "performed_independently_by_modality": True,
            "eligibility": {
                "robust_interior_family": True,
                "inside_TAC2_IQR": True,
            },
            "priority_1": (
                "prefer three-window family if available"
            ),
            "priority_2": (
                "smallest absolute distance to independently frozen "
                "TAC-2 median"
            ),
            "tie_breakers": [
                "larger window_config_support_count",
                "larger median_prominence",
                "smaller family_center_parameter",
            ],
            "uses_other_modality_position": False,
            "uses_observed_lead_sign": False,
            "uses_observed_lead_magnitude": False,
        },
        "TAC2_landmark": tac2_landmark,
        "selected_pair": (
            pair_table.iloc[
                0
            ].to_dict()
        ),
        "window_specific_pairing": (
            window_pairing.to_dict(
                orient="records"
            )
        ),
        "window_specific_direction_sign_consistent": (
            sign_consistent
        ),
        "checks": checks.to_dict(
            orient="records"
        ),
        "outputs": {
            "candidate_families": {
                "file": str(
                    OUTPUT_CANDIDATES
                ),
                "sha256": sha256_file(
                    OUTPUT_CANDIDATES
                ),
            },
            "frozen_pair": {
                "file": str(
                    OUTPUT_PAIR
                ),
                "sha256": sha256_file(
                    OUTPUT_PAIR
                ),
            },
            "window_specific_pairing": {
                "file": str(
                    OUTPUT_WINDOWS
                ),
                "sha256": sha256_file(
                    OUTPUT_WINDOWS
                ),
            },
        },
        "guardrails": {
            "bootstrap_performed": False,
            "confidence_interval_calculated": False,
            "p_value_calculated": False,
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
        OUTPUT_CANDIDATES,
        OUTPUT_PAIR,
        OUTPUT_WINDOWS,
        OUTPUT_MANIFEST,
    ]:
        print(
            path
        )

    # -----------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------

    section(
        "6. PHASE F16 DECISION"
    )

    if all_pass:
        print(
            "PHASE F16 VERDICT: GO — LOCALIZED RNA/ATAC TRANSITION-ENERGY "
            "FAMILY PAIR FROZEN BEFORE BOOTSTRAP"
        )

        print()

        print(
            f"Observed family-center separation "
            f"(RNA - ATAC): {observed_separation:.6f}"
        )

        print(
            f"Observed direction: {direction}"
        )

        print()

        print(
            "This is descriptive only."
        )

        print()

        print(
            "Next phase may bootstrap the already-frozen event-pair timing "
            "without changing family-selection rules."
        )

    else:
        print(
            "PHASE F16 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "Do not bootstrap cross-modal timing until failed pair-freeze "
            "criteria are understood."
        )

    print()

    print(
        "No bootstrap was performed."
    )

    print(
        "No confidence interval was calculated."
    )

    print(
        "No alignment null was calculated."
    )

    print(
        "No priming claim was made."
    )

    line("=")


if __name__ == "__main__":
    main()

