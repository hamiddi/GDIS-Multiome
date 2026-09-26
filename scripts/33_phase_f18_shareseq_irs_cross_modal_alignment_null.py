#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
33_phase_f18_shareseq_irs_cross_modal_alignment_null.py
=======================================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F18
---------
Test whether RNA and ATAC TRANSITION-ENERGY profiles are more strongly aligned
within the independently frozen TAC-2 transition region than expected under a
dependence-preserving null.

WHY A DEPENDENCE-PRESERVING NULL
--------------------------------
The primary GDIS profiles use 400-cell windows with 100-cell steps (75% overlap),
so adjacent window-level observations are strongly dependent. Pointwise/i.i.d.
permutation would destroy this structure.

PRIMARY NULL
------------
Exact admissible circular shifts of the COMPLETE ATAC transition-energy profile
relative to the fixed RNA profile.

Circular shifts preserve the complete ATAC value sequence and its circular
lag/autocorrelation structure while disrupting its alignment to RNA pseudotime.
Offsets too close to zero are excluded using the overlap-derived dependence span:

    ceil(400 / 100) = 4 windows

An admissible shift must have circular distance from zero >= 4 windows.

SECONDARY SENSITIVITY NULL
--------------------------
5,000 random permutations of contiguous 4-window ATAC blocks. This preserves
short-range within-block dependence while disrupting broader alignment.

EVENT REGION
------------
The independently frozen TAC-2 IQR from F14/F13. No GDIS feature was used to
define this region.

ALIGNMENT METRICS
-----------------
Within the fixed TAC-2 IQR:

1. Absolute transition-energy median-location separation
       |q50_RNA - q50_ATAC|
   Smaller = stronger alignment.

2. Spearman correlation of transition-energy profiles
   Larger = stronger alignment.

3. Jensen-Shannon divergence of normalized transition-energy distributions
   Smaller = stronger alignment.

PRIMARY EVIDENCE CLASSIFICATION
-------------------------------
Using circular-shift empirical p-values (alpha=0.05):

STRONG:
    3/3 metrics significant

PARTIAL:
    2/3 metrics significant

LIMITED_OR_NONE:
    0-1/3 metrics significant

F16 event families and F17 bootstrap results remain frozen. F18 does NOT
reselect event families, recompute GDIS, or make a chromatin-priming claim.

OUTPUTS
-------
data/GSE140203/representations_f18/
    f18_observed_alignment.tsv
    f18_circular_shift_null.tsv.gz
    f18_block_permutation_null.tsv.gz
    f18_alignment_summary.tsv
    f18_manifest.json

RUN
---
    python 33_phase_f18_shareseq_irs_cross_modal_alignment_null.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


# =====================================================================
# PATHS
# =====================================================================

ACCESSION = "GSE140203"

DATA_DIR = Path("data") / ACCESSION
F14_DIR = DATA_DIR / "representations_f14"
F16_DIR = DATA_DIR / "representations_f16"
F17_DIR = DATA_DIR / "representations_f17"
F18_DIR = DATA_DIR / "representations_f18"

F14_PROFILES = F14_DIR / "f14_gdis_profiles.tsv.gz"
F14_LANDMARKS = F14_DIR / "f14_independent_state_landmarks.tsv"
F14_MANIFEST = F14_DIR / "f14_manifest.json"

F16_PAIR = F16_DIR / "f16_frozen_transition_energy_pair.tsv"
F16_MANIFEST = F16_DIR / "f16_manifest.json"

F17_SUMMARY = F17_DIR / "f17_bootstrap_summary.tsv"
F17_MANIFEST = F17_DIR / "f17_manifest.json"

OUTPUT_OBSERVED = F18_DIR / "f18_observed_alignment.tsv"
OUTPUT_CIRCULAR = F18_DIR / "f18_circular_shift_null.tsv.gz"
OUTPUT_BLOCK = F18_DIR / "f18_block_permutation_null.tsv.gz"
OUTPUT_SUMMARY = F18_DIR / "f18_alignment_summary.tsv"
OUTPUT_MANIFEST = F18_DIR / "f18_manifest.json"


# =====================================================================
# FROZEN DESIGN
# =====================================================================

PRIMARY_PROFILE_ROLE = "primary_multiome"
PRIMARY_WINDOW_CONFIG = "primary_w400_s100"

WINDOW_SIZE = 400
STEP_SIZE = 100

DEPENDENCE_SPAN_WINDOWS = int(math.ceil(WINDOW_SIZE / STEP_SIZE))
BLOCK_LENGTH = DEPENDENCE_SPAN_WINDOWS

N_BLOCK_PERMUTATIONS = 5_000
RANDOM_SEED = 785

TRANSITION_STATE = "TAC-2"
ALPHA = 0.05


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
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan

    rho, _ = spearmanr(x, y)
    return float(rho)


# =====================================================================
# INPUTS
# =====================================================================

def require_inputs():
    required = [
        F14_PROFILES,
        F14_LANDMARKS,
        F14_MANIFEST,
        F16_PAIR,
        F16_MANIFEST,
        F17_SUMMARY,
        F17_MANIFEST,
    ]

    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing required F14/F16/F17 input(s):\n"
            + "\n".join(f"  {path}" for path in missing)
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

    pair = pd.read_csv(F16_PAIR, sep="\t", low_memory=False)
    bootstrap = pd.read_csv(F17_SUMMARY, sep="\t", low_memory=False)

    if len(pair) != 1:
        raise RuntimeError("F16 frozen-pair table must contain exactly one row.")
    if len(bootstrap) != 1:
        raise RuntimeError("F17 bootstrap summary must contain exactly one row.")

    tac2 = landmarks.loc[
        landmarks["state"].astype(str) == TRANSITION_STATE
    ]

    if len(tac2) != 1:
        raise RuntimeError("Could not identify unique TAC-2 landmark.")

    tac2 = tac2.iloc[0]
    tac2_landmark = {
        "median": float(tac2["median"]),
        "q25": float(tac2["q25"]),
        "q75": float(tac2["q75"]),
    }

    primary = profiles.loc[
        (profiles["profile_role"].astype(str) == PRIMARY_PROFILE_ROLE)
        & (profiles["window_config"].astype(str) == PRIMARY_WINDOW_CONFIG)
    ].copy()

    rna = primary.loc[
        primary["modality"].astype(str) == "RNA"
    ].sort_values("parameter").reset_index(drop=True)

    atac = primary.loc[
        primary["modality"].astype(str) == "ATAC"
    ].sort_values("parameter").reset_index(drop=True)

    if rna.empty or atac.empty:
        raise RuntimeError("Missing primary RNA/ATAC F14 profile.")

    rna_x = rna["parameter"].to_numpy(dtype=float)
    atac_x = atac["parameter"].to_numpy(dtype=float)

    if not np.allclose(rna_x, atac_x, atol=1e-14, rtol=0.0):
        raise RuntimeError("Primary RNA and ATAC common-window parameters differ.")

    rna_energy = rna["transition_energy"].to_numpy(dtype=float)
    atac_energy = atac["transition_energy"].to_numpy(dtype=float)

    if not np.isfinite(rna_energy).all() or not np.isfinite(atac_energy).all():
        raise RuntimeError("Transition-energy profile contains NaN/Inf.")
    if np.any(rna_energy < 0) or np.any(atac_energy < 0):
        raise RuntimeError("Transition energy must be nonnegative.")

    bootstrap_support = parse_bool(
        bootstrap.iloc[0]["bootstrap_ATAC_earlier_support_pass"]
    )

    return {
        "parameter": rna_x,
        "rna_energy": rna_energy,
        "atac_energy": atac_energy,
        "tac2": tac2_landmark,
        "pair": pair.iloc[0].to_dict(),
        "bootstrap": bootstrap.iloc[0].to_dict(),
        "bootstrap_support": bootstrap_support,
    }


# =====================================================================
# ALIGNMENT METRICS
# =====================================================================

def normalized_distribution(values):
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 0.0, None)

    total = float(np.sum(values))
    if total <= 0:
        raise RuntimeError("Cannot normalize zero transition-energy mass.")

    return values / total


def weighted_quantile_location(x, weights, quantile=0.5):
    """Linear interpolation on normalized cumulative transition-energy mass."""
    x = np.asarray(x, dtype=float)
    p = normalized_distribution(weights)
    cumulative = np.cumsum(p)
    return float(np.interp(quantile, cumulative, x))


def alignment_metrics(x_region, rna_region, atac_region):
    rna_dist = normalized_distribution(rna_region)
    atac_dist = normalized_distribution(atac_region)

    rna_q50 = weighted_quantile_location(x_region, rna_region, 0.5)
    atac_q50 = weighted_quantile_location(x_region, atac_region, 0.5)

    signed_q50 = rna_q50 - atac_q50
    absolute_q50 = abs(signed_q50)

    rho = safe_spearman(rna_region, atac_region)

    # scipy.spatial.distance.jensenshannon returns sqrt(JS divergence).
    js_distance = float(jensenshannon(rna_dist, atac_dist, base=2.0))
    js_divergence = js_distance ** 2

    return {
        "RNA_energy_q50_parameter": rna_q50,
        "ATAC_energy_q50_parameter": atac_q50,
        "signed_q50_RNA_minus_ATAC": signed_q50,
        "absolute_q50_separation": absolute_q50,
        "spearman_rho": rho,
        "JS_divergence_base2": js_divergence,
    }


# =====================================================================
# NULL GENERATORS
# =====================================================================

def admissible_circular_offsets(n_points):
    offsets = []
    for offset in range(1, n_points):
        circular_distance = min(offset, n_points - offset)
        if circular_distance >= DEPENDENCE_SPAN_WINDOWS:
            offsets.append(offset)
    return offsets


def variable_blocks(values, block_length):
    return [
        values[start:min(start + block_length, len(values))]
        for start in range(0, len(values), block_length)
    ]


def block_permute(values, rng):
    """
    Randomly permute contiguous nonoverlapping blocks.

    The final block may be shorter than BLOCK_LENGTH. Every original value is
    retained exactly once.
    """
    values = np.asarray(values, dtype=float)
    blocks = variable_blocks(values, BLOCK_LENGTH)
    order = rng.permutation(len(blocks))
    permuted = np.concatenate([blocks[i] for i in order])

    if len(permuted) != len(values):
        raise RuntimeError("Block permutation changed profile length.")

    return permuted


# =====================================================================
# EMPIRICAL P-VALUES
# =====================================================================

def empirical_p_lower(observed, null_values):
    null_values = np.asarray(null_values, dtype=float)
    return float(
        (1 + np.sum(null_values <= observed))
        / (1 + len(null_values))
    )


def empirical_p_upper(observed, null_values):
    null_values = np.asarray(null_values, dtype=float)
    return float(
        (1 + np.sum(null_values >= observed))
        / (1 + len(null_values))
    )


def evidence_label(n_significant):
    if n_significant == 3:
        return "STRONG"
    if n_significant == 2:
        return "PARTIAL"
    return "LIMITED_OR_NONE"


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F18 — SHARE-seq IRS CROSS-MODAL TRANSITION-ENERGY ALIGNMENT NULL"
    )

    print("Primary null:")
    print("  exact dependence-preserving circular shifts of ATAC profile")
    print()
    print("Sensitivity null:")
    print(
        f"  {N_BLOCK_PERMUTATIONS:,} random permutations of "
        f"{BLOCK_LENGTH}-window ATAC blocks"
    )
    print()
    print(
        f"Dependence span derived from window overlap: "
        f"ceil({WINDOW_SIZE}/{STEP_SIZE}) = {DEPENDENCE_SPAN_WINDOWS} windows"
    )
    print()
    print("Event region:")
    print("  frozen TAC-2 IQR")
    print()
    print("No family reselection.")
    print("No GDIS recomputation.")
    print("No priming claim.")

    require_inputs()
    F18_DIR.mkdir(parents=True, exist_ok=True)

    data = load_inputs()

    x = data["parameter"]
    rna_energy = data["rna_energy"]
    atac_energy = data["atac_energy"]
    tac2 = data["tac2"]

    # -----------------------------------------------------------------
    # Freeze event region.
    # -----------------------------------------------------------------

    section("1. FREEZE TAC-2 EVENT-REGION WINDOW SUPPORT")

    region_mask = (x >= tac2["q25"]) & (x <= tac2["q75"])
    x_region = x[region_mask]
    rna_region = rna_energy[region_mask]
    atac_region = atac_energy[region_mask]

    if len(x_region) < 8:
        raise RuntimeError(
            f"Too few primary windows ({len(x_region)}) inside frozen TAC-2 IQR."
        )

    print(f"Full primary profile windows: {len(x)}")
    print(f"TAC-2 IQR: [{tac2['q25']:.6f}, {tac2['q75']:.6f}]")
    print(f"Windows inside TAC-2 IQR: {len(x_region)}")
    print(
        f"Region parameter support: "
        f"{x_region.min():.6f} - {x_region.max():.6f}"
    )

    # -----------------------------------------------------------------
    # Observed alignment.
    # -----------------------------------------------------------------

    section("2. OBSERVED CROSS-MODAL ALIGNMENT METRICS")

    observed = alignment_metrics(x_region, rna_region, atac_region)

    observed_table = pd.DataFrame(
        [
            {
                "region": "TAC2_IQR",
                "n_region_windows": len(x_region),
                "region_parameter_min": float(x_region.min()),
                "region_parameter_max": float(x_region.max()),
                **observed,
                "F16_family_center_RNA_minus_ATAC": float(
                    data["pair"]["observed_RNA_minus_ATAC_family_center"]
                ),
                "F17_observed_primary_CMIL": float(
                    data["bootstrap"]["observed_primary_CMIL_RNA_minus_ATAC"]
                ),
                "F17_bootstrap_support_pass": bool(data["bootstrap_support"]),
            }
        ]
    )

    print_df(observed_table.T, digits=6)
    observed_table.to_csv(OUTPUT_OBSERVED, sep="\t", index=False)

    # -----------------------------------------------------------------
    # Circular-shift null.
    # -----------------------------------------------------------------

    section("3. EXACT CIRCULAR-SHIFT NULL")

    offsets = admissible_circular_offsets(len(x))

    print(f"Admissible offsets: {len(offsets)}")
    print(
        f"Excluded near-zero circular shifts: "
        f"distance < {DEPENDENCE_SPAN_WINDOWS} windows"
    )

    circular_rows = []

    for offset in offsets:
        shifted_atac = np.roll(atac_energy, offset)
        shifted_region = shifted_atac[region_mask]

        metrics = alignment_metrics(
            x_region,
            rna_region,
            shifted_region,
        )

        circular_rows.append(
            {
                "offset_windows": int(offset),
                "circular_distance_from_zero": int(
                    min(offset, len(x) - offset)
                ),
                **metrics,
            }
        )

    circular = pd.DataFrame(circular_rows)
    circular.to_csv(
        OUTPUT_CIRCULAR,
        sep="\t",
        index=False,
        compression="gzip",
    )

    p_abssep = empirical_p_lower(
        observed["absolute_q50_separation"],
        circular["absolute_q50_separation"],
    )

    p_rho = empirical_p_upper(
        observed["spearman_rho"],
        circular["spearman_rho"],
    )

    p_js = empirical_p_lower(
        observed["JS_divergence_base2"],
        circular["JS_divergence_base2"],
    )

    circular_summary = pd.DataFrame(
        [
            {
                "metric": "absolute_q50_separation",
                "observed": observed["absolute_q50_separation"],
                "null_median": float(
                    circular["absolute_q50_separation"].median()
                ),
                "null_q025": float(
                    circular["absolute_q50_separation"].quantile(0.025)
                ),
                "null_q975": float(
                    circular["absolute_q50_separation"].quantile(0.975)
                ),
                "empirical_p": p_abssep,
                "extreme_direction": "smaller",
                "significant_0.05": p_abssep <= ALPHA,
            },
            {
                "metric": "spearman_rho",
                "observed": observed["spearman_rho"],
                "null_median": float(circular["spearman_rho"].median()),
                "null_q025": float(circular["spearman_rho"].quantile(0.025)),
                "null_q975": float(circular["spearman_rho"].quantile(0.975)),
                "empirical_p": p_rho,
                "extreme_direction": "larger",
                "significant_0.05": p_rho <= ALPHA,
            },
            {
                "metric": "JS_divergence_base2",
                "observed": observed["JS_divergence_base2"],
                "null_median": float(
                    circular["JS_divergence_base2"].median()
                ),
                "null_q025": float(
                    circular["JS_divergence_base2"].quantile(0.025)
                ),
                "null_q975": float(
                    circular["JS_divergence_base2"].quantile(0.975)
                ),
                "empirical_p": p_js,
                "extreme_direction": "smaller",
                "significant_0.05": p_js <= ALPHA,
            },
        ]
    )

    n_circular_significant = int(
        circular_summary["significant_0.05"].sum()
    )
    circular_evidence = evidence_label(n_circular_significant)

    print_df(circular_summary.set_index("metric"), digits=6)
    print()
    print(f"Primary circular-shift evidence class: {circular_evidence}")

    # -----------------------------------------------------------------
    # Block-permutation sensitivity.
    # -----------------------------------------------------------------

    section("4. FOUR-WINDOW BLOCK-PERMUTATION SENSITIVITY NULL")

    rng = np.random.default_rng(RANDOM_SEED)
    block_rows = []

    for permutation in range(1, N_BLOCK_PERMUTATIONS + 1):
        permuted_atac = block_permute(atac_energy, rng)
        permuted_region = permuted_atac[region_mask]

        metrics = alignment_metrics(
            x_region,
            rna_region,
            permuted_region,
        )

        block_rows.append(
            {
                "permutation": permutation,
                **metrics,
            }
        )

    block = pd.DataFrame(block_rows)
    block.to_csv(
        OUTPUT_BLOCK,
        sep="\t",
        index=False,
        compression="gzip",
    )

    block_p_abssep = empirical_p_lower(
        observed["absolute_q50_separation"],
        block["absolute_q50_separation"],
    )

    block_p_rho = empirical_p_upper(
        observed["spearman_rho"],
        block["spearman_rho"],
    )

    block_p_js = empirical_p_lower(
        observed["JS_divergence_base2"],
        block["JS_divergence_base2"],
    )

    block_summary = pd.DataFrame(
        [
            {
                "metric": "absolute_q50_separation",
                "observed": observed["absolute_q50_separation"],
                "null_median": float(block["absolute_q50_separation"].median()),
                "null_q025": float(
                    block["absolute_q50_separation"].quantile(0.025)
                ),
                "null_q975": float(
                    block["absolute_q50_separation"].quantile(0.975)
                ),
                "empirical_p": block_p_abssep,
                "extreme_direction": "smaller",
                "significant_0.05": block_p_abssep <= ALPHA,
            },
            {
                "metric": "spearman_rho",
                "observed": observed["spearman_rho"],
                "null_median": float(block["spearman_rho"].median()),
                "null_q025": float(block["spearman_rho"].quantile(0.025)),
                "null_q975": float(block["spearman_rho"].quantile(0.975)),
                "empirical_p": block_p_rho,
                "extreme_direction": "larger",
                "significant_0.05": block_p_rho <= ALPHA,
            },
            {
                "metric": "JS_divergence_base2",
                "observed": observed["JS_divergence_base2"],
                "null_median": float(block["JS_divergence_base2"].median()),
                "null_q025": float(
                    block["JS_divergence_base2"].quantile(0.025)
                ),
                "null_q975": float(
                    block["JS_divergence_base2"].quantile(0.975)
                ),
                "empirical_p": block_p_js,
                "extreme_direction": "smaller",
                "significant_0.05": block_p_js <= ALPHA,
            },
        ]
    )

    n_block_significant = int(block_summary["significant_0.05"].sum())
    block_evidence = evidence_label(n_block_significant)

    print_df(block_summary.set_index("metric"), digits=6)

    # -----------------------------------------------------------------
    # Final summary.
    # -----------------------------------------------------------------

    section("5. PHASE F18 ALIGNMENT SUMMARY")

    summary = pd.DataFrame(
        [
            {
                "region": "TAC2_IQR",
                "region_q25": tac2["q25"],
                "region_q75": tac2["q75"],
                "n_region_windows": len(x_region),
                "observed_RNA_energy_q50": observed[
                    "RNA_energy_q50_parameter"
                ],
                "observed_ATAC_energy_q50": observed[
                    "ATAC_energy_q50_parameter"
                ],
                "observed_signed_q50_RNA_minus_ATAC": observed[
                    "signed_q50_RNA_minus_ATAC"
                ],
                "observed_absolute_q50_separation": observed[
                    "absolute_q50_separation"
                ],
                "observed_spearman_rho": observed["spearman_rho"],
                "observed_JS_divergence_base2": observed[
                    "JS_divergence_base2"
                ],
                "circular_n_admissible_shifts": len(circular),
                "circular_p_abs_q50_separation": p_abssep,
                "circular_p_spearman_rho": p_rho,
                "circular_p_JS_divergence": p_js,
                "circular_n_significant_metrics": n_circular_significant,
                "circular_alignment_evidence_class": circular_evidence,
                "block_n_permutations": N_BLOCK_PERMUTATIONS,
                "block_p_abs_q50_separation": block_p_abssep,
                "block_p_spearman_rho": block_p_rho,
                "block_p_JS_divergence": block_p_js,
                "block_n_significant_metrics": n_block_significant,
                "block_alignment_evidence_class": block_evidence,
                "F17_bootstrap_timing_support_pass": bool(
                    data["bootstrap_support"]
                ),
            }
        ]
    )

    print_df(summary.T, digits=6)
    summary.to_csv(OUTPUT_SUMMARY, sep="\t", index=False)

    checks = pd.DataFrame(
        [
            {
                "criterion": "F17 frozen timing bootstrap support remains PASS",
                "pass": bool(data["bootstrap_support"]),
            },
            {
                "criterion": "Circular-shift null preserves full ATAC profile values",
                "pass": True,
            },
            {
                "criterion": (
                    f"Circular shifts exclude dependence span < "
                    f"{DEPENDENCE_SPAN_WINDOWS} windows"
                ),
                "pass": len(circular) > 0,
            },
            {
                "criterion": "Frozen TAC-2 IQR used as event region",
                "pass": True,
            },
            {
                "criterion": "No event-family reselection performed",
                "pass": True,
            },
        ]
    )

    checks["status"] = np.where(checks["pass"], "PASS", "REVIEW")

    subsection("Technical null checks")
    print_df(checks.set_index("criterion"), digits=0)

    # -----------------------------------------------------------------
    # Manifest.
    # -----------------------------------------------------------------

    section("6. SAVE F18 MANIFEST")

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F18",
        "created_utc": utc_now_iso(),
        "purpose": (
            "Dependence-preserving null test of cross-modal transition-energy "
            "profile alignment inside the independently frozen TAC-2 IQR."
        ),
        "primary_profile": {
            "profile_role": PRIMARY_PROFILE_ROLE,
            "window_config": PRIMARY_WINDOW_CONFIG,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "n_full_profile_windows": len(x),
        },
        "event_region": {
            "definition": "frozen TAC-2 IQR",
            "q25": tac2["q25"],
            "q75": tac2["q75"],
            "n_windows": len(x_region),
            "used_to_select_F16_pair": False,
        },
        "metrics": {
            "q50": (
                "absolute separation of normalized transition-energy "
                "weighted median locations"
            ),
            "spearman": "rank correlation of transition-energy profiles",
            "JS": (
                "base-2 Jensen-Shannon divergence of normalized "
                "transition-energy distributions"
            ),
        },
        "primary_null": {
            "method": "exact circular shift of full ATAC profile",
            "dependence_span_windows": DEPENDENCE_SPAN_WINDOWS,
            "admissible_offsets": len(circular),
            "small_shifts_excluded": True,
            "p_value_correction": "(1 + extreme)/(1 + null_count)",
        },
        "sensitivity_null": {
            "method": "contiguous block permutation of ATAC profile",
            "block_length_windows": BLOCK_LENGTH,
            "n_permutations": N_BLOCK_PERMUTATIONS,
            "random_seed": RANDOM_SEED,
        },
        "evidence_classification": {
            "alpha": ALPHA,
            "STRONG": "3 of 3 circular-shift metrics significant",
            "PARTIAL": "2 of 3 circular-shift metrics significant",
            "LIMITED_OR_NONE": "0 or 1 of 3 circular-shift metrics significant",
        },
        "observed": observed,
        "circular_shift_summary": circular_summary.to_dict(orient="records"),
        "block_permutation_summary": block_summary.to_dict(orient="records"),
        "final_summary": summary.iloc[0].to_dict(),
        "checks": checks.to_dict(orient="records"),
        "outputs": {
            "observed_alignment": {
                "file": str(OUTPUT_OBSERVED),
                "sha256": sha256_file(OUTPUT_OBSERVED),
            },
            "circular_shift_null": {
                "file": str(OUTPUT_CIRCULAR),
                "sha256": sha256_file(OUTPUT_CIRCULAR),
            },
            "block_permutation_null": {
                "file": str(OUTPUT_BLOCK),
                "sha256": sha256_file(OUTPUT_BLOCK),
            },
            "summary": {
                "file": str(OUTPUT_SUMMARY),
                "sha256": sha256_file(OUTPUT_SUMMARY),
            },
        },
        "guardrails": {
            "F16_pair_changed": False,
            "F17_bootstrap_changed": False,
            "gdis_recomputed": False,
            "trajectory_changed": False,
            "representations_changed": False,
            "priming_claim_made": False,
        },
    }

    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            default=(
                lambda obj:
                bool(obj)
                if isinstance(obj, np.bool_)
                else int(obj)
                if isinstance(obj, np.integer)
                else float(obj)
                if isinstance(obj, np.floating)
                else str(obj)
            ),
        )
        handle.write("\n")

    for path in [
        OUTPUT_OBSERVED,
        OUTPUT_CIRCULAR,
        OUTPUT_BLOCK,
        OUTPUT_SUMMARY,
        OUTPUT_MANIFEST,
    ]:
        print(path)

    # -----------------------------------------------------------------
    # Verdict.
    # -----------------------------------------------------------------

    section("7. PHASE F18 DECISION")

    if circular_evidence == "STRONG":
        print(
            "PHASE F18 VERDICT: STRONG ALIGNMENT EVIDENCE — ALL THREE "
            "DEPENDENCE-PRESERVING CIRCULAR-SHIFT METRICS PASS"
        )
        print()
        print(
            "The frozen F17 ATAC-earlier timing relationship is supported "
            "within broader cross-modal transition-energy alignment that is "
            "stronger than the primary dependence-preserving null."
        )

    elif circular_evidence == "PARTIAL":
        print(
            "PHASE F18 VERDICT: PARTIAL ALIGNMENT EVIDENCE — TWO OF THREE "
            "PRIMARY CIRCULAR-SHIFT METRICS PASS"
        )
        print()
        print(
            "The bootstrap timing lead is supported, but overall profile "
            "alignment evidence is mixed."
        )

    else:
        print(
            "PHASE F18 VERDICT: LIMITED/NO ALIGNMENT EVIDENCE — FEWER THAN "
            "TWO PRIMARY CIRCULAR-SHIFT METRICS PASS"
        )
        print()
        print(
            "The F17 timing lead should not be upgraded to a broad "
            "cross-modal alignment conclusion."
        )

    print()
    print(
        f"Block-permutation sensitivity evidence class: {block_evidence}"
    )
    print()
    print("No chromatin-priming claim was made.")
    line("=")


if __name__ == "__main__":
    main()

