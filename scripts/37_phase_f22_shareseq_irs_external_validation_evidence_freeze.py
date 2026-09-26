#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
37_phase_f22_shareseq_irs_external_validation_evidence_freeze.py
================================================================

Project:
GDIS-Multiome independent positive validation

Dataset:
GSE140203 / SHARE-seq mouse skin late anagen

PHASE F22
---------
Freeze and consolidate the complete external-validation evidence chain from
F13 through F21 WITHOUT performing any new biological or statistical analysis.

WHY F22
-------
The external-validation workflow is now complete:

F13:
    RNA-only common clock validated.

F14-F15b:
    RNA and ATAC GDIS/transition-energy topology characterized without
    forcing equivalent peak structures.

F16:
    one localized RNA/ATAC transition-energy family pair frozen before
    inference.

F17:
    frozen pair bootstrap-supported with ATAC earlier.

F18:
    broad cross-modal transition-energy profile alignment NOT supported by
    the prespecified primary circular-shift null.

F19-F20:
    frozen event-window RNA/ATAC feature extraction and nearest-TSS proximity
    annotation completed.

F21:
    matched-background JASPAR2026 motif enrichment found NO motif satisfying
    the prespecified enrichment criteria.

F22 therefore creates the final reproducible external-validation evidence
table and claim-boundary table.

IMPORTANT
---------
F22 performs NO:
    - GDIS calculation
    - pseudotime calculation
    - event selection
    - bootstrap
    - permutation/null test
    - motif scan
    - enrichment test
    - P-value calculation
    - threshold optimization
    - biological rescue analysis

INPUTS
------
data/GSE140203/representations_f13/
    f13_trajectory_validation.tsv
    f13_manifest.json

data/GSE140203/representations_f16/
    f16_frozen_transition_energy_pair.tsv
    f16_window_specific_pairing.tsv
    f16_manifest.json

data/GSE140203/representations_f17/
    f17_bootstrap_summary.tsv
    f17_manifest.json

data/GSE140203/representations_f18/
    f18_alignment_summary.tsv
    f18_manifest.json

data/GSE140203/representations_f20/
    f20_cross_modal_proximity_candidates.tsv
    f20_manifest.json

data/GSE140203/representations_f21/
    f21_motif_enrichment.tsv
    f21_enriched_tf_candidates.tsv
    f21_manifest.json

OUTPUTS
-------
data/GSE140203/representations_f22/
    f22_external_validation_summary.tsv
    f22_claim_boundaries.tsv
    f22_window_specific_timing.tsv
    f22_cross_modal_candidate_summary.tsv
    f22_manifest.json

RUN
---
    python 37_phase_f22_shareseq_irs_external_validation_evidence_freeze.py

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

F13_DIR = DATA_DIR / "representations_f13"
F16_DIR = DATA_DIR / "representations_f16"
F17_DIR = DATA_DIR / "representations_f17"
F18_DIR = DATA_DIR / "representations_f18"
F20_DIR = DATA_DIR / "representations_f20"
F21_DIR = DATA_DIR / "representations_f21"
F22_DIR = DATA_DIR / "representations_f22"

F13_VALIDATION = F13_DIR / "f13_trajectory_validation.tsv"
F13_MANIFEST = F13_DIR / "f13_manifest.json"

F16_PAIR = F16_DIR / "f16_frozen_transition_energy_pair.tsv"
F16_WINDOWS = F16_DIR / "f16_window_specific_pairing.tsv"
F16_MANIFEST = F16_DIR / "f16_manifest.json"

F17_SUMMARY = F17_DIR / "f17_bootstrap_summary.tsv"
F17_MANIFEST = F17_DIR / "f17_manifest.json"

F18_SUMMARY = F18_DIR / "f18_alignment_summary.tsv"
F18_MANIFEST = F18_DIR / "f18_manifest.json"

F20_CROSS_MODAL = F20_DIR / "f20_cross_modal_proximity_candidates.tsv"
F20_MANIFEST = F20_DIR / "f20_manifest.json"

F21_ENRICHMENT = F21_DIR / "f21_motif_enrichment.tsv"
F21_ENRICHED = F21_DIR / "f21_enriched_tf_candidates.tsv"
F21_MANIFEST = F21_DIR / "f21_manifest.json"

OUTPUT_SUMMARY = F22_DIR / "f22_external_validation_summary.tsv"
OUTPUT_CLAIMS = F22_DIR / "f22_claim_boundaries.tsv"
OUTPUT_WINDOWS = F22_DIR / "f22_window_specific_timing.tsv"
OUTPUT_CANDIDATES = F22_DIR / "f22_cross_modal_candidate_summary.tsv"
OUTPUT_MANIFEST = F22_DIR / "f22_manifest.json"


# =====================================================================
# UTILITIES
# =====================================================================

def line(char="=", width=124):
    print(char * width)


def section(title):
    print()
    line("=")
    print(title)
    line("=")


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


def require_inputs():
    required = [
        F13_VALIDATION,
        F13_MANIFEST,
        F16_PAIR,
        F16_WINDOWS,
        F16_MANIFEST,
        F17_SUMMARY,
        F17_MANIFEST,
        F18_SUMMARY,
        F18_MANIFEST,
        F20_CROSS_MODAL,
        F20_MANIFEST,
        F21_ENRICHMENT,
        F21_ENRICHED,
        F21_MANIFEST,
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


def read_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


# =====================================================================
# MAIN
# =====================================================================

def main():
    section(
        "PHASE F22 — FREEZE COMPLETE SHARE-seq IRS EXTERNAL-VALIDATION EVIDENCE"
    )

    print(
        "F22 performs no new analysis."
    )

    print()

    print(
        "It consolidates the already-frozen F13-F21 evidence chain."
    )

    require_inputs()

    F22_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Load frozen outputs.
    # -----------------------------------------------------------------

    section(
        "1. LOAD FROZEN F13-F21 RESULTS"
    )

    f13_validation = pd.read_csv(
        F13_VALIDATION,
        sep="\t",
        low_memory=False,
    )

    f16_pair = pd.read_csv(
        F16_PAIR,
        sep="\t",
        low_memory=False,
    )

    f16_windows = pd.read_csv(
        F16_WINDOWS,
        sep="\t",
        low_memory=False,
    )

    f17 = pd.read_csv(
        F17_SUMMARY,
        sep="\t",
        low_memory=False,
    )

    f18 = pd.read_csv(
        F18_SUMMARY,
        sep="\t",
        low_memory=False,
    )

    f20_candidates = pd.read_csv(
        F20_CROSS_MODAL,
        sep="\t",
        low_memory=False,
    )

    f21_enrichment = pd.read_csv(
        F21_ENRICHMENT,
        sep="\t",
        low_memory=False,
    )

    f21_enriched = pd.read_csv(
        F21_ENRICHED,
        sep="\t",
        low_memory=False,
    )

    f13_manifest = read_json(
        F13_MANIFEST
    )

    f16_manifest = read_json(
        F16_MANIFEST
    )

    f17_manifest = read_json(
        F17_MANIFEST
    )

    f18_manifest = read_json(
        F18_MANIFEST
    )

    f20_manifest = read_json(
        F20_MANIFEST
    )

    f21_manifest = read_json(
        F21_MANIFEST
    )

    if len(
        f16_pair
    ) != 1:
        raise RuntimeError(
            "F16 pair table must contain exactly one row."
        )

    if len(
        f17
    ) != 1:
        raise RuntimeError(
            "F17 summary must contain exactly one row."
        )

    if len(
        f18
    ) != 1:
        raise RuntimeError(
            "F18 summary must contain exactly one row."
        )

    pair = f16_pair.iloc[
        0
    ]

    boot = f17.iloc[
        0
    ]

    align = f18.iloc[
        0
    ]

    # -----------------------------------------------------------------
    # Freeze evidence summary.
    # -----------------------------------------------------------------

    section(
        "2. EXTERNAL-VALIDATION EVIDENCE SUMMARY"
    )

    summary_rows = [
        {
            "evidence_domain": "Common developmental clock",
            "result": "SUPPORTED",
            "primary_statistic": (
                "RNA-only DPT; TAC-1 < TAC-2 < IRS"
            ),
            "value_1_name": "k20_vs_k30_Spearman",
            "value_1": float(
                f13_manifest[
                    "k_sensitivity"
                ][
                    0
                ][
                    "spearman_rho"
                ]
            ),
            "value_2_name": "k50_vs_k30_Spearman",
            "value_2": float(
                f13_manifest[
                    "k_sensitivity"
                ][
                    1
                ][
                    "spearman_rho"
                ]
            ),
            "interpretation": (
                "RNA-derived common clock validated independently of ATAC."
            ),
        },
        {
            "evidence_domain": "Frozen localized event pair",
            "result": "SUPPORTED_DESCRIPTIVELY",
            "primary_statistic": (
                "RNA transition-energy family center - "
                "ATAC transition-energy family center"
            ),
            "value_1_name": "RNA_family_center",
            "value_1": float(
                pair[
                    "RNA_family_center_parameter"
                ]
            ),
            "value_2_name": "ATAC_family_center",
            "value_2": float(
                pair[
                    "ATAC_family_center_parameter"
                ]
            ),
            "interpretation": (
                "ATAC localized event was earlier in the independently "
                "frozen event-family pair."
            ),
        },
        {
            "evidence_domain": "Bootstrap timing support",
            "result": (
                "SUPPORTED"
                if bool(
                    boot[
                        "bootstrap_ATAC_earlier_support_pass"
                    ]
                )
                else "NOT_SUPPORTED"
            ),
            "primary_statistic": "Bootstrap CMIL = RNA - ATAC",
            "value_1_name": "bootstrap_median",
            "value_1": float(
                boot[
                    "bootstrap_CMIL_median"
                ]
            ),
            "value_2_name": "paired_detection_rate",
            "value_2": float(
                boot[
                    "paired_detection_rate"
                ]
            ),
            "interpretation": (
                "Frozen localized ATAC-earlier event timing is "
                "bootstrap-supported."
            ),
        },
        {
            "evidence_domain": "Bootstrap confidence interval",
            "result": (
                "SUPPORTED"
                if float(
                    boot[
                        "bootstrap_CMIL_ci025"
                    ]
                )
                > 0
                else "NOT_SUPPORTED"
            ),
            "primary_statistic": "95% percentile CI",
            "value_1_name": "CI_lower",
            "value_1": float(
                boot[
                    "bootstrap_CMIL_ci025"
                ]
            ),
            "value_2_name": "CI_upper",
            "value_2": float(
                boot[
                    "bootstrap_CMIL_ci975"
                ]
            ),
            "interpretation": (
                "Bootstrap interval excludes zero in the ATAC-earlier "
                "direction."
            ),
        },
        {
            "evidence_domain": "Directional bootstrap probability",
            "result": (
                "SUPPORTED"
                if float(
                    boot[
                        "P_boot_CMIL_gt_0"
                    ]
                )
                >= 0.95
                else "NOT_SUPPORTED"
            ),
            "primary_statistic": "P_boot(CMIL > 0)",
            "value_1_name": "P_boot_gt_0",
            "value_1": float(
                boot[
                    "P_boot_CMIL_gt_0"
                ]
            ),
            "value_2_name": "paired_detection_n",
            "value_2": float(
                boot[
                    "paired_detection_n"
                ]
            ),
            "interpretation": (
                "Bootstrap direction strongly favors ATAC earlier."
            ),
        },
        {
            "evidence_domain": "Broad cross-modal alignment",
            "result": str(
                align[
                    "circular_alignment_evidence_class"
                ]
            ),
            "primary_statistic": (
                "Primary circular-shift alignment null"
            ),
            "value_1_name": "circular_significant_metrics",
            "value_1": float(
                align[
                    "circular_n_significant_metrics"
                ]
            ),
            "value_2_name": "block_significant_metrics",
            "value_2": float(
                align[
                    "block_n_significant_metrics"
                ]
            ),
            "interpretation": (
                "Broad RNA/ATAC transition-energy alignment is not "
                "supported by the primary circular-shift null."
            ),
        },
        {
            "evidence_domain": "Nearest-TSS cross-modal candidates",
            "result": "DESCRIPTIVE_ONLY",
            "primary_statistic": (
                "ATAC-opening nearest-TSS genes also increasing later in RNA"
            ),
            "value_1_name": "candidate_gene_count",
            "value_1": float(
                len(
                    f20_candidates
                )
            ),
            "value_2_name": "motif_supported_candidate_gene_count",
            "value_2": float(
                f21_manifest[
                    "counts"
                ][
                    "cross_modal_genes_with_enriched_motif_support"
                ]
            ),
            "interpretation": (
                "Nearest-TSS overlap provides candidate proximity, not "
                "causal regulatory linkage."
            ),
        },
        {
            "evidence_domain": "Motif/TF regulatory program",
            "result": (
                "SUPPORTED"
                if len(
                    f21_enriched
                )
                > 0
                else "NOT_SUPPORTED"
            ),
            "primary_statistic": (
                "JASPAR2026 matched-background motif enrichment"
            ),
            "value_1_name": "motifs_tested",
            "value_1": float(
                len(
                    f21_enrichment
                )
            ),
            "value_2_name": "enriched_motifs",
            "value_2": float(
                len(
                    f21_enriched
                )
            ),
            "interpretation": (
                "No coherent TF motif program passed the prespecified "
                "FDR/OR/hit thresholds."
            ),
        },
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        sep="\t",
        index=False,
    )

    print_df(
        summary.set_index(
            "evidence_domain"
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Claim boundaries.
    # -----------------------------------------------------------------

    section(
        "3. FREEZE CLAIM BOUNDARIES"
    )

    claim_rows = [
        {
            "claim": (
                "A validated RNA-derived common developmental clock exists "
                "for TAC-1 -> TAC-2 -> IRS."
            ),
            "status": "SUPPORTED",
            "reason": (
                "F13 passed connectivity, biological ordering, and k-sensitivity "
                "criteria without ATAC entering clock construction."
            ),
        },
        {
            "claim": (
                "A localized ATAC transition-energy event precedes the "
                "prospectively frozen RNA event."
            ),
            "status": "SUPPORTED",
            "reason": (
                "F16 froze the pair before inference; F17 bootstrap passed "
                "detection, CI, and directional-support criteria."
            ),
        },
        {
            "claim": (
                "The complete RNA and ATAC transition-energy landscapes are "
                "globally aligned."
            ),
            "status": "NOT_SUPPORTED",
            "reason": (
                "F18 primary circular-shift null classified broad alignment "
                "as LIMITED_OR_NONE."
            ),
        },
        {
            "claim": (
                "The external validation demonstrates a coherent enriched "
                "TF motif program at the ATAC-opening event."
            ),
            "status": "NOT_SUPPORTED",
            "reason": (
                "F21 tested the complete expressed single-TF JASPAR2026 "
                "universe and no motif passed prespecified enrichment criteria."
            ),
        },
        {
            "claim": (
                "Nearest-TSS ATAC peaks prove regulation of the associated "
                "later RNA genes."
            ),
            "status": "NOT_SUPPORTED",
            "reason": (
                "F20 proximity annotation is descriptive and does not establish "
                "causal peak-to-gene regulation."
            ),
        },
        {
            "claim": (
                "The SHARE-seq result proves universal chromatin priming."
            ),
            "status": "NOT_SUPPORTED",
            "reason": (
                "Evidence supports one localized ATAC-earlier event in one "
                "validated branch, while broad profile alignment and motif-level "
                "mechanistic enrichment were not established."
            ),
        },
        {
            "claim": (
                "GDIS-Multiome can distinguish localized cross-modal timing "
                "from broader sustained-instability architecture."
            ),
            "status": "SUPPORTED",
            "reason": (
                "RNA and ATAC showed different sustained-GDIS topology while "
                "a localized transition-energy pair could still be frozen and "
                "bootstrap-tested."
            ),
        },
    ]

    claims = pd.DataFrame(
        claim_rows
    )

    claims.to_csv(
        OUTPUT_CLAIMS,
        sep="\t",
        index=False,
    )

    print_df(
        claims.set_index(
            "status"
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Window-specific timing.
    # -----------------------------------------------------------------

    section(
        "4. FREEZE WINDOW-SPECIFIC TIMING TABLE"
    )

    f16_windows.to_csv(
        OUTPUT_WINDOWS,
        sep="\t",
        index=False,
    )

    print_df(
        f16_windows.set_index(
            "window_config"
        ),
        digits=6,
    )

    # -----------------------------------------------------------------
    # Cross-modal candidates.
    # -----------------------------------------------------------------

    section(
        "5. FREEZE DESCRIPTIVE CROSS-MODAL CANDIDATES"
    )

    if f20_candidates.empty:
        candidate_summary = pd.DataFrame(
            columns=[
                "gene",
                "minimum_abs_distance_to_TSS",
                "n_ATAC_opening_peaks",
                "strongest_delta_accessibility",
                "RNA_event_CPM",
                "RNA_event_detected_fraction",
                "log2FC_RNAevent_vs_ATACevent_CPM",
                "motif_enrichment_support",
            ]
        )

    else:
        candidate_summary = f20_candidates.copy()

        candidate_summary[
            "motif_enrichment_support"
        ] = False

    candidate_summary.to_csv(
        OUTPUT_CANDIDATES,
        sep="\t",
        index=False,
    )

    print_df(
        candidate_summary.head(
            50
        ).set_index(
            "gene"
        )
        if (
            not candidate_summary.empty
            and "gene"
            in candidate_summary.columns
        )
        else candidate_summary,
        digits=6,
    )

    # -----------------------------------------------------------------
    # Final safeguards.
    # -----------------------------------------------------------------

    section(
        "6. F22 EVIDENCE-FREEZE SAFEGUARDS"
    )

    checks = pd.DataFrame(
        [
            {
                "criterion": (
                    "F17 bootstrap support flag remains PASS"
                ),
                "pass": bool(
                    boot[
                        "bootstrap_ATAC_earlier_support_pass"
                    ]
                ),
            },
            {
                "criterion": (
                    "F18 primary alignment class remains LIMITED_OR_NONE"
                ),
                "pass": (
                    str(
                        align[
                            "circular_alignment_evidence_class"
                        ]
                    )
                    == "LIMITED_OR_NONE"
                ),
            },
            {
                "criterion": (
                    "F21 enriched motif count remains zero"
                ),
                "pass": (
                    len(
                        f21_enriched
                    )
                    == 0
                ),
            },
            {
                "criterion": (
                    "No new statistical or biological inference performed in F22"
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
        "7. SAVE F22 MANIFEST"
    )

    payload = {
        "dataset_accession": ACCESSION,
        "phase": "F22",
        "created_utc": utc_now_iso(),
        "purpose": (
            "Freeze and consolidate the completed SHARE-seq external-validation "
            "evidence chain without performing new analysis."
        ),
        "external_validation_classification": {
            "common_clock": "SUPPORTED",
            "localized_ATAC_earlier_event": (
                "SUPPORTED_BY_BOOTSTRAP"
            ),
            "broad_cross_modal_alignment": (
                str(
                    align[
                        "circular_alignment_evidence_class"
                    ]
                )
            ),
            "motif_TF_regulatory_program": (
                "NOT_SUPPORTED"
                if len(
                    f21_enriched
                )
                == 0
                else "SUPPORTED"
            ),
            "universal_chromatin_priming": "NOT_SUPPORTED",
        },
        "key_numbers": {
            "RNA_family_center": float(
                pair[
                    "RNA_family_center_parameter"
                ]
            ),
            "ATAC_family_center": float(
                pair[
                    "ATAC_family_center_parameter"
                ]
            ),
            "F16_family_center_separation": float(
                pair[
                    "observed_RNA_minus_ATAC_family_center"
                ]
            ),
            "F17_primary_observed_CMIL": float(
                boot[
                    "observed_primary_CMIL_RNA_minus_ATAC"
                ]
            ),
            "F17_bootstrap_median_CMIL": float(
                boot[
                    "bootstrap_CMIL_median"
                ]
            ),
            "F17_bootstrap_CI_lower": float(
                boot[
                    "bootstrap_CMIL_ci025"
                ]
            ),
            "F17_bootstrap_CI_upper": float(
                boot[
                    "bootstrap_CMIL_ci975"
                ]
            ),
            "F17_P_boot_CMIL_gt_0": float(
                boot[
                    "P_boot_CMIL_gt_0"
                ]
            ),
            "F17_paired_detection_rate": float(
                boot[
                    "paired_detection_rate"
                ]
            ),
            "F18_circular_significant_metrics": int(
                align[
                    "circular_n_significant_metrics"
                ]
            ),
            "F20_cross_modal_candidate_genes": int(
                len(
                    f20_candidates
                )
            ),
            "F21_motifs_tested": int(
                len(
                    f21_enrichment
                )
            ),
            "F21_enriched_motifs": int(
                len(
                    f21_enriched
                )
            ),
        },
        "checks": checks.to_dict(
            orient="records"
        ),
        "outputs": {
            "summary": {
                "file": str(
                    OUTPUT_SUMMARY
                ),
                "sha256": sha256_file(
                    OUTPUT_SUMMARY
                ),
            },
            "claims": {
                "file": str(
                    OUTPUT_CLAIMS
                ),
                "sha256": sha256_file(
                    OUTPUT_CLAIMS
                ),
            },
            "window_specific_timing": {
                "file": str(
                    OUTPUT_WINDOWS
                ),
                "sha256": sha256_file(
                    OUTPUT_WINDOWS
                ),
            },
            "cross_modal_candidates": {
                "file": str(
                    OUTPUT_CANDIDATES
                ),
                "sha256": sha256_file(
                    OUTPUT_CANDIDATES
                ),
            },
        },
        "guardrails": {
            "new_GDIS_calculation": False,
            "new_pseudotime_calculation": False,
            "new_event_selection": False,
            "new_bootstrap": False,
            "new_null_test": False,
            "new_motif_scan": False,
            "new_enrichment_test": False,
            "new_pvalue": False,
            "threshold_changed": False,
            "negative_result_rescued": False,
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
        OUTPUT_SUMMARY,
        OUTPUT_CLAIMS,
        OUTPUT_WINDOWS,
        OUTPUT_CANDIDATES,
        OUTPUT_MANIFEST,
    ]:
        print(path)

    # -----------------------------------------------------------------
    # Final verdict.
    # -----------------------------------------------------------------

    section(
        "8. PHASE F22 DECISION"
    )

    if all_pass:
        print(
            "PHASE F22 VERDICT: GO — SHARE-seq IRS EXTERNAL-VALIDATION "
            "EVIDENCE CHAIN FROZEN"
        )

        print()

        print(
            "Supported:"
        )

        print(
            "  - validated RNA-derived common clock"
        )

        print(
            "  - localized transition-energy ATAC-earlier event"
        )

        print(
            "  - bootstrap-supported timing direction"
        )

        print(
            "  - GDIS-Multiome distinguishes localized event timing from "
            "broader sustained-instability architecture"
        )

        print()

        print(
            "Not supported:"
        )

        print(
            "  - broad cross-modal profile alignment under the primary null"
        )

        print(
            "  - coherent enriched TF motif program"
        )

        print(
            "  - causal peak-to-gene regulation"
        )

        print(
            "  - universal chromatin priming"
        )

        print()

        print(
            "No additional mechanistic rescue analysis is recommended."
        )

    else:
        print(
            "PHASE F22 VERDICT: REVIEW REQUIRED"
        )

        print()

        print(
            "A frozen upstream result no longer matches its expected final "
            "interpretation. Review before figure/manuscript synthesis."
        )

    line("=")


if __name__ == "__main__":
    main()

