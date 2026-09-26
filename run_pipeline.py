#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
Master runner for the canonical GDIS-Multiome pipeline.

Examples
--------
Full pipeline, including GSE205117 trajectory-suitability branch:
    python run_pipeline.py --branch all

Manuscript core (discovery + SHARE-seq positive validation):
    python run_pipeline.py --branch core

Individual branch:
    python run_pipeline.py --branch discovery
    python run_pipeline.py --branch gse205117
    python run_pipeline.py --branch shareseq

The runner executes from the project root, streams output to the terminal, and
writes one persistent log per step under logs/. A non-zero script exit stops
that branch immediately.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
LOG_DIR = ROOT / "logs"

DISCOVERY = [
    ("01_download_gse275562.py", []),
    ("02_phase_f0_audit.py", []),
    ("03_phase_f1_batch_representation_qc.py", []),
    ("04_phase_f1b_representation_rescue_qc.py", []),
    ("05_phase_f2_common_trajectory.py", []),
    ("06_phase_f3_gdis_rna_atac.py", []),
    ("07_phase_f3b_peak_topology_qc.py", []),
    ("08_phase_f3c_transition_anchored_pairing_qc.py", []),
    ("09_phase_f4_event_specific_cmil_bootstrap.py", []),
    ("10_phase_f4b_peak_stability_diagnostics.py", []),
    ("11_phase_f4c_event_region_timing_diagnostics.py", []),
    ("12_phase_f5_cross_modal_alignment_null.py", []),
    # Publication-only outputs from frozen discovery evidence.
    ("40_phase_f24_discovery_publication_figure.py", []),
    ("41_supplementary_figure_s1_discovery_diagnostics.py", []),
    ("42_supplementary_figure_s2_discovery_alignment_null.py", []),
]

GSE205117 = [
    # Full pipeline requires the large RAW archive for F7/F8.
    ("13_phase_f6a_external_metadata_acquisition.py", ["--include-raw"]),
    ("14_phase_f6b_external_dataset_audit_corrected.py", []),
    ("15_phase_f7_external_archive_pairing_qc.py", []),
    # Use the corrected, scientifically relevant exhaustive target-cell scan.
    ("16_phase_f7b_external_barcode_universe_diagnostic.py", ["--full-atac-target-scan"]),
    ("17_phase_f8_geo_native_reconstruction_qc_dynamicbins.py", []),
    ("18_phase_f8b_external_representation_rescue_qc.py", []),
    ("19_phase_f8c_external_primary_arm_freeze_qc.py", []),
    ("20_phase_f9_external_common_trajectory.py", []),
    ("21_phase_f9b_external_trajectory_topology_diagnostic.py", []),
    # Publication-only trajectory-suitability figure from frozen evidence.
    ("43_supplementary_figure_s3_gse205117_trajectory_suitability.py", []),
]

SHARESEQ = [
    ("22_phase_f10_shareseq_acquisition_canonical.py", []),
    # Exhaustive fragment validation is canonical for a from-zero replication.
    ("24_phase_f10c_shareseq_namespace_normalized_audit.py", ["--full-fragment-scan"]),
    ("25_phase_f11_shareseq_representations_branch_continuity.py", []),
    ("26_phase_f12_shareseq_irs_freeze_rna_technical_rescue.py", []),
    ("27_phase_f13_shareseq_irs_common_trajectory.py", []),
    ("28_phase_f14_shareseq_irs_gdis_primary_profiles.py", []),
    ("29_phase_f15_shareseq_irs_peak_topology_qc.py", []),
    ("30_phase_f15b_shareseq_irs_sustained_plateau_transition_energy_qc.py", []),
    ("31_phase_f16_shareseq_irs_transition_energy_pair_freeze.py", []),
    ("32_phase_f17_shareseq_irs_frozen_pair_bootstrap.py", []),
    ("33_phase_f18_shareseq_irs_cross_modal_alignment_null.py", []),
    ("34_phase_f19_shareseq_irs_frozen_event_feature_extraction.py", []),
    ("35_phase_f20_shareseq_irs_mm10_peak_gene_annotation.py", []),
    ("36_phase_f21_shareseq_irs_motif_tf_regulatory_plausibility.py", []),
    ("37_phase_f22_shareseq_irs_external_validation_evidence_freeze.py", []),
    ("39_phase_f23b_shareseq_irs_publication_figure_revision.py", []),
    # Additional publication-only outputs from frozen SHARE-seq evidence.
    ("44_supplementary_figure_s4_tf_motif_enrichment.py", []),
    ("45_main_figure_5_exact_circular_shift_null.py", []),
    ("46_supplementary_figure_s5_shareseq_sustained_architecture.py", []),
    ("47_supplementary_table_s4_shareseq_event_family_selection.py", []),
    ("48_supplementary_table_s5_shareseq_branch_selection_qc.py", []),
]


def run_step(script_name: str, extra_args: list[str], stamp: str) -> None:
    script = SCRIPTS / script_name
    if not script.exists():
        raise FileNotFoundError(f"Canonical script missing: {script}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{stamp}_{script.stem}.log"

    command = [sys.executable, "-u", str(script), *extra_args]
    print("\n" + "=" * 120)
    print("RUNNING:", " ".join(command))
    print("LOG:    ", log_path)
    print("=" * 120, flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Pipeline stopped: {script_name} returned exit code {return_code}. "
            f"See {log_path}."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--branch",
        choices=["all", "core", "discovery", "gse205117", "shareseq"],
        default="all",
        help=(
            "all = discovery + GSE205117 suitability + SHARE-seq; "
            "core = discovery + SHARE-seq."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip dependency/version preflight (not recommended).",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.skip_preflight:
        run_step("00_preflight.py", [], stamp)

    if args.branch in {"all", "core", "discovery"}:
        for step in DISCOVERY:
            run_step(*step, stamp)

    if args.branch in {"all", "gse205117"}:
        for step in GSE205117:
            run_step(*step, stamp)

    if args.branch in {"all", "core", "shareseq"}:
        for step in SHARESEQ:
            run_step(*step, stamp)

    print("\n" + "=" * 120)
    print("CANONICAL PIPELINE COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
