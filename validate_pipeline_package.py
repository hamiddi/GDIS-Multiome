#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""Static integrity validator for the canonical pipeline package."""
from __future__ import annotations
import ast
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
SCRIPTS=ROOT/'scripts'
RUNNER=ROOT/'run_pipeline.py'

EXCLUDED={
'13_phase_f6_external_dataset_audit.py',
'14_phase_f6b_external_dataset_audit.py',
'17_phase_f8_external_representation_acquisition_qc.py',
'17_phase_f8_external_representation_acquisition_qc_v2.py',
'17_phase_f8_geo_native_reconstruction_qc.py',
'17_phase_f8_geo_native_reconstruction_qc_contigfix.py',
'22_phase_f10_shareseq_acquisition_audit.py',
'23_phase_f10b_shareseq_format_corrected_audit.py',
'38_phase_f23_shareseq_irs_publication_figures.py',
'40_phase_f24_discovery_publication_figure_v2.py',
'40_phase_f24_discovery_publication_figure_v3.py',
'41_supplementary_figure_s1_discovery_diagnostics_v2.py',
'41_supplementary_figure_s1_discovery_diagnostics_v3.py',
'41_supplementary_figure_s1_discovery_diagnostics_v4.py',
'42_supplementary_figure_s2_discovery_alignment_null_v2.py',
}


def main():
    failures=[]
    scripts=sorted(SCRIPTS.glob('*.py'))
    print(f'Canonical Python scripts: {len(scripts)}')
    for p in scripts:
        try:
            ast.parse(p.read_text(encoding='utf-8'))
            print(f'SYNTAX PASS  {p.name}')
        except Exception as exc:
            failures.append(f'Syntax failure {p.name}: {exc}')

    present={p.name for p in scripts}
    bad=sorted(EXCLUDED & present)
    if bad:
        failures.append('Excluded scripts present: '+', '.join(bad))
    else:
        print('EXCLUSION PASS  no superseded scripts in canonical scripts/')

    versioned=sorted(
        p.name for p in scripts
        if re.search(r'_v[0-9]+\.py$', p.name)
    )
    if versioned:
        failures.append(
            'Version-suffixed scripts present in canonical scripts/: '
            + ', '.join(versioned)
        )
    else:
        print('VERSION PASS  no old version-suffixed scripts in canonical scripts/')

    runner_text=RUNNER.read_text(encoding='utf-8')
    refs=set(re.findall(r'\("([0-9][^\"]+\.py)"',runner_text))
    refs.add('00_preflight.py')
    missing=sorted(refs-present)
    if missing:
        failures.append('Runner references missing scripts: '+', '.join(missing))
    else:
        print(f'RUNNER PASS  all {len(refs)} referenced scripts exist')

    required_docs=[
        ROOT/'README.md',
        ROOT/'docs/canonical_pipeline_manifest.tsv',
        ROOT/'docs/source_archive_inventory.tsv',
        ROOT/'docs/excluded_scripts.tsv',
        ROOT/'config/requirements.txt',
        ROOT/'config/environment.yml',
        ROOT/'checksums/gse140203_sha256.tsv',
    ]
    missing_docs=[str(p.relative_to(ROOT)) for p in required_docs if not p.exists()]
    if missing_docs:
        failures.append('Missing package files: '+', '.join(missing_docs))
    else:
        print('PACKAGE PASS  required manifests/configuration files exist')

    if failures:
        print('\nSTATIC VALIDATION: FAIL')
        for failure in failures:
            print(' -',failure)
        raise SystemExit(1)

    print('\nSTATIC VALIDATION: PASS')

if __name__=='__main__':
    main()
