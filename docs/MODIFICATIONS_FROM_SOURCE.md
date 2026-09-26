# Intentional modifications from the uploaded source archive

The scientific analysis code is preserved as uploaded except for the following canonicalization changes.

## 1. F8b filename normalization

Uploaded valid file:

`8_phase_f8b_external_representation_rescue_qc.py`

Canonical filename:

`18_phase_f8b_external_representation_rescue_qc.py`

The file contents are unchanged; only the filename is corrected to match its own F8b docstring and its true position between F8 and F8c.

## 2. SHARE-seq F10 acquisition-only replacement

The uploaded development scripts `22_phase_f10_shareseq_acquisition_audit.py` and `23_phase_f10b_shareseq_format_corrected_audit.py` are intentionally not executed. Their structural-audit assumptions were superseded by the valid namespace-normalized F10c audit in script 24.

The canonical package therefore adds:

`22_phase_f10_shareseq_acquisition_canonical.py`

This new script performs **data acquisition only**:

- downloads the same six frozen GSE140203 supplementary files;
- supports `.part` resume;
- validates gzip readability;
- checks the frozen SHA-256 for every file;
- performs no barcode or biological interpretation.

The scientific audit begins only in canonical script 24.

## 3. Exhaustive validation flags in the master runner

For a from-zero full replication, the master runner invokes:

- F6a with `--include-raw` so the GSE205117 RAW archive required downstream is actually acquired;
- F7b with `--full-atac-target-scan` so target metadata cells are exhaustively checked against ATAC fragments;
- F10c with `--full-fragment-scan` so raw SHARE-seq fragment namespace correspondence is exhaustively checked.

These flags strengthen completeness of reproduction; they do not change frozen analysis thresholds or biological inference.

## 4. Final figures

Only script 39/F23b is retained. Script 38/F23 is excluded because F23b is the final presentation-only revision and uses the primary 400/100 frozen event locations while leaving the scientific results unchanged.


## 5. Additional publication outputs

The updated package incorporates the publication-only scripts supplied in
`additional.zip` as canonical scripts 40–48. Duplicate development versions
are removed from `scripts/`.

Newest selected versions:
- script 40: v3 -> clean canonical filename;
- script 41: v4 -> clean canonical filename;
- script 42: v2 -> clean canonical filename;
- scripts 43–48: retained from the supplied unversioned files.

Script 46 already contains the corrected sustained-architecture v2
implementation.

A packaging-only path correction was applied to script 40 so it can run from
the canonical `scripts/` directory while reading project data from the package
root. No scientific threshold, frozen event, statistical result, or biological
interpretation was changed.

The new files are publication/reporting stages and consume frozen upstream
results. The master runner executes them after their corresponding scientific
branch.
