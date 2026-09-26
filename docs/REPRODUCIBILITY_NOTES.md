# Reproducibility notes

- Run every command from the **project root**. All scientific scripts deliberately use relative `data/...` paths.
- `pyGDIS==1.0.0` is scientifically frozen and is checked by the preflight and the external-validation GDIS scripts.
- The full GSE205117 branch downloads a large GEO RAW archive (~32.5 GB). The complete project should have substantial free disk space; 80+ GiB is recommended.
- The master runner uses the exhaustive `--full-atac-target-scan` for F7b and `--full-fragment-scan` for F10c, rather than the faster bounded diagnostics.
- F20 and F21 download frozen public reference resources into `data/GSE140203/reference_f20/` and `reference_f21/`; their scripts record SHA-256 hashes in manifests.
- Discovery scripts F0–F5 were developed as audit/analysis stages and often print results rather than saving intermediate files. The canonical runner writes every terminal result to a persistent per-step log under `logs/`.
- A scientifically negative verdict is not automatically an execution error. In particular, GSE205117 intentionally terminates as a trajectory-suitability negative result, and F21 intentionally preserves a zero-enriched-motif result.
- Do not add post hoc thresholds or substitute superseded scripts to force a desired biological direction.
- For public GitHub use, follow `docs/REPRODUCE.md` and compare reruns against `docs/REFERENCE_RESULTS.md` before changing any frozen setting.
