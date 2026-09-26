# GitHub public-release checklist

- [ ] Replace `<YOUR-GITHUB-REPOSITORY-URL>` placeholders.
- [ ] Select and add the final software license; remove `LICENSE_PENDING.md`.
- [ ] Confirm author names and affiliations.
- [ ] Update `CITATION.cff` with repository URL and release metadata.
- [ ] Add article DOI/journal citation when available.
- [ ] Run `python validate_pipeline_package.py`.
- [ ] Run `python scripts/00_preflight.py` in the intended release environment.
- [ ] Confirm no large data, generated results, credentials, tokens, private paths, or institutional secrets are staged.
- [ ] Confirm no `__pycache__`, `.pyc`, virtual environments, or temporary downloads are committed.
- [ ] Create a tagged release after the exact manuscript code is frozen.
- [ ] Archive that tagged release permanently if desired.
