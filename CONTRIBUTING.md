# Contributing

Contributions are welcome when they improve exact manuscript reproducibility, installation, portability, documentation, validation, or clearly separated scientific extensions.

## Reproducibility rule

The manuscript pipeline contains frozen analytical choices. Changes to thresholds, event-family definitions, pseudotime construction, window definitions, bootstrap rules, null models, or other inferential settings must be identified as either:

1. a bug fix restoring documented canonical behavior, or
2. a new exploratory/extended analysis kept separate from exact manuscript replication.

Do not silently modify frozen settings to obtain a different biological direction or significance result.

## Before submitting a pull request

```bash
python validate_pipeline_package.py
python -m compileall -q scripts run_pipeline.py validate_pipeline_package.py
```

If the scientific environment is installed:

```bash
python scripts/00_preflight.py
```

For branch-specific changes, rerun the affected branch when feasible and compare against `docs/REFERENCE_RESULTS.md`.

## Bug reports

Please include the operating system, Python version, exact command, failing script/phase, relevant log excerpt, and whether the issue occurs in an unmodified canonical checkout.

## Scientific extensions

Extensions to new datasets or modalities should preserve the conceptual separation between trajectory construction, modality-specific instability estimation, event definition, timing inference, broad alignment testing, and mechanistic interpretation.
