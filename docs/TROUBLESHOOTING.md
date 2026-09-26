# Troubleshooting

## pyGDIS version mismatch

Manuscript replication requires:

```text
pygdis==1.0.0
```

Recreate the environment or reinstall the pinned package before rerunning GDIS stages.

## Insufficient disk space

The full GSE205117 branch includes a raw GEO archive of approximately 32.5 GB and creates additional derived files. Use a filesystem with at least ~80 GiB free.

## Interrupted downloads

Rerun the relevant acquisition stage. Downstream analysis should proceed only after integrity checks pass.

## Long exhaustive scans

The master runner intentionally uses exhaustive barcode/fragment scans for canonical replication. Do not replace them with shortened diagnostics when reproducing the paper.

## A branch produces a negative result

This can be expected:

- GSE205117 should terminate before GDIS timing after the suitability gate.
- The SHARE-seq broad-alignment primary null should not make all three metrics significant.
- The motif analysis should preserve zero motifs passing all complete enrichment criteria.

Do not alter thresholds merely to force a positive result.

## Reference values differ

Check the Git commit, Python/dependency versions, `pygdis==1.0.0`, source-file integrity, canonical script choice, and possible upstream-resource changes before editing scientific parameters.

## HPC use

Run from the repository root on a filesystem with sufficient capacity. Long downloads and scans are best run inside a persistent terminal session or cluster job appropriate to local policy.

## Windows

Use the Python runner rather than the Bash wrapper:

```powershell
python run_pipeline.py --branch all
```
