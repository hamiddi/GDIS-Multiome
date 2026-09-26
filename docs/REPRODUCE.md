# Reproducing the GDIS-Multiome study

## 1. Requirements

Recommended:

- Linux or another Unix-like environment;
- Python 3.12;
- Conda or Mamba;
- internet access to public GEO/UCSC/JASPAR resources;
- at least ~80 GiB free disk space for a full run.

## 2. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd GDIS-Multiome
```

## 3. Create the environment

```bash
conda env create -f config/environment.yml
conda activate gdis_multiome
```

or:

```bash
mamba env create -f config/environment.yml
conda activate gdis_multiome
```

Alternative:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r config/requirements.txt
```

The manuscript analysis requires `pygdis==1.0.0`.

## 4. Validate before downloading data

```bash
python validate_pipeline_package.py
python scripts/00_preflight.py
```

## 5. Run the analysis

Full study:

```bash
python run_pipeline.py --branch all
```

Manuscript core:

```bash
python run_pipeline.py --branch core
```

Individual branches:

```bash
python run_pipeline.py --branch discovery
python run_pipeline.py --branch gse205117
python run_pipeline.py --branch shareseq
```

## 6. Understand scientific termination versus execution failure

A negative scientific result is not an execution error.

- GSE205117 is expected to stop scientifically before GDIS timing because trajectory-suitability criteria are not met.
- The SHARE-seq primary broad-alignment analysis is expected to remain limited/no overall evidence.
- The motif analysis is expected to retain zero motifs meeting all complete enrichment criteria.

## 7. Compare against frozen checkpoints

Use `docs/REFERENCE_RESULTS.md`. Material differences should be investigated before any parameter is changed.

Possible causes include dependency drift, incomplete downloads, altered upstream resources, filesystem corruption, noncanonical scripts, or platform-level numerical differences.

## 8. Archive a formal replication

Recommended:

```bash
python --version
conda env export > replication_environment.yml
git rev-parse HEAD
python validate_pipeline_package.py
python scripts/00_preflight.py
```

Archive the commit hash, exported environment, logs, and relevant generated manifests.
