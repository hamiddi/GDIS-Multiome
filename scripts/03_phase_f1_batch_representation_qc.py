#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
03_phase_f1_batch_representation_qc.py
======================================

Project:
Cross-Modal Dynamical Instability Reveals Chromatin Priming
before Transcriptional Cell-Fate Transitions

Phase F1:
Batch and representation quality control for GSE275562.

Scientific question
-------------------
Are the stored RNA-PCA and ATAC-LSI representations suitable for downstream
trajectory/GDIS analysis without being dominated by technical batch effects?

This phase deliberately performs NO pseudotime and NO GDIS.

Primary internal control
------------------------
E15.5 appears in two independent experimental/sample groups. Therefore E15.5
provides a same-developmental-stage bridge for testing technical separation.

Primary beta-lineage states
---------------------------
Ngn3 low -> Ngn3 high -> Fev+ -> Fev+ Beta -> Beta

"Ngn3 high cycling" is excluded from the primary lineage and can be evaluated
later as a sensitivity analysis.

Primary representations
-----------------------
RNA:  stored X_pca
ATAC: stored X_lsi

Primary dimensionality
----------------------
30 dimensions for each modality.

Sensitivity dimensionalities
-----------------------------
20 and 40 dimensions.

Metrics
-------
For each representation, this script reports:

1. Metadata cross-tabs for sample/batch variables.
2. E15.5 beta-lineage cell counts.
3. Biological versus technical partial multivariate R^2:
       - partial R^2 for cell type after controlling technical batch
       - partial R^2 for technical batch after controlling cell type
4. k-nearest-neighbor cross-batch mixing within the SAME cell type at E15.5.
   This avoids calling simple cell-composition differences a batch effect.
5. Silhouette scores for:
       - beta-lineage cell type
       - technical labels
6. Stability of the above metrics across 20, 30, and 40 dimensions.

Interpretation
--------------
The most important quantities are effect sizes, not p-values.

A representation is considered favorable when:
    - biological cell-state structure is preserved;
    - residual technical partial R^2 is small;
    - same-cell-type E15.5 cells mix across experiment/sample groups;
    - conclusions are stable across reasonable dimensionalities.

This script prints results to the terminal only.
It creates NO result files and NO figures.

Run
---
From the CMIL project directory:

    python 03_phase_f1_batch_representation_qc.py

Requirements
------------
    numpy
    pandas
    scipy
    scikit-learn
    mudata
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import mudata as mu

try:
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    raise SystemExit(
        "\nERROR: scikit-learn is required for Phase F1.\n"
        "Install it in the active environment with:\n\n"
        "    conda install -c conda-forge scikit-learn\n"
    )


# =====================================================================
# USER / PROJECT SETTINGS
# =====================================================================

DATA_FILE = Path("data/GSE275562_mudata_with_annotation_all.h5mu")

BETA_LINEAGE = [
    "Ngn3 low",
    "Ngn3 high",
    "Fev+",
    "Fev+ Beta",
    "Beta",
]

# Technical variables already observed in Phase F0.
TECHNICAL_COLUMNS = [
    "sample",
    "experiment_batch",
    "sequencing_batch",
    "batch",
]

PRIMARY_DIMS = 30
SENSITIVITY_DIMS = [20, 30, 40]

# Number of neighbors for local batch-mixing analysis.
K_NEIGHBORS = 30

# To keep silhouette calculations fast and reproducible.
MAX_SILHOUETTE_CELLS = 5000
RANDOM_SEED = 20260920


# =====================================================================
# FORMATTING HELPERS
# =====================================================================

def line(char: str = "=", width: int = 88) -> None:
    print(char * width)


def section(title: str) -> None:
    print()
    line("=")
    print(title)
    line("=")


def subsection(title: str) -> None:
    print()
    line("-")
    print(title)
    line("-")


def print_dataframe(df: pd.DataFrame, float_digits: int = 4) -> None:
    """Print a DataFrame without truncating rows/columns."""
    if df.empty:
        print("(empty)")
        return

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", lambda x: f"{x:.{float_digits}f}",
    ):
        print(df.to_string(index=True))


# =====================================================================
# NUMERICAL HELPERS
# =====================================================================

def as_numpy(matrix) -> np.ndarray:
    """
    Convert an obsm matrix to a dense NumPy array.

    The stored PCA/LSI objects are small low-dimensional matrices, so
    materializing them in memory is safe.
    """
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    return np.asarray(matrix, dtype=np.float64)


def zscore_columns(X: np.ndarray) -> np.ndarray:
    """
    Standardize columns to zero mean and unit variance.

    We use standardized coordinates only for the QC metrics so that one
    component with a numerically larger scale does not dominate Euclidean
    distances or multivariate variance calculations.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)

    # Prevent division by zero for constant dimensions.
    std[std == 0] = 1.0

    return (X - mean) / std


def one_hot(labels: pd.Series) -> np.ndarray:
    """
    One-hot encode a categorical variable while dropping the first level.

    The intercept is added separately in the design matrix.
    """
    encoded = pd.get_dummies(
        labels.astype(str),
        drop_first=True,
        dtype=float,
    )

    if encoded.shape[1] == 0:
        return np.empty((len(labels), 0), dtype=float)

    return encoded.to_numpy(dtype=float)


def design_matrix(*blocks: np.ndarray) -> np.ndarray:
    """Create a regression design matrix with an intercept."""
    n = blocks[0].shape[0] if blocks else 0

    columns = [np.ones((n, 1), dtype=float)]

    for block in blocks:
        if block.shape[1] > 0:
            columns.append(block)

    return np.hstack(columns)


def multivariate_sse(Y: np.ndarray, X: np.ndarray) -> float:
    """
    Residual sum of squares for multivariate ordinary least squares.

    Y has shape n_cells x n_dimensions.
    X is the design matrix.
    """
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    residuals = Y - X @ beta
    return float(np.sum(residuals ** 2))


def partial_r2(
    Y: np.ndarray,
    celltype_labels: pd.Series,
    technical_labels: pd.Series,
) -> tuple[float, float]:
    """
    Calculate partial multivariate R^2 for biology and technical batch.

    Returns
    -------
    biological_partial_r2
        Added explanatory power of cell type after controlling batch.

    technical_partial_r2
        Added explanatory power of batch after controlling cell type.
    """
    cell_block = one_hot(celltype_labels)
    tech_block = one_hot(technical_labels)

    full = design_matrix(cell_block, tech_block)
    cell_only = design_matrix(cell_block)
    tech_only = design_matrix(tech_block)

    sse_full = multivariate_sse(Y, full)
    sse_cell_only = multivariate_sse(Y, cell_only)
    sse_tech_only = multivariate_sse(Y, tech_only)

    # Contribution of technical batch after accounting for cell type.
    if sse_cell_only > 0:
        technical_r2 = (sse_cell_only - sse_full) / sse_cell_only
    else:
        technical_r2 = np.nan

    # Contribution of cell type after accounting for technical batch.
    if sse_tech_only > 0:
        biological_r2 = (sse_tech_only - sse_full) / sse_tech_only
    else:
        biological_r2 = np.nan

    # Tiny negative values can occur from floating-point precision.
    if np.isfinite(technical_r2):
        technical_r2 = max(float(technical_r2), 0.0)

    if np.isfinite(biological_r2):
        biological_r2 = max(float(biological_r2), 0.0)

    return biological_r2, technical_r2


def safe_silhouette(
    X: np.ndarray,
    labels: pd.Series,
    max_cells: int = MAX_SILHOUETTE_CELLS,
) -> float:
    """
    Calculate a reproducible silhouette score.

    Returns NaN when fewer than two valid label groups are present.
    """
    labels = labels.astype(str).reset_index(drop=True)

    if labels.nunique() < 2:
        return np.nan

    # Every class must contain at least two observations for a useful score.
    counts = labels.value_counts()
    valid_labels = counts[counts >= 2].index

    keep = labels.isin(valid_labels).to_numpy()
    X_use = X[keep]
    y_use = labels[keep].to_numpy()

    if len(np.unique(y_use)) < 2:
        return np.nan

    # Reproducible subsampling for large datasets.
    if len(y_use) > max_cells:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(y_use), size=max_cells, replace=False)
        X_use = X_use[idx]
        y_use = y_use[idx]

    return float(
        silhouette_score(
            X_use,
            y_use,
            metric="euclidean",
        )
    )


def local_cross_batch_mixing(
    X: np.ndarray,
    celltypes: pd.Series,
    technical_labels: pd.Series,
    k: int = K_NEIGHBORS,
) -> pd.DataFrame:
    """
    Measure local cross-batch mixing within each cell type.

    For every E15.5 beta-lineage cell type that occurs in >=2 technical groups,
    nearest neighbors are searched ONLY among cells of the same biological
    cell type.

    Metrics
    -------
    observed_cross_batch_fraction
        Mean fraction of a cell's neighbors belonging to a different
        technical group.

    expected_cross_batch_fraction
        Expected fraction under random mixing given the global technical-group
        proportions within that cell type:
            1 - sum(p_b^2)

    normalized_mixing
        observed / expected

        ~1.0 : mixing close to random expectation
         <1  : technical segregation
         >1  : more cross-group mixing than random expectation

    This is a descriptive QC statistic, not a formal hypothesis test.
    """
    records = []

    celltypes = celltypes.astype(str).reset_index(drop=True)
    technical_labels = technical_labels.astype(str).reset_index(drop=True)

    for cell_type in sorted(celltypes.unique()):
        mask = (celltypes == cell_type).to_numpy()

        X_ct = X[mask]
        batch_ct = technical_labels[mask].reset_index(drop=True)

        n = len(batch_ct)
        n_batches = batch_ct.nunique()

        if n < 10 or n_batches < 2:
            continue

        # Need at least one non-self neighbor.
        k_use = min(k, n - 1)

        if k_use < 1:
            continue

        nn = NearestNeighbors(
            n_neighbors=k_use + 1,
            metric="euclidean",
            algorithm="auto",
        )
        nn.fit(X_ct)

        neighbor_idx = nn.kneighbors(
            X_ct,
            return_distance=False,
        )[:, 1:]  # remove each cell itself

        labels_array = batch_ct.to_numpy()

        different = np.empty((n, k_use), dtype=float)

        for i in range(n):
            different[i, :] = (
                labels_array[neighbor_idx[i]]
                != labels_array[i]
            )

        observed = float(different.mean())

        proportions = batch_ct.value_counts(normalize=True).to_numpy()
        expected = float(1.0 - np.sum(proportions ** 2))

        normalized = (
            observed / expected
            if expected > 0
            else np.nan
        )

        records.append({
            "cell_type": cell_type,
            "n_cells": n,
            "n_technical_groups": n_batches,
            "observed_cross_batch_fraction": observed,
            "expected_cross_batch_fraction": expected,
            "normalized_mixing": normalized,
        })

    return pd.DataFrame.from_records(records)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Calculate a weighted mean while ignoring non-finite entries."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if not np.any(valid):
        return np.nan

    return float(np.average(values[valid], weights=weights[valid]))


def qc_label(
    biological_r2: float,
    technical_r2: float,
    normalized_mixing: float,
) -> str:
    """
    Convert effect-size diagnostics into a transparent QC label.

    These are pragmatic feasibility thresholds, not inferential cutoffs.

    PASS:
        technical partial R^2 < 0.05
        biology exceeds technical effect
        normalized mixing >= 0.70

    WARN:
        technical partial R^2 < 0.10
        biology exceeds technical effect
        normalized mixing >= 0.50

    REVIEW:
        anything more concerning
    """
    if not np.isfinite(technical_r2):
        return "REVIEW"

    biology_dominates = (
        np.isfinite(biological_r2)
        and biological_r2 > technical_r2
    )

    mixing_ok = (
        np.isfinite(normalized_mixing)
        and normalized_mixing >= 0.70
    )

    mixing_warn = (
        np.isfinite(normalized_mixing)
        and normalized_mixing >= 0.50
    )

    if (
        technical_r2 < 0.05
        and biology_dominates
        and mixing_ok
    ):
        return "PASS"

    if (
        technical_r2 < 0.10
        and biology_dominates
        and mixing_warn
    ):
        return "WARN"

    return "REVIEW"


# =====================================================================
# REPRESENTATION QC
# =====================================================================

def evaluate_representation(
    representation_name: str,
    X_full: np.ndarray,
    obs: pd.DataFrame,
    e155_beta_mask: np.ndarray,
    technical_columns: list[str],
) -> pd.DataFrame:
    """
    Evaluate one representation over 20/30/40 dimensions.

    Analysis is restricted to E15.5 beta-lineage cells so developmental stage
    itself cannot explain the observed technical-group separation.
    """
    results = []

    obs_e155 = obs.loc[e155_beta_mask].copy().reset_index(drop=True)
    X_e155_full = X_full[e155_beta_mask]

    for n_dims in SENSITIVITY_DIMS:
        if X_e155_full.shape[1] < n_dims:
            continue

        X = X_e155_full[:, :n_dims]
        X = zscore_columns(X)

        biological_silhouette = safe_silhouette(
            X,
            obs_e155["cell_type"],
        )

        for technical_col in technical_columns:
            labels = obs_e155[technical_col].astype(str)

            if labels.nunique() < 2:
                continue

            biological_r2, technical_r2 = partial_r2(
                Y=X,
                celltype_labels=obs_e155["cell_type"],
                technical_labels=labels,
            )

            technical_silhouette = safe_silhouette(
                X,
                labels,
            )

            mixing_table = local_cross_batch_mixing(
                X=X,
                celltypes=obs_e155["cell_type"],
                technical_labels=labels,
                k=K_NEIGHBORS,
            )

            if mixing_table.empty:
                weighted_mixing = np.nan
            else:
                weighted_mixing = weighted_mean(
                    mixing_table["normalized_mixing"],
                    mixing_table["n_cells"],
                )

            results.append({
                "representation": representation_name,
                "dimensions": n_dims,
                "technical_variable": technical_col,
                "n_cells": len(obs_e155),
                "n_technical_groups": labels.nunique(),
                "biological_partial_r2": biological_r2,
                "technical_partial_r2": technical_r2,
                "biology_to_technical_ratio": (
                    biological_r2 / technical_r2
                    if technical_r2 > 0
                    else np.inf
                ),
                "celltype_silhouette": biological_silhouette,
                "technical_silhouette": technical_silhouette,
                "weighted_normalized_mixing": weighted_mixing,
                "qc": qc_label(
                    biological_r2,
                    technical_r2,
                    weighted_mixing,
                ),
            })

    return pd.DataFrame(results)


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    section("PHASE F1 — BATCH AND REPRESENTATION QC")

    print(f"Dataset: {DATA_FILE.resolve()}")
    print()
    print("No pseudotime will be calculated.")
    print("No GDIS will be calculated.")
    print("No output files will be created.")

    # -----------------------------------------------------------------
    # 1. Load dataset.
    # -----------------------------------------------------------------

    if not DATA_FILE.exists():
        print()
        print("ERROR: Dataset file not found.")
        print(DATA_FILE.resolve())
        sys.exit(1)

    section("1. LOAD DATASET")

    print("Opening H5MU in read-only backed mode...")

    try:
        mdata = mu.read_h5mu(DATA_FILE, backed="r")
    except TypeError:
        print(
            "WARNING: This mudata version does not accept backed='r'. "
            "Falling back to standard loading."
        )
        mdata = mu.read_h5mu(DATA_FILE)

    if "rna" not in mdata.mod or "atac" not in mdata.mod:
        print("ERROR: RNA and ATAC modalities are required.")
        sys.exit(1)

    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]
    obs = rna.obs.copy()

    print(f"Cells:         {mdata.n_obs:,}")
    print(f"RNA features:  {rna.n_vars:,}")
    print(f"ATAC features: {atac.n_vars:,}")

    # -----------------------------------------------------------------
    # 2. Confirm required metadata.
    # -----------------------------------------------------------------

    section("2. REQUIRED METADATA")

    required_columns = ["stage", "cell_type"]
    missing_required = [
        col for col in required_columns
        if col not in obs.columns
    ]

    if missing_required:
        print(
            "ERROR: Required metadata missing: "
            + ", ".join(missing_required)
        )
        sys.exit(1)

    technical_columns = [
        col for col in TECHNICAL_COLUMNS
        if col in obs.columns
    ]

    print("Biological metadata:")
    print("  stage")
    print("  cell_type")

    print("\nTechnical metadata detected:")
    for col in technical_columns:
        print(
            f"  {col:<20} "
            f"{obs[col].nunique(dropna=True)} levels"
        )

    if not technical_columns:
        print("ERROR: No technical batch/sample variables were found.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # 3. Cross-tab technical variables against developmental stage.
    # -----------------------------------------------------------------

    section("3. STAGE × TECHNICAL-VARIABLE STRUCTURE")

    for col in technical_columns:
        subsection(f"stage × {col}")

        table = pd.crosstab(
            obs["stage"].astype(str),
            obs[col].astype(str),
            margins=True,
        )

        print_dataframe(table, float_digits=0)

    # -----------------------------------------------------------------
    # 4. Define primary E15.5 beta-lineage control population.
    # -----------------------------------------------------------------

    section("4. E15.5 PRIMARY BETA-LINEAGE CONTROL")

    beta_mask = obs["cell_type"].astype(str).isin(BETA_LINEAGE)
    e155_mask = obs["stage"].astype(str) == "E15.5"

    e155_beta_mask = (beta_mask & e155_mask).to_numpy()

    obs_e155 = obs.loc[e155_beta_mask].copy()

    print(f"E15.5 beta-lineage cells: {len(obs_e155):,}")
    print()

    beta_counts = pd.crosstab(
        obs_e155["cell_type"].astype(str),
        columns="n_cells",
    )

    print("Cell-state counts:")
    print_dataframe(beta_counts, float_digits=0)

    print("\nTechnical-group counts within E15.5 beta lineage:")

    for col in technical_columns:
        counts = (
            obs_e155[col]
            .astype(str)
            .value_counts()
            .rename("n_cells")
            .to_frame()
        )

        print(f"\n{col}:")
        print_dataframe(counts, float_digits=0)

    # -----------------------------------------------------------------
    # 5. Verify stored representations.
    # -----------------------------------------------------------------

    section("5. STORED REPRESENTATIONS")

    if "X_pca" not in rna.obsm:
        print("ERROR: RNA X_pca was not found.")
        sys.exit(1)

    if "X_lsi" not in atac.obsm:
        print("ERROR: ATAC X_lsi was not found.")
        sys.exit(1)

    X_rna = as_numpy(rna.obsm["X_pca"])
    X_atac = as_numpy(atac.obsm["X_lsi"])

    print(f"RNA X_pca shape:  {X_rna.shape}")
    print(f"ATAC X_lsi shape: {X_atac.shape}")

    if X_rna.shape[0] != len(obs) or X_atac.shape[0] != len(obs):
        print("ERROR: Representation rows do not match observation rows.")
        sys.exit(1)

    if X_rna.shape[1] < max(SENSITIVITY_DIMS):
        print(
            f"ERROR: RNA PCA has fewer than {max(SENSITIVITY_DIMS)} dimensions."
        )
        sys.exit(1)

    if X_atac.shape[1] < max(SENSITIVITY_DIMS):
        print(
            f"ERROR: ATAC LSI has fewer than {max(SENSITIVITY_DIMS)} dimensions."
        )
        sys.exit(1)

    print(
        f"\nPrimary QC dimensionality: {PRIMARY_DIMS}"
    )
    print(
        "Sensitivity dimensionalities: "
        + ", ".join(map(str, SENSITIVITY_DIMS))
    )

    # -----------------------------------------------------------------
    # 6. Quantitative QC.
    # -----------------------------------------------------------------

    section("6. QUANTITATIVE BATCH / BIOLOGY QC")

    print(
        "Calculating partial multivariate R², silhouette scores, "
        "and within-cell-type kNN batch mixing..."
    )

    rna_results = evaluate_representation(
        representation_name="RNA_PCA",
        X_full=X_rna,
        obs=obs,
        e155_beta_mask=e155_beta_mask,
        technical_columns=technical_columns,
    )

    atac_results = evaluate_representation(
        representation_name="ATAC_LSI",
        X_full=X_atac,
        obs=obs,
        e155_beta_mask=e155_beta_mask,
        technical_columns=technical_columns,
    )

    results = pd.concat(
        [rna_results, atac_results],
        ignore_index=True,
    )

    if results.empty:
        print("ERROR: No quantitative QC results could be computed.")
        sys.exit(1)

    # Show all dimensions for robustness.
    display_cols = [
        "representation",
        "dimensions",
        "technical_variable",
        "n_technical_groups",
        "biological_partial_r2",
        "technical_partial_r2",
        "biology_to_technical_ratio",
        "celltype_silhouette",
        "technical_silhouette",
        "weighted_normalized_mixing",
        "qc",
    ]

    print_dataframe(
        results[display_cols].set_index(
            ["representation", "dimensions", "technical_variable"]
        )
    )

    # -----------------------------------------------------------------
    # 7. Detailed E15.5 mixing for primary dimensionality.
    # -----------------------------------------------------------------

    section(
        f"7. WITHIN-CELL-TYPE E15.5 BATCH MIXING "
        f"({PRIMARY_DIMS} DIMENSIONS)"
    )

    for representation_name, X_full in [
        ("RNA_PCA", X_rna),
        ("ATAC_LSI", X_atac),
    ]:
        subsection(representation_name)

        X = X_full[e155_beta_mask, :PRIMARY_DIMS]
        X = zscore_columns(X)

        obs_local = obs.loc[e155_beta_mask].copy().reset_index(drop=True)

        for technical_col in technical_columns:
            labels = obs_local[technical_col].astype(str)

            if labels.nunique() < 2:
                continue

            print(f"\nTechnical variable: {technical_col}")

            mixing = local_cross_batch_mixing(
                X=X,
                celltypes=obs_local["cell_type"],
                technical_labels=labels,
                k=K_NEIGHBORS,
            )

            if mixing.empty:
                print(
                    "No cell type had enough cells across >=2 groups "
                    "for mixing analysis."
                )
            else:
                print_dataframe(
                    mixing.set_index("cell_type")
                )

    # -----------------------------------------------------------------
    # 8. Primary 30D summary.
    # -----------------------------------------------------------------

    section("8. PRIMARY 30-DIMENSION SUMMARY")

    primary = results.loc[
        results["dimensions"] == PRIMARY_DIMS
    ].copy()

    primary = primary.sort_values(
        ["representation", "technical_variable"]
    )

    print_dataframe(
        primary[display_cols].set_index(
            ["representation", "technical_variable"]
        )
    )

    # -----------------------------------------------------------------
    # 9. Overall Phase F1 decision.
    # -----------------------------------------------------------------

    section("9. PHASE F1 DECISION")

    # experiment_batch is scientifically the primary technical variable
    # because Phase F0 confirmed two experimental groups.
    decision_variable = (
        "experiment_batch"
        if "experiment_batch" in technical_columns
        else technical_columns[0]
    )

    primary_decision = primary.loc[
        primary["technical_variable"] == decision_variable
    ].copy()

    if primary_decision.empty:
        print(
            "REVIEW REQUIRED: Could not evaluate the primary "
            f"technical variable '{decision_variable}'."
        )
        sys.exit(1)

    print(
        f"Primary technical variable for the decision: "
        f"{decision_variable}"
    )
    print()

    decisions = {}

    for representation in ["RNA_PCA", "ATAC_LSI"]:
        row = primary_decision.loc[
            primary_decision["representation"] == representation
        ]

        if row.empty:
            decisions[representation] = "REVIEW"
            continue

        row = row.iloc[0]
        decisions[representation] = row["qc"]

        print(f"{representation}")
        print(
            f"  biological partial R²:   "
            f"{row['biological_partial_r2']:.4f}"
        )
        print(
            f"  technical partial R²:    "
            f"{row['technical_partial_r2']:.4f}"
        )
        print(
            f"  biology/technical ratio: "
            f"{row['biology_to_technical_ratio']:.2f}"
        )
        print(
            f"  normalized mixing:       "
            f"{row['weighted_normalized_mixing']:.4f}"
        )
        print(
            f"  QC status:               "
            f"{row['qc']}"
        )
        print()

    if all(status == "PASS" for status in decisions.values()):
        verdict = "GO"
        explanation = (
            "Both stored representations pass the prespecified primary "
            "batch/biology QC criteria."
        )

    elif all(status in {"PASS", "WARN"} for status in decisions.values()):
        verdict = "GO WITH CAUTION"
        explanation = (
            "No representation triggered a REVIEW flag, but at least one "
            "shows a non-negligible technical effect that must remain under "
            "explicit sensitivity control."
        )

    else:
        verdict = "REVIEW REQUIRED"
        explanation = (
            "At least one representation shows technical structure large "
            "enough to require additional correction or representation "
            "testing before constructing a common trajectory."
        )

    print(f"PHASE F1 VERDICT: {verdict}")
    print()
    print(explanation)
    print()

    # Dimensionality stability check.
    print("Dimensionality stability:")

    for representation in ["RNA_PCA", "ATAC_LSI"]:
        subset = results.loc[
            (results["representation"] == representation)
            & (results["technical_variable"] == decision_variable)
        ]

        statuses = {
            int(row["dimensions"]): row["qc"]
            for _, row in subset.iterrows()
        }

        print(f"  {representation}: {statuses}")

    print()
    print("No pseudotime was calculated.")
    print("No GDIS was calculated.")
    print("No output files were created.")
    print()
    print(
        "If Phase F1 is acceptable, the next step is "
        "04_phase_f2_common_trajectory.py."
    )
    line("=")


if __name__ == "__main__":
    main()

