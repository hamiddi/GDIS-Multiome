#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
04_phase_f1b_representation_rescue_qc.py
========================================

Project:
Cross-Modal Dynamical Instability Reveals Chromatin Priming
before Transcriptional Cell-Fate Transitions

Phase F1b:
Representation rescue and component-level QC after Phase F1 detected
residual experiment structure in the stored ATAC LSI representation.

WHY THIS PHASE EXISTS
---------------------
Phase F1 found:

RNA X_pca:
    biology partial R^2  ~ 0.241
    experiment partial R^2 ~ 0.033
    normalized E15.5 mixing ~ 0.55

ATAC X_lsi:
    biology partial R^2  ~ 0.245
    experiment partial R^2 ~ 0.063
    normalized E15.5 mixing ~ 0.41

Therefore we do NOT yet build pseudotime or calculate GDIS.

This script asks:
1. Is the residual technical structure concentrated in a few PCA/LSI components?
2. Does an alternative modality-specific latent representation improve E15.5
   batch mixing while preserving biological structure?
3. Which representation should be carried forward into trajectory analysis?

Representations tested
----------------------
RNA:
    X_pca
    X_scVI

ATAC:
    X_lsi
    X_poissonvi

Important:
    X_MultiVI is NOT used as a primary modality-specific representation because
    it combines RNA and ATAC information and would blur the modality comparison.
    It can be considered later only as a trajectory sensitivity analysis.

Primary technical variable
--------------------------
experiment_batch

At E15.5, sample, experiment_batch, sequencing_batch, and batch are perfectly
collinear in this dataset. We therefore use experiment_batch as the single
primary technical label.

Primary biological population
-----------------------------
E15.5 cells from:
    Ngn3 low
    Ngn3 high
    Fev+
    Fev+ Beta
    Beta

No pseudotime.
No GDIS.
No output files.

Run
---
    python 04_phase_f1b_representation_rescue_qc.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import mudata as mu

try:
    from scipy.stats import spearmanr
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    raise SystemExit(
        "\nERROR: scipy and scikit-learn are required.\n"
        "Install with:\n\n"
        "    conda install -c conda-forge scipy scikit-learn\n"
    )


# =====================================================================
# SETTINGS
# =====================================================================

DATA_FILE = Path("data/GSE275562_mudata_with_annotation_all.h5mu")

BETA_LINEAGE = [
    "Ngn3 low",
    "Ngn3 high",
    "Fev+",
    "Fev+ Beta",
    "Beta",
]

TECHNICAL_VARIABLE = "experiment_batch"

K_NEIGHBORS = 30
RANDOM_SEED = 20260920
MAX_SILHOUETTE_CELLS = 5000

# Component diagnostics use the full stored representations.
# Representation-level QC uses these dimension grids where possible.
DIMENSION_GRIDS = {
    "RNA_PCA": [10, 20, 30, 40, 50],
    "RNA_scVI": [5, 10],
    "ATAC_LSI": [10, 20, 30, 40, 49],
    "ATAC_poissonVI": [5, 10, 20, 22],
}

# Continuous technical covariates to inspect if present.
RNA_QC_CANDIDATES = [
    "n_counts",
    "log_counts",
    "n_genes",
    "log_genes",
    "mt_frac",
    "rp_frac",
]

ATAC_QC_CANDIDATES = [
    "n_counts",
    "log_counts",
    "n_genes",
    "log_genes",
    "mt_frac",
    "rp_frac",
]


# =====================================================================
# DISPLAY HELPERS
# =====================================================================

def line(char="=", width=94):
    print(char * width)


def section(title):
    print()
    line("=")
    print(title)
    line("=")


def subsection(title):
    print()
    line("-")
    print(title)
    line("-")


def print_df(df, digits=4):
    if df.empty:
        print("(empty)")
        return

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 240,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string(index=True))


# =====================================================================
# NUMERIC HELPERS
# =====================================================================

def as_numpy(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def zscore_columns(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def one_hot(series):
    df = pd.get_dummies(series.astype(str), drop_first=True, dtype=float)
    return df.to_numpy(dtype=float)


def design_matrix(n_rows, *blocks):
    pieces = [np.ones((n_rows, 1), dtype=float)]
    for block in blocks:
        if block.shape[1] > 0:
            pieces.append(block)
    return np.hstack(pieces)


def multivariate_sse(Y, X):
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    residual = Y - X @ beta
    return float(np.sum(residual ** 2))


def partial_r2(Y, biological, technical):
    b = one_hot(biological)
    t = one_hot(technical)
    n = len(biological)

    full = design_matrix(n, b, t)
    biological_only = design_matrix(n, b)
    technical_only = design_matrix(n, t)

    sse_full = multivariate_sse(Y, full)
    sse_bio = multivariate_sse(Y, biological_only)
    sse_tech = multivariate_sse(Y, technical_only)

    tech_r2 = (
        (sse_bio - sse_full) / sse_bio
        if sse_bio > 0 else np.nan
    )

    bio_r2 = (
        (sse_tech - sse_full) / sse_tech
        if sse_tech > 0 else np.nan
    )

    return max(float(bio_r2), 0.0), max(float(tech_r2), 0.0)


def single_component_partial_r2(values, biological, technical):
    """
    Partial R^2 for one latent component.

    Returns biological and technical partial R^2.
    """
    Y = np.asarray(values, dtype=float).reshape(-1, 1)
    return partial_r2(Y, biological, technical)


def safe_silhouette(X, labels):
    labels = labels.astype(str).reset_index(drop=True)

    if labels.nunique() < 2:
        return np.nan

    counts = labels.value_counts()
    valid_groups = counts[counts >= 2].index
    keep = labels.isin(valid_groups).to_numpy()

    X_use = X[keep]
    y_use = labels[keep].to_numpy()

    if len(np.unique(y_use)) < 2:
        return np.nan

    if len(y_use) > MAX_SILHOUETTE_CELLS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(
            len(y_use),
            size=MAX_SILHOUETTE_CELLS,
            replace=False,
        )
        X_use = X_use[idx]
        y_use = y_use[idx]

    return float(silhouette_score(X_use, y_use))


def local_cross_batch_mixing(X, celltypes, batches, k=K_NEIGHBORS):
    """
    Calculate within-cell-type normalized batch mixing.

    normalized_mixing ~ 1:
        observed cross-batch mixing is close to random expectation.

    normalized_mixing < 1:
        batch segregation remains.
    """
    celltypes = celltypes.astype(str).reset_index(drop=True)
    batches = batches.astype(str).reset_index(drop=True)

    rows = []

    for cell_type in sorted(celltypes.unique()):
        mask = (celltypes == cell_type).to_numpy()

        X_ct = X[mask]
        batch_ct = batches[mask].reset_index(drop=True)

        n = len(batch_ct)

        if n < 10 or batch_ct.nunique() < 2:
            continue

        k_use = min(k, n - 1)

        nn = NearestNeighbors(
            n_neighbors=k_use + 1,
            metric="euclidean",
        )
        nn.fit(X_ct)

        indices = nn.kneighbors(
            X_ct,
            return_distance=False,
        )[:, 1:]

        batch_array = batch_ct.to_numpy()

        cross = np.empty((n, k_use), dtype=float)

        for i in range(n):
            cross[i] = (
                batch_array[indices[i]]
                != batch_array[i]
            )

        observed = float(cross.mean())

        p = batch_ct.value_counts(normalize=True).to_numpy()
        expected = float(1.0 - np.sum(p ** 2))

        normalized = (
            observed / expected
            if expected > 0 else np.nan
        )

        rows.append({
            "cell_type": cell_type,
            "n_cells": n,
            "observed_cross_batch": observed,
            "expected_cross_batch": expected,
            "normalized_mixing": normalized,
        })

    return pd.DataFrame(rows)


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    return float(np.average(values[valid], weights=weights[valid]))


def representation_qc(X, obs_local, dims):
    """
    Evaluate one representation using the first `dims` latent coordinates.
    """
    X_use = zscore_columns(X[:, :dims])

    bio_r2, tech_r2 = partial_r2(
        X_use,
        obs_local["cell_type"],
        obs_local[TECHNICAL_VARIABLE],
    )

    bio_sil = safe_silhouette(
        X_use,
        obs_local["cell_type"],
    )

    tech_sil = safe_silhouette(
        X_use,
        obs_local[TECHNICAL_VARIABLE],
    )

    mixing = local_cross_batch_mixing(
        X_use,
        obs_local["cell_type"],
        obs_local[TECHNICAL_VARIABLE],
    )

    if mixing.empty:
        weighted_mix = np.nan
    else:
        weighted_mix = weighted_mean(
            mixing["normalized_mixing"],
            mixing["n_cells"],
        )

    return {
        "dimensions": dims,
        "biological_partial_r2": bio_r2,
        "technical_partial_r2": tech_r2,
        "biology_to_technical_ratio": (
            bio_r2 / tech_r2
            if tech_r2 > 0 else np.inf
        ),
        "celltype_silhouette": bio_sil,
        "technical_silhouette": tech_sil,
        "weighted_normalized_mixing": weighted_mix,
    }


def qc_class(row):
    """
    Conservative representation classification.

    STRONG:
        technical partial R2 < 0.05
        biology/technical ratio >= 4
        mixing >= 0.70

    ACCEPTABLE:
        technical partial R2 < 0.08
        biology/technical ratio >= 3
        mixing >= 0.55

    CAUTION:
        technical partial R2 < 0.10
        biology/technical ratio >= 2
        mixing >= 0.40

    REVIEW:
        otherwise

    These are feasibility thresholds, not statistical significance cutoffs.
    """
    t = row["technical_partial_r2"]
    ratio = row["biology_to_technical_ratio"]
    mix = row["weighted_normalized_mixing"]

    if t < 0.05 and ratio >= 4 and mix >= 0.70:
        return "STRONG"

    if t < 0.08 and ratio >= 3 and mix >= 0.55:
        return "ACCEPTABLE"

    if t < 0.10 and ratio >= 2 and mix >= 0.40:
        return "CAUTION"

    return "REVIEW"


def component_diagnostics(
    X,
    obs_local,
    component_prefix,
    qc_covariates,
):
    """
    Inspect individual latent components for biological, batch, and QC effects.
    """
    rows = []

    biological = obs_local["cell_type"]
    technical = obs_local[TECHNICAL_VARIABLE]

    for j in range(X.shape[1]):
        values = np.asarray(X[:, j], dtype=float)

        bio_r2, tech_r2 = single_component_partial_r2(
            values,
            biological,
            technical,
        )

        row = {
            "component": f"{component_prefix}{j + 1}",
            "biological_partial_r2": bio_r2,
            "technical_partial_r2": tech_r2,
            "tech_minus_bio_r2": tech_r2 - bio_r2,
            "batch_dominant": (
                tech_r2 > 0.05
                and tech_r2 > bio_r2
            ),
        }

        for covariate in qc_covariates:
            if covariate not in obs_local.columns:
                continue

            values_qc = pd.to_numeric(
                obs_local[covariate],
                errors="coerce",
            ).to_numpy(dtype=float)

            valid = np.isfinite(values) & np.isfinite(values_qc)

            if valid.sum() < 10:
                rho = np.nan
            else:
                rho, _ = spearmanr(
                    values[valid],
                    values_qc[valid],
                )

            row[f"rho_{covariate}"] = rho

        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# MAIN
# =====================================================================

def main():

    section("PHASE F1b — REPRESENTATION RESCUE AND COMPONENT QC")

    print(f"Dataset: {DATA_FILE.resolve()}")
    print()
    print("Primary technical variable: experiment_batch")
    print("Analysis population: E15.5 primary beta lineage")
    print()
    print("No pseudotime.")
    print("No GDIS.")
    print("No output files.")

    if not DATA_FILE.exists():
        print("\nERROR: Dataset file not found.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Load.
    # -----------------------------------------------------------------

    section("1. LOAD DATASET")

    try:
        mdata = mu.read_h5mu(DATA_FILE, backed="r")
    except TypeError:
        mdata = mu.read_h5mu(DATA_FILE)

    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]

    obs = rna.obs.copy()

    required = [
        "stage",
        "cell_type",
        TECHNICAL_VARIABLE,
    ]

    missing = [c for c in required if c not in obs.columns]

    if missing:
        print("ERROR: Missing metadata:", ", ".join(missing))
        sys.exit(1)

    # -----------------------------------------------------------------
    # Confirm technical-variable collinearity at E15.5.
    # -----------------------------------------------------------------

    section("2. TECHNICAL-LABEL REDUNDANCY CHECK")

    e155_all = obs["stage"].astype(str) == "E15.5"

    candidate_tech = [
        c for c in [
            "sample",
            "experiment_batch",
            "sequencing_batch",
            "batch",
        ]
        if c in obs.columns
    ]

    table = obs.loc[e155_all, candidate_tech].astype(str).drop_duplicates()

    print("Unique technical-label combinations at E15.5:")
    print_df(table.reset_index(drop=True), digits=0)

    print()
    print(
        "If each experiment_batch maps one-to-one to the other technical "
        "labels, we treat experiment_batch as the single primary variable."
    )

    # -----------------------------------------------------------------
    # Define E15.5 beta-lineage bridge.
    # -----------------------------------------------------------------

    section("3. DEFINE E15.5 BETA-LINEAGE BRIDGE")

    mask = (
        (obs["stage"].astype(str) == "E15.5")
        & obs["cell_type"].astype(str).isin(BETA_LINEAGE)
    )

    obs_local = obs.loc[mask].copy().reset_index(drop=True)

    print(f"Cells: {len(obs_local):,}")

    counts = pd.crosstab(
        obs_local["cell_type"].astype(str),
        obs_local[TECHNICAL_VARIABLE].astype(str),
        margins=True,
    )

    print()
    print_df(counts, digits=0)

    # -----------------------------------------------------------------
    # Retrieve candidate representations.
    # -----------------------------------------------------------------

    section("4. CANDIDATE MODALITY-SPECIFIC REPRESENTATIONS")

    candidates = {}

    if "X_pca" in rna.obsm:
        candidates["RNA_PCA"] = (
            as_numpy(rna.obsm["X_pca"])[mask.to_numpy()],
            "RNA",
        )

    if "X_scVI" in rna.obsm:
        candidates["RNA_scVI"] = (
            as_numpy(rna.obsm["X_scVI"])[mask.to_numpy()],
            "RNA",
        )

    if "X_lsi" in atac.obsm:
        candidates["ATAC_LSI"] = (
            as_numpy(atac.obsm["X_lsi"])[mask.to_numpy()],
            "ATAC",
        )

    if "X_poissonvi" in atac.obsm:
        candidates["ATAC_poissonVI"] = (
            as_numpy(atac.obsm["X_poissonvi"])[mask.to_numpy()],
            "ATAC",
        )

    for name, (X, modality) in candidates.items():
        print(
            f"{name:<18} modality={modality:<4} "
            f"shape={X.shape}"
        )

    required_candidates = {"RNA_PCA", "ATAC_LSI"}

    if not required_candidates.issubset(candidates):
        print("\nERROR: Primary stored representations are missing.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Component diagnostics.
    # -----------------------------------------------------------------

    section("5. COMPONENT-LEVEL DIAGNOSTICS")

    component_tables = {}

    for name in ["RNA_PCA", "ATAC_LSI"]:
        X, modality = candidates[name]

        subsection(name)

        qc_covariates = (
            RNA_QC_CANDIDATES
            if modality == "RNA"
            else ATAC_QC_CANDIDATES
        )

        comp = component_diagnostics(
            X=X,
            obs_local=obs_local,
            component_prefix="PC" if modality == "RNA" else "LSI",
            qc_covariates=qc_covariates,
        )

        component_tables[name] = comp

        # Show components with strongest technical effect first.
        display = comp.sort_values(
            "technical_partial_r2",
            ascending=False,
        ).head(15)

        print(
            "Top 15 components by residual experiment effect "
            "(within E15.5, controlling cell type):"
        )

        print_df(
            display.set_index("component")
        )

        batch_dominant = comp.loc[
            comp["batch_dominant"]
        ]

        print()
        print(
            f"Batch-dominant components under the prespecified rule: "
            f"{len(batch_dominant)}"
        )

        if not batch_dominant.empty:
            print(
                ", ".join(
                    batch_dominant["component"].astype(str)
                )
            )

    # -----------------------------------------------------------------
    # Candidate representation comparison.
    # -----------------------------------------------------------------

    section("6. REPRESENTATION COMPARISON")

    rows = []

    for name, (X, modality) in candidates.items():
        grid = DIMENSION_GRIDS.get(
            name,
            [X.shape[1]],
        )

        for dims in grid:
            if dims > X.shape[1]:
                continue

            result = representation_qc(
                X=X,
                obs_local=obs_local,
                dims=dims,
            )

            result["representation"] = name
            result["modality"] = modality
            rows.append(result)

    results = pd.DataFrame(rows)

    results["qc_class"] = results.apply(
        qc_class,
        axis=1,
    )

    columns = [
        "modality",
        "representation",
        "dimensions",
        "biological_partial_r2",
        "technical_partial_r2",
        "biology_to_technical_ratio",
        "celltype_silhouette",
        "technical_silhouette",
        "weighted_normalized_mixing",
        "qc_class",
    ]

    print_df(
        results[columns].set_index(
            ["modality", "representation", "dimensions"]
        )
    )

    # -----------------------------------------------------------------
    # Optional diagnostic: remove only clearly batch-dominant LSI
    # components identified using the E15.5 bridge.
    # -----------------------------------------------------------------

    section("7. DIAGNOSTIC LSI COMPONENT EXCLUSION")

    X_lsi, _ = candidates["ATAC_LSI"]
    lsi_diag = component_tables["ATAC_LSI"]

    batch_dominant_indices = (
        lsi_diag.index[
            lsi_diag["batch_dominant"]
        ]
        .to_numpy(dtype=int)
    )

    if len(batch_dominant_indices) == 0:
        print(
            "No LSI component met the prespecified batch-dominant rule."
        )
        print(
            "No component-exclusion diagnostic is performed."
        )

    else:
        keep_indices = np.array(
            [
                i for i in range(X_lsi.shape[1])
                if i not in set(batch_dominant_indices)
            ],
            dtype=int,
        )

        X_filtered = X_lsi[:, keep_indices]

        print(
            "Excluded only components with:"
        )
        print(
            "  technical partial R² > 0.05 AND "
            "technical partial R² > biological partial R²"
        )

        print()
        print(
            "Excluded components: "
            + ", ".join(
                lsi_diag.loc[
                    batch_dominant_indices,
                    "component",
                ].astype(str)
            )
        )

        print(
            f"Remaining LSI dimensions: {X_filtered.shape[1]}"
        )

        diagnostic_dims = min(
            30,
            X_filtered.shape[1],
        )

        diagnostic = representation_qc(
            X=X_filtered,
            obs_local=obs_local,
            dims=diagnostic_dims,
        )

        diagnostic["qc_class"] = qc_class(diagnostic)

        diagnostic_df = pd.DataFrame(
            [diagnostic]
        ).set_index("dimensions")

        print()
        print(
            "Diagnostic filtered-LSI QC "
            "(NOT automatically selected as primary):"
        )
        print_df(diagnostic_df)

    # -----------------------------------------------------------------
    # Select best empirical candidate within each modality.
    # -----------------------------------------------------------------

    section("8. EMPIRICAL CANDIDATE SUMMARY")

    rank_map = {
        "STRONG": 0,
        "ACCEPTABLE": 1,
        "CAUTION": 2,
        "REVIEW": 3,
    }

    results["_rank"] = results["qc_class"].map(rank_map)

    # Prefer better QC class, then higher mixing, then lower technical R2.
    selected_rows = []

    for modality in ["RNA", "ATAC"]:
        subset = results.loc[
            results["modality"] == modality
        ].copy()

        subset = subset.sort_values(
            [
                "_rank",
                "weighted_normalized_mixing",
                "technical_partial_r2",
            ],
            ascending=[True, False, True],
        )

        if not subset.empty:
            selected_rows.append(
                subset.iloc[0]
            )

    selected = pd.DataFrame(selected_rows)

    if not selected.empty:
        print_df(
            selected[
                [
                    "modality",
                    "representation",
                    "dimensions",
                    "biological_partial_r2",
                    "technical_partial_r2",
                    "biology_to_technical_ratio",
                    "weighted_normalized_mixing",
                    "qc_class",
                ]
            ].set_index("modality")
        )

    # -----------------------------------------------------------------
    # Final decision.
    # -----------------------------------------------------------------

    section("9. PHASE F1b DECISION")

    atac_selected = selected.loc[
        selected["modality"] == "ATAC"
    ]

    if atac_selected.empty:
        verdict = "REVIEW REQUIRED"
        reason = "No ATAC candidate representation could be evaluated."

    else:
        atac_row = atac_selected.iloc[0]

        if atac_row["qc_class"] in {"STRONG", "ACCEPTABLE"}:
            verdict = "GO"
            reason = (
                "At least one modality-specific ATAC representation "
                "shows acceptable E15.5 batch/biology behavior."
            )

        elif atac_row["qc_class"] == "CAUTION":
            verdict = "GO WITH CAUTION"
            reason = (
                "The best ATAC representation remains imperfect but "
                "improves sufficiently to permit trajectory construction "
                "with explicit representation sensitivity analyses."
            )

        else:
            verdict = "REVIEW REQUIRED"
            reason = (
                "No modality-specific ATAC representation adequately "
                "controls experiment structure. Do not construct the "
                "common trajectory yet."
            )

    print(f"PHASE F1b VERDICT: {verdict}")
    print()
    print(reason)

    print()
    print(
        "Important: the diagnostic removal of batch-dominant LSI "
        "components is NOT automatically adopted as the final representation."
    )
    print(
        "It is used only to determine whether the problem is concentrated "
        "in a small number of dimensions."
    )

    print()
    print("No pseudotime was calculated.")
    print("No GDIS was calculated.")
    print("No output files were created.")

    if verdict.startswith("GO"):
        print()
        print(
            "Next step: 05_phase_f2_common_trajectory.py"
        )

    line("=")


if __name__ == "__main__":
    main()

