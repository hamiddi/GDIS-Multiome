#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
06_phase_f3_gdis_rna_atac.py
=============================

Project:
Cross-Modal Dynamical Instability Reveals Chromatin Priming
before Transcriptional Cell-Fate Transitions

Phase F3:
Compute modality-specific RNA and ATAC GDIS profiles on one shared,
RNA-derived developmental ordering.

Design
------
- Common pseudotime is reconstructed from RNA X_scVI only.
- ATAC never defines pseudotime.
- Same cells, ordering, windows, and 10 latent dimensions for both modalities.
- RNA = X_scVI (10D)
- ATAC = X_poissonvi (10D)

Primary timing readout:
    pyGDIS with transition weight lambda_t = 0.0.
    In the reference formulation this equals the sustained-instability
    functional and avoids modality-specific transition localization.

Secondary reference readout:
    standard pyGDIS transition weight lambda_t = 0.18.

For the standard score both modalities receive the SAME independent
biological critical value:
    midpoint between median Fev+ and Fev+ Beta common pseudotime.

Primary window:
    400 cells / 100-cell step

Sensitivity:
    300 / 75
    500 / 125

This phase does NOT calculate CMIL or bootstrap uncertainty.
It creates no output files.

Run:
    python 06_phase_f3_gdis_rna_atac.py

If needed:
    python -m pip install pygdis
"""

from __future__ import annotations

from pathlib import Path
import importlib.metadata
import sys

import numpy as np
import pandas as pd
import mudata as mu
import anndata as ad
import scanpy as sc
from scipy.stats import spearmanr

try:
    from gdis import GDIS, transition_weight_sensitivity
except ImportError:
    raise SystemExit(
        "\nERROR: pyGDIS is not installed.\n\n"
        "Install it with:\n"
        "    python -m pip install pygdis\n\n"
        "Then rerun this script.\n"
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

RNA_REPRESENTATION = "X_scVI"
ATAC_REPRESENTATION = "X_poissonvi"
N_LATENT_DIMS = 10

DPT_NEIGHBORS = 30
N_DIFFUSION_COMPONENTS = 10
RANDOM_SEED = 20260920

WINDOW_DESIGNS = [
    ("300_75", 300, 75),
    ("400_100", 400, 100),   # primary
    ("500_125", 500, 125),
]

PRIMARY_WINDOW_DESIGN = "400_100"

PRIMARY_TRANSITION_WEIGHT = 0.0
REFERENCE_TRANSITION_WEIGHT = 0.18

COMMON_GRID_POINTS = 300
EPS = 1e-10


# =====================================================================
# DISPLAY
# =====================================================================

def line(char="=", width=100):
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


def print_df(df, digits=5):
    if df.empty:
        print("(empty)")
        return
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 260,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string(index=True))


# =====================================================================
# HELPERS
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


# =====================================================================
# RECONSTRUCT ACCEPTED PHASE-F2 COMMON PSEUDOTIME
# =====================================================================

def choose_root_medoid(X, obs):
    mask = (
        (obs["stage"].astype(str) == "E14.5")
        & (obs["cell_type"].astype(str) == "Ngn3 low")
    ).to_numpy()

    indices = np.flatnonzero(mask)

    if len(indices) == 0:
        raise RuntimeError("No E14.5 Ngn3-low root candidates were found.")

    candidates = X[indices]
    centroid = candidates.mean(axis=0)
    distances = np.linalg.norm(candidates - centroid, axis=1)

    return int(indices[np.argmin(distances)])


def reconstruct_common_pseudotime(X_rna, obs):
    X = zscore_columns(X_rna)

    adata = ad.AnnData(
        X=np.zeros((X.shape[0], 1), dtype=np.float32)
    )

    obs_copy = obs.copy()
    obs_copy.index = obs_copy.index.astype(str)
    adata.obs = obs_copy
    adata.obsm["X_common_rna"] = X.astype(np.float32)

    root_index = choose_root_medoid(X, adata.obs)

    sc.pp.neighbors(
        adata,
        n_neighbors=DPT_NEIGHBORS,
        use_rep="X_common_rna",
        metric="euclidean",
        random_state=RANDOM_SEED,
    )

    sc.tl.diffmap(
        adata,
        n_comps=max(N_DIFFUSION_COMPONENTS + 1, 15),
        random_state=RANDOM_SEED,
    )

    adata.uns["iroot"] = root_index

    sc.tl.dpt(
        adata,
        n_dcs=N_DIFFUSION_COMPONENTS,
    )

    pt = adata.obs["dpt_pseudotime"].to_numpy(dtype=float)
    pt[~np.isfinite(pt)] = np.nan

    return pt


# =====================================================================
# WINDOWS
# =====================================================================

def create_windows(X, pseudotime, window_size, step_size):
    valid = np.isfinite(pseudotime)

    X = X[valid]
    pt = pseudotime[valid]

    order = np.argsort(pt, kind="mergesort")
    X = X[order]
    pt = pt[order]

    trajectories = []
    parameters = []
    metadata = []

    start = 0
    window_id = 0

    while start + window_size <= len(pt):
        stop = start + window_size

        X_window = X[start:stop]
        pt_window = pt[start:stop]

        center = float(np.mean(pt_window))

        trajectories.append(X_window.copy())
        parameters.append(center)

        metadata.append(
            {
                "window_id": window_id,
                "start_rank": start,
                "stop_rank_exclusive": stop,
                "n_cells": window_size,
                "pt_min": float(pt_window[0]),
                "pt_max": float(pt_window[-1]),
                "pt_mean": center,
                "pt_median": float(np.median(pt_window)),
            }
        )

        start += step_size
        window_id += 1

    parameters = np.asarray(parameters, dtype=float)

    if len(trajectories) < 9:
        raise RuntimeError(
            f"Only {len(trajectories)} windows were produced."
        )

    if np.any(np.diff(parameters) <= 0):
        raise RuntimeError(
            "Window-center pseudotimes are not strictly increasing."
        )

    return trajectories, parameters, pd.DataFrame(metadata)


# =====================================================================
# INDEPENDENT BIOLOGICAL LANDMARK
# =====================================================================

def biological_critical_landmark(pseudotime, obs):
    medians = {}

    for state in ["Fev+", "Fev+ Beta"]:
        mask = (obs["cell_type"].astype(str) == state).to_numpy()
        values = pseudotime[mask]
        values = values[np.isfinite(values)]

        if len(values) == 0:
            raise RuntimeError(f"No finite pseudotime for {state}.")

        medians[state] = float(np.median(values))

    midpoint = 0.5 * (medians["Fev+"] + medians["Fev+ Beta"])

    return midpoint, medians["Fev+"], medians["Fev+ Beta"]


# =====================================================================
# GDIS
# =====================================================================

def run_gdis(trajectories, parameters, critical_value):
    model = GDIS(
        transition_weight=REFERENCE_TRANSITION_WEIGHT,
    )

    result = model.fit_transform(
        trajectories,
        parameters,
        critical_value=critical_value,
    )

    rescored = transition_weight_sensitivity(
        result,
        weights=(
            PRIMARY_TRANSITION_WEIGHT,
            REFERENCE_TRANSITION_WEIGHT,
        ),
    )

    primary = (
        rescored.loc[
            np.isclose(
                rescored["transition_weight"],
                PRIMARY_TRANSITION_WEIGHT,
            )
        ]
        .sort_values("parameter")
        .reset_index(drop=True)
    )

    reference = (
        rescored.loc[
            np.isclose(
                rescored["transition_weight"],
                REFERENCE_TRANSITION_WEIGHT,
            )
        ]
        .sort_values("parameter")
        .reset_index(drop=True)
    )

    return result, primary, reference


def validate_scores(values):
    values = np.asarray(values, dtype=float)

    return {
        "all_finite": bool(np.all(np.isfinite(values))),
        "all_bounded": bool(
            np.all((values >= -EPS) & (values < 1.0 + EPS))
        ),
        "dynamic_range": float(np.nanmax(values) - np.nanmin(values)),
    }


def summarize_profile(modality, design_name, primary_df, reference_df):
    p = primary_df["parameter"].to_numpy(dtype=float)
    sustained = primary_df["gdis"].to_numpy(dtype=float)
    reference = reference_df["gdis"].to_numpy(dtype=float)

    i_s = int(np.argmax(sustained))
    i_r = int(np.argmax(reference))

    qc_s = validate_scores(sustained)
    qc_r = validate_scores(reference)

    rho, _ = spearmanr(sustained, reference)

    return {
        "modality": modality,
        "window_design": design_name,
        "n_windows": len(p),

        "sustained_peak_pt": float(p[i_s]),
        "sustained_peak_score": float(sustained[i_s]),
        "sustained_dynamic_range": qc_s["dynamic_range"],

        "reference_peak_pt": float(p[i_r]),
        "reference_peak_score": float(reference[i_r]),
        "reference_dynamic_range": qc_r["dynamic_range"],

        "sustained_vs_reference_rho": float(rho),

        "all_finite": bool(
            qc_s["all_finite"] and qc_r["all_finite"]
        ),
        "all_bounded": bool(
            qc_s["all_bounded"] and qc_r["all_bounded"]
        ),

        "sustained_peak_at_edge": bool(
            i_s == 0 or i_s == len(p) - 1
        ),
    }


# =====================================================================
# CURVE SENSITIVITY
# =====================================================================

def curve_correlation(x1, y1, x2, y2):
    lower = max(float(np.min(x1)), float(np.min(x2)))
    upper = min(float(np.max(x1)), float(np.max(x2)))

    if upper <= lower:
        return np.nan

    grid = np.linspace(lower, upper, COMMON_GRID_POINTS)

    a = np.interp(grid, x1, y1)
    b = np.interp(grid, x2, y2)

    rho, _ = spearmanr(a, b)

    return float(rho)


# =====================================================================
# MAIN
# =====================================================================

def main():
    section("PHASE F3 — RNA AND ATAC GDIS ON ONE COMMON TRAJECTORY")

    print(f"Dataset: {DATA_FILE.resolve()}")

    try:
        version = importlib.metadata.version("pygdis")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"

    print(f"pyGDIS version: {version}")
    print()
    print(f"RNA representation:  {RNA_REPRESENTATION}, {N_LATENT_DIMS}D")
    print(f"ATAC representation: {ATAC_REPRESENTATION}, {N_LATENT_DIMS}D")
    print()
    print("Primary timing profile: lambda_t = 0.0 (sustained instability)")
    print("Secondary reference profile: lambda_t = 0.18")
    print()
    print("No CMIL is calculated.")
    print("No bootstrap is performed.")
    print("No output files are created.")

    if not DATA_FILE.exists():
        print("\nERROR: Dataset file not found.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    section("1. LOAD DATASET")

    try:
        mdata = mu.read_h5mu(DATA_FILE, backed="r")
    except TypeError:
        mdata = mu.read_h5mu(DATA_FILE)

    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]
    obs_full = rna.obs.copy()

    for key in ["stage", "cell_type", "experiment_batch"]:
        if key not in obs_full.columns:
            print(f"ERROR: Required metadata '{key}' is missing.")
            sys.exit(1)

    if RNA_REPRESENTATION not in rna.obsm:
        print(f"ERROR: RNA {RNA_REPRESENTATION} not found.")
        sys.exit(1)

    if ATAC_REPRESENTATION not in atac.obsm:
        print(f"ERROR: ATAC {ATAC_REPRESENTATION} not found.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Beta lineage
    # -----------------------------------------------------------------

    section("2. PRIMARY BETA-LINEAGE DATA")

    beta_mask = (
        obs_full["cell_type"]
        .astype(str)
        .isin(BETA_LINEAGE)
    ).to_numpy()

    obs = obs_full.loc[beta_mask].copy().reset_index(drop=True)
    obs.index = pd.Index([f"beta_{i}" for i in range(len(obs))])

    X_rna = as_numpy(
        rna.obsm[RNA_REPRESENTATION]
    )[beta_mask, :N_LATENT_DIMS]

    X_atac = as_numpy(
        atac.obsm[ATAC_REPRESENTATION]
    )[beta_mask, :N_LATENT_DIMS]

    if X_rna.shape != X_atac.shape:
        print("ERROR: RNA and ATAC matched latent matrices differ in shape.")
        sys.exit(1)

    print(f"Cells: {len(obs):,}")
    print(f"RNA shape:  {X_rna.shape}")
    print(f"ATAC shape: {X_atac.shape}")

    X_rna_z = zscore_columns(X_rna)
    X_atac_z = zscore_columns(X_atac)

    # -----------------------------------------------------------------
    # Common pseudotime
    # -----------------------------------------------------------------

    section("3. RECONSTRUCT COMMON RNA-ONLY PSEUDOTIME")

    pseudotime = reconstruct_common_pseudotime(
        X_rna=X_rna,
        obs=obs,
    )

    finite_fraction = float(np.isfinite(pseudotime).mean())

    rows = []
    for state in BETA_LINEAGE:
        mask = (obs["cell_type"].astype(str) == state).to_numpy()
        values = pseudotime[mask]
        values = values[np.isfinite(values)]
        rows.append(
            {
                "cell_type": state,
                "n_cells": len(values),
                "median_pseudotime": (
                    float(np.median(values))
                    if len(values) else np.nan
                ),
            }
        )

    print(f"Finite pseudotime fraction: {finite_fraction:.6f}")
    print()
    print_df(pd.DataFrame(rows).set_index("cell_type"))

    if finite_fraction < 0.99:
        print("\nERROR: Accepted Phase-F2 trajectory was not reproduced.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Common landmark
    # -----------------------------------------------------------------

    section("4. COMMON BIOLOGICAL CRITICAL LANDMARK")

    critical_value, fev_median, fev_beta_median = (
        biological_critical_landmark(
            pseudotime,
            obs,
        )
    )

    print(f"Median Fev+ pseudotime:      {fev_median:.6f}")
    print(f"Median Fev+ Beta pseudotime: {fev_beta_median:.6f}")
    print(f"Common midpoint landmark:    {critical_value:.6f}")
    print()
    print(
        "The same annotation-defined landmark is supplied to RNA and ATAC "
        "for the lambda_t=0.18 reference score."
    )

    # -----------------------------------------------------------------
    # Compute profiles
    # -----------------------------------------------------------------

    section("5. COMPUTE MODALITY-SPECIFIC GDIS PROFILES")

    profiles = {}
    summaries = []

    for design_name, window_size, step_size in WINDOW_DESIGNS:
        subsection(
            f"{design_name}: window={window_size}, step={step_size}"
        )

        rna_traj, parameters, _ = create_windows(
            X_rna_z,
            pseudotime,
            window_size,
            step_size,
        )

        atac_traj, atac_parameters, _ = create_windows(
            X_atac_z,
            pseudotime,
            window_size,
            step_size,
        )

        if not np.allclose(
            parameters,
            atac_parameters,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                "RNA and ATAC window centers are not identical."
            )

        print(f"Windows: {len(parameters)}")
        print(
            f"Center range: {parameters[0]:.6f} to "
            f"{parameters[-1]:.6f}"
        )

        rna_result, rna_primary, rna_reference = run_gdis(
            rna_traj,
            parameters,
            critical_value,
        )

        atac_result, atac_primary, atac_reference = run_gdis(
            atac_traj,
            parameters,
            critical_value,
        )

        profiles[("RNA", design_name)] = {
            "parameters": parameters.copy(),
            "primary": rna_primary["gdis"].to_numpy(dtype=float),
            "reference": rna_reference["gdis"].to_numpy(dtype=float),
            "result": rna_result,
        }

        profiles[("ATAC", design_name)] = {
            "parameters": parameters.copy(),
            "primary": atac_primary["gdis"].to_numpy(dtype=float),
            "reference": atac_reference["gdis"].to_numpy(dtype=float),
            "result": atac_result,
        }

        summaries.append(
            summarize_profile(
                "RNA",
                design_name,
                rna_primary,
                rna_reference,
            )
        )

        summaries.append(
            summarize_profile(
                "ATAC",
                design_name,
                atac_primary,
                atac_reference,
            )
        )

        print(
            "RNA critical source:  "
            f"{rna_result.metadata.get('critical_value_source')}"
        )
        print(
            "ATAC critical source: "
            f"{atac_result.metadata.get('critical_value_source')}"
        )

    summary = pd.DataFrame(summaries)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    section("6. GDIS PROFILE SUMMARY")

    display_cols = [
        "modality",
        "window_design",
        "n_windows",
        "sustained_peak_pt",
        "sustained_peak_score",
        "sustained_dynamic_range",
        "reference_peak_pt",
        "reference_peak_score",
        "reference_dynamic_range",
        "sustained_vs_reference_rho",
        "all_finite",
        "all_bounded",
        "sustained_peak_at_edge",
    ]

    print_df(
        summary[display_cols].set_index(
            ["modality", "window_design"]
        )
    )

    print()
    print(
        "Peak locations are printed independently for QC only. "
        "No RNA-ATAC peak subtraction is performed here."
    )

    # -----------------------------------------------------------------
    # Top primary windows
    # -----------------------------------------------------------------

    section("7. PRIMARY 400/100 — TOP SUSTAINED-INSTABILITY WINDOWS")

    for modality in ["RNA", "ATAC"]:
        subsection(modality)

        profile = profiles[(modality, PRIMARY_WINDOW_DESIGN)]

        df = pd.DataFrame(
            {
                "window_center_pt": profile["parameters"],
                "sustained_gdis_lambda0": profile["primary"],
                "reference_gdis_lambda018": profile["reference"],
            }
        )

        top = (
            df.sort_values(
                "sustained_gdis_lambda0",
                ascending=False,
            )
            .head(10)
        )

        print_df(top.set_index("window_center_pt"))

    # -----------------------------------------------------------------
    # Sensitivity
    # -----------------------------------------------------------------

    section("8. WITHIN-MODALITY WINDOW-DESIGN SENSITIVITY")

    sensitivity_rows = []

    for modality in ["RNA", "ATAC"]:
        primary_profile = profiles[
            (modality, PRIMARY_WINDOW_DESIGN)
        ]

        primary_row = summary.loc[
            (summary["modality"] == modality)
            & (summary["window_design"] == PRIMARY_WINDOW_DESIGN)
        ].iloc[0]

        for design_name, _, _ in WINDOW_DESIGNS:
            if design_name == PRIMARY_WINDOW_DESIGN:
                continue

            candidate = profiles[(modality, design_name)]

            candidate_row = summary.loc[
                (summary["modality"] == modality)
                & (summary["window_design"] == design_name)
            ].iloc[0]

            rho = curve_correlation(
                primary_profile["parameters"],
                primary_profile["primary"],
                candidate["parameters"],
                candidate["primary"],
            )

            peak_shift = abs(
                float(primary_row["sustained_peak_pt"])
                - float(candidate_row["sustained_peak_pt"])
            )

            sensitivity_rows.append(
                {
                    "modality": modality,
                    "comparison": (
                        f"{PRIMARY_WINDOW_DESIGN} vs {design_name}"
                    ),
                    "curve_spearman_rho": rho,
                    "absolute_peak_shift": peak_shift,
                }
            )

    sensitivity = pd.DataFrame(sensitivity_rows)

    print_df(
        sensitivity.set_index(
            ["modality", "comparison"]
        )
    )

    # -----------------------------------------------------------------
    # Component peaks
    # -----------------------------------------------------------------

    section("9. PRIMARY pyGDIS COMPONENT PEAKS")

    component_names = [
        "jacobian_raw",
        "stretching_raw",
        "expansion_raw",
        "entropy_raw",
        "temporal_raw",
        "transition_energy",
    ]

    component_rows = []

    for modality in ["RNA", "ATAC"]:
        result = profiles[
            (modality, PRIMARY_WINDOW_DESIGN)
        ]["result"]

        for component in component_names:
            if component not in result.components:
                continue

            values = np.asarray(
                result.components[component],
                dtype=float,
            )

            i = int(np.nanargmax(values))

            component_rows.append(
                {
                    "modality": modality,
                    "component": component,
                    "peak_window_pt": float(result.parameters[i]),
                    "peak_value": float(values[i]),
                }
            )

    print_df(
        pd.DataFrame(component_rows).set_index(
            ["modality", "component"]
        )
    )

    # -----------------------------------------------------------------
    # F3 decision
    # -----------------------------------------------------------------

    section("10. PHASE F3 DECISION")

    checks = []

    for modality in ["RNA", "ATAC"]:
        row = summary.loc[
            (summary["modality"] == modality)
            & (summary["window_design"] == PRIMARY_WINDOW_DESIGN)
        ].iloc[0]

        local_sens = sensitivity.loc[
            sensitivity["modality"] == modality
        ]

        criteria = [
            (
                f"{modality}: all scores finite",
                bool(row["all_finite"]),
            ),
            (
                f"{modality}: all scores bounded [0,1)",
                bool(row["all_bounded"]),
            ),
            (
                f"{modality}: sustained peak not at edge",
                not bool(row["sustained_peak_at_edge"]),
            ),
            (
                f"{modality}: sustained dynamic range >0.02",
                float(row["sustained_dynamic_range"]) > 0.02,
            ),
            (
                f"{modality}: sensitivity curve rho >=0.75",
                bool(
                    np.all(
                        local_sens["curve_spearman_rho"]
                        .to_numpy(dtype=float) >= 0.75
                    )
                ),
            ),
            (
                f"{modality}: sensitivity peak shift <=0.10",
                bool(
                    np.all(
                        local_sens["absolute_peak_shift"]
                        .to_numpy(dtype=float) <= 0.10
                    )
                ),
            ),
        ]

        for criterion, passed in criteria:
            checks.append(
                {
                    "criterion": criterion,
                    "pass": passed,
                    "status": "PASS" if passed else "REVIEW",
                }
            )

    checks_df = pd.DataFrame(checks)

    print_df(
        checks_df.set_index("criterion"),
        digits=0,
    )

    n_pass = int(checks_df["pass"].sum())
    n_total = len(checks_df)

    print()
    print(f"Criteria passed: {n_pass}/{n_total}")

    hard_fail = any(
        (
            (
                "finite" in row["criterion"]
                or "bounded" in row["criterion"]
                or "not at edge" in row["criterion"]
            )
            and not row["pass"]
        )
        for _, row in checks_df.iterrows()
    )

    if hard_fail:
        verdict = "REVIEW REQUIRED"
        reason = (
            "At least one modality has an invalid or boundary-dominated "
            "primary GDIS profile. Do not calculate CMIL."
        )
    elif n_pass == n_total:
        verdict = "GO"
        reason = (
            "Both modality-specific sustained-instability profiles are "
            "finite, bounded, internally peaked, non-flat, and stable under "
            "the prespecified window sensitivities."
        )
    elif n_pass >= n_total - 2:
        verdict = "GO WITH CAUTION"
        reason = (
            "Primary GDIS profiles are valid, but one or two sensitivity "
            "criteria require explicit attention in the CMIL/bootstrap phase."
        )
    else:
        verdict = "REVIEW REQUIRED"
        reason = (
            "The RNA/ATAC GDIS profiles are not sufficiently robust to "
            "proceed to cross-modal timing inference."
        )

    print()
    print(f"PHASE F3 VERDICT: {verdict}")
    print()
    print(reason)

    print()
    print("No CMIL was calculated.")
    print("No bootstrap was performed.")
    print("No output files were created.")

    if verdict.startswith("GO"):
        print()
        print("Next step: 07_phase_f4_cmil_bootstrap.py")

    line("=")


if __name__ == "__main__":
    main()

