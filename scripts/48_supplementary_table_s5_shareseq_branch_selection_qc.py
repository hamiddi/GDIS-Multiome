#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
48_supplementary_table_s5_shareseq_branch_selection_qc.py
=========================================================

Create Supplementary Table S5 for the GDIS-Multiome manuscript:

    Supplementary Table S5.
    RNA-only branch-selection quality-control summary for the
    SHARE-seq TAC-derived hair-follicle trajectories.

PURPOSE
-------
This table documents that TAC-1 -> TAC-2 -> IRS was selected from RNA topology
before pseudotime, GDIS, CMIL, or RNA-ATAC lead/lag timing was evaluated.

NO NEW BRANCH SELECTION IS PERFORMED.

Part A reports the frozen F11 RNA primary-view continuity metrics for the three
prespecified TAC-derived candidate branches:
    - TAC-1 -> TAC-2 -> IRS
    - TAC-1 -> TAC-2 -> Medulla
    - TAC-1 -> TAC-2 -> Hair Shaft-cuticle/cortex

Part B summarizes continuity stability across all F11 RNA sensitivity runs
(10/20/30/40/50 dimensions x k=20/30/50).

Part C records the chronology/guardrails from the frozen F11/F12 manifests,
including that the F12 branch-selection basis was explicitly RNA-only and that
pseudotime, GDIS, CMIL, and lead/lag analysis had not been performed.

INPUTS
------
data/GSE140203/representations_f11/
    f11_branch_continuity.tsv.gz
    f11_manifest.json

data/GSE140203/representations_f12/
    f12_manifest.json

OUTPUTS
-------
data/GSE140203/tables_supplementary/
    Supplementary_Table_S5A_RNA_primary_branch_QC.tsv
    Supplementary_Table_S5B_RNA_branch_sensitivity.tsv
    Supplementary_Table_S5C_selection_chronology.tsv
    Supplementary_Table_S5_combined.md
    Supplementary_Table_S5_validation.tsv

RUN
---
python 48_supplementary_table_s5_shareseq_branch_selection_qc.py
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

DATA_DIR = Path("data") / "GSE140203"
F11_DIR = DATA_DIR / "representations_f11"
F12_DIR = DATA_DIR / "representations_f12"
OUT_DIR = DATA_DIR / "tables_supplementary"

F11_CONTINUITY = F11_DIR / "f11_branch_continuity.tsv.gz"
F11_MANIFEST = F11_DIR / "f11_manifest.json"
F12_MANIFEST = F12_DIR / "f12_manifest.json"

OUTPUT_A = OUT_DIR / "Supplementary_Table_S5A_RNA_primary_branch_QC.tsv"
OUTPUT_B = OUT_DIR / "Supplementary_Table_S5B_RNA_branch_sensitivity.tsv"
OUTPUT_C = OUT_DIR / "Supplementary_Table_S5C_selection_chronology.tsv"
OUTPUT_MD = OUT_DIR / "Supplementary_Table_S5_combined.md"
OUTPUT_VALIDATION = OUT_DIR / "Supplementary_Table_S5_validation.tsv"


# =====================================================================
# FROZEN F11 PRIMARY VIEW
# =====================================================================

RNA_MODALITY = "RNA_PCA"
PRIMARY_DIMS = 10
PRIMARY_K = 30

BRANCH_ORDER = ["IRS", "Medulla", "CuticleCortex"]

BRANCH_LABELS = {
    "IRS": "TAC-1 → TAC-2 → IRS",
    "Medulla": "TAC-1 → TAC-2 → Medulla",
    "CuticleCortex": "TAC-1 → TAC-2 → Hair Shaft cuticle/cortex",
}

SELECTED_BRANCH = "IRS"


# =====================================================================
# HELPERS
# =====================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen input not found:\n{path}")


def load_json(path: Path) -> dict:
    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt6(x) -> str:
    if pd.isna(x):
        return "—"
    return f"{float(x):.6f}"


def fmt3(x) -> str:
    if pd.isna(x):
        return "—"
    return f"{float(x):.3f}"


def yes_no(x) -> str:
    return "Yes" if bool(x) else "No"


def markdown_table(df: pd.DataFrame) -> str:
    d = df.copy().astype(str)
    cols = list(d.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in d.iterrows():
        vals = [str(row[c]).replace("|", r"\|") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# =====================================================================
# LOAD
# =====================================================================

def load_inputs():
    require_file(F11_CONTINUITY)
    continuity = pd.read_csv(
        F11_CONTINUITY,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )
    f11_manifest = load_json(F11_MANIFEST)
    f12_manifest = load_json(F12_MANIFEST)
    return continuity, f11_manifest, f12_manifest


# =====================================================================
# PART A — PRIMARY RNA BRANCH QC
# =====================================================================

def build_part_a(continuity: pd.DataFrame) -> pd.DataFrame:
    required = [
        "branch",
        "terminal_state",
        "modality",
        "n_dims",
        "k",
        "n_cells",
        "connected_components",
        "cross_state_neighbor_fraction",
        "adjacent_fraction_of_cross_edges",
        "bypass_fraction_of_cross_edges",
        "min_required_bridge_fraction",
        "endpoint_distance_is_largest",
        "centroid_chain_ratio",
        "silhouette_state_labels",
    ]
    missing = [c for c in required if c not in continuity.columns]
    if missing:
        raise RuntimeError(
            "F11 continuity table is missing required column(s): "
            + ", ".join(missing)
        )

    primary = continuity.loc[
        continuity["modality"].astype(str).eq(RNA_MODALITY)
        & pd.to_numeric(continuity["n_dims"], errors="coerce").eq(PRIMARY_DIMS)
        & pd.to_numeric(continuity["k"], errors="coerce").eq(PRIMARY_K)
    ].copy()

    if set(primary["branch"].astype(str)) != set(BRANCH_ORDER):
        raise RuntimeError(
            "F11 RNA primary view does not contain exactly the three "
            "prespecified branches."
        )

    primary["branch_order"] = (
        primary["branch"].astype(str)
        .map({b: i for i, b in enumerate(BRANCH_ORDER)})
    )
    primary = primary.sort_values("branch_order").reset_index(drop=True)

    out = pd.DataFrame({
        "Candidate branch": primary["branch"].astype(str).map(BRANCH_LABELS),
        "Cells": pd.to_numeric(primary["n_cells"], errors="raise").astype(int),
        "Connected components": (
            pd.to_numeric(primary["connected_components"], errors="raise")
            .astype(int)
        ),
        "Cross-state neighbor fraction": (
            primary["cross_state_neighbor_fraction"].map(fmt6)
        ),
        "Adjacent fraction of cross-state edges": (
            primary["adjacent_fraction_of_cross_edges"].map(fmt6)
        ),
        "Bypass fraction of cross-state edges": (
            primary["bypass_fraction_of_cross_edges"].map(fmt6)
        ),
        "Minimum required bridge fraction": (
            primary["min_required_bridge_fraction"].map(fmt6)
        ),
        "Endpoint distance largest": (
            primary["endpoint_distance_is_largest"].map(yes_no)
        ),
        "Centroid chain ratio": primary["centroid_chain_ratio"].map(fmt6),
        "State-label silhouette": primary["silhouette_state_labels"].map(fmt6),
        "Frozen for downstream validation": [
            "Yes" if b == SELECTED_BRANCH else "No"
            for b in primary["branch"].astype(str)
        ],
    })

    return out


# =====================================================================
# PART B — RNA SENSITIVITY SUMMARY
# =====================================================================

def build_part_b(continuity: pd.DataFrame) -> pd.DataFrame:
    rna = continuity.loc[
        continuity["modality"].astype(str).eq(RNA_MODALITY)
    ].copy()

    # Exactly 5 dimensions x 3 neighborhood sizes = 15 runs per branch.
    grouped_rows = []

    for branch in BRANCH_ORDER:
        sub = rna.loc[rna["branch"].astype(str).eq(branch)].copy()

        if sub.empty:
            raise RuntimeError(f"No RNA sensitivity rows found for {branch}.")

        grouped_rows.append({
            "Candidate branch": BRANCH_LABELS[branch],
            "Sensitivity runs": int(len(sub)),
            "Max connected components": int(
                pd.to_numeric(
                    sub["connected_components"], errors="raise"
                ).max()
            ),
            "Endpoint-geometry pass fraction": fmt3(
                sub["endpoint_distance_is_largest"].astype(bool).mean()
            ),
            "Median cross-state neighbor fraction": fmt6(
                pd.to_numeric(
                    sub["cross_state_neighbor_fraction"], errors="coerce"
                ).median()
            ),
            "Median adjacent fraction": fmt6(
                pd.to_numeric(
                    sub["adjacent_fraction_of_cross_edges"], errors="coerce"
                ).median()
            ),
            "Median bypass fraction": fmt6(
                pd.to_numeric(
                    sub["bypass_fraction_of_cross_edges"], errors="coerce"
                ).median()
            ),
            "Minimum bridge support across runs": fmt6(
                pd.to_numeric(
                    sub["min_required_bridge_fraction"], errors="coerce"
                ).min()
            ),
            "Median bridge support": fmt6(
                pd.to_numeric(
                    sub["min_required_bridge_fraction"], errors="coerce"
                ).median()
            ),
            "Median centroid chain ratio": fmt6(
                pd.to_numeric(
                    sub["centroid_chain_ratio"], errors="coerce"
                ).median()
            ),
        })

    return pd.DataFrame(grouped_rows)


# =====================================================================
# PART C — SELECTION CHRONOLOGY / GUARDRAILS
# =====================================================================

def build_part_c(f11_manifest: dict, f12_manifest: dict) -> pd.DataFrame:
    f11_guard = f11_manifest.get("guardrails", {})
    f12_guard = f12_manifest.get("guardrails", {})
    f12_branch = f12_manifest.get("branch", {})

    rows = [
        {
            "Phase": "F11",
            "Item": "Branch continuity screen",
            "Frozen record": "RNA topology screened before branch freeze",
        },
        {
            "Phase": "F11",
            "Item": "Pseudotime calculated",
            "Frozen record": yes_no(
                f11_guard.get("pseudotime_calculated", False)
            ),
        },
        {
            "Phase": "F11",
            "Item": "GDIS calculated",
            "Frozen record": yes_no(
                f11_guard.get("gdis_calculated", False)
            ),
        },
        {
            "Phase": "F11",
            "Item": "CMIL calculated",
            "Frozen record": yes_no(
                f11_guard.get("cmil_calculated", False)
            ),
        },
        {
            "Phase": "F11",
            "Item": "Lead/lag tested",
            "Frozen record": yes_no(
                f11_guard.get("lead_lag_tested", False)
            ),
        },
        {
            "Phase": "F12",
            "Item": "Frozen branch",
            "Frozen record": "TAC-1 → TAC-2 → IRS",
        },
        {
            "Phase": "F12",
            "Item": "Selection basis",
            "Frozen record": str(
                f12_branch.get(
                    "selection_basis",
                    "F11 RNA-only continuity screen; ATAC not used for branch choice",
                )
            ),
        },
        {
            "Phase": "F12",
            "Item": "Pseudotime calculated",
            "Frozen record": yes_no(
                f12_guard.get("pseudotime_calculated", False)
            ),
        },
        {
            "Phase": "F12",
            "Item": "GDIS calculated",
            "Frozen record": yes_no(
                f12_guard.get("gdis_calculated", False)
            ),
        },
        {
            "Phase": "F12",
            "Item": "CMIL calculated",
            "Frozen record": yes_no(
                f12_guard.get("cmil_calculated", False)
            ),
        },
        {
            "Phase": "F12",
            "Item": "Lead/lag tested",
            "Frozen record": yes_no(
                f12_guard.get("lead_lag_tested", False)
            ),
        },
    ]

    return pd.DataFrame(rows)


# =====================================================================
# VALIDATION
# =====================================================================

def validate(
    continuity: pd.DataFrame,
    f11_manifest: dict,
    f12_manifest: dict,
) -> pd.DataFrame:

    checks = []

    primary = continuity.loc[
        continuity["modality"].astype(str).eq(RNA_MODALITY)
        & pd.to_numeric(continuity["n_dims"], errors="coerce").eq(PRIMARY_DIMS)
        & pd.to_numeric(continuity["k"], errors="coerce").eq(PRIMARY_K)
    ].copy()

    checks.append({
        "check": "Primary F11 RNA view contains all three candidate branches",
        "pass": set(primary["branch"].astype(str)) == set(BRANCH_ORDER),
    })

    rna = continuity.loc[
        continuity["modality"].astype(str).eq(RNA_MODALITY)
    ]
    counts = rna.groupby("branch").size().to_dict()

    checks.append({
        "check": "Each branch has 15 F11 RNA sensitivity runs",
        "pass": all(int(counts.get(b, 0)) == 15 for b in BRANCH_ORDER),
    })

    f11_guard = f11_manifest.get("guardrails", {})
    checks.append({
        "check": "F11 performed no pseudotime/GDIS/CMIL/lead-lag analysis",
        "pass": all(
            not bool(f11_guard.get(key, False))
            for key in [
                "pseudotime_calculated",
                "gdis_calculated",
                "cmil_calculated",
                "lead_lag_tested",
            ]
        ),
    })

    f12_branch = f12_manifest.get("branch", {})
    selection_basis = str(f12_branch.get("selection_basis", "")).lower()

    checks.append({
        "check": "F12 manifest records RNA-only branch-selection basis",
        "pass": (
            "rna-only" in selection_basis
            and "atac not used" in selection_basis
        ),
    })

    states = f12_branch.get("states", [])
    checks.append({
        "check": "F12 frozen branch is TAC-1 -> TAC-2 -> IRS",
        "pass": list(states) == ["TAC-1", "TAC-2", "IRS"],
    })

    f12_guard = f12_manifest.get("guardrails", {})
    checks.append({
        "check": "F12 still performed no pseudotime/GDIS/CMIL/lead-lag analysis",
        "pass": all(
            not bool(f12_guard.get(key, False))
            for key in [
                "pseudotime_calculated",
                "gdis_calculated",
                "cmil_calculated",
                "lead_lag_tested",
            ]
        ),
    })

    return pd.DataFrame(checks)


# =====================================================================
# MARKDOWN
# =====================================================================

def build_markdown(
    part_a: pd.DataFrame,
    part_b: pd.DataFrame,
    part_c: pd.DataFrame,
) -> str:

    caption = (
        "**Supplementary Table S5. RNA-only branch-selection quality-control "
        "summary for the SHARE-seq TAC-derived hair-follicle trajectories.** "
        "Candidate branches were compared using RNA topology before pseudotime, "
        "GDIS, CMIL, or cross-modal timing analysis. The TAC-1 → TAC-2 → IRS "
        "branch was subsequently frozen for independent validation."
    )

    note = (
        "Part A reports the frozen F11 primary RNA view (10 dimensions, k=30). "
        "Part B summarizes all 15 prespecified RNA representation/neighborhood "
        "sensitivity runs per candidate branch. Part C records the frozen "
        "analysis chronology and confirms that ATAC timing information was not "
        "used to select the IRS branch. These tables are descriptive QC and do "
        "not introduce a new branch-selection score."
    )

    return "\n".join([
        caption,
        "",
        "### A. Primary RNA branch-continuity view",
        "",
        markdown_table(part_a),
        "",
        "### B. RNA continuity across prespecified sensitivity settings",
        "",
        markdown_table(part_b),
        "",
        "### C. Branch-selection chronology and timing guardrails",
        "",
        markdown_table(part_c),
        "",
        f"*Note.* {note}",
        "",
    ])


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 112)
    print("SUPPLEMENTARY TABLE S5 — SHARE-seq RNA-ONLY BRANCH-SELECTION QC")
    print("=" * 112)

    continuity, f11_manifest, f12_manifest = load_inputs()

    validation = validate(
        continuity,
        f11_manifest,
        f12_manifest,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    validation.to_csv(
        OUTPUT_VALIDATION,
        sep="\t",
        index=False,
    )

    if not bool(validation["pass"].all()):
        print("\nVALIDATION FAILED:\n")
        print(validation.to_string(index=False))
        raise RuntimeError(
            "Frozen F11/F12 chronology or branch-QC integrity checks failed. "
            "Supplementary Table S5 was not written."
        )

    part_a = build_part_a(continuity)
    part_b = build_part_b(continuity)
    part_c = build_part_c(f11_manifest, f12_manifest)

    part_a.to_csv(OUTPUT_A, sep="\t", index=False)
    part_b.to_csv(OUTPUT_B, sep="\t", index=False)
    part_c.to_csv(OUTPUT_C, sep="\t", index=False)

    markdown = build_markdown(part_a, part_b, part_c)
    OUTPUT_MD.write_text(markdown, encoding="utf-8")

    print("\nPart A — primary RNA branch QC:")
    print(part_a.to_string(index=False))

    print("\nPart B — RNA sensitivity summary:")
    print(part_b.to_string(index=False))

    print("\nPart C — chronology/guardrails:")
    print(part_c.to_string(index=False))

    print("\nAll frozen F11/F12 validation checks passed.")
    print("\nSaved:")
    for path in [
        OUTPUT_A,
        OUTPUT_B,
        OUTPUT_C,
        OUTPUT_MD,
        OUTPUT_VALIDATION,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()

