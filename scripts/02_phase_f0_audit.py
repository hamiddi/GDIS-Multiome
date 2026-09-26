#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
Phase F0 — GSE275562 Multiome Feasibility Audit
================================================

Project:
Cross-Modal Dynamical Instability Reveals Chromatin Priming
before Transcriptional Cell-Fate Transitions

Purpose
-------
This script performs ONLY the initial feasibility audit.

It does NOT:
    - calculate GDIS
    - calculate pseudotime
    - run PCA or LSI
    - perform differential expression/accessibility
    - make any chromatin-priming claim

It checks:
    1. Dataset dimensions
    2. Available modalities
    3. RNA/ATAC cell pairing
    4. Metadata fields
    5. Developmental-stage counts
    6. Cell-type counts
    7. Stage × cell-type counts
    8. Primary beta-lineage counts
    9. Experiment/batch metadata
   10. Existing latent representations

Run:
    python phase_f0_gse275562.py

Expected dataset location:
    data/GSE275562_mudata_with_annotation_all.h5mu
"""

from pathlib import Path
import sys

import pandas as pd
import mudata as mu


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

DATA_FILE = Path(
    "data/GSE275562_mudata_with_annotation_all.h5mu"
)

# Primary beta-cell developmental trajectory for the feasibility study.
# "Ngn3 high cycling" is deliberately excluded from the primary lineage.
BETA_LINEAGE = [
    "Ngn3 low",
    "Ngn3 high",
    "Fev+",
    "Fev+ Beta",
    "Beta",
]

# Candidate names used to locate important metadata columns.
STAGE_CANDIDATES = [
    "stage",
    "developmental_stage",
    "timepoint",
    "time_point",
]

CELLTYPE_CANDIDATES = [
    "cell_type",
    "celltype",
    "annotation",
    "cell_type_refined",
]

BATCH_CANDIDATES = [
    "experiment",
    "batch",
    "sample",
    "sample_id",
    "library",
    "library_id",
    "replicate",
    "orig.ident",
    "orig_ident",
]


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def line(char="=", width=78):
    """Print a separator line."""
    print(char * width)


def section(title):
    """Print a formatted section heading."""
    print()
    line("=")
    print(title)
    line("=")


def normalize_name(name):
    """Normalize metadata column names for robust matching."""
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(obs, candidates):
    """
    Find a metadata column using a list of possible names.

    MuData objects sometimes prefix fields with modality names, so the
    function also checks suffix matches such as 'rna:cell_type'.
    """
    normalized = {
        normalize_name(col): col
        for col in obs.columns
    }

    # Exact normalized match.
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    # Suffix match.
    for candidate in candidates:
        key = normalize_name(candidate)

        matches = []
        for col in obs.columns:
            c = normalize_name(col)

            if (
                c.endswith(":" + key)
                or c.endswith("_" + key)
            ):
                matches.append(col)

        if len(matches) == 1:
            return matches[0]

    return None


def choose_obs_table(mdata):
    """
    Select the most informative observation table.

    The RNA modality is preferred because published moscot examples use
    RNA observation metadata for 'stage' and 'cell_type'.
    """
    if "rna" in mdata.mod:
        rna_obs = mdata.mod["rna"].obs

        stage = find_column(rna_obs, STAGE_CANDIDATES)
        cell_type = find_column(rna_obs, CELLTYPE_CANDIDATES)

        if stage is not None or cell_type is not None:
            return rna_obs, "mdata.mod['rna'].obs"

    return mdata.obs, "mdata.obs"


def print_value_counts(obs, column, title):
    """Print counts for one categorical metadata column."""
    section(title)

    counts = (
        obs[column]
        .astype(str)
        .value_counts(dropna=False)
    )

    print(counts.to_string())
    print()
    print(f"Total: {int(counts.sum()):,}")


# ---------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------

def main():

    section("PHASE F0 — GSE275562 MULTIOME FEASIBILITY AUDIT")

    print(f"Dataset expected at:\n{DATA_FILE.resolve()}")

    # --------------------------------------------------------------
    # 1. Confirm that the dataset exists.
    # --------------------------------------------------------------

    if not DATA_FILE.exists():
        print()
        print("ERROR: Dataset file was not found.")
        print()
        print("Expected file:")
        print(DATA_FILE.resolve())
        print()
        print("Place the H5MU file in the data/ directory and run:")
        print()
        print("    python phase_f0_gse275562.py")
        sys.exit(1)

    file_size_gb = DATA_FILE.stat().st_size / (1024 ** 3)

    print(f"\nFile size: {file_size_gb:.2f} GB")

    # --------------------------------------------------------------
    # 2. Load the MuData object in read-only backed mode.
    # --------------------------------------------------------------

    section("1. LOADING DATASET")

    print("Opening H5MU in read-only backed mode...")

    try:
        mdata = mu.read_h5mu(DATA_FILE, backed="r")
    except TypeError:
        # Compatibility fallback for older mudata releases.
        print(
            "This mudata version does not accept backed='r'; "
            "falling back to standard loading."
        )
        mdata = mu.read_h5mu(DATA_FILE)

    print("Dataset loaded successfully.")

    # --------------------------------------------------------------
    # 3. Dataset-level dimensions.
    # --------------------------------------------------------------

    section("2. DATASET STRUCTURE")

    print(f"Total cells:    {mdata.n_obs:,}")
    print(f"Total features: {mdata.n_vars:,}")

    print("\nModalities:")

    for modality_name, adata in mdata.mod.items():
        print(
            f"  {modality_name:<12}"
            f"{adata.n_obs:>10,} cells   "
            f"{adata.n_vars:>12,} features"
        )

    required_modalities = {"rna", "atac"}
    available_modalities = set(mdata.mod.keys())

    if required_modalities.issubset(available_modalities):
        print("\nPASS: Both RNA and ATAC modalities are present.")
    else:
        print("\nFAIL: Expected RNA and ATAC modalities were not both found.")
        print(f"Available modalities: {sorted(available_modalities)}")
        sys.exit(1)

    # --------------------------------------------------------------
    # 4. RNA/ATAC barcode pairing.
    # --------------------------------------------------------------

    section("3. RNA / ATAC CELL PAIRING")

    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]

    rna_cells = pd.Index(rna.obs_names.astype(str))
    atac_cells = pd.Index(atac.obs_names.astype(str))

    same_number = len(rna_cells) == len(atac_cells)
    same_set = same_number and set(rna_cells) == set(atac_cells)
    same_order = same_number and rna_cells.equals(atac_cells)

    print(f"RNA cells:                       {len(rna_cells):,}")
    print(f"ATAC cells:                      {len(atac_cells):,}")
    print(f"Same number of cells:            {same_number}")
    print(f"Same barcode set:                {same_set}")
    print(f"Same barcode order:              {same_order}")
    print(f"RNA barcodes unique:             {rna_cells.is_unique}")
    print(f"ATAC barcodes unique:            {atac_cells.is_unique}")

    if (
        same_number
        and same_set
        and same_order
        and rna_cells.is_unique
        and atac_cells.is_unique
    ):
        print("\nPASS: RNA and ATAC are paired cell-by-cell.")
    else:
        print()
        print("FAIL: RNA/ATAC cell pairing is not clean.")
        print("Do not proceed to Phase F1 until this is resolved.")
        sys.exit(1)

    # --------------------------------------------------------------
    # 5. Metadata audit.
    # --------------------------------------------------------------

    obs, obs_source = choose_obs_table(mdata)

    section("4. OBSERVATION METADATA")

    print(f"Metadata source: {obs_source}")
    print(f"Number of metadata columns: {len(obs.columns)}")
    print()

    for col in obs.columns:
        n_unique = obs[col].nunique(dropna=True)

        print(
            f"{str(col):<45}"
            f"unique={n_unique:>6}"
            f"   dtype={str(obs[col].dtype)}"
        )

    stage_col = find_column(obs, STAGE_CANDIDATES)
    celltype_col = find_column(obs, CELLTYPE_CANDIDATES)
    batch_col = find_column(obs, BATCH_CANDIDATES)

    print()
    print(f"Detected stage column:     {stage_col}")
    print(f"Detected cell-type column: {celltype_col}")
    print(f"Detected batch column:     {batch_col}")

    # --------------------------------------------------------------
    # 6. Developmental-stage counts.
    # --------------------------------------------------------------

    if stage_col is None:
        section("5. DEVELOPMENTAL STAGES")
        print("FAIL: No developmental-stage column was detected.")
        sys.exit(1)

    print_value_counts(
        obs,
        stage_col,
        "5. DEVELOPMENTAL-STAGE COUNTS",
    )

    # --------------------------------------------------------------
    # 7. Cell-type counts.
    # --------------------------------------------------------------

    if celltype_col is None:
        section("6. CELL TYPES")
        print("FAIL: No cell-type annotation column was detected.")
        sys.exit(1)

    print_value_counts(
        obs,
        celltype_col,
        "6. CELL-TYPE COUNTS",
    )

    # --------------------------------------------------------------
    # 8. Stage × cell-type table.
    # --------------------------------------------------------------

    section("7. STAGE × CELL-TYPE COUNTS")

    stage_celltype = pd.crosstab(
        obs[celltype_col].astype(str),
        obs[stage_col].astype(str),
        margins=True,
    )

    print(stage_celltype.to_string())

    # --------------------------------------------------------------
    # 9. Primary beta-lineage audit.
    # --------------------------------------------------------------

    section("8. PRIMARY BETA-LINEAGE AUDIT")

    available_celltypes = set(
        obs[celltype_col]
        .dropna()
        .astype(str)
    )

    missing_beta_states = [
        state
        for state in BETA_LINEAGE
        if state not in available_celltypes
    ]

    if missing_beta_states:
        print("WARNING: Some prespecified beta-lineage states were not found:")
        for state in missing_beta_states:
            print(f"  - {state}")
    else:
        print("PASS: All prespecified beta-lineage states are present.")

    beta_mask = (
        obs[celltype_col]
        .astype(str)
        .isin(BETA_LINEAGE)
    )

    beta_obs = obs.loc[beta_mask].copy()

    print(f"\nTotal beta-lineage cells: {len(beta_obs):,}")

    beta_table = pd.crosstab(
        beta_obs[celltype_col].astype(str),
        beta_obs[stage_col].astype(str),
        margins=True,
    )

    print()
    print(beta_table.to_string())

    # --------------------------------------------------------------
    # 10. Batch / experiment audit.
    # --------------------------------------------------------------

    section("9. BATCH / EXPERIMENT AUDIT")

    if batch_col is None:
        print("WARNING: No obvious experiment/batch column was detected.")
        print()
        print(
            "This does not automatically stop the study, but the two "
            "E15.5 experiments must eventually be distinguishable."
        )

        print("\nPotential metadata columns worth inspecting manually:")

        for col in obs.columns:
            n_unique = obs[col].nunique(dropna=True)

            if 2 <= n_unique <= 20:
                values = (
                    obs[col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                print(f"\n{col}:")
                print("  " + ", ".join(values[:20]))

    else:
        print(f"Using batch/experiment column: {batch_col}")

        batch_counts = (
            obs[batch_col]
            .astype(str)
            .value_counts()
        )

        print("\nOverall batch counts:")
        print(batch_counts.to_string())

        stage_batch = pd.crosstab(
            obs[stage_col].astype(str),
            obs[batch_col].astype(str),
            margins=True,
        )

        print("\nStage × batch counts:")
        print(stage_batch.to_string())

        # E15.5 bridge check.
        e155_mask = (
            obs[stage_col]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.upper()
            == "E15.5"
        )

        if e155_mask.any():
            n_e155_batches = (
                obs.loc[e155_mask, batch_col]
                .dropna()
                .astype(str)
                .nunique()
            )

            print(
                f"\nNumber of batch/experiment labels within E15.5: "
                f"{n_e155_batches}"
            )

            if n_e155_batches >= 2:
                print(
                    "PASS: E15.5 can be used as an internal "
                    "cross-experiment bridge."
                )
            else:
                print(
                    "WARNING: Only one E15.5 batch label was detected "
                    "in this metadata column."
                )

    # --------------------------------------------------------------
    # 11. Existing dimensional representations.
    # --------------------------------------------------------------

    section("10. EXISTING LOW-DIMENSIONAL REPRESENTATIONS")

    print("Top-level MuData obsm keys:")
    if len(mdata.obsm.keys()) == 0:
        print("  None")
    else:
        for key in mdata.obsm.keys():
            print(f"  {key}")

    for modality_name, adata in mdata.mod.items():

        print(f"\n{modality_name} obsm keys:")

        keys = list(adata.obsm.keys())

        if len(keys) == 0:
            print("  None")
        else:
            for key in keys:
                try:
                    shape = adata.obsm[key].shape
                    print(f"  {key:<35} shape={shape}")
                except Exception:
                    print(f"  {key}")

    # --------------------------------------------------------------
    # 12. Layer audit.
    # --------------------------------------------------------------

    section("11. DATA LAYERS")

    for modality_name, adata in mdata.mod.items():

        layers = list(adata.layers.keys())

        print(f"{modality_name}:")
        if layers:
            for layer in layers:
                print(f"  {layer}")
        else:
            print("  No additional layers")

        print()

    # --------------------------------------------------------------
    # 13. Final Phase-F0 summary.
    # --------------------------------------------------------------

    section("12. PHASE F0 SUMMARY")

    hard_pass = (
        {"rna", "atac"}.issubset(set(mdata.mod.keys()))
        and same_order
        and rna_cells.is_unique
        and atac_cells.is_unique
        and stage_col is not None
        and celltype_col is not None
        and len(beta_obs) >= 4000
    )

    print(f"Paired RNA + ATAC:             PASS")
    print(f"Exact RNA/ATAC cell pairing:   PASS")
    print(f"Developmental stage metadata:  PASS")
    print(f"Cell-type metadata:            PASS")
    print(
        f"Beta-lineage support:          "
        f"{'PASS' if len(beta_obs) >= 4000 else 'WARNING'} "
        f"({len(beta_obs):,} cells)"
    )

    if batch_col is None:
        print("Batch/experiment metadata:     WARNING")
    else:
        print("Batch/experiment metadata:     DETECTED")

    print()

    if hard_pass:
        print("PHASE F0 VERDICT: GO")
        print()
        print(
            "The dataset is structurally suitable for Phase F1: "
            "RNA-PCA and ATAC-LSI reconstruction plus batch-concordance analysis."
        )
    else:
        print("PHASE F0 VERDICT: REVIEW REQUIRED")
        print()
        print(
            "At least one structural requirement requires attention "
            "before Phase F1."
        )

    print()
    print("No GDIS calculation was performed.")
    line("=")


if __name__ == "__main__":
    main()

