#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
14_phase_f6b_external_dataset_audit_corrected.py
=================================================

Corrected Phase F6b structural audit for GSE205117.

This version explicitly uses the official GEO canonical fields:
    stage
    sample
    genotype
    celltype.mapped
    pass_rnaQC
    pass_atacQC
    doublet_call

It replaces the earlier heuristic-field version that incorrectly selected:
    sample   as stage
    genotype as cell type

No pseudotime, GDIS, or CMIL is calculated.
"""

from __future__ import annotations

from pathlib import Path
import gzip
import hashlib
import json
import re
import tarfile

import numpy as np
import pandas as pd


ACCESSION = "GSE205117"
DATA_DIR = Path("data") / ACCESSION

METADATA_FILE = DATA_DIR / "GSE205117_cell_metadata.txt.gz"
MANIFEST_FILE = DATA_DIR / "acquisition_manifest.json"
RAW_TAR = DATA_DIR / "GSE205117_RAW.tar"

CELL_COL = "cell"
BARCODE_COL = "barcode"
SAMPLE_COL = "sample"
STAGE_COL = "stage"
GENOTYPE_COL = "genotype"
CELLTYPE_COL = "celltype.mapped"

RNA_QC_COL = "pass_rnaQC"
ATAC_QC_COL = "pass_atacQC"
DOUBLET_CALL_COL = "doublet_call"

REQUIRED_COLUMNS = [
    CELL_COL,
    BARCODE_COL,
    SAMPLE_COL,
    STAGE_COL,
    GENOTYPE_COL,
    CELLTYPE_COL,
]

NUMERIC_QC_COLUMNS = [
    "nFeature_RNA",
    "nCount_RNA",
    "mitochondrial_percent_RNA",
    "ribosomal_percent_RNA",
    "doublet_score",
    "TSSEnrichment_atac",
    "ReadsInTSS_atac",
    "PromoterRatio_atac",
    "NucleosomeRatio_atac",
    "nFrags_atac",
    "BlacklistRatio_atac",
]

TARGET_TERMS = [
    "NMP",
    "neuromesoderm",
    "neuromesodermal",
    "paraxial",
    "somitic",
    "somite",
    "presomitic",
    "PSM",
    "caudal mesoderm",
    "caudal_mesoderm",
    "nascent mesoderm",
    "nascent_mesoderm",
    "primitive streak",
    "primitive_streak",
    "spinal cord",
    "spinal_cord",
]

EXPECTED_MIN_STAGES = 4


def line(char="=", width=116):
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
    if df is None or df.empty:
        print("(empty)")
        return

    with pd.option_context(
        "display.max_rows", 150,
        "display.max_columns", None,
        "display.width", 340,
        "display.max_colwidth", 100,
        "display.float_format", lambda x: f"{x:.{digits}f}",
    ):
        print(df.to_string())


def human_size(n_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n_bytes)

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{n_bytes} B"


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def verify_provenance():
    if not METADATA_FILE.exists():
        return False, {"reason": "Metadata file not found."}

    if not MANIFEST_FILE.exists():
        return False, {
            "reason": (
                "Acquisition manifest not found. Run "
                "13_phase_f6a_external_metadata_acquisition.py first."
            )
        }

    with MANIFEST_FILE.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    record = manifest["files"]["cell_metadata"]

    expected_size = int(record["size_bytes"])
    expected_sha = str(record["sha256"])

    observed_size = METADATA_FILE.stat().st_size
    observed_sha = sha256_file(METADATA_FILE)

    details = {
        "source_url": record.get("source_url", "UNKNOWN"),
        "expected_size": expected_size,
        "observed_size": observed_size,
        "expected_sha256": expected_sha,
        "observed_sha256": observed_sha,
        "size_match": expected_size == observed_size,
        "checksum_match": expected_sha == observed_sha,
    }

    return bool(
        details["size_match"] and details["checksum_match"]
    ), details


def validate_gzip():
    with gzip.open(
        METADATA_FILE,
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        header = handle.readline().strip()

    if not header:
        raise RuntimeError("Metadata gzip has no readable header.")

    return header


def load_metadata():
    return pd.read_csv(
        METADATA_FILE,
        sep="\t",
        compression="gzip",
        low_memory=False,
    ).dropna(axis=1, how="all")


def normalize_bool(series):
    mapping = {
        "true": True,
        "t": True,
        "1": True,
        "yes": True,
        "pass": True,
        "passed": True,
        "false": False,
        "f": False,
        "0": False,
        "no": False,
        "fail": False,
        "failed": False,
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


def target_mask(series):
    text = (
        series.fillna("")
        .astype(str)
        .str.lower()
    )

    pattern = "|".join(
        re.escape(term.lower())
        for term in TARGET_TERMS
    )

    return text.str.contains(
        pattern,
        regex=True,
        na=False,
    )


def numeric_summary(df, columns):
    rows = []

    for column in columns:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        rows.append(
            {
                "metric": column,
                "n": len(values),
                "min": float(values.min()),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def labeled(series):
    return (
        series.astype("object")
        .where(series.notna(), "<MISSING>")
        .astype(str)
    )


def inventory_tar(path):
    with tarfile.open(path, "r") as tar:
        members = [
            member.name
            for member in tar.getmembers()
            if member.isfile()
        ]

    lower = [name.lower() for name in members]

    atac = [
        members[i]
        for i, name in enumerate(lower)
        if any(
            token in name
            for token in [
                "atac",
                "fragment",
                "peak",
            ]
        )
    ]

    rna = [
        members[i]
        for i, name in enumerate(lower)
        if any(
            token in name
            for token in [
                "gex",
                "rna",
                "matrix.mtx",
                "features.tsv",
                "barcodes.tsv",
            ]
        )
        and not any(
            token in name
            for token in [
                "atac",
                "fragment",
                "peak",
            ]
        )
    ]

    return members, rna, atac


def main():

    section(
        "PHASE F6b — CORRECTED EXTERNAL STRUCTURAL AUDIT: GSE205117"
    )

    print(f"Metadata: {METADATA_FILE.resolve()}")
    print("Canonical GEO fields are used explicitly.")
    print()
    print("GSE275562 remains frozen.")
    print("No pseudotime.")
    print("No GDIS.")
    print("No CMIL.")

    section("1. ACQUISITION PROVENANCE")

    verified, provenance = verify_provenance()

    for key, value in provenance.items():
        print(f"{key}: {value}")

    if not verified:
        print()
        print("PHASE F6b VERDICT: PROVENANCE REVIEW REQUIRED")
        return

    print()
    print("Provenance verification: PASS")

    section("2. LOAD OFFICIAL GEO METADATA")

    header = validate_gzip()
    metadata = load_metadata()

    print(f"File size: {human_size(METADATA_FILE.stat().st_size)}")
    print(f"Rows:      {len(metadata):,}")
    print(f"Columns:   {metadata.shape[1]:,}")
    print()
    print(f"Header preview: {header[:260]}")

    section("3. CANONICAL FIELD VERIFICATION")

    rows = []

    for column in REQUIRED_COLUMNS:
        rows.append(
            {
                "column": column,
                "present": column in metadata.columns,
                "n_unique": (
                    metadata[column].nunique(dropna=True)
                    if column in metadata.columns
                    else np.nan
                ),
                "n_missing": (
                    int(metadata[column].isna().sum())
                    if column in metadata.columns
                    else np.nan
                ),
            }
        )

    canonical = pd.DataFrame(rows).set_index("column")
    print_df(canonical, digits=0)

    missing_required = [
        column
        for column in REQUIRED_COLUMNS
        if column not in metadata.columns
    ]

    if missing_required:
        print()
        print("Missing required canonical fields:")

        for column in missing_required:
            print(f"  {column}")

        print()
        print("PHASE F6b VERDICT: REVIEW REQUIRED")
        return

    print()
    print("Frozen mapping:")
    print(f"  stage/time = {STAGE_COL}")
    print(f"  sample     = {SAMPLE_COL}")
    print(f"  genotype   = {GENOTYPE_COL}")
    print(f"  cell type  = {CELLTYPE_COL}")

    section("4. CORE BIOLOGICAL INVENTORIES")

    for title, column in [
        ("Developmental stages", STAGE_COL),
        ("Samples", SAMPLE_COL),
        ("Genotypes", GENOTYPE_COL),
        ("Mapped cell types", CELLTYPE_COL),
    ]:
        subsection(title)

        counts = (
            labeled(metadata[column])
            .value_counts(dropna=False)
            .rename("n_cells")
            .to_frame()
        )

        print_df(counts)

    section("5. PAIRED RNA + ATAC QC AUDIT")

    rna_qc_available = RNA_QC_COL in metadata.columns
    atac_qc_available = ATAC_QC_COL in metadata.columns

    print(f"{RNA_QC_COL} present:  {rna_qc_available}")
    print(f"{ATAC_QC_COL} present: {atac_qc_available}")

    joint_qc_mask = pd.Series(
        True,
        index=metadata.index,
    )

    if rna_qc_available:
        rna_pass = normalize_bool(metadata[RNA_QC_COL])

        print()
        print(f"{RNA_QC_COL} values:")

        print_df(
            labeled(metadata[RNA_QC_COL])
            .value_counts()
            .rename("n_cells")
            .to_frame()
        )

        joint_qc_mask &= rna_pass.fillna(False)

    if atac_qc_available:
        atac_pass = normalize_bool(metadata[ATAC_QC_COL])

        print()
        print(f"{ATAC_QC_COL} values:")

        print_df(
            labeled(metadata[ATAC_QC_COL])
            .value_counts()
            .rename("n_cells")
            .to_frame()
        )

        joint_qc_mask &= atac_pass.fillna(False)

    joint_qc_count = int(joint_qc_mask.sum())
    joint_qc_fraction = float(joint_qc_mask.mean())

    print()
    print(
        f"Cells passing BOTH available RNA/ATAC QC flags: "
        f"{joint_qc_count:,}/{len(metadata):,} "
        f"({joint_qc_fraction:.4f})"
    )

    if DOUBLET_CALL_COL in metadata.columns:
        print()
        print(f"{DOUBLET_CALL_COL} values:")

        print_df(
            labeled(metadata[DOUBLET_CALL_COL])
            .value_counts()
            .rename("n_cells")
            .to_frame()
        )

    qcdf = numeric_summary(
        metadata,
        NUMERIC_QC_COLUMNS,
    )

    if not qcdf.empty:
        print()
        print("Numeric QC summary:")

        print_df(
            qcdf.set_index("metric"),
            digits=3,
        )

    section("6. STRUCTURAL CROSS-TABS — ALL METADATA CELLS")

    subsection("Stage × sample")

    print_df(
        pd.crosstab(
            labeled(metadata[STAGE_COL]),
            labeled(metadata[SAMPLE_COL]),
            margins=True,
        )
    )

    subsection("Stage × genotype")

    print_df(
        pd.crosstab(
            labeled(metadata[STAGE_COL]),
            labeled(metadata[GENOTYPE_COL]),
            margins=True,
        )
    )

    section("7. TARGET NMP / PARAXIAL-SOMITIC MESODERM AXIS")

    all_target_mask = target_mask(
        metadata[CELLTYPE_COL]
    )

    target_all = metadata.loc[
        all_target_mask
    ].copy()

    target_qc = metadata.loc[
        all_target_mask & joint_qc_mask
    ].copy()

    print(
        f"Target-axis cells before joint QC: "
        f"{len(target_all):,}"
    )

    print(
        f"Target-axis cells after joint QC:  "
        f"{len(target_qc):,}"
    )

    lineage_present = len(target_all) > 0
    lineage_qc_present = len(target_qc) > 0

    if lineage_present:

        subsection("Matched target annotations — all cells")

        print_df(
            target_all[CELLTYPE_COL]
            .astype(str)
            .value_counts()
            .rename("n_cells")
            .to_frame()
        )

        subsection("Target annotation × stage — all cells")

        print_df(
            pd.crosstab(
                target_all[CELLTYPE_COL].astype(str),
                labeled(target_all[STAGE_COL]),
                margins=True,
            )
        )

        subsection(
            "Matched target annotations — joint RNA+ATAC QC"
        )

        print_df(
            target_qc[CELLTYPE_COL]
            .astype(str)
            .value_counts()
            .rename("n_cells")
            .to_frame()
        )

        subsection(
            "Target annotation × stage — joint RNA+ATAC QC"
        )

        print_df(
            pd.crosstab(
                target_qc[CELLTYPE_COL].astype(str),
                labeled(target_qc[STAGE_COL]),
                margins=True,
            )
        )

        subsection(
            "Target annotation × genotype — joint RNA+ATAC QC"
        )

        print_df(
            pd.crosstab(
                target_qc[CELLTYPE_COL].astype(str),
                labeled(target_qc[GENOTYPE_COL]),
                margins=True,
            )
        )

        subsection(
            "Target stage × sample — joint RNA+ATAC QC"
        )

        print_df(
            pd.crosstab(
                labeled(target_qc[STAGE_COL]),
                labeled(target_qc[SAMPLE_COL]),
                margins=True,
            )
        )

    else:
        print()
        print(
            "No target annotations matched the prespecified broad term list."
        )
        print(
            "Inspect the exact celltype.mapped labels above before "
            "changing any lineage definition."
        )

    section("8. WT / BRACHYURY-T KO STRUCTURE")

    genotype_text = (
        metadata[GENOTYPE_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    wt_mask = genotype_text.str.upper().eq("WT")
    ko_mask = genotype_text.str.upper().eq("T_KO")

    print(f"WT cells:   {int(wt_mask.sum()):,}")
    print(f"T_KO cells: {int(ko_mask.sum()):,}")
    print(
        f"Missing genotype: "
        f"{int(metadata[GENOTYPE_COL].isna().sum()):,}"
    )

    wt_stages = (
        metadata.loc[
            wt_mask,
            STAGE_COL,
        ]
        .dropna()
        .astype(str)
        .nunique()
    )

    print(f"WT developmental stages: {wt_stages}")

    subsection("WT stage counts")

    print_df(
        labeled(
            metadata.loc[
                wt_mask,
                STAGE_COL,
            ]
        )
        .value_counts()
        .sort_index()
        .rename("n_cells")
        .to_frame()
    )

    subsection("T_KO stage counts")

    print_df(
        labeled(
            metadata.loc[
                ko_mask,
                STAGE_COL,
            ]
        )
        .value_counts()
        .sort_index()
        .rename("n_cells")
        .to_frame()
    )

    section("9. RAW PAIRED-MULTIOME ARCHIVE STATUS")

    raw_present = RAW_TAR.exists()
    paired_raw_evidence = False

    if not raw_present:
        print("GSE205117_RAW.tar is not present locally.")
        print()
        print("This is expected at F6b.")
        print(
            "If F6b passes, acquire it reproducibly using:"
        )
        print(
            "  python "
            "13_phase_f6a_external_metadata_acquisition.py "
            "--include-raw"
        )

    else:
        print(f"Archive: {RAW_TAR.resolve()}")
        print(f"Size:    {human_size(RAW_TAR.stat().st_size)}")

        members, rna_members, atac_members = inventory_tar(
            RAW_TAR
        )

        paired_raw_evidence = bool(
            rna_members and atac_members
        )

        print(f"Archive members:      {len(members):,}")
        print(f"RNA/GEX-like members: {len(rna_members):,}")
        print(f"ATAC-like members:    {len(atac_members):,}")

    section("10. PHASE F6b SUITABILITY CHECKS")

    n_stages = metadata[STAGE_COL].nunique(dropna=True)
    n_samples = metadata[SAMPLE_COL].nunique(dropna=True)
    n_celltypes = metadata[CELLTYPE_COL].nunique(dropna=True)

    checks = pd.DataFrame(
        [
            {
                "criterion": "Acquisition provenance verified",
                "pass": verified,
            },
            {
                "criterion": "Canonical stage field present",
                "pass": STAGE_COL in metadata.columns,
            },
            {
                "criterion": "Canonical sample field present",
                "pass": SAMPLE_COL in metadata.columns,
            },
            {
                "criterion": "Canonical genotype field present",
                "pass": GENOTYPE_COL in metadata.columns,
            },
            {
                "criterion": "Canonical mapped cell-type field present",
                "pass": CELLTYPE_COL in metadata.columns,
            },
            {
                "criterion": (
                    f">={EXPECTED_MIN_STAGES} developmental stages"
                ),
                "pass": n_stages >= EXPECTED_MIN_STAGES,
            },
            {
                "criterion": "Multiple independent samples represented",
                "pass": n_samples >= 2,
            },
            {
                "criterion": "Multiple mapped cell types represented",
                "pass": n_celltypes >= 2,
            },
            {
                "criterion": "Both RNA and ATAC QC flags present",
                "pass": rna_qc_available and atac_qc_available,
            },
            {
                "criterion": "Joint RNA+ATAC QC retains cells",
                "pass": joint_qc_count > 0,
            },
            {
                "criterion": "Target NMP/mesoderm annotations represented",
                "pass": lineage_present,
            },
            {
                "criterion": (
                    "Target NMP/mesoderm annotations survive joint QC"
                ),
                "pass": lineage_qc_present,
            },
            {
                "criterion": "WT cells represented",
                "pass": bool(wt_mask.any()),
            },
            {
                "criterion": "WT spans >=3 stages",
                "pass": wt_stages >= 3,
            },
            {
                "criterion": "T_KO cells represented",
                "pass": bool(ko_mask.any()),
            },
        ]
    )

    checks["status"] = np.where(
        checks["pass"],
        "PASS",
        "REVIEW",
    )

    print_df(
        checks.set_index("criterion"),
        digits=0,
    )

    n_pass = int(checks["pass"].sum())

    print()
    print(
        f"Criteria passed: {n_pass}/{len(checks)}"
    )

    section("11. PHASE F6b DECISION")

    core_ok = bool(
        verified
        and n_stages >= EXPECTED_MIN_STAGES
        and n_samples >= 2
        and n_celltypes >= 2
        and rna_qc_available
        and atac_qc_available
        and joint_qc_count > 0
        and lineage_present
        and lineage_qc_present
        and wt_mask.any()
        and wt_stages >= 3
        and ko_mask.any()
    )

    if core_ok:
        if raw_present and paired_raw_evidence:
            verdict = (
                "GO — EXTERNAL DATASET STRUCTURALLY SUITABLE"
            )
            reason = (
                "Canonical metadata fields, joint RNA+ATAC QC, the target "
                "developmental axis, multistage WT samples, and T_KO cells "
                "are all represented, and the local archive contains both "
                "RNA/GEX- and ATAC-like data."
            )
        else:
            verdict = (
                "GO — STRUCTURALLY SUITABLE; ACQUIRE PAIRED MULTIOME DATA"
            )
            reason = (
                "Canonical metadata fields, paired-modality QC, the target "
                "developmental axis, multistage WT samples, and T_KO cells "
                "are present. The next step is reproducible acquisition of "
                "the paired RNA/ATAC data; no trajectory should be built yet."
            )
    else:
        verdict = "REVIEW REQUIRED"
        reason = (
            "At least one essential external-validation structural criterion "
            "was not verified after using the correct canonical GEO fields."
        )

    print(f"PHASE F6b VERDICT: {verdict}")
    print()
    print(reason)
    print()
    print("No pseudotime was calculated.")
    print("No GDIS was calculated.")
    print("No CMIL was calculated.")
    print("No discovery-dataset setting was changed.")

    if verdict.startswith(
        "GO — STRUCTURALLY SUITABLE"
    ):
        print()
        print("Next reproducible step:")
        print(
            "  python "
            "13_phase_f6a_external_metadata_acquisition.py "
            "--include-raw"
        )
        print()
        print("After paired-data acquisition:")
        print(
            "  15_phase_f7_external_pairing_representation_qc.py"
        )

    line("=")


if __name__ == "__main__":
    main()

