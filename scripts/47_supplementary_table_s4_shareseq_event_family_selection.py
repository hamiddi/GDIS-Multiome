#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
47_supplementary_table_s4_shareseq_event_family_selection.py
=============================================================

Create Supplementary Table S4 for the GDIS-Multiome manuscript:

    Supplementary Table S4.
    SHARE-seq transition-energy event-family selection and
    window-resolution sensitivity.

PURPOSE
-------
The prospective event-freezing rule is central to GDIS-Multiome. This script
creates a compact, auditable table showing:

Part A. Eligible within-modality transition-energy families
    - modality
    - family ID
    - family center
    - window-support count
    - three-window support
    - robust-interior status
    - location inside the frozen TAC-2 IQR
    - distance to the TAC-2 median
    - median prominence
    - whether the family was frozen for cross-modal timing

Part B. Window-specific observations of the independently frozen family pair
    - 300/75, 400/100, and 500/125 window configurations
    - RNA event position
    - ATAC event position
    - RNA - ATAC separation
    - direction

NO new event selection or timing inference is performed. The script reads
only the frozen Phase F16 outputs and reformats them for publication.

EXPECTED INPUTS
---------------
data/GSE140203/representations_f16/
    f16_candidate_families.tsv
    f16_frozen_transition_energy_pair.tsv
    f16_window_specific_pairing.tsv

OUTPUTS
-------
data/GSE140203/tables_supplementary/
    Supplementary_Table_S4A_event_family_selection.tsv
    Supplementary_Table_S4B_window_sensitivity.tsv
    Supplementary_Table_S4_combined.md
    Supplementary_Table_S4_validation.tsv

RUN
---
python 47_supplementary_table_s4_shareseq_event_family_selection.py
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

DATA_DIR = Path("data") / "GSE140203"
F16_DIR = DATA_DIR / "representations_f16"
OUT_DIR = DATA_DIR / "tables_supplementary"

F16_CANDIDATES = F16_DIR / "f16_candidate_families.tsv"
F16_PAIR = F16_DIR / "f16_frozen_transition_energy_pair.tsv"
F16_WINDOWS = F16_DIR / "f16_window_specific_pairing.tsv"

OUTPUT_A = OUT_DIR / "Supplementary_Table_S4A_event_family_selection.tsv"
OUTPUT_B = OUT_DIR / "Supplementary_Table_S4B_window_sensitivity.tsv"
OUTPUT_MD = OUT_DIR / "Supplementary_Table_S4_combined.md"
OUTPUT_VALIDATION = OUT_DIR / "Supplementary_Table_S4_validation.tsv"


# =====================================================================
# DISPLAY SETTINGS
# =====================================================================

WINDOW_ORDER = [
    "sensitivity_w300_s75",
    "primary_w400_s100",
    "sensitivity_w500_s125",
]

WINDOW_LABELS = {
    "sensitivity_w300_s75": "300/75",
    "primary_w400_s100": "400/100",
    "sensitivity_w500_s125": "500/125",
}

MODALITY_ORDER = ["RNA", "ATAC"]


# =====================================================================
# HELPERS
# =====================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen F16 input was not found:\n{path}"
        )


def normalize_bool(series: pd.Series) -> pd.Series:
    """
    Normalize booleans after TSV round trip.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(False)
        .astype(bool)
    )


def yes_no(value) -> str:
    return "Yes" if bool(value) else "No"


def fmt6(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.6f}"


def fmt4(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.4f}"


def markdown_table(df: pd.DataFrame) -> str:
    """
    Convert a small dataframe to a clean Markdown table without requiring
    tabulate.
    """
    display = df.copy().astype(str)

    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in display.iterrows():
        safe_values = [
            str(row[col]).replace("|", r"\|")
            for col in headers
        ]
        lines.append("| " + " | ".join(safe_values) + " |")

    return "\n".join(lines)


# =====================================================================
# LOAD AND VALIDATE FROZEN OUTPUTS
# =====================================================================

def load_inputs():
    for path in [F16_CANDIDATES, F16_PAIR, F16_WINDOWS]:
        require_file(path)

    candidates = pd.read_csv(F16_CANDIDATES, sep="\t", low_memory=False)
    pair = pd.read_csv(F16_PAIR, sep="\t", low_memory=False)
    windows = pd.read_csv(F16_WINDOWS, sep="\t", low_memory=False)

    if len(pair) != 1:
        raise RuntimeError(
            "f16_frozen_transition_energy_pair.tsv must contain exactly one row."
        )

    required_candidate = [
        "modality",
        "family_id",
        "family_center_parameter",
        "window_config_support_count",
        "robust_interior_family",
        "three_window_family",
        "median_prominence",
        "inside_TAC2_IQR",
        "absolute_distance_to_TAC2_median",
    ]

    required_pair = [
        "RNA_family_id",
        "RNA_family_center_parameter",
        "RNA_three_window_family",
        "ATAC_family_id",
        "ATAC_family_center_parameter",
        "ATAC_three_window_family",
        "observed_RNA_minus_ATAC_family_center",
        "observed_direction",
    ]

    required_windows = [
        "window_config",
        "RNA_family_id",
        "RNA_peak_available",
        "RNA_peak_parameter",
        "RNA_peak_prominence",
        "ATAC_family_id",
        "ATAC_peak_available",
        "ATAC_peak_parameter",
        "ATAC_peak_prominence",
        "paired_observation_available",
        "RNA_minus_ATAC_parameter",
        "direction",
    ]

    for df, required, label in [
        (candidates, required_candidate, "F16 candidate-family table"),
        (pair, required_pair, "F16 frozen-pair table"),
        (windows, required_windows, "F16 window-specific table"),
    ]:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise RuntimeError(
                f"{label} is missing required column(s): "
                + ", ".join(missing)
            )

    # Normalize candidate booleans.
    for col in [
        "robust_interior_family",
        "three_window_family",
        "inside_TAC2_IQR",
    ]:
        candidates[col] = normalize_bool(candidates[col])

    # Normalize window booleans.
    for col in [
        "RNA_peak_available",
        "ATAC_peak_available",
        "paired_observation_available",
    ]:
        windows[col] = normalize_bool(windows[col])

    return candidates, pair.iloc[0], windows


# =====================================================================
# PART A — EVENT-FAMILY SELECTION
# =====================================================================

def build_part_a(
    candidates: pd.DataFrame,
    pair_row: pd.Series,
) -> pd.DataFrame:

    selected_ids = {
        "RNA": int(pair_row["RNA_family_id"]),
        "ATAC": int(pair_row["ATAC_family_id"]),
    }

    table = candidates.copy()

    table["family_id"] = pd.to_numeric(
        table["family_id"], errors="raise"
    ).astype(int)

    table["Frozen for timing"] = [
        "Yes"
        if int(fid) == selected_ids[str(modality)]
        else "No"
        for modality, fid in zip(
            table["modality"].astype(str),
            table["family_id"],
        )
    ]

    table["Modality order"] = (
        table["modality"]
        .astype(str)
        .map({m: i for i, m in enumerate(MODALITY_ORDER)})
    )

    table = table.sort_values(
        ["Modality order", "family_center_parameter"],
        kind="mergesort",
    ).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "Modality": table["modality"].astype(str),
            "Family": table["family_id"].astype(int),
            "Center pseudotime": table["family_center_parameter"].map(fmt6),
            "Window support": table["window_config_support_count"].astype(int),
            "3-window support": table["three_window_family"].map(yes_no),
            "Robust interior": table["robust_interior_family"].map(yes_no),
            "Within TAC-2 IQR": table["inside_TAC2_IQR"].map(yes_no),
            "|Center − TAC-2 median|": (
                table["absolute_distance_to_TAC2_median"].map(fmt6)
            ),
            "Median prominence": table["median_prominence"].map(fmt6),
            "Frozen for timing": table["Frozen for timing"],
        }
    )

    return out


# =====================================================================
# PART B — WINDOW SENSITIVITY
# =====================================================================

def build_part_b(
    windows: pd.DataFrame,
) -> pd.DataFrame:

    table = windows.copy()

    table["order"] = (
        table["window_config"]
        .astype(str)
        .map({name: i for i, name in enumerate(WINDOW_ORDER)})
    )

    unexpected = table["order"].isna()
    if unexpected.any():
        bad = table.loc[unexpected, "window_config"].astype(str).tolist()
        raise RuntimeError(
            "Unexpected window configuration(s): " + ", ".join(bad)
        )

    table = table.sort_values("order").reset_index(drop=True)

    out = pd.DataFrame(
        {
            "Window / step": (
                table["window_config"].astype(str).map(WINDOW_LABELS)
            ),
            "RNA event pseudotime": table["RNA_peak_parameter"].map(fmt6),
            "ATAC event pseudotime": table["ATAC_peak_parameter"].map(fmt6),
            "RNA − ATAC": table["RNA_minus_ATAC_parameter"].map(fmt6),
            "Direction": table["direction"].astype(str),
        }
    )

    return out


# =====================================================================
# PUBLICATION INTEGRITY CHECKS
# =====================================================================

def validate_frozen_tables(
    candidates: pd.DataFrame,
    pair_row: pd.Series,
    windows: pd.DataFrame,
) -> pd.DataFrame:

    checks = []

    # Selected family IDs should each appear exactly once among the candidate
    # families for the corresponding modality.
    for modality in MODALITY_ORDER:
        selected_id = int(pair_row[f"{modality}_family_id"])
        n_matches = int(
            (
                candidates["modality"].astype(str).eq(modality)
                & pd.to_numeric(
                    candidates["family_id"], errors="coerce"
                ).eq(selected_id)
            ).sum()
        )

        checks.append(
            {
                "check": f"{modality} frozen family occurs once in candidate table",
                "pass": n_matches == 1,
                "observed": n_matches,
                "expected": 1,
            }
        )

    # Frozen family centers must match the candidate-family table.
    for modality in MODALITY_ORDER:
        selected_id = int(pair_row[f"{modality}_family_id"])
        frozen_center = float(
            pair_row[f"{modality}_family_center_parameter"]
        )

        candidate_center = float(
            candidates.loc[
                candidates["modality"].astype(str).eq(modality)
                & pd.to_numeric(
                    candidates["family_id"], errors="coerce"
                ).eq(selected_id),
                "family_center_parameter",
            ].iloc[0]
        )

        checks.append(
            {
                "check": f"{modality} family center matches frozen pair",
                "pass": bool(
                    np.isclose(
                        candidate_center,
                        frozen_center,
                        atol=1e-12,
                        rtol=0.0,
                    )
                ),
                "observed": candidate_center,
                "expected": frozen_center,
            }
        )

    # The window-specific table should contain all three prespecified windows.
    present = set(windows["window_config"].astype(str))
    expected = set(WINDOW_ORDER)

    checks.append(
        {
            "check": "All three prespecified window configurations are present",
            "pass": present == expected,
            "observed": ",".join(sorted(present)),
            "expected": ",".join(WINDOW_ORDER),
        }
    )

    # All three windows should recover both frozen families in F16.
    paired_all = bool(
        windows["paired_observation_available"].astype(bool).all()
    )

    checks.append(
        {
            "check": "Frozen pair observed in all three window configurations",
            "pass": paired_all,
            "observed": paired_all,
            "expected": True,
        }
    )

    # Verify the window-specific difference equals RNA minus ATAC.
    computed_delta = (
        pd.to_numeric(windows["RNA_peak_parameter"], errors="coerce")
        - pd.to_numeric(windows["ATAC_peak_parameter"], errors="coerce")
    )
    stored_delta = pd.to_numeric(
        windows["RNA_minus_ATAC_parameter"], errors="coerce"
    )

    delta_match = bool(
        np.allclose(
            computed_delta.to_numpy(dtype=float),
            stored_delta.to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
            equal_nan=True,
        )
    )

    checks.append(
        {
            "check": "Window-specific RNA−ATAC separations recompute exactly",
            "pass": delta_match,
            "observed": delta_match,
            "expected": True,
        }
    )

    # Direction should be ATAC earlier in all three frozen observations.
    all_atac_earlier = bool(
        windows["direction"]
        .astype(str)
        .str.lower()
        .eq("atac earlier than rna")
        .all()
    )

    checks.append(
        {
            "check": "Direction is ATAC earlier in all three windows",
            "pass": all_atac_earlier,
            "observed": all_atac_earlier,
            "expected": True,
        }
    )

    validation = pd.DataFrame(checks)
    return validation


# =====================================================================
# MARKDOWN OUTPUT
# =====================================================================

def build_markdown(
    part_a: pd.DataFrame,
    part_b: pd.DataFrame,
    pair_row: pd.Series,
) -> str:

    rna_center = float(pair_row["RNA_family_center_parameter"])
    atac_center = float(pair_row["ATAC_family_center_parameter"])
    family_delta = float(
        pair_row["observed_RNA_minus_ATAC_family_center"]
    )

    caption = (
        "**Supplementary Table S4. SHARE-seq transition-energy event-family "
        "selection and window-resolution sensitivity.** "
        "Part A lists the robust-interior TAC-2-region event families eligible "
        "under the prospectively frozen within-modality selection rule and "
        "identifies the RNA and ATAC families frozen before cross-modal timing "
        "inference. Part B reports the observed positions of the frozen event "
        "pair across the three prespecified window configurations. Positive "
        "RNA−ATAC values indicate an earlier ATAC event."
    )

    note = (
        f"Frozen family centers: RNA = {rna_center:.6f}; "
        f"ATAC = {atac_center:.6f}; "
        f"RNA−ATAC family-center separation = {family_delta:.6f}. "
        "Family selection was performed independently within RNA and ATAC; "
        "the sign or magnitude of the cross-modal separation was not used as "
        "a selection criterion."
    )

    text = [
        caption,
        "",
        "### A. Eligible event families and frozen selection",
        "",
        markdown_table(part_a),
        "",
        "### B. Window-resolution sensitivity of the frozen pair",
        "",
        markdown_table(part_b),
        "",
        f"*Note.* {note}",
        "",
    ]

    return "\n".join(text)


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 110)
    print("SUPPLEMENTARY TABLE S4 — SHARE-seq EVENT-FAMILY SELECTION")
    print("=" * 110)

    candidates, pair_row, windows = load_inputs()

    validation = validate_frozen_tables(
        candidates,
        pair_row,
        windows,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    validation.to_csv(
        OUTPUT_VALIDATION,
        sep="\t",
        index=False,
    )

    if not bool(validation["pass"].all()):
        print()
        print("VALIDATION FAILED:")
        print(validation.to_string(index=False))
        raise RuntimeError(
            "Frozen F16 integrity checks failed. "
            "Supplementary Table S4 was not written."
        )

    part_a = build_part_a(
        candidates,
        pair_row,
    )

    part_b = build_part_b(
        windows,
    )

    part_a.to_csv(
        OUTPUT_A,
        sep="\t",
        index=False,
    )

    part_b.to_csv(
        OUTPUT_B,
        sep="\t",
        index=False,
    )

    markdown = build_markdown(
        part_a,
        part_b,
        pair_row,
    )

    OUTPUT_MD.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("Part A — eligible event families:")
    print(part_a.to_string(index=False))

    print()
    print("Part B — window sensitivity:")
    print(part_b.to_string(index=False))

    print()
    print("Frozen pair:")
    print(
        f"  RNA center  = "
        f"{float(pair_row['RNA_family_center_parameter']):.6f}"
    )
    print(
        f"  ATAC center = "
        f"{float(pair_row['ATAC_family_center_parameter']):.6f}"
    )
    print(
        f"  RNA−ATAC    = "
        f"{float(pair_row['observed_RNA_minus_ATAC_family_center']):.6f}"
    )
    print(
        f"  Direction   = {pair_row['observed_direction']}"
    )

    print()
    print("All frozen F16 integrity checks passed.")
    print()
    print("Saved:")
    print(f"  {OUTPUT_A}")
    print(f"  {OUTPUT_B}")
    print(f"  {OUTPUT_MD}")
    print(f"  {OUTPUT_VALIDATION}")


if __name__ == "__main__":
    main()

