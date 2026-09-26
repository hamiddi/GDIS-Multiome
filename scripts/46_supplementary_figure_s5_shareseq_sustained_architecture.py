#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
46_supplementary_figure_s5_shareseq_sustained_architecture.py
================================================================

Publication-only visualization for:

Supplementary Figure S5.
SHARE-seq sustained-instability architecture.

NO new statistical inference is performed.

Panels
------
A. RNA sustained-GDIS profiles across 300/75, 400/100, and 500/125.
B. ATAC sustained-GDIS profiles across the same window configurations.
C. 95% near-maximum sustained-GDIS envelopes and global maxima.
D. Compact decision summary explaining why sustained-profile argmaxes
   were not paired across modalities.

Inputs
------
data/GSE140203/representations_f14/
    f14_gdis_profiles.tsv.gz
    f14_independent_state_landmarks.tsv

data/GSE140203/representations_f15b/
    f15b_sustained_gdis_envelopes.tsv
    f15b_sustained_gdis_shape_summary.tsv
    f15b_topology_summary.tsv

Outputs
-------
data/GSE140203/figures_supplementary/
    Supplementary_Figure_S5_SHAREseq_sustained_instability_architecture.png
    Supplementary_Figure_S5_SHAREseq_sustained_instability_architecture.pdf
    Supplementary_Figure_S5_SHAREseq_sustained_instability_summary.tsv

Run
---
python 46_supplementary_figure_s5_shareseq_sustained_architecture.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# PATHS
# =====================================================================

DATA_DIR = Path("data") / "GSE140203"
F14_DIR = DATA_DIR / "representations_f14"
F15B_DIR = DATA_DIR / "representations_f15b"
OUT_DIR = DATA_DIR / "figures_supplementary"

F14_PROFILES = F14_DIR / "f14_gdis_profiles.tsv.gz"
F14_LANDMARKS = F14_DIR / "f14_independent_state_landmarks.tsv"

F15B_ENVELOPES = F15B_DIR / "f15b_sustained_gdis_envelopes.tsv"
F15B_SHAPE = F15B_DIR / "f15b_sustained_gdis_shape_summary.tsv"
F15B_TOPOLOGY = F15B_DIR / "f15b_topology_summary.tsv"

OUTPUT_PNG = OUT_DIR / "Supplementary_Figure_S5_SHAREseq_sustained_instability_architecture.png"
OUTPUT_PDF = OUT_DIR / "Supplementary_Figure_S5_SHAREseq_sustained_instability_architecture.pdf"
OUTPUT_SUMMARY = OUT_DIR / "Supplementary_Figure_S5_SHAREseq_sustained_instability_summary.tsv"


# =====================================================================
# FROZEN ANALYSIS SETTINGS / DISPLAY ORDER
# =====================================================================

PRIMARY_PROFILE_ROLE = "primary_multiome"
TRANSITION_STATE = "TAC-2"
NEAR_MAX_LEVEL = 0.95

WINDOW_CONFIGS = [
    "sensitivity_w300_s75",
    "primary_w400_s100",
    "sensitivity_w500_s125",
]

WINDOW_LABELS = {
    "sensitivity_w300_s75": "300 / 75",
    "primary_w400_s100": "400 / 100",
    "sensitivity_w500_s125": "500 / 125",
}

# Publication settings
DPI = 600
FIGSIZE = (14.0, 9.5)
TITLE_SIZE = 16
PANEL_TITLE_SIZE = 13
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 9.5
TEXT_SIZE = 10.5
PANEL_LABEL_SIZE = 15

COLORS = {
    "sensitivity_w300_s75": "#1f77b4",
    "primary_w400_s100": "#d62728",
    "sensitivity_w500_s125": "#2ca02c",
}

LINESTYLES = {
    "sensitivity_w300_s75": ":",
    "primary_w400_s100": "-",
    "sensitivity_w500_s125": "--",
}


# =====================================================================
# UTILITIES
# =====================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen input not found:\n{path}")


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.10, 1.05, label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def style_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.20, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def get_tac2_landmark(landmarks: pd.DataFrame) -> pd.Series:
    row = landmarks.loc[
        landmarks["state"].astype(str).eq(TRANSITION_STATE)
    ]
    if row.empty:
        raise RuntimeError("TAC-2 landmark was not found.")
    return row.iloc[0]


def load_data():
    for path in [
        F14_PROFILES,
        F14_LANDMARKS,
        F15B_ENVELOPES,
        F15B_SHAPE,
        F15B_TOPOLOGY,
    ]:
        require_file(path)

    profiles = pd.read_csv(F14_PROFILES, sep="\t", compression="gzip")
    landmarks = pd.read_csv(F14_LANDMARKS, sep="\t")
    envelopes = pd.read_csv(F15B_ENVELOPES, sep="\t")
    shape = pd.read_csv(F15B_SHAPE, sep="\t")
    topology = pd.read_csv(F15B_TOPOLOGY, sep="\t")

    return profiles, landmarks, envelopes, shape, topology


# =====================================================================
# SUMMARY TABLE
# =====================================================================

def build_summary(envelopes: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per modality.

    IMPORTANT FIX:
    The 95%-near-maximum width is calculated directly from the frozen envelope
    table. We intentionally do NOT merge the same-named width column from the
    F15b topology table, which previously caused pandas to generate _x / _y
    suffixes and led to the KeyError seen in v1.
    """

    env95 = envelopes.loc[
        np.isclose(
            pd.to_numeric(envelopes["near_max_level"], errors="coerce"),
            NEAR_MAX_LEVEL,
        )
    ].copy()

    numeric_cols = [
        "component_width_parameter",
        "global_max_parameter",
        "component_start_parameter",
        "component_end_parameter",
    ]
    for col in numeric_cols:
        env95[col] = pd.to_numeric(env95[col], errors="coerce")

    rows = []

    for modality in ["RNA", "ATAC"]:
        sub = env95.loc[
            env95["modality"].astype(str).eq(modality)
        ].copy()

        if sub.empty:
            raise RuntimeError(
                f"No {NEAR_MAX_LEVEL:.2f} near-maximum envelope rows for {modality}."
            )

        rows.append(
            {
                "modality": modality,
                "median_95pct_nearmax_width": float(
                    sub["component_width_parameter"].median()
                ),
                "min_95pct_nearmax_width": float(
                    sub["component_width_parameter"].min()
                ),
                "max_95pct_nearmax_width": float(
                    sub["component_width_parameter"].max()
                ),
                "median_global_max_parameter": float(
                    sub["global_max_parameter"].median()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    # Merge only NON-DUPLICATE topology descriptors.
    if "modality" in topology.columns:
        topology_keep = [
            col
            for col in [
                "modality",
                "n_robust_energy_families",
                "n_robust_interior_energy_families",
                "n_three_window_energy_families",
                "median_95pct_nearmax_fraction_points",
            ]
            if col in topology.columns
        ]

        if len(topology_keep) > 1:
            summary = summary.merge(
                topology[topology_keep],
                on="modality",
                how="left",
                validate="one_to_one",
            )

    required = {
        "modality",
        "median_95pct_nearmax_width",
        "median_global_max_parameter",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise RuntimeError(
            "Internal summary is missing: " + ", ".join(sorted(missing))
        )

    return summary, env95


# =====================================================================
# PROFILE PANELS
# =====================================================================

def plot_sustained_profiles(
    ax,
    profiles: pd.DataFrame,
    modality: str,
    tac2: pd.Series,
) -> None:

    primary = profiles.loc[
        profiles["profile_role"].astype(str).eq(PRIMARY_PROFILE_ROLE)
        & profiles["modality"].astype(str).eq(modality)
    ].copy()

    ax.axvspan(
        float(tac2["q25"]),
        float(tac2["q75"]),
        color="0.85",
        alpha=0.35,
        label="TAC-2 IQR",
        zorder=0,
    )

    ax.axvline(
        float(tac2["median"]),
        color="black",
        linestyle=(0, (4, 3)),
        linewidth=1.1,
        label="TAC-2 median",
    )

    for config in WINDOW_CONFIGS:
        sub = primary.loc[
            primary["window_config"].astype(str).eq(config)
        ].sort_values("parameter")

        if sub.empty:
            raise RuntimeError(f"Missing {modality} profile for {config}")

        ax.plot(
            sub["parameter"],
            sub["gdis"],
            color=COLORS[config],
            linestyle=LINESTYLES[config],
            linewidth=2.1,
            label=WINDOW_LABELS[config],
        )

    ax.set_xlabel(
        "RNA-derived common pseudotime",
        fontsize=LABEL_SIZE,
    )
    ax.set_ylabel(
        "Sustained GDIS ($\\lambda_t=0$)",
        fontsize=LABEL_SIZE,
    )
    ax.set_title(
        f"{modality} sustained-instability profiles",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
    )

    ax.legend(
        fontsize=LEGEND_SIZE,
        frameon=False,
        loc="best",
    )

    style_axis(ax)


# =====================================================================
# ENVELOPE PANEL
# =====================================================================

def plot_envelope_topology(
    ax,
    env95: pd.DataFrame,
    tac2: pd.Series,
) -> None:

    rows = []

    for modality in ["RNA", "ATAC"]:
        for config in WINDOW_CONFIGS:
            sub = env95.loc[
                env95["modality"].astype(str).eq(modality)
                & env95["window_config"].astype(str).eq(config)
            ]

            if sub.empty:
                raise RuntimeError(
                    f"Missing 95% envelope for {modality}/{config}"
                )

            row = sub.iloc[0].copy()
            rows.append(
                {
                    "modality": modality,
                    "window_config": config,
                    "label": f"{modality}  {WINDOW_LABELS[config]}",
                    "start": float(row["component_start_parameter"]),
                    "end": float(row["component_end_parameter"]),
                    "max": float(row["global_max_parameter"]),
                }
            )

    plot_df = pd.DataFrame(rows)

    y = np.arange(len(plot_df))[::-1]

    ax.axvspan(
        float(tac2["q25"]),
        float(tac2["q75"]),
        color="0.85",
        alpha=0.35,
        zorder=0,
    )

    ax.axvline(
        float(tac2["median"]),
        color="black",
        linestyle=(0, (4, 3)),
        linewidth=1.1,
    )

    for ypos, row in zip(y, plot_df.itertuples(index=False)):
        ax.hlines(
            ypos,
            row.start,
            row.end,
            color=COLORS[row.window_config],
            linewidth=5.5,
        )
        ax.scatter(
            row.max,
            ypos,
            s=52,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=TICK_SIZE)

    ax.set_xlabel(
        "RNA-derived common pseudotime",
        fontsize=LABEL_SIZE,
    )

    ax.set_title(
        "95% near-maximum sustained-GDIS envelopes",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
    )

    style_axis(ax)


# =====================================================================
# SUMMARY PANEL
# =====================================================================

def plot_decision_summary(
    ax,
    summary: pd.DataFrame,
    tac2: pd.Series,
) -> None:

    ax.axis("off")
    s = summary.set_index("modality")

    rna_width = float(s.loc["RNA", "median_95pct_nearmax_width"])
    atac_width = float(s.loc["ATAC", "median_95pct_nearmax_width"])

    rna_max = float(s.loc["RNA", "median_global_max_parameter"])
    atac_max = float(s.loc["ATAC", "median_global_max_parameter"])

    width_ratio = rna_width / atac_width if atac_width > 0 else np.nan

    ax.set_title(
        "Sustained-profile pairing decision",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        pad=8,
    )

    summary_text = (
        f"TAC-2 IQR: {float(tac2['q25']):.3f}–{float(tac2['q75']):.3f}\n\n"
        f"RNA median 95% near-max width: {rna_width:.3f}\n"
        f"ATAC median 95% near-max width: {atac_width:.3f}\n"
        f"RNA / ATAC width ratio: {width_ratio:.1f}×\n\n"
        f"Median global maximum:\n"
        f"RNA = {rna_max:.3f}; ATAC = {atac_max:.3f}\n\n"
        "Decision:\n"
        "Sustained-profile argmaxes were NOT paired.\n\n"
        "Rationale:\n"
        "RNA forms a broad sustained-instability plateau,\n"
        "whereas ATAC shows a much narrower early maximum.\n"
        "A broad RNA plateau should not be converted into an\n"
        "artificially precise cross-modal timing event.\n\n"
        "Formal timing inference therefore used only the\n"
        "prospectively frozen localized transition-energy families."
    )

    ax.text(
        0.04,
        0.96,
        summary_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TEXT_SIZE,
        linespacing=1.22,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
        ),
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 120)
    print("SUPPLEMENTARY FIGURE S5 — SHARE-seq SUSTAINED-INSTABILITY ARCHITECTURE")
    print("=" * 120)

    profiles, landmarks, envelopes, shape, topology = load_data()
    tac2 = get_tac2_landmark(landmarks)

    summary, env95 = build_summary(envelopes, topology)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        OUTPUT_SUMMARY,
        sep="\t",
        index=False,
    )

    print()
    print("Frozen TAC-2 region:")
    print(
        f"  q25={float(tac2['q25']):.6f}, "
        f"median={float(tac2['median']):.6f}, "
        f"q75={float(tac2['q75']):.6f}"
    )
    print()
    print("Corrected sustained-envelope summary:")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE,
    )

    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    plot_sustained_profiles(
        ax_a,
        profiles,
        "RNA",
        tac2,
    )
    panel_label(ax_a, "A")

    plot_sustained_profiles(
        ax_b,
        profiles,
        "ATAC",
        tac2,
    )
    panel_label(ax_b, "B")

    plot_envelope_topology(
        ax_c,
        env95,
        tac2,
    )
    panel_label(ax_c, "C")

    plot_decision_summary(
        ax_d,
        summary,
        tac2,
    )
    panel_label(ax_d, "D")

    fig.suptitle(
        "Supplementary Figure S5. SHARE-seq sustained-instability architecture",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        y=0.985,
    )

    fig.tight_layout(
        rect=[0.015, 0.015, 0.985, 0.955],
        h_pad=2.2,
        w_pad=2.4,
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print()
    print("Saved:")
    print(f"  {OUTPUT_PNG}")
    print(f"  {OUTPUT_PDF}")
    print(f"  {OUTPUT_SUMMARY}")
    print()
    print("No new inference was performed.")


if __name__ == "__main__":
    main()

