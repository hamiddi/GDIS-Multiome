#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""Canonical environment preflight for the GDIS-Multiome pipeline."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path
import shutil
import sys

MIN_PYTHON = (3, 10)
REQUIRED_IMPORTS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("anndata", "anndata"),
    ("mudata", "mudata"),
    ("scanpy", "scanpy"),
    ("gdis", "pygdis"),
    ("py2bit", "py2bit"),
    ("MOODS.scan", "MOODS-python"),
]
EXPECTED_PYGDIS = "1.0.0"
RECOMMENDED_FREE_GIB = 80.0


def main() -> None:
    print("=" * 100)
    print("GDIS-MULTIOME CANONICAL PIPELINE — ENVIRONMENT PREFLIGHT")
    print("=" * 100)

    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
            f"Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} is required; "
            f"found {sys.version.split()[0]}."
        )

    print(f"Python: {sys.version.split()[0]} — PASS")

    failures = []
    for import_name, package_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(import_name)
            try:
                version = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                version = "installed"
            print(f"{package_name:<18} {version:<16} PASS")
        except Exception as exc:
            failures.append((package_name, str(exc)))
            print(f"{package_name:<18} MISSING/FAILED")

    if failures:
        details = "\n".join(f"  {name}: {err}" for name, err in failures)
        raise RuntimeError(
            "Required Python dependencies are missing or failed to import:\n"
            + details
            + "\nInstall config/requirements.txt or config/environment.yml first."
        )

    pygdis_version = importlib.metadata.version("pygdis")
    if pygdis_version != EXPECTED_PYGDIS:
        raise RuntimeError(
            f"pyGDIS must be exactly {EXPECTED_PYGDIS}; found {pygdis_version}."
        )
    print(f"pyGDIS exact-version safeguard: {pygdis_version} — PASS")

    root = Path.cwd()
    usage = shutil.disk_usage(root)
    free_gib = usage.free / (1024 ** 3)
    print(f"Free disk space: {free_gib:.1f} GiB")
    if free_gib < RECOMMENDED_FREE_GIB:
        print(
            f"WARNING: < {RECOMMENDED_FREE_GIB:.0f} GiB free. The full pipeline "
            "includes the ~32.5-GB GSE205117 RAW archive plus other large inputs."
        )

    print("PREFLIGHT VERDICT: PASS")


if __name__ == "__main__":
    main()
