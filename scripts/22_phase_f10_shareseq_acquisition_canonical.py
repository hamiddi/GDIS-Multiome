#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
22_phase_f10_shareseq_acquisition_canonical.py
===============================================

Canonical acquisition-only phase for GSE140203 SHARE-seq late-anagen data.

This replaces the superseded F10/F10b development scripts. It performs only:
    - download of the six frozen GEO supplementary files;
    - safe resume through .part files when possible;
    - SHA-256 verification against the frozen checksums from the completed study;
    - gzip-readability validation.

It deliberately performs no barcode parsing or biological audit. The next
canonical analysis step is:

    24_phase_f10c_shareseq_namespace_normalized_audit.py

RUN
---
    python scripts/22_phase_f10_shareseq_acquisition_canonical.py
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import shutil
import urllib.error
import urllib.request

ACCESSION = "GSE140203"
DATA_DIR = Path("data") / ACCESSION
CHUNK_SIZE = 8 * 1024 * 1024
USER_AGENT = "GDIS-Multiome-canonical-SHAREseq-acquisition/1.0"

ATAC_BASE = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/"
    "GSM4156nnn/GSM4156597/suppl/"
)
RNA_BASE = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/"
    "GSM4156nnn/GSM4156608/suppl/"
)

FILES = [
    (
        "GSM4156597_skin_celltype.txt.gz",
        ATAC_BASE + "GSM4156597_skin_celltype.txt.gz",
        "22699fe209c4413f2083c4e85f7b6c589336e2b6a3414b622e81d63000c200d1",
    ),
    (
        "GSM4156597_skin.late.anagen.counts.txt.gz",
        ATAC_BASE + "GSM4156597_skin.late.anagen.counts.txt.gz",
        "1db64b652da36364c3931c45464fc5894307f85df5cbd56b5310cfcbbe3ef3cc",
    ),
    (
        "GSM4156597_skin.late.anagen.barcodes.txt.gz",
        ATAC_BASE + "GSM4156597_skin.late.anagen.barcodes.txt.gz",
        "356fbf14fe38ea2985c327d8d08577252f1aef8a0efd7053f19afa878cad7130",
    ),
    (
        "GSM4156597_skin.late.anagen.peaks.bed.gz",
        ATAC_BASE + "GSM4156597_skin.late.anagen.peaks.bed.gz",
        "5e18c8662658b4ac94857e8203270c9ceddb767ffacfb58ff631ed9187796572",
    ),
    (
        "GSM4156597_skin.late.anagen.atac.fragments.bed.gz",
        ATAC_BASE + "GSM4156597_skin.late.anagen.atac.fragments.bed.gz",
        "02743849e7daf7a06dc8e5f5a23c49e23a947ce373f145bc636f2777af019904",
    ),
    (
        "GSM4156608_skin.late.anagen.rna.counts.txt.gz",
        RNA_BASE + "GSM4156608_skin.late.anagen.rna.counts.txt.gz",
        "431568ae8a2a3c7e7bf121027fe875f9707abadeedf82a71a8ead979cc98ef66",
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def validate_gzip(path: Path) -> None:
    try:
        with gzip.open(path, "rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise RuntimeError(f"Unreadable gzip file: {path}: {exc}") from exc


def download_with_resume(url: str, destination: Path) -> None:
    part = Path(str(destination) + ".part")
    total = remote_size(url)
    existing = part.stat().st_size if part.exists() else 0

    headers = {"User-Agent": USER_AGENT}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    request = urllib.request.Request(url, headers=headers)

    try:
        response = urllib.request.urlopen(request, timeout=180)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc

    # If the server ignored Range, restart rather than append duplicate bytes.
    append_mode = existing > 0 and getattr(response, "status", None) == 206
    if existing > 0 and not append_mode:
        existing = 0

    mode = "ab" if append_mode else "wb"
    with response, part.open(mode) as output:
        shutil.copyfileobj(response, output, length=CHUNK_SIZE)

    if total is not None and part.stat().st_size != total:
        raise RuntimeError(
            f"Incomplete download for {destination.name}: "
            f"local={part.stat().st_size:,} remote={total:,} bytes"
        )

    part.replace(destination)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("CANONICAL GSE140203 SHARE-seq ACQUISITION")
    print("=" * 120)

    for filename, url, expected_sha in FILES:
        path = DATA_DIR / filename
        print(f"\n{filename}")

        if path.exists():
            observed = sha256_file(path)
            if observed != expected_sha:
                raise RuntimeError(
                    f"Existing file has the wrong SHA-256:\n  {path}\n"
                    f"Expected: {expected_sha}\nObserved: {observed}\n"
                    "Move/remove the incorrect file before rerunning."
                )
            validate_gzip(path)
            print("  existing file: SHA-256 PASS; gzip PASS")
            continue

        print(f"  source: {url}")
        download_with_resume(url, path)

        observed = sha256_file(path)
        if observed != expected_sha:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file failed SHA-256:\n  {path}\n"
                f"Expected: {expected_sha}\nObserved: {observed}"
            )

        validate_gzip(path)
        print("  SHA-256 PASS; gzip PASS")

    print("\n" + "=" * 120)
    print("ACQUISITION VERDICT: PASS — ALL SIX FROZEN SHARE-seq FILES VERIFIED")
    print("=" * 120)
    print("Next: python scripts/24_phase_f10c_shareseq_namespace_normalized_audit.py --full-fragment-scan")


if __name__ == "__main__":
    main()
