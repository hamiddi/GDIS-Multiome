#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
13_phase_f6a_external_metadata_acquisition.py
==============================================

Project:
GDIS-Multiome external validation

External dataset:
GSE205117
"Decoding gene regulation in the mouse embryo using single-cell multi-omics"

PHASE F6a
---------
Reproducible acquisition and provenance recording for the independent
GSE205117 paired single-cell multiome validation dataset.

DEFAULT BEHAVIOR
----------------
The script downloads ONLY the small official GEO cell-metadata file:

    GSE205117_cell_metadata.txt.gz

It then:
    1. creates data/GSE205117 if needed;
    2. downloads from the official NCBI GEO HTTPS endpoint;
    3. supports safe resume through a temporary .part file;
    4. validates that the result is readable gzip text;
    5. records file size;
    6. computes SHA-256;
    7. writes a machine-readable provenance manifest.

LARGE RAW ARCHIVE
-----------------
The ~32.5-GB GEO supplementary archive is NOT downloaded by default.

After the metadata audit approves the dataset, it can be acquired explicitly:

    python 13_phase_f6a_external_metadata_acquisition.py --include-raw

This explicit gate is intentional so a large transfer cannot occur merely by
running the reproducibility pipeline.

NO ANALYSIS
-----------
This script performs:
    - no cell filtering;
    - no pseudotime;
    - no GDIS;
    - no CMIL;
    - no biological inference.

Run:
    python 13_phase_f6a_external_metadata_acquisition.py

Optional:
    python 13_phase_f6a_external_metadata_acquisition.py --include-raw
    python 13_phase_f6a_external_metadata_acquisition.py --force
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.error
import urllib.request


# =====================================================================
# FROZEN DATA-SOURCE CONFIGURATION
# =====================================================================

GEO_ACCESSION = "GSE205117"

DATA_DIR = Path("data") / GEO_ACCESSION

METADATA_FILENAME = "GSE205117_cell_metadata.txt.gz"
RAW_FILENAME = "GSE205117_RAW.tar"

METADATA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE205nnn/"
    "GSE205117/suppl/GSE205117_cell_metadata.txt.gz"
)

RAW_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE205nnn/"
    "GSE205117/suppl/GSE205117_RAW.tar"
)

MANIFEST_FILENAME = "acquisition_manifest.json"

USER_AGENT = (
    "GDIS-Multiome-reproducible-pipeline/1.0 "
    "(GSE205117 external validation)"
)

DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


# =====================================================================
# DISPLAY
# =====================================================================

def line(char="=", width=108):
    print(char * width)


def section(title):
    print()
    line("=")
    print(title)
    line("=")


def human_size(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(n_bytes)

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"

        value /= 1024.0

    return f"{n_bytes} B"


# =====================================================================
# HASHING / VALIDATION
# =====================================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(DOWNLOAD_CHUNK_SIZE)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def validate_gzip_text(path: Path) -> dict:
    """
    Validate that the metadata file:
        - exists;
        - is non-empty;
        - opens as gzip;
        - contains a non-empty first line.

    Returns a small content diagnostic dictionary.
    """
    if not path.exists():
        raise RuntimeError(
            f"Metadata file does not exist: {path}"
        )

    size = path.stat().st_size

    if size <= 0:
        raise RuntimeError(
            f"Metadata file is empty: {path}"
        )

    try:
        with gzip.open(
            path,
            "rt",
            encoding="utf-8",
            errors="replace",
        ) as handle:
            first_line = handle.readline().strip()

            second_line = handle.readline().strip()

    except OSError as exc:
        raise RuntimeError(
            f"Downloaded metadata is not a readable gzip file: {exc}"
        ) from exc

    if not first_line:
        raise RuntimeError(
            "Downloaded metadata gzip contains no readable header line."
        )

    return {
        "first_line": first_line,
        "second_line_present": bool(second_line),
    }


# =====================================================================
# HTTP DOWNLOAD WITH RESUME
# =====================================================================

def get_remote_size(url: str) -> int | None:
    """
    Try HEAD first. If HEAD is unsupported, return None rather than failing.
    """
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={
            "User-Agent": USER_AGENT,
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:
            content_length = response.headers.get(
                "Content-Length"
            )

            if content_length is None:
                return None

            return int(content_length)

    except Exception:
        return None


def download_with_resume(
    url: str,
    destination: Path,
    force: bool = False,
) -> dict:
    """
    Download `url` to `destination`.

    Behavior:
      - an existing complete destination is reused unless --force is given;
      - partial content is kept in destination.part;
      - if the server honors HTTP Range, downloading resumes;
      - if Range is ignored, the .part file is restarted safely.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    remote_size = get_remote_size(
        url
    )

    if destination.exists() and not force:
        local_size = destination.stat().st_size

        if (
            remote_size is None
            or local_size == remote_size
        ):
            print(
                f"Existing file retained: {destination}"
            )
            print(
                f"Local size: {human_size(local_size)}"
            )

            return {
                "downloaded": False,
                "resumed": False,
                "remote_size": remote_size,
                "local_size": local_size,
            }

        print(
            "Existing destination size differs from the remote file; "
            "it will be reacquired safely."
        )

    if force:
        if destination.exists():
            destination.unlink()

        part_path = destination.with_suffix(
            destination.suffix + ".part"
        )

        if part_path.exists():
            part_path.unlink()

    part_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    existing = (
        part_path.stat().st_size
        if part_path.exists()
        else 0
    )

    headers = {
        "User-Agent": USER_AGENT,
    }

    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

        print(
            f"Found partial file: {part_path}"
        )
        print(
            f"Attempting resume from {human_size(existing)}"
        )

    request = urllib.request.Request(
        url,
        headers=headers,
    )

    try:
        response = urllib.request.urlopen(
            request,
            timeout=120,
        )

    except urllib.error.HTTPError as exc:
        if existing > 0 and exc.code == 416:
            # Requested range is not satisfiable. If the partial file already
            # has the expected full length, promote it.
            if (
                remote_size is not None
                and existing == remote_size
            ):
                part_path.replace(
                    destination
                )

                return {
                    "downloaded": True,
                    "resumed": True,
                    "remote_size": remote_size,
                    "local_size": destination.stat().st_size,
                }

        raise

    status = getattr(
        response,
        "status",
        None,
    )

    # HTTP 206 means Range was honored. HTTP 200 means the server returned
    # the complete resource, so restart instead of appending duplicate data.
    range_honored = (
        existing > 0
        and status == 206
    )

    if existing > 0 and not range_honored:
        print(
            "Server did not honor the resume request; restarting the "
            "temporary download from byte 0."
        )

        existing = 0
        mode = "wb"

    else:
        mode = (
            "ab"
            if range_honored
            else "wb"
        )

    total_written = existing

    with response:
        with part_path.open(
            mode
        ) as handle:
            while True:
                chunk = response.read(
                    DOWNLOAD_CHUNK_SIZE
                )

                if not chunk:
                    break

                handle.write(
                    chunk
                )

                total_written += len(
                    chunk
                )

                if remote_size:
                    percent = (
                        100.0
                        * total_written
                        / remote_size
                    )

                    print(
                        f"\rDownloaded "
                        f"{human_size(total_written)} / "
                        f"{human_size(remote_size)} "
                        f"({percent:6.2f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\rDownloaded "
                        f"{human_size(total_written)}",
                        end="",
                        flush=True,
                    )

    print()

    if (
        remote_size is not None
        and total_written != remote_size
    ):
        raise RuntimeError(
            "Download size mismatch: "
            f"expected {remote_size} bytes, "
            f"received {total_written} bytes."
        )

    part_path.replace(
        destination
    )

    return {
        "downloaded": True,
        "resumed": range_honored,
        "remote_size": remote_size,
        "local_size": destination.stat().st_size,
    }


# =====================================================================
# PROVENANCE
# =====================================================================

def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def build_file_record(
    path: Path,
    url: str,
    acquisition_result: dict,
    validation: dict | None = None,
) -> dict:
    record = {
        "filename": path.name,
        "relative_path": str(path),
        "source_url": url,
        "size_bytes": path.stat().st_size,
        "size_human": human_size(
            path.stat().st_size
        ),
        "sha256": sha256_file(
            path
        ),
        "downloaded_this_run": bool(
            acquisition_result.get(
                "downloaded",
                False,
            )
        ),
        "resumed_this_run": bool(
            acquisition_result.get(
                "resumed",
                False,
            )
        ),
    }

    if validation is not None:
        record[
            "validation"
        ] = validation

    return record


def write_manifest(
    metadata_record: dict,
    raw_record: dict | None,
    include_raw: bool,
) -> Path:
    manifest = {
        "dataset_accession": GEO_ACCESSION,
        "dataset_role": (
            "independent paired single-cell multiome "
            "external validation"
        ),
        "acquisition_timestamp_utc": utc_now_iso(),
        "pipeline_phase": "F6a",
        "source_repository": "NCBI GEO",
        "metadata_download_is_default": True,
        "raw_archive_requires_explicit_flag": True,
        "include_raw_requested": bool(
            include_raw
        ),
        "files": {
            "cell_metadata": metadata_record,
            "raw_archive": raw_record,
        },
        "analysis_guardrails": {
            "pseudotime_calculated": False,
            "gdis_calculated": False,
            "cmil_calculated": False,
            "discovery_dataset_modified": False,
        },
    }

    path = DATA_DIR / MANIFEST_FILENAME

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    return path


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reproducibly acquire GSE205117 metadata and, only when "
            "explicitly requested, the large raw supplementary archive."
        )
    )

    parser.add_argument(
        "--include-raw",
        action="store_true",
        help=(
            "Also download GSE205117_RAW.tar (~32.5 GB). "
            "Not recommended until the F6b metadata audit passes."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-download requested files even if complete local files exist."
        ),
    )

    return parser.parse_args()


# =====================================================================
# MAIN
# =====================================================================

def main():
    args = parse_args()

    section(
        "PHASE F6a — GSE205117 REPRODUCIBLE DATA ACQUISITION"
    )

    print(
        f"Dataset: {GEO_ACCESSION}"
    )

    print(
        f"Destination: {DATA_DIR.resolve()}"
    )

    print()

    print(
        "Default acquisition: official GEO cell metadata only."
    )

    print(
        "Large raw archive requires explicit --include-raw."
    )

    print()

    print(
        "No pseudotime."
    )

    print(
        "No GDIS."
    )

    print(
        "No CMIL."
    )

    # -----------------------------------------------------------------
    # Create project data directory.
    # -----------------------------------------------------------------

    section(
        "1. CREATE DATA DIRECTORY"
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Directory ready: {DATA_DIR.resolve()}"
    )

    # -----------------------------------------------------------------
    # Metadata.
    # -----------------------------------------------------------------

    section(
        "2. ACQUIRE OFFICIAL GEO CELL METADATA"
    )

    metadata_path = (
        DATA_DIR
        / METADATA_FILENAME
    )

    print(
        f"Source:      {METADATA_URL}"
    )

    print(
        f"Destination: {metadata_path}"
    )

    metadata_result = download_with_resume(
        METADATA_URL,
        metadata_path,
        force=args.force,
    )

    print()

    print(
        "Validating gzip text..."
    )

    metadata_validation = validate_gzip_text(
        metadata_path
    )

    print(
        "Gzip validation: PASS"
    )

    print(
        f"Header preview: "
        f"{metadata_validation['first_line'][:180]}"
    )

    print()

    print(
        "Computing SHA-256..."
    )

    metadata_record = build_file_record(
        metadata_path,
        METADATA_URL,
        metadata_result,
        validation={
            "gzip_readable": True,
            "nonempty_header": True,
            "second_line_present": metadata_validation[
                "second_line_present"
            ],
        },
    )

    print(
        f"File size: "
        f"{metadata_record['size_human']}"
    )

    print(
        f"SHA-256: "
        f"{metadata_record['sha256']}"
    )

    # -----------------------------------------------------------------
    # Optional large raw archive.
    # -----------------------------------------------------------------

    raw_record = None

    section(
        "3. LARGE RAW GEO ARCHIVE GATE"
    )

    raw_path = (
        DATA_DIR
        / RAW_FILENAME
    )

    if not args.include_raw:
        print(
            "Raw archive download: SKIPPED BY DESIGN"
        )

        print()

        print(
            "Reason: the metadata-level structural audit should pass "
            "before transferring the ~32.5-GB raw supplementary archive."
        )

        print()

        print(
            "After F6b passes, acquisition can be reproduced with:"
        )

        print(
            "  python "
            "13_phase_f6a_external_metadata_acquisition.py "
            "--include-raw"
        )

    else:
        print(
            "Explicit --include-raw flag received."
        )

        print(
            f"Source:      {RAW_URL}"
        )

        print(
            f"Destination: {raw_path}"
        )

        raw_result = download_with_resume(
            RAW_URL,
            raw_path,
            force=args.force,
        )

        print()

        print(
            "Computing SHA-256 for raw archive..."
        )

        raw_record = build_file_record(
            raw_path,
            RAW_URL,
            raw_result,
        )

        print(
            f"File size: "
            f"{raw_record['size_human']}"
        )

        print(
            f"SHA-256: "
            f"{raw_record['sha256']}"
        )

    # -----------------------------------------------------------------
    # Manifest.
    # -----------------------------------------------------------------

    section(
        "4. WRITE ACQUISITION PROVENANCE MANIFEST"
    )

    manifest_path = write_manifest(
        metadata_record=metadata_record,
        raw_record=raw_record,
        include_raw=args.include_raw,
    )

    print(
        f"Manifest: {manifest_path.resolve()}"
    )

    print()

    print(
        "The manifest records:"
    )

    print(
        "  - GEO accession"
    )

    print(
        "  - official source URL"
    )

    print(
        "  - acquisition timestamp"
    )

    print(
        "  - exact local file size"
    )

    print(
        "  - SHA-256 checksum"
    )

    print(
        "  - whether the large raw archive was explicitly requested"
    )

    # -----------------------------------------------------------------
    # Decision.
    # -----------------------------------------------------------------

    section(
        "5. PHASE F6a DECISION"
    )

    print(
        "PHASE F6a VERDICT: METADATA ACQUISITION COMPLETE"
    )

    print()

    print(
        "Next step:"
    )

    print(
        "  python 14_phase_f6b_external_dataset_audit.py"
    )

    print()

    print(
        "Do not download/process the large raw archive unless the "
        "metadata structural audit supports the external validation design."
    )

    line("=")


if __name__ == "__main__":
    main()

