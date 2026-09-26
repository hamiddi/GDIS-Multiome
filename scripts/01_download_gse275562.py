#!/usr/bin/env python3

# =============================================================================
# Paper Title:
# GDIS-Multiome: A Dynamical Instability Framework for Resolving Cross-Modal
# State-Transition Architecture in Paired Single-Cell Multiomics
#
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
01_download_gse275562.py
========================

Project:
Cross-Modal Dynamical Instability Reveals Chromatin Priming
before Transcriptional Cell-Fate Transitions

Step 01:
Download the processed GSE275562 paired RNA + ATAC MuData object from NCBI GEO.

What this script does
---------------------
1. Creates a local ./data directory if it does not already exist.
2. Downloads:
       GSE275562_mudata_with_annotation_all.h5mu
   from NCBI GEO.
3. Uses a temporary ".part" file while downloading.
4. Resumes an interrupted download when the server supports HTTP Range requests.
5. Compares the final local file size with the remote Content-Length when available.
6. Renames the completed ".part" file to the final H5MU filename only after
   the download appears complete.

What this script does NOT do
----------------------------
- It does not open or analyze the H5MU file.
- It does not preprocess RNA or ATAC data.
- It does not calculate pseudotime.
- It does not calculate GDIS.

Run
---
From the CMIL project directory:

    python 01_download_gse275562.py

No third-party Python packages are required.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

GEO_ACCESSION = "GSE275562"

FILENAME = "GSE275562_mudata_with_annotation_all.h5mu"

URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE275nnn/GSE275562/suppl/"
    "GSE275562_mudata_with_annotation_all.h5mu"
)

DATA_DIR = Path("data")
FINAL_PATH = DATA_DIR / FILENAME
PART_PATH = DATA_DIR / f"{FILENAME}.part"

# Read/write the remote file in 8 MiB chunks.
CHUNK_SIZE = 8 * 1024 * 1024

# Update the screen roughly every 5 seconds during download.
PROGRESS_INTERVAL_SECONDS = 5


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def separator(char: str = "=", width: int = 78) -> None:
    """Print a clean terminal separator."""
    print(char * width)


def format_bytes(n_bytes: int | None) -> str:
    """Convert a byte count to a human-readable value."""
    if n_bytes is None:
        return "unknown"

    value = float(n_bytes)

    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} TiB"


def get_remote_size(url: str) -> int | None:
    """
    Ask the server for the remote file size.

    Returns
    -------
    int or None
        Remote Content-Length in bytes, if NCBI provides it.
    """
    request = Request(
        url,
        method="HEAD",
        headers={"User-Agent": "GDIS-Multiome-GSE275562-Downloader/1.0"},
    )

    try:
        with urlopen(request, timeout=60) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value is not None else None

    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"WARNING: Could not obtain remote file size: {exc}")
        return None


def enough_disk_space(required_bytes: int | None) -> bool:
    """
    Check whether the current filesystem has enough free space.

    We keep a 5 GiB safety margin beyond the estimated amount still required.
    """
    if required_bytes is None:
        return True

    usage = shutil.disk_usage(DATA_DIR)
    safety_margin = 5 * 1024**3

    return usage.free >= required_bytes + safety_margin


def print_disk_status() -> None:
    """Print filesystem capacity information for the data directory."""
    usage = shutil.disk_usage(DATA_DIR)

    print(f"Filesystem total: {format_bytes(usage.total)}")
    print(f"Filesystem used:  {format_bytes(usage.used)}")
    print(f"Filesystem free:  {format_bytes(usage.free)}")


def download_file(url: str, remote_size: int | None) -> None:
    """
    Download the H5MU file, resuming from an existing .part file if possible.

    The final filename is created only after the temporary download passes the
    available size checks.
    """
    existing = PART_PATH.stat().st_size if PART_PATH.exists() else 0

    if existing > 0:
        print(f"Partial download found: {format_bytes(existing)}")
        print("Attempting to resume it.")
    else:
        print("No partial download found. Starting from the beginning.")

    headers = {
        "User-Agent": "GDIS-Multiome-GSE275562-Downloader/1.0",
    }

    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    request = Request(url, headers=headers)

    try:
        response = urlopen(request, timeout=120)
    except HTTPError as exc:
        # HTTP 416 means the requested range begins beyond the end of the file.
        # This can happen when the partial file is already complete.
        if exc.code == 416 and remote_size is not None and existing == remote_size:
            print("The partial file already matches the remote file size.")
            PART_PATH.replace(FINAL_PATH)
            return
        raise

    status = getattr(response, "status", None)

    # If we asked to resume but the server ignored Range and returned a full
    # response (HTTP 200), restart safely instead of appending duplicate bytes.
    if existing > 0 and status == 200:
        print(
            "The server did not honor the resume request. "
            "Restarting the download from byte 0."
        )
        existing = 0
        mode = "wb"
    else:
        mode = "ab" if existing > 0 else "wb"

    downloaded = existing
    start_time = time.time()
    last_report = start_time

    with response, PART_PATH.open(mode) as output:
        while True:
            chunk = response.read(CHUNK_SIZE)

            if not chunk:
                break

            output.write(chunk)
            downloaded += len(chunk)

            now = time.time()

            if now - last_report >= PROGRESS_INTERVAL_SECONDS:
                elapsed = max(now - start_time, 1e-6)
                transferred_this_run = downloaded - existing
                speed = transferred_this_run / elapsed

                if remote_size:
                    percent = 100.0 * downloaded / remote_size
                    remaining = max(remote_size - downloaded, 0)
                    eta_seconds = remaining / speed if speed > 0 else None

                    eta_text = (
                        f"{eta_seconds / 60:.1f} min"
                        if eta_seconds is not None
                        else "unknown"
                    )

                    print(
                        f"Downloaded {format_bytes(downloaded)} / "
                        f"{format_bytes(remote_size)} "
                        f"({percent:6.2f}%) | "
                        f"{format_bytes(int(speed))}/s | "
                        f"ETA {eta_text}"
                    )
                else:
                    print(
                        f"Downloaded {format_bytes(downloaded)} | "
                        f"{format_bytes(int(speed))}/s"
                    )

                last_report = now

    print()
    print(f"Temporary file size: {format_bytes(PART_PATH.stat().st_size)}")


def verify_and_finalize(remote_size: int | None) -> bool:
    """
    Verify the completed temporary file and rename it to the final H5MU filename.

    Returns
    -------
    bool
        True if finalization succeeds.
    """
    if not PART_PATH.exists():
        print("ERROR: Temporary download file was not created.")
        return False

    local_size = PART_PATH.stat().st_size

    if local_size == 0:
        print("ERROR: Downloaded file is empty.")
        return False

    if remote_size is not None and local_size != remote_size:
        print()
        print("DOWNLOAD INCOMPLETE")
        print(f"Remote size: {format_bytes(remote_size)}")
        print(f"Local size:  {format_bytes(local_size)}")
        print()
        print(
            "Keep the .part file and run this script again. "
            "It will attempt to resume."
        )
        return False

    PART_PATH.replace(FINAL_PATH)

    print()
    print("Download verification passed.")
    print(f"Final file: {FINAL_PATH.resolve()}")
    print(f"Final size: {format_bytes(FINAL_PATH.stat().st_size)}")

    return True


# ---------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------

def main() -> None:
    separator()
    print("STEP 01 — DOWNLOAD GSE275562 MULTIOME DATA")
    separator()

    print(f"GEO accession: {GEO_ACCESSION}")
    print(f"File:          {FILENAME}")
    print(f"Source:        {URL}")
    print()

    # Create the project's data directory.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {DATA_DIR.resolve()}")
    print_disk_status()
    print()

    # Obtain the expected file size from NCBI when available.
    print("Checking the remote file...")
    remote_size = get_remote_size(URL)

    print(f"Remote size: {format_bytes(remote_size)}")
    print()

    # If the final file already exists and matches the server size, do nothing.
    if FINAL_PATH.exists():
        local_size = FINAL_PATH.stat().st_size

        print("A final H5MU file already exists.")
        print(f"Local size:  {format_bytes(local_size)}")

        if remote_size is not None and local_size == remote_size:
            print()
            print("PASS: Local file size matches the remote file size.")
            print("No download is necessary.")
            print()
            print("STEP 01 COMPLETE")
            separator()
            return

        if remote_size is None and local_size > 0:
            print()
            print(
                "WARNING: The local file exists, but the remote size could not "
                "be obtained automatically."
            )
            print(
                "The script will not overwrite the existing final file."
            )
            print()
            print("Review the file before proceeding.")
            sys.exit(1)

        print()
        print(
            "WARNING: Existing final file size does not match the remote size."
        )
        print(
            "Rename or remove the existing final file before attempting a "
            "fresh/resumed download."
        )
        sys.exit(1)

    # Estimate how much additional space is required.
    partial_size = PART_PATH.stat().st_size if PART_PATH.exists() else 0

    if remote_size is not None:
        remaining = max(remote_size - partial_size, 0)

        print(f"Already downloaded: {format_bytes(partial_size)}")
        print(f"Still required:     {format_bytes(remaining)}")
        print()

        if not enough_disk_space(remaining):
            print("ERROR: Insufficient free disk space.")
            print()
            print_disk_status()
            print()
            print(
                "Free additional space or change DATA_DIR before downloading."
            )
            sys.exit(1)

    # Download.
    print("Starting download from NCBI GEO...")
    print("Press Ctrl+C if necessary; rerunning the script will resume the .part file.")
    print()

    try:
        download_file(URL, remote_size)

    except KeyboardInterrupt:
        print()
        print()
        print("Download interrupted by user.")
        print(f"Partial data retained at: {PART_PATH.resolve()}")
        print("Run the same script again to resume.")
        sys.exit(130)

    except (HTTPError, URLError, TimeoutError) as exc:
        print()
        print(f"NETWORK ERROR: {exc}")
        print()
        if PART_PATH.exists():
            print(f"Partial data retained at: {PART_PATH.resolve()}")
            print("Run the same script again to resume.")
        sys.exit(1)

    # Verify and rename the finished download.
    success = verify_and_finalize(remote_size)

    if not success:
        sys.exit(1)

    print()
    print("STEP 01 COMPLETE")
    print()
    print("The processed paired RNA + ATAC H5MU dataset is ready.")
    print("Next project step: 02_phase_f0_audit.py")
    separator()


if __name__ == "__main__":
    main()

