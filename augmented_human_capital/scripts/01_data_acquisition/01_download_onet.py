#!/usr/bin/env python3
"""
Phase 0 — S0.2: Download O*NET 30.2 database files.
Downloads all task, ability, skill, and activity data needed for AHC index construction.
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ONET_DIR = PROJECT_ROOT / "data" / "raw" / "onet"
ONET_DIR.mkdir(parents=True, exist_ok=True)

# O*NET Database 30.2 bulk download URL
ONET_DB_URL = "https://www.onetcenter.org/dl_files/database/db_30_2_text.zip"

# Individual files we need (inside the ZIP)
REQUIRED_FILES = [
    "Task Statements.txt",
    "Task Ratings.txt",
    "Abilities.txt",
    "Skills.txt",
    "Knowledge.txt",
    "Work Activities.txt",
    "Detailed Work Activities.txt",
    "DWA Reference.txt",
    "Occupation Data.txt",
    "Technology Skills.txt",
    "Work Context.txt",
    "Interests.txt",
    "Work Styles.txt",
    "Content Model Reference.txt",
    "Scales Reference.txt",
]

# Crosswalk files
CROSSWALK_URLS = {
    "soc_isco_crosswalk.xls": "https://www.bls.gov/soc/isco_soc_crosswalk.xls",
    # 2018 SOC to ISCO-08 from BLS
}


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar."""
    if dest.exists():
        print(f"  [SKIP] Already exists: {dest.name}")
        return True

    print(f"  [DOWNLOAD] {desc or dest.name}...")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(dest, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_onet_database():
    """Download and extract O*NET database."""
    zip_path = ONET_DIR / "db_30_2_text.zip"

    # Check if already extracted
    marker = ONET_DIR / "Occupation Data.txt"
    if marker.exists():
        print("[SKIP] O*NET database already extracted")
        return True

    if not download_file(ONET_DB_URL, zip_path, "O*NET 30.2 Database"):
        # Try alternate version
        alt_url = "https://www.onetcenter.org/dl_files/database/db_30_1_text.zip"
        zip_path_alt = ONET_DIR / "db_30_1_text.zip"
        if not download_file(alt_url, zip_path_alt, "O*NET 30.1 Database (fallback)"):
            print("[ERROR] Could not download O*NET database")
            return False
        zip_path = zip_path_alt

    print("  [EXTRACT] Extracting O*NET database...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            # O*NET zips have a subdirectory; extract all to onet dir
            members = z.namelist()
            for member in tqdm(members, desc="Extracting"):
                # Flatten directory structure
                filename = os.path.basename(member)
                if filename and filename.endswith(".txt"):
                    with z.open(member) as src, open(ONET_DIR / filename, "wb") as dst:
                        dst.write(src.read())
        print(f"  [OK] Extracted {len([m for m in members if m.endswith('.txt')])} files")
        return True
    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}")
        return False


def download_crosswalks():
    """Download SOC-ISCO crosswalk from BLS."""
    crosswalk_dir = PROJECT_ROOT / "data" / "crosswalks"
    crosswalk_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in CROSSWALK_URLS.items():
        download_file(url, crosswalk_dir / filename, f"Crosswalk: {filename}")


def validate_downloads():
    """Check that all required files are present."""
    print("\n--- Validation ---")
    all_ok = True
    for fname in REQUIRED_FILES:
        path = ONET_DIR / fname
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {fname}")
            all_ok = False
    return all_ok


def summarize_onet():
    """Quick summary of downloaded O*NET data."""
    import pandas as pd

    print("\n--- O*NET Data Summary ---")

    for fname in ["Occupation Data.txt", "Task Statements.txt", "Abilities.txt", "Skills.txt"]:
        path = ONET_DIR / fname
        if path.exists():
            try:
                df = pd.read_csv(path, sep="\t", encoding="utf-8", on_bad_lines="skip")
                n_occ = df["O*NET-SOC Code"].nunique() if "O*NET-SOC Code" in df.columns else "N/A"
                print(f"  {fname}: {len(df):,} rows, {len(df.columns)} cols, {n_occ} occupations")
            except Exception as e:
                print(f"  {fname}: error reading ({e})")


def main():
    print("=" * 60)
    print("O*NET Database Download — Phase 0, Sprint 0.2")
    print("=" * 60)

    download_onet_database()
    download_crosswalks()
    ok = validate_downloads()

    try:
        summarize_onet()
    except ImportError:
        print("  (pandas not available for summary)")

    if ok:
        print("\n[SUCCESS] All O*NET files downloaded and validated.")
    else:
        # Check if critical files are present (non-critical can be missing)
        critical = ["Task Statements.txt", "Abilities.txt", "Occupation Data.txt", "Task Ratings.txt"]
        critical_ok = all((ONET_DIR / f).exists() for f in critical)
        if critical_ok:
            print("\n[OK] Critical O*NET files present (some optional files missing).")
        else:
            print("\n[WARNING] Critical O*NET files missing. Check logs above.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
