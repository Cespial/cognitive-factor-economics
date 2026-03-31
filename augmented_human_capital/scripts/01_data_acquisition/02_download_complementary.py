#!/usr/bin/env python3
"""
Phase 0 — S0.5: Download complementary datasets.
Felten AIOE, Eloundou GPT exposure, IFR robot data, patent data.
"""

import os
import sys
import json
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DOWNLOADS = {
    # Felten et al. AIOE data (GitHub)
    "felten/aioe_scores.csv": "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/Appendix%20A%20-%20AIOE%20by%206-digit%20SOC.csv",
    "felten/aiie_scores.csv": "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/Appendix%20B%20-%20AIIE%20by%204-digit%20NAICS.csv",
    "felten/aige_scores.csv": "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/Appendix%20C%20-%20AIGE%20by%20county%20FIPS.csv",
    "felten/ability_exposure.csv": "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/Appendix%20E%20-%20Standardized%20Ability-Level%20AI%20Exposure.csv",
    "felten/lm_aioe.csv": "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/Language%20Modeling%20AIOE%20by%206-digit%20SOC.csv",
    "felten/lm_aiie.csv": "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/Language%20Modeling%20AIIE%20by%204-digit%20NAICS.csv",
    "felten/image_aioe.csv": "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/Image%20Generation%20AIOE%20by%206-digit%20SOC.csv",

    # IFR robot density (Our World in Data proxy)
    "ifr/industrial_robots_installed.csv": "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Annual%20industrial%20robots%20installed%20-%20IFR%20(2020)/Annual%20industrial%20robots%20installed%20-%20IFR%20(2020).csv",

    # ESCO bulk download info page (actual CSV needs manual download from portal)
    # We'll create a placeholder script for ESCO API access
}

# USPTO PatentsView API for AI patents
PATENTSVIEW_API = "https://api.patentsview.org/patents/query"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100:
        print(f"  [SKIP] {dest.relative_to(RAW_DIR)}")
        return True

    print(f"  [GET] {desc or dest.name}...")
    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": "AHC-Research/1.0"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"  [OK] {dest.name} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  [ERROR] {dest.name}: {e}")
        return False


def download_felten_aioe():
    """Download all Felten et al. AIOE data from GitHub."""
    print("\n--- Felten et al. AIOE Data ---")
    success = 0
    for rel_path, url in DOWNLOADS.items():
        if rel_path.startswith("felten/"):
            if download_file(url, RAW_DIR / rel_path):
                success += 1
    print(f"  Downloaded {success}/{sum(1 for k in DOWNLOADS if k.startswith('felten/'))} Felten files")


def download_ifr_robot_data():
    """Download IFR robot density data (OWID proxy)."""
    print("\n--- IFR Robot Data ---")
    for rel_path, url in DOWNLOADS.items():
        if rel_path.startswith("ifr/"):
            download_file(url, RAW_DIR / rel_path)

    # Also try to get the newer OWID GitHub data
    owid_url = "https://raw.githubusercontent.com/owid/etl/master/etl/steps/data/garden/ifr/2024-10-03/industrial_robots.csv"
    download_file(owid_url, RAW_DIR / "ifr" / "industrial_robots_owid_2024.csv", "OWID IFR 2024")


def download_patent_data():
    """Download AI patent counts by CPC class from PatentsView."""
    print("\n--- USPTO Patent Data (AI Classification) ---")
    patent_dir = RAW_DIR / "patents"
    patent_dir.mkdir(parents=True, exist_ok=True)

    output_file = patent_dir / "ai_patents_by_year_cpc.json"
    if output_file.exists():
        print("  [SKIP] Patent data already downloaded")
        return

    # Query PatentsView for AI-classified patents (CPC class G06N = Machine learning)
    # and Y10S706 = AI applications
    ai_cpc_classes = ["G06N", "G06F18", "G06V", "G10L15", "G06Q"]

    all_results = {}
    for cpc in ai_cpc_classes:
        print(f"  [QUERY] CPC {cpc}...")
        try:
            params = {
                "q": json.dumps({"_and": [
                    {"_gte": {"patent_date": "2010-01-01"}},
                    {"_begins": {"cpc_group_id": cpc}}
                ]}),
                "f": json.dumps(["patent_number", "patent_date", "patent_year"]),
                "o": json.dumps({"page": 1, "per_page": 100}),
                "s": json.dumps([{"patent_year": "desc"}])
            }
            resp = requests.get(PATENTSVIEW_API, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                count = data.get("total_patent_count", 0)
                all_results[cpc] = {"total": count, "sample": data.get("patents", [])[:5]}
                print(f"  [OK] CPC {cpc}: {count:,} patents")
            else:
                print(f"  [WARN] CPC {cpc}: HTTP {resp.status_code}")
                all_results[cpc] = {"total": 0, "error": resp.status_code}
        except Exception as e:
            print(f"  [ERROR] CPC {cpc}: {e}")
            all_results[cpc] = {"total": 0, "error": str(e)}

    output_file.write_text(json.dumps(all_results, indent=2))
    print(f"  [SAVED] {output_file.name}")


def copy_upstream_data():
    """Copy/link key datasets from automatizacion_colombia."""
    print("\n--- Upstream Data Integration ---")
    upstream = Path("/Users/cristianespinal/Claude Code/Projects/Research/Cognitive Factor Economics (CFE)/automatizacion_colombia/data")

    files_to_link = [
        "automation_analysis_dataset.csv",
        "eam_panel_constructed.csv",
        "international_panel.csv",
        "automation_occupation_summary.csv",
        "automation_sector_summary.csv",
    ]

    link_dir = PROJECT_ROOT / "data" / "upstream_auto_col"
    if not link_dir.exists() and upstream.exists():
        link_dir.symlink_to(upstream)
        print(f"  [LINK] Created symlink to upstream data")
    elif link_dir.exists():
        print(f"  [SKIP] Upstream symlink exists")
    else:
        print(f"  [WARN] Upstream data not found at {upstream}")

    # Validate key files
    for f in files_to_link:
        path = link_dir / f if link_dir.exists() else upstream / f
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {f} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {f}")


def copy_upstream_references():
    """Copy references.bib from automatizacion_colombia."""
    print("\n--- Literature References ---")
    upstream_bib = Path("/Users/cristianespinal/Claude Code/Projects/Research/Cognitive Factor Economics (CFE)/automatizacion_colombia/references.bib")
    dest_bib = PROJECT_ROOT / "literature" / "upstream_references.bib"

    if upstream_bib.exists():
        import shutil
        shutil.copy2(upstream_bib, dest_bib)
        n_entries = sum(1 for line in upstream_bib.read_text().split("\n") if line.startswith("@"))
        print(f"  [OK] Copied references.bib ({n_entries} entries)")
    else:
        print(f"  [WARN] Upstream references.bib not found")


def download_eloundou_data():
    """Download Eloundou et al. (2023) GPT exposure scores."""
    print("\n--- Eloundou et al. GPT Exposure ---")
    eloundou_dir = RAW_DIR / "eloundou"
    eloundou_dir.mkdir(parents=True, exist_ok=True)

    # The supplementary data from OpenAI/Science paper
    urls = {
        "gpt_exposure_scores.csv": "https://raw.githubusercontent.com/openai/openai-cookbook/main/data/gpts_are_gpts_supplementary.csv",
    }

    for fname, url in urls.items():
        download_file(url, eloundou_dir / fname, f"Eloundou: {fname}")

    # Create a note about where to find the full supplementary data
    note = eloundou_dir / "README.md"
    if not note.exists():
        note.write_text(
            "# Eloundou et al. (2023) Data\n\n"
            "Full supplementary data from:\n"
            "- arXiv: https://arxiv.org/abs/2303.10130\n"
            "- Science: https://www.science.org/doi/10.1126/science.adj0998\n"
            "- OpenAI: https://openai.com/index/gpts-are-gpts/\n\n"
            "If CSV download fails, extract from the paper's supplementary materials.\n"
        )


def main():
    print("=" * 60)
    print("Complementary Data Download — Phase 0, Sprint 0.5")
    print("=" * 60)

    download_felten_aioe()
    download_ifr_robot_data()
    download_patent_data()
    download_eloundou_data()
    copy_upstream_data()
    copy_upstream_references()

    print("\n" + "=" * 60)
    print("Complementary data download complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
