#!/usr/bin/env python3
"""
Phase 0 — S0.1: Environment setup and validation.
Creates conda env, installs packages, validates data paths.
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPSTREAM = Path("/Users/cristianespinal/Claude Code/Projects/Research/Cognitive Factor Economics (CFE)/automatizacion_colombia/data")

REQUIRED_PACKAGES = [
    "pandas>=2.2", "numpy>=1.26", "scipy>=1.12", "statsmodels>=0.14",
    "linearmodels>=6.0", "scikit-learn>=1.4", "matplotlib>=3.8",
    "seaborn>=0.13", "duckdb>=0.10", "pyarrow>=15.0", "requests>=2.31",
    "beautifulsoup4>=4.12", "tqdm>=4.66", "openpyxl>=3.1",
    "anthropic>=0.40", "openai>=1.50", "rapidfuzz>=3.6",
    "xlrd>=2.0",
]


def check_python():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    assert v.major == 3 and v.minor >= 11, "Requires Python >= 3.11"


def install_packages():
    for pkg in REQUIRED_PACKAGES:
        name = pkg.split(">=")[0].split(">=")[0]
        try:
            __import__(name.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


def validate_upstream():
    print(f"\nValidating upstream data at: {UPSTREAM}")
    expected = [
        "automation_analysis_dataset.csv",
        "eam_panel_constructed.csv",
        "international_panel.csv",
    ]
    found = []
    missing = []
    for f in expected:
        if (UPSTREAM / f).exists():
            found.append(f)
        else:
            missing.append(f)

    print(f"  Found: {len(found)}/{len(expected)}")
    if missing:
        print(f"  Missing: {missing}")
    return len(missing) == 0


def create_symlink():
    link = DATA_DIR / "upstream_auto_col"
    if link.exists() or link.is_symlink():
        print(f"Symlink already exists: {link}")
        return
    if UPSTREAM.exists():
        link.symlink_to(UPSTREAM)
        print(f"Created symlink: {link} -> {UPSTREAM}")
    else:
        print(f"WARNING: Upstream directory not found at {UPSTREAM}")


def validate_directories():
    dirs = [
        "data/raw/onet", "data/raw/esco", "data/raw/geih", "data/raw/eam",
        "data/raw/ifr", "data/raw/patents", "data/raw/revelio", "data/raw/pwt",
        "data/processed", "data/crosswalks", "data/indices",
        "output/tables", "output/figures", "output/logs",
    ]
    for d in dirs:
        p = PROJECT_ROOT / d
        p.mkdir(parents=True, exist_ok=True)
    print(f"Validated {len(dirs)} directories")


def create_env_template():
    env_path = PROJECT_ROOT / ".env.template"
    if not env_path.exists():
        env_path.write_text(
            "# API Keys\n"
            "ANTHROPIC_API_KEY=\n"
            "OPENAI_API_KEY=\n"
            "ONET_API_USERNAME=\n"
            "ONET_API_PASSWORD=\n"
            "\n"
            "# LLM Scoring Config\n"
            "LLM_PROVIDER=anthropic  # anthropic | openai | ollama\n"
            "LLM_MODEL=claude-sonnet-4-6\n"
            "LLM_BATCH_SIZE=50\n"
            "LLM_MAX_RETRIES=3\n"
            "\n"
            "# Paths\n"
            "PROJECT_ROOT=/Users/cristianespinal/Claude Code/Projects/Research/augmented_human_capital\n"
        )
        print(f"Created .env.template at {env_path}")


def main():
    print("=" * 60)
    print("AHC Project — Environment Setup")
    print("=" * 60)

    check_python()
    validate_directories()
    install_packages()
    create_symlink()
    validate_upstream()
    create_env_template()

    print("\n" + "=" * 60)
    print("Setup complete. Ready for overnight processing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
