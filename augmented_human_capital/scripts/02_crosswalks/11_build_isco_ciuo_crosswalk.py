#!/usr/bin/env python3
"""
Phase 1 — S1.2: Build ISCO-08 → CIUO-08 AC crosswalk.
CIUO-08 AC is Colombia's direct adaptation of ISCO-08, so this is mostly 1:1.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CROSSWALK_DIR = PROJECT_ROOT / "data" / "crosswalks"
UPSTREAM = PROJECT_ROOT / "data" / "upstream_auto_col"
OUTPUT = CROSSWALK_DIR / "isco_ciuo.parquet"


# ISCO-08 Major Groups → CIUO-08 AC (Colombia) — direct 1:1 at major group level
# At 4-digit level, CIUO-08 AC adds Colombian-specific codes but maintains ISCO structure
ISCO_MAJOR_GROUPS = {
    "0": {"title_en": "Armed Forces Occupations", "title_es": "Fuerzas Militares"},
    "1": {"title_en": "Managers", "title_es": "Directores y Gerentes"},
    "2": {"title_en": "Professionals", "title_es": "Profesionales Científicos e Intelectuales"},
    "3": {"title_en": "Technicians and Associate Professionals", "title_es": "Técnicos y Profesionales de Nivel Medio"},
    "4": {"title_en": "Clerical Support Workers", "title_es": "Personal de Apoyo Administrativo"},
    "5": {"title_en": "Service and Sales Workers", "title_es": "Trabajadores de Servicios y Vendedores"},
    "6": {"title_en": "Skilled Agricultural Workers", "title_es": "Agricultores y Trabajadores Calificados"},
    "7": {"title_en": "Craft and Related Trades Workers", "title_es": "Oficiales, Operarios y Artesanos"},
    "8": {"title_en": "Plant and Machine Operators", "title_es": "Operadores de Instalaciones y Máquinas"},
    "9": {"title_en": "Elementary Occupations", "title_es": "Ocupaciones Elementales"},
}


def extract_ciuo_from_geih() -> pd.DataFrame:
    """Extract unique CIUO codes from GEIH data to build empirical crosswalk."""
    geih_path = UPSTREAM / "automation_analysis_dataset.csv"
    if not geih_path.exists():
        print("  [WARN] GEIH data not found at upstream path")
        return pd.DataFrame()

    print("  Loading GEIH data for CIUO code extraction...")
    df = pd.read_csv(geih_path, usecols=["OFICIO_C8", "ocu_1d", "ocu_2d", "ocu_group", "weight"])

    # OFICIO_C8 is 4-digit CIUO-08 AC code
    df["CIUO_4d"] = df["OFICIO_C8"].astype(str).str.zfill(4)
    df["CIUO_2d"] = df["CIUO_4d"].str[:2]
    df["CIUO_1d"] = df["CIUO_4d"].str[:1]

    # Aggregate by 4-digit code with weighted employment
    occ_summary = df.groupby(["CIUO_4d", "CIUO_2d", "CIUO_1d"]).agg(
        n_workers=("weight", "sum"),
        n_obs=("OFICIO_C8", "count"),
    ).reset_index()

    occ_summary["employment_share"] = occ_summary["n_workers"] / occ_summary["n_workers"].sum()

    print(f"  Unique CIUO codes: {occ_summary['CIUO_4d'].nunique()} (4-digit)")
    print(f"  Total weighted employment: {occ_summary['n_workers'].sum():,.0f}")
    return occ_summary


def build_isco_ciuo_map(geih_codes: pd.DataFrame) -> pd.DataFrame:
    """
    Build ISCO-08 → CIUO-08 AC mapping.
    Since CIUO-08 AC is a direct adaptation of ISCO-08:
    - At 1-digit, 2-digit, 3-digit levels: codes are identical
    - At 4-digit level: most codes are 1:1, some Colombian additions exist
    """
    rows = []

    if geih_codes.empty:
        # Use theoretical mapping only
        for isco_1d, info in ISCO_MAJOR_GROUPS.items():
            rows.append({
                "ISCO_code": isco_1d,
                "CIUO_code": isco_1d,
                "level": "1d",
                "mapping_type": "identical",
                "title_en": info["title_en"],
                "title_es": info["title_es"],
                "confidence": "high",
                "weight": 1.0,
            })
    else:
        # Use empirical CIUO codes from GEIH
        for _, row in geih_codes.iterrows():
            ciuo_4d = row["CIUO_4d"]
            # CIUO-08 AC maintains ISCO-08 structure → same code
            isco_4d = ciuo_4d

            # Get major group info
            major = ciuo_4d[0]
            info = ISCO_MAJOR_GROUPS.get(major, {"title_en": "Unknown", "title_es": "Desconocido"})

            rows.append({
                "ISCO_code": isco_4d,
                "CIUO_code": ciuo_4d,
                "CIUO_2d": row["CIUO_2d"],
                "CIUO_1d": row["CIUO_1d"],
                "level": "4d",
                "mapping_type": "identical",
                "confidence": "high",
                "weight": 1.0,
                "n_workers": row["n_workers"],
                "employment_share": row["employment_share"],
                "title_en": info["title_en"],
                "title_es": info["title_es"],
            })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("ISCO-08 → CIUO-08 AC Crosswalk — Phase 1, Sprint 1.2")
    print("=" * 60)

    # Extract empirical CIUO codes from GEIH
    geih_codes = extract_ciuo_from_geih()

    # Build crosswalk
    crosswalk = build_isco_ciuo_map(geih_codes)

    # Save
    crosswalk.to_parquet(OUTPUT, index=False)
    print(f"\n  [SAVED] {OUTPUT.name}: {len(crosswalk)} rows")

    # Summary
    print(f"\n--- ISCO-CIUO Crosswalk Summary ---")
    print(f"  Total mappings: {len(crosswalk)}")
    print(f"  Unique ISCO codes: {crosswalk['ISCO_code'].nunique()}")
    print(f"  Unique CIUO codes: {crosswalk['CIUO_code'].nunique()}")

    if "n_workers" in crosswalk.columns:
        coverage = crosswalk["employment_share"].sum()
        print(f"  Employment coverage: {coverage:.1%}")

        # Top 10 occupations by employment
        top10 = crosswalk.nlargest(10, "n_workers")[["CIUO_code", "n_workers", "employment_share"]]
        print(f"\n  Top 10 occupations by employment:")
        for _, row in top10.iterrows():
            print(f"    CIUO {row['CIUO_code']}: {row['n_workers']:,.0f} workers ({row['employment_share']:.1%})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
