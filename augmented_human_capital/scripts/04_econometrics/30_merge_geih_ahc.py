#!/usr/bin/env python3
"""
Phase 4 — S4.1: Merge AHC index onto GEIH microdata for estimation.
Constructs the estimation sample with all variables needed for the augmented Mincer equation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPSTREAM = PROJECT_ROOT / "data" / "upstream_auto_col"
INDICES_DIR = PROJECT_ROOT / "data" / "indices"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT = PROCESSED_DIR / "estimation_sample.parquet"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_geih() -> pd.DataFrame:
    """Load GEIH microdata from upstream automatizacion_colombia."""
    path = UPSTREAM / "automation_analysis_dataset.csv"
    if not path.exists():
        print(f"[ERROR] GEIH data not found at {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"  GEIH loaded: {len(df):,} observations")

    # Parse CIUO-08 4-digit code
    df["CIUO_4d"] = df["OFICIO_C8"].astype(str).str.zfill(4)
    df["CIUO_2d"] = df["CIUO_4d"].str[:2]
    df["CIUO_1d"] = df["CIUO_4d"].str[:1]

    return df


def load_ahc_index() -> pd.DataFrame:
    """Load AHC index by occupation."""
    path = INDICES_DIR / "ahc_index_by_occupation.parquet"
    if not path.exists():
        print("[WARN] AHC index not yet built. Creating placeholder merge.")
        return pd.DataFrame()

    index = pd.read_parquet(path)
    print(f"  AHC index loaded: {len(index)} occupations")
    return index


def load_eam_sector_adoption() -> pd.DataFrame:
    """Load sector-level AI/software adoption proxy from EAM panel."""
    path = UPSTREAM / "eam_panel_constructed.csv"
    if not path.exists():
        print("[WARN] EAM panel not found. Skipping D_f construction.")
        return pd.DataFrame()

    eam = pd.read_csv(path)
    print(f"  EAM loaded: {len(eam):,} firm-year observations")

    # Construct sector-level AI adoption proxy
    # Using machinery investment + software as proxy for D (digital labor stock)
    eam["ciiu2"] = eam["ciiu4"].astype(str).str[:2]

    sector_adoption = eam.groupby(["ciiu2", "periodo"]).agg(
        mean_automation_proxy=("automation_proxy", "mean"),
        mean_capital_intensity=("capital_intensity", lambda x: np.nanmean(x) if x.notna().any() else np.nan),
        total_investment=("invebrta", "sum"),
        n_firms=("nordemp", "nunique"),
        mean_labor_productivity=("labor_productivity", "mean"),
    ).reset_index()

    # Log transform
    for col in ["mean_automation_proxy", "mean_capital_intensity", "total_investment", "mean_labor_productivity"]:
        sector_adoption[f"ln_{col}"] = np.log(sector_adoption[col].clip(lower=1))

    return sector_adoption


def construct_variables(geih: pd.DataFrame, ahc: pd.DataFrame) -> pd.DataFrame:
    """Construct all estimation variables."""
    # Merge AHC index
    if not ahc.empty:
        geih = geih.merge(
            ahc[["CIUO_code", "AHC_score", "AHC_normalized", "SUB_score", "PHY_score", "ROU_score"]],
            left_on="CIUO_4d",
            right_on="CIUO_code",
            how="left",
        )
        coverage = geih["AHC_score"].notna().mean()
        print(f"  AHC merge coverage: {coverage:.1%} of workers")
    else:
        # Placeholder
        for col in ["AHC_score", "AHC_normalized", "SUB_score", "PHY_score", "ROU_score"]:
            geih[col] = np.nan

    # Education years mapping (GEIH education_num → years)
    education_years_map = {
        0: 0,   # None
        1: 5,   # Primary
        2: 9,   # Lower secondary
        3: 11,  # Upper secondary
        4: 14,  # Technical/technological
        5: 17,  # University
        6: 19,  # Postgraduate
    }
    geih["education_years"] = geih["education_num"].map(education_years_map).fillna(
        geih["education_num"] * 3  # Rough fallback
    )

    # Mincer experience (potential)
    geih["experience"] = (geih["age"] - geih["education_years"] - 6).clip(lower=0)
    geih["experience_sq"] = geih["experience"] ** 2

    # Log income (already exists, but verify)
    if "log_income" not in geih.columns:
        geih["log_income"] = np.log(geih["income"].clip(lower=1))

    # Interaction terms (will be filled once sector D is available)
    geih["AHC_x_lnD"] = np.nan  # Placeholder
    geih["ROU_x_lnD"] = np.nan  # Placeholder
    geih["SUB_x_lnD"] = np.nan  # Placeholder

    return geih


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for Table 1."""
    vars_of_interest = [
        "income", "log_income", "age", "education_years", "experience",
        "hours_worked", "formal", "female", "urban",
        "AHC_score", "SUB_score", "PHY_score", "ROU_score",
        "automation_prob",
    ]

    stats_records = []
    for var in vars_of_interest:
        if var not in df.columns:
            continue
        series = df[var].dropna()
        stats_records.append({
            "variable": var,
            "n": len(series),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "p25": series.quantile(0.25),
            "median": series.median(),
            "p75": series.quantile(0.75),
            "max": series.max(),
        })

    return pd.DataFrame(stats_records)


def apply_sample_restrictions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard sample restrictions for estimation."""
    n_start = len(df)
    restrictions = []

    # Age 18-65
    df = df[(df["age"] >= 18) & (df["age"] <= 65)]
    restrictions.append(("Age 18-65", n_start - len(df)))

    # Positive income
    n = len(df)
    df = df[df["income"] > 0]
    restrictions.append(("Positive income", n - len(df)))

    # Has occupation code
    n = len(df)
    df = df[df["CIUO_4d"].notna() & (df["CIUO_4d"] != "")]
    restrictions.append(("Valid occupation", n - len(df)))

    print(f"\n  Sample restrictions:")
    for desc, dropped in restrictions:
        print(f"    {desc}: dropped {dropped:,}")
    print(f"  Final sample: {len(df):,} (from {n_start:,})")

    return df


def main():
    print("=" * 60)
    print("GEIH-AHC Merge & Estimation Sample — Phase 4, Sprint 4.1")
    print("=" * 60)

    geih = load_geih()
    ahc = load_ahc_index()

    # Construct variables
    sample = construct_variables(geih, ahc)

    # Apply restrictions
    sample = apply_sample_restrictions(sample)

    # Save estimation sample
    sample.to_parquet(OUTPUT, index=False)
    print(f"\n  [SAVED] {OUTPUT.name}: {len(sample):,} observations")

    # Descriptive statistics
    stats = compute_descriptive_stats(sample)
    stats_path = TABLE_DIR / "descriptive_statistics.csv"
    stats.to_csv(stats_path, index=False)
    print(f"  [SAVED] {stats_path.name}")

    # Print key stats
    print(f"\n--- Sample Summary ---")
    print(stats.to_string(index=False, float_format="{:.3f}".format))

    # AHC coverage report
    if "AHC_score" in sample.columns:
        ahc_coverage = sample["AHC_score"].notna().mean()
        print(f"\n  AHC coverage: {ahc_coverage:.1%}")
        if ahc_coverage > 0:
            print(f"  AHC mean: {sample['AHC_score'].mean():.2f}")
            print(f"  AHC std: {sample['AHC_score'].std():.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
