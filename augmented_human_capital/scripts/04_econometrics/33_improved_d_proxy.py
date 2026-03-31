#!/usr/bin/env python3
"""
Phase 4 — Improved AI Adoption Proxy.

The EAM-based sector mapping lost 91% of variation (median fill).
Alternative: construct D from GEIH-observable sector characteristics
that correlate with AI adoption:
  - Sector formality rate (formal firms adopt more tech)
  - Sector mean education (educated workforce → more AI tools)
  - Sector mean income (higher productivity → more investment)
  - Sector firm size distribution (large firms adopt first)

Then re-estimate the augmented Mincer equation.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
UPSTREAM = PROJECT_ROOT / "data" / "upstream_auto_col"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"


def construct_better_d_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct D proxy at sector × occupation-group level.
    Uses within-sector variation to get better identification.
    """
    # Strategy: D varies at sector × CIUO-2digit level
    # This gives ~20 sectors × ~40 occupation groups = ~800 cells
    df["CIUO_2d"] = df["CIUO_4d"].str[:2]
    df["sector_occ"] = df["sector"] + "_" + df["CIUO_2d"]

    # Compute cell-level characteristics that proxy AI adoption
    cell_stats = df.groupby("sector_occ").agg(
        cell_formal_rate=("formal", "mean"),
        cell_mean_education=("education_years", "mean"),
        cell_mean_income=("income", "mean"),
        cell_large_firm_share=("firm_size", lambda x: (x == "Grande (201+)").mean() if x.dtype == object else 0),
        cell_n=("weight", "count"),
    ).reset_index()

    # D = PCA-style composite of these indicators
    # Simple weighted average (normalized components)
    for col in ["cell_formal_rate", "cell_mean_education", "cell_mean_income", "cell_large_firm_share"]:
        mean = cell_stats[col].mean()
        std = cell_stats[col].std()
        if std > 0:
            cell_stats[f"{col}_z"] = (cell_stats[col] - mean) / std
        else:
            cell_stats[f"{col}_z"] = 0

    # Composite D proxy (equal weights on standardized components)
    cell_stats["D_composite"] = (
        0.30 * cell_stats["cell_formal_rate_z"] +
        0.25 * cell_stats["cell_mean_education_z"] +
        0.20 * cell_stats["cell_mean_income_z"] +
        0.25 * cell_stats["cell_large_firm_share_z"]
    )

    # Also use sector-level Frey-Osborne as proxy for automation exposure (inverse = AI complement)
    sector_auto = df.groupby("sector_occ")["automation_prob"].mean().reset_index()
    sector_auto.columns = ["sector_occ", "cell_auto_prob"]
    cell_stats = cell_stats.merge(sector_auto, on="sector_occ", how="left")

    # Alternative D: use automation_prob as proxy (sectors with more automatable jobs → more AI investment)
    cell_stats["D_auto"] = cell_stats["cell_auto_prob"]

    df = df.merge(cell_stats[["sector_occ", "D_composite", "D_auto", "cell_formal_rate",
                               "cell_mean_education", "cell_large_firm_share"]], on="sector_occ", how="left")

    # Log transform and standardize
    for d_col in ["D_composite", "D_auto"]:
        df[f"ln_{d_col}"] = df[d_col]  # Already standardized for composite
        mean = df[d_col].mean()
        std = df[d_col].std()
        if std > 0:
            df[f"{d_col}_z"] = (df[d_col] - mean) / std
        else:
            df[f"{d_col}_z"] = 0

    print(f"  D_composite: mean={df['D_composite'].mean():.3f}, std={df['D_composite'].std():.3f}")
    print(f"  D_auto: mean={df['D_auto'].mean():.3f}, std={df['D_auto'].std():.3f}")
    print(f"  Unique sector_occ cells: {df['sector_occ'].nunique()}")
    print(f"  cor(D_composite, D_auto) = {df['D_composite'].corr(df['D_auto']):.3f}")

    return df


def estimate_all_specs(df: pd.DataFrame) -> list:
    """Estimate with improved D proxy."""
    results = []

    est = df.dropna(subset=["log_income", "AHC_score", "D_composite", "education_years"]).copy()
    est = est[est["log_income"] > 0]

    # Standardize AHC indices
    for col in ["AHC_score", "SUB_score", "PHY_score", "ROU_score"]:
        if col in est.columns and est[col].notna().any():
            est[f"{col}_z"] = (est[col] - est[col].mean()) / est[col].std()

    # Interaction terms with composite D
    est["AHC_x_D"] = est["AHC_score_z"] * est["D_composite_z"]
    est["ROU_x_D"] = est["ROU_score_z"] * est["D_composite_z"]
    est["SUB_x_D"] = est["SUB_score_z"] * est["D_composite_z"]

    # Interaction with automation-based D
    est["AHC_x_Dauto"] = est["AHC_score_z"] * est["D_auto_z"]
    est["ROU_x_Dauto"] = est["ROU_score_z"] * est["D_auto_z"]

    # Sector FE
    sector_dummies = pd.get_dummies(est["sector"], prefix="sec", drop_first=True, dtype=float)
    est = pd.concat([est.reset_index(drop=True), sector_dummies.reset_index(drop=True)], axis=1)
    sector_cols = list(sector_dummies.columns)

    print(f"\n  Estimation sample: {len(est):,}")

    def run_ols(y, x_vars, name, data=est):
        sample = data.dropna(subset=[y] + x_vars)
        Y = sample[y]
        X = sm.add_constant(sample[x_vars].astype(float))
        result = sm.OLS(Y, X).fit(cov_type="HC1")
        coefs = {}
        for v in x_vars:
            coefs[v] = {
                "coef": result.params.get(v, np.nan),
                "se": result.bse.get(v, np.nan),
                "t": result.tvalues.get(v, np.nan),
                "p": result.pvalues.get(v, np.nan),
            }
        return {"name": name, "n": len(sample), "r2": result.rsquared,
                "r2_adj": result.rsquared_adj, "coefficients": coefs}

    # M1: Standard Mincer
    r1 = run_ols("log_income", ["education_years", "experience", "experience_sq"], "M1: Mincer")
    results.append(r1)

    # M2: + AHC levels
    r2 = run_ols("log_income", [
        "education_years", "experience", "experience_sq",
        "AHC_score_z", "SUB_score_z", "ROU_score_z",
    ], "M2: + AHC levels")
    results.append(r2)

    # M3a: + Composite D interactions
    r3a = run_ols("log_income", [
        "education_years", "experience", "experience_sq",
        "AHC_score_z", "SUB_score_z", "ROU_score_z",
        "D_composite_z", "AHC_x_D", "ROU_x_D",
    ], "M3a: + D_composite interactions")
    results.append(r3a)

    # M3b: + Auto D interactions
    r3b = run_ols("log_income", [
        "education_years", "experience", "experience_sq",
        "AHC_score_z", "SUB_score_z", "ROU_score_z",
        "D_auto_z", "AHC_x_Dauto", "ROU_x_Dauto",
    ], "M3b: + D_auto interactions")
    results.append(r3b)

    # M4: + Controls
    r4 = run_ols("log_income", [
        "education_years", "experience", "experience_sq",
        "AHC_score_z", "SUB_score_z", "ROU_score_z",
        "D_composite_z", "AHC_x_D", "ROU_x_D",
        "female", "urban",
    ], "M4: + Controls")
    results.append(r4)

    # M5: + Sector FE
    m5_vars = [
        "education_years", "experience", "experience_sq",
        "AHC_score_z", "SUB_score_z", "ROU_score_z",
        "D_composite_z", "AHC_x_D", "ROU_x_D",
        "female", "urban",
    ] + sector_cols[:15]
    r5 = run_ols("log_income", m5_vars, "M5: + Sector FE")
    results.append(r5)

    # M6: Formal only
    formal = est[est["formal"] == 1]
    r6 = run_ols("log_income", [
        "education_years", "experience", "experience_sq",
        "AHC_score_z", "SUB_score_z", "ROU_score_z",
        "D_composite_z", "AHC_x_D", "ROU_x_D",
        "female", "urban",
    ], "M6: Formal only", data=formal)
    results.append(r6)

    # M7: Informal only
    informal = est[est["formal"] == 0]
    r7 = run_ols("log_income", [
        "education_years", "experience", "experience_sq",
        "AHC_score_z", "SUB_score_z", "ROU_score_z",
        "D_composite_z", "AHC_x_D", "ROU_x_D",
        "female", "urban",
    ], "M7: Informal only", data=informal)
    results.append(r7)

    return results


def print_results(results):
    print("\n" + "=" * 80)
    print("RESULTS — Augmented Mincer with Improved D Proxy")
    print("=" * 80)

    key_vars = ["AHC_score_z", "SUB_score_z", "ROU_score_z",
                "D_composite_z", "D_auto_z",
                "AHC_x_D", "ROU_x_D", "SUB_x_D",
                "AHC_x_Dauto", "ROU_x_Dauto",
                "education_years", "female", "urban"]

    for r in results:
        name = r["name"]
        print(f"\n{'─'*70}")
        print(f"  {name}  (N={r['n']:,}, R²={r['r2']:.4f}, R²adj={r['r2_adj']:.4f})")
        print(f"{'─'*70}")

        coefs = r["coefficients"]
        for var in key_vars:
            if var in coefs:
                c = coefs[var]
                stars = "***" if c["p"] < 0.01 else "**" if c["p"] < 0.05 else "*" if c["p"] < 0.10 else "   "
                print(f"  {var:22s}  {c['coef']:9.4f}{stars}  ({c['se']:.4f})  t={c['t']:6.2f}")

        # CFE signature test
        for d_int, r_int in [("AHC_x_D", "ROU_x_D"), ("AHC_x_Dauto", "ROU_x_Dauto")]:
            if d_int in coefs and r_int in coefs:
                b2 = coefs[d_int]
                b3 = coefs[r_int]
                print(f"\n  >>> CFE TEST ({d_int} / {r_int}):")
                s2 = "✓" if b2["coef"] > 0 and b2["p"] < 0.10 else "✗"
                s3 = "✓" if b3["coef"] < 0 and b3["p"] < 0.10 else "✗"
                print(f"      β₂={b2['coef']:+.4f} (p={b2['p']:.3f}) [{s2}]  β₃={b3['coef']:+.4f} (p={b3['p']:.3f}) [{s3}]")


def save_tables(results):
    rows = []
    for r in results:
        row = {"model": r["name"], "n": r["n"], "r2": r["r2"], "r2_adj": r["r2_adj"]}
        for var, c in r["coefficients"].items():
            stars = "***" if c["p"] < 0.01 else "**" if c["p"] < 0.05 else "*" if c["p"] < 0.10 else ""
            row[f"{var}_b"] = round(c["coef"], 4)
            row[f"{var}_se"] = round(c["se"], 4)
            row[f"{var}_p"] = round(c["p"], 4)
            row[f"{var}_sig"] = stars
        rows.append(row)

    pd.DataFrame(rows).to_csv(TABLE_DIR / "table4_augmented_mincer_v2.csv", index=False)
    print(f"\n  [SAVED] table4_augmented_mincer_v2.csv")


def main():
    print("=" * 80)
    print("AUGMENTED MINCER — Improved D Proxy (sector × occupation)")
    print("=" * 80)

    df = pd.read_parquet(PROCESSED_DIR / "estimation_sample.parquet")
    print(f"  Sample: {len(df):,}")

    print("\n--- Constructing Improved D Proxy ---")
    df = construct_better_d_proxy(df)

    results = estimate_all_specs(df)
    print_results(results)
    save_tables(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
