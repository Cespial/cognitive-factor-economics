#!/usr/bin/env python3
"""
Phase 4 — S4.2-S4.4: Estimate the Augmented Mincer Equation.

The CFE signature prediction:
  β₂ > 0 (augmentation premium: AHC × AI adoption)
  β₃ < 0 (routine displacement: ROU × AI adoption)

Models estimated:
  M1: Standard Mincer (education + experience)
  M2: + AHC, SUB, ROU, PHY (level effects)
  M3: + AHC×lnD, ROU×lnD (interaction effects — THE KEY)
  M4: + full controls (firm size, urban, gender, department)
  M5: + sector FE
  M6: Heterogeneity by formality
  M7: Quantile regressions
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
UPSTREAM = PROJECT_ROOT / "data" / "upstream_auto_col"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_estimation_sample() -> pd.DataFrame:
    path = PROCESSED_DIR / "estimation_sample.parquet"
    df = pd.read_parquet(path)
    print(f"Estimation sample: {len(df):,} observations")
    return df


def construct_ai_adoption_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct sector-level AI adoption proxy (D_s) from EAM data.
    Uses machinery/software investment intensity as proxy for digital labor stock.
    """
    eam_path = UPSTREAM / "eam_panel_constructed.csv"
    if not eam_path.exists():
        print("[WARN] EAM data not found. Using sector-level automation_prob as D proxy.")
        # Fallback: use sector mean automation probability as D proxy
        sector_d = df.groupby("sector")["automation_prob"].mean().reset_index()
        sector_d.columns = ["sector", "D_sector"]
        sector_d["ln_D"] = np.log(sector_d["D_sector"].clip(lower=0.01))
        df = df.merge(sector_d, on="sector", how="left")
        return df

    eam = pd.read_csv(eam_path)
    print(f"  EAM loaded: {len(eam):,} firm-year obs")

    # Use latest year (2024) or average of recent years
    eam_recent = eam[eam["periodo"] >= 2022].copy()
    eam_recent["ciiu2"] = eam_recent["ciiu4"].astype(str).str[:2]

    # Compute sector-level automation intensity
    sector_d = eam_recent.groupby("ciiu2").agg(
        mean_capital_intensity=("capital_intensity", "mean"),
        mean_investment_rate=("investment_rate", "mean"),
        mean_automation_proxy=("automation_proxy", "mean"),
        mean_labor_productivity=("labor_productivity", "mean"),
        n_firms=("nordemp", "nunique"),
    ).reset_index()

    # The D proxy = capital intensity (captures machinery + software per worker)
    sector_d["D_sector"] = sector_d["mean_capital_intensity"].clip(lower=1)
    sector_d["ln_D"] = np.log(sector_d["D_sector"])

    # Map CIIU 2-digit to GEIH sector names
    # GEIH sectors use text names; we need to map via RAMA2D_R4
    if "RAMA2D_R4" in df.columns:
        rama_map = df.groupby("RAMA2D_R4")["sector"].first().reset_index()
        rama_map["ciiu2"] = rama_map["RAMA2D_R4"].astype(str).str.zfill(2)
        sector_d = sector_d.merge(rama_map[["ciiu2", "sector"]], on="ciiu2", how="left")

        df = df.merge(sector_d[["sector", "D_sector", "ln_D"]], on="sector", how="left")
    else:
        # Fallback: use sector-level stats from GEIH
        sector_stats = df.groupby("sector").agg(
            mean_income=("income", "mean"),
            formal_rate=("formal", "mean"),
        ).reset_index()
        # Use formality rate × mean income as rough D proxy
        sector_stats["D_sector"] = (sector_stats["formal_rate"] * sector_stats["mean_income"]).clip(lower=1)
        sector_stats["ln_D"] = np.log(sector_stats["D_sector"])
        df = df.merge(sector_stats[["sector", "D_sector", "ln_D"]], on="sector", how="left")

    # Fill missing with median
    if df["ln_D"].isna().any():
        median_d = df["ln_D"].median()
        print(f"  Filling {df['ln_D'].isna().sum():,} missing D values with median={median_d:.2f}")
        df["ln_D"] = df["ln_D"].fillna(median_d)

    print(f"  D proxy constructed: mean ln_D={df['ln_D'].mean():.2f}, std={df['ln_D'].std():.2f}")
    return df


def construct_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Construct AHC×lnD and ROU×lnD interaction terms."""
    # Standardize indices for interpretability
    for col in ["AHC_score", "SUB_score", "PHY_score", "ROU_score"]:
        if col in df.columns and df[col].notna().any():
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f"{col}_z"] = (df[col] - mean) / std
            else:
                df[f"{col}_z"] = 0

    # Standardize ln_D
    if "ln_D" in df.columns:
        df["ln_D_z"] = (df["ln_D"] - df["ln_D"].mean()) / df["ln_D"].std()

        # Interaction terms (using standardized versions)
        df["AHC_x_lnD"] = df["AHC_score_z"] * df["ln_D_z"]
        df["ROU_x_lnD"] = df["ROU_score_z"] * df["ln_D_z"]
        df["SUB_x_lnD"] = df["SUB_score_z"] * df["ln_D_z"]

    return df


def estimate_ols(df: pd.DataFrame, y_col: str, x_cols: list, weights: str = None,
                 cluster_col: str = None, name: str = "") -> dict:
    """Estimate OLS with robust/clustered standard errors."""
    sample = df.dropna(subset=[y_col] + x_cols)
    if len(sample) < 100:
        print(f"  [SKIP] {name}: insufficient observations ({len(sample)})")
        return {}

    Y = sample[y_col]
    X = sm.add_constant(sample[x_cols])

    if weights and weights in sample.columns:
        w = sample[weights]
        model = sm.WLS(Y, X, weights=w)
    else:
        model = sm.OLS(Y, X)

    if cluster_col and cluster_col in sample.columns:
        # Clustered standard errors
        groups = sample[cluster_col]
        result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        result = model.fit(cov_type="HC1")  # Heteroskedasticity-robust

    # Extract results
    coef_dict = {}
    for var in x_cols:
        coef_dict[var] = {
            "coef": result.params.get(var, np.nan),
            "se": result.bse.get(var, np.nan),
            "t": result.tvalues.get(var, np.nan),
            "p": result.pvalues.get(var, np.nan),
        }

    return {
        "name": name,
        "n": len(sample),
        "r2": result.rsquared,
        "r2_adj": result.rsquared_adj,
        "f_stat": result.fvalue,
        "f_pval": result.f_pvalue,
        "coefficients": coef_dict,
        "result_obj": result,
    }


def run_progressive_specifications(df: pd.DataFrame) -> list:
    """Run Models M1-M5 (progressive specifications)."""
    results = []

    # Ensure we have the estimation sample
    est = df.dropna(subset=["log_income", "AHC_score", "ln_D"]).copy()
    est = est[est["log_income"] > 0]

    # Create dummy variables (ensure numeric dtype)
    if "sector" in est.columns:
        sector_dummies = pd.get_dummies(est["sector"], prefix="sec", drop_first=True, dtype=float)
        est = pd.concat([est.reset_index(drop=True), sector_dummies.reset_index(drop=True)], axis=1)
        sector_cols = list(sector_dummies.columns)
    else:
        sector_cols = []

    # Firm size dummies
    if "firm_size" in est.columns:
        size_dummies = pd.get_dummies(est["firm_size"], prefix="size", drop_first=True, dtype=float)
        est = pd.concat([est.reset_index(drop=True), size_dummies.reset_index(drop=True)], axis=1)
        size_cols = list(size_dummies.columns)
    else:
        size_cols = []

    print(f"\n  Estimation sample: {len(est):,} observations")
    print(f"  Sectors: {est['sector'].nunique() if 'sector' in est.columns else 'N/A'}")

    # M1: Standard Mincer
    m1_vars = ["education_years", "experience", "experience_sq"]
    m1_sample = est.dropna(subset=m1_vars)
    r1 = estimate_ols(m1_sample, "log_income", m1_vars, name="M1: Standard Mincer")
    results.append(r1)

    # M2: + AHC level effects
    m2_vars = m1_vars + ["AHC_score_z", "SUB_score_z", "ROU_score_z", "PHY_score_z"]
    m2_available = [v for v in m2_vars if v in est.columns]
    r2 = estimate_ols(est, "log_income", m2_available, name="M2: + AHC indices")
    results.append(r2)

    # M3: + Interaction terms (THE KEY SPECIFICATION)
    m3_vars = m2_available + ["AHC_x_lnD", "ROU_x_lnD", "ln_D_z"]
    m3_available = [v for v in m3_vars if v in est.columns]
    r3 = estimate_ols(est, "log_income", m3_available, name="M3: + AHC×D interactions")
    results.append(r3)

    # M4: + Controls
    control_vars = []
    if "female" in est.columns:
        control_vars.append("female")
    if "urban" in est.columns:
        control_vars.append("urban")
    control_vars += size_cols[:3]  # Limit to avoid too many dummies

    m4_vars = m3_available + control_vars
    r4 = estimate_ols(est, "log_income", m4_vars, name="M4: + Controls")
    results.append(r4)

    # M5: + Sector FE
    m5_vars = m4_vars + sector_cols[:15]  # Limit sectors
    r5 = estimate_ols(est, "log_income", m5_vars, name="M5: + Sector FE")
    results.append(r5)

    return results


def run_formality_heterogeneity(df: pd.DataFrame) -> list:
    """M6: Split sample by formality status."""
    results = []
    est = df.dropna(subset=["log_income", "AHC_score", "ln_D", "education_years"]).copy()

    base_vars = ["education_years", "experience", "experience_sq",
                 "AHC_score_z", "SUB_score_z", "ROU_score_z",
                 "AHC_x_lnD", "ROU_x_lnD", "ln_D_z"]
    available = [v for v in base_vars if v in est.columns]

    for label, mask in [("Formal", est["formal"] == 1), ("Informal", est["formal"] == 0)]:
        subset = est[mask]
        r = estimate_ols(subset, "log_income", available, name=f"M6: {label} workers")
        results.append(r)

    return results


def run_quantile_regressions(df: pd.DataFrame) -> list:
    """M7: Quantile regressions at τ = 0.10, 0.25, 0.50, 0.75, 0.90."""
    results = []
    est = df.dropna(subset=["log_income", "AHC_score", "ln_D", "education_years"]).copy()

    base_vars = ["education_years", "experience", "experience_sq",
                 "AHC_score_z", "AHC_x_lnD", "ROU_x_lnD", "ln_D_z"]
    available = [v for v in base_vars if v in est.columns]

    Y = est["log_income"]
    X = sm.add_constant(est[available].dropna())
    valid_idx = X.index.intersection(Y.index)
    Y = Y.loc[valid_idx]
    X = X.loc[valid_idx]

    for tau in [0.10, 0.25, 0.50, 0.75, 0.90]:
        try:
            qr = sm.QuantReg(Y, X).fit(q=tau)
            coef_dict = {}
            for var in available:
                if var in qr.params.index:
                    coef_dict[var] = {
                        "coef": qr.params[var],
                        "se": qr.bse[var],
                        "t": qr.tvalues[var],
                        "p": qr.pvalues[var],
                    }
            results.append({
                "name": f"Q{int(tau*100)}",
                "tau": tau,
                "n": len(Y),
                "coefficients": coef_dict,
            })
        except Exception as e:
            print(f"  [WARN] Quantile {tau}: {e}")

    return results


def format_results_table(results: list) -> pd.DataFrame:
    """Format regression results into a publication-style table."""
    key_vars = [
        "education_years", "experience", "experience_sq",
        "AHC_score_z", "SUB_score_z", "ROU_score_z", "PHY_score_z",
        "AHC_x_lnD", "ROU_x_lnD", "ln_D_z",
        "female", "urban",
    ]

    rows = []
    for var in key_vars:
        row = {"variable": var}
        for r in results:
            if not r:
                continue
            name = r.get("name", "")
            coefs = r.get("coefficients", {})
            if var in coefs:
                c = coefs[var]
                stars = ""
                if c["p"] < 0.01:
                    stars = "***"
                elif c["p"] < 0.05:
                    stars = "**"
                elif c["p"] < 0.10:
                    stars = "*"
                row[f"{name}_coef"] = f"{c['coef']:.4f}{stars}"
                row[f"{name}_se"] = f"({c['se']:.4f})"
            else:
                row[f"{name}_coef"] = ""
                row[f"{name}_se"] = ""
        rows.append(row)

    # Add N and R²
    n_row = {"variable": "N"}
    r2_row = {"variable": "R²"}
    r2adj_row = {"variable": "R² adj"}
    for r in results:
        if not r:
            continue
        name = r.get("name", "")
        n_row[f"{name}_coef"] = f"{r.get('n', ''):,}"
        n_row[f"{name}_se"] = ""
        r2_row[f"{name}_coef"] = f"{r.get('r2', 0):.4f}" if "r2" in r else ""
        r2_row[f"{name}_se"] = ""
        r2adj_row[f"{name}_coef"] = f"{r.get('r2_adj', 0):.4f}" if "r2_adj" in r else ""
        r2adj_row[f"{name}_se"] = ""

    rows.extend([n_row, r2_row, r2adj_row])
    return pd.DataFrame(rows)


def print_key_results(results: list):
    """Print the key coefficients for quick interpretation."""
    print("\n" + "=" * 70)
    print("KEY RESULTS — Augmented Mincer Equation")
    print("=" * 70)

    for r in results:
        if not r:
            continue
        name = r.get("name", "")
        coefs = r.get("coefficients", {})
        n = r.get("n", 0)
        r2 = r.get("r2", 0)

        print(f"\n--- {name} (N={n:,}, R²={r2:.4f}) ---")

        for var in ["AHC_score_z", "AHC_x_lnD", "ROU_x_lnD", "education_years"]:
            if var in coefs:
                c = coefs[var]
                stars = "***" if c["p"] < 0.01 else "**" if c["p"] < 0.05 else "*" if c["p"] < 0.10 else ""
                print(f"  {var:20s}: β={c['coef']:8.4f}{stars:3s}  (se={c['se']:.4f}, t={c['t']:.2f}, p={c['p']:.4f})")

        # Check CFE signature
        if "AHC_x_lnD" in coefs and "ROU_x_lnD" in coefs:
            b2 = coefs["AHC_x_lnD"]["coef"]
            b3 = coefs["ROU_x_lnD"]["coef"]
            p2 = coefs["AHC_x_lnD"]["p"]
            p3 = coefs["ROU_x_lnD"]["p"]
            sig2 = "YES" if p2 < 0.05 else "marginal" if p2 < 0.10 else "NO"
            sig3 = "YES" if p3 < 0.05 else "marginal" if p3 < 0.10 else "NO"
            print(f"\n  >>> CFE SIGNATURE TEST:")
            print(f"      β₂ (AHC×D) = {b2:.4f} (p={p2:.4f}) — Expected > 0: {'CONFIRMED' if b2 > 0 and p2 < 0.10 else 'NOT confirmed'} [{sig2}]")
            print(f"      β₃ (ROU×D) = {b3:.4f} (p={p3:.4f}) — Expected < 0: {'CONFIRMED' if b3 < 0 and p3 < 0.10 else 'NOT confirmed'} [{sig3}]")


def main():
    print("=" * 70)
    print("AUGMENTED MINCER EQUATION — Phase 4, Sprints 4.2-4.4")
    print("=" * 70)

    df = load_estimation_sample()

    # Construct AI adoption proxy
    print("\n--- Constructing AI Adoption Proxy (D_s) ---")
    df = construct_ai_adoption_proxy(df)

    # Construct interaction terms
    print("\n--- Constructing Interaction Terms ---")
    df = construct_interaction_terms(df)

    # Progressive specifications (M1-M5)
    print("\n--- Progressive OLS Specifications ---")
    ols_results = run_progressive_specifications(df)

    # Formality heterogeneity (M6)
    print("\n--- Formality Heterogeneity ---")
    formality_results = run_formality_heterogeneity(df)

    # Quantile regressions (M7)
    print("\n--- Quantile Regressions ---")
    quantile_results = run_quantile_regressions(df)

    # Print key results
    print_key_results(ols_results)
    print_key_results(formality_results)

    # Format and save tables
    print("\n--- Saving Tables ---")

    # Table 4: OLS results
    table_ols = format_results_table(ols_results)
    table_ols.to_csv(TABLE_DIR / "table4_ols_regression.csv", index=False)
    print(f"  [SAVED] table4_ols_regression.csv")

    # Table 6: Formality heterogeneity
    table_form = format_results_table(formality_results)
    table_form.to_csv(TABLE_DIR / "table6_formality_heterogeneity.csv", index=False)
    print(f"  [SAVED] table6_formality_heterogeneity.csv")

    # Table: Quantile results
    qr_rows = []
    for qr in quantile_results:
        row = {"quantile": qr.get("tau", ""), "n": qr.get("n", "")}
        for var in ["AHC_score_z", "AHC_x_lnD", "ROU_x_lnD", "education_years"]:
            if var in qr.get("coefficients", {}):
                c = qr["coefficients"][var]
                stars = "***" if c["p"] < 0.01 else "**" if c["p"] < 0.05 else "*" if c["p"] < 0.10 else ""
                row[f"{var}_coef"] = f"{c['coef']:.4f}{stars}"
                row[f"{var}_se"] = f"({c['se']:.4f})"
        qr_rows.append(row)
    if qr_rows:
        pd.DataFrame(qr_rows).to_csv(TABLE_DIR / "table_quantile_regressions.csv", index=False)
        print(f"  [SAVED] table_quantile_regressions.csv")

    # Print quantile results summary
    if quantile_results:
        print("\n--- Quantile Regression: AHC×D across distribution ---")
        for qr in quantile_results:
            coefs = qr.get("coefficients", {})
            if "AHC_x_lnD" in coefs:
                c = coefs["AHC_x_lnD"]
                stars = "***" if c["p"] < 0.01 else "**" if c["p"] < 0.05 else "*" if c["p"] < 0.10 else ""
                print(f"  τ={qr['tau']:.2f}: β₂={c['coef']:.4f}{stars} (p={c['p']:.4f})")

    print("\n" + "=" * 70)
    print("ESTIMATION COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
