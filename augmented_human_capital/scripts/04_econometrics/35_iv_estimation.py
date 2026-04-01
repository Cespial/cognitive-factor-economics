#!/usr/bin/env python3
"""
Sprint 3 — Stronger Identification: IV Estimation + Event Study + Oaxaca-Blinder

Three identification strategies:
1. IV (2SLS): Use pre-period sector characteristics as instrument for D
2. Event study: Pre/post ChatGPT (Nov 2022) wage dynamics by AHC level
3. Oaxaca-Blinder: Decompose formal/informal wage gap into AHC vs other components
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDICES_DIR = PROJECT_ROOT / "data" / "indices"
UPSTREAM = PROJECT_ROOT / "data" / "upstream_auto_col"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare():
    """Load estimation sample with AHC v2 and D proxy."""
    est = pd.read_parquet(PROCESSED_DIR / "estimation_sample.parquet")
    est = est[est["log_income"] > 0].copy()

    # Load AHC v2
    ahc_v2 = pd.read_parquet(INDICES_DIR / "ahc_index_v2_improved_crosswalk.parquet")
    est = est.merge(ahc_v2[["CIUO_code", "AHC_score", "SUB_score"]],
                    left_on="CIUO_4d", right_on="CIUO_code", how="left",
                    suffixes=("_v1", "_v2"))

    # D proxy at sector×occupation level
    est["CIUO_2d"] = est["CIUO_4d"].str[:2]
    est["sector_occ"] = est["sector"].astype(str) + "_" + est["CIUO_2d"].astype(str)

    cell = est.groupby("sector_occ").agg(
        cell_formal=("formal", "mean"),
        cell_educ=("education_years", "mean"),
        cell_income=("income", "mean"),
        cell_large=("firm_size", lambda x: (x == "Grande (201+)").mean() if x.dtype == object else 0),
    ).reset_index()
    for col in ["cell_formal", "cell_educ", "cell_income", "cell_large"]:
        m, s = cell[col].mean(), cell[col].std()
        cell[f"{col}_z"] = (cell[col] - m) / s if s > 0 else 0
    cell["D_z"] = 0.30 * cell["cell_formal_z"] + 0.25 * cell["cell_educ_z"] + \
                  0.20 * cell["cell_income_z"] + 0.25 * cell["cell_large_z"]
    est = est.merge(cell[["sector_occ", "D_z"]], on="sector_occ", how="left")

    # Standardize
    for col in ["AHC_score_v2", "SUB_score_v2"]:
        if col in est.columns and est[col].notna().any():
            m, s = est[col].mean(), est[col].std()
            est[f"{col}_z"] = (est[col] - m) / s if s > 0 else 0

    est["AHC_x_D"] = est["AHC_score_v2_z"] * est["D_z"]

    sample = est.dropna(subset=["log_income", "AHC_score_v2_z", "D_z", "education_years"])
    print(f"Estimation sample: {len(sample):,}")
    return sample


# ============================================================
# STRATEGY 1: IV Estimation (2SLS)
# ============================================================

def construct_iv(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Construct instrument for D using EAM pre-period data.

    IV = sector-level capital intensity in 2019 (pre-COVID, pre-ChatGPT)
    Relevance: sectors with higher historical capital intensity adopt more AI
    Exogeneity: 2019 capital intensity is predetermined relative to 2024 wages
    """
    eam_path = UPSTREAM / "eam_panel_constructed.csv"
    if not eam_path.exists():
        print("  [WARN] EAM not found. Using alternative IV: sector automation probability")
        # Alternative: use Frey-Osborne sector-level automation as IV
        # (predetermined measure of sector's technical automation potential)
        sector_auto = sample.groupby("sector")["automation_prob"].mean().reset_index()
        sector_auto.columns = ["sector", "IV_auto_prob"]
        m, s = sector_auto["IV_auto_prob"].mean(), sector_auto["IV_auto_prob"].std()
        sector_auto["IV_z"] = (sector_auto["IV_auto_prob"] - m) / s
        sample = sample.merge(sector_auto[["sector", "IV_z"]], on="sector", how="left")
        return sample

    eam = pd.read_csv(eam_path)

    # Pre-period: 2018-2019 (before COVID and ChatGPT)
    eam_pre = eam[eam["periodo"].isin([2018, 2019])].copy()
    eam_pre["ciiu2"] = eam_pre["ciiu4"].astype(str).str[:2]

    # Sector-level pre-period capital intensity
    iv_data = eam_pre.groupby("ciiu2").agg(
        pre_capital_intensity=("capital_intensity", "mean"),
        pre_investment_rate=("investment_rate", "mean"),
        pre_labor_productivity=("labor_productivity", "mean"),
        n_firms_pre=("nordemp", "nunique"),
    ).reset_index()

    # Standardize
    for col in ["pre_capital_intensity", "pre_investment_rate", "pre_labor_productivity"]:
        m, s = iv_data[col].mean(), iv_data[col].std()
        iv_data[f"{col}_z"] = (iv_data[col] - m) / s if s > 0 else 0

    # Primary IV: pre-period capital intensity
    iv_data["IV_z"] = iv_data["pre_capital_intensity_z"]
    # Secondary IV: pre-period investment rate
    iv_data["IV2_z"] = iv_data["pre_investment_rate_z"]

    print(f"  IV data: {len(iv_data)} sectors")

    # Map to GEIH via sector→CIIU2
    # Use RAMA2D_R4 if available
    if "RAMA2D_R4" in sample.columns:
        rama_to_ciiu = sample.groupby("RAMA2D_R4")["sector"].first().reset_index()
        rama_to_ciiu["ciiu2"] = rama_to_ciiu["RAMA2D_R4"].astype(str).str.zfill(2)
        iv_mapped = iv_data.merge(rama_to_ciiu[["ciiu2", "sector"]], on="ciiu2", how="inner")
        sample = sample.merge(iv_mapped[["sector", "IV_z", "IV2_z"]], on="sector", how="left")
    else:
        # Fallback: use sector-level automation probability as IV
        sector_auto = sample.groupby("sector")["automation_prob"].mean().reset_index()
        sector_auto.columns = ["sector", "IV_auto"]
        m, s = sector_auto["IV_auto"].mean(), sector_auto["IV_auto"].std()
        sector_auto["IV_z"] = (sector_auto["IV_auto"] - m) / s
        sample = sample.merge(sector_auto[["sector", "IV_z"]], on="sector", how="left")

    iv_coverage = sample["IV_z"].notna().mean()
    print(f"  IV coverage: {iv_coverage:.1%}")

    # Fill missing with median
    if sample["IV_z"].isna().any():
        sample["IV_z"] = sample["IV_z"].fillna(sample["IV_z"].median())

    return sample


def run_2sls(sample: pd.DataFrame):
    """Run 2SLS IV estimation."""
    print("\n" + "=" * 70)
    print("STRATEGY 1: IV ESTIMATION (2SLS)")
    print("=" * 70)

    if "IV_z" not in sample.columns:
        print("  [SKIP] No IV available")
        return None

    est = sample.dropna(subset=["IV_z", "AHC_score_v2_z", "D_z", "education_years"]).copy()

    # Construct IV interaction: AHC × IV (instrument for AHC × D)
    est["AHC_x_IV"] = est["AHC_score_v2_z"] * est["IV_z"]

    print(f"  Sample: {len(est):,}")
    print(f"  cor(IV, D) = {est['IV_z'].corr(est['D_z']):.3f}")
    print(f"  cor(AHC×IV, AHC×D) = {est['AHC_x_IV'].corr(est['AHC_x_D']):.3f}")

    # --- First Stage ---
    print("\n--- First Stage: D_z ~ IV_z + controls ---")
    fs_vars = ["IV_z", "AHC_score_v2_z", "education_years", "experience",
               "experience_sq", "female", "urban"]
    fs_available = [v for v in fs_vars if v in est.columns]
    X_fs = sm.add_constant(est[fs_available].astype(float))
    fs_result = sm.OLS(est["D_z"], X_fs).fit(cov_type="HC1")

    iv_coef = fs_result.params["IV_z"]
    iv_t = fs_result.tvalues["IV_z"]
    iv_p = fs_result.pvalues["IV_z"]
    f_stat = iv_t ** 2  # For single IV, F = t²
    print(f"  IV coefficient: {iv_coef:.4f} (t={iv_t:.2f}, p={iv_p:.4f})")
    print(f"  First-stage F-statistic: {f_stat:.1f} {'(STRONG)' if f_stat > 10 else '(WEAK)' if f_stat < 10 else ''}")

    # --- Reduced Form ---
    print("\n--- Reduced Form: ln_w ~ AHC×IV + controls ---")
    rf_vars = ["AHC_x_IV", "IV_z", "AHC_score_v2_z", "education_years",
               "experience", "experience_sq", "female", "urban"]
    rf_available = [v for v in rf_vars if v in est.columns]
    X_rf = sm.add_constant(est[rf_available].astype(float))
    rf_result = sm.OLS(est["log_income"], X_rf).fit(cov_type="HC1")
    rf_coef = rf_result.params.get("AHC_x_IV", np.nan)
    rf_p = rf_result.pvalues.get("AHC_x_IV", 1.0)
    print(f"  AHC×IV (reduced form): {rf_coef:.4f} (p={rf_p:.4f})")

    # --- 2SLS with linearmodels ---
    print("\n--- Second Stage: 2SLS ---")
    try:
        dep = est["log_income"]
        exog = est[["AHC_score_v2_z", "education_years", "experience",
                     "experience_sq", "female", "urban"]].astype(float)
        exog = sm.add_constant(exog)
        endog = est[["AHC_x_D"]].astype(float)  # Endogenous: AHC×D
        instruments = est[["AHC_x_IV"]].astype(float)  # Instrument: AHC×IV

        iv_model = IV2SLS(dep, exog, endog, instruments).fit(cov_type="robust")
        print(f"\n  2SLS Results:")
        print(f"  AHC×D (instrumented): {iv_model.params['AHC_x_D']:.4f} "
              f"(se={iv_model.std_errors['AHC_x_D']:.4f}, "
              f"p={iv_model.pvalues['AHC_x_D']:.4f})")
        print(f"  AHC level: {iv_model.params['AHC_score_v2_z']:.4f}")
        print(f"  Education: {iv_model.params['education_years']:.4f}")

        # Diagnostics
        print(f"\n  Diagnostics:")
        print(f"  First-stage F: {f_stat:.1f}")
        if hasattr(iv_model, 'wooldridge_overid'):
            print(f"  Wooldridge overid: {iv_model.wooldridge_overid}")

        return {
            "ols_ahc_x_d": est["AHC_x_D"].corr(est["log_income"]),
            "iv_ahc_x_d": iv_model.params["AHC_x_D"],
            "iv_se": iv_model.std_errors["AHC_x_D"],
            "iv_p": iv_model.pvalues["AHC_x_D"],
            "first_stage_f": f_stat,
            "n": len(est),
        }

    except Exception as e:
        print(f"  [ERROR] 2SLS failed: {e}")
        # Fallback: manual 2SLS
        print("\n  Fallback: Manual 2SLS (OLS on predicted values)")
        D_hat = fs_result.predict(X_fs)
        est["AHC_x_D_hat"] = est["AHC_score_v2_z"] * D_hat

        ss_vars = ["AHC_x_D_hat", "AHC_score_v2_z", "education_years",
                    "experience", "experience_sq", "female", "urban"]
        X_ss = sm.add_constant(est[ss_vars].astype(float))
        ss_result = sm.OLS(est["log_income"], X_ss).fit(cov_type="HC1")
        b = ss_result.params["AHC_x_D_hat"]
        p = ss_result.pvalues["AHC_x_D_hat"]
        print(f"  AHC×D_hat (manual 2SLS): {b:.4f} (p={p:.4f})")

        return {
            "manual_2sls_coef": b,
            "manual_2sls_p": p,
            "first_stage_f": f_stat,
            "n": len(est),
        }


# ============================================================
# STRATEGY 2: Oaxaca-Blinder Decomposition
# ============================================================

def oaxaca_blinder(sample: pd.DataFrame):
    """Decompose formal/informal wage gap into AHC vs other components."""
    print("\n" + "=" * 70)
    print("STRATEGY 2: OAXACA-BLINDER DECOMPOSITION")
    print("=" * 70)

    est = sample.dropna(subset=["log_income", "AHC_score_v2_z", "D_z", "education_years"]).copy()

    formal = est[est["formal"] == 1]
    informal = est[est["formal"] == 0]

    print(f"  Formal: {len(formal):,}, Informal: {len(informal):,}")
    print(f"  Mean log wage — Formal: {formal['log_income'].mean():.3f}, "
          f"Informal: {informal['log_income'].mean():.3f}")
    print(f"  Raw gap: {formal['log_income'].mean() - informal['log_income'].mean():.3f} "
          f"({(np.exp(formal['log_income'].mean() - informal['log_income'].mean()) - 1) * 100:.1f}%)")

    # Estimate separate regressions
    vars_ob = ["AHC_score_v2_z", "D_z", "AHC_x_D", "education_years",
               "experience", "experience_sq", "female", "urban"]
    available = [v for v in vars_ob if v in est.columns]

    X_f = sm.add_constant(formal[available].astype(float))
    X_i = sm.add_constant(informal[available].astype(float))

    res_f = sm.OLS(formal["log_income"], X_f).fit()
    res_i = sm.OLS(informal["log_income"], X_i).fit()

    # Blinder decomposition: gap = (X_f - X_i) * beta_f (explained) + X_i * (beta_f - beta_i) (unexplained)
    mean_f = X_f.mean()
    mean_i = X_i.mean()

    total_gap = formal["log_income"].mean() - informal["log_income"].mean()

    # Explained part (endowment differences)
    explained = {}
    for var in available:
        diff = mean_f[var] - mean_i[var]
        contrib = diff * res_f.params[var]
        explained[var] = contrib

    total_explained = sum(explained.values())
    total_unexplained = total_gap - total_explained

    print(f"\n--- Oaxaca-Blinder Decomposition ---")
    print(f"  Total gap: {total_gap:.4f} ({total_gap/total_gap*100:.1f}%)")
    print(f"  Explained (endowments): {total_explained:.4f} ({total_explained/total_gap*100:.1f}%)")
    print(f"  Unexplained (coefficients): {total_unexplained:.4f} ({total_unexplained/total_gap*100:.1f}%)")

    print(f"\n  Component decomposition (explained):")
    for var, contrib in sorted(explained.items(), key=lambda x: -abs(x[1])):
        pct = contrib / total_gap * 100
        print(f"    {var:22s}: {contrib:+.4f} ({pct:+5.1f}%)")

    # Key insight: how much of gap is due to AHC + AHC×D?
    ahc_contrib = explained.get("AHC_score_v2_z", 0) + explained.get("AHC_x_D", 0)
    d_contrib = explained.get("D_z", 0)
    educ_contrib = explained.get("education_years", 0)

    print(f"\n  >>> Key decomposition:")
    print(f"      AHC + AHC×D contribution: {ahc_contrib:.4f} ({ahc_contrib/total_gap*100:.1f}% of gap)")
    print(f"      D (AI adoption) contribution: {d_contrib:.4f} ({d_contrib/total_gap*100:.1f}% of gap)")
    print(f"      Education contribution: {educ_contrib:.4f} ({educ_contrib/total_gap*100:.1f}% of gap)")

    return {
        "total_gap": total_gap,
        "explained": total_explained,
        "unexplained": total_unexplained,
        "ahc_contribution": ahc_contrib,
        "ahc_pct": ahc_contrib / total_gap * 100,
        "d_contribution": d_contrib,
        "education_contribution": educ_contrib,
        "components": explained,
    }


# ============================================================
# STRATEGY 3: Wage distribution analysis
# ============================================================

def wage_distribution_analysis(sample: pd.DataFrame):
    """Analyze how AHC reshapes the wage distribution."""
    print("\n" + "=" * 70)
    print("STRATEGY 3: WAGE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    est = sample.dropna(subset=["log_income", "AHC_score_v2_z", "D_z", "education_years"]).copy()

    # Quantile regressions with v2 AHC
    vars_qr = ["AHC_score_v2_z", "D_z", "AHC_x_D", "education_years",
                "experience", "experience_sq", "female", "urban"]
    available = [v for v in vars_qr if v in est.columns]

    Y = est["log_income"]
    X = sm.add_constant(est[available].astype(float))

    print(f"\n--- Quantile Regressions (AHC v2) ---")
    print(f"{'τ':>6s} {'AHC':>10s} {'AHC×D':>10s} {'Educ':>10s}")
    print("─" * 40)

    qr_results = []
    for tau in [0.10, 0.25, 0.50, 0.75, 0.90]:
        try:
            qr = sm.QuantReg(Y, X).fit(q=tau)
            ahc_b = qr.params.get("AHC_score_v2_z", np.nan)
            ahc_d_b = qr.params.get("AHC_x_D", np.nan)
            educ_b = qr.params.get("education_years", np.nan)
            ahc_p = qr.pvalues.get("AHC_score_v2_z", 1)
            ahc_d_p = qr.pvalues.get("AHC_x_D", 1)

            stars_ahc = "***" if ahc_p < 0.01 else "**" if ahc_p < 0.05 else "*" if ahc_p < 0.10 else "   "
            stars_d = "***" if ahc_d_p < 0.01 else "**" if ahc_d_p < 0.05 else "*" if ahc_d_p < 0.10 else "   "

            print(f"{tau:6.2f} {ahc_b:+8.4f}{stars_ahc} {ahc_d_b:+8.4f}{stars_d} {educ_b:+8.4f}")

            qr_results.append({
                "tau": tau,
                "AHC_coef": ahc_b, "AHC_p": ahc_p,
                "AHC_x_D_coef": ahc_d_b, "AHC_x_D_p": ahc_d_p,
                "education_coef": educ_b,
            })
        except Exception as e:
            print(f"{tau:6.2f} ERROR: {e}")

    # Key insight: does AHC×D increase across quantiles?
    if len(qr_results) >= 3:
        low = qr_results[0]["AHC_x_D_coef"]
        high = qr_results[-1]["AHC_x_D_coef"]
        print(f"\n  >>> AHC×D at τ=0.10: {low:+.4f}")
        print(f"  >>> AHC×D at τ=0.90: {high:+.4f}")
        if high > low:
            print(f"  >>> Augmentation premium INCREASES across distribution (inequality-increasing)")
        else:
            print(f"  >>> Augmentation premium DECREASES across distribution (inequality-reducing)")

    return qr_results


def save_sprint3_results(iv_results, ob_results, qr_results):
    """Save all Sprint 3 results."""
    import json

    results = {
        "iv_estimation": iv_results or {},
        "oaxaca_blinder": {k: v for k, v in (ob_results or {}).items() if k != "components"},
        "quantile_regressions": qr_results or [],
    }

    path = TABLE_DIR / "sprint3_identification_results.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  [SAVED] {path.name}")


def main():
    print("=" * 70)
    print("SPRINT 3 — STRONGER IDENTIFICATION")
    print("=" * 70)

    sample = load_and_prepare()
    sample = construct_iv(sample)

    iv_results = run_2sls(sample)
    ob_results = oaxaca_blinder(sample)
    qr_results = wage_distribution_analysis(sample)

    save_sprint3_results(iv_results, ob_results, qr_results)

    print("\n" + "=" * 70)
    print("SPRINT 3 COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
