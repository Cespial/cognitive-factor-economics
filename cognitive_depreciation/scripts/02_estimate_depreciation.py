#!/usr/bin/env python3
"""
Sprint 3 — Estimate endogenous depreciation from GEIH data.
Key test: γ₄ (Ω̇ × Experience) < 0 — faster AI → lower experience returns.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
P1 = PROJECT / "data" / "paper1"
PROCESSED = PROJECT / "data" / "processed"
OUTPUT = PROJECT / "output" / "tables"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load GEIH + AHC + Ω̇ merged dataset."""
    est = pd.read_parquet(P1 / "processed" / "estimation_sample.parquet")
    est = est[est["log_income"] > 0].copy()

    # Load Ω̇ by SOC
    omega = pd.read_csv(PROCESSED / "omega_dot_by_occupation.csv")

    # Map SOC → CIUO via major group
    soc_to_ciuo1 = {
        "11": "1", "13": "2", "15": "2", "17": "2", "19": "2",
        "21": "2", "23": "2", "25": "2", "27": "2", "29": "2",
        "31": "3", "33": "5", "35": "5", "37": "9", "39": "5",
        "41": "5", "43": "4", "45": "6", "47": "7", "49": "7",
        "51": "8", "53": "8", "55": "0",
    }

    omega["SOC_2d"] = omega["SOC"].str[:2]
    omega["CIUO_1d"] = omega["SOC_2d"].map(soc_to_ciuo1)
    omega_ciuo = omega.groupby("CIUO_1d")["omega_dot"].mean().reset_index()

    est["CIUO_1d"] = est["CIUO_4d"].str[0]
    est = est.merge(omega_ciuo, on="CIUO_1d", how="left")

    # Also load AHC v2
    ahc = pd.read_csv(P1 / "indices" / "ahc_index_v2_improved_crosswalk.csv")
    ahc_ciuo1 = ahc.groupby(ahc["CIUO_code"].astype(str).str[0])["AHC_score"].mean().reset_index()
    ahc_ciuo1.columns = ["CIUO_1d", "AHC_occ"]
    est = est.merge(ahc_ciuo1, on="CIUO_1d", how="left")

    # Standardize
    for col in ["omega_dot", "AHC_occ"]:
        m, s = est[col].mean(), est[col].std()
        est[f"{col}_z"] = (est[col] - m) / s if s > 0 else 0

    # Interaction terms
    est["omega_x_exp"] = est["omega_dot_z"] * est["experience"]
    est["ahc_x_exp"] = est.get("AHC_occ_z", pd.Series(0, index=est.index)) * est["experience"]
    est["omega_x_exp_sq"] = est["omega_dot_z"] * est["experience_sq"]

    sample = est.dropna(subset=["log_income", "omega_dot_z", "education_years", "experience"])
    print(f"Estimation sample: {len(sample):,}")
    print(f"  Ω̇ coverage: {sample['omega_dot_z'].notna().mean():.1%}")
    print(f"  Ω̇ mean: {sample['omega_dot'].mean():.4f}, std: {sample['omega_dot'].std():.4f}")
    return sample


def estimate_depreciation_model(sample):
    """Estimate the key depreciation equation."""
    print("\n" + "=" * 70)
    print("ENDOGENOUS DEPRECIATION MODEL")
    print("=" * 70)

    def run(data, vars_, name):
        X = sm.add_constant(data[vars_].astype(float))
        r = sm.OLS(data["log_income"], X).fit(cov_type="HC1")
        return r, name

    results = []

    # M1: Standard Mincer
    r1, _ = run(sample, ["education_years", "experience", "experience_sq"], "M1: Mincer")
    results.append(r1)

    # M2: + Ω̇ level (does AI advancement affect wages?)
    r2, _ = run(sample, ["education_years", "experience", "experience_sq",
                          "omega_dot_z"], "M2: + Ω̇")
    results.append(r2)

    # M3: + Ω̇ × Experience (THE KEY TEST)
    r3, _ = run(sample, ["education_years", "experience", "experience_sq",
                          "omega_dot_z", "omega_x_exp"], "M3: + Ω̇×Exp")
    results.append(r3)

    # M4: + AHC (augmentability control)
    r4, _ = run(sample, ["education_years", "experience", "experience_sq",
                          "omega_dot_z", "omega_x_exp",
                          "AHC_occ_z", "ahc_x_exp"], "M4: + AHC×Exp")
    results.append(r4)

    # M5: + controls
    r5, _ = run(sample, ["education_years", "experience", "experience_sq",
                          "omega_dot_z", "omega_x_exp",
                          "AHC_occ_z", "ahc_x_exp",
                          "female", "urban", "formal"], "M5: + Controls")
    results.append(r5)

    # Print results
    names = ["M1: Mincer", "M2: + Ω̇", "M3: + Ω̇×Exp", "M4: + AHC×Exp", "M5: + Controls"]
    key_vars = ["experience", "omega_dot_z", "omega_x_exp", "AHC_occ_z", "ahc_x_exp", "education_years"]

    print(f"\n{'Variable':<20s}", end="")
    for n in names:
        print(f" {n[:15]:>15s}", end="")
    print()
    print("─" * (20 + 15 * len(names)))

    for var in key_vars:
        print(f"{var:<20s}", end="")
        for r in results:
            if var in r.params:
                b = r.params[var]
                p = r.pvalues[var]
                s = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "   "
                print(f" {b:>+11.4f}{s:3s}", end="")
            else:
                print(f" {'':>15s}", end="")
        print()

    print(f"{'R²':<20s}", end="")
    for r in results:
        print(f" {r.rsquared:>14.4f}", end="")
    print()
    print(f"{'N':<20s}", end="")
    for r in results:
        print(f" {r.nobs:>14,.0f}", end="")
    print()

    # THE KEY RESULT
    gamma4 = r5.params.get("omega_x_exp", 0)
    p_gamma4 = r5.pvalues.get("omega_x_exp", 1)
    print(f"\n>>> KEY TEST: γ₄ (Ω̇ × Experience) = {gamma4:+.4f} (p={p_gamma4:.4f})")
    if gamma4 < 0 and p_gamma4 < 0.05:
        print(">>> CONFIRMED: Faster AI advancement → lower experience returns (endogenous depreciation)")
    elif gamma4 < 0 and p_gamma4 < 0.10:
        print(">>> MARGINALLY CONFIRMED (p < 0.10)")
    else:
        print(">>> NOT CONFIRMED at conventional levels")

    # Implied depreciation rates
    beta_exp = r5.params.get("experience", 0)
    print(f"\n--- Implied Depreciation Rates ---")
    print(f"  Base experience return (β₂): {beta_exp:.4f}")
    print(f"  Ω̇ interaction (γ₄): {gamma4:.4f}")

    omega_values = sample["omega_dot"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    omega_z_values = sample["omega_dot_z"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])

    print(f"\n  Experience return by Ω̇ percentile:")
    print(f"  {'Percentile':>12s} {'Ω̇':>8s} {'Exp return':>12s} {'Implied δ':>10s} {'Half-life':>10s}")
    for pct, (omega, omega_z) in zip([10, 25, 50, 75, 90],
                                       zip(omega_values, omega_z_values)):
        exp_return = beta_exp + gamma4 * omega_z
        implied_delta = max(0.001, -exp_return)  # δ ≈ -d(ln w)/d(exp)
        half_life = np.log(2) / implied_delta if implied_delta > 0 else 999
        print(f"  {pct:>10d}th {omega:>8.4f} {exp_return:>+12.4f} {implied_delta:>10.4f} {half_life:>8.1f} yr")

    return results


def heterogeneity(sample):
    """Estimate by formality and education level."""
    print("\n" + "=" * 70)
    print("HETEROGENEITY: Formal vs Informal")
    print("=" * 70)

    vars_ = ["education_years", "experience", "experience_sq",
             "omega_dot_z", "omega_x_exp", "AHC_occ_z", "ahc_x_exp",
             "female", "urban"]

    for label, mask in [("Formal", sample["formal"] == 1),
                         ("Informal", sample["formal"] == 0)]:
        sub = sample[mask]
        avail = [v for v in vars_ if v in sub.columns]
        X = sm.add_constant(sub[avail].astype(float))
        r = sm.OLS(sub["log_income"], X).fit(cov_type="HC1")
        g4 = r.params.get("omega_x_exp", 0)
        p4 = r.pvalues.get("omega_x_exp", 1)
        stars = "***" if p4 < 0.01 else "**" if p4 < 0.05 else "*" if p4 < 0.10 else ""
        print(f"  {label:10s}: γ₄ = {g4:+.4f}{stars} (p={p4:.4f}, N={len(sub):,})")

    print("\n--- By Age Cohort ---")
    for label, lo, hi in [("18-30", 18, 30), ("31-45", 31, 45), ("46-65", 46, 65)]:
        sub = sample[(sample["age"] >= lo) & (sample["age"] <= hi)]
        avail = [v for v in vars_ if v in sub.columns]
        X = sm.add_constant(sub[avail].astype(float))
        r = sm.OLS(sub["log_income"], X).fit(cov_type="HC1")
        g4 = r.params.get("omega_x_exp", 0)
        p4 = r.pvalues.get("omega_x_exp", 1)
        stars = "***" if p4 < 0.01 else "**" if p4 < 0.05 else "*" if p4 < 0.10 else ""
        print(f"  Age {label:5s}: γ₄ = {g4:+.4f}{stars} (p={p4:.4f}, N={len(sub):,})")


def main():
    print("=" * 70)
    print("SPRINT 3 — Endogenous Depreciation Estimation")
    print("=" * 70)

    sample = load_data()
    results = estimate_depreciation_model(sample)
    heterogeneity(sample)

    return 0


if __name__ == "__main__":
    sys.exit(main())
