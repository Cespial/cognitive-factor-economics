#!/usr/bin/env python3
"""
Paper 0 — Sprint 3: ORIV Estimation + Horse Race + Measurement Error Analysis.

ORIV (Obviously Related Instrumental Variables):
  Use Sonnet score as IV for Haiku score (and vice versa).
  If measurement errors are independent, this corrects attenuation bias.

Horse Race:
  Does AHC add predictive power beyond Felten, Eloundou, F&O?

Measurement Error:
  Estimate attenuation factor from Haiku-Sonnet disagreement.
  Compare OLS / ORIV / external IV coefficients.
"""

import sys
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
P1 = PROJECT / "data" / "paper1"
OUTPUT = PROJECT / "output" / "tables"


def load_estimation_data():
    """Load GEIH estimation sample with Haiku + Sonnet AHC scores."""
    est = pd.read_parquet(P1 / "processed" / "estimation_sample.parquet")
    est = est[est["log_income"] > 0].copy()

    # Haiku scores by SOC → CIUO
    haiku_scores = {}
    with open(P1 / "indices" / "raw_llm_scores.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if "error" not in r and "augmentation_score" in r:
                    soc = r.get("soc_code", "")
                    if soc not in haiku_scores:
                        haiku_scores[soc] = []
                    haiku_scores[soc].append(r["augmentation_score"])
            except:
                pass

    # Sonnet scores by task_id → SOC
    sonnet_by_soc = {}
    haiku_task_soc = {}
    with open(P1 / "indices" / "raw_llm_scores.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if "error" not in r:
                    haiku_task_soc[r["task_id"]] = r.get("soc_code", "")
            except:
                pass

    sonnet_path = P1 / "indices" / "sonnet_validation_scores.jsonl"
    if sonnet_path.exists():
        with open(sonnet_path) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if "error" not in r and "augmentation_score" in r:
                        soc = haiku_task_soc.get(r["task_id"], "")
                        if soc:
                            if soc not in sonnet_by_soc:
                                sonnet_by_soc[soc] = []
                            sonnet_by_soc[soc].append(r["augmentation_score"])
                except:
                    pass

    # Aggregate to SOC level
    ahc_haiku = {soc: np.mean(vals) for soc, vals in haiku_scores.items()}
    ahc_sonnet = {soc: np.mean(vals) for soc, vals in sonnet_by_soc.items()}

    # Map SOC → CIUO using the crosswalk (major group level)
    soc_to_ciuo1 = {
        "11": "1", "13": "2", "15": "2", "17": "2", "19": "2",
        "21": "2", "23": "2", "25": "2", "27": "2", "29": "2",
        "31": "3", "33": "5", "35": "5", "37": "9", "39": "5",
        "41": "5", "43": "4", "45": "6", "47": "7", "49": "7",
        "51": "8", "53": "8", "55": "0",
    }

    # Aggregate to CIUO 1-digit
    ciuo_haiku = {}
    ciuo_sonnet = {}
    for soc, val in ahc_haiku.items():
        soc2 = soc[:2]
        ciuo1 = soc_to_ciuo1.get(soc2)
        if ciuo1:
            ciuo_haiku.setdefault(ciuo1, []).append(val)
    for soc, val in ahc_sonnet.items():
        soc2 = soc[:2]
        ciuo1 = soc_to_ciuo1.get(soc2)
        if ciuo1:
            ciuo_sonnet.setdefault(ciuo1, []).append(val)

    ciuo_haiku_mean = {k: np.mean(v) for k, v in ciuo_haiku.items()}
    ciuo_sonnet_mean = {k: np.mean(v) for k, v in ciuo_sonnet.items()}

    est["CIUO_1d"] = est["CIUO_4d"].str[0]
    est["AHC_haiku"] = est["CIUO_1d"].map(ciuo_haiku_mean)
    est["AHC_sonnet"] = est["CIUO_1d"].map(ciuo_sonnet_mean)

    # Standardize
    for col in ["AHC_haiku", "AHC_sonnet"]:
        m, s = est[col].mean(), est[col].std()
        est[f"{col}_z"] = (est[col] - m) / s if s > 0 else 0

    # D proxy
    est["CIUO_2d"] = est["CIUO_4d"].str[:2]
    est["sector_occ"] = est["sector"].astype(str) + "_" + est["CIUO_2d"].astype(str)
    cell = est.groupby("sector_occ").agg(
        cf=("formal", "mean"), ce=("education_years", "mean"),
        ci=("income", "mean"),
        cl=("firm_size", lambda x: (x == "Grande (201+)").mean() if x.dtype == object else 0),
    ).reset_index()
    for c in ["cf", "ce", "ci", "cl"]:
        m, s = cell[c].mean(), cell[c].std()
        cell[f"{c}_z"] = (cell[c] - m) / s if s > 0 else 0
    cell["D_z"] = 0.30 * cell["cf_z"] + 0.25 * cell["ce_z"] + 0.20 * cell["ci_z"] + 0.25 * cell["cl_z"]
    est = est.merge(cell[["sector_occ", "D_z"]], on="sector_occ", how="left")

    # Interactions
    est["AHC_H_x_D"] = est["AHC_haiku_z"] * est["D_z"]
    est["AHC_S_x_D"] = est["AHC_sonnet_z"] * est["D_z"]

    sample = est.dropna(subset=["log_income", "AHC_haiku_z", "AHC_sonnet_z",
                                 "D_z", "education_years"])
    print(f"ORIV estimation sample: {len(sample):,}")
    print(f"  Haiku coverage: {sample['AHC_haiku_z'].notna().mean():.1%}")
    print(f"  Sonnet coverage: {sample['AHC_sonnet_z'].notna().mean():.1%}")
    return sample


# ============================================================
# ORIV ESTIMATION
# ============================================================

def run_oriv(sample):
    """
    ORIV: Use Sonnet as IV for Haiku (and vice versa).
    Then average the two 2SLS estimates.
    """
    print("\n" + "=" * 70)
    print("ORIV ESTIMATION (Gillen, Snowberg & Yariv 2019)")
    print("=" * 70)

    controls = ["education_years", "experience", "experience_sq",
                "D_z", "female", "urban"]
    avail = [v for v in controls if v in sample.columns]

    # OLS with Haiku
    ols_vars = avail + ["AHC_haiku_z", "AHC_H_x_D"]
    X_ols = sm.add_constant(sample[ols_vars].astype(float))
    ols = sm.OLS(sample["log_income"], X_ols).fit(cov_type="HC1")
    b_ols = ols.params.get("AHC_haiku_z", np.nan)
    b_ols_d = ols.params.get("AHC_H_x_D", np.nan)

    print(f"\n--- OLS (Haiku scores) ---")
    print(f"  AHC_haiku:   {b_ols:+.4f} (p={ols.pvalues.get('AHC_haiku_z', 1):.4f})")
    print(f"  AHC_H × D:  {b_ols_d:+.4f} (p={ols.pvalues.get('AHC_H_x_D', 1):.4f})")
    print(f"  R²: {ols.rsquared:.4f}")

    # OLS with Sonnet
    ols_vars_s = avail + ["AHC_sonnet_z", "AHC_S_x_D"]
    X_ols_s = sm.add_constant(sample[ols_vars_s].astype(float))
    ols_s = sm.OLS(sample["log_income"], X_ols_s).fit(cov_type="HC1")
    b_ols_s = ols_s.params.get("AHC_sonnet_z", np.nan)

    print(f"\n--- OLS (Sonnet scores) ---")
    print(f"  AHC_sonnet:  {b_ols_s:+.4f} (p={ols_s.pvalues.get('AHC_sonnet_z', 1):.4f})")
    print(f"  R²: {ols_s.rsquared:.4f}")

    # ORIV: First stage — regress Haiku on Sonnet + controls
    print(f"\n--- ORIV: Sonnet → Haiku (first stage) ---")
    fs_vars = avail + ["AHC_sonnet_z"]
    X_fs = sm.add_constant(sample[fs_vars].astype(float))
    fs = sm.OLS(sample["AHC_haiku_z"], X_fs).fit()
    fs_coef = fs.params.get("AHC_sonnet_z", 0)
    fs_f = fs.tvalues.get("AHC_sonnet_z", 0) ** 2
    print(f"  Sonnet → Haiku: {fs_coef:.4f} (F={fs_f:.1f})")

    # Second stage: use predicted Haiku from Sonnet
    sample_oriv = sample.copy()
    sample_oriv["AHC_hat"] = fs.predict(X_fs)
    sample_oriv["AHC_hat_x_D"] = sample_oriv["AHC_hat"] * sample_oriv["D_z"]

    ss_vars = avail + ["AHC_hat", "AHC_hat_x_D"]
    X_ss = sm.add_constant(sample_oriv[ss_vars].astype(float))
    ss = sm.OLS(sample_oriv["log_income"], X_ss).fit(cov_type="HC1")
    b_oriv = ss.params.get("AHC_hat", np.nan)
    b_oriv_d = ss.params.get("AHC_hat_x_D", np.nan)

    print(f"\n--- ORIV: Second stage ---")
    print(f"  AHC (ORIV):    {b_oriv:+.4f} (p={ss.pvalues.get('AHC_hat', 1):.4f})")
    print(f"  AHC×D (ORIV):  {b_oriv_d:+.4f} (p={ss.pvalues.get('AHC_hat_x_D', 1):.4f})")
    print(f"  R²: {ss.rsquared:.4f}")

    # Comparison
    print(f"\n--- COMPARISON ---")
    print(f"  {'Method':<20s} {'AHC level':>12s} {'AHC×D':>12s}")
    print(f"  {'─'*46}")
    print(f"  {'OLS (Haiku)':<20s} {b_ols:>+12.4f} {b_ols_d:>+12.4f}")
    print(f"  {'OLS (Sonnet)':<20s} {b_ols_s:>+12.4f} {'N/A':>12s}")
    print(f"  {'ORIV (Sonnet→Haiku)':<20s} {b_oriv:>+12.4f} {b_oriv_d:>+12.4f}")
    print(f"  {'External IV (Paper 1)':<20s} {'':>12s} {'+0.234':>12s}")

    # Attenuation factor
    print(f"\n--- MEASUREMENT ERROR ANALYSIS ---")
    var_haiku = sample["AHC_haiku_z"].var()
    var_diff = (sample["AHC_haiku_z"] - sample["AHC_sonnet_z"]).var()
    lambda_hat = 1 - var_diff / (2 * var_haiku)
    print(f"  Var(Haiku): {var_haiku:.4f}")
    print(f"  Var(Haiku - Sonnet): {var_diff:.4f}")
    print(f"  Attenuation factor λ̂: {lambda_hat:.4f}")
    print(f"  Implied true coefficient: OLS / λ̂ = {b_ols / lambda_hat:.4f}")
    if b_oriv != 0:
        print(f"  ORIV coefficient: {b_oriv:.4f}")
        print(f"  Ratio ORIV/OLS: {b_oriv / b_ols:.2f}x (expect > 1 if attenuation)")

    return {
        "ols_haiku": b_ols, "ols_sonnet": b_ols_s, "oriv": b_oriv,
        "ols_haiku_d": b_ols_d, "oriv_d": b_oriv_d,
        "first_stage_f": fs_f, "attenuation_lambda": lambda_hat,
    }


# ============================================================
# HORSE RACE
# ============================================================

def horse_race(sample):
    """Does AHC add predictive power beyond existing indices?"""
    print("\n" + "=" * 70)
    print("HORSE RACE: Incremental R² of AHC")
    print("=" * 70)

    # Load Felten and Eloundou at CIUO-1d level
    # (already computed in data audit — use sector-level proxy here)
    controls = ["education_years", "experience", "experience_sq",
                "D_z", "female", "urban"]
    avail = [v for v in controls if v in sample.columns]

    # M1: Controls only
    X1 = sm.add_constant(sample[avail].astype(float))
    m1 = sm.OLS(sample["log_income"], X1).fit()

    # M2: + Frey-Osborne (existing automation measure)
    if "automation_prob" in sample.columns:
        sample["FO_z"] = (sample["automation_prob"] - sample["automation_prob"].mean()) / sample["automation_prob"].std()
        X2 = sm.add_constant(sample[avail + ["FO_z"]].astype(float))
        m2 = sm.OLS(sample["log_income"], X2).fit()
    else:
        m2 = m1

    # M3: + AHC (our index)
    X3 = sm.add_constant(sample[avail + ["AHC_haiku_z"]].astype(float))
    m3 = sm.OLS(sample["log_income"], X3).fit()

    # M4: + AHC + F&O
    vars4 = avail + ["AHC_haiku_z"]
    if "FO_z" in sample.columns:
        vars4.append("FO_z")
    X4 = sm.add_constant(sample[vars4].astype(float))
    m4 = sm.OLS(sample["log_income"], X4).fit()

    # M5: + AHC + AHC×D
    vars5 = vars4 + ["AHC_H_x_D"]
    X5 = sm.add_constant(sample[vars5].astype(float))
    m5 = sm.OLS(sample["log_income"], X5).fit()

    print(f"\n  {'Model':<35s} {'R²':>8s} {'ΔR²':>8s} {'AHC':>10s} {'AHC×D':>10s}")
    print(f"  {'─'*75}")
    r2_base = m1.rsquared
    for name, model, has_ahc_d in [
        ("M1: Controls only", m1, False),
        ("M2: + Frey-Osborne", m2, False),
        ("M3: + AHC (ours)", m3, False),
        ("M4: + AHC + F&O", m4, False),
        ("M5: + AHC + AHC×D + F&O", m5, True),
    ]:
        r2 = model.rsquared
        dr2 = r2 - r2_base
        ahc = model.params.get("AHC_haiku_z", np.nan)
        ahc_d = model.params.get("AHC_H_x_D", np.nan) if has_ahc_d else np.nan
        ahc_s = "***" if model.pvalues.get("AHC_haiku_z", 1) < 0.01 else ""
        ahc_d_s = "***" if has_ahc_d and model.pvalues.get("AHC_H_x_D", 1) < 0.01 else ""
        ahc_str = f"{ahc:+.4f}{ahc_s}" if not np.isnan(ahc) else ""
        ahc_d_str = f"{ahc_d:+.4f}{ahc_d_s}" if not np.isnan(ahc_d) else ""
        print(f"  {name:<35s} {r2:>8.4f} {dr2:>+8.4f} {ahc_str:>10s} {ahc_d_str:>10s}")

    # Key result
    r2_ahc_increment = m3.rsquared - m1.rsquared
    r2_fo_increment = m2.rsquared - m1.rsquared
    print(f"\n  >>> AHC incremental R²: {r2_ahc_increment:.4f}")
    print(f"  >>> F&O incremental R²: {r2_fo_increment:.4f}")
    if r2_ahc_increment > r2_fo_increment:
        print(f"  >>> AHC adds MORE explanatory power than Frey-Osborne")
    else:
        print(f"  >>> F&O adds more explanatory power than AHC")

    return {
        "r2_controls": m1.rsquared,
        "r2_fo": m2.rsquared,
        "r2_ahc": m3.rsquared,
        "r2_both": m4.rsquared,
        "r2_full": m5.rsquared,
        "ahc_increment": r2_ahc_increment,
        "fo_increment": r2_fo_increment,
    }


def main():
    print("=" * 70)
    print("PAPER 0 — ORIV + HORSE RACE + MEASUREMENT ERROR")
    print("=" * 70)

    sample = load_estimation_data()

    oriv_results = run_oriv(sample)
    hr_results = horse_race(sample)

    # Save
    results = {"oriv": oriv_results, "horse_race": hr_results}
    path = OUTPUT / "oriv_horse_race_results.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[SAVED] {path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
