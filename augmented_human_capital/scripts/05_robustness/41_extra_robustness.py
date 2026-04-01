#!/usr/bin/env python3
"""
Sprint 5 — Extra robustness checks:
1. Leave-one-sector-out jackknife
2. Bootstrap CIs for AHC index
3. Alternative D proxy: sector formality only
4. Weighted regression (GEIH sampling weights)
"""

import sys
import warnings
from pathlib import Path
import json

import pandas as pd
import numpy as np
import statsmodels.api as sm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDICES_DIR = PROJECT_ROOT / "data" / "indices"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"


def load_sample():
    est = pd.read_parquet(PROCESSED_DIR / "estimation_sample.parquet")
    est = est[est["log_income"] > 0].copy()
    ahc = pd.read_parquet(INDICES_DIR / "ahc_index_v2_improved_crosswalk.parquet")
    est = est.merge(ahc[["CIUO_code", "AHC_score", "SUB_score"]],
                    left_on="CIUO_4d", right_on="CIUO_code", how="left",
                    suffixes=("_v1", "_v2"))
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

    for c in ["AHC_score_v2", "SUB_score_v2"]:
        if c in est.columns:
            m, s = est[c].mean(), est[c].std()
            est[f"{c}_z"] = (est[c] - m) / s if s > 0 else 0
    est["AHC_x_D"] = est["AHC_score_v2_z"] * est["D_z"]

    return est.dropna(subset=["log_income", "AHC_score_v2_z", "D_z", "education_years"])


def run_ols(data, name=""):
    vars_ = ["education_years", "experience", "experience_sq",
             "AHC_score_v2_z", "SUB_score_v2_z", "D_z", "AHC_x_D",
             "female", "urban"]
    avail = [v for v in vars_ if v in data.columns]
    X = sm.add_constant(data[avail].astype(float))
    r = sm.OLS(data["log_income"], X).fit(cov_type="HC1")
    return {
        "name": name, "n": len(data), "r2": r.rsquared,
        "AHC_b": r.params.get("AHC_score_v2_z", np.nan),
        "AHC_p": r.pvalues.get("AHC_score_v2_z", 1),
        "AHC_x_D_b": r.params.get("AHC_x_D", np.nan),
        "AHC_x_D_p": r.pvalues.get("AHC_x_D", 1),
    }


def jackknife_sectors(est):
    """Leave-one-sector-out jackknife."""
    print("\n--- Leave-One-Sector-Out Jackknife ---")
    sectors = est["sector"].unique()
    results = []
    for sec in sectors:
        subset = est[est["sector"] != sec]
        r = run_ols(subset, f"Drop: {sec[:20]}")
        results.append(r)

    betas = [r["AHC_x_D_b"] for r in results]
    print(f"  Sectors: {len(sectors)}")
    print(f"  AHC×D range: [{min(betas):.4f}, {max(betas):.4f}]")
    print(f"  AHC×D mean: {np.mean(betas):.4f}, std: {np.std(betas):.4f}")
    print(f"  Sign changes: {sum(1 for b in betas if b < 0)}/{len(betas)}")

    full = run_ols(est, "Full sample")
    print(f"  Full sample AHC×D: {full['AHC_x_D_b']:.4f}")
    print(f"  Max deviation: {max(abs(b - full['AHC_x_D_b']) for b in betas):.4f}")

    return results


def bootstrap_ahc_index():
    """Bootstrap CIs for occupation-level AHC scores."""
    print("\n--- Bootstrap CIs for AHC Index ---")
    scores_path = INDICES_DIR / "raw_llm_scores.jsonl"
    scores = []
    with open(scores_path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if "error" not in r and "augmentation_score" in r:
                    scores.append(r)
            except:
                pass

    df = pd.DataFrame(scores)
    n_boot = 200
    np.random.seed(42)

    occ_scores = {}
    for tid, group in df.groupby("task_id"):
        occ_scores[tid] = group["augmentation_score"].values

    # Bootstrap: resample tasks within each group
    boot_means = []
    all_tasks = df["task_id"].unique()
    for b in range(n_boot):
        resampled = np.random.choice(all_tasks, size=len(all_tasks), replace=True)
        boot_aug = [df[df["task_id"] == t]["augmentation_score"].mean() for t in resampled[:500]]
        boot_means.append(np.mean(boot_aug))

    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)
    print(f"  Overall AHC mean: {np.mean(boot_means):.2f}")
    print(f"  95% Bootstrap CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  Width: {ci_hi - ci_lo:.2f}")

    return {"mean": np.mean(boot_means), "ci_lo": ci_lo, "ci_hi": ci_hi}


def alternative_d_proxies(est):
    """Test with alternative D proxies."""
    print("\n--- Alternative D Proxies ---")
    results = []

    # D1: Composite (baseline)
    r1 = run_ols(est, "D1: Composite")
    results.append(r1)

    # D2: Formality only
    sector_f = est.groupby("sector")["formal"].mean().reset_index()
    sector_f.columns = ["sector", "D_formal"]
    m, s = sector_f["D_formal"].mean(), sector_f["D_formal"].std()
    sector_f["D_f_z"] = (sector_f["D_formal"] - m) / s
    est2 = est.merge(sector_f[["sector", "D_f_z"]], on="sector", how="left")
    est2["AHC_x_Df"] = est2["AHC_score_v2_z"] * est2["D_f_z"]
    vars2 = ["education_years", "experience", "experience_sq",
             "AHC_score_v2_z", "SUB_score_v2_z", "D_f_z", "AHC_x_Df", "female", "urban"]
    X2 = sm.add_constant(est2[vars2].astype(float).dropna())
    valid = est2.dropna(subset=vars2 + ["log_income"])
    r2 = sm.OLS(valid["log_income"], sm.add_constant(valid[vars2].astype(float))).fit(cov_type="HC1")
    results.append({"name": "D2: Formality only", "n": len(valid), "r2": r2.rsquared,
                     "AHC_b": r2.params.get("AHC_score_v2_z", np.nan),
                     "AHC_p": r2.pvalues.get("AHC_score_v2_z", 1),
                     "AHC_x_D_b": r2.params.get("AHC_x_Df", np.nan),
                     "AHC_x_D_p": r2.pvalues.get("AHC_x_Df", 1)})

    # D3: Education only
    sector_e = est.groupby("sector")["education_years"].mean().reset_index()
    sector_e.columns = ["sector", "D_educ"]
    m, s = sector_e["D_educ"].mean(), sector_e["D_educ"].std()
    sector_e["D_e_z"] = (sector_e["D_educ"] - m) / s
    est3 = est.merge(sector_e[["sector", "D_e_z"]], on="sector", how="left")
    est3["AHC_x_De"] = est3["AHC_score_v2_z"] * est3["D_e_z"]
    vars3 = ["education_years", "experience", "experience_sq",
             "AHC_score_v2_z", "SUB_score_v2_z", "D_e_z", "AHC_x_De", "female", "urban"]
    valid3 = est3.dropna(subset=vars3 + ["log_income"])
    r3 = sm.OLS(valid3["log_income"], sm.add_constant(valid3[vars3].astype(float))).fit(cov_type="HC1")
    results.append({"name": "D3: Education only", "n": len(valid3), "r2": r3.rsquared,
                     "AHC_b": r3.params.get("AHC_score_v2_z", np.nan),
                     "AHC_p": r3.pvalues.get("AHC_score_v2_z", 1),
                     "AHC_x_D_b": r3.params.get("AHC_x_De", np.nan),
                     "AHC_x_D_p": r3.pvalues.get("AHC_x_De", 1)})

    # Print
    print(f"  {'Model':<25s} {'N':>7s} {'R²':>6s} {'AHC':>10s} {'AHC×D':>10s}")
    print("  " + "─" * 62)
    for r in results:
        s1 = "***" if r["AHC_p"] < 0.01 else ""
        s2 = "***" if r["AHC_x_D_p"] < 0.01 else "**" if r["AHC_x_D_p"] < 0.05 else ""
        print(f"  {r['name']:<25s} {r['n']:>7,d} {r['r2']:>6.3f} "
              f"{r['AHC_b']:>+8.4f}{s1:3s} {r['AHC_x_D_b']:>+8.4f}{s2:3s}")

    return results


def weighted_regression(est):
    """Regression with GEIH sampling weights."""
    print("\n--- Weighted Regression (GEIH weights) ---")
    if "weight" not in est.columns:
        print("  [SKIP] No weights available")
        return None

    vars_ = ["education_years", "experience", "experience_sq",
             "AHC_score_v2_z", "SUB_score_v2_z", "D_z", "AHC_x_D",
             "female", "urban"]
    valid = est.dropna(subset=vars_ + ["log_income", "weight"])
    X = sm.add_constant(valid[vars_].astype(float))
    r_wls = sm.WLS(valid["log_income"], X, weights=valid["weight"]).fit(cov_type="HC1")
    r_ols = sm.OLS(valid["log_income"], X).fit(cov_type="HC1")

    print(f"  OLS  AHC×D: {r_ols.params['AHC_x_D']:+.4f} (p={r_ols.pvalues['AHC_x_D']:.4f})")
    print(f"  WLS  AHC×D: {r_wls.params['AHC_x_D']:+.4f} (p={r_wls.pvalues['AHC_x_D']:.4f})")
    print(f"  OLS  R²: {r_ols.rsquared:.4f}")
    print(f"  WLS  R²: {r_wls.rsquared:.4f}")

    return {
        "ols_b": r_ols.params["AHC_x_D"], "ols_p": r_ols.pvalues["AHC_x_D"],
        "wls_b": r_wls.params["AHC_x_D"], "wls_p": r_wls.pvalues["AHC_x_D"],
    }


def main():
    print("=" * 60)
    print("SPRINT 5 — EXTRA ROBUSTNESS")
    print("=" * 60)

    est = load_sample()
    print(f"Sample: {len(est):,}")

    jk = jackknife_sectors(est)
    boot = bootstrap_ahc_index()
    alt_d = alternative_d_proxies(est)
    wls = weighted_regression(est)

    # Save summary
    summary = {
        "jackknife": {
            "n_sectors": len(jk),
            "ahc_x_d_range": [min(r["AHC_x_D_b"] for r in jk), max(r["AHC_x_D_b"] for r in jk)],
            "sign_changes": sum(1 for r in jk if r["AHC_x_D_b"] < 0),
        },
        "bootstrap": boot,
        "alternative_d": [{k: v for k, v in r.items()} for r in alt_d],
        "weighted": wls,
    }
    path = TABLE_DIR / "sprint5_robustness_extra.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  [SAVED] {path.name}")

    print("\n" + "=" * 60)
    print("SPRINT 5 COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
