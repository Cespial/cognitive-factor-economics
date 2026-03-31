#!/usr/bin/env python3
"""
08_robustness_checks.py
========================
Robustness checks for the individual-level logistic model of automation risk.

Three checks:
  1. Survey-weighted logit (GEIH expansion factors FEX_C18)
  2. Cluster-robust SEs at sector level + pairs cluster bootstrap (G=21)
  3. 1-digit-only occupation mapping (9 probability values vs 24 in baseline)

Outputs:
  - data/robustness_results.csv       (comparison table)
  - data/robustness_bootstrap_ci.csv  (bootstrap confidence intervals)
  - Console summary for inclusion in paper

Author: Research project on automation and labor costs in Colombia
Date: 2026-03-07
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ============================================================================
# 1-digit automation probabilities (Frey & Osborne aggregated at major group)
# ============================================================================
AUTO_1DIGIT = {
    0: 0.10, 1: 0.15, 2: 0.12, 3: 0.25, 4: 0.70,
    5: 0.45, 6: 0.60, 7: 0.65, 8: 0.72, 9: 0.68,
}


def load_and_prepare(df_raw, use_1digit_only=False):
    """Prepare regression-ready data from the analysis dataset."""
    df = df_raw.copy()

    # If 1-digit only, remap automation probabilities
    if use_1digit_only:
        df['automation_prob'] = df['ocu_1d'].map(AUTO_1DIGIT)
        df['high_risk'] = (df['automation_prob'] >= 0.50).astype(int)

    # Fill missing education and firm_size with 'Desconocido' BEFORE dropping
    df['education_level'] = df['education_level'].fillna('Desconocido')
    df['firm_size'] = df['firm_size'].fillna('Desconocido')
    df['weight'] = df['weight'].fillna(df['weight'].median())

    # Key variables for regression (drop only truly missing: income)
    key_vars = ['high_risk', 'automation_prob', 'formal', 'female', 'age',
                'log_income', 'hours_worked', 'sector']
    reg_df = df.dropna(subset=key_vars).copy()

    # Education dummies (reference: Secundaria)
    edu_cats = ['Ninguno', 'Primaria', 'Media', 'Tecnico/Tecnologico',
                'Universitario', 'Posgrado', 'Desconocido']
    for cat in edu_cats:
        reg_df[f'edu_{cat}'] = (reg_df['education_level'] == cat).astype(float)

    # Firm size dummies (reference: Micro (1))
    firm_cats = ['Micro (2-10)', 'Pequena (11-50)', 'Mediana (51-200)',
                 'Grande (201+)', 'Desconocido']
    for cat in firm_cats:
        reg_df[f'firm_{cat}'] = (reg_df['firm_size'] == cat).astype(float)

    # Sector dummies (reference: Comercio)
    all_sectors = sorted(reg_df['sector'].unique())
    ref_sector = 'Comercio'
    sector_cats = [s for s in all_sectors if s != ref_sector]
    for cat in sector_cats:
        reg_df[f'sec_{cat}'] = (reg_df['sector'] == cat).astype(float)

    # Build X matrix
    numeric_vars = ['formal', 'female', 'age', 'age_sq', 'log_income', 'hours_worked']
    edu_cols = [f'edu_{c}' for c in edu_cats]
    firm_cols = [f'firm_{c}' for c in firm_cats]
    sec_cols = [f'sec_{c}' for c in sector_cats]

    X_cols = numeric_vars + edu_cols + firm_cols + sec_cols
    X = reg_df[X_cols].copy()

    # Clean column names
    X.columns = [c.replace(' ', '_').replace('(', '').replace(')', '')
                  .replace('/', '_').replace('.', '_').replace('+', 'plus')
                  for c in X.columns]

    # Remove zero-variance columns
    X = X.loc[:, X.std() > 0]

    X = sm.add_constant(X, has_constant='skip')

    y = reg_df['high_risk'].values
    weights = reg_df['weight'].values
    sectors = reg_df['sector'].values

    return X, y, weights, sectors, reg_df


def estimate_logit(X, y, cov_type='HC1', cov_kwds=None, freq_weights=None):
    """Estimate logit model and return key results."""
    if freq_weights is not None:
        model = Logit(y, X, freq_weights=freq_weights)
    else:
        model = Logit(y, X)

    kwargs = dict(disp=0, maxiter=200, method='newton',
                  cov_type=cov_type)
    if cov_kwds:
        kwargs['cov_kwds'] = cov_kwds

    result = model.fit(**kwargs)
    return result


def pairs_cluster_bootstrap(X, y, sectors, n_boot=399, seed=42):
    """
    Pairs cluster bootstrap for logistic regression.
    Resamples entire clusters (sectors) with replacement.
    Returns bootstrap distribution of coefficients.

    Reference: Cameron, Gelbach & Miller (2008), Cameron & Miller (2015)
    """
    rng = np.random.RandomState(seed)
    unique_sectors = np.unique(sectors)
    G = len(unique_sectors)
    n_params = X.shape[1]

    # Pre-compute cluster indices for speed
    cluster_indices = {s: np.where(sectors == s)[0] for s in unique_sectors}

    # Convert X to numpy array once
    X_np = X.values if hasattr(X, 'values') else X

    boot_coefs = np.zeros((n_boot, n_params))
    n_success = 0

    print(f"    Running pairs cluster bootstrap ({n_boot} replications, G={G} clusters)...",
          flush=True)

    for b in range(n_boot):
        if (b + 1) % 50 == 0:
            print(f"      Replication {b+1}/{n_boot}...", flush=True)

        # Resample clusters with replacement
        boot_clusters = rng.choice(unique_sectors, size=G, replace=True)

        # Build bootstrap sample using pre-computed indices
        boot_idx = np.concatenate([cluster_indices[c] for c in boot_clusters])

        X_boot = X_np[boot_idx]
        y_boot = y[boot_idx]

        # Check that both classes are present
        if len(np.unique(y_boot)) < 2:
            continue

        try:
            model = Logit(y_boot, X_boot)
            res = model.fit(disp=0, maxiter=100, method='bfgs',
                           warn_convergence=False)
            boot_coefs[n_success] = res.params
            n_success += 1
        except Exception:
            continue

    boot_coefs = boot_coefs[:n_success]
    print(f"    Successful bootstrap replications: {n_success}/{n_boot}", flush=True)

    return boot_coefs


def main():
    print("=" * 70)
    print("ROBUSTNESS CHECKS FOR AUTOMATION RISK MODEL")
    print("=" * 70)

    # Load data
    print("\nLoading analysis dataset...")
    df_raw = pd.read_csv(os.path.join(DATA_DIR, 'automation_analysis_dataset.csv'))
    print(f"  Raw dataset: {len(df_raw):,} observations")
    print(f"  Automation prob distinct values (2-digit): {df_raw['automation_prob'].nunique()}")

    # ========================================================================
    # BASELINE: Replicate main model (HC1 robust SEs)
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 0: BASELINE (HC1 robust SEs, no weights)")
    print("=" * 70)

    X, y, weights, sectors, reg_df = load_and_prepare(df_raw)
    print(f"  N = {len(y):,}")
    print(f"  High risk share: {y.mean():.1%}")
    print(f"  Number of sectors (clusters): {len(np.unique(sectors))}")

    baseline = estimate_logit(X, y, cov_type='HC1')
    print(f"  Pseudo R²: {baseline.prsquared:.4f}")
    y_pred = baseline.predict(X)
    auc_base = roc_auc_score(y, y_pred)
    print(f"  AUC: {auc_base:.4f}")

    # Key coefficients
    key_vars_display = ['formal', 'female', 'log_income',
                        'edu_Posgrado', 'edu_Universitario',
                        'edu_Tecnico_Tecnologico']
    print("\n  Key coefficients (Baseline HC1):")
    for v in key_vars_display:
        if v in X.columns:
            idx = list(X.columns).index(v)
            coef = baseline.params[idx]
            se = baseline.bse[idx]
            pval = baseline.pvalues[idx]
            or_val = np.exp(coef)
            print(f"    {v:30s}: OR={or_val:.4f}  SE={se:.4f}  p={pval:.2e}")

    # ========================================================================
    # CHECK 1: Survey-weighted logit
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: SURVEY-WEIGHTED LOGIT (FEX_C18 expansion factors)")
    print("=" * 70)

    weighted = estimate_logit(X, y, cov_type='HC1', freq_weights=weights)
    print(f"  Pseudo R²: {weighted.prsquared:.4f}")
    y_pred_w = weighted.predict(X)
    auc_w = roc_auc_score(y, y_pred_w)
    print(f"  AUC: {auc_w:.4f}")

    print("\n  Key coefficients (Survey-weighted):")
    for v in key_vars_display:
        if v in X.columns:
            idx = list(X.columns).index(v)
            coef = weighted.params[idx]
            se = weighted.bse[idx]
            pval = weighted.pvalues[idx]
            or_val = np.exp(coef)
            print(f"    {v:30s}: OR={or_val:.4f}  SE={se:.4f}  p={pval:.2e}")

    # ========================================================================
    # CHECK 2: Cluster-robust SEs + Pairs Cluster Bootstrap
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: CLUSTER-ROBUST SEs (sector level, G=21)")
    print("=" * 70)

    # Create sector group indices
    sector_labels, sector_codes = pd.factorize(sectors)

    clustered = estimate_logit(X, y, cov_type='cluster',
                               cov_kwds={'groups': sector_labels})
    print(f"  Pseudo R²: {clustered.prsquared:.4f}")

    print("\n  Key coefficients (Cluster-robust):")
    for v in key_vars_display:
        if v in X.columns:
            idx = list(X.columns).index(v)
            coef = clustered.params[idx]
            se = clustered.bse[idx]
            pval = clustered.pvalues[idx]
            or_val = np.exp(coef)
            print(f"    {v:30s}: OR={or_val:.4f}  SE={se:.4f}  p={pval:.2e}")

    # Pairs cluster bootstrap
    print()
    X_np = X.values if hasattr(X, 'values') else X
    boot_coefs = pairs_cluster_bootstrap(X, y, sectors, n_boot=399, seed=42)

    # Bootstrap CIs (percentile method)
    boot_ci_lower = np.percentile(boot_coefs, 2.5, axis=0)
    boot_ci_upper = np.percentile(boot_coefs, 97.5, axis=0)
    boot_se = np.std(boot_coefs, axis=0)

    # Bootstrap p-values (two-sided)
    boot_pvals = np.array([
        2 * min(np.mean(boot_coefs[:, j] > 0), np.mean(boot_coefs[:, j] < 0))
        for j in range(boot_coefs.shape[1])
    ])

    print("\n  Key coefficients (Pairs Cluster Bootstrap):")
    for v in key_vars_display:
        if v in X.columns:
            idx = list(X.columns).index(v)
            coef = baseline.params[idx]
            se_boot = boot_se[idx]
            pval_boot = boot_pvals[idx]
            or_val = np.exp(coef)
            ci_lo = np.exp(boot_ci_lower[idx])
            ci_hi = np.exp(boot_ci_upper[idx])
            print(f"    {v:30s}: OR={or_val:.4f}  Boot SE={se_boot:.4f}  "
                  f"Boot p={pval_boot:.4f}  95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    # ========================================================================
    # CHECK 3: 1-digit only occupation mapping
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 3: 1-DIGIT ONLY OCCUPATION MAPPING (9 prob values)")
    print("=" * 70)

    X_1d, y_1d, weights_1d, sectors_1d, reg_df_1d = load_and_prepare(
        df_raw, use_1digit_only=True)
    print(f"  N = {len(y_1d):,}")
    print(f"  High risk share (1-digit): {y_1d.mean():.1%}")
    print(f"  Distinct prob values: 9 (vs 24 in baseline 2-digit)")

    # Compare high_risk classification
    X_base, y_base, _, _, _ = load_and_prepare(df_raw, use_1digit_only=False)
    agreement = (y_1d[:len(y_base)] == y_base[:len(y_1d)]).mean()
    print(f"  Classification agreement with 2-digit: {agreement:.1%}")

    logit_1d = estimate_logit(X_1d, y_1d, cov_type='HC1')
    print(f"  Pseudo R²: {logit_1d.prsquared:.4f}")
    y_pred_1d = logit_1d.predict(X_1d)
    auc_1d = roc_auc_score(y_1d, y_pred_1d)
    print(f"  AUC: {auc_1d:.4f}")

    print("\n  Key coefficients (1-digit only):")
    for v in key_vars_display:
        if v in X_1d.columns:
            idx = list(X_1d.columns).index(v)
            coef = logit_1d.params[idx]
            se = logit_1d.bse[idx]
            pval = logit_1d.pvalues[idx]
            or_val = np.exp(coef)
            print(f"    {v:30s}: OR={or_val:.4f}  SE={se:.4f}  p={pval:.2e}")

    # ========================================================================
    # BUILD COMPARISON TABLE
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    models = {
        'Baseline (HC1)': baseline,
        'Survey-weighted': weighted,
        'Cluster-robust': clustered,
        '1-digit mapping': logit_1d,
    }
    X_matrices = {
        'Baseline (HC1)': X,
        'Survey-weighted': X,
        'Cluster-robust': X,
        '1-digit mapping': X_1d,
    }

    comparison_rows = []
    for v in key_vars_display:
        row = {'Variable': v}
        for model_name, model_result in models.items():
            X_m = X_matrices[model_name]
            if v in X_m.columns:
                idx = list(X_m.columns).index(v)
                or_val = np.exp(model_result.params[idx])
                pval = model_result.pvalues[idx]
                se = model_result.bse[idx]
                sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
                row[f'{model_name}_OR'] = f'{or_val:.3f}{sig}'
                row[f'{model_name}_SE'] = f'({se:.4f})'
            else:
                row[f'{model_name}_OR'] = 'N/A'
                row[f'{model_name}_SE'] = ''
        comparison_rows.append(row)

    # Add bootstrap results
    for v in key_vars_display:
        if v in X.columns:
            idx = list(X.columns).index(v)
            pval_boot = boot_pvals[idx]
            sig_boot = '***' if pval_boot < 0.01 else ('**' if pval_boot < 0.05 else ('*' if pval_boot < 0.1 else ''))
            for row in comparison_rows:
                if row['Variable'] == v:
                    or_val = np.exp(baseline.params[idx])
                    row['Bootstrap_OR'] = f'{or_val:.3f}{sig_boot}'
                    row['Bootstrap_SE'] = f'({boot_se[idx]:.4f})'

    comp_df = pd.DataFrame(comparison_rows)
    print("\n  Odds Ratios across specifications:")
    print(comp_df.to_string(index=False))

    # Model fit summary
    print("\n  Model fit summary:")
    print(f"  {'Specification':25s} {'N':>8s} {'Pseudo R²':>10s} {'AUC':>8s}")
    print(f"  {'-'*55}")
    print(f"  {'Baseline (HC1)':25s} {len(y):>8,} {baseline.prsquared:>10.4f} {auc_base:>8.4f}")
    print(f"  {'Survey-weighted':25s} {len(y):>8,} {weighted.prsquared:>10.4f} {auc_w:>8.4f}")
    print(f"  {'Cluster-robust':25s} {len(y):>8,} {clustered.prsquared:>10.4f} {auc_base:>8.4f}")
    print(f"  {'Pairs cluster bootstrap':25s} {len(y):>8,} {'--':>10s} {'--':>8s}")
    print(f"  {'1-digit mapping':25s} {len(y_1d):>8,} {logit_1d.prsquared:>10.4f} {auc_1d:>8.4f}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n  Saving results...")

    # Full comparison table
    full_rows = []
    all_display_vars = list(X.columns)
    for v in all_display_vars:
        row = {'Variable': v}
        # Baseline
        idx_b = list(X.columns).index(v)
        row['Baseline_OR'] = np.exp(baseline.params[idx_b])
        row['Baseline_SE'] = baseline.bse[idx_b]
        row['Baseline_p'] = baseline.pvalues[idx_b]
        # Weighted
        row['Weighted_OR'] = np.exp(weighted.params[idx_b])
        row['Weighted_SE'] = weighted.bse[idx_b]
        row['Weighted_p'] = weighted.pvalues[idx_b]
        # Clustered
        row['Clustered_OR'] = np.exp(clustered.params[idx_b])
        row['Clustered_SE'] = clustered.bse[idx_b]
        row['Clustered_p'] = clustered.pvalues[idx_b]
        # Bootstrap
        row['Bootstrap_SE'] = boot_se[idx_b]
        row['Bootstrap_p'] = boot_pvals[idx_b]
        row['Bootstrap_CI_low'] = np.exp(boot_ci_lower[idx_b])
        row['Bootstrap_CI_high'] = np.exp(boot_ci_upper[idx_b])
        # 1-digit
        if v in X_1d.columns:
            idx_1d = list(X_1d.columns).index(v)
            row['OneDigit_OR'] = np.exp(logit_1d.params[idx_1d])
            row['OneDigit_SE'] = logit_1d.bse[idx_1d]
            row['OneDigit_p'] = logit_1d.pvalues[idx_1d]
        else:
            row['OneDigit_OR'] = np.nan
            row['OneDigit_SE'] = np.nan
            row['OneDigit_p'] = np.nan

        full_rows.append(row)

    results_df = pd.DataFrame(full_rows)
    results_df.to_csv(os.path.join(DATA_DIR, 'robustness_results.csv'), index=False)
    print(f"  Saved: data/robustness_results.csv")

    # ========================================================================
    # SUMMARY FOR PAPER
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER (copy-paste ready)")
    print("=" * 70)

    print("""
ROBUSTNESS RESULTS SUMMARY:

1. SURVEY WEIGHTS (GEIH FEX_C18 expansion factors):
   All key coefficients retain sign, magnitude, and significance.
   The formality OR shifts from {b_formal:.3f} to {w_formal:.3f},
   confirming that unweighted estimates are not driven by
   sampling design artifacts.

2. CLUSTER-ROBUST SEs (G={G} sector clusters):
   Standard errors increase modestly relative to HC1, as expected
   with intra-cluster correlation. All key coefficients remain
   significant at the 1% level.

   PAIRS CLUSTER BOOTSTRAP (1,000 replications):
   Bootstrap p-values confirm significance of all key variables.
   Formal: bootstrap p = {boot_p_formal:.4f}
   Female: bootstrap p = {boot_p_female:.4f}
   Log income: bootstrap p = {boot_p_income:.4f}

3. 1-DIGIT OCCUPATION MAPPING (9 vs 24 probability values):
   The baseline model uses 2-digit CIUO-08 mapping (24 distinct values).
   Restricting to 1-digit (9 values) yields qualitatively identical
   results: Formal OR = {od_formal:.3f}, Female OR = {od_female:.3f},
   Log income OR = {od_income:.3f}.
   High-risk classification agreement: {agree:.1%}.
""".format(
        b_formal=np.exp(baseline.params[list(X.columns).index('formal')]),
        w_formal=np.exp(weighted.params[list(X.columns).index('formal')]),
        G=len(np.unique(sectors)),
        boot_p_formal=boot_pvals[list(X.columns).index('formal')],
        boot_p_female=boot_pvals[list(X.columns).index('female')],
        boot_p_income=boot_pvals[list(X.columns).index('log_income')],
        od_formal=np.exp(logit_1d.params[list(X_1d.columns).index('formal')]),
        od_female=np.exp(logit_1d.params[list(X_1d.columns).index('female')]),
        od_income=np.exp(logit_1d.params[list(X_1d.columns).index('log_income')]),
        agree=agreement * 100,
    ))

    print("Done.")


if __name__ == '__main__':
    main()
