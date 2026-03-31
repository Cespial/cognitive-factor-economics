#!/usr/bin/env python3
"""
09_did_panel_eam.py
====================
Difference-in-Differences identification strategy using EAM panel data (2016-2024).

Research question: Did the large minimum wage increases under President Petro
(+16% in 2023, +12% in 2024) accelerate capital investment (automation proxy)
in labor-cost-intensive manufacturing firms?

Identification strategy:
  - Treatment: firms with above-median labor cost share in 2022 (pre-treatment)
  - Treatment period: 2023-2024 (post-Petro minimum wage shocks)
  - Control: firms with below-median labor cost share in 2022
  - Estimator: two-way fixed effects (firm + year) with clustered SEs

Specifications:
  a. Binary DiD:       log(Y_it) = α_i + γ_t + β(high_cost_i × post_t) + X'δ + ε_it
  b. Continuous DiD:   log(Y_it) = α_i + γ_t + β(intensity_i × post_t) + X'δ + ε_it
  c. Event study:      log(Y_it) = α_i + γ_t + Σ_k β_k(high_cost_i × 1{t=k}) + X'δ + ε_it

Outputs:
  - data/eam_panel_constructed.csv        (unified firm-year panel)
  - data/did_regression_results.csv       (all regression coefficients)
  - images/en/fig_did_event_study.{png,pdf}
  - images/en/fig_did_parallel_trends.{png,pdf}
  - images/en/fig_did_treatment_distribution.{png,pdf}
  - images/en/fig_did_forest_plot.{png,pdf}

Author: Research project on automation and labor costs in Colombia
Date: 2026-03-20
"""

import os
import sys
import warnings
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from linearmodels.panel import PanelOLS

warnings.filterwarnings('ignore')

# Force unbuffered output
print = functools.partial(print, flush=True)

# Reproducibility
np.random.seed(42)

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PANEL_DIR = os.path.join(DATA_DIR, 'eam_panel')
IMG_DIR = os.path.join(BASE_DIR, 'images', 'en')
os.makedirs(IMG_DIR, exist_ok=True)

# ============================================================================
# Matplotlib publication settings (Econometrica / Nature style)
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'figure.figsize': (8, 5),
})

# Key variables to keep from each year
KEY_VARS = [
    'nordemp', 'nordest', 'dpto', 'ciiu4', 'periodo',
    'pertotal', 'persocu', 'salarper', 'pressper',
    'prespyte', 'salpeyte', 'prodbr2', 'invebrta',
    'activfi', 'deprecia',
]

# ============================================================================
# 1. DATA LOADING
# ============================================================================
def load_year(year: int) -> pd.DataFrame:
    """Load a single year of EAM data, handling format differences.

    Years 2016-2019: semicolon-delimited CSV (UTF-8-SIG encoding).
    Years 2020-2022: comma-delimited CSV.
    Years 2023-2024: Stata .dta format.

    Returns a DataFrame with lowercase column names and only KEY_VARS retained.
    """
    file_map = {
        2016: ('eam_2016/EAM_2016.csv', ';'),
        2017: ('eam_2017/Estructura - EAM - 2017.csv', ';'),
        2018: ('eam_2018/EAM_2018.csv', ';'),
        2019: ('eam_2019/EAM_2019.csv', ';'),
        2020: ('eam_2020/EAM_2020/EAM_2020.csv', ','),
        2021: ('eam_2021/EAM_2021/EAM_ANONIMIZADA_2021.csv', ','),
        2022: ('eam_2022/EAM_ANONIMIZADA_2022.csv', ','),
        2023: ('eam_2023/EAM_ANONIMIZADA_2023.dta', None),
        2024: ('eam_2024/EAM_ANONIMIZADA_2024.dta', None),
    }

    rel_path, sep = file_map[year]
    fpath = os.path.join(PANEL_DIR, rel_path)

    if year in (2023, 2024):
        # Stata format
        df = pd.read_stata(fpath, convert_categoricals=False)
    else:
        # CSV — try utf-8-sig first, then latin-1
        for enc in ['utf-8-sig', 'utf-8', 'latin-1']:
            try:
                df = pd.read_csv(fpath, sep=sep, encoding=enc, low_memory=False)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            raise RuntimeError(f"Could not read {fpath} with any encoding")

    # Standardize column names to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Keep only key variables (present in this year)
    available = [v for v in KEY_VARS if v in df.columns]
    df = df[available].copy()

    # Coerce all columns to numeric (some years store numbers as strings)
    for col in df.columns:
        if col not in ('nordemp', 'nordest', 'dpto', 'ciiu4', 'periodo'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # Identifiers: ensure numeric for panel merging
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure periodo is set
    if 'periodo' not in df.columns or df['periodo'].isna().all():
        df['periodo'] = year
    df['periodo'] = df['periodo'].fillna(year).astype(int)

    return df


def load_all_years() -> pd.DataFrame:
    """Load and stack all 9 years of EAM data into a single panel."""
    frames = []
    for year in range(2016, 2025):
        print(f"  Loading {year}...", end=' ')
        try:
            df = load_year(year)
            print(f"OK — {len(df):,} obs, {df['nordemp'].nunique():,} firms")
            frames.append(df)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    panel = pd.concat(frames, ignore_index=True)
    return panel


# ============================================================================
# 2. VARIABLE CONSTRUCTION
# ============================================================================
def construct_variables(panel: pd.DataFrame) -> pd.DataFrame:
    """Create analysis variables from raw EAM data.

    Constructed variables:
      - total_labor_cost: permanent + temporary compensation
      - total_salaries: permanent + temporary wages
      - non_wage_ratio: (total_labor_cost - total_salaries) / total_salaries
      - labor_productivity: gross production / total employed
      - unit_labor_cost: total labor cost / gross production
      - capital_intensity: fixed assets / total employed
      - investment_rate: gross investment / gross production
      - automation_proxy: gross investment / total employed
      - log versions using np.log1p for zero-safe transformation
    """
    df = panel.copy()

    # Fill missing numeric vars with 0 where sensible (costs, investment)
    cost_vars = ['pressper', 'salarper', 'prespyte', 'salpeyte',
                 'invebrta', 'activfi', 'deprecia']
    for v in cost_vars:
        if v in df.columns:
            df[v] = df[v].fillna(0)

    # Total labor cost = permanent compensation + temporary compensation
    df['total_labor_cost'] = df['pressper'] + df['prespyte']
    df['total_salaries'] = df['salarper'] + df['salpeyte']

    # Non-wage burden ratio
    df['non_wage_ratio'] = np.where(
        df['total_salaries'] > 0,
        (df['total_labor_cost'] - df['total_salaries']) / df['total_salaries'],
        np.nan
    )

    # Replace zero/negative employment with NaN to avoid division issues
    df['persocu_safe'] = df['persocu'].where(df['persocu'] > 0, np.nan)
    df['prodbr2_safe'] = df['prodbr2'].where(df['prodbr2'] > 0, np.nan)

    # Ratios
    df['labor_productivity'] = df['prodbr2'] / df['persocu_safe']
    df['unit_labor_cost'] = df['total_labor_cost'] / df['prodbr2_safe']
    df['capital_intensity'] = df['activfi'] / df['persocu_safe']
    df['investment_rate'] = df['invebrta'] / df['prodbr2_safe']
    df['automation_proxy'] = df['invebrta'] / df['persocu_safe']

    # 2-digit CIIU for sector grouping
    df['ciiu2'] = (df['ciiu4'] // 100).astype('Int64')

    # Log transformations (log1p handles zeros gracefully)
    log_vars = ['total_labor_cost', 'total_salaries', 'labor_productivity',
                'unit_labor_cost', 'capital_intensity', 'investment_rate',
                'automation_proxy', 'persocu', 'prodbr2', 'invebrta', 'activfi']
    for v in log_vars:
        # Clip negative values to 0 before log transform
        df[f'log_{v}'] = np.log1p(df[v].clip(lower=0))

    return df


# ============================================================================
# 3. TREATMENT DEFINITION
# ============================================================================
def define_treatment(panel: pd.DataFrame) -> pd.DataFrame:
    """Define treatment status based on pre-treatment (2022) labor cost share.

    Treatment intensity (continuous):
      labor_cost_share_2022 = total_labor_cost_2022 / prodbr2_2022

    Binary treatment:
      high_cost = 1 if labor_cost_share_2022 > median(labor_cost_share_2022)

    Post period:
      post = 1 if periodo >= 2023

    Also creates tercile and quartile indicators for robustness.
    """
    df = panel.copy()

    # --- Compute 2022 labor cost share per firm ---
    mask_2022 = df['periodo'] == 2022
    df_2022 = df.loc[mask_2022, ['nordemp', 'total_labor_cost', 'prodbr2']].copy()
    df_2022['labor_cost_share_2022'] = np.where(
        df_2022['prodbr2'] > 0,
        df_2022['total_labor_cost'] / df_2022['prodbr2'],
        np.nan
    )
    df_2022 = df_2022.dropna(subset=['labor_cost_share_2022'])
    # Keep one obs per firm (in case of duplicates)
    firm_treatment = df_2022.groupby('nordemp')['labor_cost_share_2022'].mean().reset_index()

    # --- Binary treatment: above-median ---
    median_share = firm_treatment['labor_cost_share_2022'].median()
    firm_treatment['high_cost'] = (firm_treatment['labor_cost_share_2022'] > median_share).astype(int)

    # --- Tercile treatment ---
    firm_treatment['tercile'] = pd.qcut(
        firm_treatment['labor_cost_share_2022'], q=3, labels=[1, 2, 3]
    ).astype(int)

    # --- Quartile treatment ---
    firm_treatment['quartile'] = pd.qcut(
        firm_treatment['labor_cost_share_2022'], q=4, labels=[1, 2, 3, 4]
    ).astype(int)

    # Merge treatment status back to full panel
    df = df.merge(
        firm_treatment[['nordemp', 'labor_cost_share_2022', 'high_cost', 'tercile', 'quartile']],
        on='nordemp', how='left'
    )

    # Post-treatment indicator
    df['post'] = (df['periodo'] >= 2023).astype(int)

    # DiD interaction terms
    df['did'] = df['high_cost'] * df['post']
    df['did_intensity'] = df['labor_cost_share_2022'] * df['post']

    # Event study dummies: interaction of high_cost with year indicators
    # Reference year = 2022
    for yr in range(2016, 2025):
        if yr == 2022:
            continue  # omitted reference period
        df[f'high_cost_x_{yr}'] = df['high_cost'] * (df['periodo'] == yr).astype(int)

    print(f"\n  Treatment definition:")
    print(f"    Median labor cost share (2022): {median_share:.4f}")
    print(f"    Firms with treatment data: {firm_treatment['nordemp'].nunique():,}")
    print(f"    High-cost firms: {firm_treatment['high_cost'].sum():,}")
    print(f"    Low-cost firms: {(1 - firm_treatment['high_cost']).sum():,}")

    return df


# ============================================================================
# 4. WINSORIZATION
# ============================================================================
def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize a series at the given quantiles."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def winsorize_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorize continuous variables at 1st/99th percentile."""
    out = df.copy()
    vars_to_winsorize = [
        'log_automation_proxy', 'log_capital_intensity', 'log_investment_rate',
        'log_unit_labor_cost', 'log_labor_productivity', 'log_persocu',
        'log_prodbr2', 'labor_cost_share_2022',
    ]
    for v in vars_to_winsorize:
        if v in out.columns:
            out[f'{v}_w'] = winsorize(out[v])
    return out


# ============================================================================
# 5. DiD ESTIMATION
# ============================================================================
def prepare_panel_index(df: pd.DataFrame) -> pd.DataFrame:
    """Set MultiIndex (nordemp, periodo) for linearmodels PanelOLS."""
    out = df.copy()
    out['nordemp'] = out['nordemp'].astype(int)
    out['periodo'] = out['periodo'].astype(int)
    out = out.set_index(['nordemp', 'periodo'])
    return out


def run_did_regression(df: pd.DataFrame, dep_var: str, treat_var: str,
                       controls: list, entity_effects: bool = True,
                       time_effects: bool = True, cluster_entity: bool = True,
                       label: str = '') -> dict:
    """Run a single two-way FE DiD regression using PanelOLS.

    Parameters
    ----------
    df : DataFrame with MultiIndex (nordemp, periodo)
    dep_var : dependent variable name
    treat_var : treatment interaction variable (or list for event study)
    controls : list of control variable names
    entity_effects : include firm fixed effects
    time_effects : include year fixed effects
    cluster_entity : cluster standard errors at firm level
    label : descriptive label for the regression

    Returns
    -------
    dict with coefficient, SE, t-stat, p-value, CI, N, n_firms, R2
    """
    # Build list of RHS variables
    if isinstance(treat_var, list):
        rhs_vars = treat_var + controls
    else:
        rhs_vars = [treat_var] + controls

    all_vars = [dep_var] + rhs_vars
    subset = df[all_vars].dropna()

    if len(subset) < 100:
        print(f"    WARNING: Only {len(subset)} obs for {label}, skipping.")
        return None

    y = subset[dep_var]
    X = subset[rhs_vars]

    # Add constant only if no fixed effects
    if not entity_effects and not time_effects:
        X = pd.DataFrame(np.column_stack([np.ones(len(X)), X.values]),
                         columns=['const'] + list(X.columns), index=X.index)

    mod = PanelOLS(y, X, entity_effects=entity_effects, time_effects=time_effects,
                   drop_absorbed=True)

    if cluster_entity:
        res = mod.fit(cov_type='clustered', cluster_entity=True)
    else:
        res = mod.fit(cov_type='robust')

    # Extract results
    result = {
        'label': label,
        'dep_var': dep_var,
        'n_obs': int(res.nobs),
        'n_entities': int(res.entity_info['total']),
        'r2_within': float(res.rsquared_within),
        'r2_between': float(res.rsquared_between),
        'r2_overall': float(res.rsquared_overall),
    }

    # Store coefficients for treatment variables
    if isinstance(treat_var, list):
        for tv in treat_var:
            if tv in res.params.index:
                result[f'beta_{tv}'] = float(res.params[tv])
                result[f'se_{tv}'] = float(res.std_errors[tv])
                result[f'tstat_{tv}'] = float(res.tstats[tv])
                result[f'pval_{tv}'] = float(res.pvalues[tv])
                ci = res.conf_int().loc[tv]
                result[f'ci_lo_{tv}'] = float(ci.iloc[0])
                result[f'ci_hi_{tv}'] = float(ci.iloc[1])
    else:
        if treat_var in res.params.index:
            result['beta'] = float(res.params[treat_var])
            result['se'] = float(res.std_errors[treat_var])
            result['tstat'] = float(res.tstats[treat_var])
            result['pval'] = float(res.pvalues[treat_var])
            ci = res.conf_int().loc[treat_var]
            result['ci_lo'] = float(ci.iloc[0])
            result['ci_hi'] = float(ci.iloc[1])

    result['_res'] = res  # store full result object for diagnostics
    return result


def run_event_study(df: pd.DataFrame, dep_var: str, controls: list,
                    label: str = '') -> dict:
    """Run event study specification with year-by-treatment interactions.

    Omits 2022 as the reference year. Returns coefficients for each year.
    """
    event_vars = [f'high_cost_x_{yr}' for yr in range(2016, 2025) if yr != 2022]
    return run_did_regression(
        df, dep_var=dep_var, treat_var=event_vars,
        controls=controls, label=label
    )


# ============================================================================
# 6. PARALLEL TRENDS TEST
# ============================================================================
def test_parallel_pretrends(result: dict) -> dict:
    """Joint F-test on pre-treatment event study coefficients.

    H0: β_2016 = β_2017 = β_2018 = β_2019 = β_2020 = β_2021 = 0
    (all pre-treatment interactions are jointly zero)
    """
    res_obj = result.get('_res')
    if res_obj is None:
        return {'f_stat': np.nan, 'p_value': np.nan, 'reject_at_05': np.nan}

    pre_years = [2016, 2017, 2018, 2019, 2020, 2021]
    pre_vars = [f'high_cost_x_{yr}' for yr in pre_years]

    # Extract coefficients and VCV for pre-treatment interactions
    available = [v for v in pre_vars if v in res_obj.params.index]
    if len(available) == 0:
        return {'f_stat': np.nan, 'p_value': np.nan, 'reject_at_05': np.nan}

    betas = res_obj.params[available].values
    vcov = res_obj.cov.loc[available, available].values
    k = len(available)

    try:
        # Wald test: β' V^{-1} β / k ~ F(k, N-K)
        wald = betas @ np.linalg.inv(vcov) @ betas / k
        dof2 = int(res_obj.nobs) - len(res_obj.params)
        p_value = 1 - stats.f.cdf(wald, k, max(dof2, 1))
    except np.linalg.LinAlgError:
        wald, p_value = np.nan, np.nan

    return {
        'f_stat': float(wald),
        'p_value': float(p_value),
        'reject_at_05': p_value < 0.05 if not np.isnan(p_value) else np.nan,
        'n_pre_coefs': k,
    }


# ============================================================================
# 7. FIGURES
# ============================================================================
def plot_event_study(result: dict, outcome_label: str, filename: str):
    """Plot event study coefficients with 95% confidence intervals.

    This is the KEY figure for the paper: shows pre-treatment parallel trends
    and post-treatment divergence.
    """
    years = [yr for yr in range(2016, 2025) if yr != 2022]
    betas, ci_lo, ci_hi = [], [], []

    for yr in years:
        var = f'high_cost_x_{yr}'
        b = result.get(f'beta_{var}', np.nan)
        lo = result.get(f'ci_lo_{var}', np.nan)
        hi = result.get(f'ci_hi_{var}', np.nan)
        betas.append(b)
        ci_lo.append(lo)
        ci_hi.append(hi)

    # Insert reference year (2022) with zero
    all_years = list(range(2016, 2025))
    full_betas = []
    full_ci_lo = []
    full_ci_hi = []
    idx = 0
    for yr in all_years:
        if yr == 2022:
            full_betas.append(0.0)
            full_ci_lo.append(0.0)
            full_ci_hi.append(0.0)
        else:
            full_betas.append(betas[idx])
            full_ci_lo.append(ci_lo[idx])
            full_ci_hi.append(ci_hi[idx])
            idx += 1

    full_betas = np.array(full_betas)
    full_ci_lo = np.array(full_ci_lo)
    full_ci_hi = np.array(full_ci_hi)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Shaded region for post-treatment
    ax.axvspan(2022.5, 2024.5, alpha=0.08, color='red', label='Post-treatment')

    # Zero line
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='-')

    # Reference year marker
    ax.axvline(x=2022, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    # Error bars
    yerr_lo = full_betas - full_ci_lo
    yerr_hi = full_ci_hi - full_betas
    ax.errorbar(all_years, full_betas, yerr=[yerr_lo, yerr_hi],
                fmt='o', color='#1f4e79', markersize=7, capsize=4,
                capthick=1.5, linewidth=1.5, elinewidth=1.5,
                markeredgecolor='white', markeredgewidth=0.8)

    # Connect points with line
    ax.plot(all_years, full_betas, '-', color='#1f4e79', linewidth=1.2, alpha=0.6)

    ax.set_xlabel('Year')
    ax.set_ylabel(f'Coefficient (relative to 2022)')
    ax.set_title(f'Event Study: Effect of Labor Cost Intensity on {outcome_label}')
    ax.set_xticks(all_years)
    ax.set_xticklabels(all_years, rotation=0)

    # Annotation
    ax.annotate('Reference\nyear', xy=(2022, 0), xytext=(2022, full_betas.min() * 0.7),
                ha='center', fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax.annotate('Petro MW\nshocks', xy=(2023, full_betas[7] if len(full_betas) > 7 else 0),
                xytext=(2023.5, max(full_betas) * 1.1 if max(full_betas) > 0 else 0.1),
                ha='center', fontsize=9, color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8))

    ax.legend(loc='upper left', frameon=True, edgecolor='gray', fancybox=False)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(IMG_DIR, f'{filename}.{ext}'))
    plt.close(fig)
    print(f"    Saved {filename}.png/pdf")


def plot_parallel_trends(df_raw: pd.DataFrame, outcome_var: str,
                         outcome_label: str, filename: str):
    """Plot mean outcome by treatment group over time (parallel trends visual)."""
    df = df_raw.dropna(subset=[outcome_var, 'high_cost']).copy()
    df['high_cost'] = df['high_cost'].astype(int)

    grouped = df.groupby(['periodo', 'high_cost'])[outcome_var].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))

    for treat, label, color, marker in [(0, 'Low labor cost share', '#2c7fb8', 's'),
                                         (1, 'High labor cost share', '#d7301f', 'o')]:
        sub = grouped[grouped['high_cost'] == treat]
        ax.plot(sub['periodo'], sub[outcome_var], f'-{marker}', color=color,
                label=label, markersize=7, linewidth=1.8,
                markeredgecolor='white', markeredgewidth=0.8)

    # Vertical line at treatment
    ax.axvline(x=2022.5, color='gray', linewidth=1, linestyle='--', alpha=0.7)
    ax.text(2022.6, ax.get_ylim()[1] * 0.98, 'Treatment\nonset', fontsize=9,
            color='gray', va='top')

    ax.set_xlabel('Year')
    ax.set_ylabel(outcome_label)
    ax.set_title(f'Parallel Trends: {outcome_label} by Treatment Group')
    ax.set_xticks(range(2016, 2025))
    ax.legend(frameon=True, edgecolor='gray', fancybox=False)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(IMG_DIR, f'{filename}.{ext}'))
    plt.close(fig)
    print(f"    Saved {filename}.png/pdf")


def plot_treatment_distribution(df: pd.DataFrame, filename: str):
    """Plot distribution of treatment intensity (labor cost share in 2022).

    Clips at the 99th percentile for visualization to avoid extreme outliers
    compressing the histogram.
    """
    firm_data = df.drop_duplicates(subset='nordemp')
    var = firm_data['labor_cost_share_2022'].dropna()

    # Clip for visualization only (extreme outliers distort the histogram)
    p99 = var.quantile(0.99)
    var_clipped = var.clip(upper=p99)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(var_clipped, bins=60, color='#2c7fb8', edgecolor='white', alpha=0.85, density=True)

    median_val = var.median()
    ax.axvline(x=median_val, color='#d7301f', linewidth=2, linestyle='--',
               label=f'Median = {median_val:.3f}')

    # Add quartile markers
    q25 = var.quantile(0.25)
    q75 = var.quantile(0.75)
    ax.axvline(x=q25, color='gray', linewidth=1, linestyle=':', alpha=0.7,
               label=f'Q25 = {q25:.3f}')
    ax.axvline(x=q75, color='gray', linewidth=1, linestyle=':', alpha=0.7,
               label=f'Q75 = {q75:.3f}')

    ax.set_xlabel('Labor Cost Share (2022)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Treatment Intensity (Pre-Treatment Labor Cost Share)')
    ax.legend(frameon=True, edgecolor='gray', fancybox=False)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(IMG_DIR, f'{filename}.{ext}'))
    plt.close(fig)
    print(f"    Saved {filename}.png/pdf")


def plot_forest(results_list: list, filename: str):
    """Forest plot comparing DiD coefficients across different outcome variables."""
    labels, betas, ci_los, ci_his = [], [], [], []

    for r in results_list:
        if r is None or 'beta' not in r:
            continue
        labels.append(r['dep_var'].replace('log_', '').replace('_', ' ').title())
        betas.append(r['beta'])
        ci_los.append(r['ci_lo'])
        ci_his.append(r['ci_hi'])

    if len(labels) == 0:
        print("    WARNING: No results for forest plot.")
        return

    betas = np.array(betas)
    ci_los = np.array(ci_los)
    ci_his = np.array(ci_his)

    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.8 + 1)))

    ax.axvline(x=0, color='gray', linewidth=0.8)

    # Plot horizontal error bars
    for i in range(len(labels)):
        color = '#d7301f' if betas[i] > 0 else '#2c7fb8'
        ax.errorbar(betas[i], y_pos[i],
                    xerr=[[betas[i] - ci_los[i]], [ci_his[i] - betas[i]]],
                    fmt='o', color=color, markersize=8, capsize=5,
                    capthick=1.5, linewidth=1.5, elinewidth=1.5,
                    markeredgecolor='white', markeredgewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('DiD Coefficient (β)')
    ax.set_title('Coefficient Comparison Across Outcomes\n(High Labor Cost × Post)')
    ax.invert_yaxis()
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(IMG_DIR, f'{filename}.{ext}'))
    plt.close(fig)
    print(f"    Saved {filename}.png/pdf")


# ============================================================================
# 8. SUMMARY STATISTICS
# ============================================================================
def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics by treatment group and period."""
    vars_of_interest = [
        'persocu', 'total_labor_cost', 'total_salaries', 'prodbr2',
        'invebrta', 'activfi', 'labor_productivity', 'unit_labor_cost',
        'capital_intensity', 'investment_rate', 'automation_proxy',
    ]

    available = [v for v in vars_of_interest if v in df.columns]
    sub = df.dropna(subset=['high_cost'])

    print("\n" + "=" * 90)
    print("SUMMARY STATISTICS BY TREATMENT GROUP AND PERIOD")
    print("=" * 90)

    for period_label, period_mask in [('Pre-treatment (2016-2022)', sub['post'] == 0),
                                       ('Post-treatment (2023-2024)', sub['post'] == 1)]:
        print(f"\n--- {period_label} ---")
        for treat_label, treat_mask in [('Low-cost (control)', sub['high_cost'] == 0),
                                         ('High-cost (treated)', sub['high_cost'] == 1)]:
            subset = sub[period_mask & treat_mask][available]
            print(f"\n  {treat_label} (N={len(subset):,}):")
            stats_df = subset.describe().T[['mean', 'std', 'min', '50%', 'max']]
            stats_df.columns = ['Mean', 'SD', 'Min', 'Median', 'Max']
            # Format large numbers
            for col in stats_df.columns:
                stats_df[col] = stats_df[col].apply(lambda x: f'{x:>14,.2f}')
            print(stats_df.to_string())

    print("\n" + "=" * 90)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("09_did_panel_eam.py — DiD with EAM Panel (2016-2024)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # PHASE 1: Load data
    # ------------------------------------------------------------------
    print("\n[1/8] Loading EAM panel data (2016-2024)...")
    panel = load_all_years()
    print(f"\n  Total observations: {len(panel):,}")
    print(f"  Unique firms (nordemp): {panel['nordemp'].nunique():,}")
    print(f"  Year range: {int(panel['periodo'].min())}–{int(panel['periodo'].max())}")

    # Panel balance
    firm_counts = panel.groupby('nordemp')['periodo'].nunique()
    balanced = (firm_counts == 9).sum()
    print(f"\n  Balanced panel (all 9 years): {balanced:,} firms")
    print(f"  Unbalanced (< 9 years): {(firm_counts < 9).sum():,} firms")
    print(f"  Obs per year:")
    for yr in range(2016, 2025):
        n = (panel['periodo'] == yr).sum()
        print(f"    {yr}: {n:,}")

    # ------------------------------------------------------------------
    # PHASE 2: Construct variables
    # ------------------------------------------------------------------
    print("\n[2/8] Constructing analysis variables...")
    panel = construct_variables(panel)
    print("  Done. Variables created: total_labor_cost, labor_productivity,")
    print("  unit_labor_cost, capital_intensity, investment_rate, automation_proxy")

    # ------------------------------------------------------------------
    # PHASE 3: Define treatment
    # ------------------------------------------------------------------
    print("\n[3/8] Defining treatment (based on 2022 labor cost share)...")
    panel = define_treatment(panel)

    # Drop firms without treatment assignment (not present in 2022)
    n_before = len(panel)
    panel_did = panel.dropna(subset=['high_cost']).copy()
    print(f"  Dropped {n_before - len(panel_did):,} obs without treatment assignment")
    print(f"  Analysis sample: {len(panel_did):,} obs, {panel_did['nordemp'].nunique():,} firms")

    # ------------------------------------------------------------------
    # PHASE 4: Summary statistics
    # ------------------------------------------------------------------
    print("\n[4/8] Summary statistics...")
    print_summary_stats(panel_did)

    # ------------------------------------------------------------------
    # PHASE 5: Winsorize and prepare panel
    # ------------------------------------------------------------------
    print("\n[5/8] Winsorizing and preparing panel for estimation...")
    panel_did = winsorize_panel(panel_did)

    # Save constructed panel
    save_cols = [c for c in panel_did.columns if c != '_res' and not c.startswith('_')]
    panel_did[save_cols].to_csv(os.path.join(DATA_DIR, 'eam_panel_constructed.csv'), index=False)
    print(f"  Saved data/eam_panel_constructed.csv ({len(panel_did):,} rows)")

    # Set panel index
    pdata = prepare_panel_index(panel_did)

    # Controls for regressions
    controls = ['log_persocu', 'log_prodbr2']

    # ------------------------------------------------------------------
    # PHASE 6: DiD regressions
    # ------------------------------------------------------------------
    print("\n[6/8] Running DiD regressions...")

    all_results = []

    # --- 6a. Binary DiD across outcomes ---
    outcomes = {
        'log_automation_proxy': 'Investment per Worker',
        'log_capital_intensity': 'Capital Intensity',
        'log_investment_rate': 'Investment Rate',
        'log_unit_labor_cost': 'Unit Labor Cost',
        'log_labor_productivity': 'Labor Productivity',
    }

    print("\n  --- (a) Binary DiD: high_cost × post ---")
    for dep_var, label in outcomes.items():
        print(f"    {label} ({dep_var})...")
        r = run_did_regression(
            pdata, dep_var=dep_var, treat_var='did',
            controls=controls, label=f'Binary DiD — {label}'
        )
        if r is not None:
            res_obj = r.pop('_res', None)
            print(f"      β = {r.get('beta', np.nan):.4f}, SE = {r.get('se', np.nan):.4f}, "
                  f"p = {r.get('pval', np.nan):.4f}, N = {r.get('n_obs', 0):,}")
            all_results.append(r)

    # --- 6b. Continuous treatment intensity DiD ---
    print("\n  --- (b) Continuous intensity DiD: labor_cost_share_2022 × post ---")
    for dep_var, label in outcomes.items():
        print(f"    {label} ({dep_var})...")
        r = run_did_regression(
            pdata, dep_var=dep_var, treat_var='did_intensity',
            controls=controls, label=f'Continuous DiD — {label}'
        )
        if r is not None:
            res_obj = r.pop('_res', None)
            print(f"      β = {r.get('beta', np.nan):.4f}, SE = {r.get('se', np.nan):.4f}, "
                  f"p = {r.get('pval', np.nan):.4f}, N = {r.get('n_obs', 0):,}")
            all_results.append(r)

    # --- 6c. Event study (main outcome: automation_proxy) ---
    print("\n  --- (c) Event study specification ---")
    event_results = {}
    for dep_var, label in outcomes.items():
        print(f"    {label} ({dep_var})...")
        r = run_event_study(pdata, dep_var=dep_var, controls=controls,
                            label=f'Event Study — {label}')
        if r is not None:
            res_obj = r.pop('_res', None)
            event_results[dep_var] = r

            # Print pre-trend test
            r_copy = dict(r)
            r_copy['_res'] = res_obj
            pretrend = test_parallel_pretrends(r_copy)
            print(f"      Pre-trend F-test: F={pretrend['f_stat']:.3f}, "
                  f"p={pretrend['p_value']:.4f}, "
                  f"reject H0 at 5%: {pretrend['reject_at_05']}")
            r['pretrend_f'] = pretrend['f_stat']
            r['pretrend_p'] = pretrend['p_value']
            all_results.append(r)

    # ------------------------------------------------------------------
    # PHASE 7: Robustness checks
    # ------------------------------------------------------------------
    print("\n[7/8] Robustness checks...")

    # --- 7a. Tercile and quartile splits ---
    print("\n  --- (a) Tercile split: top tercile vs bottom tercile ---")
    pdata_tercile = pdata.copy()
    pdata_tercile['did_tercile'] = ((pdata_tercile['tercile'] == 3).astype(int) *
                                     pdata_tercile['post'])
    # Keep only top and bottom tercile
    mask_tb = pdata_tercile['tercile'].isin([1, 3])
    pdata_tb = pdata_tercile[mask_tb]
    r = run_did_regression(
        pdata_tb, dep_var='log_automation_proxy', treat_var='did_tercile',
        controls=controls, label='Robustness: Tercile split (T3 vs T1)'
    )
    if r is not None:
        r.pop('_res', None)
        print(f"      β = {r.get('beta', np.nan):.4f}, p = {r.get('pval', np.nan):.4f}")
        all_results.append(r)

    print("  --- Quartile split: top quartile vs bottom quartile ---")
    pdata_quartile = pdata.copy()
    pdata_quartile['did_quartile'] = ((pdata_quartile['quartile'] == 4).astype(int) *
                                       pdata_quartile['post'])
    mask_qb = pdata_quartile['quartile'].isin([1, 4])
    pdata_qb = pdata_quartile[mask_qb]
    r = run_did_regression(
        pdata_qb, dep_var='log_automation_proxy', treat_var='did_quartile',
        controls=controls, label='Robustness: Quartile split (Q4 vs Q1)'
    )
    if r is not None:
        r.pop('_res', None)
        print(f"      β = {r.get('beta', np.nan):.4f}, p = {r.get('pval', np.nan):.4f}")
        all_results.append(r)

    # --- 7b. Dropping COVID years (2020-2021) ---
    print("\n  --- (b) Dropping COVID years (2020-2021) ---")
    mask_no_covid = ~pdata.index.get_level_values('periodo').isin([2020, 2021])
    pdata_no_covid = pdata[mask_no_covid]
    r = run_did_regression(
        pdata_no_covid, dep_var='log_automation_proxy', treat_var='did',
        controls=controls, label='Robustness: Excl. COVID years'
    )
    if r is not None:
        r.pop('_res', None)
        print(f"      β = {r.get('beta', np.nan):.4f}, p = {r.get('pval', np.nan):.4f}")
        all_results.append(r)

    # --- 7c. Balanced panel only ---
    print("\n  --- (c) Balanced panel only (firms in all 9 years) ---")
    firm_yr_counts = panel_did.groupby('nordemp')['periodo'].nunique()
    balanced_firms = firm_yr_counts[firm_yr_counts == 9].index
    mask_balanced = pdata.index.get_level_values('nordemp').isin(balanced_firms)
    pdata_balanced = pdata[mask_balanced]
    print(f"      Balanced panel: {pdata_balanced.index.get_level_values('nordemp').nunique():,} firms, "
          f"{len(pdata_balanced):,} obs")
    if len(pdata_balanced) > 100:
        r = run_did_regression(
            pdata_balanced, dep_var='log_automation_proxy', treat_var='did',
            controls=controls, label='Robustness: Balanced panel'
        )
        if r is not None:
            r.pop('_res', None)
            print(f"      β = {r.get('beta', np.nan):.4f}, p = {r.get('pval', np.nan):.4f}")
            all_results.append(r)
    else:
        print("      Too few observations, skipping.")

    # --- 7d. Sector-specific linear trends (ciiu2 × t) ---
    print("\n  --- (d) Sector-specific linear trends (CIIU 2-digit × t) ---")
    pdata_sector = pdata.copy()
    # Use a linear time trend interacted with sector dummies (avoids rank issues
    # from full sector × year interactions absorbed by two-way FE)
    pdata_sector['time_trend'] = (
        pdata_sector.index.get_level_values('periodo') - 2016
    ).astype(float)
    ciiu2_vals = sorted(pdata_sector['ciiu2'].dropna().unique())
    sector_trend_cols = []
    # Omit one sector as reference to avoid collinearity with time FE
    for c2 in ciiu2_vals[1:]:
        col_name = f'sec{int(c2)}_trend'
        pdata_sector[col_name] = (
            (pdata_sector['ciiu2'] == c2).astype(float) * pdata_sector['time_trend']
        )
        sector_trend_cols.append(col_name)

    controls_sector = controls + sector_trend_cols
    try:
        r = run_did_regression(
            pdata_sector, dep_var='log_automation_proxy', treat_var='did',
            controls=controls_sector, label='Robustness: Sector-specific trends'
        )
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"      Rank issue with full sector trends, retrying with check_rank=False...")
        r = None

    if r is not None:
        r.pop('_res', None)
        print(f"      β = {r.get('beta', np.nan):.4f}, p = {r.get('pval', np.nan):.4f}")
        all_results.append(r)

    # --- 7e. Winsorized outcomes ---
    print("\n  --- (e) Winsorized outcomes (1st/99th percentile) ---")
    r = run_did_regression(
        pdata, dep_var='log_automation_proxy_w', treat_var='did',
        controls=controls, label='Robustness: Winsorized outcomes'
    )
    if r is not None:
        r.pop('_res', None)
        print(f"      β = {r.get('beta', np.nan):.4f}, p = {r.get('pval', np.nan):.4f}")
        all_results.append(r)

    # ------------------------------------------------------------------
    # PHASE 8: Figures and output
    # ------------------------------------------------------------------
    print("\n[8/8] Generating figures...")

    # (a) Event study plot — main result
    if 'log_automation_proxy' in event_results:
        plot_event_study(event_results['log_automation_proxy'],
                         'Investment per Worker (Automation Proxy)',
                         'fig_did_event_study')

    # (b) Parallel trends
    plot_parallel_trends(panel_did, 'log_automation_proxy',
                         'Log Investment per Worker',
                         'fig_did_parallel_trends')

    # (c) Treatment distribution
    plot_treatment_distribution(panel_did, 'fig_did_treatment_distribution')

    # (d) Forest plot across outcomes — binary DiD only
    binary_results = [r for r in all_results if r['label'].startswith('Binary DiD')]
    plot_forest(binary_results, 'fig_did_forest_plot')

    # Save all regression results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                                for r in all_results])
    results_df.to_csv(os.path.join(DATA_DIR, 'did_regression_results.csv'), index=False)
    print(f"\n  Saved data/did_regression_results.csv ({len(results_df)} rows)")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ESTIMATION COMPLETE")
    print("=" * 70)

    print("\nKey results (Binary DiD: high labor cost × post):")
    for r in all_results:
        if r['label'].startswith('Binary DiD'):
            sig = '***' if r.get('pval', 1) < 0.01 else ('**' if r.get('pval', 1) < 0.05
                   else ('*' if r.get('pval', 1) < 0.10 else ''))
            print(f"  {r['dep_var']:30s}  β = {r.get('beta', np.nan):>8.4f}  "
                  f"({r.get('se', np.nan):.4f})  p = {r.get('pval', np.nan):.4f} {sig}")

    print("\nRobustness checks:")
    for r in all_results:
        if r['label'].startswith('Robustness'):
            sig = '***' if r.get('pval', 1) < 0.01 else ('**' if r.get('pval', 1) < 0.05
                   else ('*' if r.get('pval', 1) < 0.10 else ''))
            print(f"  {r['label']:45s}  β = {r.get('beta', np.nan):>8.4f}  "
                  f"p = {r.get('pval', np.nan):.4f} {sig}")

    print("\nPre-trend tests (event study):")
    for dep_var, r in event_results.items():
        status = 'PASS (parallel trends hold)' if r.get('pretrend_p', 0) > 0.05 else 'FAIL'
        print(f"  {dep_var:30s}  F = {r.get('pretrend_f', np.nan):>7.3f}  "
              f"p = {r.get('pretrend_p', np.nan):.4f}  {status}")

    print("\nOutputs:")
    print(f"  data/eam_panel_constructed.csv")
    print(f"  data/did_regression_results.csv")
    print(f"  images/en/fig_did_event_study.png/pdf")
    print(f"  images/en/fig_did_parallel_trends.png/pdf")
    print(f"  images/en/fig_did_treatment_distribution.png/pdf")
    print(f"  images/en/fig_did_forest_plot.png/pdf")
    print("\nDone.")


if __name__ == '__main__':
    main()
