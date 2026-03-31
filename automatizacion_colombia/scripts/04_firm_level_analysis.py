#!/usr/bin/env python3
"""
04_firm_level_analysis.py
==========================
Firm-level analysis: Merging EAM and EDIT microdata to study the relationship
between labor costs and automation/innovation investment in Colombian manufacturing.

Research questions:
  1. Do higher labor costs (unit labor cost, labor share) drive automation investment?
  2. Is there a positive relationship between labor cost per worker and capital intensity?
  3. What firm and sector characteristics predict innovation adoption?

Data:
  - EAM 2023 (Encuesta Anual Manufacturera): 6,714 establishments, tab-separated
  - EDIT X 2019-2020 (Encuesta de Desarrollo e Innovacion Tecnologica): 6,798 firms, semicolon-separated
  - Merge key: NORDEMP (anonymized firm ID) -- ~96% merge rate from EAM side

Author: Research team
Date: 2026-03-06
"""

import os
import sys
import zipfile
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMG_DIR = os.path.join(BASE_DIR, 'images')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')

EAM_ZIP = os.path.join(DATA_DIR, 'BDATOS-EAM-2023.zip')
EDIT_ZIP = os.path.join(DATA_DIR, 'EDIT_X_2019_2020.zip')
EAM_TEMP = os.path.join(DATA_DIR, 'eam_temp')
EDIT_TEMP = os.path.join(DATA_DIR, 'edit_temp')

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(EAM_TEMP, exist_ok=True)
os.makedirs(EDIT_TEMP, exist_ok=True)

# Publication style
MAINBLUE = '#1B4F72'
ACCENTRED = '#C0392B'
DARKGRAY = '#2C3E50'
LIGHTGRAY = '#BDC3C7'
MEDGRAY = '#7F8C8D'
TEAL = '#117A65'
ORANGE = '#E67E22'
PURPLE = '#6C3483'

SECTOR_PALETTE = [MAINBLUE, ACCENTRED, TEAL, ORANGE, PURPLE,
                  '#2E86C1', '#D4AC0D', '#A93226', '#1ABC9C', '#8E44AD',
                  '#F39C12', '#2C3E50', '#27AE60', '#E74C3C', '#3498DB']

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': DARKGRAY,
    'axes.labelcolor': DARKGRAY,
    'xtick.color': DARKGRAY,
    'ytick.color': DARKGRAY,
})

# CIIU Rev 4 sector names (2-digit manufacturing)
CIIU_2D_NAMES = {
    10: 'Alimentos',
    11: 'Bebidas',
    12: 'Tabaco',
    13: 'Textiles',
    14: 'Confecciones',
    15: 'Cuero y calzado',
    16: 'Madera',
    17: 'Papel',
    18: 'Imprenta',
    19: 'Coque y refinacion',
    20: 'Quimicos',
    21: 'Farmaceuticos',
    22: 'Caucho y plastico',
    23: 'Min. no metalicos',
    24: 'Metalurgia basica',
    25: 'Prod. metalicos',
    26: 'Electronica',
    27: 'Equipo electrico',
    28: 'Maquinaria',
    29: 'Vehiculos',
    30: 'Otro eq. transporte',
    31: 'Muebles',
    32: 'Otras manufacturas',
    33: 'Rep. e instalacion',
}


def print_section(title):
    """Print formatted section header."""
    width = 80
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def safe_log(x, min_val=1):
    """Safe log transformation: log(max(x, min_val))."""
    return np.log(np.maximum(x, min_val))


# =============================================================================
# STEP 1: EXTRACT AND LOAD DATASETS
# =============================================================================
def step1_load_data():
    """Extract and load EAM and EDIT microdata."""
    print_section("STEP 1: Loading EAM 2023 and EDIT X 2019-2020")

    # --- Extract EAM ---
    eam_file = os.path.join(EAM_TEMP, 'EAM_ANONIMIZADA_2023.txt')
    if not os.path.exists(eam_file):
        print("  Extracting EAM from ZIP...")
        with zipfile.ZipFile(EAM_ZIP, 'r') as z:
            for f in z.infolist():
                if 'EAM_ANONIMIZADA_2023.txt' in f.filename:
                    f.filename = os.path.basename(f.filename)
                    z.extract(f, EAM_TEMP)
                    break

    print(f"  Loading EAM: {eam_file}")
    df_eam = pd.read_csv(eam_file, sep='\t', encoding='latin-1')
    print(f"    Shape: {df_eam.shape[0]:,} rows x {df_eam.shape[1]} columns")

    # --- Extract EDIT ---
    edit_file = os.path.join(EDIT_TEMP, 'EDIT_X_2019_2020.csv')
    if not os.path.exists(edit_file):
        print("  Extracting EDIT from ZIP...")
        with zipfile.ZipFile(EDIT_ZIP, 'r') as z:
            for f in z.infolist():
                if 'EDIT_X_2019_2020.csv' in f.filename:
                    f.filename = os.path.basename(f.filename)
                    z.extract(f, EDIT_TEMP)
                    break

    print(f"  Loading EDIT: {edit_file}")
    df_edit = pd.read_csv(edit_file, sep=';', encoding='latin-1')
    print(f"    Shape: {df_edit.shape[0]:,} rows x {df_edit.shape[1]} columns")

    # Quick diagnostics
    print(f"\n  EAM unique firms (nordemp): {df_eam['nordemp'].nunique():,}")
    print(f"  EAM unique establishments (nordest): {df_eam['nordest'].nunique():,}")
    print(f"  EDIT unique firms (NORDEMP): {df_edit['NORDEMP'].nunique():,}")

    return df_eam, df_edit


# =============================================================================
# STEP 2: CLEAN AND PREPARE EAM DATA
# =============================================================================
def step2_clean_eam(df_eam):
    """Clean EAM and construct firm-level analytical variables."""
    print_section("STEP 2: Cleaning and preparing EAM data")

    df = df_eam.copy()

    # --- Aggregate establishments to firm level ---
    # EAM has 6,714 establishments but 6,119 unique firms.
    # Some firms have multiple establishments. We aggregate to firm level.
    print(f"  Establishments: {len(df):,}")
    print(f"  Unique firms: {df['nordemp'].nunique():,}")

    # Numeric columns to sum at firm level
    sum_cols = ['SALARPER', 'PRESSPER', 'SALPEYTE', 'PRESPYTE', 'REMUTEMP',
                'SALAPREN', 'PERTOTAL', 'PERSOCU', 'PPERYTEM', 'PERTEM3',
                'PERAPREN', 'PRODBR2', 'VALAGRI', 'INVEBRTA', 'ACTIVFI',
                'DEPRECIA', 'VALORVEN', 'VALORCOM', 'VALORCX', 'EELEC',
                'CONSMATE']

    # Check which columns exist
    sum_cols = [c for c in sum_cols if c in df.columns]

    # For ciiu4 and dpto, take the mode (most common across establishments)
    agg_dict = {c: 'sum' for c in sum_cols}
    agg_dict['ciiu4'] = 'first'  # sector of primary establishment
    agg_dict['dpto'] = 'first'
    agg_dict['nordest'] = 'count'  # count establishments per firm

    df_firm = df.groupby('nordemp').agg(agg_dict).reset_index()
    df_firm = df_firm.rename(columns={'nordest': 'n_establishments'})
    print(f"  Firm-level aggregation: {len(df_firm):,} firms")

    # --- Construct analytical variables ---
    # Total labor cost: salaries + benefits for permanent + temporary workers
    # SALPEYTE already includes perm+temp salaries; PRESPYTE includes perm+temp benefits
    df_firm['total_labor_cost'] = df_firm['SALPEYTE'] + df_firm['PRESPYTE']

    # Employment: use PPERYTEM (permanent + temporary) as denominator for per-worker metrics
    # PERTOTAL can include outsourced workers not covered by the salary data
    df_firm['total_employment'] = df_firm['PPERYTEM']

    # Flag and handle problematic observations
    df_firm['flag_zero_employment'] = (df_firm['total_employment'] <= 0).astype(int)
    df_firm['flag_zero_va'] = (df_firm['VALAGRI'] <= 0).astype(int)
    df_firm['flag_zero_production'] = (df_firm['PRODBR2'] <= 0).astype(int)
    df_firm['flag_zero_labor_cost'] = (df_firm['total_labor_cost'] <= 0).astype(int)

    n_zero_emp = df_firm['flag_zero_employment'].sum()
    n_zero_va = df_firm['flag_zero_va'].sum()
    n_zero_prod = df_firm['flag_zero_production'].sum()
    n_zero_lc = df_firm['flag_zero_labor_cost'].sum()
    print(f"\n  Data quality flags:")
    print(f"    Zero employment:   {n_zero_emp:,} ({n_zero_emp/len(df_firm)*100:.1f}%)")
    print(f"    Zero/neg VA:       {n_zero_va:,} ({n_zero_va/len(df_firm)*100:.1f}%)")
    print(f"    Zero/neg prod:     {n_zero_prod:,} ({n_zero_prod/len(df_firm)*100:.1f}%)")
    print(f"    Zero labor cost:   {n_zero_lc:,} ({n_zero_lc/len(df_firm)*100:.1f}%)")

    # Compute per-worker and ratio variables (only for valid observations)
    valid = (df_firm['total_employment'] > 0) & (df_firm['VALAGRI'] > 0)
    print(f"    Valid sample (emp>0 & VA>0): {valid.sum():,} ({valid.sum()/len(df_firm)*100:.1f}%)")

    # Labor cost per worker (thousands of COP)
    df_firm['labor_cost_per_worker'] = np.where(
        df_firm['total_employment'] > 0,
        df_firm['total_labor_cost'] / df_firm['total_employment'],
        np.nan
    )

    # Value added
    df_firm['value_added'] = df_firm['VALAGRI']
    df_firm['gross_production'] = df_firm['PRODBR2']

    # Labor share of value added
    df_firm['labor_share_va'] = np.where(
        df_firm['VALAGRI'] > 0,
        df_firm['total_labor_cost'] / df_firm['VALAGRI'],
        np.nan
    )

    # Labor productivity (VA per worker)
    df_firm['labor_productivity'] = np.where(
        df_firm['total_employment'] > 0,
        df_firm['VALAGRI'] / df_firm['total_employment'],
        np.nan
    )

    # Investment variables
    df_firm['investment_total'] = df_firm['INVEBRTA']

    # Capital intensity (fixed assets per worker)
    df_firm['capital_intensity'] = np.where(
        df_firm['total_employment'] > 0,
        df_firm['ACTIVFI'] / df_firm['total_employment'],
        np.nan
    )

    # Investment rate (investment / VA)
    df_firm['investment_rate'] = np.where(
        df_firm['VALAGRI'] > 0,
        df_firm['INVEBRTA'] / df_firm['VALAGRI'],
        np.nan
    )

    # Unit labor cost (labor cost per worker / labor productivity)
    # = total_labor_cost / value_added (same as labor_share_va!)
    # More precisely: ULC = (w*L) / (Y/L) * (1/L) = w / (Y/L)
    df_firm['unit_labor_cost'] = np.where(
        (df_firm['labor_productivity'] > 0) & (df_firm['labor_productivity'].notna()),
        df_firm['labor_cost_per_worker'] / df_firm['labor_productivity'],
        np.nan
    )

    # Sector codes
    df_firm['sector_2d'] = (df_firm['ciiu4'] // 100).astype(int)
    df_firm['sector_name'] = df_firm['sector_2d'].map(CIIU_2D_NAMES).fillna('Otro')

    # Firm size categories
    def classify_size(emp):
        if pd.isna(emp) or emp <= 0:
            return 'Sin dato'
        elif emp < 10:
            return 'Micro (<10)'
        elif emp <= 50:
            return 'Pequena (10-50)'
        elif emp <= 200:
            return 'Mediana (51-200)'
        else:
            return 'Grande (>200)'

    df_firm['firm_size'] = df_firm['total_employment'].apply(classify_size)
    size_order = ['Micro (<10)', 'Pequena (10-50)', 'Mediana (51-200)', 'Grande (>200)', 'Sin dato']
    df_firm['firm_size'] = pd.Categorical(df_firm['firm_size'], categories=size_order, ordered=True)

    # Log transformations (safe: using log(max(x, 1)))
    for var in ['total_labor_cost', 'total_employment', 'labor_cost_per_worker',
                'value_added', 'gross_production', 'labor_productivity',
                'investment_total', 'capital_intensity']:
        df_firm[f'log_{var}'] = np.where(
            df_firm[var] > 0,
            np.log(df_firm[var]),
            np.nan
        )

    # Export share
    df_firm['export_share'] = np.where(
        df_firm['VALORVEN'] > 0,
        df_firm['VALORCX'] / df_firm['VALORVEN'],
        0
    )

    # Print summary statistics
    print(f"\n  Summary of key constructed variables (valid obs only):")
    key_vars = ['total_labor_cost', 'total_employment', 'labor_cost_per_worker',
                'value_added', 'labor_share_va', 'labor_productivity',
                'capital_intensity', 'investment_rate', 'unit_labor_cost']
    for v in key_vars:
        vals = df_firm[v].dropna()
        vals_pos = vals[vals > 0]
        print(f"    {v:30s}: N={len(vals_pos):,}, mean={vals_pos.mean():>15,.1f}, "
              f"median={vals_pos.median():>15,.1f}, sd={vals_pos.std():>15,.1f}")

    print(f"\n  Firm size distribution:")
    for size, count in df_firm['firm_size'].value_counts().sort_index().items():
        print(f"    {size:20s}: {count:,} ({count/len(df_firm)*100:.1f}%)")

    return df_firm


# =============================================================================
# STEP 3: CLEAN AND PREPARE EDIT DATA
# =============================================================================
def step3_clean_edit(df_edit):
    """Clean EDIT and construct innovation/technology variables."""
    print_section("STEP 3: Cleaning and preparing EDIT data")

    df = df_edit.copy()

    # --- Innovation investment variables ---
    # Sum year 1 (2019) + year 2 (2020) for each investment component
    # Fill NaN with 0 for summation within observed firms (NaN = not surveyed for that chapter)

    def safe_sum_cols(df, col1, col2):
        """Sum two columns, treating NaN as 0 only if at least one is non-NaN."""
        s1 = df[col1] if col1 in df.columns else pd.Series(0, index=df.index)
        s2 = df[col2] if col2 in df.columns else pd.Series(0, index=df.index)
        both_nan = s1.isna() & s2.isna()
        result = s1.fillna(0) + s2.fillna(0)
        result[both_nan] = np.nan
        return result

    # Machinery & equipment for innovation
    df['machinery_innovation'] = safe_sum_cols(df, 'II1R3C1', 'II1R3C2')

    # ICT investment
    df['ict_investment'] = safe_sum_cols(df, 'II1R4C1', 'II1R4C2')

    # Software for innovation
    df['software_investment'] = safe_sum_cols(df, 'II1R10C1', 'II1R10C2')

    # Internal R&D
    df['rd_internal'] = safe_sum_cols(df, 'II1R1C1', 'II1R1C2')

    # External R&D
    df['rd_external'] = safe_sum_cols(df, 'II1R2C1', 'II1R2C2')

    # Technology transfer
    df['tech_transfer'] = safe_sum_cols(df, 'II1R5C1', 'II1R5C2')

    # Training for innovation
    df['training_innovation'] = safe_sum_cols(df, 'II1R8C1', 'II1R8C2')

    # Total ACTI -- compute as sum of all components (more reliable than II1R12)
    acti_components = ['rd_internal', 'rd_external', 'machinery_innovation',
                       'ict_investment', 'tech_transfer', 'training_innovation',
                       'software_investment']
    # Also add remaining components
    df['engineering_design'] = safe_sum_cols(df, 'II1R7C1', 'II1R7C2')
    df['consulting'] = safe_sum_cols(df, 'II1R6C1', 'II1R6C2')
    df['market_intro'] = safe_sum_cols(df, 'II1R9C1', 'II1R9C2')
    df['ip_costs'] = safe_sum_cols(df, 'II1R11C1', 'II1R11C2')

    all_components = acti_components + ['engineering_design', 'consulting',
                                        'market_intro', 'ip_costs']

    # Total ACTI = sum of all components
    df['total_acti'] = df[all_components].sum(axis=1, min_count=1)

    # R&D spending (internal + external)
    df['rd_spending'] = df['rd_internal'].fillna(0) + df['rd_external'].fillna(0)
    both_rd_nan = df['rd_internal'].isna() & df['rd_external'].isna()
    df.loc[both_rd_nan, 'rd_spending'] = np.nan

    # Automation proxy: machinery + ICT + software (technology-embodied investment)
    df['automation_proxy'] = (df['machinery_innovation'].fillna(0) +
                              df['ict_investment'].fillna(0) +
                              df['software_investment'].fillna(0))
    all_three_nan = (df['machinery_innovation'].isna() &
                     df['ict_investment'].isna() &
                     df['software_investment'].isna())
    df.loc[all_three_nan, 'automation_proxy'] = np.nan

    # --- Innovation typology ---
    df['innovator'] = df['TIPOLO'].isin(['ESTRIC', 'AMPLIA']).astype(int)
    df['strict_innovator'] = (df['TIPOLO'] == 'ESTRIC').astype(int)
    df['broad_innovator'] = (df['TIPOLO'] == 'AMPLIA').astype(int)
    df['potential_innovator'] = (df['TIPOLO'] == 'POTENC').astype(int)
    df['non_innovator'] = (df['TIPOLO'] == 'NOINNO').astype(int)

    # Process innovation indicators (Chapter IV)
    # IV1R5 = new process, IV1R6 = improved process
    if 'IV1R5C1' in df.columns and 'IV1R6C1' in df.columns:
        df['process_innovator'] = ((df['IV1R5C1'].fillna(0) > 0) |
                                    (df['IV1R5C2'].fillna(0) > 0) |
                                    (df['IV1R6C1'].fillna(0) > 0) |
                                    (df['IV1R6C2'].fillna(0) > 0)).astype(int)
    else:
        df['process_innovator'] = 0

    # Sector
    df['sector_2d'] = (df['CIIU4'] // 100).astype(int)
    df['sector_name'] = df['sector_2d'].map(CIIU_2D_NAMES).fillna('Otro')

    # Log transformations
    for var in ['automation_proxy', 'total_acti', 'machinery_innovation',
                'ict_investment', 'software_investment', 'rd_spending']:
        df[f'log_{var}'] = np.where(df[var] > 0, np.log(df[var]), np.nan)

    # Print summary
    print(f"  Innovation typology distribution:")
    for cat, count in df['TIPOLO'].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {cat:10s}: {count:5,} ({pct:5.1f}%)")

    print(f"\n  Innovation investment summary (thousands of COP, both years summed):")
    inv_vars = ['machinery_innovation', 'ict_investment', 'software_investment',
                'rd_internal', 'rd_external', 'total_acti', 'automation_proxy']
    for v in inv_vars:
        vals = df[v].dropna()
        n_nonzero = (vals > 0).sum()
        print(f"    {v:25s}: N_valid={len(vals):,}, N_nonzero={n_nonzero:,}, "
              f"mean={vals.mean():>12,.0f}, median={vals.median():>10,.0f}")

    return df


# =============================================================================
# STEP 4: MERGE EAM AND EDIT
# =============================================================================
def step4_merge(df_eam_clean, df_edit_clean):
    """Merge EAM and EDIT at the firm level using NORDEMP."""
    print_section("STEP 4: Merging EAM and EDIT on firm ID (NORDEMP)")

    # Rename for consistency
    eam = df_eam_clean.copy()
    edit = df_edit_clean.copy()

    # Prepare EDIT columns for merge (avoid name conflicts)
    edit_cols = ['NORDEMP', 'TIPOLO', 'innovator', 'strict_innovator', 'broad_innovator',
                 'potential_innovator', 'non_innovator', 'process_innovator',
                 'machinery_innovation', 'ict_investment', 'software_investment',
                 'rd_internal', 'rd_external', 'rd_spending', 'tech_transfer',
                 'training_innovation', 'total_acti', 'automation_proxy',
                 'log_automation_proxy', 'log_total_acti', 'log_machinery_innovation',
                 'log_ict_investment', 'log_software_investment', 'log_rd_spending']

    # Keep only columns that exist
    edit_cols = [c for c in edit_cols if c in edit.columns]
    edit_merge = edit[edit_cols].copy()

    # Merge
    merged = eam.merge(edit_merge, left_on='nordemp', right_on='NORDEMP', how='inner')

    # Report merge statistics
    n_eam = len(eam)
    n_edit = len(edit)
    n_merged = len(merged)
    n_eam_only = n_eam - n_merged
    n_edit_only = n_edit - len(edit[edit['NORDEMP'].isin(eam['nordemp'])])

    print(f"  Merge results:")
    print(f"    EAM firms:          {n_eam:,}")
    print(f"    EDIT firms:         {n_edit:,}")
    print(f"    Matched (inner):    {n_merged:,}")
    print(f"    EAM only:           {n_eam_only:,} ({n_eam_only/n_eam*100:.1f}%)")
    print(f"    EDIT only:          {n_edit_only:,} ({n_edit_only/n_edit*100:.1f}%)")
    print(f"    Merge rate (EAM):   {n_merged/n_eam*100:.1f}%")
    print(f"    Merge rate (EDIT):  {n_merged/n_edit*100:.1f}%")

    # Innovation distribution in merged sample
    print(f"\n  Innovation distribution in merged sample:")
    for cat, count in merged['TIPOLO'].value_counts().items():
        pct = count / len(merged) * 100
        print(f"    {cat:10s}: {count:5,} ({pct:5.1f}%)")

    # Has automation investment
    has_auto = (merged['automation_proxy'] > 0) & merged['automation_proxy'].notna()
    print(f"\n  Firms with positive automation proxy: {has_auto.sum():,} ({has_auto.sum()/len(merged)*100:.1f}%)")

    return merged


# =============================================================================
# STEP 5: ECONOMETRIC ANALYSIS
# =============================================================================
def step5_econometrics(df_eam_clean, df_merged):
    """Run econometric models."""
    print_section("STEP 5: Econometric analysis")

    results = {}

    # -------------------------------------------------------------------------
    # MODEL A: What drives automation investment? (Merged sample)
    # log(automation_proxy) = b0 + b1*log(unit_labor_cost) + b2*log(labor_productivity)
    #                        + b3*firm_size + b4*sector_dummies + e
    # -------------------------------------------------------------------------
    print("\n  --- Model A: Determinants of automation investment (merged sample) ---")
    ma = df_merged.copy()
    # Filter: positive automation proxy and valid covariates
    ma = ma[(ma['automation_proxy'] > 0) &
            (ma['unit_labor_cost'] > 0) & ma['unit_labor_cost'].notna() &
            (ma['labor_productivity'] > 0) & ma['labor_productivity'].notna() &
            (ma['total_employment'] > 0)].copy()
    ma = ma[ma['firm_size'] != 'Sin dato']

    if len(ma) > 50:
        ma['log_ulc'] = np.log(ma['unit_labor_cost'])
        ma['log_lp'] = np.log(ma['labor_productivity'])
        ma['log_auto'] = np.log(ma['automation_proxy'])
        ma['log_emp'] = np.log(ma['total_employment'])

        # Firm size dummies (reference: Micro)
        size_dums = pd.get_dummies(ma['firm_size'], prefix='size', drop_first=True,
                                    dtype=float)
        # Remove ordered category issues
        size_dums.columns = [str(c) for c in size_dums.columns]

        # Sector dummies (top 10 sectors)
        top_sectors = ma['sector_2d'].value_counts().head(10).index
        ma['sector_top'] = np.where(ma['sector_2d'].isin(top_sectors),
                                     ma['sector_2d'].astype(str), 'Other')
        sector_dums = pd.get_dummies(ma['sector_top'], prefix='sec', drop_first=True,
                                      dtype=float)

        X_vars = pd.concat([
            ma[['log_ulc', 'log_lp', 'log_emp']].reset_index(drop=True),
            size_dums.reset_index(drop=True),
            sector_dums.reset_index(drop=True)
        ], axis=1)
        X = sm.add_constant(X_vars)
        y = ma['log_auto'].reset_index(drop=True)

        # Drop rows with any NaN
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        if len(y) > 20:
            model_a = sm.OLS(y, X).fit(cov_type='HC3')
            results['Model_A'] = model_a
            print(f"    N = {model_a.nobs:.0f}, R2 = {model_a.rsquared:.4f}, "
                  f"Adj.R2 = {model_a.rsquared_adj:.4f}, F = {model_a.fvalue:.2f}")
            print(f"    Key coefficients:")
            for var in ['log_ulc', 'log_lp', 'log_emp']:
                if var in model_a.params.index:
                    coef = model_a.params[var]
                    se = model_a.bse[var]
                    pval = model_a.pvalues[var]
                    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                    print(f"      {var:15s}: {coef:8.4f} ({se:.4f}) {sig}")
        else:
            print("    Insufficient observations for Model A")
    else:
        print("    Insufficient observations for Model A")

    # -------------------------------------------------------------------------
    # MODEL B: Investment rate and labor share (EAM only)
    # investment_rate = b0 + b1*labor_share_va + b2*log(labor_productivity)
    #                  + b3*firm_size + b4*sector + e
    # -------------------------------------------------------------------------
    print("\n  --- Model B: Investment rate and labor share (EAM full sample) ---")
    mb = df_eam_clean.copy()
    mb = mb[(mb['investment_rate'].notna()) & np.isfinite(mb['investment_rate']) &
            (mb['labor_share_va'].notna()) & np.isfinite(mb['labor_share_va']) &
            (mb['labor_productivity'] > 0) & mb['labor_productivity'].notna() &
            (mb['total_employment'] > 0)].copy()
    mb = mb[mb['firm_size'] != 'Sin dato']

    # Winsorize extreme values (top/bottom 1%)
    for v in ['investment_rate', 'labor_share_va']:
        p01 = mb[v].quantile(0.01)
        p99 = mb[v].quantile(0.99)
        mb[v] = mb[v].clip(p01, p99)

    if len(mb) > 50:
        mb['log_lp'] = np.log(mb['labor_productivity'])
        mb['log_emp'] = np.log(mb['total_employment'])

        size_dums = pd.get_dummies(mb['firm_size'], prefix='size', drop_first=True,
                                    dtype=float)
        size_dums.columns = [str(c) for c in size_dums.columns]

        top_sectors = mb['sector_2d'].value_counts().head(10).index
        mb['sector_top'] = np.where(mb['sector_2d'].isin(top_sectors),
                                     mb['sector_2d'].astype(str), 'Other')
        sector_dums = pd.get_dummies(mb['sector_top'], prefix='sec', drop_first=True,
                                      dtype=float)

        X_vars = pd.concat([
            mb[['labor_share_va', 'log_lp', 'log_emp']].reset_index(drop=True),
            size_dums.reset_index(drop=True),
            sector_dums.reset_index(drop=True)
        ], axis=1)
        X = sm.add_constant(X_vars)
        y = mb['investment_rate'].reset_index(drop=True)

        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        if len(y) > 20:
            model_b = sm.OLS(y, X).fit(cov_type='HC3')
            results['Model_B'] = model_b
            print(f"    N = {model_b.nobs:.0f}, R2 = {model_b.rsquared:.4f}, "
                  f"Adj.R2 = {model_b.rsquared_adj:.4f}, F = {model_b.fvalue:.2f}")
            print(f"    Key coefficients:")
            for var in ['labor_share_va', 'log_lp', 'log_emp']:
                if var in model_b.params.index:
                    coef = model_b.params[var]
                    se = model_b.bse[var]
                    pval = model_b.pvalues[var]
                    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                    print(f"      {var:15s}: {coef:8.4f} ({se:.4f}) {sig}")
        else:
            print("    Insufficient observations for Model B")
    else:
        print("    Insufficient observations for Model B")

    # -------------------------------------------------------------------------
    # MODEL C: Capital intensity and labor cost per worker (EAM only)
    # log(capital_intensity) = b0 + b1*log(labor_cost_per_worker) + b2*firm_size
    #                         + b3*sector + e
    # -------------------------------------------------------------------------
    print("\n  --- Model C: Capital intensity vs labor cost per worker (EAM) ---")
    mc = df_eam_clean.copy()
    mc = mc[(mc['capital_intensity'] > 0) & mc['capital_intensity'].notna() &
            (mc['labor_cost_per_worker'] > 0) & mc['labor_cost_per_worker'].notna() &
            (mc['total_employment'] > 0)].copy()
    mc = mc[mc['firm_size'] != 'Sin dato']

    if len(mc) > 50:
        mc['log_ci'] = np.log(mc['capital_intensity'])
        mc['log_lcpw'] = np.log(mc['labor_cost_per_worker'])
        mc['log_emp'] = np.log(mc['total_employment'])

        size_dums = pd.get_dummies(mc['firm_size'], prefix='size', drop_first=True,
                                    dtype=float)
        size_dums.columns = [str(c) for c in size_dums.columns]

        top_sectors = mc['sector_2d'].value_counts().head(10).index
        mc['sector_top'] = np.where(mc['sector_2d'].isin(top_sectors),
                                     mc['sector_2d'].astype(str), 'Other')
        sector_dums = pd.get_dummies(mc['sector_top'], prefix='sec', drop_first=True,
                                      dtype=float)

        X_vars = pd.concat([
            mc[['log_lcpw', 'log_emp']].reset_index(drop=True),
            size_dums.reset_index(drop=True),
            sector_dums.reset_index(drop=True)
        ], axis=1)
        X = sm.add_constant(X_vars)
        y = mc['log_ci'].reset_index(drop=True)

        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        if len(y) > 20:
            model_c = sm.OLS(y, X).fit(cov_type='HC3')
            results['Model_C'] = model_c
            print(f"    N = {model_c.nobs:.0f}, R2 = {model_c.rsquared:.4f}, "
                  f"Adj.R2 = {model_c.rsquared_adj:.4f}, F = {model_c.fvalue:.2f}")
            print(f"    Key coefficients:")
            for var in ['log_lcpw', 'log_emp']:
                if var in model_c.params.index:
                    coef = model_c.params[var]
                    se = model_c.bse[var]
                    pval = model_c.pvalues[var]
                    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                    print(f"      {var:15s}: {coef:8.4f} ({se:.4f}) {sig}")
        else:
            print("    Insufficient observations for Model C")
    else:
        print("    Insufficient observations for Model C")

    # -------------------------------------------------------------------------
    # MODEL D: Probability of being an innovator (Logit, merged sample)
    # P(innovator=1) = Logit(b0 + b1*log(ulc) + b2*firm_size + b3*sector + e)
    # Uses full merged sample (innovator is defined from TIPOLO for all firms)
    # -------------------------------------------------------------------------
    print("\n  --- Model D: Probability of innovation (Logit, merged) ---")
    md = df_merged.copy()
    md = md[(md['unit_labor_cost'] > 0) & md['unit_labor_cost'].notna() &
            (md['labor_productivity'] > 0) & md['labor_productivity'].notna() &
            (md['total_employment'] > 0)].copy()
    md = md[md['firm_size'] != 'Sin dato']

    if len(md) > 50:
        md['log_ulc'] = np.log(md['unit_labor_cost'])
        md['log_lp'] = np.log(md['labor_productivity'])
        md['log_emp'] = np.log(md['total_employment'])

        size_dums = pd.get_dummies(md['firm_size'], prefix='size', drop_first=True,
                                    dtype=float)
        size_dums.columns = [str(c) for c in size_dums.columns]

        # Use fewer sector dummies (top 5) to avoid collinearity in Logit
        top_sectors_d = md['sector_2d'].value_counts().head(5).index
        md['sector_top'] = np.where(md['sector_2d'].isin(top_sectors_d),
                                     md['sector_2d'].astype(str), 'Other')
        sector_dums = pd.get_dummies(md['sector_top'], prefix='sec', drop_first=True,
                                      dtype=float)

        X_vars = pd.concat([
            md[['log_ulc', 'log_lp', 'log_emp']].reset_index(drop=True),
            size_dums.reset_index(drop=True),
            sector_dums.reset_index(drop=True)
        ], axis=1)
        X = sm.add_constant(X_vars)
        y = md['innovator'].reset_index(drop=True)

        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        # Drop any columns with zero variance to avoid singularity
        col_var = X.var()
        zero_var_cols = col_var[col_var == 0].index.tolist()
        if zero_var_cols:
            print(f"    Dropping zero-variance columns: {zero_var_cols}")
            X = X.drop(columns=zero_var_cols)

        if len(y) > 20 and y.nunique() > 1:
            try:
                model_d = Logit(y, X).fit(disp=0, maxiter=200, method='newton')
                results['Model_D'] = model_d
                print(f"    N = {model_d.nobs:.0f}, Pseudo R2 = {model_d.prsquared:.4f}, "
                      f"Log-Lik = {model_d.llf:.1f}")
                print(f"    Key coefficients (log-odds):")
                for var in ['log_ulc', 'log_lp', 'log_emp']:
                    if var in model_d.params.index:
                        coef = model_d.params[var]
                        se = model_d.bse[var]
                        pval = model_d.pvalues[var]
                        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                        print(f"      {var:15s}: {coef:8.4f} ({se:.4f}) {sig}")
                # Marginal effects at means
                try:
                    mfx = model_d.get_margeff(at='mean')
                    print(f"    Marginal effects at means:")
                    for var in ['log_ulc', 'log_lp', 'log_emp']:
                        if var in X.columns:
                            idx = list(X.columns).index(var)
                            me = mfx.margeff[idx]
                            me_se = mfx.margeff_se[idx]
                            print(f"      {var:15s}: {me:8.4f} ({me_se:.4f})")
                except Exception:
                    pass
            except Exception as e:
                # Fallback: try with bfgs optimizer
                try:
                    print(f"    Newton failed ({e}), trying BFGS...")
                    model_d = Logit(y, X).fit(disp=0, maxiter=200, method='bfgs')
                    results['Model_D'] = model_d
                    print(f"    N = {model_d.nobs:.0f}, Pseudo R2 = {model_d.prsquared:.4f}, "
                          f"Log-Lik = {model_d.llf:.1f}")
                    for var in ['log_ulc', 'log_lp', 'log_emp']:
                        if var in model_d.params.index:
                            coef = model_d.params[var]
                            se = model_d.bse[var]
                            pval = model_d.pvalues[var]
                            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                            print(f"      {var:15s}: {coef:8.4f} ({se:.4f}) {sig}")
                except Exception as e2:
                    print(f"    Logit failed with both methods: {e2}")
        else:
            print("    Insufficient observations or no variation in outcome")
    else:
        print("    Insufficient observations for Model D")

    # -------------------------------------------------------------------------
    # MODEL E: Tobit for automation investment (many zeros)
    # Since statsmodels doesn't have a built-in Tobit, we use:
    # (1) Heckman two-step: Probit(has_investment) then OLS(log_investment | invested)
    # (2) OLS on full sample with log(automation_proxy + 1)
    # -------------------------------------------------------------------------
    print("\n  --- Model E: Tobit-like model for automation investment ---")
    print("    (Two-part model: Probit for extensive margin + OLS for intensive margin)")

    me = df_merged.copy()
    me = me[(me['unit_labor_cost'] > 0) & me['unit_labor_cost'].notna() &
            (me['labor_productivity'] > 0) & me['labor_productivity'].notna() &
            (me['total_employment'] > 0) &
            me['automation_proxy'].notna()].copy()
    me = me[me['firm_size'] != 'Sin dato']

    if len(me) > 50:
        me['log_ulc'] = np.log(me['unit_labor_cost'])
        me['log_lp'] = np.log(me['labor_productivity'])
        me['log_emp'] = np.log(me['total_employment'])
        me['has_auto'] = (me['automation_proxy'] > 0).astype(int)
        me['log_auto_p1'] = np.log(me['automation_proxy'] + 1)

        size_dums = pd.get_dummies(me['firm_size'], prefix='size', drop_first=True,
                                    dtype=float)
        size_dums.columns = [str(c) for c in size_dums.columns]

        # Use fewer sectors (top 5) to avoid collinearity in Probit
        top_sectors_e = me['sector_2d'].value_counts().head(5).index
        me['sector_top'] = np.where(me['sector_2d'].isin(top_sectors_e),
                                     me['sector_2d'].astype(str), 'Other')
        sector_dums = pd.get_dummies(me['sector_top'], prefix='sec', drop_first=True,
                                      dtype=float)

        X_vars = pd.concat([
            me[['log_ulc', 'log_lp', 'log_emp']].reset_index(drop=True),
            size_dums.reset_index(drop=True),
            sector_dums.reset_index(drop=True)
        ], axis=1)
        X = sm.add_constant(X_vars)

        # Part 1: Probit for extensive margin
        y_ext = me['has_auto'].reset_index(drop=True)
        valid_idx = X.notna().all(axis=1) & y_ext.notna()
        X_valid = X[valid_idx]
        y_ext_valid = y_ext[valid_idx]

        if len(y_ext_valid) > 20 and y_ext_valid.nunique() > 1:
            # Drop zero-variance columns
            col_var = X_valid.var()
            zero_var = col_var[col_var == 0].index.tolist()
            if zero_var:
                print(f"    Dropping zero-variance columns: {zero_var}")
                X_valid = X_valid.drop(columns=zero_var)

            try:
                model_e1 = Probit(y_ext_valid, X_valid).fit(disp=0, maxiter=200, method='bfgs')
                results['Model_E1_Probit'] = model_e1
                print(f"\n    Part 1 (Probit: P(automation>0)):")
                print(f"      N = {model_e1.nobs:.0f}, Pseudo R2 = {model_e1.prsquared:.4f}")
                print(f"      N(auto>0) = {y_ext_valid.sum():,}, N(auto=0) = {(y_ext_valid==0).sum():,}")
                for var in ['log_ulc', 'log_lp', 'log_emp']:
                    if var in model_e1.params.index:
                        coef = model_e1.params[var]
                        se = model_e1.bse[var]
                        pval = model_e1.pvalues[var]
                        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                        print(f"      {var:15s}: {coef:8.4f} ({se:.4f}) {sig}")
            except Exception as e:
                print(f"    Probit failed: {e}")

        # Part 2: OLS on intensive margin (firms that invested)
        me_pos = me[me['automation_proxy'] > 0].copy()
        if len(me_pos) > 30:
            me_pos['log_auto'] = np.log(me_pos['automation_proxy'])

            size_dums2 = pd.get_dummies(me_pos['firm_size'], prefix='size', drop_first=True,
                                         dtype=float)
            size_dums2.columns = [str(c) for c in size_dums2.columns]

            me_pos['sector_top'] = np.where(me_pos['sector_2d'].isin(top_sectors),
                                             me_pos['sector_2d'].astype(str), 'Other')
            sector_dums2 = pd.get_dummies(me_pos['sector_top'], prefix='sec', drop_first=True,
                                           dtype=float)

            X_vars2 = pd.concat([
                me_pos[['log_ulc', 'log_lp', 'log_emp']].reset_index(drop=True),
                size_dums2.reset_index(drop=True),
                sector_dums2.reset_index(drop=True)
            ], axis=1)
            X2 = sm.add_constant(X_vars2)
            y2 = me_pos['log_auto'].reset_index(drop=True)

            valid_idx2 = X2.notna().all(axis=1) & y2.notna()
            X2 = X2[valid_idx2]
            y2 = y2[valid_idx2]

            if len(y2) > 20:
                model_e2 = sm.OLS(y2, X2).fit(cov_type='HC3')
                results['Model_E2_OLS'] = model_e2
                print(f"\n    Part 2 (OLS: log(automation) | automation > 0):")
                print(f"      N = {model_e2.nobs:.0f}, R2 = {model_e2.rsquared:.4f}, "
                      f"Adj.R2 = {model_e2.rsquared_adj:.4f}")
                for var in ['log_ulc', 'log_lp', 'log_emp']:
                    if var in model_e2.params.index:
                        coef = model_e2.params[var]
                        se = model_e2.bse[var]
                        pval = model_e2.pvalues[var]
                        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                        print(f"      {var:15s}: {coef:8.4f} ({se:.4f}) {sig}")
    else:
        print("    Insufficient observations for Model E")

    return results


# =============================================================================
# STEP 6: SECTORAL ANALYSIS
# =============================================================================
def step6_sectoral(df_eam_clean, df_merged):
    """Compute sectoral averages and rankings from EAM data."""
    print_section("STEP 6: Sectoral analysis")

    # EAM sectoral averages
    valid = df_eam_clean[
        (df_eam_clean['total_employment'] > 0) &
        (df_eam_clean['VALAGRI'] > 0)
    ].copy()

    sector_stats = valid.groupby('sector_2d').agg(
        n_firms=('nordemp', 'count'),
        total_employment=('total_employment', 'sum'),
        avg_employment=('total_employment', 'mean'),
        avg_labor_cost_pw=('labor_cost_per_worker', 'mean'),
        median_labor_cost_pw=('labor_cost_per_worker', 'median'),
        avg_labor_share=('labor_share_va', 'mean'),
        median_labor_share=('labor_share_va', 'median'),
        avg_labor_productivity=('labor_productivity', 'mean'),
        median_labor_productivity=('labor_productivity', 'median'),
        avg_capital_intensity=('capital_intensity', 'mean'),
        avg_investment_rate=('investment_rate', 'mean'),
        avg_unit_labor_cost=('unit_labor_cost', 'mean'),
        total_value_added=('value_added', 'sum'),
        total_gross_production=('gross_production', 'sum'),
        total_investment=('investment_total', 'sum'),
        avg_export_share=('export_share', 'mean'),
    ).reset_index()

    sector_stats['sector_name'] = sector_stats['sector_2d'].map(CIIU_2D_NAMES).fillna('Otro')

    # Winsorize extreme sector averages
    for v in ['avg_labor_share', 'avg_investment_rate', 'avg_unit_labor_cost']:
        p99 = sector_stats[v].quantile(0.95)
        sector_stats[v] = sector_stats[v].clip(upper=p99)

    # Rank by labor cost burden
    sector_stats = sector_stats.sort_values('avg_labor_cost_pw', ascending=False)

    print(f"\n  Sectoral averages (sorted by avg labor cost per worker):")
    print(f"  {'Sector':25s} {'N':>5s} {'LCost/W':>10s} {'LabShare':>10s} "
          f"{'LabProd':>12s} {'CapInt':>12s} {'InvRate':>8s}")
    print(f"  {'-'*85}")
    for _, row in sector_stats.iterrows():
        print(f"  {row['sector_name']:25s} {row['n_firms']:5.0f} "
              f"{row['avg_labor_cost_pw']:>10,.0f} {row['avg_labor_share']:>10.3f} "
              f"{row['avg_labor_productivity']:>12,.0f} {row['avg_capital_intensity']:>12,.0f} "
              f"{row['avg_investment_rate']:>8.3f}")

    # Merged sample: add innovation stats by sector
    if len(df_merged) > 0:
        merged_valid = df_merged[
            (df_merged['total_employment'] > 0) &
            (df_merged['VALAGRI'] > 0)
        ].copy()

        sector_innov = merged_valid.groupby('sector_2d').agg(
            n_merged=('nordemp', 'count'),
            pct_innovator=('innovator', 'mean'),
            avg_automation_proxy=('automation_proxy', lambda x: x[x>0].mean() if (x>0).any() else 0),
            pct_process_innov=('process_innovator', 'mean'),
        ).reset_index()

        sector_stats = sector_stats.merge(sector_innov, on='sector_2d', how='left')

    return sector_stats


# =============================================================================
# STEP 7: PUBLICATION-QUALITY FIGURES
# =============================================================================
def step7_figures(df_eam_clean, df_merged, sector_stats, reg_results):
    """Generate all publication-quality figures."""
    print_section("STEP 7: Generating publication-quality figures")

    # ---- Figure A: Scatter: labor cost per worker vs investment rate ----
    print("  (a) Scatter: labor cost per worker vs investment rate")
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_df = df_eam_clean[
        (df_eam_clean['labor_cost_per_worker'] > 0) &
        (df_eam_clean['investment_rate'] > 0) &
        (df_eam_clean['investment_rate'] < df_eam_clean['investment_rate'].quantile(0.99)) &
        (df_eam_clean['total_employment'] > 0)
    ].copy()

    # Top 6 sectors by firm count for coloring
    top_secs = plot_df['sector_2d'].value_counts().head(6).index.tolist()
    plot_df['sector_label'] = plot_df.apply(
        lambda r: CIIU_2D_NAMES.get(r['sector_2d'], 'Otro')
        if r['sector_2d'] in top_secs else 'Otros sectores', axis=1
    )

    colors = dict(zip(
        [CIIU_2D_NAMES.get(s, 'Otro') for s in top_secs] + ['Otros sectores'],
        SECTOR_PALETTE[:6] + [LIGHTGRAY]
    ))

    for label in [CIIU_2D_NAMES.get(s, 'Otro') for s in top_secs] + ['Otros sectores']:
        subset = plot_df[plot_df['sector_label'] == label]
        alpha = 0.3 if label == 'Otros sectores' else 0.6
        size = 15 if label == 'Otros sectores' else 30
        ax.scatter(subset['labor_cost_per_worker'], subset['investment_rate'],
                   c=colors[label], alpha=alpha, s=size, label=label, edgecolors='none')

    ax.set_xscale('log')
    ax.set_xlabel('Costo laboral por trabajador (miles COP, log)')
    ax.set_ylabel('Tasa de inversion (Inversion / Valor agregado)')
    ax.set_title('Costo laboral por trabajador vs. tasa de inversion\n(EAM 2023, nivel de firma)',
                 fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(IMG_DIR, f'fig_scatter_lcpw_vs_invrate.{fmt}'), dpi=300)
    plt.close(fig)

    # ---- Figure B: Scatter: unit labor cost vs capital intensity ----
    print("  (b) Scatter: unit labor cost vs capital intensity")
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_df = df_eam_clean[
        (df_eam_clean['unit_labor_cost'] > 0) &
        (df_eam_clean['capital_intensity'] > 0)
    ].copy()
    # Winsorize
    for v in ['unit_labor_cost', 'capital_intensity']:
        p01, p99 = plot_df[v].quantile(0.01), plot_df[v].quantile(0.99)
        plot_df = plot_df[(plot_df[v] >= p01) & (plot_df[v] <= p99)]

    ax.scatter(plot_df['unit_labor_cost'], plot_df['capital_intensity'],
               c=MAINBLUE, alpha=0.25, s=15, edgecolors='none')

    # Add OLS fit line
    log_x = np.log(plot_df['unit_labor_cost'])
    log_y = np.log(plot_df['capital_intensity'])
    valid = np.isfinite(log_x) & np.isfinite(log_y)
    if valid.sum() > 10:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_x[valid], log_y[valid])
        x_line = np.linspace(plot_df['unit_labor_cost'].quantile(0.05),
                             plot_df['unit_labor_cost'].quantile(0.95), 100)
        y_line = np.exp(intercept) * x_line ** slope
        ax.plot(x_line, y_line, color=ACCENTRED, linewidth=2, linestyle='--',
                label=f'OLS: elasticidad = {slope:.2f} (R$^2$={r_value**2:.3f})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Costo laboral unitario (log)')
    ax.set_ylabel('Intensidad de capital (Activos fijos / Trabajador, log)')
    ax.set_title('Costo laboral unitario vs. intensidad de capital\n(EAM 2023)',
                 fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(IMG_DIR, f'fig_scatter_ulc_vs_capint.{fmt}'), dpi=300)
    plt.close(fig)

    # ---- Figure C: Bar chart: avg labor cost per worker by sector ----
    print("  (c) Bar chart: average labor cost per worker by sector")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df = sector_stats[sector_stats['n_firms'] >= 10].sort_values('avg_labor_cost_pw')

    bars = ax.barh(plot_df['sector_name'], plot_df['avg_labor_cost_pw'],
                   color=MAINBLUE, alpha=0.85, edgecolor='white', height=0.7)

    # Add value labels
    for bar, val in zip(bars, plot_df['avg_labor_cost_pw']):
        ax.text(bar.get_width() + plot_df['avg_labor_cost_pw'].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:,.0f}', va='center', fontsize=9, color=DARKGRAY)

    ax.set_xlabel('Costo laboral promedio por trabajador (miles COP)')
    ax.set_title('Costo laboral promedio por trabajador, por sector\n(EAM 2023)',
                 fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(IMG_DIR, f'fig_bar_labor_cost_by_sector.{fmt}'), dpi=300)
    plt.close(fig)

    # ---- Figure D: Bar chart: avg investment rate by sector ----
    print("  (d) Bar chart: average investment rate by sector")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df = sector_stats[sector_stats['n_firms'] >= 10].sort_values('avg_investment_rate')

    bars = ax.barh(plot_df['sector_name'], plot_df['avg_investment_rate'],
                   color=TEAL, alpha=0.85, edgecolor='white', height=0.7)

    for bar, val in zip(bars, plot_df['avg_investment_rate']):
        ax.text(bar.get_width() + plot_df['avg_investment_rate'].max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9, color=DARKGRAY)

    ax.set_xlabel('Tasa de inversion promedio (Inversion / Valor agregado)')
    ax.set_title('Tasa de inversion promedio por sector\n(EAM 2023)',
                 fontweight='bold')

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(IMG_DIR, f'fig_bar_investment_rate_by_sector.{fmt}'), dpi=300)
    plt.close(fig)

    # ---- Figure E: Box plot: labor share of VA by firm size ----
    print("  (e) Box plot: labor share of VA by firm size")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = df_eam_clean[
        (df_eam_clean['labor_share_va'].notna()) &
        (df_eam_clean['labor_share_va'] > 0) &
        (df_eam_clean['labor_share_va'] < df_eam_clean['labor_share_va'].quantile(0.99)) &
        (df_eam_clean['firm_size'] != 'Sin dato')
    ].copy()

    size_order = ['Micro (<10)', 'Pequena (10-50)', 'Mediana (51-200)', 'Grande (>200)']
    box_colors = [MAINBLUE, TEAL, ORANGE, ACCENTRED]

    bp = ax.boxplot(
        [plot_df[plot_df['firm_size'] == s]['labor_share_va'].dropna() for s in size_order],
        tick_labels=size_order, patch_artist=True, widths=0.6,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color=DARKGRAY),
        capprops=dict(color=DARKGRAY),
        flierprops=dict(marker='o', markersize=3, alpha=0.3, color=MEDGRAY)
    )
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel('Participacion laboral en el valor agregado')
    ax.set_title('Participacion laboral en el valor agregado por tamano de firma\n(EAM 2023)',
                 fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.axhline(y=1.0, color=ACCENTRED, linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(4.6, 1.02, 'VA = Costo laboral', color=ACCENTRED, fontsize=8, ha='right')

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(IMG_DIR, f'fig_box_laborshare_by_size.{fmt}'), dpi=300)
    plt.close(fig)

    # ---- Figure F: Coefficient plot from main regressions ----
    print("  (f) Coefficient plot from regression models")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Define models and their key variables to plot
    model_specs = [
        ('Model_A', 'Modelo A: Det. de inversion en\nautomatizacion (log)',
         ['log_ulc', 'log_lp', 'log_emp'],
         ['log(CLU)', 'log(Prod. laboral)', 'log(Empleo)']),
        ('Model_B', 'Modelo B: Tasa de inversion\n(EAM completa)',
         ['labor_share_va', 'log_lp', 'log_emp'],
         ['Part. laboral VA', 'log(Prod. laboral)', 'log(Empleo)']),
        ('Model_C', 'Modelo C: Intensidad de capital\n(log)',
         ['log_lcpw', 'log_emp'],
         ['log(Costo lab/trab)', 'log(Empleo)']),
    ]

    for idx, (model_key, title, vars_list, var_labels) in enumerate(model_specs):
        ax = axes[idx]
        if model_key in reg_results:
            model = reg_results[model_key]
            coefs = []
            ses = []
            pvals = []
            labels = []
            for var, label in zip(vars_list, var_labels):
                if var in model.params.index:
                    coefs.append(model.params[var])
                    ses.append(model.bse[var])
                    pvals.append(model.pvalues[var])
                    labels.append(label)

            if coefs:
                y_pos = range(len(coefs))
                colors_coef = [ACCENTRED if p < 0.05 else MAINBLUE if p < 0.1 else MEDGRAY
                               for p in pvals]

                ax.barh(y_pos, coefs, xerr=[1.96*s for s in ses],
                        color=colors_coef, alpha=0.8, height=0.5,
                        edgecolor='white', capsize=3,
                        error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'color': DARKGRAY})
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)
                ax.axvline(x=0, color=DARKGRAY, linewidth=0.8, linestyle='-')

                # Add significance stars
                for i, (c, p) in enumerate(zip(coefs, pvals)):
                    star = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                    offset = max(abs(c) * 0.1, ses[i] * 2.5)
                    ax.text(c + offset if c > 0 else c - offset,
                            i, star, va='center', ha='left' if c > 0 else 'right',
                            fontsize=10, fontweight='bold', color=DARKGRAY)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Coeficiente')

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(IMG_DIR, f'fig_coefficient_plots.{fmt}'), dpi=300)
    plt.close(fig)

    # ---- Figure G: Sector-level scatter: labor cost burden vs automation ----
    print("  (g) Sector scatter: labor cost vs automation investment")
    if 'avg_automation_proxy' in sector_stats.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_df = sector_stats[
            (sector_stats['n_firms'] >= 10) &
            (sector_stats['avg_automation_proxy'].notna()) &
            (sector_stats['avg_automation_proxy'] > 0)
        ].copy()

        if len(plot_df) > 3:
            # Bubble size proportional to employment
            max_emp = plot_df['total_employment'].max()
            sizes = (plot_df['total_employment'] / max_emp) * 1500 + 100

            scatter = ax.scatter(
                plot_df['avg_labor_cost_pw'],
                plot_df['avg_automation_proxy'],
                s=sizes, c=MAINBLUE, alpha=0.6, edgecolors=DARKGRAY, linewidth=0.5
            )

            # Label each bubble
            for _, row in plot_df.iterrows():
                ax.annotate(
                    row['sector_name'],
                    (row['avg_labor_cost_pw'], row['avg_automation_proxy']),
                    fontsize=8, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points',
                    color=DARKGRAY
                )

            ax.set_xlabel('Costo laboral promedio por trabajador (miles COP)')
            ax.set_ylabel('Inversion promedio en automatizacion (miles COP)')
            ax.set_title('Costo laboral vs. inversion en automatizacion por sector\n'
                         '(Tamano de burbuja = empleo total del sector)',
                         fontweight='bold')
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

            for fmt in ['png', 'pdf']:
                fig.savefig(os.path.join(IMG_DIR, f'fig_sector_bubble_lcost_auto.{fmt}'), dpi=300)
            plt.close(fig)
        else:
            print("    Not enough sectors with automation data for bubble plot")
            plt.close(fig)
    else:
        print("    No automation proxy data available for sector bubble plot")

    # ---- Additional Figure: Innovation rate by firm size ----
    print("  (h) Bar chart: innovation rate by firm size (merged sample)")
    if len(df_merged) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = df_merged[
            (df_merged['firm_size'] != 'Sin dato') &
            (df_merged['firm_size'].notna())
        ].copy()
        # Use only the 4 real size categories
        valid_sizes = ['Micro (<10)', 'Pequena (10-50)', 'Mediana (51-200)', 'Grande (>200)']
        plot_df = plot_df[plot_df['firm_size'].isin(valid_sizes)]
        innov_by_size = plot_df.groupby('firm_size', observed=True)['innovator'].agg(['mean', 'count']).reset_index()
        innov_by_size.columns = ['firm_size', 'innovation_rate', 'n_firms']
        # Sort by size order
        size_order_map = {s: i for i, s in enumerate(valid_sizes)}
        innov_by_size['sort_key'] = innov_by_size['firm_size'].map(size_order_map)
        innov_by_size = innov_by_size.sort_values('sort_key').reset_index(drop=True)

        bars = ax.bar(range(len(innov_by_size)), innov_by_size['innovation_rate'],
                      color=[MAINBLUE, TEAL, ORANGE, ACCENTRED][:len(innov_by_size)],
                      alpha=0.85, edgecolor='white')

        ax.set_xticks(range(len(innov_by_size)))
        ax.set_xticklabels(innov_by_size['firm_size'], rotation=15, ha='right')
        ax.set_ylabel('Tasa de innovacion')
        ax.set_title('Tasa de innovacion por tamano de firma\n(Muestra EAM-EDIT fusionada)',
                     fontweight='bold')

        for bar, rate, n in zip(bars, innov_by_size['innovation_rate'], innov_by_size['n_firms']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{rate:.1%}\n(n={n:,})', ha='center', va='bottom',
                    fontsize=9, color=DARKGRAY)

        ax.set_ylim(0, innov_by_size['innovation_rate'].max() * 1.25)

        for fmt in ['png', 'pdf']:
            fig.savefig(os.path.join(IMG_DIR, f'fig_bar_innovation_by_size.{fmt}'), dpi=300)
        plt.close(fig)

    print("  All figures saved to images/")


# =============================================================================
# STEP 8: OUTPUT TABLES
# =============================================================================
def step8_tables(df_eam_clean, df_merged, sector_stats, reg_results):
    """Save summary tables and regression results."""
    print_section("STEP 8: Saving output tables")

    # ---- Table 1: EAM summary statistics ----
    print("  Saving EAM summary statistics...")
    valid_eam = df_eam_clean[
        (df_eam_clean['total_employment'] > 0) &
        (df_eam_clean['VALAGRI'] > 0)
    ].copy()

    sum_vars = ['total_labor_cost', 'total_employment', 'labor_cost_per_worker',
                'value_added', 'gross_production', 'labor_share_va',
                'labor_productivity', 'capital_intensity', 'investment_rate',
                'unit_labor_cost', 'export_share']

    desc_eam = valid_eam[sum_vars].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]).T
    desc_eam.index.name = 'Variable'
    desc_eam = desc_eam.round(2)
    desc_eam.to_csv(os.path.join(DATA_DIR, 'table_eam_summary_stats.csv'))
    print(f"    Saved: data/table_eam_summary_stats.csv ({len(desc_eam)} variables)")

    # ---- Table 2: Merged sample summary statistics ----
    if len(df_merged) > 0:
        print("  Saving merged sample summary statistics...")
        valid_merged = df_merged[
            (df_merged['total_employment'] > 0) &
            (df_merged['VALAGRI'] > 0)
        ].copy()

        merge_vars = sum_vars + ['automation_proxy', 'machinery_innovation',
                                  'ict_investment', 'software_investment',
                                  'rd_spending', 'total_acti', 'innovator']
        merge_vars = [v for v in merge_vars if v in valid_merged.columns]

        desc_merged = valid_merged[merge_vars].describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]).T
        desc_merged.index.name = 'Variable'
        desc_merged = desc_merged.round(2)
        desc_merged.to_csv(os.path.join(DATA_DIR, 'table_merged_summary_stats.csv'))
        print(f"    Saved: data/table_merged_summary_stats.csv ({len(desc_merged)} variables)")

    # ---- Table 3: Regression results ----
    print("  Saving regression results...")
    reg_rows = []
    for model_name, model in reg_results.items():
        params = model.params
        bse = model.bse
        pvalues = model.pvalues

        for var in params.index:
            sig = '***' if pvalues[var] < 0.01 else '**' if pvalues[var] < 0.05 else '*' if pvalues[var] < 0.1 else ''
            reg_rows.append({
                'Model': model_name,
                'Variable': var,
                'Coefficient': round(params[var], 6),
                'Std_Error': round(bse[var], 6),
                'p_value': round(pvalues[var], 6),
                'Significance': sig,
            })

        # Add model summary row
        if hasattr(model, 'rsquared'):
            reg_rows.append({
                'Model': model_name,
                'Variable': '_R_squared',
                'Coefficient': round(model.rsquared, 4),
                'Std_Error': np.nan,
                'p_value': np.nan,
                'Significance': '',
            })
            reg_rows.append({
                'Model': model_name,
                'Variable': '_Adj_R_squared',
                'Coefficient': round(model.rsquared_adj, 4),
                'Std_Error': np.nan,
                'p_value': np.nan,
                'Significance': '',
            })
        if hasattr(model, 'prsquared'):
            reg_rows.append({
                'Model': model_name,
                'Variable': '_Pseudo_R_squared',
                'Coefficient': round(model.prsquared, 4),
                'Std_Error': np.nan,
                'p_value': np.nan,
                'Significance': '',
            })
        reg_rows.append({
            'Model': model_name,
            'Variable': '_N_obs',
            'Coefficient': model.nobs,
            'Std_Error': np.nan,
            'p_value': np.nan,
            'Significance': '',
        })

    df_reg = pd.DataFrame(reg_rows)
    df_reg.to_csv(os.path.join(DATA_DIR, 'table_regression_results.csv'), index=False)
    print(f"    Saved: data/table_regression_results.csv ({len(df_reg)} rows)")

    # ---- Table 4: Sectoral averages ----
    print("  Saving sectoral averages...")
    sector_stats.to_csv(os.path.join(DATA_DIR, 'table_sectoral_averages.csv'), index=False)
    print(f"    Saved: data/table_sectoral_averages.csv ({len(sector_stats)} sectors)")

    # ---- Table 5: Firm-level merged dataset (for reproducibility) ----
    print("  Saving firm-level analytical dataset...")
    save_cols = ['nordemp', 'ciiu4', 'sector_2d', 'sector_name', 'dpto', 'firm_size',
                 'total_employment', 'total_labor_cost', 'labor_cost_per_worker',
                 'value_added', 'gross_production', 'labor_share_va',
                 'labor_productivity', 'capital_intensity', 'investment_rate',
                 'unit_labor_cost', 'investment_total', 'export_share',
                 'n_establishments']
    if len(df_merged) > 0:
        merge_save_cols = save_cols + [
            'TIPOLO', 'innovator', 'process_innovator',
            'automation_proxy', 'machinery_innovation', 'ict_investment',
            'software_investment', 'rd_spending', 'total_acti'
        ]
        merge_save_cols = [c for c in merge_save_cols if c in df_merged.columns]
        df_merged[merge_save_cols].to_csv(
            os.path.join(DATA_DIR, 'firm_level_merged_dataset.csv'), index=False)
        print(f"    Saved: data/firm_level_merged_dataset.csv ({len(df_merged)} rows)")

    # Also save EAM-only analytical dataset
    eam_save_cols = [c for c in save_cols if c in df_eam_clean.columns]
    df_eam_clean[eam_save_cols].to_csv(
        os.path.join(DATA_DIR, 'firm_level_eam_dataset.csv'), index=False)
    print(f"    Saved: data/firm_level_eam_dataset.csv ({len(df_eam_clean)} rows)")


# =============================================================================
# PRINT COMPREHENSIVE REGRESSION TABLE
# =============================================================================
def print_regression_table(results):
    """Print a formatted regression summary table."""
    print_section("REGRESSION RESULTS SUMMARY")

    # Key variables to show across models
    key_vars_map = {
        'log_ulc': 'log(CLU)',
        'log_lp': 'log(Productividad lab.)',
        'log_emp': 'log(Empleo)',
        'labor_share_va': 'Part. laboral VA',
        'log_lcpw': 'log(Costo lab./trab.)',
        'const': 'Constante',
    }

    # Ordered list of models
    model_order = ['Model_A', 'Model_B', 'Model_C', 'Model_D', 'Model_E1_Probit', 'Model_E2_OLS']
    model_labels = {
        'Model_A': 'A: log(Auto)',
        'Model_B': 'B: Inv.Rate',
        'Model_C': 'C: log(K/L)',
        'Model_D': 'D: Logit(Innov)',
        'Model_E1_Probit': 'E1: Probit(Auto>0)',
        'Model_E2_OLS': 'E2: OLS(log Auto|>0)',
    }

    active_models = [m for m in model_order if m in results]

    if not active_models:
        print("  No regression results to display.")
        return

    # Header
    header = f"  {'Variable':30s}"
    for m in active_models:
        header += f"  {model_labels.get(m, m):>20s}"
    print(header)
    print(f"  {'-'*30}" + f"  {'-'*20}" * len(active_models))

    # Coefficients
    all_vars_shown = set()
    for var, label in key_vars_map.items():
        row_coef = f"  {label:30s}"
        row_se = f"  {'':30s}"
        has_any = False
        for m in active_models:
            model = results[m]
            if var in model.params.index:
                coef = model.params[var]
                se = model.bse[var]
                pval = model.pvalues[var]
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                row_coef += f"  {coef:>17.4f}{sig:>3s}"
                row_se += f"  ({se:>17.4f})"
                has_any = True
                all_vars_shown.add(var)
            else:
                row_coef += f"  {'':>20s}"
                row_se += f"  {'':>20s}"
        if has_any:
            print(row_coef)
            print(row_se)

    # Size dummies
    size_vars = [c for c in (results[active_models[0]].params.index if active_models else [])
                 if c.startswith('size_')]
    if size_vars:
        print(f"  {'Firm size dummies':30s}", end='')
        for m in active_models:
            model = results[m]
            has_size = any(c.startswith('size_') for c in model.params.index)
            print(f"  {'Yes':>20s}" if has_size else f"  {'No':>20s}", end='')
        print()

    # Sector dummies
    sec_vars = [c for c in (results[active_models[0]].params.index if active_models else [])
                if c.startswith('sec_')]
    if sec_vars:
        print(f"  {'Sector dummies':30s}", end='')
        for m in active_models:
            model = results[m]
            has_sec = any(c.startswith('sec_') for c in model.params.index)
            print(f"  {'Yes':>20s}" if has_sec else f"  {'No':>20s}", end='')
        print()

    print(f"  {'-'*30}" + f"  {'-'*20}" * len(active_models))

    # Model fit
    print(f"  {'N':30s}", end='')
    for m in active_models:
        print(f"  {results[m].nobs:>20,.0f}", end='')
    print()

    print(f"  {'R2 / Pseudo R2':30s}", end='')
    for m in active_models:
        model = results[m]
        if hasattr(model, 'rsquared'):
            print(f"  {model.rsquared:>20.4f}", end='')
        elif hasattr(model, 'prsquared'):
            print(f"  {model.prsquared:>20.4f}", end='')
        else:
            print(f"  {'':>20s}", end='')
    print()

    print(f"\n  Notes: Robust standard errors (HC3) for OLS models.")
    print(f"         *p<0.1, **p<0.05, ***p<0.01")
    print(f"         All monetary variables in thousands of COP.")
    print(f"         CLU = Costo laboral unitario (unit labor cost).")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("  FIRM-LEVEL ANALYSIS: LABOR COSTS AND AUTOMATION IN COLOMBIAN MANUFACTURING")
    print("  EAM 2023 + EDIT X 2019-2020")
    print("=" * 80)

    # Step 1: Load data
    df_eam_raw, df_edit_raw = step1_load_data()

    # Step 2: Clean EAM
    df_eam_clean = step2_clean_eam(df_eam_raw)

    # Step 3: Clean EDIT
    df_edit_clean = step3_clean_edit(df_edit_raw)

    # Step 4: Merge
    df_merged = step4_merge(df_eam_clean, df_edit_clean)

    # Step 5: Econometrics
    reg_results = step5_econometrics(df_eam_clean, df_merged)

    # Print regression table
    print_regression_table(reg_results)

    # Step 6: Sectoral analysis
    sector_stats = step6_sectoral(df_eam_clean, df_merged)

    # Step 7: Figures
    step7_figures(df_eam_clean, df_merged, sector_stats, reg_results)

    # Step 8: Tables
    step8_tables(df_eam_clean, df_merged, sector_stats, reg_results)

    print_section("ANALYSIS COMPLETE")
    print(f"  Output files in: {IMG_DIR}")
    print(f"  Data tables in:  {DATA_DIR}")
    print(f"\n  Key findings summary:")
    if 'Model_A' in reg_results:
        m = reg_results['Model_A']
        if 'log_ulc' in m.params.index:
            coef = m.params['log_ulc']
            pval = m.pvalues['log_ulc']
            print(f"    - Unit labor cost -> automation investment: "
                  f"elasticity = {coef:.3f} (p={pval:.4f})")
    if 'Model_C' in reg_results:
        m = reg_results['Model_C']
        if 'log_lcpw' in m.params.index:
            coef = m.params['log_lcpw']
            pval = m.pvalues['log_lcpw']
            print(f"    - Labor cost/worker -> capital intensity: "
                  f"elasticity = {coef:.3f} (p={pval:.4f})")
    if 'Model_D' in reg_results:
        m = reg_results['Model_D']
        if 'log_ulc' in m.params.index:
            coef = m.params['log_ulc']
            pval = m.pvalues['log_ulc']
            print(f"    - Unit labor cost -> P(innovator): "
                  f"log-odds = {coef:.3f} (p={pval:.4f})")


if __name__ == '__main__':
    main()
