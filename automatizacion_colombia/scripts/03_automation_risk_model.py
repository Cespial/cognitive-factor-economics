#!/usr/bin/env python3
"""
03_automation_risk_model.py
============================
Individual-Level Automation Risk Model using GEIH 2024 Microdata

Research question: What worker and job characteristics predict high automation
risk in Colombia, and how does formality status relate to automation exposure?

Methodology:
  - Assigns Frey & Osborne (2017) automation probabilities to Colombian workers
    via CIUO-08 / ISCO-08 occupation classification crosswalk
  - Merges GEIH 2024 modules (Ocupados + Caracteristicas generales + Fuerza de trabajo)
  - Estimates logit, probit, LPM, and OLS models of automation risk
  - Generates publication-quality figures and tables

Data: DANE GEIH 2024, months: January, May, July, October
Classification: CIUO-08 (Colombian adaptation of ISCO-08) via OFICIO_C8

Author: Research project on automation and labor costs in Colombia
Date: 2026-03-06
"""

import os
import sys
import warnings
import zipfile
import gc
import tempfile
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Logit, Probit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.iolib.summary2 import summary_col

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMG_DIR = os.path.join(BASE_DIR, 'images')
GEIH_DIR = os.path.join(DATA_DIR, 'dane',
                        'Gran Encuesta Integrada de Hogares - GEIH - 2024')
GEIH_TEMP = os.path.join(DATA_DIR, 'geih_temp')

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(GEIH_TEMP, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Months to load (avoiding March/April which have nested ZIPs)
MONTHS_TO_LOAD = {
    'Ene_2024.zip': 'Enero',
    'Mayo_2024 1.zip': 'Mayo',
    'Julio_2024.zip': 'Julio',
    'Octubre_2024.zip': 'Octubre',
}

# Color scheme (publication quality, consistent with project)
MAINBLUE = '#1B4F72'
ACCENTRED = '#C0392B'
DARKGRAY = '#2C3E50'
LIGHTGRAY = '#BDC3C7'
MIDBLUE = '#2E86C1'
TEAL = '#1ABC9C'
AMBER = '#F39C12'
PURPLE = '#8E44AD'
GREEN = '#27AE60'
ORANGE = '#E67E22'

# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia', 'serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})


# ==============================================================================
# STEP 1: OCCUPATION-AUTOMATION CROSSWALK
# ==============================================================================
def build_automation_crosswalk():
    """
    Build the CIUO-08 (ISCO-08) to Frey & Osborne automation probability crosswalk.

    Uses Frey & Osborne (2017) probabilities aggregated at ISCO-08 major group
    and 2-digit sub-major group levels. GEIH's OFICIO_C8 follows CIUO-08
    (Colombian adaptation of ISCO-08), so the first 1-2 digits map directly.

    Returns:
        dict: Two dictionaries - 1-digit and 2-digit automation probabilities
    """
    print("=" * 70)
    print("STEP 1: Building occupation-automation crosswalk")
    print("=" * 70)

    # --- 1-digit (Major Group) automation probabilities ---
    # Source: Frey & Osborne (2017), aggregated by ISCO-08 major groups
    auto_1digit = {
        0: 0.10,   # Armed forces occupations
        1: 0.15,   # Managers
        2: 0.12,   # Professionals
        3: 0.25,   # Technicians and associate professionals
        4: 0.70,   # Clerical support workers
        5: 0.45,   # Service and sales workers
        6: 0.60,   # Skilled agricultural, forestry and fishery workers
        7: 0.65,   # Craft and related trades workers
        8: 0.72,   # Plant and machine operators, and assemblers
        9: 0.68,   # Elementary occupations
    }

    # --- 2-digit (Sub-Major Group) automation probabilities ---
    # Finer-grained mapping from Frey & Osborne (2017) literature,
    # cross-referenced with Arntz et al. (2016) and Nedelkoska & Quintini (2018)
    auto_2digit = {
        # Armed forces
        1:  0.08,   # Commissioned armed forces officers
        2:  0.10,   # Non-commissioned armed forces officers
        3:  0.12,   # Armed forces occupations, other ranks

        # Managers
        11: 0.10,   # Chief executives, senior officials and legislators
        12: 0.14,   # Administrative and commercial managers
        13: 0.17,   # Production and specialised services managers
        14: 0.20,   # Hospitality, retail and other services managers

        # Professionals
        21: 0.08,   # Science and engineering professionals
        22: 0.03,   # Health professionals
        23: 0.05,   # Teaching professionals
        24: 0.15,   # Business and administration professionals
        25: 0.12,   # ICT professionals
        26: 0.08,   # Legal, social and cultural professionals

        # Technicians and associate professionals
        31: 0.22,   # Science and engineering associate professionals
        32: 0.15,   # Health associate professionals
        33: 0.30,   # Business and administration associate professionals
        34: 0.20,   # Legal, social, cultural and related associate professionals
        35: 0.28,   # Information and communications technicians

        # Clerical support workers
        41: 0.72,   # General and keyboard clerks
        42: 0.68,   # Customer services clerks
        43: 0.75,   # Numerical and material recording clerks
        44: 0.62,   # Other clerical support workers

        # Service and sales workers
        51: 0.40,   # Personal service workers
        52: 0.48,   # Sales workers
        53: 0.30,   # Personal care workers
        54: 0.55,   # Protective services workers

        # Skilled agricultural workers
        61: 0.58,   # Market-oriented skilled agricultural workers
        62: 0.55,   # Market-oriented skilled forestry, fishery and hunting workers
        63: 0.65,   # Subsistence farmers, fishers, hunters and gatherers

        # Craft and related trades workers
        71: 0.60,   # Building and related trades workers (excl. electricians)
        72: 0.65,   # Metal, machinery and related trades workers
        73: 0.70,   # Handicraft and printing workers
        74: 0.58,   # Electrical and electronic trades workers
        75: 0.72,   # Food processing, wood working, garment and other craft workers

        # Plant and machine operators
        81: 0.75,   # Stationary plant and machine operators
        82: 0.70,   # Assemblers
        83: 0.68,   # Drivers and mobile plant operators

        # Elementary occupations
        91: 0.65,   # Cleaners and helpers
        92: 0.70,   # Agricultural, forestry and fishery labourers
        93: 0.72,   # Labourers in mining, construction, manufacturing and transport
        94: 0.60,   # Food preparation assistants
        95: 0.50,   # Street and related sales and service workers
        96: 0.75,   # Refuse workers and other elementary workers
    }

    print(f"  1-digit crosswalk: {len(auto_1digit)} major groups")
    print(f"  2-digit crosswalk: {len(auto_2digit)} sub-major groups")
    print()
    for k, v in auto_1digit.items():
        risk = "HIGH" if v >= 0.50 else "LOW"
        names = {0: 'Armed forces', 1: 'Managers', 2: 'Professionals',
                 3: 'Technicians', 4: 'Clerical', 5: 'Service/Sales',
                 6: 'Agric. skilled', 7: 'Craft/Trades', 8: 'Operators',
                 9: 'Elementary'}
        print(f"    ISCO {k} ({names[k]:15s}): prob={v:.2f}  [{risk}]")

    return auto_1digit, auto_2digit


# ==============================================================================
# STEP 2: LOAD AND PROCESS GEIH DATA
# ==============================================================================
def load_geih_data(auto_1digit, auto_2digit):
    """
    Load GEIH 2024 data from 4 months, merge modules, and assign automation risk.

    Extracts ZIPs to temporary directories, reads Ocupados + Caracteristicas
    generales + Fuerza de trabajo modules, merges on link keys, and assigns
    Frey & Osborne automation probabilities.

    Returns:
        pd.DataFrame: Merged dataset with automation risk variables
    """
    print("\n" + "=" * 70)
    print("STEP 2: Loading and processing GEIH 2024 data")
    print("=" * 70)

    all_data = []

    for zip_name, month_label in MONTHS_TO_LOAD.items():
        zip_path = os.path.join(GEIH_DIR, zip_name)
        print(f"\n  Processing {month_label} ({zip_name})...")

        if not os.path.exists(zip_path):
            print(f"    WARNING: {zip_path} not found, skipping.")
            continue

        # Extract to a temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f'geih_{month_label}_')

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)

            # Find CSV files - they can be in various subdirectory structures
            csv_dir = None
            for root, dirs, files in os.walk(temp_dir):
                if any(f.endswith('.CSV') or f.endswith('.csv') for f in files):
                    csv_files = [f for f in files if f.upper().endswith('.CSV')]
                    if len(csv_files) >= 6:  # At least 6 of the 8 modules
                        csv_dir = root
                        break

            if csv_dir is None:
                print(f"    WARNING: Could not find CSV directory in {zip_name}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue

            # Find exact filenames (handle encoding variations)
            csv_files = os.listdir(csv_dir)

            ocupados_file = None
            caract_file = None
            fuerza_file = None

            for f in csv_files:
                fl = f.lower()
                if 'ocupados' in fl and 'no ocupados' not in fl and 'otras' not in fl:
                    ocupados_file = f
                elif 'caracter' in fl or 'general' in fl:
                    caract_file = f
                elif 'fuerza' in fl:
                    fuerza_file = f

            if not all([ocupados_file, caract_file, fuerza_file]):
                print(f"    WARNING: Missing modules. Found: ocu={ocupados_file}, "
                      f"car={caract_file}, ft={fuerza_file}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue

            # Load modules with key variables only (memory efficiency)
            ocu_cols = ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN', 'HOGAR', 'MES',
                        'AREA', 'CLASE', 'DPTO', 'FEX_C18',
                        'OFICIO_C8', 'RAMA2D_R4', 'RAMA4D_R4',
                        'P6430',  # Occupational position
                        'P6500',  # Salary
                        'INGLABO',  # Total labor income
                        'P6800',  # Hours worked
                        'P6460', 'P6460S1',  # Firm size
                        'P6920',  # Pension contribution
                        'P6915',  # Health contribution
                        'P6440',  # Business registry
                        'P6450',  # Accounting records
                        'OCI',  # Ocupados flag
                        ]

            car_cols = ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN', 'HOGAR',
                        'P6040',  # Age
                        'P3271',  # Sex
                        'P3042',  # Education level
                        'FEX_C18',
                        ]

            ft_cols = ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN', 'HOGAR',
                       'FT',  # Labor force
                       'PET',  # Working-age population
                       'P6240',  # Main activity
                       ]

            # Read CSVs
            df_ocu = pd.read_csv(
                os.path.join(csv_dir, ocupados_file),
                sep=';', encoding='latin-1', low_memory=False,
                usecols=lambda c: c in ocu_cols
            )

            df_car = pd.read_csv(
                os.path.join(csv_dir, caract_file),
                sep=';', encoding='latin-1', low_memory=False,
                usecols=lambda c: c in car_cols
            )

            df_ft = pd.read_csv(
                os.path.join(csv_dir, fuerza_file),
                sep=';', encoding='latin-1', low_memory=False,
                usecols=lambda c: c in ft_cols
            )

            print(f"    Ocupados: {len(df_ocu):,} rows")
            print(f"    Caracteristicas: {len(df_car):,} rows")
            print(f"    Fuerza trabajo: {len(df_ft):,} rows")

            # --- Merge modules ---
            link_keys = ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN', 'HOGAR']

            # Drop FEX_C18 from car to avoid duplicate
            if 'FEX_C18' in df_car.columns:
                df_car = df_car.drop(columns=['FEX_C18'])

            df_merged = df_ocu.merge(df_car, on=link_keys, how='left')
            df_merged = df_merged.merge(df_ft, on=link_keys, how='left')

            # Add month identifier
            df_merged['month'] = month_label

            print(f"    Merged: {len(df_merged):,} rows, {len(df_merged.columns)} cols")

            all_data.append(df_merged)

        finally:
            # Clean up temporary extraction directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    if not all_data:
        raise RuntimeError("No GEIH data could be loaded. Check ZIP file paths.")

    # Concatenate all months
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n  Total combined dataset: {len(df):,} rows from {len(all_data)} months")

    # --- Assign automation probabilities ---
    print("\n  Assigning automation probabilities from Frey & Osborne crosswalk...")

    # Extract 1-digit and 2-digit occupation codes from OFICIO_C8
    df['ocu_1d'] = df['OFICIO_C8'].apply(
        lambda x: int(str(int(x))[0]) if pd.notna(x) and x > 0 else np.nan
    )
    df['ocu_2d'] = df['OFICIO_C8'].apply(
        lambda x: int(str(int(x))[:2]) if pd.notna(x) and x > 0 else np.nan
    )

    # Assign probability: prefer 2-digit, fall back to 1-digit
    df['automation_prob'] = df['ocu_2d'].map(auto_2digit)
    mask_missing = df['automation_prob'].isna()
    df.loc[mask_missing, 'automation_prob'] = df.loc[mask_missing, 'ocu_1d'].map(auto_1digit)

    n_assigned = df['automation_prob'].notna().sum()
    n_total = len(df)
    print(f"  Automation prob assigned: {n_assigned:,}/{n_total:,} "
          f"({100*n_assigned/n_total:.1f}%)")

    # Distribution of automation probabilities
    print(f"\n  Automation probability distribution:")
    print(f"    Mean:   {df['automation_prob'].mean():.3f}")
    print(f"    Median: {df['automation_prob'].median():.3f}")
    print(f"    Std:    {df['automation_prob'].std():.3f}")
    print(f"    Min:    {df['automation_prob'].min():.3f}")
    print(f"    Max:    {df['automation_prob'].max():.3f}")

    return df


# ==============================================================================
# STEP 3: CONSTRUCT ANALYSIS VARIABLES
# ==============================================================================
def construct_variables(df):
    """
    Construct all analysis variables from raw GEIH data.

    Creates:
      - high_risk: binary (automation_prob >= 0.50)
      - formal: binary (contributes to pension or health)
      - female: binary
      - age, age_sq
      - education_level: categorical
      - log_income
      - hours_worked
      - firm_size: categorical
      - sector: 2-digit CIIU
      - informal: constructed from pension/health contributions

    Returns:
        pd.DataFrame: Analysis-ready dataset
    """
    print("\n" + "=" * 70)
    print("STEP 3: Constructing analysis variables")
    print("=" * 70)

    n_start = len(df)

    # --- Filter to valid observations ---
    # Keep only those with valid automation probability
    df = df[df['automation_prob'].notna()].copy()
    print(f"  After filtering valid automation prob: {len(df):,} (dropped {n_start - len(df):,})")

    # --- Binary outcome: high automation risk ---
    df['high_risk'] = (df['automation_prob'] >= 0.50).astype(int)
    print(f"  High risk (prob >= 0.50): {df['high_risk'].mean():.1%} of workers")

    # --- Formality ---
    # P6920: pension contribution (1=yes, 2=no, 3=already pensioned)
    # P6915: health contribution (1=contributivo cotizante, others=no)
    # Formal = contributes to pension OR is a contributing health affiliate
    df['cotiza_pension'] = (df['P6920'] == 1).astype(int)
    df['cotiza_salud'] = (df['P6915'] == 1).astype(int)
    df['formal'] = ((df['P6920'] == 1) | (df['P6915'] == 1)).astype(int)
    df['informal'] = 1 - df['formal']
    print(f"  Formal workers: {df['formal'].mean():.1%}")

    # --- Demographics ---
    # Sex: P3271 (1=male, 2=female)
    df['female'] = (df['P3271'] == 2).astype(int)
    print(f"  Female: {df['female'].mean():.1%}")

    # Age
    df['age'] = pd.to_numeric(df['P6040'], errors='coerce')
    # Filter working-age population (15-65 for main analysis)
    df = df[(df['age'] >= 15) & (df['age'] <= 65)].copy()
    df['age_sq'] = df['age'] ** 2
    print(f"  After age filter (15-65): {len(df):,}")
    print(f"  Mean age: {df['age'].mean():.1f}")

    # --- Education ---
    # P3042 ISCLED categories in GEIH 2024:
    # 1 = Ninguno / Preescolar
    # 2 = Basica primaria (1-5)
    # 3 = Basica secundaria (6-9)
    # 4 = Media (10-13) / bachillerato
    # 5 = Superior o universitaria
    # 6 = No sabe / No informa -> will be treated as missing
    # Some newer GEIH versions have different coding (up to 13 categories)
    # We'll map to broader categories
    edu_map = {
        1: 'Ninguno',
        2: 'Primaria',
        3: 'Secundaria',
        4: 'Media',
        5: 'Superior',
        6: 'Tecnico/Tecnologico',
        7: 'Tecnico/Tecnologico',
        8: 'Universitario',
        9: 'Universitario',
        10: 'Posgrado',
        11: 'Posgrado',
        12: 'Posgrado',
        13: 'Posgrado',
        99: np.nan,
    }
    df['education_level'] = df['P3042'].map(edu_map)

    # Create ordered category for regression
    edu_order = ['Ninguno', 'Primaria', 'Secundaria', 'Media',
                 'Tecnico/Tecnologico', 'Universitario', 'Posgrado']
    df['education_level'] = pd.Categorical(df['education_level'],
                                            categories=edu_order, ordered=True)

    # Numeric education for continuous measure
    edu_num_map = {
        'Ninguno': 0, 'Primaria': 1, 'Secundaria': 2, 'Media': 3,
        'Tecnico/Tecnologico': 4, 'Universitario': 5, 'Posgrado': 6
    }
    df['education_num'] = df['education_level'].map(edu_num_map).astype(float)

    print(f"  Education distribution:")
    for cat in edu_order:
        n = (df['education_level'] == cat).sum()
        pct = 100 * n / len(df)
        print(f"    {cat:25s}: {n:>7,} ({pct:5.1f}%)")

    # --- Income ---
    df['income'] = pd.to_numeric(df['INGLABO'], errors='coerce')
    df['log_income'] = np.where(df['income'] > 0, np.log(df['income']), np.nan)
    print(f"  Median monthly income: ${df['income'].median():,.0f} COP")
    print(f"  Mean monthly income:   ${df['income'].mean():,.0f} COP")

    # --- Hours worked ---
    df['hours_worked'] = pd.to_numeric(df['P6800'], errors='coerce')
    df['hours_worked'] = df['hours_worked'].clip(1, 120)  # Cap at reasonable bounds
    print(f"  Mean hours/week: {df['hours_worked'].mean():.1f}")

    # --- Firm size ---
    # P6460: 1=solo, 2=con otros; P6460S1: size range code when P6460=2
    # P6460S1 codes (GEIH 2024 coding):
    # 0=solo, 1=2 personas, 2=3 personas, 3=4-5, 4=6-10, 5=11-19,
    # 6=20-50, 7=31-50, 8=51-100, 9=101-200, 10=201-500, 11=501+, 12=NS/NR
    def classify_firm_size(row):
        p6460 = row.get('P6460')
        p6460s1 = row.get('P6460S1')

        if pd.isna(p6460):
            return 'Desconocido'

        if p6460 == 1:
            return 'Micro (1)'

        if p6460 == 2 and pd.notna(p6460s1):
            s = int(p6460s1)
            if s <= 3:       # 1-5 personas
                return 'Micro (2-10)'
            elif s <= 4:     # 6-10
                return 'Micro (2-10)'
            elif s <= 5:     # 11-19
                return 'Pequena (11-50)'
            elif s <= 7:     # 20-50
                return 'Pequena (11-50)'
            elif s <= 8:     # 51-100
                return 'Mediana (51-200)'
            elif s <= 9:     # 101-200
                return 'Mediana (51-200)'
            elif s <= 11:    # 201-500, 501+
                return 'Grande (201+)'
            else:            # 12=NS/NR or higher
                return 'Desconocido'

        return 'Desconocido'

    df['firm_size'] = df.apply(classify_firm_size, axis=1)
    firm_order = ['Micro (1)', 'Micro (2-10)', 'Pequena (11-50)',
                  'Mediana (51-200)', 'Grande (201+)', 'Desconocido']
    df['firm_size'] = pd.Categorical(df['firm_size'], categories=firm_order, ordered=True)

    print(f"  Firm size distribution:")
    for cat in firm_order:
        n = (df['firm_size'] == cat).sum()
        pct = 100 * n / len(df)
        print(f"    {cat:25s}: {n:>7,} ({pct:5.1f}%)")

    # --- Sector ---
    # RAMA2D_R4: CIIU Rev.4 2-digit sector code
    sector_map = {
        1: 'Agricultura', 2: 'Agricultura', 3: 'Agricultura',
        5: 'Mineria', 6: 'Mineria', 7: 'Mineria', 8: 'Mineria', 9: 'Mineria',
        10: 'Manufactura', 11: 'Manufactura', 12: 'Manufactura',
        13: 'Manufactura', 14: 'Manufactura', 15: 'Manufactura',
        16: 'Manufactura', 17: 'Manufactura', 18: 'Manufactura',
        19: 'Manufactura', 20: 'Manufactura', 21: 'Manufactura',
        22: 'Manufactura', 23: 'Manufactura', 24: 'Manufactura',
        25: 'Manufactura', 26: 'Manufactura', 27: 'Manufactura',
        28: 'Manufactura', 29: 'Manufactura', 30: 'Manufactura',
        31: 'Manufactura', 32: 'Manufactura', 33: 'Manufactura',
        35: 'Serv. publicos', 36: 'Serv. publicos',
        37: 'Serv. publicos', 38: 'Serv. publicos', 39: 'Serv. publicos',
        41: 'Construccion', 42: 'Construccion', 43: 'Construccion',
        45: 'Comercio', 46: 'Comercio', 47: 'Comercio',
        49: 'Transporte', 50: 'Transporte', 51: 'Transporte',
        52: 'Transporte', 53: 'Transporte',
        55: 'Alojamiento/Comida', 56: 'Alojamiento/Comida',
        58: 'Info/Comunicaciones', 59: 'Info/Comunicaciones',
        60: 'Info/Comunicaciones', 61: 'Info/Comunicaciones',
        62: 'Info/Comunicaciones', 63: 'Info/Comunicaciones',
        64: 'Financiero', 65: 'Financiero', 66: 'Financiero',
        68: 'Inmobiliario',
        69: 'Serv. profesionales', 70: 'Serv. profesionales',
        71: 'Serv. profesionales', 72: 'Serv. profesionales',
        73: 'Serv. profesionales', 74: 'Serv. profesionales',
        75: 'Serv. profesionales',
        77: 'Adm. y apoyo', 78: 'Adm. y apoyo', 79: 'Adm. y apoyo',
        80: 'Adm. y apoyo', 81: 'Adm. y apoyo', 82: 'Adm. y apoyo',
        84: 'Admin. publica',
        85: 'Educacion',
        86: 'Salud', 87: 'Salud', 88: 'Salud',
        90: 'Arte/Entretenimiento', 91: 'Arte/Entretenimiento',
        92: 'Arte/Entretenimiento', 93: 'Arte/Entretenimiento',
        94: 'Otros servicios', 95: 'Otros servicios', 96: 'Otros servicios',
        97: 'Hogares empleadores', 98: 'Hogares empleadores',
        99: 'Org. extraterritoriales',
        0: 'No especificado',
    }

    df['sector'] = df['RAMA2D_R4'].map(sector_map).fillna('Otro')

    print(f"\n  Sector distribution:")
    for sec in df['sector'].value_counts().head(15).index:
        n = (df['sector'] == sec).sum()
        pct = 100 * n / len(df)
        print(f"    {sec:25s}: {n:>7,} ({pct:5.1f}%)")

    # --- Occupational position ---
    pos_map = {
        1: 'Empleado empresa', 2: 'Empleado gobierno',
        3: 'Empleado domestico', 4: 'Cuenta propia',
        5: 'Patron/empleador', 6: 'Trab. familiar sin pago',
        7: 'Trab. sin remuneracion', 8: 'Jornalero/peon'
    }
    df['position'] = df['P6430'].map(pos_map).fillna('Otro')

    # --- Area (urban/rural) ---
    df['urban'] = (df['CLASE'] == 1).astype(int)

    # --- Survey weight ---
    df['weight'] = pd.to_numeric(df['FEX_C18'], errors='coerce')
    df['weight'] = df['weight'].fillna(df['weight'].median())

    # --- Occupation group label ---
    ocu_label_map = {
        0: 'Fuerzas armadas', 1: 'Directivos', 2: 'Profesionales',
        3: 'Tecnicos', 4: 'Administrativos', 5: 'Servicios/Ventas',
        6: 'Agric. calificados', 7: 'Oficios/Artesanos',
        8: 'Operadores', 9: 'Elementales'
    }
    df['ocu_group'] = df['ocu_1d'].map(ocu_label_map)

    # --- Department name ---
    dpto_map = {
        5: 'Antioquia', 8: 'Atlantico', 11: 'Bogota', 13: 'Bolivar',
        15: 'Boyaca', 17: 'Caldas', 18: 'Caqueta', 19: 'Cauca',
        20: 'Cesar', 23: 'Cordoba', 25: 'Cundinamarca', 27: 'Choco',
        41: 'Huila', 44: 'La Guajira', 47: 'Magdalena', 50: 'Meta',
        52: 'Narino', 54: 'N. Santander', 63: 'Quindio', 66: 'Risaralda',
        68: 'Santander', 70: 'Sucre', 73: 'Tolima', 76: 'Valle del Cauca',
        81: 'Arauca', 85: 'Casanare', 86: 'Putumayo', 88: 'San Andres',
        91: 'Amazonas', 94: 'Guainia', 95: 'Guaviare', 97: 'Vaupes',
        99: 'Vichada'
    }
    df['departamento'] = df['DPTO'].map(dpto_map).fillna('Otro')

    # --- Final summary ---
    n_final = len(df)
    print(f"\n  Final analysis dataset: {n_final:,} observations")
    print(f"  Variables created: {len(df.columns)}")

    return df


# ==============================================================================
# STEP 4: ECONOMETRIC MODELS
# ==============================================================================
def run_econometric_models(df):
    """
    Estimate automation risk models:
      a) Logistic regression (main model)
      b) Probit (robustness)
      c) Linear Probability Model (robustness)
      d) OLS on continuous automation_prob
      e) Logit with formal x sector interaction

    Returns:
        dict: Model results
    """
    print("\n" + "=" * 70)
    print("STEP 4: Econometric models")
    print("=" * 70)

    results = {}

    # --- Prepare regression data ---
    # Create dummy variables for categorical variables
    reg_df = df.copy()

    # Drop observations with missing key variables
    key_vars = ['high_risk', 'automation_prob', 'formal', 'female', 'age',
                'log_income', 'hours_worked', 'education_level', 'sector']
    reg_df = reg_df.dropna(subset=key_vars).copy()

    # Drop unknown firm sizes for regression
    reg_df = reg_df[reg_df['firm_size'] != 'Desconocido'].copy()

    print(f"  Regression sample: {len(reg_df):,} observations")
    print(f"  High risk share in sample: {reg_df['high_risk'].mean():.1%}")

    # Create dummies for education (reference: Secundaria)
    edu_dummies = pd.get_dummies(reg_df['education_level'], prefix='edu',
                                  drop_first=False, dtype=float)
    # Drop reference category
    ref_edu = 'edu_Secundaria'
    if ref_edu in edu_dummies.columns:
        edu_dummies = edu_dummies.drop(columns=[ref_edu])

    # Create dummies for firm size (reference: Micro (2-10))
    firm_dummies = pd.get_dummies(reg_df['firm_size'], prefix='firm',
                                   drop_first=False, dtype=float)
    ref_firm = 'firm_Micro (2-10)'
    if ref_firm in firm_dummies.columns:
        firm_dummies = firm_dummies.drop(columns=[ref_firm])

    # Sector dummies (reference: Comercio, largest sector)
    sector_dummies = pd.get_dummies(reg_df['sector'], prefix='sec',
                                     drop_first=False, dtype=float)
    ref_sec = 'sec_Comercio'
    if ref_sec in sector_dummies.columns:
        sector_dummies = sector_dummies.drop(columns=[ref_sec])

    # Build X matrix
    X_vars = pd.concat([
        reg_df[['formal', 'female', 'age', 'age_sq', 'log_income', 'hours_worked']].reset_index(drop=True),
        edu_dummies.reset_index(drop=True),
        firm_dummies.reset_index(drop=True),
        sector_dummies.reset_index(drop=True),
    ], axis=1)

    # Clean column names (remove special characters for statsmodels)
    X_vars.columns = [c.replace(' ', '_').replace('(', '').replace(')', '')
                      .replace('/', '_').replace('.', '_').replace('+', 'plus')
                      for c in X_vars.columns]

    # Remove any columns with zero variance (perfect collinearity source)
    zero_var_cols = X_vars.columns[X_vars.std() == 0].tolist()
    if zero_var_cols:
        print(f"  Dropping zero-variance columns: {zero_var_cols}")
        X_vars = X_vars.drop(columns=zero_var_cols)

    # Drop any duplicate columns
    X_vars = X_vars.loc[:, ~X_vars.columns.duplicated()]

    # Check for and drop perfectly collinear columns
    # Use correlation matrix to find columns with correlation > 0.999
    corr_matrix = X_vars.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []
    for col in upper_tri.columns:
        high_corr = upper_tri[col][upper_tri[col] > 0.999]
        if len(high_corr) > 0:
            to_drop.append(col)
    if to_drop:
        print(f"  Dropping near-perfectly correlated columns: {to_drop}")
        X_vars = X_vars.drop(columns=to_drop, errors='ignore')

    X_vars = sm.add_constant(X_vars, has_constant='skip')
    y_binary = reg_df['high_risk'].values
    y_cont = reg_df['automation_prob'].values
    weights = reg_df['weight'].values

    # Normalize weights for WLS (mean = 1)
    weights_norm = weights / weights.mean()

    print(f"  Final X matrix: {X_vars.shape[1]} variables (incl. constant)")
    print(f"  Variables: {list(X_vars.columns[:10])}... ({X_vars.shape[1]} total)")

    # Check for multicollinearity (core numeric vars only)
    print("\n  Checking for multicollinearity...")
    numeric_vars = ['formal', 'female', 'age', 'age_sq', 'log_income', 'hours_worked']
    numeric_vars = [v for v in numeric_vars if v in X_vars.columns]
    for v in numeric_vars:
        vif_data = X_vars[numeric_vars].copy()
        idx = list(vif_data.columns).index(v)
        vif_val = variance_inflation_factor(vif_data.values, idx)
        if vif_val > 10:
            print(f"    WARNING: High VIF for {v}: {vif_val:.1f}")
    print("    VIF check complete (age/age_sq naturally collinear, acceptable)")

    # =========================================================================
    # Model (a): Logistic Regression (MAIN MODEL)
    # =========================================================================
    print("\n  --- Model (a): Logistic Regression ---")
    try:
        logit_model = Logit(y_binary, X_vars)
        logit_result = logit_model.fit(disp=0, maxiter=200,
                                        method='bfgs',
                                        cov_type='HC1')  # robust SE
        results['logit'] = logit_result

        print(f"    Pseudo R-squared: {logit_result.prsquared:.4f}")
        print(f"    Log-Likelihood: {logit_result.llf:.1f}")
        print(f"    AIC: {logit_result.aic:.1f}")
        print(f"    BIC: {logit_result.bic:.1f}")

        # Odds ratios
        odds_ratios = np.exp(logit_result.params)
        ci = logit_result.conf_int()
        or_ci_low = np.exp(ci[0])
        or_ci_high = np.exp(ci[1])

        or_df = pd.DataFrame({
            'Variable': X_vars.columns,
            'Coef': logit_result.params,
            'OR': odds_ratios,
            'OR_CI_low': or_ci_low,
            'OR_CI_high': or_ci_high,
            'p_value': logit_result.pvalues,
            'SE': logit_result.bse,
        })
        results['odds_ratios'] = or_df

        print(f"\n    Key Odds Ratios:")
        for _, row in or_df.iterrows():
            if row['Variable'] in ['formal', 'female', 'age', 'log_income', 'hours_worked']:
                sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05
                       else ('*' if row['p_value'] < 0.1 else ''))
                print(f"      {row['Variable']:20s}: OR={row['OR']:.4f} "
                      f"[{row['OR_CI_low']:.4f}, {row['OR_CI_high']:.4f}] {sig}")

        # Classification accuracy
        y_pred_prob = logit_result.predict(X_vars)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        acc = accuracy_score(y_binary, y_pred)
        auc = roc_auc_score(y_binary, y_pred_prob)
        print(f"\n    Classification accuracy: {acc:.1%}")
        print(f"    ROC-AUC: {auc:.4f}")

        results['y_pred_prob_logit'] = y_pred_prob
        results['auc_logit'] = auc
        results['accuracy_logit'] = acc

        # Average Marginal Effects
        print("\n    Computing average marginal effects...")
        mfx = logit_result.get_margeff(at='overall', method='dydx')
        results['marginal_effects'] = mfx

        mfx_df = pd.DataFrame({
            'Variable': X_vars.columns[1:],  # exclude constant
            'AME': mfx.margeff,
            'SE': mfx.margeff_se,
            'p_value': mfx.pvalues,
        })
        results['mfx_df'] = mfx_df

        print(f"    Key Average Marginal Effects:")
        for _, row in mfx_df.iterrows():
            if row['Variable'] in ['formal', 'female', 'age', 'log_income', 'hours_worked']:
                sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05
                       else ('*' if row['p_value'] < 0.1 else ''))
                print(f"      {row['Variable']:20s}: AME={row['AME']:.4f} "
                      f"(SE={row['SE']:.4f}) {sig}")

    except Exception as e:
        print(f"    ERROR in logit: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Model (b): Probit (Robustness)
    # =========================================================================
    print("\n  --- Model (b): Probit Regression ---")
    try:
        probit_model = Probit(y_binary, X_vars)
        probit_result = probit_model.fit(disp=0, maxiter=200, method='bfgs',
                                          cov_type='HC1')
        results['probit'] = probit_result

        print(f"    Pseudo R-squared: {probit_result.prsquared:.4f}")
        print(f"    Log-Likelihood: {probit_result.llf:.1f}")

        # Marginal effects
        mfx_probit = probit_result.get_margeff(at='overall', method='dydx')
        results['mfx_probit'] = mfx_probit

    except Exception as e:
        print(f"    ERROR in probit: {e}")

    # =========================================================================
    # Model (c): Linear Probability Model (OLS on binary outcome)
    # =========================================================================
    print("\n  --- Model (c): Linear Probability Model ---")
    try:
        lpm_model = sm.OLS(y_binary, X_vars)
        lpm_result = lpm_model.fit(cov_type='HC1')
        results['lpm'] = lpm_result

        print(f"    R-squared: {lpm_result.rsquared:.4f}")
        print(f"    Adj R-squared: {lpm_result.rsquared_adj:.4f}")

        # Key coefficients
        for v in ['formal', 'female', 'age', 'log_income']:
            if v in lpm_result.params.index:
                coef = lpm_result.params[v]
                se = lpm_result.bse[v]
                pv = lpm_result.pvalues[v]
                sig = '***' if pv < 0.01 else ('**' if pv < 0.05 else ('*' if pv < 0.1 else ''))
                print(f"      {v:20s}: {coef:.4f} (SE={se:.4f}) {sig}")

    except Exception as e:
        print(f"    ERROR in LPM: {e}")

    # =========================================================================
    # Model (d): OLS on continuous automation_prob
    # =========================================================================
    print("\n  --- Model (d): OLS on Continuous Automation Probability ---")
    try:
        ols_model = sm.OLS(y_cont, X_vars)
        ols_result = ols_model.fit(cov_type='HC1')
        results['ols_cont'] = ols_result

        print(f"    R-squared: {ols_result.rsquared:.4f}")
        print(f"    Adj R-squared: {ols_result.rsquared_adj:.4f}")

        for v in ['formal', 'female', 'age', 'log_income']:
            if v in ols_result.params.index:
                coef = ols_result.params[v]
                se = ols_result.bse[v]
                pv = ols_result.pvalues[v]
                sig = '***' if pv < 0.01 else ('**' if pv < 0.05 else ('*' if pv < 0.1 else ''))
                print(f"      {v:20s}: {coef:.4f} (SE={se:.4f}) {sig}")

    except Exception as e:
        print(f"    ERROR in OLS: {e}")

    # =========================================================================
    # Model (e): Logit with Formal x Sector Interaction
    # =========================================================================
    print("\n  --- Model (e): Logit with Formal x Sector Interaction ---")
    try:
        # Create interaction terms: formal * each sector dummy
        interaction_cols = []
        for col in sector_dummies.columns:
            clean_col = col.replace(' ', '_').replace('(', '').replace(')', '')\
                         .replace('/', '_').replace('.', '_').replace('+', 'plus')
            int_name = f'formal_x_{clean_col}'
            interaction_cols.append(int_name)

        # Rebuild X with interactions
        X_interact = X_vars.copy()
        sec_dummy_clean = sector_dummies.copy()
        sec_dummy_clean.columns = [c.replace(' ', '_').replace('(', '').replace(')', '')
                                    .replace('/', '_').replace('.', '_').replace('+', 'plus')
                                    for c in sec_dummy_clean.columns]
        sec_dummy_clean = sec_dummy_clean.reset_index(drop=True)

        for col, int_name in zip(sec_dummy_clean.columns, interaction_cols):
            X_interact[int_name] = X_interact['formal'] * sec_dummy_clean[col].values

        # Drop zero-variance and near-collinear interaction columns
        zero_var_int = X_interact.columns[X_interact.std() == 0].tolist()
        if zero_var_int:
            print(f"    Dropping zero-variance interaction cols: {len(zero_var_int)}")
            X_interact = X_interact.drop(columns=zero_var_int)
            interaction_cols = [c for c in interaction_cols if c in X_interact.columns]

        X_interact = X_interact.loc[:, ~X_interact.columns.duplicated()]

        logit_interact = Logit(y_binary, X_interact)
        logit_int_result = logit_interact.fit(disp=0, maxiter=200, cov_type='HC1',
                                               method='bfgs')
        results['logit_interact'] = logit_int_result

        print(f"    Pseudo R-squared: {logit_int_result.prsquared:.4f}")
        print(f"    AIC: {logit_int_result.aic:.1f}")

        # Test joint significance of interactions
        # Show significant interaction terms
        int_params = logit_int_result.params[interaction_cols]
        int_pvals = logit_int_result.pvalues[interaction_cols]
        sig_ints = int_pvals[int_pvals < 0.10]
        if len(sig_ints) > 0:
            print(f"    Significant interactions (p<0.10):")
            for name in sig_ints.index:
                coef = logit_int_result.params[name]
                pv = logit_int_result.pvalues[name]
                sig = '***' if pv < 0.01 else ('**' if pv < 0.05 else '*')
                print(f"      {name:45s}: {coef:.4f} (p={pv:.4f}) {sig}")
        else:
            print("    No individually significant interaction terms (p<0.10)")

    except Exception as e:
        print(f"    ERROR in interaction model: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Model comparison table
    # =========================================================================
    print("\n  --- Model Comparison ---")
    try:
        model_list = []
        model_names = []

        if 'logit' in results:
            model_list.append(results['logit'])
            model_names.append('Logit')
        if 'probit' in results:
            model_list.append(results['probit'])
            model_names.append('Probit')
        if 'lpm' in results:
            model_list.append(results['lpm'])
            model_names.append('LPM')
        if 'ols_cont' in results:
            model_list.append(results['ols_cont'])
            model_names.append('OLS (cont.)')

        if model_list:
            comparison = summary_col(
                model_list,
                model_names=model_names,
                stars=True,
                float_format='%.4f',
                info_dict={
                    'N': lambda x: f"{int(x.nobs):,}",
                    'R-sq / Pseudo R-sq': lambda x: f"{getattr(x, 'rsquared', getattr(x, 'prsquared', 0)):.4f}",
                }
            )
            results['comparison_table'] = comparison
            print(comparison)
    except Exception as e:
        print(f"    WARNING: Could not create comparison table: {e}")

    # Store regression data for plots
    results['X_vars'] = X_vars
    results['y_binary'] = y_binary
    results['y_cont'] = y_cont
    results['reg_df'] = reg_df

    return results


# ==============================================================================
# STEP 5: PUBLICATION-QUALITY FIGURES
# ==============================================================================
def generate_figures(df, results):
    """
    Generate all publication-quality figures.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Generating publication-quality figures")
    print("=" * 70)

    # =========================================================================
    # Figure (a): Distribution of automation risk by formality
    # =========================================================================
    print("\n  (a) Automation risk density by formality status...")
    fig, ax = plt.subplots(figsize=(8, 5))

    formal_probs = df[df['formal'] == 1]['automation_prob'].dropna()
    informal_probs = df[df['formal'] == 0]['automation_prob'].dropna()

    # KDE plots
    from scipy.stats import gaussian_kde

    x_range = np.linspace(0, 1, 200)

    if len(formal_probs) > 10:
        kde_formal = gaussian_kde(formal_probs, bw_method=0.08)
        ax.fill_between(x_range, kde_formal(x_range), alpha=0.35, color=MAINBLUE,
                        label=f'Formal (n={len(formal_probs):,})')
        ax.plot(x_range, kde_formal(x_range), color=MAINBLUE, linewidth=2)

    if len(informal_probs) > 10:
        kde_informal = gaussian_kde(informal_probs, bw_method=0.08)
        ax.fill_between(x_range, kde_informal(x_range), alpha=0.35, color=ACCENTRED,
                        label=f'Informal (n={len(informal_probs):,})')
        ax.plot(x_range, kde_informal(x_range), color=ACCENTRED, linewidth=2)

    ax.axvline(x=0.50, color=DARKGRAY, linestyle='--', linewidth=1, alpha=0.7,
               label='Umbral alto riesgo (0.50)')

    ax.set_xlabel('Probabilidad de automatizacion (Frey & Osborne)')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribucion del riesgo de automatizacion por estatus de formalidad')
    ax.legend(frameon=True, fancybox=False, edgecolor=LIGHTGRAY)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'fig_automation_risk_density.png'), dpi=300)
    fig.savefig(os.path.join(IMG_DIR, 'fig_automation_risk_density.pdf'))
    plt.close(fig)
    print("    Saved: fig_automation_risk_density.png/pdf")

    # =========================================================================
    # Figure (b): Automation risk by sector (box plot)
    # =========================================================================
    print("  (b) Automation risk by sector (box plot)...")

    # Get sectors ordered by median automation risk
    sector_medians = df.groupby('sector')['automation_prob'].median().sort_values(ascending=True)
    sector_order = sector_medians.index.tolist()

    # Filter to sectors with enough observations
    sector_counts = df['sector'].value_counts()
    valid_sectors = sector_counts[sector_counts >= 50].index
    sector_order = [s for s in sector_order if s in valid_sectors]

    fig, ax = plt.subplots(figsize=(10, 7))

    plot_data = df[df['sector'].isin(sector_order)].copy()

    # Color by risk level
    palette = {}
    for sec in sector_order:
        med = sector_medians[sec]
        if med >= 0.60:
            palette[sec] = ACCENTRED
        elif med >= 0.45:
            palette[sec] = AMBER
        else:
            palette[sec] = MAINBLUE

    bp = ax.boxplot(
        [plot_data[plot_data['sector'] == s]['automation_prob'].dropna().values
         for s in sector_order],
        vert=False, patch_artist=True, widths=0.6,
        medianprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='.', markersize=2, alpha=0.3),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
    )

    for patch, sec in zip(bp['boxes'], sector_order):
        patch.set_facecolor(palette[sec])
        patch.set_alpha(0.7)
        patch.set_edgecolor(DARKGRAY)

    ax.set_yticklabels(sector_order)
    ax.axvline(x=0.50, color=DARKGRAY, linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Probabilidad de automatizacion')
    ax.set_title('Riesgo de automatizacion por sector economico')
    ax.text(0.51, len(sector_order) + 0.3, 'Alto riesgo', fontsize=8,
            color=DARKGRAY, style='italic')

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'fig_automation_risk_by_sector.png'), dpi=300)
    fig.savefig(os.path.join(IMG_DIR, 'fig_automation_risk_by_sector.pdf'))
    plt.close(fig)
    print("    Saved: fig_automation_risk_by_sector.png/pdf")

    # =========================================================================
    # Figure (c): Automation risk by education level
    # =========================================================================
    print("  (c) Automation risk by education level...")
    fig, ax = plt.subplots(figsize=(8, 5))

    edu_order = ['Ninguno', 'Primaria', 'Secundaria', 'Media',
                 'Tecnico/Tecnologico', 'Universitario', 'Posgrado']

    means = []
    ci_low = []
    ci_high = []
    labels = []
    counts = []

    for edu in edu_order:
        subset = df[df['education_level'] == edu]['automation_prob'].dropna()
        if len(subset) >= 30:
            m = subset.mean()
            se = subset.std() / np.sqrt(len(subset))
            means.append(m)
            ci_low.append(m - 1.96 * se)
            ci_high.append(m + 1.96 * se)
            labels.append(edu)
            counts.append(len(subset))

    x_pos = np.arange(len(labels))
    colors = [ACCENTRED if m >= 0.50 else MAINBLUE for m in means]

    bars = ax.bar(x_pos, means, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.errorbar(x_pos, means,
                yerr=[np.array(means) - np.array(ci_low),
                      np.array(ci_high) - np.array(means)],
                fmt='none', color=DARKGRAY, capsize=3, linewidth=1.5)

    ax.axhline(y=0.50, color=DARKGRAY, linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Probabilidad promedio de automatizacion')
    ax.set_title('Riesgo de automatizacion por nivel educativo')
    ax.set_ylim(0, max(means) * 1.15)

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'n={count:,}', ha='center', va='bottom', fontsize=7, color=DARKGRAY)

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'fig_automation_risk_by_education.png'), dpi=300)
    fig.savefig(os.path.join(IMG_DIR, 'fig_automation_risk_by_education.pdf'))
    plt.close(fig)
    print("    Saved: fig_automation_risk_by_education.png/pdf")

    # =========================================================================
    # Figure (d): ROC curve
    # =========================================================================
    print("  (d) ROC curve for logit model...")
    if 'logit' in results and 'y_pred_prob_logit' in results:
        fig, ax = plt.subplots(figsize=(6, 6))

        y_true = results['y_binary']
        y_prob = results['y_pred_prob_logit']

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_val = results['auc_logit']

        ax.plot(fpr, tpr, color=MAINBLUE, linewidth=2,
                label=f'Logit (AUC = {auc_val:.3f})')
        ax.plot([0, 1], [0, 1], color=LIGHTGRAY, linestyle='--', linewidth=1,
                label='Clasificacion aleatoria')

        # Find optimal threshold (Youden's J)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_thresh = thresholds[optimal_idx]
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color=ACCENTRED,
                   s=80, zorder=5, label=f'Umbral optimo ({optimal_thresh:.2f})')

        ax.set_xlabel('Tasa de falsos positivos (1 - Especificidad)')
        ax.set_ylabel('Tasa de verdaderos positivos (Sensibilidad)')
        ax.set_title('Curva ROC - Modelo Logit de riesgo de automatizacion')
        ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor=LIGHTGRAY)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')

        plt.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, 'fig_roc_curve.png'), dpi=300)
        fig.savefig(os.path.join(IMG_DIR, 'fig_roc_curve.pdf'))
        plt.close(fig)
        print("    Saved: fig_roc_curve.png/pdf")

    # =========================================================================
    # Figure (e): Forest plot of odds ratios
    # =========================================================================
    print("  (e) Forest plot of odds ratios...")
    if 'odds_ratios' in results:
        or_df = results['odds_ratios'].copy()

        # Select key variables (not constant, not all sector/firm dummies)
        key_vars = ['formal', 'female', 'age', 'log_income', 'hours_worked']
        edu_vars = [v for v in or_df['Variable'] if v.startswith('edu_')]
        firm_vars = [v for v in or_df['Variable'] if v.startswith('firm_')]

        # Take top significant sector dummies
        sec_vars = [v for v in or_df['Variable'] if v.startswith('sec_')]
        sec_df = or_df[or_df['Variable'].isin(sec_vars)].copy()
        sec_df = sec_df.sort_values('p_value').head(8)['Variable'].tolist()

        plot_vars = key_vars + edu_vars + firm_vars[:4] + sec_df
        plot_df = or_df[or_df['Variable'].isin(plot_vars)].copy()

        # Rename for display
        rename_map = {
            'formal': 'Formal (vs. informal)',
            'female': 'Mujer (vs. hombre)',
            'age': 'Edad (anos)',
            'log_income': 'Log ingreso laboral',
            'hours_worked': 'Horas trabajadas/semana',
        }
        for old, new in rename_map.items():
            plot_df.loc[plot_df['Variable'] == old, 'Variable'] = new

        # Clean up remaining names
        plot_df['Variable'] = (plot_df['Variable']
                               .str.replace('edu_', 'Educ: ', regex=False)
                               .str.replace('firm_', 'Tamano: ', regex=False)
                               .str.replace('sec_', 'Sector: ', regex=False)
                               .str.replace('_', ' ', regex=False))

        fig, ax = plt.subplots(figsize=(9, max(7, len(plot_df) * 0.35)))

        y_pos = np.arange(len(plot_df))

        # Color by significance
        colors = [ACCENTRED if p < 0.05 else (AMBER if p < 0.10 else LIGHTGRAY)
                  for p in plot_df['p_value']]

        ax.hlines(y_pos, plot_df['OR_CI_low'], plot_df['OR_CI_high'],
                  colors=colors, linewidth=2, alpha=0.8)
        ax.scatter(plot_df['OR'], y_pos, color=colors, s=50, zorder=5,
                   edgecolors='white', linewidths=0.5)

        ax.axvline(x=1, color=DARKGRAY, linestyle='-', linewidth=0.8, alpha=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['Variable'].values)
        ax.set_xlabel('Odds Ratio (IC 95%)')
        ax.set_title('Odds Ratios del modelo logit: determinantes del alto riesgo de automatizacion')

        # Add significance legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=ACCENTRED,
                   markersize=8, label='p < 0.05'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=AMBER,
                   markersize=8, label='p < 0.10'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=LIGHTGRAY,
                   markersize=8, label='No significativo'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True,
                  fancybox=False, edgecolor=LIGHTGRAY)

        plt.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, 'fig_forest_plot_odds_ratios.png'), dpi=300)
        fig.savefig(os.path.join(IMG_DIR, 'fig_forest_plot_odds_ratios.pdf'))
        plt.close(fig)
        print("    Saved: fig_forest_plot_odds_ratios.png/pdf")

    # =========================================================================
    # Figure (f): Marginal effects plot
    # =========================================================================
    print("  (f) Marginal effects plot...")
    if 'mfx_df' in results:
        mfx_df = results['mfx_df'].copy()

        # Select key variables
        key_mfx_vars = ['formal', 'female', 'age', 'log_income', 'hours_worked']
        edu_mfx = [v for v in mfx_df['Variable'] if v.startswith('edu_')]

        plot_vars = key_mfx_vars + edu_mfx
        plot_mfx = mfx_df[mfx_df['Variable'].isin(plot_vars)].copy()

        # Rename
        rename_map = {
            'formal': 'Formal',
            'female': 'Mujer',
            'age': 'Edad',
            'log_income': 'Log ingreso',
            'hours_worked': 'Horas/semana',
        }
        for old, new in rename_map.items():
            plot_mfx.loc[plot_mfx['Variable'] == old, 'Variable'] = new
        plot_mfx['Variable'] = (plot_mfx['Variable']
                                .str.replace('edu_', 'Educ: ', regex=False)
                                .str.replace('_', ' ', regex=False))

        fig, ax = plt.subplots(figsize=(8, max(5, len(plot_mfx) * 0.4)))

        y_pos = np.arange(len(plot_mfx))
        ci_low = plot_mfx['AME'] - 1.96 * plot_mfx['SE']
        ci_high = plot_mfx['AME'] + 1.96 * plot_mfx['SE']

        colors = [ACCENTRED if p < 0.05 else (AMBER if p < 0.10 else LIGHTGRAY)
                  for p in plot_mfx['p_value']]

        ax.hlines(y_pos, ci_low, ci_high, colors=colors, linewidth=2, alpha=0.8)
        ax.scatter(plot_mfx['AME'], y_pos, color=colors, s=50, zorder=5,
                   edgecolors='white', linewidths=0.5)
        ax.axvline(x=0, color=DARKGRAY, linestyle='-', linewidth=0.8, alpha=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_mfx['Variable'].values)
        ax.set_xlabel('Efecto marginal promedio sobre P(alto riesgo)')
        ax.set_title('Efectos marginales promedio - Modelo logit')

        plt.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, 'fig_marginal_effects.png'), dpi=300)
        fig.savefig(os.path.join(IMG_DIR, 'fig_marginal_effects.pdf'))
        plt.close(fig)
        print("    Saved: fig_marginal_effects.png/pdf")

    # =========================================================================
    # Figure (g): Heatmap - sector x formality showing avg automation risk
    # =========================================================================
    print("  (g) Heatmap: sector x formality...")
    fig, ax = plt.subplots(figsize=(8, 8))

    # Compute weighted mean automation prob by sector and formality
    heat_data = df.groupby(['sector', 'formal']).apply(
        lambda x: np.average(x['automation_prob'].dropna(),
                             weights=x.loc[x['automation_prob'].notna(), 'weight']
                             if x.loc[x['automation_prob'].notna(), 'weight'].sum() > 0
                             else None),
        include_groups=False
    ).unstack(fill_value=np.nan)

    # Rename columns
    heat_data.columns = ['Informal', 'Formal']

    # Sort by overall risk
    heat_data['mean_risk'] = heat_data.mean(axis=1)
    heat_data = heat_data.sort_values('mean_risk', ascending=True)
    heat_data = heat_data.drop(columns=['mean_risk'])

    # Filter to sectors with valid data
    heat_data = heat_data.dropna(how='all')

    # Custom colormap: blue (low risk) -> white -> red (high risk)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'risk', [(0, MAINBLUE), (0.5, '#FFFFFF'), (1.0, ACCENTRED)], N=256
    )

    im = ax.imshow(heat_data.values, cmap=cmap, aspect='auto',
                   vmin=0.1, vmax=0.75)

    ax.set_xticks(np.arange(len(heat_data.columns)))
    ax.set_xticklabels(heat_data.columns)
    ax.set_yticks(np.arange(len(heat_data.index)))
    ax.set_yticklabels(heat_data.index)

    # Add text annotations
    for i in range(len(heat_data.index)):
        for j in range(len(heat_data.columns)):
            val = heat_data.iloc[i, j]
            if pd.notna(val):
                text_color = 'white' if val > 0.55 or val < 0.2 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color=text_color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Probabilidad promedio de automatizacion')

    ax.set_title('Riesgo de automatizacion por sector y formalidad')

    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'fig_heatmap_sector_formality.png'), dpi=300)
    fig.savefig(os.path.join(IMG_DIR, 'fig_heatmap_sector_formality.pdf'))
    plt.close(fig)
    print("    Saved: fig_heatmap_sector_formality.png/pdf")

    # =========================================================================
    # Figure (h): Demographic profile of high-risk workers
    # =========================================================================
    print("  (h) Demographic profile of high-risk workers...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    hr_df = df[df['high_risk'] == 1].copy()
    lr_df = df[df['high_risk'] == 0].copy()

    # (h1) Age distribution
    ax = axes[0, 0]
    bins = np.arange(15, 70, 5)
    ax.hist(lr_df['age'].dropna(), bins=bins, alpha=0.6, color=MAINBLUE,
            label='Bajo riesgo', density=True, edgecolor='white')
    ax.hist(hr_df['age'].dropna(), bins=bins, alpha=0.6, color=ACCENTRED,
            label='Alto riesgo', density=True, edgecolor='white')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Densidad')
    ax.set_title('(a) Distribucion por edad')
    ax.legend(frameon=True, fancybox=False, edgecolor=LIGHTGRAY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (h2) Sex composition
    ax = axes[0, 1]
    sex_data = pd.DataFrame({
        'Alto riesgo': [hr_df['female'].mean(), 1 - hr_df['female'].mean()],
        'Bajo riesgo': [lr_df['female'].mean(), 1 - lr_df['female'].mean()],
    }, index=['Mujer', 'Hombre'])

    x_pos = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, sex_data['Alto riesgo'], width,
                   color=ACCENTRED, alpha=0.8, label='Alto riesgo')
    bars2 = ax.bar(x_pos + width/2, sex_data['Bajo riesgo'], width,
                   color=MAINBLUE, alpha=0.8, label='Bajo riesgo')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Mujer', 'Hombre'])
    ax.set_ylabel('Proporcion')
    ax.set_title('(b) Composicion por sexo')
    ax.legend(frameon=True, fancybox=False, edgecolor=LIGHTGRAY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.0%}', ha='center', va='bottom', fontsize=9)

    # (h3) Education composition
    ax = axes[1, 0]
    edu_order_short = ['Ninguno', 'Primaria', 'Secundaria', 'Media',
                       'Tecnico/\nTecnologico', 'Universitario', 'Posgrado']
    edu_order_full = ['Ninguno', 'Primaria', 'Secundaria', 'Media',
                      'Tecnico/Tecnologico', 'Universitario', 'Posgrado']

    hr_edu = hr_df['education_level'].value_counts(normalize=True)
    lr_edu = lr_df['education_level'].value_counts(normalize=True)

    x_pos = np.arange(len(edu_order_full))
    width = 0.35

    hr_vals = [hr_edu.get(e, 0) for e in edu_order_full]
    lr_vals = [lr_edu.get(e, 0) for e in edu_order_full]

    ax.bar(x_pos - width/2, hr_vals, width, color=ACCENTRED, alpha=0.8, label='Alto riesgo')
    ax.bar(x_pos + width/2, lr_vals, width, color=MAINBLUE, alpha=0.8, label='Bajo riesgo')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(edu_order_short, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Proporcion')
    ax.set_title('(c) Distribucion por nivel educativo')
    ax.legend(frameon=True, fancybox=False, edgecolor=LIGHTGRAY, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (h4) Formality composition
    ax = axes[1, 1]
    form_data = pd.DataFrame({
        'Alto riesgo': [hr_df['formal'].mean(), hr_df['informal'].mean()],
        'Bajo riesgo': [lr_df['formal'].mean(), lr_df['informal'].mean()],
    }, index=['Formal', 'Informal'])

    x_pos = np.arange(2)
    bars1 = ax.bar(x_pos - width/2, form_data['Alto riesgo'], width,
                   color=ACCENTRED, alpha=0.8, label='Alto riesgo')
    bars2 = ax.bar(x_pos + width/2, form_data['Bajo riesgo'], width,
                   color=MAINBLUE, alpha=0.8, label='Bajo riesgo')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Formal', 'Informal'])
    ax.set_ylabel('Proporcion')
    ax.set_title('(d) Composicion por formalidad')
    ax.legend(frameon=True, fancybox=False, edgecolor=LIGHTGRAY)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.0%}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Perfil demografico de trabajadores de alto vs. bajo riesgo de automatizacion',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'fig_demographic_profile_risk.png'), dpi=300,
                bbox_inches='tight')
    fig.savefig(os.path.join(IMG_DIR, 'fig_demographic_profile_risk.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print("    Saved: fig_demographic_profile_risk.png/pdf")

    print("\n  All figures generated successfully.")


# ==============================================================================
# STEP 6: SUMMARY STATISTICS AND TABLES
# ==============================================================================
def generate_tables(df, results):
    """
    Generate descriptive statistics tables and cross-tabulations.
    """
    print("\n" + "=" * 70)
    print("STEP 6: Summary statistics and tables")
    print("=" * 70)

    # =========================================================================
    # Table 1: Descriptive statistics by high_risk status
    # =========================================================================
    print("\n  Table 1: Descriptive statistics by automation risk level...")

    desc_vars = {
        'automation_prob': 'Prob. automatizacion',
        'formal': 'Formal',
        'female': 'Mujer',
        'age': 'Edad',
        'income': 'Ingreso laboral (COP)',
        'hours_worked': 'Horas trabajadas/semana',
        'education_num': 'Nivel educativo (0-6)',
        'urban': 'Urbano',
        'cotiza_pension': 'Cotiza pension',
        'cotiza_salud': 'Cotiza salud',
    }

    rows = []
    for var, label in desc_vars.items():
        if var not in df.columns:
            continue

        total = pd.to_numeric(df[var], errors='coerce').dropna()
        hr = pd.to_numeric(df[df['high_risk'] == 1][var], errors='coerce').dropna()
        lr = pd.to_numeric(df[df['high_risk'] == 0][var], errors='coerce').dropna()

        row = {
            'Variable': label,
            'Total_mean': total.mean(),
            'Total_sd': total.std(),
            'Total_N': len(total),
            'HighRisk_mean': hr.mean(),
            'HighRisk_sd': hr.std(),
            'HighRisk_N': len(hr),
            'LowRisk_mean': lr.mean(),
            'LowRisk_sd': lr.std(),
            'LowRisk_N': len(lr),
        }

        # T-test for difference
        if len(hr) > 1 and len(lr) > 1:
            t_stat, p_val = stats.ttest_ind(hr, lr, nan_policy='omit')
            row['diff_pvalue'] = p_val
        else:
            row['diff_pvalue'] = np.nan

        rows.append(row)

    desc_table = pd.DataFrame(rows)

    # Print formatted table
    print("\n  " + "-" * 95)
    print(f"  {'Variable':30s} | {'Total':>12s} | {'Alto riesgo':>12s} | {'Bajo riesgo':>12s} | {'p-val':>8s}")
    print("  " + "-" * 95)
    for _, row in desc_table.iterrows():
        total_str = f"{row['Total_mean']:.3f}"
        hr_str = f"{row['HighRisk_mean']:.3f}"
        lr_str = f"{row['LowRisk_mean']:.3f}"
        pv_str = f"{row['diff_pvalue']:.4f}" if pd.notna(row['diff_pvalue']) else '-'

        if row['Variable'] == 'Ingreso laboral (COP)':
            total_str = f"{row['Total_mean']:,.0f}"
            hr_str = f"{row['HighRisk_mean']:,.0f}"
            lr_str = f"{row['LowRisk_mean']:,.0f}"

        print(f"  {row['Variable']:30s} | {total_str:>12s} | {hr_str:>12s} | {lr_str:>12s} | {pv_str:>8s}")
    print("  " + "-" * 95)

    n_hr = df['high_risk'].sum()
    n_lr = (df['high_risk'] == 0).sum()
    n_total = len(df)
    print(f"  N: Total={n_total:,}, Alto riesgo={n_hr:,} ({100*n_hr/n_total:.1f}%), "
          f"Bajo riesgo={n_lr:,} ({100*n_lr/n_total:.1f}%)")

    # Save to CSV
    desc_path = os.path.join(DATA_DIR, 'automation_descriptive_stats.csv')
    desc_table.to_csv(desc_path, index=False)
    print(f"\n  Saved: {desc_path}")

    # =========================================================================
    # Table 2: Cross-tabulation sector x risk x formality
    # =========================================================================
    print("\n  Table 2: Cross-tabulation sector x risk x formality...")

    cross_tab = df.groupby(['sector', 'high_risk', 'formal']).agg(
        count=('automation_prob', 'count'),
        mean_prob=('automation_prob', 'mean'),
        mean_income=('income', lambda x: x.mean()),
        weighted_count=('weight', 'sum'),
    ).reset_index()

    cross_tab['high_risk_label'] = cross_tab['high_risk'].map({0: 'Bajo riesgo', 1: 'Alto riesgo'})
    cross_tab['formal_label'] = cross_tab['formal'].map({0: 'Informal', 1: 'Formal'})

    cross_path = os.path.join(DATA_DIR, 'automation_cross_tabulation.csv')
    cross_tab.to_csv(cross_path, index=False)
    print(f"  Saved: {cross_path}")

    # Print summary
    print("\n  Sector-level summary:")
    sector_summary = df.groupby('sector').agg(
        n=('automation_prob', 'count'),
        mean_prob=('automation_prob', 'mean'),
        pct_high_risk=('high_risk', 'mean'),
        pct_formal=('formal', 'mean'),
        mean_income=('income', 'mean'),
    ).sort_values('pct_high_risk', ascending=False)

    print(f"\n  {'Sector':25s} | {'N':>7s} | {'Prob':>6s} | {'%Alto':>6s} | {'%Formal':>7s} | {'Ingreso':>12s}")
    print("  " + "-" * 80)
    for sec, row in sector_summary.iterrows():
        print(f"  {sec:25s} | {row['n']:>7,.0f} | {row['mean_prob']:>6.3f} | "
              f"{row['pct_high_risk']:>6.1%} | {row['pct_formal']:>7.1%} | "
              f"{row['mean_income']:>12,.0f}")

    sector_path = os.path.join(DATA_DIR, 'automation_sector_summary.csv')
    sector_summary.to_csv(sector_path)
    print(f"\n  Saved: {sector_path}")

    # =========================================================================
    # Table 3: Regression results (main models)
    # =========================================================================
    print("\n  Table 3: Regression results summary...")

    if 'odds_ratios' in results:
        or_path = os.path.join(DATA_DIR, 'automation_odds_ratios.csv')
        results['odds_ratios'].to_csv(or_path, index=False)
        print(f"  Saved: {or_path}")

    if 'mfx_df' in results:
        mfx_path = os.path.join(DATA_DIR, 'automation_marginal_effects.csv')
        results['mfx_df'].to_csv(mfx_path, index=False)
        print(f"  Saved: {mfx_path}")

    # =========================================================================
    # Table 4: Occupation group summary
    # =========================================================================
    print("\n  Table 4: Automation risk by occupation group...")

    ocu_summary = df.groupby('ocu_group').agg(
        n=('automation_prob', 'count'),
        mean_prob=('automation_prob', 'mean'),
        pct_high_risk=('high_risk', 'mean'),
        pct_formal=('formal', 'mean'),
        mean_income=('income', 'mean'),
        pct_female=('female', 'mean'),
    ).sort_values('mean_prob', ascending=False)

    print(f"\n  {'Ocupacion':25s} | {'N':>7s} | {'Prob':>6s} | {'%Alto':>6s} | {'%Formal':>7s} | {'%Mujer':>7s}")
    print("  " + "-" * 75)
    for ocu, row in ocu_summary.iterrows():
        print(f"  {ocu:25s} | {row['n']:>7,.0f} | {row['mean_prob']:>6.3f} | "
              f"{row['pct_high_risk']:>6.1%} | {row['pct_formal']:>7.1%} | {row['pct_female']:>7.1%}")

    ocu_path = os.path.join(DATA_DIR, 'automation_occupation_summary.csv')
    ocu_summary.to_csv(ocu_path)
    print(f"\n  Saved: {ocu_path}")

    # =========================================================================
    # Save full analysis dataset (compressed)
    # =========================================================================
    print("\n  Saving analysis dataset...")
    analysis_cols = ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN', 'HOGAR', 'month',
                     'OFICIO_C8', 'ocu_1d', 'ocu_2d', 'ocu_group',
                     'RAMA2D_R4', 'sector',
                     'automation_prob', 'high_risk',
                     'formal', 'informal', 'cotiza_pension', 'cotiza_salud',
                     'female', 'age', 'age_sq',
                     'education_level', 'education_num',
                     'income', 'log_income',
                     'hours_worked', 'firm_size',
                     'position', 'urban', 'departamento',
                     'weight']

    existing_cols = [c for c in analysis_cols if c in df.columns]
    analysis_df = df[existing_cols].copy()

    analysis_path = os.path.join(DATA_DIR, 'automation_analysis_dataset.csv')
    analysis_df.to_csv(analysis_path, index=False)
    print(f"  Saved: {analysis_path} ({len(analysis_df):,} rows)")

    return desc_table, cross_tab, sector_summary


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """Main execution pipeline."""
    print("\n" + "#" * 70)
    print("#  AUTOMATION RISK MODEL - GEIH 2024 MICRODATA")
    print("#  Frey & Osborne (2017) x DANE GEIH 2024")
    print("#" * 70)
    print(f"\n  Base directory: {BASE_DIR}")
    print(f"  GEIH source: {GEIH_DIR}")
    print(f"  Images output: {IMG_DIR}")
    print(f"  Data output: {DATA_DIR}")

    # Step 1: Build crosswalk
    auto_1digit, auto_2digit = build_automation_crosswalk()

    # Step 2: Load and process GEIH data
    df_raw = load_geih_data(auto_1digit, auto_2digit)

    # Step 3: Construct analysis variables
    df = construct_variables(df_raw)
    del df_raw
    gc.collect()

    # Step 4: Econometric models
    results = run_econometric_models(df)

    # Step 5: Generate figures
    generate_figures(df, results)

    # Step 6: Summary tables
    desc_table, cross_tab, sector_summary = generate_tables(df, results)

    # Final summary
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE")
    print("=" * 70)
    print(f"\n  Total observations: {len(df):,}")
    print(f"  High-risk workers: {df['high_risk'].sum():,} ({df['high_risk'].mean():.1%})")
    print(f"  Formal workers: {df['formal'].sum():,} ({df['formal'].mean():.1%})")
    print(f"\n  Figures saved to: {IMG_DIR}")
    print(f"  Tables saved to: {DATA_DIR}")

    if 'logit' in results:
        print(f"\n  Logit model performance:")
        print(f"    Pseudo R-squared: {results['logit'].prsquared:.4f}")
        print(f"    ROC-AUC: {results.get('auc_logit', 'N/A')}")
        print(f"    Accuracy: {results.get('accuracy_logit', 'N/A')}")

    print("\n  Key output files:")
    output_files = [
        'automation_analysis_dataset.csv',
        'automation_descriptive_stats.csv',
        'automation_cross_tabulation.csv',
        'automation_sector_summary.csv',
        'automation_occupation_summary.csv',
        'automation_odds_ratios.csv',
        'automation_marginal_effects.csv',
    ]
    for f in output_files:
        fp = os.path.join(DATA_DIR, f)
        if os.path.exists(fp):
            size = os.path.getsize(fp) / 1024
            print(f"    {f}: {size:.0f} KB")

    image_files = [f for f in os.listdir(IMG_DIR)
                   if f.startswith('fig_') and (f.endswith('.png') or f.endswith('.pdf'))
                   and 'automation' in f or 'roc' in f or 'forest' in f
                   or 'marginal' in f or 'heatmap_sector' in f or 'demographic' in f]
    print(f"\n  Figures generated: {len(image_files)}")
    for f in sorted(image_files):
        print(f"    {f}")

    return df, results


if __name__ == '__main__':
    df, results = main()
