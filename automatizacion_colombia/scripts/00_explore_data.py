#!/usr/bin/env python3
"""
00_explore_data.py
==================
Data exploration script for the research project:
"Automation, Labor Costs, and the Future of Work in Colombian Manufacturing"

This script explores the three core DANE datasets:
  1. GEIH 2024 (Gran Encuesta Integrada de Hogares) - Labor force microdata
  2. EAM 2023 (Encuesta Anual Manufacturera) - Manufacturing survey
  3. EDIT X 2019-2020 (Encuesta de Desarrollo e Innovacion Tecnologica) - Innovation survey

It extracts samples, identifies key variables for the econometric exercises,
and saves a comprehensive report to data/data_exploration_report.txt.

Usage:
    python scripts/00_explore_data.py

Author: Research team
Date: 2026-03-06
"""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Base project directory (adjust if running from a different location)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Paths to source ZIP files
GEIH_DIR = os.path.join(DATA_DIR, "dane",
                        "Gran Encuesta Integrada de Hogares - GEIH - 2024")
EAM_ZIP = os.path.join(DATA_DIR, "BDATOS-EAM-2023.zip")
EDIT_ZIP = os.path.join(DATA_DIR, "EDIT_X_2019_2020.zip")

# Temporary extraction directories
GEIH_TEMP = os.path.join(DATA_DIR, "geih_temp")
EAM_TEMP = os.path.join(DATA_DIR, "eam_temp")
EDIT_TEMP = os.path.join(DATA_DIR, "edit_temp")

# Output report
REPORT_PATH = os.path.join(DATA_DIR, "data_exploration_report.txt")

# GEIH month to use for detailed exploration (January 2024)
GEIH_SAMPLE_ZIP = os.path.join(GEIH_DIR, "Ene_2024.zip")

# Maximum rows to read for exploration (use None for full dataset)
MAX_ROWS_GEIH = 5000  # Sample size for GEIH (full month is ~28K-70K per module)
MAX_ROWS_EAM = None   # EAM is small (~6,700 rows), read all
MAX_ROWS_EDIT = None   # EDIT is small (~6,800 rows), read all


# ==============================================================================
# KEY VARIABLE DEFINITIONS
# These are the variables needed for the econometric exercises
# ==============================================================================

# GEIH: Variables for labor market analysis
# Organized by the CSV module they appear in
GEIH_KEY_VARS = {
    "Ocupados": {
        # Income variables
        "P6500": "Monthly salary/wage from primary job (pesos)",
        "INGLABO": "Total labor income (constructed aggregate)",
        "P6510": "Monthly income from secondary job",
        "P6590": "Income in-kind: food",
        "P6600": "Income in-kind: housing",
        "P6610": "Income in-kind: transport",
        "P6620": "Income in-kind: other",
        "P6585S1": "Subsidio de transporte (transport subsidy)",
        "P6585S2": "Subsidio de alimentacion (food subsidy)",
        "P6585S3": "Subsidio de vivienda (housing subsidy)",
        "P6585S4": "Subsidio educativo (education subsidy)",
        "P6545": "Net profit last month (self-employed)",
        "P6580": "Net profit last month (employer)",
        "P6630S1": "Prima de servicios (service bonus)",
        "P6630S2": "Prima de navidad (Christmas bonus)",
        "P6630S3": "Prima de vacaciones (vacation bonus)",
        "P6630S4": "Viaticos permanentes (permanent travel expenses)",
        "P6630S6": "Bonificaciones (bonuses)",

        # Occupation and sector codes
        "OFICIO_C8": "Occupation code (CIUO-08 adapted, 4 digits)",
        "RAMA2D_R4": "Economic sector code (CIIU Rev.4, 2 digits)",
        "RAMA4D_R4": "Economic sector code (CIIU Rev.4, 4 digits)",
        "P6430": "Occupational position (employee, self-employed, employer, etc.)",
        "P6430S1": "Occupational position - other (specify)",

        # Hours worked
        "P6800": "Hours worked last week in primary job",

        # Firm size indicators
        "P6870": "Number of workers at workplace (if exists)",
        "P6440": "Does the firm where you work have a business registry?",
        "P6450": "Does the firm keep accounting records?",
        "P6460": "Number of workers at the firm (categorical)",

        # Formality indicators
        "P6920": "Contributes to pension fund? (1=yes, 2=no)",
        "P6915": "Contributes to health insurance? (1=yes, 2=no)",
        "P6585S1": "Receives transport subsidy?",
        "P6585S2": "Receives food subsidy?",

        # Contract type
        "P6440": "Has business registration (formal indicator)",
        "P6460": "Firm size category",
        "P7040": "Wants to work more hours?",
        "P7045": "Has been looking for another job?",
        "P7050": "Available to work more hours?",
        "OCI": "Ocupado informal (informal worker flag)",

        # Second job
        "P7040": "Wants to work more hours?",
        "P7050": "Available to work more hours?",
    },

    "Caracteristicas_generales": {
        # Demographics
        "P6040": "Age in years",
        "P3271": "Sex (1=male, 2=female)",
        "P6050": "Relationship to household head",
        "P6016": "Person number within household",
        "DPTO": "Department code (geographic)",
        "AREA": "Area (urban/rural)",
        "CLASE": "Class (1=cabecera, 2=resto)",

        # Education
        "P3042": "Highest education level achieved (ISCLED)",
        "P3042S1": "Last year approved at that education level",
        "P3042S2": "Education level - title obtained",
        "P3038": "Can read and write?",
        "P3039": "Currently studying?",
        "P3041": "Type of educational institution",

        # Social security
        "P6090": "Affiliated to health system?",
        "P6100": "Type of health affiliation",
        "P6110": "Entity of health affiliation",
        "P6120": "Monthly health contribution amount",

        # Expansion factor
        "FEX_C18": "Expansion factor (population weight)",
    },

    "Fuerza_de_trabajo": {
        "FT": "Belongs to labor force? (1=yes)",
        "FFT": "Outside labor force? (1=yes)",
        "PET": "Working-age population (1=yes)",
        "P6240": "Main activity last week (1=working, 2=looking, etc.)",
        "P6300": "Worked at least 1 hour last week?",
        "P6310": "Has a job but did not work last week?",
        "P6320": "Why did not work?",
        "P6340": "Helps in family business without pay?",
    },

    "Datos_hogar_vivienda": {
        "P4030S1": "Wall material",
        "P4030S2": "Floor material",
        "P5000": "Number of rooms",
        "P5010": "Number of bedrooms",
        "P5090": "Tenure status (owner, renter, etc.)",
        "P5140": "Monthly rent amount",
        "P6008": "Household size",
    },

    "Otros_ingresos": {
        "P7495": "Receives other income?",
        "P7500S1": "Rental income",
        "P7500S2": "Pension income",
        "P7500S3": "Government transfer (subsidio)",
        "P7510S1": "Interest/dividends",
        "P7510S2": "Remittances from abroad",
    },
}

# EAM: Variables for manufacturing analysis
EAM_KEY_VARS = {
    # Identifiers
    "nordemp": "Firm ID (anonymized)",
    "nordest": "Establishment ID (anonymized)",
    "dpto": "Department code",
    "ciiu4": "CIIU Rev.4 sector code (4 digits)",
    "periodo": "Survey year",

    # Labor cost variables
    "SALARPER": "Total salaries - permanent workers (thousands of pesos)",
    "PRESSPER": "Total benefits (prestaciones) - permanent workers",
    "SALPEYTE": "Total salaries - permanent + temporary workers",
    "PRESPYTE": "Total benefits (prestaciones) - permanent + temporary",
    "REMUTEMP": "Remuneration of temporary workers",
    "SALAPREN": "Salaries of apprentices (SENA)",

    # Employment variables
    "PERTOTAL": "Total personnel (permanent + temporary + outsourced)",
    "PERSOCU": "Permanently employed personnel",
    "PERSOESC": "Permanently employed personnel (scalar)",
    "PERTEM3": "Average temporary workers",
    "PPERYTEM": "Permanent + temporary workers",
    "PERAPREN": "Number of apprentices",

    # Production and value added
    "PRODBR2": "Gross production (produccion bruta, thousands of pesos)",
    "PRODBIND": "Industrial gross production",
    "VALAGRI": "Industrial value added (valor agregado industrial)",
    "VALORVEN": "Total sales value",
    "VALORCOM": "Value of commercialized goods",
    "VALORCX": "Value of exports",

    # Costs and intermediate consumption
    "CONSMATE": "Cost of raw materials consumed",
    "CONSIN": "Cost of industrial inputs",
    "CONSIN2": "Total intermediate consumption",
    "EELEC": "Electricity expenditure",

    # Investment and capital
    "INVEBRTA": "Gross fixed investment (inversion bruta)",
    "ACTIVFI": "Fixed assets (activos fijos)",
    "DEPRECIA": "Depreciation",

    # Energy consumption (detailed fuel variables c298-c358)
    "TOTALV": "Total energy value consumed",

    # Employment by category (Chapter 4 variables - c4rXcY pattern)
    # c4r1c1n = permanent admin male, c4r1c2n = permanent admin female
    # c4r2c1e = temporary admin male, c4r2c2e = temporary admin female
    # etc. (rows: admin, production, other; cols: male/female, permanent/temp)
}

# EDIT: Variables for innovation and technology analysis
EDIT_KEY_VARS = {
    # Identifiers
    "NORDEMP": "Firm ID (anonymized)",
    "CIIU4": "CIIU Rev.4 sector code (4 digits)",
    "TIPOLO": "Innovation typology (ESTRIC/AMPLIA/POTENC/INTENC/NOINNO)",

    # Chapter I: General characteristics and personnel
    "I1R1C1N": "R&D personnel - professionals (national, year 1)",
    "I1R1C2N": "R&D personnel - professionals (national, year 2)",
    "I1R4C1": "Total R&D personnel (year 1)",
    "I1R4C2": "Total R&D personnel (year 2)",

    # Chapter II: Innovation investment (ACTI - Actividades de CTI)
    "II1R1C1": "Internal R&D investment (year 1, thousands of pesos)",
    "II1R1C2": "Internal R&D investment (year 2)",
    "II1R2C1": "External R&D investment (year 1)",
    "II1R2C2": "External R&D investment (year 2)",
    "II1R3C1": "Machinery & equipment for innovation (year 1)",
    "II1R3C2": "Machinery & equipment for innovation (year 2)",
    "II1R4C1": "Information & communication technology (year 1)",
    "II1R4C2": "Information & communication technology (year 2)",
    "II1R5C1": "Technology transfer (year 1)",
    "II1R5C2": "Technology transfer (year 2)",
    "II1R6C1": "Technical assistance and consulting (year 1)",
    "II1R6C2": "Technical assistance and consulting (year 2)",
    "II1R7C1": "Engineering and industrial design (year 1)",
    "II1R7C2": "Engineering and industrial design (year 2)",
    "II1R8C1": "Training for innovation (year 1)",
    "II1R8C2": "Training for innovation (year 2)",
    "II1R9C1": "Market introduction of innovations (year 1)",
    "II1R9C2": "Market introduction of innovations (year 2)",
    "II1R10C1": "Software for innovation (year 1)",
    "II1R10C2": "Software for innovation (year 2)",
    "II1R11C1": "Intellectual property (year 1)",
    "II1R11C2": "Intellectual property (year 2)",
    "II1R12C1": "Total ACTI investment (year 1)",
    "II1R12C2": "Total ACTI investment (year 2)",
    "II2R1C1": "Innovation investment as % of sales",
    "II3R1C1": "Total innovation investment financing (year 1)",
    "II3R1C2": "Total innovation investment financing (year 2)",

    # Chapter III: R&D projects
    "III1R1C1": "R&D projects - internal (year 1)",
    "III1R1C2": "R&D projects - internal (year 2)",
    "III3R1C1": "Total R&D expenditure",

    # Chapter IV: Innovation results
    "IV1R1C1": "New product - new to international market (year 1)",
    "IV1R2C1": "New product - new to national market (year 1)",
    "IV1R3C1": "New product - new to firm (year 1)",
    "IV1R4C1": "Improved product (year 1)",
    "IV1R5C1": "New process (year 1)",
    "IV1R6C1": "Improved process (year 1)",
    "IV1R7C1": "Organizational innovation (year 1)",
    "IV1R8C1": "Marketing innovation (year 1)",

    # Chapter V: Innovation objectives and results
    "V1R1C1": "Objective: new products",
    "V1R2C1": "Objective: improved quality",
    "V1R3C1": "Objective: extended product range",

    # Chapter VI: Intellectual property
    "VI1R1C1": "Patents filed (year 1)",
    "VI1R1C2": "Patents filed (year 2)",

    # Chapter VIII: Obstacles to innovation
    "VIII1R1C1": "Obstacle: lack of own resources",
    "VIII2R1C1": "Obstacle: lack of external financing",
    "VIII3R1C1": "Obstacle: high innovation costs",
    "VIII4R1C1": "Obstacle: lack of qualified personnel",
    "VIII5R1C1": "Obstacle: lack of market information",
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def extract_geih_csvs(zip_path, output_dir):
    """Extract CSV files from a GEIH monthly ZIP using Python's zipfile
    (handles non-UTF8 filenames in ZIP created on Windows)."""
    os.makedirs(output_dir, exist_ok=True)
    extracted = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [f for f in z.infolist()
                     if f.filename.upper().endswith('.CSV')]
        for f in csv_files:
            safe_name = f.filename.split('/')[-1]
            target = os.path.join(output_dir, safe_name)
            if not os.path.exists(target):
                with z.open(f) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
            extracted[safe_name] = target
    return extracted


def extract_single_file(zip_path, filename_pattern, output_dir):
    """Extract a single file from a ZIP, matching by pattern."""
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        for f in z.infolist():
            if filename_pattern in f.filename:
                target = os.path.join(output_dir, os.path.basename(f.filename))
                if not os.path.exists(target):
                    z.extract(f, output_dir)
                    # Handle nested extraction
                    extracted_path = os.path.join(output_dir, f.filename)
                    if extracted_path != target and os.path.exists(extracted_path):
                        os.rename(extracted_path, target)
                return target
    return None


def describe_dataframe(df, name, key_vars_dict=None, report_lines=None):
    """Generate comprehensive description of a DataFrame."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  DATASET: {name}")
    lines.append(f"{'='*80}")
    lines.append(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    lines.append(f"  Memory usage: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")

    # Data types summary
    lines.append(f"\n  Data types:")
    for dtype, count in df.dtypes.value_counts().items():
        lines.append(f"    {str(dtype):20s}: {count:4d} columns")

    # Missing values summary
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100)
    lines.append(f"\n  Missing values (top 10 by % missing):")
    for col in missing_pct.sort_values(ascending=False).head(10).index:
        lines.append(f"    {col:25s}: {missing[col]:>8,} ({missing_pct[col]:5.1f}%)")

    # All columns
    lines.append(f"\n  All columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        dtype = str(df[col].dtype)
        n_valid = df[col].count()
        n_unique = df[col].nunique()
        lines.append(f"    [{i:3d}] {col:25s}  dtype={dtype:10s}  "
                     f"valid={n_valid:>7,}  unique={n_unique:>6,}")

    # Key variables analysis
    if key_vars_dict:
        lines.append(f"\n  KEY VARIABLES FOR RESEARCH:")
        lines.append(f"  {'-'*76}")
        for var, description in key_vars_dict.items():
            if var in df.columns:
                col = df[var]
                n_valid = col.count()
                n_unique = col.nunique()
                status = "FOUND"
                detail = f"valid={n_valid:,}, unique={n_unique}"
                if pd.api.types.is_numeric_dtype(col):
                    non_null = col.dropna()
                    if len(non_null) > 0:
                        detail += (f", mean={non_null.mean():,.1f}, "
                                  f"min={non_null.min():,.0f}, "
                                  f"max={non_null.max():,.0f}")
                else:
                    top_vals = col.value_counts().head(3)
                    detail += f", top: {dict(top_vals)}"
            else:
                status = "NOT FOUND"
                detail = ""
            lines.append(f"    [{status:>9s}] {var:20s} - {description}")
            if detail:
                lines.append(f"               {'':20s}   {detail}")

    # Basic statistics for numeric columns
    lines.append(f"\n  Descriptive statistics (numeric columns):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe().T
        lines.append(f"    {'Column':25s} {'count':>10s} {'mean':>15s} "
                     f"{'std':>15s} {'min':>15s} {'max':>15s}")
        lines.append(f"    {'-'*95}")
        for col in stats.index[:30]:  # Limit to first 30 for readability
            row = stats.loc[col]
            lines.append(f"    {col:25s} {row['count']:>10,.0f} "
                        f"{row['mean']:>15,.1f} {row['std']:>15,.1f} "
                        f"{row['min']:>15,.0f} {row['max']:>15,.0f}")
        if len(stats) > 30:
            lines.append(f"    ... and {len(stats)-30} more numeric columns")

    result = '\n'.join(lines)
    print(result)
    if report_lines is not None:
        report_lines.append(result)
    return lines


# ==============================================================================
# MAIN EXPLORATION
# ==============================================================================

def main():
    report = []
    report.append("=" * 80)
    report.append("  DATA EXPLORATION REPORT")
    report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"  Project: Automation, Labor Costs, and the Future of Work")
    report.append(f"           in Colombian Manufacturing")
    report.append("=" * 80)

    # ------------------------------------------------------------------
    # 1. LIST CONTENTS OF ALL ZIP FILES
    # ------------------------------------------------------------------
    report.append("\n\n" + "#" * 80)
    report.append("# SECTION 1: ZIP FILE CONTENTS")
    report.append("#" * 80)

    # 1a. GEIH monthly ZIPs
    report.append("\n--- GEIH 2024: Monthly ZIP files ---")
    if os.path.exists(GEIH_DIR):
        zips = sorted([f for f in os.listdir(GEIH_DIR) if f.endswith('.zip')])
        report.append(f"  Found {len(zips)} monthly ZIP files:")
        for zf in zips:
            zpath = os.path.join(GEIH_DIR, zf)
            size_mb = os.path.getsize(zpath) / 1e6
            report.append(f"    {zf:30s}  {size_mb:7.1f} MB")

        # List contents of sample month
        report.append(f"\n  Contents of sample month (Ene_2024.zip):")
        with zipfile.ZipFile(GEIH_SAMPLE_ZIP, 'r') as z:
            for info in z.infolist():
                if not info.is_dir():
                    size_mb = info.file_size / 1e6
                    report.append(f"    {info.filename:70s}  {size_mb:8.1f} MB")

        report.append("\n  Structure: Each month contains 8 CSV/DTA/SAV modules:")
        report.append("    1. Caracteristicas generales (demographics, education, health)")
        report.append("    2. Datos del hogar y la vivienda (household and dwelling)")
        report.append("    3. Fuerza de trabajo (labor force participation)")
        report.append("    4. Migracion (migration)")
        report.append("    5. No ocupados (unemployed/inactive)")
        report.append("    6. Ocupados (employed - MAIN MODULE for our research)")
        report.append("    7. Otras formas de trabajo (other forms of work)")
        report.append("    8. Otros ingresos e impuestos (other income and taxes)")
    else:
        report.append("  WARNING: GEIH directory not found!")

    # 1b. EAM ZIP
    report.append("\n--- EAM 2023: Annual Manufacturing Survey ---")
    if os.path.exists(EAM_ZIP):
        size_mb = os.path.getsize(EAM_ZIP) / 1e6
        report.append(f"  ZIP file: {os.path.basename(EAM_ZIP)} ({size_mb:.1f} MB)")
        with zipfile.ZipFile(EAM_ZIP, 'r') as z:
            for info in z.infolist():
                size_mb = info.file_size / 1e6
                report.append(f"    {info.filename:50s}  {size_mb:8.1f} MB")
    else:
        report.append("  WARNING: EAM ZIP not found!")

    # 1c. EDIT ZIP
    report.append("\n--- EDIT X 2019-2020: Innovation & Technology Survey ---")
    if os.path.exists(EDIT_ZIP):
        size_mb = os.path.getsize(EDIT_ZIP) / 1e6
        report.append(f"  ZIP file: {os.path.basename(EDIT_ZIP)} ({size_mb:.1f} MB)")
        with zipfile.ZipFile(EDIT_ZIP, 'r') as z:
            for info in z.infolist():
                size_mb = info.file_size / 1e6
                report.append(f"    {info.filename:50s}  {size_mb:8.1f} MB")
    else:
        report.append("  WARNING: EDIT ZIP not found!")

    print('\n'.join(report))

    # ------------------------------------------------------------------
    # 2. EXTRACT AND EXPLORE GEIH (January 2024)
    # ------------------------------------------------------------------
    report.append("\n\n" + "#" * 80)
    report.append("# SECTION 2: GEIH 2024 - DETAILED EXPLORATION (January)")
    report.append("#" * 80)

    print("\n\nExtracting GEIH January 2024 CSV files...")
    geih_files = extract_geih_csvs(GEIH_SAMPLE_ZIP, GEIH_TEMP)
    print(f"  Extracted {len(geih_files)} CSV files to {GEIH_TEMP}")

    # Map of filenames to their key variable dictionaries
    geih_modules = {
        "Ocupados.CSV": ("Ocupados (Employed Workers)", GEIH_KEY_VARS["Ocupados"]),
        "Características generales, seguridad social en salud y educación.CSV":
            ("Caracteristicas Generales (Demographics & Education)",
             GEIH_KEY_VARS["Caracteristicas_generales"]),
        "Fuerza de trabajo.CSV":
            ("Fuerza de Trabajo (Labor Force)", GEIH_KEY_VARS["Fuerza_de_trabajo"]),
        "Datos del hogar y la vivienda.CSV":
            ("Datos del Hogar (Household)", GEIH_KEY_VARS["Datos_hogar_vivienda"]),
        "Otros ingresos e impuestos.CSV":
            ("Otros Ingresos (Other Income)", GEIH_KEY_VARS["Otros_ingresos"]),
        "No ocupados.CSV": ("No Ocupados (Unemployed/Inactive)", None),
        "Migración.CSV": ("Migracion", None),
        "Otras formas de trabajo.CSV": ("Otras Formas de Trabajo", None),
    }

    # GEIH files use semicolon separator
    for fname, (label, key_vars) in geih_modules.items():
        fpath = os.path.join(GEIH_TEMP, fname)
        if os.path.exists(fpath):
            print(f"\n  Reading: {label}...")
            df = pd.read_csv(fpath, sep=';', encoding='latin-1',
                           nrows=MAX_ROWS_GEIH)
            describe_dataframe(df, f"GEIH - {label}",
                             key_vars, report)

            # For Ocupados, show merge keys
            if 'Ocupados' in fname:
                report.append("\n  MERGE KEYS for combining GEIH modules:")
                report.append("    DIRECTORIO + SECUENCIA_P + ORDEN + HOGAR")
                report.append("    These uniquely identify a person across modules.")
        else:
            msg = f"\n  WARNING: File not found: {fname}"
            print(msg)
            report.append(msg)

    # ------------------------------------------------------------------
    # 3. EXTRACT AND EXPLORE EAM 2023
    # ------------------------------------------------------------------
    report.append("\n\n" + "#" * 80)
    report.append("# SECTION 3: EAM 2023 - ANNUAL MANUFACTURING SURVEY")
    report.append("#" * 80)

    print("\n\nExtracting EAM 2023...")
    eam_file = os.path.join(EAM_TEMP, "EAM_ANONIMIZADA_2023.txt")
    if not os.path.exists(eam_file):
        extract_single_file(EAM_ZIP, "EAM_ANONIMIZADA_2023.txt", EAM_TEMP)

    if os.path.exists(eam_file):
        print(f"  Reading: {eam_file}")
        # EAM TXT uses tab separator
        df_eam = pd.read_csv(eam_file, sep='\t', encoding='latin-1',
                           nrows=MAX_ROWS_EAM)
        describe_dataframe(df_eam, "EAM 2023 - Encuesta Anual Manufacturera",
                         EAM_KEY_VARS, report)

        # Additional EAM analysis: sector distribution
        report.append("\n  SECTOR DISTRIBUTION (CIIU Rev.4, 4-digit):")
        sector_counts = df_eam['ciiu4'].value_counts().head(20)
        for sector, count in sector_counts.items():
            report.append(f"    CIIU {sector}: {count:,} establishments")

        report.append(f"\n  DEPARTMENT DISTRIBUTION:")
        dept_counts = df_eam['dpto'].value_counts().head(10)
        for dept, count in dept_counts.items():
            report.append(f"    Dept {dept:2d}: {count:,} establishments")

        # Key ratios
        report.append(f"\n  KEY RATIOS (means across establishments):")
        if 'VALAGRI' in df_eam.columns and 'PRODBR2' in df_eam.columns:
            va_ratio = (df_eam['VALAGRI'] / df_eam['PRODBR2']).mean()
            report.append(f"    Value added / Gross production: {va_ratio:.3f}")
        if 'SALPEYTE' in df_eam.columns and 'VALAGRI' in df_eam.columns:
            labor_share = (df_eam['SALPEYTE'] / df_eam['VALAGRI']).mean()
            report.append(f"    Labor cost / Value added: {labor_share:.3f}")
        if 'INVEBRTA' in df_eam.columns and 'PRODBR2' in df_eam.columns:
            inv_ratio = (df_eam['INVEBRTA'] / df_eam['PRODBR2']).mean()
            report.append(f"    Investment / Gross production: {inv_ratio:.3f}")

        # EAM variable structure explanation
        report.append("\n  EAM VARIABLE NAMING CONVENTION:")
        report.append("    Chapter 4 (c4): Employment by category")
        report.append("      c4r{row}c{col}{suffix}")
        report.append("      row 1-5: admin, production, sales, other, total")
        report.append("      col 1-10: M/F pairs for permanent/temp")
        report.append("      suffix: n=national, e=extranjero")
        report.append("    Chapter 3 (c3): Costs and production details")
        report.append("    Chapter 5 (c5): Energy consumption")
        report.append("    Chapter 6 (c6): Water consumption")
        report.append("    Chapter 7 (c7): Investment in fixed assets")
        report.append("    Aggregate vars: SALARPER, PRESSPER, PRODBR2, etc.")
    else:
        msg = "  WARNING: EAM file not found!"
        print(msg)
        report.append(msg)

    # ------------------------------------------------------------------
    # 4. EXTRACT AND EXPLORE EDIT X
    # ------------------------------------------------------------------
    report.append("\n\n" + "#" * 80)
    report.append("# SECTION 4: EDIT X 2019-2020 - INNOVATION & TECHNOLOGY")
    report.append("#" * 80)

    print("\n\nExtracting EDIT X...")
    edit_file = os.path.join(EDIT_TEMP, "EDIT_X_2019_2020.csv")
    if not os.path.exists(edit_file):
        extract_single_file(EDIT_ZIP, "EDIT_X_2019_2020.csv", EDIT_TEMP)

    if os.path.exists(edit_file):
        print(f"  Reading: {edit_file}")
        # EDIT CSV uses semicolon separator
        df_edit = pd.read_csv(edit_file, sep=';', encoding='latin-1',
                            nrows=MAX_ROWS_EDIT)
        describe_dataframe(df_edit, "EDIT X 2019-2020 - Innovation Survey",
                         EDIT_KEY_VARS, report)

        # Innovation typology distribution
        report.append("\n  INNOVATION TYPOLOGY DISTRIBUTION:")
        if 'TIPOLO' in df_edit.columns:
            tipolo = df_edit['TIPOLO'].value_counts()
            for cat, count in tipolo.items():
                pct = count / len(df_edit) * 100
                report.append(f"    {cat:10s}: {count:5,} firms ({pct:5.1f}%)")

            report.append("\n  Typology definitions:")
            report.append("    ESTRIC  = Innovadora en sentido estricto (strict innovator)")
            report.append("    AMPLIA  = Innovadora en sentido amplio (broad innovator)")
            report.append("    POTENC  = Potencialmente innovadora (potential innovator)")
            report.append("    INTENC  = Con intenciones de innovar (intends to innovate)")
            report.append("    NOINNO  = No innovadora (non-innovator)")

        # Sector distribution
        if 'CIIU4' in df_edit.columns:
            report.append(f"\n  SECTOR DISTRIBUTION (CIIU Rev.4, top 20):")
            sector_counts = df_edit['CIIU4'].value_counts().head(20)
            for sector, count in sector_counts.items():
                report.append(f"    CIIU {sector}: {count:,} firms")

        # EDIT variable structure explanation
        report.append("\n  EDIT X VARIABLE NAMING CONVENTION:")
        report.append("    Chapters are encoded in variable prefixes:")
        report.append("    I   = Chap I: General characteristics and personnel")
        report.append("    II  = Chap II: Innovation investment (ACTI)")
        report.append("    III = Chap III: R&D projects")
        report.append("    IV  = Chap IV: Innovation results (products/processes)")
        report.append("    V   = Chap V: Innovation objectives and impacts")
        report.append("    VI  = Chap VI: Intellectual property")
        report.append("    VIII= Chap VIII: Obstacles to innovation")
        report.append("    Variable format: {Chapter}{Row}R{N}C{Col}")
        report.append("    C1 usually = year 2019, C2 = year 2020")
    else:
        msg = "  WARNING: EDIT file not found!"
        print(msg)
        report.append(msg)

    # ------------------------------------------------------------------
    # 5. SUMMARY: LINKING STRATEGY
    # ------------------------------------------------------------------
    report.append("\n\n" + "#" * 80)
    report.append("# SECTION 5: LINKING STRATEGY AND RESEARCH VARIABLES")
    report.append("#" * 80)

    report.append("""
  LINKING DATASETS:
  =================

  1. GEIH (household survey) <-> EAM/EDIT (firm surveys):
     - Cannot be directly linked at the firm/individual level
     - Link via sector codes: GEIH.RAMA4D_R4 <-> EAM.ciiu4 <-> EDIT.CIIU4
     - Geographic link: GEIH.DPTO <-> EAM.dpto
     - This enables sector-level analysis of automation vs. labor outcomes

  2. EAM <-> EDIT:
     - Both contain CIIU4 sector codes
     - EAM has NORDEMP (firm ID) and EDIT has NORDEMP
     - Potential direct firm-level merge for overlapping firms
     - Note: EAM is 2023, EDIT is 2019-2020 (different periods)

  3. Within GEIH:
     - Modules are linked by: DIRECTORIO + SECUENCIA_P + ORDEN + HOGAR
     - Must merge Ocupados + Caracteristicas to get demographics + job info
     - Weight variable: FEX_C18 (expansion factor)

  KEY VARIABLES FOR ECONOMETRIC EXERCISES:
  ========================================

  Exercise 1: Labor cost analysis by sector (EAM)
    - Dependent: SALARPER + PRESSPER (total labor cost)
    - Independent: CIIU4, PRODBR2, INVEBRTA, PERTOTAL
    - Compute: unit labor cost = (SALPEYTE+PRESPYTE) / VALAGRI

  Exercise 2: Automation exposure and wages (GEIH + sector aggregates)
    - Dependent: INGLABO (labor income)
    - Independent: RAMA4D_R4, OFICIO_C8, P6040, P3271, P3042, AREA
    - Formality: P6920, P6915, OCI
    - Hours: P6800

  Exercise 3: Innovation investment and labor displacement (EDIT + EAM)
    - EDIT investment vars: II1R3C1/C2 (machinery for innovation),
      II1R4C1/C2 (ICT), II1R10C1/C2 (software)
    - EAM labor vars: PERTOTAL, PERSOCU, PPERYTEM
    - Link via CIIU4 sector code

  Exercise 4: Technology adoption and skill composition
    - EDIT: TIPOLO (innovation type), II1R1-R12 (investment components)
    - GEIH: P3042 (education level), OFICIO_C8 (occupation code)
    - EAM: c4r1-r5 (employment by category: admin/production/other)

  DATA QUALITY NOTES:
  ===================
  - GEIH: Well-structured, large sample (~28K employed per month)
  - EAM: 6,714 establishments, 386 variables, all integer types
  - EDIT: 6,798 firms, 729 variables, many sparse (innovation-related)
  - All GEIH CSVs use semicolon (;) as separator, latin-1 encoding
  - EAM TXT uses tab separator, latin-1 encoding
  - EDIT CSV uses semicolon (;) separator, latin-1 encoding
""")

    # ------------------------------------------------------------------
    # SAVE REPORT
    # ------------------------------------------------------------------
    report_text = '\n'.join(report)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n\n{'='*80}")
    print(f"  Report saved to: {REPORT_PATH}")
    print(f"  Report size: {len(report_text):,} characters")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
