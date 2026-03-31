#!/usr/bin/env python3
"""
10_did_geih_mw_shock.py
========================
Difference-in-Differences analysis of minimum wage shocks on automation risk
and labor formality in Colombia, using DANE's GEIH microdata (2022-2024).

Treatment:
    President Petro's above-inflation minimum wage increases:
        2022: $1,000,000 COP (baseline)
        2023: $1,160,000 COP (+16%)
        2024: $1,300,000 COP (+12%)

Identification strategy:
    Sector-level minimum wage bite (share of workers near MW in 2022)
    as continuous and binary treatment intensity. Sectors with higher
    pre-treatment MW bite are more exposed to the MW shock.

Outcomes:
    1. Formality rate (pension contribution)
    2. Mean automation risk (Frey-Osborne, CIUO-08 crosswalk)
    3. Share of high-risk workers (automation risk >= 0.50)
    4. Mean log labor income
    5. Total employment (expansion-factor weighted)

Outputs:
    - data/geih_did_dataset.csv        (sector-year-quarter aggregates)
    - data/geih_did_results.csv        (regression coefficients)
    - images/en/fig_did_event_formality.{png,pdf}
    - images/en/fig_did_event_automation.{png,pdf}
    - images/en/fig_did_parallel_trends.{png,pdf}
    - images/en/fig_did_mw_bite_distribution.{png,pdf}
    - images/en/fig_did_scatter_bite_formality.{png,pdf}

Author: Cristian Espinal / Fourier.dev
Date: 2026-03-20
"""

import os
import sys
import io
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    pass  # Older pandas versions don't have this warning class

# ──────────────────────────────────────────────────────────────────────
# 0. PATHS AND CONSTANTS
# ──────────────────────────────────────────────────────────────────────

PROJECT = Path("/Users/cristianespinal/Claude Code/Projects/Research/automatizacion_colombia")
DATA_DIR = PROJECT / "data"
IMG_DIR = PROJECT / "images" / "en"
SCRIPTS_DIR = PROJECT / "scripts"

GEIH_2022 = DATA_DIR / "geih_panel" / "GEIH_2022"
GEIH_2023 = DATA_DIR / "geih_panel" / "GEIH_2023"
GEIH_2024 = DATA_DIR / "dane" / "Gran Encuesta Integrada de Hogares - GEIH - 2024"

IMG_DIR.mkdir(parents=True, exist_ok=True)

# Minimum wage (SMMLV) and transport subsidy by year
SMMLV = {2022: 1_000_000, 2023: 1_160_000, 2024: 1_300_000,
          2025: 1_423_500, 2026: 1_623_500}
AUXILIO_TRANSPORTE = {2022: 117_172, 2023: 140_606, 2024: 162_000}

# Months to sample (quarterly: Jan, Apr, Jul, Oct)
SAMPLE_MONTHS = [1, 4, 7, 10]

# Zip file mapping: (year, month) -> (base_path, filename)
ZIP_FILES = {
    (2022, 1):  (GEIH_2022, "GEIH_Enero_2022_Marco_2018.zip"),
    (2022, 4):  (GEIH_2022, "GEIH_Abril_2022_Marco_2018_Act.zip"),
    (2022, 7):  (GEIH_2022, "GEIH_Julio_2022_Marco_2018.zip"),
    (2022, 10): (GEIH_2022, "GEIH_Octubre_Marco_2018.zip"),
    (2023, 1):  (GEIH_2023, "Enero.zip"),
    (2023, 4):  (GEIH_2023, "Abril.zip"),
    (2023, 7):  (GEIH_2023, "Julio.zip"),
    (2023, 10): (GEIH_2023, "Octubre.zip"),
    (2024, 1):  (GEIH_2024, "Ene_2024.zip"),
    (2024, 4):  (GEIH_2024, "Abril 2024.zip"),
    (2024, 7):  (GEIH_2024, "Julio_2024.zip"),
    (2024, 10): (GEIH_2024, "Octubre_2024.zip"),
}

# Frey-Osborne automation risk by CIUO-08 2-digit occupation code
AUTOMATION_RISK_MAP = {
    "01": 0.03, "11": 0.15, "12": 0.16, "13": 0.10, "14": 0.18,
    "21": 0.08, "22": 0.04, "23": 0.03, "24": 0.35, "25": 0.15,
    "26": 0.12, "31": 0.35, "32": 0.30, "33": 0.55, "34": 0.48,
    "35": 0.50, "41": 0.75, "42": 0.68, "43": 0.75, "44": 0.70,
    "51": 0.30, "52": 0.62, "53": 0.15, "54": 0.30, "61": 0.55,
    "62": 0.52, "63": 0.45, "71": 0.58, "72": 0.65, "73": 0.60,
    "74": 0.55, "75": 0.65, "81": 0.72, "82": 0.70, "83": 0.68,
    "91": 0.52, "92": 0.60, "93": 0.58, "94": 0.55, "95": 0.40,
    "96": 0.50,
}

# ──────────────────────────────────────────────────────────────────────
# 1. MATPLOTLIB PUBLICATION STYLE
# ──────────────────────────────────────────────────────────────────────

def set_publication_style():
    """Configure matplotlib for publication-quality figures (Nature style)."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (7, 5),
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


# ──────────────────────────────────────────────────────────────────────
# 2. DATA LOADING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def _find_csv_in_namelist(namelist, keyword):
    """Find a CSV file in a zip namelist matching keyword (case-insensitive).

    Uses basename matching to avoid false positives (e.g., 'No ocupados'
    matching 'ocupados'). Requires the keyword to match as a word boundary
    in the filename.
    """
    keyword_lower = keyword.lower()
    for name in namelist:
        name_lower = name.lower()
        if not name_lower.endswith(".csv"):
            continue
        # Extract just the filename part (after last /)
        basename = name_lower.rsplit("/", 1)[-1] if "/" in name_lower else name_lower
        # For "ocupados", ensure it's not "no ocupados"
        if keyword_lower == "ocupados":
            if "ocupados" in basename and "no ocupados" not in basename:
                return name
        elif keyword_lower in basename:
            return name
    return None


def _read_csv_from_file_obj(file_obj, year, module_name=""):
    """Read a CSV from a file object, auto-detecting delimiter.

    For 2022, most modules use comma; for 2023-2024, semicolon.
    We sniff the first line to be safe.
    """
    raw = file_obj.read()
    text = raw.decode("utf-8", errors="replace")

    # Sniff delimiter from header line
    first_line = text.split("\n")[0]
    n_semi = first_line.count(";")
    n_comma = first_line.count(",")
    sep = ";" if n_semi > n_comma else ","

    df = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        low_memory=False,
        dtype=str,  # Read everything as string initially for safety
        na_values=["", " ", ".", "NA", "NaN", "nan"],
    )
    return df


def _open_csv_from_zip(zip_path, keyword):
    """Open a CSV matching keyword from a (possibly nested) zip file.

    Handles three structures:
      1. Flat: CSV files directly inside outer zip
      2. Nested: outer zip contains a CSV.zip which contains the CSVs
      3. Double-nested: outer zip/subdir/CSV.zip/CSV/files
    """
    z_outer = zipfile.ZipFile(zip_path, "r")
    all_names = z_outer.namelist()

    # Strategy 1: Look for CSV file directly in outer zip
    csv_match = _find_csv_in_namelist(all_names, keyword)
    if csv_match:
        return z_outer.open(csv_match), z_outer, None

    # Strategy 2: Look for a nested zip containing "CSV" in its name
    inner_zip_names = [n for n in all_names if n.lower().endswith(".zip")
                       and ("csv" in n.lower() or "cvs" in n.lower())]
    if inner_zip_names:
        inner_zip_name = inner_zip_names[0]
        with z_outer.open(inner_zip_name) as inner_f:
            inner_bytes = io.BytesIO(inner_f.read())
        z_inner = zipfile.ZipFile(inner_bytes)
        csv_match = _find_csv_in_namelist(z_inner.namelist(), keyword)
        if csv_match:
            return z_inner.open(csv_match), z_outer, z_inner

    raise FileNotFoundError(
        f"Could not find '{keyword}' CSV in {zip_path}. "
        f"Contents: {all_names[:10]}"
    )


def load_geih_month(year, month):
    """Load and merge Ocupados + Caracteristicas generales for one GEIH month.

    Returns a DataFrame with standardized lowercase column names,
    plus 'year' and 'month' columns.
    """
    base_path, fname = ZIP_FILES[(year, month)]
    zip_path = str(base_path / fname)

    print(f"  Loading GEIH {year}-{month:02d} from {fname}...", end=" ", flush=True)

    # --- Load Ocupados ---
    try:
        f_ocup, z1, z1_inner = _open_csv_from_zip(zip_path, "ocupados")
        df_ocup = _read_csv_from_file_obj(f_ocup, year, "Ocupados")
        f_ocup.close()
    except Exception as e:
        print(f"ERROR loading Ocupados: {e}")
        return None

    # --- Load Caracteristicas generales ---
    try:
        f_caract, z2, z2_inner = _open_csv_from_zip(zip_path, "sticas")
        df_caract = _read_csv_from_file_obj(f_caract, year, "Caracteristicas")
        f_caract.close()
    except Exception as e:
        print(f"ERROR loading Caracteristicas: {e}")
        return None

    # Standardize column names to uppercase for merging, then lowercase
    df_ocup.columns = [c.strip().upper() for c in df_ocup.columns]
    df_caract.columns = [c.strip().upper() for c in df_caract.columns]

    # Deduplicate columns within each DataFrame (keep first occurrence)
    df_ocup = df_ocup.loc[:, ~df_ocup.columns.duplicated()]
    df_caract = df_caract.loc[:, ~df_caract.columns.duplicated()]

    # Determine merge keys
    merge_keys = ["DIRECTORIO", "SECUENCIA_P", "ORDEN"]
    if "HOGAR" in df_ocup.columns and "HOGAR" in df_caract.columns:
        merge_keys.append("HOGAR")

    # Only keep columns from df_caract that are either merge keys or
    # NOT already in df_ocup (avoid duplicates in the merged result)
    ocup_cols_set = set(df_ocup.columns)
    caract_cols_to_keep = list(dict.fromkeys(
        merge_keys + [
            c for c in df_caract.columns
            if c not in ocup_cols_set
        ]
    ))
    df_caract_slim = df_caract[caract_cols_to_keep].copy()

    # Merge
    df = pd.merge(df_ocup, df_caract_slim, on=merge_keys, how="left")

    # Add year and month
    df["YEAR"] = year
    df["MONTH"] = month

    # Convert to lowercase
    df.columns = [c.lower() for c in df.columns]

    n = len(df)
    print(f"OK ({n:,} obs)")
    return df


def load_all_geih():
    """Load all quarterly GEIH data for 2022-2024 and stack into one DataFrame."""
    frames = []
    for year in [2022, 2023, 2024]:
        for month in SAMPLE_MONTHS:
            df = load_geih_month(year, month)
            if df is not None:
                frames.append(df)

    if not frames:
        raise RuntimeError("No GEIH data could be loaded!")

    full = pd.concat(frames, ignore_index=True)
    print(f"\n  Total stacked observations: {len(full):,}")
    for yr in sorted(full["year"].unique()):
        n = len(full[full["year"] == yr])
        print(f"    {yr}: {n:,} obs")

    return full


# ──────────────────────────────────────────────────────────────────────
# 3. VARIABLE CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────

def construct_variables(df):
    """Construct all analysis variables from raw GEIH microdata.

    Adds: formal, near_minimum_wage, at_minimum_wage, automation_risk,
    high_risk, education_level, age_group, log_income, quarter, post,
    ocup_2d, sector_2d.
    """
    print("\nConstructing analysis variables...")

    # --- Convert numeric columns ---
    numeric_cols = ["p6920", "p6500", "inglabo", "p6800", "fex_c18",
                    "p6040", "p3271", "p3041", "p3042", "p6430", "p6870",
                    "rama2d_r4", "dpto"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Formality: pension contribution ---
    # P6920: 1 = yes contributes to pension, 2 = no
    df["formal"] = np.where(df["p6920"] == 1, 1, 0)
    df.loc[df["p6920"].isna(), "formal"] = np.nan

    # --- Monthly wage (P6500) ---
    df["wage"] = df["p6500"].copy()
    df.loc[df["wage"] <= 0, "wage"] = np.nan

    # --- Total labor income ---
    df["income"] = df["inglabo"].copy()
    df.loc[df["income"] <= 0, "income"] = np.nan
    df["log_income"] = np.log(df["income"].clip(lower=1))

    # --- Near-minimum and at-minimum wage indicators ---
    for _, row_year in df.groupby("year"):
        pass  # we do it vectorized below

    df["near_mw"] = 0
    df["at_mw"] = 0
    for yr in df["year"].unique():
        mw = SMMLV.get(yr, np.nan)
        if pd.isna(mw):
            continue
        mask_yr = df["year"] == yr
        wage_yr = df.loc[mask_yr, "wage"]
        df.loc[mask_yr, "near_mw"] = np.where(
            wage_yr.between(mw * 0.8, mw * 1.2), 1, 0
        )
        df.loc[mask_yr, "at_mw"] = np.where(
            wage_yr.between(mw * 0.95, mw * 1.05), 1, 0
        )

    # --- Occupation 2-digit code from OFICIO_C8 ---
    if "oficio_c8" in df.columns:
        df["oficio_str"] = df["oficio_c8"].astype(str).str.strip()
        # OFICIO_C8 can be 4-digit or 8-digit; first 2 digits = major group
        df["ocup_2d"] = df["oficio_str"].str[:2]
        # Clean: if code is too short or non-numeric, set NaN
        df.loc[~df["ocup_2d"].str.match(r"^\d{2}$", na=False), "ocup_2d"] = np.nan
    else:
        print("  WARNING: oficio_c8 column not found!")
        df["ocup_2d"] = np.nan

    # --- Automation risk ---
    df["automation_risk"] = df["ocup_2d"].map(AUTOMATION_RISK_MAP)
    df["high_risk"] = np.where(df["automation_risk"] >= 0.50, 1, 0)
    df.loc[df["automation_risk"].isna(), "high_risk"] = np.nan

    # --- Sector 2-digit ---
    df["sector_2d"] = df["rama2d_r4"].copy()
    df.loc[df["sector_2d"] <= 0, "sector_2d"] = np.nan

    # --- Education level recoding ---
    # P3041: 1=None, 2=Preschool, 3=Primary, 4=Secondary, 5=Middle,
    #         6=Technical, 7=Technological, 8=Professional, 9=Specialization,
    #         10=Master, 11=Doctorate, 13=Don't know
    edu_map = {
        1: 1, 2: 1, 3: 1,       # None/Preschool/Primary -> 1
        4: 2, 5: 2,              # Secondary/Media -> 2
        6: 3, 7: 3,              # Technical/Technological -> 3
        8: 4,                    # Professional -> 4
        9: 5, 10: 5, 11: 5,     # Postgraduate -> 5
    }
    df["education_level"] = df["p3041"].map(edu_map)

    # --- Age group ---
    df["age"] = df["p6040"].copy()
    df["age_group"] = pd.cut(
        df["age"], bins=[14, 24, 34, 44, 54, 100],
        labels=["15-24", "25-34", "35-44", "45-54", "55+"],
        right=True
    )

    # --- Sex ---
    df["female"] = np.where(df["p3271"] == 2, 1, 0)

    # --- Quarter ---
    month_to_q = {1: 1, 4: 2, 7: 3, 10: 4}
    df["quarter"] = df["month"].map(month_to_q)
    df["year_quarter"] = df["year"].astype(str) + "q" + df["quarter"].astype(str)

    # --- Post period ---
    df["post"] = np.where(df["year"] >= 2023, 1, 0)

    # --- Expansion factor ---
    df["weight"] = df["fex_c18"].copy()
    df.loc[df["weight"].isna() | (df["weight"] <= 0), "weight"] = 1.0

    # Report
    print(f"  Formal rate (unweighted): {df['formal'].mean():.3f}")
    print(f"  Mean automation risk: {df['automation_risk'].mean():.3f}")
    print(f"  Share high-risk: {df['high_risk'].mean():.3f}")
    print(f"  Near-MW share: {df['near_mw'].mean():.3f}")
    print(f"  Observations with valid sector: {df['sector_2d'].notna().sum():,}")

    return df


# ──────────────────────────────────────────────────────────────────────
# 4. SECTOR-LEVEL MINIMUM WAGE BITE
# ──────────────────────────────────────────────────────────────────────

def compute_sector_mw_bite(df):
    """Compute pre-treatment (2022) sector-level minimum wage bite.

    MW bite = weighted share of workers earning <= 1.2 x SMMLV_2022.
    Returns a DataFrame with sector_2d, sector_mw_bite_2022, high_bite.
    """
    print("\nComputing sector-level MW bite (2022 baseline)...")

    mw_2022 = SMMLV[2022]
    threshold = mw_2022 * 1.2

    pre = df[(df["year"] == 2022) & df["sector_2d"].notna() & df["wage"].notna()].copy()
    pre["below_threshold"] = np.where(pre["wage"] <= threshold, 1, 0)

    bite = pre.groupby("sector_2d").apply(
        lambda g: np.average(g["below_threshold"], weights=g["weight"])
    ).reset_index()
    bite.columns = ["sector_2d", "sector_mw_bite_2022"]

    # Binary treatment: above-median bite
    median_bite = bite["sector_mw_bite_2022"].median()
    bite["high_bite"] = np.where(bite["sector_mw_bite_2022"] >= median_bite, 1, 0)

    print(f"  Number of sectors: {len(bite)}")
    print(f"  Median MW bite: {median_bite:.3f}")
    print(f"  High-bite sectors: {bite['high_bite'].sum()}")
    print(f"  Low-bite sectors: {(bite['high_bite'] == 0).sum()}")
    print(f"  MW bite range: [{bite['sector_mw_bite_2022'].min():.3f}, "
          f"{bite['sector_mw_bite_2022'].max():.3f}]")

    return bite


# ──────────────────────────────────────────────────────────────────────
# 5. AGGREGATE TO SECTOR-YEAR-QUARTER CELLS
# ──────────────────────────────────────────────────────────────────────

def aggregate_to_cells(df, bite_df):
    """Aggregate individual data to sector x year x quarter cells.

    Merges in the sector MW bite and computes weighted outcome means.
    """
    print("\nAggregating to sector-year-quarter cells...")

    # Merge sector bite into individual data
    df = df.merge(bite_df, on="sector_2d", how="left")

    # Filter to valid observations
    valid = df[
        df["sector_2d"].notna()
        & df["weight"].notna()
        & (df["weight"] > 0)
    ].copy()

    def weighted_mean(x, w):
        mask = x.notna() & w.notna()
        if mask.sum() == 0:
            return np.nan
        return np.average(x[mask], weights=w[mask])

    cells = valid.groupby(["sector_2d", "year", "quarter"]).apply(
        lambda g: pd.Series({
            "formality_rate": weighted_mean(g["formal"], g["weight"]),
            "mean_automation_risk": weighted_mean(g["automation_risk"], g["weight"]),
            "share_high_risk": weighted_mean(g["high_risk"], g["weight"]),
            "mean_log_income": weighted_mean(g["log_income"], g["weight"]),
            "total_employment": g["weight"].sum(),
            "n_obs": len(g),
            "sector_mw_bite_2022": g["sector_mw_bite_2022"].iloc[0],
            "high_bite": g["high_bite"].iloc[0],
        })
    ).reset_index()

    cells["post"] = np.where(cells["year"] >= 2023, 1, 0)
    cells["year_quarter"] = (
        cells["year"].astype(int).astype(str) + "q"
        + cells["quarter"].astype(int).astype(str)
    )
    cells["did_term"] = cells["high_bite"] * cells["post"]

    # Time index for event study (0-based: 2022q1=0, ..., 2024q4=11)
    cells["time_idx"] = (
        (cells["year"].astype(int) - 2022) * 4
        + cells["quarter"].astype(int) - 1
    )

    print(f"  Total cells: {len(cells)}")
    print(f"  Sectors: {cells['sector_2d'].nunique()}")
    print(f"  Year-quarters: {sorted(cells['year_quarter'].unique())}")

    return cells, df  # Return df with bite merged


# ──────────────────────────────────────────────────────────────────────
# 6. DiD ESTIMATION
# ──────────────────────────────────────────────────────────────────────

def run_cell_did(cells, outcome, outcome_label):
    """Run cell-level DiD regression with sector and time fixed effects.

    Y_st = alpha_s + gamma_t + beta(high_bite_s x post_t) + eps_st
    Weighted by total_employment, clustered at sector level.
    """
    data = cells.dropna(subset=[outcome, "high_bite", "post"]).copy()
    data["sector_fe"] = data["sector_2d"].astype(str)
    data["time_fe"] = data["year_quarter"].astype(str)

    # Create dummies manually for more control
    sector_dummies = pd.get_dummies(data["sector_fe"], prefix="s", drop_first=True)
    time_dummies = pd.get_dummies(data["time_fe"], prefix="t", drop_first=True)

    X = pd.concat([
        data[["did_term"]],
        sector_dummies,
        time_dummies,
    ], axis=1).astype(float)
    X = sm.add_constant(X)

    y = data[outcome].astype(float)
    w = data["total_employment"].astype(float)

    # WLS with cluster-robust SEs at sector level
    model = sm.WLS(y, X, weights=w)
    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["sector_fe"].values},
    )

    beta = results.params["did_term"]
    se = results.bse["did_term"]
    ci_lo = results.conf_int().loc["did_term", 0]
    ci_hi = results.conf_int().loc["did_term", 1]
    pval = results.pvalues["did_term"]
    nobs = int(results.nobs)

    print(f"\n  Cell-level DiD: {outcome_label}")
    print(f"    beta(DiD) = {beta:.4f}  (SE = {se:.4f})")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    p-value = {pval:.4f}  |  N = {nobs}")

    return {
        "outcome": outcome_label,
        "model": "cell_did",
        "beta": beta,
        "se": se,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "p_value": pval,
        "n_obs": nobs,
    }


def run_individual_did(df_merged, outcome, outcome_label):
    """Run individual-level DiD for robustness.

    Y_ist = alpha_s + gamma_t + beta(high_bite_s x post_t) + X_i'delta + eps_ist
    Clustered at sector level.
    """
    cols_needed = [outcome, "high_bite", "post", "age", "female",
                   "education_level", "log_income", "sector_2d",
                   "year_quarter", "weight"]
    data = df_merged.dropna(subset=[c for c in cols_needed if c != "log_income"]).copy()

    # Filter to working-age with valid sector
    data = data[(data["age"] >= 15) & (data["age"] <= 65)].copy()
    data["did_term"] = data["high_bite"] * data["post"]
    data["age_sq"] = data["age"] ** 2

    # Subsample if too large (> 500K) for computational feasibility
    if len(data) > 500_000:
        print(f"    Subsampling from {len(data):,} to 500,000 for tractability...")
        data = data.sample(n=500_000, weights="weight", random_state=42,
                           replace=False).copy()

    data["sector_fe"] = data["sector_2d"].astype(str)
    data["time_fe"] = data["year_quarter"].astype(str)

    sector_dummies = pd.get_dummies(data["sector_fe"], prefix="s", drop_first=True)
    time_dummies = pd.get_dummies(data["time_fe"], prefix="t", drop_first=True)

    X = pd.concat([
        data[["did_term", "age", "age_sq", "female", "education_level"]],
        sector_dummies,
        time_dummies,
    ], axis=1).astype(float)
    X = sm.add_constant(X)

    y = data[outcome].astype(float)
    w = data["weight"].astype(float)

    model = sm.WLS(y, X, weights=w)
    try:
        results = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": data["sector_fe"].values},
        )
    except Exception as e:
        print(f"    Individual DiD failed: {e}")
        return None

    beta = results.params["did_term"]
    se = results.bse["did_term"]
    ci_lo = results.conf_int().loc["did_term", 0]
    ci_hi = results.conf_int().loc["did_term", 1]
    pval = results.pvalues["did_term"]
    nobs = int(results.nobs)

    print(f"\n  Individual-level DiD (robustness): {outcome_label}")
    print(f"    beta(DiD) = {beta:.4f}  (SE = {se:.4f})")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    p-value = {pval:.4f}  |  N = {nobs:,}")

    return {
        "outcome": outcome_label,
        "model": "individual_did",
        "beta": beta,
        "se": se,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "p_value": pval,
        "n_obs": nobs,
    }


def run_event_study(cells, outcome, outcome_label):
    """Run event study regression at the cell level.

    Y_st = alpha_s + gamma_t + sum_k beta_k (mw_bite_s x 1{time=k}) + eps_st
    Omits 2022q4 (time_idx=3) as reference period.

    Returns coefficients and CIs for the event-study plot.
    """
    data = cells.dropna(subset=[outcome, "sector_mw_bite_2022"]).copy()
    data["sector_fe"] = data["sector_2d"].astype(str)

    # Reference period: 2022q4 = time_idx 3
    ref_idx = 3
    time_indices = sorted(data["time_idx"].unique())
    time_indices_no_ref = [t for t in time_indices if t != ref_idx]

    # Create interaction terms: mw_bite x 1{time=k}
    for t in time_indices_no_ref:
        data[f"bite_x_t{t}"] = data["sector_mw_bite_2022"] * (data["time_idx"] == t).astype(float)

    interact_cols = [f"bite_x_t{t}" for t in time_indices_no_ref]

    sector_dummies = pd.get_dummies(data["sector_fe"], prefix="s", drop_first=True)
    time_dummies = pd.get_dummies(data["time_idx"].astype(str), prefix="t", drop_first=True)

    X = pd.concat([
        data[interact_cols],
        sector_dummies,
        time_dummies,
    ], axis=1).astype(float)
    X = sm.add_constant(X)

    y = data[outcome].astype(float)
    w = data["total_employment"].astype(float)

    model = sm.WLS(y, X, weights=w)
    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["sector_fe"].values},
    )

    # Extract coefficients
    event_coefs = []
    for t in time_indices:
        col = f"bite_x_t{t}"
        if t == ref_idx:
            event_coefs.append({
                "time_idx": t,
                "beta": 0.0,
                "se": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
            })
        elif col in results.params.index:
            event_coefs.append({
                "time_idx": t,
                "beta": results.params[col],
                "se": results.bse[col],
                "ci_low": results.conf_int().loc[col, 0],
                "ci_high": results.conf_int().loc[col, 1],
            })

    event_df = pd.DataFrame(event_coefs)
    # Map time_idx to year-quarter labels
    idx_to_label = {}
    for yr in [2022, 2023, 2024]:
        for q_i, q in enumerate([1, 2, 3, 4], start=0):
            idx = (yr - 2022) * 4 + q_i
            idx_to_label[idx] = f"{yr}q{q}"
    event_df["year_quarter"] = event_df["time_idx"].map(idx_to_label)

    print(f"\n  Event study: {outcome_label}")
    print(event_df[["year_quarter", "beta", "se", "ci_low", "ci_high"]].to_string(index=False))

    return event_df


# ──────────────────────────────────────────────────────────────────────
# 7. PLOTTING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

# Color palette
COLOR_TREATED = "#D62728"   # red
COLOR_CONTROL = "#1F77B4"   # blue
COLOR_COEF = "#2C3E50"      # dark navy
COLOR_CI = "#BDC3C7"        # light gray


def plot_event_study(event_df, outcome_label, filename_stem):
    """Plot event study coefficients with 95% CI bands."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = event_df["time_idx"].values
    y = event_df["beta"].values
    ci_lo = event_df["ci_low"].values
    ci_hi = event_df["ci_high"].values

    # Shade post-treatment period
    ax.axvspan(3.5, x.max() + 0.5, alpha=0.08, color=COLOR_TREATED,
               label="Post-treatment (2023+)")

    # Reference line
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)

    # Vertical line at treatment onset
    ax.axvline(x=3.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    # CI band
    ax.fill_between(x, ci_lo, ci_hi, alpha=0.25, color=COLOR_COEF)

    # Point estimates
    ax.plot(x, y, "o-", color=COLOR_COEF, markersize=6, linewidth=1.5, zorder=5)

    # Mark reference period
    ref_idx = 3
    if ref_idx in x:
        ax.plot(ref_idx, 0, "D", color=COLOR_TREATED, markersize=8, zorder=6,
                label="Reference (2022q4)")

    # Labels
    labels = event_df["year_quarter"].values
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Quarter")
    ax.set_ylabel(f"Coefficient (MW bite interaction)")
    ax.set_title(f"Event Study: Effect of MW Shock on {outcome_label}")
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    # Add annotation
    ax.annotate(
        "MW +16%\n(Jan 2023)",
        xy=(4, y[x == 4][0] if 4 in x else 0),
        xytext=(5.5, max(ci_hi) * 0.8),
        fontsize=9, ha="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(IMG_DIR / f"{filename_stem}.{ext}")
    plt.close(fig)
    print(f"  Saved: {filename_stem}.png/pdf")


def plot_parallel_trends(cells, outcome, outcome_label, filename_stem):
    """Plot mean outcome over time by treatment group (high vs low MW bite)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for hb, label, color, marker in [
        (1, "High MW bite (treated)", COLOR_TREATED, "s"),
        (0, "Low MW bite (control)", COLOR_CONTROL, "o"),
    ]:
        group = cells[cells["high_bite"] == hb].copy()
        ts = group.groupby("year_quarter").apply(
            lambda g: np.average(g[outcome].dropna(),
                                 weights=g.loc[g[outcome].notna(), "total_employment"])
            if g[outcome].notna().sum() > 0 else np.nan
        ).reset_index()
        ts.columns = ["year_quarter", "mean"]

        # Sort by time
        order = ["2022q1", "2022q2", "2022q3", "2022q4",
                 "2023q1", "2023q2", "2023q3", "2023q4",
                 "2024q1", "2024q2", "2024q3", "2024q4"]
        ts["sort_key"] = ts["year_quarter"].map(
            {v: i for i, v in enumerate(order)}
        )
        ts = ts.sort_values("sort_key")

        ax.plot(ts["sort_key"], ts["mean"], f"{marker}-", color=color,
                label=label, markersize=6, linewidth=1.5)

    # Treatment onset line
    ax.axvline(x=3.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(3.6, ax.get_ylim()[1] * 0.98, "MW +16%", fontsize=9,
            va="top", color="gray")

    # Shade post period
    ax.axvspan(3.5, 11.5, alpha=0.06, color=COLOR_TREATED)

    order_present = sorted(cells["year_quarter"].unique(),
                           key=lambda x: (int(x[:4]), int(x[-1])))
    order_map = {v: i for i, v in enumerate(order)}
    ticks = [order_map[yq] for yq in order_present if yq in order_map]
    tick_labels = [yq for yq in order_present if yq in order_map]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_xlabel("Quarter")
    ax.set_ylabel(outcome_label)
    ax.set_title(f"Parallel Trends: {outcome_label} by MW Bite Group")
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(IMG_DIR / f"{filename_stem}.{ext}")
    plt.close(fig)
    print(f"  Saved: {filename_stem}.png/pdf")


def plot_mw_bite_distribution(bite_df, filename_stem):
    """Plot histogram of sector-level MW bite in 2022."""
    fig, ax = plt.subplots(figsize=(7, 5))

    bite_vals = bite_df["sector_mw_bite_2022"].dropna()
    median_val = bite_vals.median()

    ax.hist(bite_vals, bins=20, color=COLOR_COEF, alpha=0.7, edgecolor="white")
    ax.axvline(x=median_val, color=COLOR_TREATED, linewidth=2, linestyle="--",
               label=f"Median = {median_val:.2f}")
    ax.set_xlabel("Minimum Wage Bite (share earning $\\leq$ 1.2 SMMLV, 2022)")
    ax.set_ylabel("Number of Sectors")
    ax.set_title("Distribution of Sector-Level MW Bite (Pre-Treatment 2022)")
    ax.legend(frameon=True)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(IMG_DIR / f"{filename_stem}.{ext}")
    plt.close(fig)
    print(f"  Saved: {filename_stem}.png/pdf")


def plot_scatter_bite_vs_formality(cells, bite_df, filename_stem):
    """Scatter: sector MW bite vs change in formality (2022 -> 2024)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Compute mean formality by sector and year
    sector_year = cells.groupby(["sector_2d", "year"]).apply(
        lambda g: np.average(g["formality_rate"].dropna(),
                             weights=g.loc[g["formality_rate"].notna(),
                                           "total_employment"])
        if g["formality_rate"].notna().sum() > 0 else np.nan
    ).reset_index()
    sector_year.columns = ["sector_2d", "year", "formality"]

    # Pivot to get 2022 and 2024 columns
    pivot = sector_year.pivot(index="sector_2d", columns="year",
                              values="formality")
    if 2022 in pivot.columns and 2024 in pivot.columns:
        pivot["delta_formality"] = pivot[2024] - pivot[2022]
    elif 2022 in pivot.columns and 2023 in pivot.columns:
        pivot["delta_formality"] = pivot[2023] - pivot[2022]
    else:
        print("  WARNING: Cannot compute formality change, skipping scatter.")
        plt.close(fig)
        return

    pivot = pivot.reset_index().merge(bite_df, on="sector_2d")
    pivot = pivot.dropna(subset=["delta_formality", "sector_mw_bite_2022"])

    # Color by treatment group
    for hb, label, color, marker in [
        (1, "High MW bite", COLOR_TREATED, "s"),
        (0, "Low MW bite", COLOR_CONTROL, "o"),
    ]:
        mask = pivot["high_bite"] == hb
        ax.scatter(
            pivot.loc[mask, "sector_mw_bite_2022"],
            pivot.loc[mask, "delta_formality"],
            c=color, marker=marker, s=60, alpha=0.8, edgecolors="white",
            linewidths=0.5, label=label, zorder=5,
        )

    # Fit line
    x_all = pivot["sector_mw_bite_2022"].values
    y_all = pivot["delta_formality"].values
    if len(x_all) > 2:
        slope, intercept, r, p, se = stats.linregress(x_all, y_all)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, "--", color="gray",
                linewidth=1, alpha=0.7,
                label=f"OLS: slope={slope:.3f} (p={p:.3f})")

    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="-", alpha=0.5)
    ax.set_xlabel("Sector MW Bite (2022)")
    last_year = 2024 if 2024 in sector_year["year"].values else 2023
    ax.set_ylabel(f"Change in Formality Rate (2022 to {last_year})")
    ax.set_title("MW Bite vs. Formality Change by Sector")
    ax.legend(frameon=True, loc="best")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(IMG_DIR / f"{filename_stem}.{ext}")
    plt.close(fig)
    print(f"  Saved: {filename_stem}.png/pdf")


# ──────────────────────────────────────────────────────────────────────
# 8. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def main():
    """Execute the full DiD analysis pipeline."""
    set_publication_style()

    print("=" * 70)
    print("DiD ANALYSIS: MINIMUM WAGE SHOCKS, AUTOMATION & FORMALITY")
    print("GEIH 2022-2024 (quarterly samples)")
    print("=" * 70)

    # ── Step 1: Load data ──
    print("\n[STEP 1] Loading GEIH microdata...")
    df = load_all_geih()

    # ── Step 2: Construct variables ──
    print("\n[STEP 2] Constructing analysis variables...")
    df = construct_variables(df)

    # ── Step 3: Compute sector MW bite ──
    print("\n[STEP 3] Computing sector-level MW bite...")
    bite_df = compute_sector_mw_bite(df)

    # ── Step 4: Aggregate to cells ──
    print("\n[STEP 4] Aggregating to sector-year-quarter cells...")
    cells, df_merged = aggregate_to_cells(df, bite_df)

    # Save cell-level dataset
    cells_path = DATA_DIR / "geih_did_dataset.csv"
    cells.to_csv(cells_path, index=False)
    print(f"\n  Saved cell-level dataset: {cells_path}")

    # ── Step 5: DiD estimation ──
    print("\n[STEP 5] Running DiD regressions...")
    print("=" * 50)

    results = []
    outcomes = [
        ("formality_rate", "Formality Rate"),
        ("mean_automation_risk", "Mean Automation Risk"),
        ("share_high_risk", "Share High-Risk Workers"),
        ("mean_log_income", "Mean Log Income"),
        ("total_employment", "Total Employment"),
    ]

    # 5a. Cell-level DiD
    print("\n--- Cell-level DiD (main specification) ---")
    for var, label in outcomes:
        try:
            res = run_cell_did(cells, var, label)
            results.append(res)
        except Exception as e:
            print(f"  ERROR in cell DiD for {label}: {e}")

    # 5b. Individual-level DiD (robustness)
    print("\n--- Individual-level DiD (robustness) ---")
    individual_outcomes = [
        ("formal", "Formality (Individual)"),
        ("automation_risk", "Automation Risk (Individual)"),
        ("high_risk", "High Risk (Individual)"),
    ]
    for var, label in individual_outcomes:
        try:
            res = run_individual_did(df_merged, var, label)
            if res:
                results.append(res)
        except Exception as e:
            print(f"  ERROR in individual DiD for {label}: {e}")

    # Save results
    results_df = pd.DataFrame(results)
    results_path = DATA_DIR / "geih_did_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Saved regression results: {results_path}")
    print("\n  Results summary:")
    print(results_df.to_string(index=False))

    # ── Step 6: Event studies ──
    print("\n[STEP 6] Running event study regressions...")
    print("=" * 50)

    try:
        event_formality = run_event_study(cells, "formality_rate", "Formality Rate")
    except Exception as e:
        print(f"  ERROR in event study (formality): {e}")
        event_formality = None

    try:
        event_automation = run_event_study(cells, "mean_automation_risk",
                                           "Mean Automation Risk")
    except Exception as e:
        print(f"  ERROR in event study (automation): {e}")
        event_automation = None

    # ── Step 7: Figures ──
    print("\n[STEP 7] Generating publication-quality figures...")
    print("=" * 50)

    # (a) Event study: formality rate
    if event_formality is not None:
        plot_event_study(event_formality, "Formality Rate",
                         "fig_did_event_formality")

    # (b) Event study: automation risk
    if event_automation is not None:
        plot_event_study(event_automation, "Mean Automation Risk",
                         "fig_did_event_automation")

    # (c) Parallel trends: formality
    plot_parallel_trends(cells, "formality_rate", "Formality Rate",
                         "fig_did_parallel_trends")

    # (d) Distribution of MW bite
    plot_mw_bite_distribution(bite_df, "fig_did_mw_bite_distribution")

    # (e) Scatter: MW bite vs formality change
    plot_scatter_bite_vs_formality(cells, bite_df,
                                   "fig_did_scatter_bite_formality")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Data:    {cells_path}")
    print(f"  Results: {results_path}")
    print(f"  Figures: {IMG_DIR}/fig_did_*.png/pdf")
    print()

    # Print key finding
    if len(results) > 0:
        formality_res = [r for r in results
                         if r["outcome"] == "Formality Rate"
                         and r["model"] == "cell_did"]
        if formality_res:
            r = formality_res[0]
            direction = "decrease" if r["beta"] < 0 else "increase"
            sig = "significant" if r["p_value"] < 0.05 else "not significant"
            print(f"  KEY FINDING: MW shock is associated with a {direction} "
                  f"in formality ({sig}, p={r['p_value']:.3f})")
            print(f"    DiD coefficient: {r['beta']:.4f} "
                  f"(95% CI: [{r['ci_low']:.4f}, {r['ci_high']:.4f}])")


if __name__ == "__main__":
    main()
