#!/usr/bin/env python3
"""
06_regenerate_figures.py
========================
Regenerates ALL 21 figures used in the paper in both English and Spanish.
Output: images/en/*.pdf and images/es/*.pdf
Style: Q1 journal quality (Nature / Econometrica style)
"""

import os
import sys
import warnings
import traceback

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(BASE_DIR, "images")
for lang in ("en", "es"):
    os.makedirs(os.path.join(IMG_DIR, lang), exist_ok=True)

# ---------------------------------------------------------------------------
# Q1 Journal Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.5,
    "axes.grid": False,
    "grid.linewidth": 0.3,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.5,
})

# Professional muted palette
COLORS = {
    "primary": "#2C3E50",
    "secondary": "#E74C3C",
    "tertiary": "#3498DB",
    "quaternary": "#27AE60",
    "quinary": "#F39C12",
    "light_gray": "#BDC3C7",
    "dark_gray": "#7F8C8D",
}
PALETTE = [
    "#2C3E50", "#E74C3C", "#3498DB", "#27AE60", "#F39C12",
    "#8E44AD", "#1ABC9C", "#D35400", "#7F8C8D",
]

# ---------------------------------------------------------------------------
# Single-column and double-column sizes
# ---------------------------------------------------------------------------
SINGLE = (3.5, 2.5)
DOUBLE = (7.0, 4.0)
SINGLE_TALL = (3.5, 4.0)
DOUBLE_TALL = (7.0, 5.5)

# ---------------------------------------------------------------------------
# Translation dictionaries
# ---------------------------------------------------------------------------
LABELS = {
    "en": {
        "robot_density": "Robot density (per 10,000 mfg. workers)",
        "gdp_per_worker": "GDP per worker (log, PPP USD)",
        "year": "Year",
        "robot_installations": "Annual robot installations",
        "country": "Country",
        "coefficient": "Coefficient",
        "variable": "Variable",
        "log_robots": "log(Robot installations)",
        "log_productivity": "log(Productivity)",
        "rd_gdp_pct": "R&D (% GDP)",
        "log_gdp_per_worker": "log(GDP per worker)",
        "pooled_ols": "Pooled OLS",
        "fixed_effects": "Fixed Effects",
        "random_effects": "Random Effects",
        "dynamic_ols": "Dynamic OLS",
        "correlation": "Correlation",
        # Sectoral
        "sector": "Sector",
        "avi": "Automation Vulnerability Index",
        "labor_share": "Labor share (normalized)",
        "prod_growth": "Productivity growth (normalized)",
        "cap_intensity": "Capital intensity (normalized)",
        "vulnerability_index": "Vulnerability index",
        "productivity_growth": "Productivity growth (norm.)",
        "capital_intensity": "Capital intensity (norm.)",
        "labor_share_norm": "Labor share (norm.)",
        # Automation
        "automation_prob": "Automation probability",
        "high_risk_pct": "High-risk share (%)",
        "fpr": "False positive rate",
        "tpr": "True positive rate",
        "odds_ratio": "Odds ratio",
        "education_level": "Education level",
        "formal": "Formal",
        "informal": "Informal",
        "mean_prob": "Mean automation probability",
        "count": "Workers (thousands)",
        "roc_auc": "AUC = {:.3f}",
        "high_risk": "High risk",
        "low_risk": "Low risk",
        "edu_none": "None",
        "edu_primary": "Primary",
        "edu_secondary": "Secondary",
        "edu_media": "Upper sec.",
        "edu_technical": "Technical",
        "edu_university": "University",
        "edu_postgrad": "Postgrad.",
        # Firm-level
        "log_ulc": "log(Unit labor cost)",
        "log_lp": "log(Labor productivity)",
        "log_emp": "log(Employment)",
        "log_lcpw": "log(Labor cost/worker)",
        "investment_rate": "Investment rate",
        "labor_cost_pw": "Labor cost per worker (COP M)",
        "employment": "Employment (thousands)",
        "model_a": "Model A: log(VA)",
        "model_b": "Model B: Investment rate",
        "model_c": "Model C: log(Inv. total)",
        "automation_proxy": "Automation investment (COP M)",
        # Simulation
        "formal_employment": "Formal employment (millions)",
        "displacement": "Cumulative displacement (millions)",
        "net_change": "Net change (millions)",
        "scenario": "Scenario",
        "status_quo": "Status Quo",
        "labor_reform": "Labor Reform",
        "parafiscal_reform": "Parafiscal Reform",
        "ai_acceleration": "AI Acceleration",
        "combined_worst": "Combined Worst",
        "parameter": "Parameter",
        "low_value": "Low",
        "high_value": "High",
        "central": "Central",
        "pct_change": "Change (%)",
        "displaced_m": "Displaced (millions)",
        "p5_p95": "90% CI",
        "median": "Median",
        "incremental": "Incremental displacement (millions)",
        "base_displacement": "Base (Status Quo)",
        "reform_effect": "Reform effect",
        "total_labor_reform": "Total (Labor Reform)",
    },
    "es": {
        "robot_density": "Densidad rob\u00f3tica (por 10.000 trab. manuf.)",
        "gdp_per_worker": "PIB por trabajador (log, USD PPA)",
        "year": "A\u00f1o",
        "robot_installations": "Instalaciones anuales de robots",
        "country": "Pa\u00eds",
        "coefficient": "Coeficiente",
        "variable": "Variable",
        "log_robots": "log(Instalaciones de robots)",
        "log_productivity": "log(Productividad)",
        "rd_gdp_pct": "I+D (% PIB)",
        "log_gdp_per_worker": "log(PIB por trabajador)",
        "pooled_ols": "MCO agrupado",
        "fixed_effects": "Efectos fijos",
        "random_effects": "Efectos aleatorios",
        "dynamic_ols": "MCO din\u00e1mico",
        "correlation": "Correlaci\u00f3n",
        # Sectoral
        "sector": "Sector",
        "avi": "\u00cdndice de vulnerabilidad a la automatizaci\u00f3n",
        "labor_share": "Participaci\u00f3n laboral (normalizada)",
        "prod_growth": "Crecimiento productividad (normalizado)",
        "cap_intensity": "Intensidad de capital (normalizada)",
        "vulnerability_index": "\u00cdndice de vulnerabilidad",
        "productivity_growth": "Crecimiento productividad (norm.)",
        "capital_intensity": "Intensidad de capital (norm.)",
        "labor_share_norm": "Participaci\u00f3n laboral (norm.)",
        # Automation
        "automation_prob": "Probabilidad de automatizaci\u00f3n",
        "high_risk_pct": "Proporci\u00f3n alto riesgo (%)",
        "fpr": "Tasa de falsos positivos",
        "tpr": "Tasa de verdaderos positivos",
        "odds_ratio": "Raz\u00f3n de momios",
        "education_level": "Nivel educativo",
        "formal": "Formal",
        "informal": "Informal",
        "mean_prob": "Probabilidad media de automatizaci\u00f3n",
        "count": "Trabajadores (miles)",
        "roc_auc": "AUC = {:.3f}",
        "high_risk": "Alto riesgo",
        "low_risk": "Bajo riesgo",
        "edu_none": "Ninguno",
        "edu_primary": "Primaria",
        "edu_secondary": "Secundaria",
        "edu_media": "Media",
        "edu_technical": "T\u00e9cnico",
        "edu_university": "Universitario",
        "edu_postgrad": "Posgrado",
        # Firm-level
        "log_ulc": "log(Costo laboral unitario)",
        "log_lp": "log(Productividad laboral)",
        "log_emp": "log(Empleo)",
        "log_lcpw": "log(Costo laboral/trabajador)",
        "investment_rate": "Tasa de inversi\u00f3n",
        "labor_cost_pw": "Costo laboral por trabajador (COP M)",
        "employment": "Empleo (miles)",
        "model_a": "Modelo A: log(VA)",
        "model_b": "Modelo B: Tasa de inversi\u00f3n",
        "model_c": "Modelo C: log(Inv. total)",
        "automation_proxy": "Inversi\u00f3n en automatizaci\u00f3n (COP M)",
        # Simulation
        "formal_employment": "Empleo formal (millones)",
        "displacement": "Desplazamiento acumulado (millones)",
        "net_change": "Cambio neto (millones)",
        "scenario": "Escenario",
        "status_quo": "Statu Quo",
        "labor_reform": "Reforma laboral",
        "parafiscal_reform": "Reforma parafiscal",
        "ai_acceleration": "Aceleraci\u00f3n IA",
        "combined_worst": "Peor combinado",
        "parameter": "Par\u00e1metro",
        "low_value": "Bajo",
        "high_value": "Alto",
        "central": "Central",
        "pct_change": "Cambio (%)",
        "displaced_m": "Desplazados (millones)",
        "p5_p95": "IC 90%",
        "median": "Mediana",
        "incremental": "Desplazamiento incremental (millones)",
        "base_displacement": "Base (Statu Quo)",
        "reform_effect": "Efecto reforma",
        "total_labor_reform": "Total (Reforma laboral)",
    },
}

SCENARIO_NAMES = {
    "en": {
        "Status Quo": "Status Quo",
        "Labor Reform": "Labor Reform",
        "Parafiscal Reform": "Parafiscal Reform",
        "AI Acceleration": "AI Acceleration",
        "Combined Worst": "Combined Worst",
    },
    "es": {
        "Status Quo": "Statu Quo",
        "Labor Reform": "Reforma laboral",
        "Parafiscal Reform": "Reforma parafiscal",
        "AI Acceleration": "Aceleraci\u00f3n IA",
        "Combined Worst": "Peor combinado",
    },
}

SECTOR_NAMES = {
    "en": {
        "Agriculture": "Agriculture",
        "Manufacturing": "Manufacturing",
        "Construction": "Construction",
        "Commerce/Transport": "Commerce/Transport",
        "Public Admin/Educ/Health": "Public Admin/Educ/Health",
        "Financial Services": "Financial Services",
        "BPO/Professional": "BPO/Professional",
        "Mining": "Mining",
        "Other Services": "Other Services",
    },
    "es": {
        "Agriculture": "Agricultura",
        "Manufacturing": "Manufactura",
        "Construction": "Construcci\u00f3n",
        "Commerce/Transport": "Comercio/Transporte",
        "Public Admin/Educ/Health": "Adm. p\u00fablica/Educ./Salud",
        "Financial Services": "Servicios financieros",
        "BPO/Professional": "BPO/Profesional",
        "Mining": "Miner\u00eda",
        "Other Services": "Otros servicios",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def L(key, lang):
    """Get label by key and language."""
    return LABELS[lang].get(key, key)


def savefig(fig, name, lang):
    path = os.path.join(IMG_DIR, lang, f"{name}.pdf")
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  Saved: {path}")


def despine(ax):
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
_cache = {}


def load(name, **kwargs):
    # Build a hashable cache key from kwargs
    def _make_hashable(v):
        if isinstance(v, list):
            return tuple(v)
        return v
    cache_key = (name, tuple(sorted((k, _make_hashable(v)) for k, v in kwargs.items())) if kwargs else ())
    if cache_key not in _cache:
        path = os.path.join(DATA_DIR, name)
        _cache[cache_key] = pd.read_csv(path, **kwargs)
    return _cache[cache_key].copy()


# ===================================================================
# FIGURE FUNCTIONS
# ===================================================================

# -------------------------------------------------------------------
# 1. fig1_productivity_vs_robots
# -------------------------------------------------------------------
def fig1_productivity_vs_robots(lang="en"):
    df = load("international_panel.csv")
    # Use latest year per country
    latest = df.groupby("country")["year"].max().reset_index()
    df_latest = df.merge(latest, on=["country", "year"])
    df_latest = df_latest.dropna(subset=["log_robots", "log_gdp_per_worker"])

    fig, ax = plt.subplots(figsize=SINGLE)
    ax.scatter(
        df_latest["log_robots"],
        df_latest["log_gdp_per_worker"],
        s=15, alpha=0.7, color=COLORS["primary"],
        edgecolors="white", linewidths=0.3, zorder=3,
    )
    # Fit line
    mask = np.isfinite(df_latest["log_robots"]) & np.isfinite(df_latest["log_gdp_per_worker"])
    z = np.polyfit(df_latest.loc[mask, "log_robots"], df_latest.loc[mask, "log_gdp_per_worker"], 1)
    p = np.poly1d(z)
    xr = np.linspace(df_latest["log_robots"].min(), df_latest["log_robots"].max(), 100)
    ax.plot(xr, p(xr), color=COLORS["secondary"], linewidth=0.8, linestyle="--", zorder=2)

    # Label Colombia
    col = df_latest[df_latest["country"] == "Colombia"]
    if not col.empty:
        ax.annotate(
            "Colombia", xy=(col["log_robots"].iloc[0], col["log_gdp_per_worker"].iloc[0]),
            fontsize=6, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", lw=0.4, color=COLORS["dark_gray"]),
        )

    # Label a few key countries
    for c in ["United States", "South Korea", "Germany", "China", "Japan", "Brazil", "Mexico"]:
        row = df_latest[df_latest["country"] == c]
        if not row.empty:
            ax.annotate(
                c if lang == "en" else c, xy=(row["log_robots"].iloc[0], row["log_gdp_per_worker"].iloc[0]),
                fontsize=5, ha="left", va="bottom", color=COLORS["dark_gray"],
                xytext=(3, 2), textcoords="offset points",
            )

    ax.set_xlabel(L("log_robots", lang))
    ax.set_ylabel(L("gdp_per_worker", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig1_productivity_vs_robots", lang)


# -------------------------------------------------------------------
# 2. fig2_robot_timeseries
# -------------------------------------------------------------------
def fig2_robot_timeseries(lang="en"):
    df = load("international_panel.csv")
    # Select key countries
    countries = ["South Korea", "Germany", "Japan", "United States", "China", "Colombia", "Brazil", "Mexico"]
    sub = df[df["country"].isin(countries)].copy()
    sub = sub.sort_values("year")

    fig, ax = plt.subplots(figsize=SINGLE)
    for i, c in enumerate(countries):
        d = sub[sub["country"] == c]
        lw = 1.2 if c == "Colombia" else 0.7
        ls = "-" if c == "Colombia" else "-"
        ax.plot(d["year"], d["robot_installations"], label=c,
                color=PALETTE[i % len(PALETTE)], linewidth=lw, linestyle=ls)

    ax.set_xlabel(L("year", lang))
    ax.set_ylabel(L("robot_installations", lang))
    ax.legend(fontsize=5.5, ncol=2, loc="upper left", frameon=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig2_robot_timeseries", lang)


# -------------------------------------------------------------------
# 3. fig3_coefficient_plot (international panel FE model)
# -------------------------------------------------------------------
def fig3_coefficient_plot(lang="en"):
    # Parse regression_results.csv
    reg = load("regression_results.csv")
    reg.columns = ["Variable", "Pooled OLS", "Fixed Effects", "Random Effects", "Dynamic OLS"]

    # Extract coefficients and SEs for Fixed Effects model
    vars_of_interest = ["log_productivity", "rd_gdp_pct", "log_gdp_per_worker"]
    var_labels = {
        "log_productivity": L("log_productivity", lang),
        "rd_gdp_pct": L("rd_gdp_pct", lang),
        "log_gdp_per_worker": L("log_gdp_per_worker", lang),
    }

    coefs = []
    ses = []
    labels = []
    for v in vars_of_interest:
        row_coef = reg[reg["Variable"] == v]
        row_se = reg[reg["Variable"] == f"{v}_se"]
        if not row_coef.empty and not row_se.empty:
            c = float(row_coef["Fixed Effects"].iloc[0])
            se_str = str(row_se["Fixed Effects"].iloc[0])
            # Parse SE from format like "(0.4007)"
            se_val = float(se_str.replace("(", "").replace(")", "").replace("*", ""))
            coefs.append(c)
            ses.append(se_val)
            labels.append(var_labels.get(v, v))

    coefs = np.array(coefs)
    ses = np.array(ses)

    fig, ax = plt.subplots(figsize=SINGLE)
    y_pos = np.arange(len(labels))
    ax.errorbar(coefs, y_pos, xerr=1.96 * ses, fmt="o", color=COLORS["primary"],
                markersize=4, capsize=2, capthick=0.5, elinewidth=0.7)
    ax.axvline(0, color=COLORS["light_gray"], linewidth=0.5, linestyle="--", zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(L("coefficient", lang))
    ax.invert_yaxis()
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig3_coefficient_plot", lang)


# -------------------------------------------------------------------
# 5. fig5_robot_density_comparison
# -------------------------------------------------------------------
def fig5_robot_density_comparison(lang="en"):
    df = load("international_panel.csv")
    # Latest year, compute robot density as robots_per_gdp or use robot_installations/employment
    latest = df.groupby("country")["year"].max().reset_index()
    df_latest = df.merge(latest, on=["country", "year"])
    # Robot density: robots per 10k manufacturing workers ~ use robot_installations / (employment_millions * 1000) * 10000
    # Approximate: robot_installations / employment_millions
    df_latest["robot_density"] = df_latest["robot_installations"] / df_latest["employment_millions"]
    df_latest = df_latest.dropna(subset=["robot_density"]).sort_values("robot_density")

    # Take top 20 + Colombia
    top20 = df_latest.nlargest(20, "robot_density")
    col = df_latest[df_latest["country"] == "Colombia"]
    show = pd.concat([top20, col]).drop_duplicates(subset=["country"]).sort_values("robot_density")

    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    colors_bar = [COLORS["secondary"] if c == "Colombia" else COLORS["primary"]
                  for c in show["country"]]
    ax.barh(range(len(show)), show["robot_density"], color=colors_bar, height=0.7, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show["country"], fontsize=6)
    ax.set_xlabel(L("robot_density", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig5_robot_density_comparison", lang)


# -------------------------------------------------------------------
# 6. fig6_correlation_matrix
# -------------------------------------------------------------------
def fig6_correlation_matrix(lang="en"):
    df = load("international_panel.csv")
    cols = ["log_robots", "log_gdp_per_worker", "rd_gdp_pct", "tfp", "labor_share", "human_capital"]
    col_labels = {
        "log_robots": L("log_robots", lang),
        "log_gdp_per_worker": L("log_gdp_per_worker", lang),
        "rd_gdp_pct": L("rd_gdp_pct", lang),
        "tfp": "TFP",
        "labor_share": L("labor_share", lang),
        "human_capital": "Human capital" if lang == "en" else "Capital humano",
    }
    sub = df[cols].dropna()
    corr = sub.corr()
    labels_list = [col_labels.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 6},
        xticklabels=labels_list, yticklabels=labels_list,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.7, "label": L("correlation", lang)},
        ax=ax, vmin=-1, vmax=1,
        square=True,
    )
    ax.tick_params(axis="both", labelsize=5.5, rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    fig.tight_layout()
    savefig(fig, "fig6_correlation_matrix", lang)


# -------------------------------------------------------------------
# 7. fig_bar_vulnerability_ranking
# -------------------------------------------------------------------
def fig_bar_vulnerability_ranking(lang="en"):
    df = load("sectoral_analysis_results.csv", encoding="utf-8-sig")
    df = df.sort_values("indice_vulnerabilidad", ascending=True)

    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    colors_bar = [COLORS["secondary"] if v > 60 else COLORS["primary"]
                  for v in df["indice_vulnerabilidad"]]
    ax.barh(range(len(df)), df["indice_vulnerabilidad"], color=colors_bar,
            height=0.7, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["sector"], fontsize=6)
    ax.set_xlabel(L("avi", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_bar_vulnerability_ranking", lang)


# -------------------------------------------------------------------
# 8. fig_heatmap_sectoral_indicators
# -------------------------------------------------------------------
def fig_heatmap_sectoral_indicators(lang="en"):
    df = load("sectoral_analysis_results.csv", encoding="utf-8-sig")
    cols = ["labor_share_norm", "prod_growth_norm", "cap_intensity_norm"]
    col_labels = [L("labor_share_norm", lang), L("productivity_growth", lang), L("capital_intensity", lang)]

    mat = df[cols].values
    sectors = df["sector"].tolist()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(
        mat, cmap=cmap, annot=True, fmt=".0f", annot_kws={"size": 6},
        xticklabels=col_labels, yticklabels=sectors,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="x", labelsize=6)
    plt.setp(ax.get_xticklabels(), ha="right", rotation=30, rotation_mode="anchor")
    fig.tight_layout()
    savefig(fig, "fig_heatmap_sectoral_indicators", lang)


# -------------------------------------------------------------------
# 9. fig_scatter_productivity_vs_laborshare
# -------------------------------------------------------------------
def fig_scatter_productivity_vs_laborshare(lang="en"):
    df = load("sectoral_analysis_results.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=SINGLE)
    ax.scatter(
        df["prod_growth_norm"], df["labor_share_norm"],
        s=df["va_share_pct_2024"] * 8,  # size by VA share
        alpha=0.7, color=COLORS["tertiary"], edgecolors=COLORS["primary"], linewidths=0.4,
        zorder=3,
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["sector"], xy=(row["prod_growth_norm"], row["labor_share_norm"]),
            fontsize=5, ha="center", va="bottom",
            xytext=(0, 3), textcoords="offset points",
        )
    ax.set_xlabel(L("productivity_growth", lang))
    ax.set_ylabel(L("labor_share_norm", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_scatter_productivity_vs_laborshare", lang)


# -------------------------------------------------------------------
# 10. fig_roc_curve
# -------------------------------------------------------------------
def fig_roc_curve(lang="en"):
    """Reconstruct ROC from automation_analysis_dataset (logit: high_risk ~ features).
    We use a simplified approach: compute predicted probabilities from automation_prob
    and binary high_risk outcome with different thresholds."""
    df = load("automation_analysis_dataset.csv", usecols=["automation_prob", "high_risk"])
    df = df.dropna()

    # Use automation_prob as predicted probability for high_risk
    y_true = df["high_risk"].values
    y_score = df["automation_prob"].values

    # Compute ROC manually
    thresholds = np.linspace(0, 1, 200)
    tprs, fprs = [], []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)

    # AUC by trapezoidal rule (sort by fpr)
    fprs, tprs = np.array(fprs), np.array(tprs)
    sort_idx = np.argsort(fprs)
    fprs_s, tprs_s = fprs[sort_idx], tprs[sort_idx]
    auc_val = np.trapz(tprs_s, fprs_s)

    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(fprs_s, tprs_s, color=COLORS["primary"], linewidth=1.0, zorder=3)
    ax.plot([0, 1], [0, 1], color=COLORS["light_gray"], linewidth=0.5, linestyle="--", zorder=1)
    ax.fill_between(fprs_s, tprs_s, alpha=0.08, color=COLORS["tertiary"])
    ax.text(0.6, 0.2, L("roc_auc", lang).format(auc_val), fontsize=7,
            transform=ax.transAxes, ha="center")
    ax.set_xlabel(L("fpr", lang))
    ax.set_ylabel(L("tpr", lang))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_roc_curve", lang)


# -------------------------------------------------------------------
# 11. fig_forest_plot_odds_ratios
# -------------------------------------------------------------------
def fig_forest_plot_odds_ratios(lang="en"):
    df = load("automation_odds_ratios.csv")
    # Exclude const and sector dummies, focus on individual-level
    individual_vars = ["formal", "female", "age", "age_sq", "log_income", "hours_worked",
                       "edu_Ninguno", "edu_Media", "edu_Tecnico_Tecnologico", "edu_Universitario", "edu_Posgrado",
                       "firm_Micro_1", "firm_Pequena_11-50", "firm_Mediana_51-200", "firm_Grande_201plus"]

    var_labels_en = {
        "formal": "Formal", "female": "Female", "age": "Age", "age_sq": "Age\u00b2",
        "log_income": "log(Income)", "hours_worked": "Hours worked",
        "edu_Ninguno": "Educ.: None", "edu_Media": "Educ.: Upper sec.",
        "edu_Tecnico_Tecnologico": "Educ.: Technical", "edu_Universitario": "Educ.: University",
        "edu_Posgrado": "Educ.: Postgrad.",
        "firm_Micro_1": "Firm: Micro(1)", "firm_Pequena_11-50": "Firm: Small",
        "firm_Mediana_51-200": "Firm: Medium", "firm_Grande_201plus": "Firm: Large",
    }
    var_labels_es = {
        "formal": "Formal", "female": "Mujer", "age": "Edad", "age_sq": "Edad\u00b2",
        "log_income": "log(Ingreso)", "hours_worked": "Horas trabajadas",
        "edu_Ninguno": "Educ.: Ninguno", "edu_Media": "Educ.: Media",
        "edu_Tecnico_Tecnologico": "Educ.: T\u00e9cnico", "edu_Universitario": "Educ.: Universitario",
        "edu_Posgrado": "Educ.: Posgrado",
        "firm_Micro_1": "Firma: Micro(1)", "firm_Pequena_11-50": "Firma: Peque\u00f1a",
        "firm_Mediana_51-200": "Firma: Mediana", "firm_Grande_201plus": "Firma: Grande",
    }
    vl = var_labels_en if lang == "en" else var_labels_es

    sub = df[df["Variable"].isin(individual_vars)].copy()
    # Preserve order
    sub["order"] = sub["Variable"].apply(lambda x: individual_vars.index(x) if x in individual_vars else 99)
    sub = sub.sort_values("order", ascending=False)

    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    y_pos = np.arange(len(sub))
    colors_dots = [COLORS["secondary"] if p < 0.05 else COLORS["light_gray"]
                   for p in sub["p_value"]]

    ax.errorbar(
        sub["OR"], y_pos,
        xerr=[sub["OR"] - sub["OR_CI_low"], sub["OR_CI_high"] - sub["OR"]],
        fmt="none", ecolor=COLORS["dark_gray"], elinewidth=0.5, capsize=1.5, capthick=0.4,
    )
    ax.scatter(sub["OR"], y_pos, c=colors_dots, s=18, zorder=3, edgecolors="white", linewidths=0.3)
    ax.axvline(1, color=COLORS["light_gray"], linewidth=0.5, linestyle="--", zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([vl.get(v, v) for v in sub["Variable"]], fontsize=6)
    ax.set_xlabel(L("odds_ratio", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_forest_plot_odds_ratios", lang)


# -------------------------------------------------------------------
# 12. fig_automation_risk_by_sector
# -------------------------------------------------------------------
def fig_automation_risk_by_sector(lang="en"):
    df = load("automation_sector_summary.csv")
    df = df[df["sector"] != "No especificado"].copy()
    df = df.sort_values("mean_prob", ascending=True)

    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    y_pos = np.arange(len(df))

    # Bar for mean_prob
    colors_bar = [COLORS["secondary"] if v > 0.5 else COLORS["tertiary"] for v in df["mean_prob"]]
    ax.barh(y_pos, df["mean_prob"], color=colors_bar, height=0.7,
            edgecolor="white", linewidth=0.3, alpha=0.85)

    # Mark 0.5 threshold
    ax.axvline(0.5, color=COLORS["dark_gray"], linewidth=0.5, linestyle=":", zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["sector"], fontsize=5.5)
    ax.set_xlabel(L("mean_prob", lang))
    ax.set_xlim(0, 0.75)
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_automation_risk_by_sector", lang)


# -------------------------------------------------------------------
# 13. fig_automation_risk_by_education
# -------------------------------------------------------------------
def fig_automation_risk_by_education(lang="en"):
    df = load("automation_analysis_dataset.csv", usecols=["education_level", "automation_prob", "high_risk"])
    df = df.dropna(subset=["education_level"])

    edu_order = ["Ninguno", "Primaria", "Secundaria", "Media", "Tecnico/Tecnologico", "Universitario", "Posgrado"]
    edu_labels_en = ["None", "Primary", "Secondary", "Upper sec.", "Technical", "University", "Postgrad."]
    edu_labels_es = ["Ninguno", "Primaria", "Secundaria", "Media", "T\u00e9cnico", "Universitario", "Posgrado"]
    edu_labels = edu_labels_en if lang == "en" else edu_labels_es

    means = []
    counts = []
    for e in edu_order:
        sub = df[df["education_level"] == e]
        means.append(sub["automation_prob"].mean() if len(sub) > 0 else 0)
        counts.append(len(sub))

    fig, ax = plt.subplots(figsize=SINGLE)
    x = np.arange(len(edu_order))
    colors_bar = [COLORS["secondary"] if m > 0.5 else COLORS["tertiary"] for m in means]
    ax.bar(x, means, color=colors_bar, width=0.65, edgecolor="white", linewidth=0.3)
    ax.axhline(0.5, color=COLORS["dark_gray"], linewidth=0.5, linestyle=":", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(edu_labels, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel(L("mean_prob", lang))
    ax.set_ylim(0, 0.75)
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_automation_risk_by_education", lang)


# -------------------------------------------------------------------
# 14. fig_heatmap_sector_formality
# -------------------------------------------------------------------
def fig_heatmap_sector_formality(lang="en"):
    df = load("automation_cross_tabulation.csv")
    # Pivot: sector x (formal x high_risk) -> mean_prob
    df["label"] = df["formal_label"] + " / " + df["high_risk_label"]
    piv = df.pivot_table(index="sector", columns="label", values="mean_prob", aggfunc="first")
    piv = piv.sort_index()

    # Translate column labels
    if lang == "es":
        piv.columns = [c.replace("Formal", "Formal").replace("Informal", "Informal")
                        .replace("Alto riesgo", "Alto riesgo").replace("Bajo riesgo", "Bajo riesgo")
                       for c in piv.columns]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(
        piv, cmap=cmap, annot=True, fmt=".2f", annot_kws={"size": 5},
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.6, "label": L("mean_prob", lang)},
        ax=ax,
    )
    ax.tick_params(axis="y", labelsize=5.5)
    ax.tick_params(axis="x", labelsize=5.5)
    plt.setp(ax.get_xticklabels(), ha="right", rotation=35, rotation_mode="anchor")
    fig.tight_layout()
    savefig(fig, "fig_heatmap_sector_formality", lang)


# -------------------------------------------------------------------
# 15. fig_coefficient_plots (firm models A-C)
# -------------------------------------------------------------------
def fig_coefficient_plots(lang="en"):
    df = load("table_regression_results.csv")

    # Focus on Models A, B, C - key economic variables only
    models = ["Model_A", "Model_B", "Model_C"]
    model_labels = [L("model_a", lang), L("model_b", lang), L("model_c", lang)]

    # Variables of interest (excluding sector dummies and diagnostics)
    vars_a = ["log_ulc", "log_lp", "log_emp"]
    vars_b = ["labor_share_va", "log_lp", "log_emp"]
    vars_c = ["log_lcpw", "log_emp"]
    var_lists = [vars_a, vars_b, vars_c]

    var_labels_en = {
        "log_ulc": "log(Unit labor cost)", "log_lp": "log(Labor prod.)",
        "log_emp": "log(Employment)", "labor_share_va": "Labor share (VA)",
        "log_lcpw": "log(Labor cost/worker)",
    }
    var_labels_es = {
        "log_ulc": "log(CLU)", "log_lp": "log(Prod. laboral)",
        "log_emp": "log(Empleo)", "labor_share_va": "Partic. laboral (VA)",
        "log_lcpw": "log(Costo lab./trab.)",
    }
    vl = var_labels_en if lang == "en" else var_labels_es

    fig, axes = plt.subplots(1, 3, figsize=DOUBLE, sharey=False)

    for idx, (model, var_list, mlabel) in enumerate(zip(models, var_lists, model_labels)):
        ax = axes[idx]
        mdata = df[(df["Model"] == model) & (df["Variable"].isin(var_list))].copy()
        mdata = mdata.dropna(subset=["Coefficient", "Std_Error"])

        y_pos = np.arange(len(mdata))
        colors_dots = [COLORS["secondary"] if p < 0.05 else COLORS["light_gray"]
                       for p in mdata["p_value"]]

        ax.errorbar(
            mdata["Coefficient"], y_pos,
            xerr=1.96 * mdata["Std_Error"],
            fmt="none", ecolor=COLORS["dark_gray"], elinewidth=0.5, capsize=2, capthick=0.4,
        )
        ax.scatter(mdata["Coefficient"], y_pos, c=colors_dots, s=20, zorder=3,
                   edgecolors="white", linewidths=0.3)
        ax.axvline(0, color=COLORS["light_gray"], linewidth=0.5, linestyle="--", zorder=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([vl.get(v, v) for v in mdata["Variable"]], fontsize=6)
        ax.set_xlabel(L("coefficient", lang), fontsize=7)
        ax.set_title(mlabel, fontsize=7, fontweight="bold")
        despine(ax)

    fig.tight_layout()
    savefig(fig, "fig_coefficient_plots", lang)


# -------------------------------------------------------------------
# 16. fig_scatter_lcpw_vs_invrate
# -------------------------------------------------------------------
def fig_scatter_lcpw_vs_invrate(lang="en"):
    df = load("firm_level_merged_dataset.csv")
    df = df.dropna(subset=["labor_cost_per_worker", "investment_rate"])
    # Filter outliers
    df = df[(df["investment_rate"] >= 0) & (df["investment_rate"] < 1)]
    df = df[df["labor_cost_per_worker"] > 0]

    fig, ax = plt.subplots(figsize=SINGLE)
    ax.scatter(
        np.log(df["labor_cost_per_worker"]), df["investment_rate"],
        s=3, alpha=0.15, color=COLORS["primary"], rasterized=True,
    )
    # Fit line
    x = np.log(df["labor_cost_per_worker"])
    y = df["investment_rate"]
    mask = np.isfinite(x) & np.isfinite(y)
    z = np.polyfit(x[mask], y[mask], 1)
    p = np.poly1d(z)
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, p(xr), color=COLORS["secondary"], linewidth=0.8, linestyle="--")

    ax.set_xlabel(L("log_lcpw", lang))
    ax.set_ylabel(L("investment_rate", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_scatter_lcpw_vs_invrate", lang)


# -------------------------------------------------------------------
# 17. fig_sector_bubble_lcost_auto
# -------------------------------------------------------------------
def fig_sector_bubble_lcost_auto(lang="en"):
    df = load("table_sectoral_averages.csv")
    df = df.dropna(subset=["avg_labor_cost_pw", "avg_automation_proxy", "total_employment"])
    df = df[df["avg_automation_proxy"] > 0]

    fig, ax = plt.subplots(figsize=SINGLE)
    sizes = df["total_employment"] / df["total_employment"].max() * 300
    sc = ax.scatter(
        df["avg_labor_cost_pw"] / 1e3, df["avg_automation_proxy"] / 1e6,
        s=sizes, alpha=0.6, color=COLORS["tertiary"],
        edgecolors=COLORS["primary"], linewidths=0.5, zorder=3,
    )
    for _, row in df.iterrows():
        if row["avg_automation_proxy"] > 0:
            ax.annotate(
                row["sector_name"][:12],
                xy=(row["avg_labor_cost_pw"] / 1e3, row["avg_automation_proxy"] / 1e6),
                fontsize=4.5, ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points",
            )
    ax.set_xlabel(L("labor_cost_pw", lang))
    ax.set_ylabel(L("automation_proxy", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_sector_bubble_lcost_auto", lang)


# -------------------------------------------------------------------
# 18. fig_sim_formal_employment_trajectories
# -------------------------------------------------------------------
def fig_sim_formal_employment_trajectories(lang="en"):
    df = load("simulation_full_results.csv")
    agg = df.groupby(["year", "scenario"])["formal_employment"].sum().reset_index()

    fig, ax = plt.subplots(figsize=SINGLE)
    scenarios = ["Status Quo", "Labor Reform", "Parafiscal Reform", "AI Acceleration", "Combined Worst"]
    for i, s in enumerate(scenarios):
        d = agg[agg["scenario"] == s].sort_values("year")
        label = SCENARIO_NAMES[lang].get(s, s)
        ax.plot(d["year"], d["formal_employment"], label=label,
                color=PALETTE[i], linewidth=1.0 if s != "Combined Worst" else 1.2,
                linestyle="-" if s != "Combined Worst" else "--")

    ax.set_xlabel(L("year", lang))
    ax.set_ylabel(L("formal_employment", lang))
    ax.legend(fontsize=5.5, loc="lower left", frameon=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_sim_formal_employment_trajectories", lang)


# -------------------------------------------------------------------
# 19. fig_sim_waterfall_labor_reform
# -------------------------------------------------------------------
def fig_sim_waterfall_labor_reform(lang="en"):
    """Waterfall chart showing incremental displacement from Status Quo to Labor Reform."""
    sc = load("simulation_scenario_comparison.csv")
    sq = sc[sc["Scenario"] == "Status Quo"].iloc[0]
    lr = sc[sc["Scenario"] == "Labor Reform"].iloc[0]

    # Components: Status Quo displaced, additional from reform, informalized
    values = [
        sq["Cumulative Displaced (M)"],
        lr["Cumulative Displaced (M)"] - sq["Cumulative Displaced (M)"],
        lr["Cumulative Informalized (M)"],
    ]
    labels_en = ["Base displacement\n(Status Quo)", "Additional\ndisplacement", "Informalized"]
    labels_es = ["Desplazamiento\nbase (Statu Quo)", "Desplazamiento\nadicional", "Informalizado"]
    labels = labels_en if lang == "en" else labels_es

    # Waterfall logic
    cumulative = [0]
    for v in values[:-1]:
        cumulative.append(cumulative[-1] + v)

    fig, ax = plt.subplots(figsize=SINGLE)
    bar_colors = [COLORS["primary"], COLORS["secondary"], COLORS["quinary"]]

    for i, (val, bottom) in enumerate(zip(values, cumulative)):
        ax.bar(i, val, bottom=bottom, color=bar_colors[i], width=0.5,
               edgecolor="white", linewidth=0.3)
        ax.text(i, bottom + val / 2, f"{val:.2f}M", ha="center", va="center", fontsize=6, color="white")

    # Total line
    total = sum(values)
    ax.plot([-0.4, len(values) - 0.6], [total, total], color=COLORS["dark_gray"],
            linewidth=0.5, linestyle="--")
    total_label = f"Total: {total:.2f}M"
    ax.text(len(values) - 1, total + 0.1, total_label, fontsize=6, ha="center", color=COLORS["dark_gray"])

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(L("displacement", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_sim_waterfall_labor_reform", lang)


# -------------------------------------------------------------------
# 20. fig_sim_heatmap_displacement
# -------------------------------------------------------------------
def fig_sim_heatmap_displacement(lang="en"):
    df = load("simulation_sectoral_breakdown.csv")
    piv = df.pivot_table(index="Sector", columns="Scenario", values="Pct Change (%)", aggfunc="first")

    # Reorder scenarios
    scenario_order = ["Status Quo", "Parafiscal Reform", "AI Acceleration", "Labor Reform", "Combined Worst"]
    piv = piv[[s for s in scenario_order if s in piv.columns]]

    # Translate
    if lang == "es":
        piv.columns = [SCENARIO_NAMES["es"].get(c, c) for c in piv.columns]
        piv.index = [SECTOR_NAMES["es"].get(s, s) for s in piv.index]
    else:
        piv.index = [SECTOR_NAMES["en"].get(s, s) for s in piv.index]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    sns.heatmap(
        piv, cmap=cmap, annot=True, fmt=".1f", annot_kws={"size": 5.5},
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.7, "label": L("pct_change", lang)},
        ax=ax, center=-25,
    )
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="x", labelsize=6)
    plt.setp(ax.get_xticklabels(), ha="right", rotation=30, rotation_mode="anchor")
    fig.tight_layout()
    savefig(fig, "fig_sim_heatmap_displacement", lang)


# -------------------------------------------------------------------
# 21. fig_sim_tornado_sensitivity
# -------------------------------------------------------------------
def fig_sim_tornado_sensitivity(lang="en"):
    df = load("simulation_sensitivity_analysis.csv")
    df = df.sort_values("Range (M)", ascending=True)

    param_labels_en = {
        "Automation risk (+/-10pp)": "Automation risk (\u00b110pp)",
        "Elasticity (sigma)": "Elasticity (\u03c3)",
        "Tech cost decline": "Tech. cost decline",
        "Reinstatement rate": "Reinstatement rate",
        "Min wage growth": "Min. wage growth",
    }
    param_labels_es = {
        "Automation risk (+/-10pp)": "Riesgo automatizaci\u00f3n (\u00b110pp)",
        "Elasticity (sigma)": "Elasticidad (\u03c3)",
        "Tech cost decline": "Ca\u00edda costo tecnol.",
        "Reinstatement rate": "Tasa de reinstalaci\u00f3n",
        "Min wage growth": "Crecimiento salario m\u00edn.",
    }
    pl = param_labels_en if lang == "en" else param_labels_es
    central = df["Central Displacement (M)"].iloc[0]

    fig, ax = plt.subplots(figsize=SINGLE)
    y_pos = np.arange(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        low = row["Low Value Displacement (M)"]
        high = row["High Value Displacement (M)"]
        ax.barh(i, high - central, left=central, color=COLORS["secondary"],
                height=0.5, edgecolor="white", linewidth=0.3)
        ax.barh(i, low - central, left=central, color=COLORS["tertiary"],
                height=0.5, edgecolor="white", linewidth=0.3)

    ax.axvline(central, color=COLORS["dark_gray"], linewidth=0.5, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([pl.get(p, p) for p in df["Parameter"]], fontsize=6)
    ax.set_xlabel(L("displaced_m", lang))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["tertiary"], label=L("low_value", lang)),
        Patch(facecolor=COLORS["secondary"], label=L("high_value", lang)),
    ]
    ax.legend(handles=legend_elements, fontsize=6, loc="lower right", frameon=False)
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_sim_tornado_sensitivity", lang)


# -------------------------------------------------------------------
# 22. fig_sim_fan_chart_baseline
# -------------------------------------------------------------------
def fig_sim_fan_chart_baseline(lang="en"):
    """Fan chart using Monte Carlo summary data for Status Quo scenario,
    extrapolated across years using full simulation results."""
    mc = load("simulation_montecarlo_summary.csv")
    full = load("simulation_full_results.csv")

    # Get Status Quo trajectory
    sq = full[full["scenario"] == "Status Quo"]
    agg = sq.groupby("year")["formal_employment"].sum().reset_index().sort_values("year")

    # Get MC summary for Status Quo
    sq_mc = mc[mc["Scenario"] == "Status Quo"].iloc[0]
    mean_disp = sq_mc["Mean Displaced (M)"]
    std_dev = sq_mc["Std Dev (M)"]

    # Initial total formal employment
    initial_formal = agg["formal_employment"].iloc[0]
    years = agg["year"].values
    formal = agg["formal_employment"].values

    # Build uncertainty bands proportionally
    # At final year the displacement has SD. Scale proportionally over time.
    final_formal = formal[-1]
    displacement_profile = initial_formal - formal  # cumulative displacement at each year
    max_displacement = displacement_profile[-1]

    # Scale factor: at each year, fraction of total displacement
    frac = np.where(max_displacement > 0, displacement_profile / max_displacement, 0)

    # Uncertainty at each point
    sd_at_year = frac * std_dev

    fig, ax = plt.subplots(figsize=SINGLE)

    # Bands: 50%, 80%, 90% CI
    bands = [(1.645, 0.08), (1.28, 0.12), (0.674, 0.18)]
    for z_val, alpha_val in bands:
        upper = formal + z_val * sd_at_year
        lower = formal - z_val * sd_at_year
        ax.fill_between(years, lower, upper, alpha=alpha_val, color=COLORS["tertiary"], linewidth=0)

    ax.plot(years, formal, color=COLORS["primary"], linewidth=1.0,
            label=L("median", lang))

    # Label bands
    ci_labels = ["90%", "80%", "50%"]
    ax.text(years[-1] + 0.15, formal[-1] + 1.645 * sd_at_year[-1], "90%", fontsize=5,
            color=COLORS["dark_gray"], va="center")
    ax.text(years[-1] + 0.15, formal[-1] + 0.674 * sd_at_year[-1], "50%", fontsize=5,
            color=COLORS["dark_gray"], va="center")

    ax.set_xlabel(L("year", lang))
    ax.set_ylabel(L("formal_employment", lang))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=6, loc="upper right", frameon=False)
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_sim_fan_chart_baseline", lang)


# ===================================================================
# MAIN EXECUTION
# ===================================================================
FIGURES = [
    ("fig1_productivity_vs_robots", fig1_productivity_vs_robots),
    ("fig2_robot_timeseries", fig2_robot_timeseries),
    ("fig3_coefficient_plot", fig3_coefficient_plot),
    ("fig5_robot_density_comparison", fig5_robot_density_comparison),
    ("fig6_correlation_matrix", fig6_correlation_matrix),
    ("fig_bar_vulnerability_ranking", fig_bar_vulnerability_ranking),
    ("fig_heatmap_sectoral_indicators", fig_heatmap_sectoral_indicators),
    ("fig_scatter_productivity_vs_laborshare", fig_scatter_productivity_vs_laborshare),
    ("fig_roc_curve", fig_roc_curve),
    ("fig_forest_plot_odds_ratios", fig_forest_plot_odds_ratios),
    ("fig_automation_risk_by_sector", fig_automation_risk_by_sector),
    ("fig_automation_risk_by_education", fig_automation_risk_by_education),
    ("fig_heatmap_sector_formality", fig_heatmap_sector_formality),
    ("fig_coefficient_plots", fig_coefficient_plots),
    ("fig_scatter_lcpw_vs_invrate", fig_scatter_lcpw_vs_invrate),
    ("fig_sector_bubble_lcost_auto", fig_sector_bubble_lcost_auto),
    ("fig_sim_formal_employment_trajectories", fig_sim_formal_employment_trajectories),
    ("fig_sim_waterfall_labor_reform", fig_sim_waterfall_labor_reform),
    ("fig_sim_heatmap_displacement", fig_sim_heatmap_displacement),
    ("fig_sim_tornado_sensitivity", fig_sim_tornado_sensitivity),
    ("fig_sim_fan_chart_baseline", fig_sim_fan_chart_baseline),
]


def main():
    print(f"{'='*60}")
    print("Regenerating ALL figures in EN and ES")
    print(f"Output: {IMG_DIR}/en/ and {IMG_DIR}/es/")
    print(f"{'='*60}\n")

    success = 0
    failures = []

    for name, func in FIGURES:
        for lang in ("en", "es"):
            try:
                print(f"[{lang.upper()}] {name} ...", end=" ")
                func(lang=lang)
                success += 1
            except Exception as e:
                failures.append((name, lang, str(e)))
                print(f"  FAILED: {e}")
                traceback.print_exc()
                plt.close("all")

    print(f"\n{'='*60}")
    print(f"Results: {success} succeeded, {len(failures)} failed")
    if failures:
        print("\nFailures:")
        for name, lang, err in failures:
            print(f"  [{lang}] {name}: {err}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
