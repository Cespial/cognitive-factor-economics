#!/usr/bin/env python3
"""
07_improved_figures.py
======================
Q1-quality figures for the automation paper.
Fixes: ROC AUC bug, sector name translations, label overlaps,
colorblind-friendly palettes, confidence bands, overplotting.
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
from matplotlib.patches import Patch
import seaborn as sns
from adjustText import adjust_text

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
# Q1 Journal Style (Nature / Econometrica)
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

# Colorblind-friendly palette (Okabe-Ito inspired)
COLORS = {
    "primary": "#2C3E50",
    "secondary": "#C0392B",
    "tertiary": "#2980B9",
    "quaternary": "#27AE60",
    "quinary": "#D4880F",
    "light_gray": "#BDC3C7",
    "dark_gray": "#7F8C8D",
}
PALETTE = [
    "#2C3E50", "#C0392B", "#2980B9", "#27AE60", "#D4880F",
    "#8E44AD", "#16A085", "#D35400", "#7F8C8D",
]

SINGLE = (3.5, 2.5)
DOUBLE = (7.0, 4.0)
SINGLE_TALL = (3.5, 4.0)
DOUBLE_TALL = (7.0, 5.5)

# ---------------------------------------------------------------------------
# Sector name translations (data is in Spanish)
# ---------------------------------------------------------------------------
SECTOR_TRANS_EN = {
    "Transporte": "Transport",
    "Agricultura": "Agriculture",
    "Hogares empleadores": "Household employers",
    "Adm. y apoyo": "Admin. & support",
    "Construccion": "Construction",
    "Construcción": "Construction",
    "Mineria": "Mining",
    "Minería": "Mining",
    "Manufactura": "Manufacturing",
    "Serv. publicos": "Utilities",
    "Serv. públicos": "Utilities",
    "Inmobiliario": "Real estate",
    "Admin. publica": "Public admin.",
    "Admin. pública": "Public admin.",
    "Financiero": "Financial",
    "Alojamiento/Comida": "Accommodation/Food",
    "Arte/Entretenimiento": "Arts/Entertainment",
    "Otros servicios": "Other services",
    "Comercio": "Commerce",
    "Org. extraterritoriales": "Intl. organizations",
    "Info/Comunicaciones": "ICT",
    "Salud": "Health",
    "Serv. profesionales": "Professional serv.",
    "Educacion": "Education",
    "Educación": "Education",
    "No especificado": "Not specified",
    # Sectoral analysis
    "Construcción": "Construction",
    "Adm./Educ./Salud": "Admin./Educ./Health",
    "Manufactura": "Manufacturing",
    "Serv. prof.": "Prof. services",
    "Comercio/Transp.": "Commerce/Transport",
    "Minería": "Mining",
    "TIC": "ICT",
    "Artes/Otros": "Arts/Other",
    "Inmobiliario": "Real estate",
    "Elec. y agua": "Utilities",
    "Financiero": "Financial",
}

# Heatmap formality/risk labels
FORMALITY_RISK_EN = {
    "Formal / Alto riesgo": "Formal / High risk",
    "Formal / Bajo riesgo": "Formal / Low risk",
    "Informal / Alto riesgo": "Informal / High risk",
    "Informal / Bajo riesgo": "Informal / Low risk",
}

# Simulation sector names
SIM_SECTOR_EN = {
    "Agriculture": "Agriculture",
    "BPO/Professional": "BPO/Professional",
    "Commerce/Transport": "Commerce/Transport",
    "Construction": "Construction",
    "Financial Services": "Financial services",
    "Manufacturing": "Manufacturing",
    "Mining": "Mining",
    "Other Services": "Other services",
    "Public Admin/Educ/Health": "Public admin./Educ./Health",
}
SIM_SECTOR_ES = {
    "Agriculture": "Agricultura",
    "BPO/Professional": "BPO/Profesional",
    "Commerce/Transport": "Comercio/Transporte",
    "Construction": "Construcción",
    "Financial Services": "Serv. financieros",
    "Manufacturing": "Manufactura",
    "Mining": "Minería",
    "Other Services": "Otros servicios",
    "Public Admin/Educ/Health": "Adm. púb./Educ./Salud",
}

# Manufacturing subsector translations
MFG_SUBSECTOR_EN = {
    "Coque y refi": "Coke & refining",
    "Papel": "Paper",
    "Metalurgia b": "Basic metals",
    "Min. no meta": "Non-metallic min.",
    "Bebidas": "Beverages",
    "Otras manufa": "Other mfg.",
    "Alimentos": "Food",
    "Vehiculos": "Vehicles",
    "Vehículos": "Vehicles",
    "Imprenta": "Printing",
    "Textil": "Textiles",
    "Caucho y pla": "Rubber & plastics",
    "Otro eq. tra": "Other transport eq.",
    "Confecciones": "Apparel",
    "Prod. metali": "Metal products",
    "Maquinaria": "Machinery",
    "Muebles": "Furniture",
    "Madera": "Wood",
    "Cuero y calz": "Leather & footwear",
    "Equipo elect": "Electrical eq.",
    "Quimicos": "Chemicals",
    "Químicos": "Chemicals",
    "Farmaceutico": "Pharmaceuticals",
    "Farmacéutico": "Pharmaceuticals",
}

# ---------------------------------------------------------------------------
# Translation dictionaries
# ---------------------------------------------------------------------------
LABELS = {
    "en": {
        "robot_density": "Robot density (per 10,000 mfg. workers)",
        "gdp_per_worker": "GDP per worker (log, PPP USD)",
        "year": "Year",
        "robot_installations": "Annual robot installations",
        "coefficient": "Coefficient",
        "log_robots": "log(Robot installations)",
        "log_productivity": "log(Productivity)",
        "rd_gdp_pct": "R&D (% GDP)",
        "log_gdp_per_worker": "log(GDP per worker)",
        "correlation": "Correlation",
        "avi": "Automation Vulnerability Index",
        "labor_share_norm": "Labor share (norm.)",
        "productivity_growth": "Productivity growth (norm.)",
        "capital_intensity": "Capital intensity (norm.)",
        "automation_prob": "Automation probability",
        "fpr": "False positive rate",
        "tpr": "True positive rate",
        "odds_ratio": "Odds ratio",
        "mean_prob": "Mean automation probability",
        "roc_auc": "AUC = {:.3f}",
        "log_ulc": "log(Unit labor cost)",
        "log_lp": "log(Labor prod.)",
        "log_emp": "log(Employment)",
        "log_lcpw": "log(Labor cost/worker)",
        "investment_rate": "Investment rate",
        "labor_cost_pw": "Labor cost per worker (COP M)",
        "automation_proxy": "Automation investment (COP M)",
        "formal_employment": "Formal employment (millions)",
        "displacement": "Cumulative displacement (millions)",
        "scenario": "Scenario",
        "pct_change": "Change (%)",
        "displaced_m": "Displaced (millions)",
        "median": "Median",
        "low_value": "Low",
        "high_value": "High",
        "model_a": "Model A: log(Autom. proxy)",
        "model_b": "Model B: Investment rate",
        "model_c": "Model C: log(Capital int.)",
        "labor_share_va": "Labor share (VA)",
        "log_lcpw_short": "log(Labor cost/wkr)",
        "status_quo": "Status Quo",
        "labor_reform": "Labor Reform",
        "parafiscal_reform": "Parafiscal Reform",
        "ai_acceleration": "AI Acceleration",
        "combined_worst": "Combined Worst",
        "sector": "Sector",
        "formality_risk": "Formality / Risk category",
    },
    "es": {
        "robot_density": "Densidad robótica (por 10.000 trab. manuf.)",
        "gdp_per_worker": "PIB por trabajador (log, USD PPA)",
        "year": "Año",
        "robot_installations": "Instalaciones anuales de robots",
        "coefficient": "Coeficiente",
        "log_robots": "log(Instalaciones de robots)",
        "log_productivity": "log(Productividad)",
        "rd_gdp_pct": "I+D (% PIB)",
        "log_gdp_per_worker": "log(PIB por trabajador)",
        "correlation": "Correlación",
        "avi": "Índice de vulnerabilidad a la automatización",
        "labor_share_norm": "Participación laboral (norm.)",
        "productivity_growth": "Crecimiento productividad (norm.)",
        "capital_intensity": "Intensidad de capital (norm.)",
        "automation_prob": "Probabilidad de automatización",
        "fpr": "Tasa de falsos positivos",
        "tpr": "Tasa de verdaderos positivos",
        "odds_ratio": "Razón de momios",
        "mean_prob": "Probabilidad media de automatización",
        "roc_auc": "AUC = {:.3f}",
        "log_ulc": "log(Costo laboral unitario)",
        "log_lp": "log(Prod. laboral)",
        "log_emp": "log(Empleo)",
        "log_lcpw": "log(Costo laboral/trabajador)",
        "investment_rate": "Tasa de inversión",
        "labor_cost_pw": "Costo laboral por trabajador (COP M)",
        "automation_proxy": "Inversión en automatización (COP M)",
        "formal_employment": "Empleo formal (millones)",
        "displacement": "Desplazamiento acumulado (millones)",
        "scenario": "Escenario",
        "pct_change": "Cambio (%)",
        "displaced_m": "Desplazados (millones)",
        "median": "Mediana",
        "low_value": "Bajo",
        "high_value": "Alto",
        "model_a": "Modelo A: log(Proxy autom.)",
        "model_b": "Modelo B: Tasa de inversión",
        "model_c": "Modelo C: log(Intens. capital)",
        "labor_share_va": "Partic. laboral (VA)",
        "log_lcpw_short": "log(Costo lab./trab.)",
        "status_quo": "Statu Quo",
        "labor_reform": "Reforma laboral",
        "parafiscal_reform": "Reforma parafiscal",
        "ai_acceleration": "Aceleración IA",
        "combined_worst": "Peor combinado",
        "sector": "Sector",
        "formality_risk": "Formalidad / Categoría de riesgo",
    },
}

SCENARIO_NAMES = {
    "en": {"Status Quo": "Status Quo", "Labor Reform": "Labor Reform",
           "Parafiscal Reform": "Parafiscal Reform", "AI Acceleration": "AI Acceleration",
           "Combined Worst": "Combined Worst"},
    "es": {"Status Quo": "Statu Quo", "Labor Reform": "Reforma laboral",
           "Parafiscal Reform": "Reforma parafiscal", "AI Acceleration": "Aceleración IA",
           "Combined Worst": "Peor combinado"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def L(key, lang):
    return LABELS[lang].get(key, key)


def savefig(fig, name, lang):
    path = os.path.join(IMG_DIR, lang, f"{name}.pdf")
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  Saved: {path}")


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def translate_sector(name, lang):
    """Translate sector name for the given language."""
    if lang == "en":
        return SECTOR_TRANS_EN.get(name, name)
    return name


def translate_sectors_list(names, lang):
    return [translate_sector(n, lang) for n in names]


_cache = {}
def load(name, **kwargs):
    def _h(v):
        return tuple(v) if isinstance(v, list) else v
    ck = (name, tuple(sorted((k, _h(v)) for k, v in kwargs.items())) if kwargs else ())
    if ck not in _cache:
        _cache[ck] = pd.read_csv(os.path.join(DATA_DIR, name), **kwargs)
    return _cache[ck].copy()


# ===================================================================
# FIGURE FUNCTIONS
# ===================================================================

# 1. Productivity vs Robots (scatter with adjustText + CI band)
def fig1_productivity_vs_robots(lang="en"):
    df = load("international_panel.csv")
    latest = df.groupby("country")["year"].max().reset_index()
    df_l = df.merge(latest, on=["country", "year"])
    df_l = df_l.dropna(subset=["log_robots", "log_gdp_per_worker"])

    fig, ax = plt.subplots(figsize=SINGLE)
    ax.scatter(df_l["log_robots"], df_l["log_gdp_per_worker"],
               s=18, alpha=0.7, color=COLORS["primary"],
               edgecolors="white", linewidths=0.3, zorder=3)

    # Fit line + confidence band
    x, y = df_l["log_robots"].values, df_l["log_gdp_per_worker"].values
    mask = np.isfinite(x) & np.isfinite(y)
    x_m, y_m = x[mask], y[mask]
    z = np.polyfit(x_m, y_m, 1)
    p = np.poly1d(z)
    xr = np.linspace(x_m.min(), x_m.max(), 100)
    y_hat = p(xr)
    # SE of prediction
    residuals = y_m - p(x_m)
    se = np.std(residuals)
    ax.plot(xr, y_hat, color=COLORS["secondary"], linewidth=0.8, linestyle="--", zorder=2)
    ax.fill_between(xr, y_hat - 1.96 * se, y_hat + 1.96 * se,
                     alpha=0.08, color=COLORS["secondary"], linewidth=0)

    # Labels with adjustText
    key_countries = ["Colombia", "United States", "South Korea", "Germany",
                     "China", "Japan", "Brazil", "Mexico"]
    texts = []
    for c in key_countries:
        row = df_l[df_l["country"] == c]
        if not row.empty:
            cx, cy = row["log_robots"].iloc[0], row["log_gdp_per_worker"].iloc[0]
            t = ax.text(cx, cy, c, fontsize=5.5, color=COLORS["dark_gray"], ha="left")
            texts.append(t)

    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.3, color=COLORS["light_gray"]),
                    force_text=(0.5, 0.5), force_points=(0.3, 0.3))

    ax.set_xlabel(L("log_robots", lang))
    ax.set_ylabel(L("gdp_per_worker", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig1_productivity_vs_robots", lang)


# 2. Robot timeseries
def fig2_robot_timeseries(lang="en"):
    df = load("international_panel.csv")
    countries = ["South Korea", "Germany", "Japan", "United States", "China", "Colombia", "Brazil", "Mexico"]
    sub = df[df["country"].isin(countries)].sort_values("year")

    fig, ax = plt.subplots(figsize=SINGLE)
    for i, c in enumerate(countries):
        d = sub[sub["country"] == c]
        lw = 1.4 if c == "Colombia" else 0.8
        ax.plot(d["year"], d["robot_installations"], label=c,
                color=PALETTE[i % len(PALETTE)], linewidth=lw)

    ax.set_xlabel(L("year", lang))
    ax.set_ylabel(L("robot_installations", lang))
    ax.legend(fontsize=5.5, ncol=2, loc="upper left", frameon=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig2_robot_timeseries", lang)


# 3. Coefficient plot (international panel FE)
def fig3_coefficient_plot(lang="en"):
    reg = load("regression_results.csv")
    reg.columns = ["Variable", "Pooled OLS", "Fixed Effects", "Random Effects", "Dynamic OLS"]

    vars_of_interest = ["log_productivity", "rd_gdp_pct", "log_gdp_per_worker"]
    var_labels = {
        "log_productivity": L("log_productivity", lang),
        "rd_gdp_pct": L("rd_gdp_pct", lang),
        "log_gdp_per_worker": L("log_gdp_per_worker", lang),
    }

    coefs, ses, labels = [], [], []
    for v in vars_of_interest:
        rc = reg[reg["Variable"] == v]
        rs = reg[reg["Variable"] == f"{v}_se"]
        if not rc.empty and not rs.empty:
            c = float(rc["Fixed Effects"].iloc[0])
            se_str = str(rs["Fixed Effects"].iloc[0])
            se_val = float(se_str.replace("(", "").replace(")", "").replace("*", ""))
            coefs.append(c)
            ses.append(se_val)
            labels.append(var_labels.get(v, v))

    coefs, ses = np.array(coefs), np.array(ses)

    fig, ax = plt.subplots(figsize=SINGLE)
    y_pos = np.arange(len(labels))
    ax.errorbar(coefs, y_pos, xerr=1.96 * ses, fmt="o", color=COLORS["primary"],
                markersize=5, capsize=3, capthick=0.6, elinewidth=0.8)
    ax.axvline(0, color=COLORS["light_gray"], linewidth=0.5, linestyle="--", zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(L("coefficient", lang))
    ax.invert_yaxis()
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig3_coefficient_plot", lang)


# 5. Robot density comparison (bar)
def fig5_robot_density_comparison(lang="en"):
    df = load("international_panel.csv")
    latest = df.groupby("country")["year"].max().reset_index()
    df_l = df.merge(latest, on=["country", "year"])
    df_l["robot_density"] = df_l["robot_installations"] / df_l["employment_millions"]
    df_l = df_l.dropna(subset=["robot_density"]).sort_values("robot_density")

    top20 = df_l.nlargest(20, "robot_density")
    col = df_l[df_l["country"] == "Colombia"]
    show = pd.concat([top20, col]).drop_duplicates(subset=["country"]).sort_values("robot_density")

    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    colors_bar = [COLORS["secondary"] if c == "Colombia" else COLORS["primary"]
                  for c in show["country"]]
    ax.barh(range(len(show)), show["robot_density"], color=colors_bar,
            height=0.7, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show["country"], fontsize=6)
    ax.set_xlabel(L("robot_density", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig5_robot_density_comparison", lang)


# 6. Correlation matrix (colorblind-friendly)
def fig6_correlation_matrix(lang="en"):
    df = load("international_panel.csv")
    cols = ["log_robots", "log_gdp_per_worker", "rd_gdp_pct", "tfp", "labor_share", "human_capital"]
    col_labels = {
        "log_robots": L("log_robots", lang),
        "log_gdp_per_worker": L("log_gdp_per_worker", lang),
        "rd_gdp_pct": L("rd_gdp_pct", lang),
        "tfp": "TFP",
        "labor_share": L("labor_share_norm", lang),
        "human_capital": "Human capital" if lang == "en" else "Capital humano",
    }
    sub = df[cols].dropna()
    corr = sub.corr()
    labels_list = [col_labels.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    # Colorblind-friendly diverging: RdBu
    cmap = plt.cm.RdBu_r
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                xticklabels=labels_list, yticklabels=labels_list,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.7, "label": L("correlation", lang)},
                ax=ax, vmin=-1, vmax=1, square=True)
    ax.tick_params(axis="both", labelsize=5.5, rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    fig.tight_layout()
    savefig(fig, "fig6_correlation_matrix", lang)


# 7. AVI vulnerability ranking
def fig_bar_vulnerability_ranking(lang="en"):
    df = load("sectoral_analysis_results.csv", encoding="utf-8-sig")
    df = df.sort_values("indice_vulnerabilidad", ascending=True)

    sector_names = translate_sectors_list(df["sector"].tolist(), lang)

    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    colors_bar = [COLORS["secondary"] if v > 60 else COLORS["primary"]
                  for v in df["indice_vulnerabilidad"]]
    ax.barh(range(len(df)), df["indice_vulnerabilidad"], color=colors_bar,
            height=0.7, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(sector_names, fontsize=6)
    ax.set_xlabel(L("avi", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_bar_vulnerability_ranking", lang)


# 8. Sectoral indicators heatmap (colorblind: viridis)
def fig_heatmap_sectoral_indicators(lang="en"):
    df = load("sectoral_analysis_results.csv", encoding="utf-8-sig")
    cols = ["labor_share_norm", "prod_growth_norm", "cap_intensity_norm"]
    col_labels = [L("labor_share_norm", lang), L("productivity_growth", lang), L("capital_intensity", lang)]
    mat = df[cols].values
    sectors = translate_sectors_list(df["sector"].tolist(), lang)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    cmap = plt.cm.YlOrRd
    sns.heatmap(mat, cmap=cmap, annot=True, fmt=".0f", annot_kws={"size": 6},
                xticklabels=col_labels, yticklabels=sectors,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="x", labelsize=6)
    plt.setp(ax.get_xticklabels(), ha="right", rotation=30, rotation_mode="anchor")
    fig.tight_layout()
    savefig(fig, "fig_heatmap_sectoral_indicators", lang)


# 9. Productivity vs labor share scatter (with adjustText)
def fig_scatter_productivity_vs_laborshare(lang="en"):
    df = load("sectoral_analysis_results.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=SINGLE)
    ax.scatter(df["prod_growth_norm"], df["labor_share_norm"],
               s=df["va_share_pct_2024"] * 8, alpha=0.7,
               color=COLORS["tertiary"], edgecolors=COLORS["primary"], linewidths=0.4, zorder=3)

    texts = []
    sector_names = translate_sectors_list(df["sector"].tolist(), lang)
    for i, (_, row) in enumerate(df.iterrows()):
        t = ax.text(row["prod_growth_norm"], row["labor_share_norm"],
                    sector_names[i], fontsize=5, ha="center")
        texts.append(t)

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.3, color=COLORS["light_gray"]),
                force_text=(0.8, 0.8))

    ax.set_xlabel(L("productivity_growth", lang))
    ax.set_ylabel(L("labor_share_norm", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_scatter_productivity_vs_laborshare", lang)


# 10. ROC curve -- FIXED: use actual logistic model, not circular prediction
def fig_roc_curve(lang="en"):
    """Fit logistic model on the data and compute proper ROC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import StandardScaler

    df = load("automation_analysis_dataset.csv")
    # Fill missing education with a numeric code for "Unknown"
    df["education_num"] = df["education_num"].fillna(-1)
    df = df.dropna(subset=["high_risk", "formal", "female", "age", "log_income",
                           "hours_worked"])

    # Features for logistic model (NOT automation_prob -- that would be circular)
    feature_cols = ["formal", "female", "age", "age_sq", "log_income",
                    "hours_worked", "education_num"]
    X = df[feature_cols].values
    y = df["high_risk"].values

    # Handle any remaining NaN
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Compute ROC
    fpr_vals, tpr_vals, _ = roc_curve(y, y_prob)
    auc_val = auc(fpr_vals, tpr_vals)

    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(fpr_vals, tpr_vals, color=COLORS["primary"], linewidth=1.0, zorder=3)
    ax.plot([0, 1], [0, 1], color=COLORS["light_gray"], linewidth=0.5, linestyle="--", zorder=1)
    ax.fill_between(fpr_vals, tpr_vals, alpha=0.06, color=COLORS["tertiary"])
    ax.text(0.55, 0.15, L("roc_auc", lang).format(auc_val), fontsize=8,
            transform=ax.transAxes, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["light_gray"], alpha=0.8))
    ax.set_xlabel(L("fpr", lang))
    ax.set_ylabel(L("tpr", lang))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_roc_curve", lang)


# 11. Forest plot odds ratios
def fig_forest_plot_odds_ratios(lang="en"):
    df = load("automation_odds_ratios_corrected.csv")
    individual_vars = ["formal", "female", "age", "age_sq", "log_income", "hours_worked",
                       "edu_Ninguno", "edu_Media", "edu_Tecnico_Tecnologico", "edu_Universitario", "edu_Posgrado",
                       "firm_Micro_1", "firm_Pequena_11-50", "firm_Mediana_51-200", "firm_Grande_201plus"]

    vl_en = {"formal": "Formal", "female": "Female", "age": "Age", "age_sq": "Age²",
             "log_income": "log(Income)", "hours_worked": "Hours worked",
             "edu_Ninguno": "Educ.: None", "edu_Media": "Educ.: Upper sec.",
             "edu_Tecnico_Tecnologico": "Educ.: Technical", "edu_Universitario": "Educ.: University",
             "edu_Posgrado": "Educ.: Postgrad.",
             "firm_Micro_1": "Firm: Micro(1)", "firm_Pequena_11-50": "Firm: Small",
             "firm_Mediana_51-200": "Firm: Medium", "firm_Grande_201plus": "Firm: Large"}
    vl_es = {"formal": "Formal", "female": "Mujer", "age": "Edad", "age_sq": "Edad²",
             "log_income": "log(Ingreso)", "hours_worked": "Horas trabajadas",
             "edu_Ninguno": "Educ.: Ninguno", "edu_Media": "Educ.: Media",
             "edu_Tecnico_Tecnologico": "Educ.: Técnico", "edu_Universitario": "Educ.: Universitario",
             "edu_Posgrado": "Educ.: Posgrado",
             "firm_Micro_1": "Firma: Micro(1)", "firm_Pequena_11-50": "Firma: Pequeña",
             "firm_Mediana_51-200": "Firma: Mediana", "firm_Grande_201plus": "Firma: Grande"}
    vl = vl_en if lang == "en" else vl_es

    sub = df[df["Variable"].isin(individual_vars)].copy()
    sub["order"] = sub["Variable"].apply(lambda x: individual_vars.index(x) if x in individual_vars else 99)
    sub = sub.sort_values("order", ascending=False)

    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    y_pos = np.arange(len(sub))
    colors_dots = [COLORS["secondary"] if p < 0.05 else COLORS["light_gray"] for p in sub["p_value"]]

    ax.errorbar(sub["OR"], y_pos,
                xerr=[sub["OR"] - sub["OR_CI_low"], sub["OR_CI_high"] - sub["OR"]],
                fmt="none", ecolor=COLORS["dark_gray"], elinewidth=0.5, capsize=1.5, capthick=0.4)
    ax.scatter(sub["OR"], y_pos, c=colors_dots, s=20, zorder=3, edgecolors="white", linewidths=0.3)
    ax.axvline(1, color=COLORS["light_gray"], linewidth=0.5, linestyle="--", zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([vl.get(v, v) for v in sub["Variable"]], fontsize=6)
    ax.set_xlabel(L("odds_ratio", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_forest_plot_odds_ratios", lang)


# 12. Automation risk by sector -- with translated sector names
def fig_automation_risk_by_sector(lang="en"):
    df = load("automation_sector_summary.csv")
    df = df[df["sector"] != "No especificado"].copy()
    df = df.sort_values("mean_prob", ascending=True)

    sector_names = translate_sectors_list(df["sector"].tolist(), lang)

    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    colors_bar = [COLORS["secondary"] if v > 0.5 else COLORS["tertiary"] for v in df["mean_prob"]]
    ax.barh(range(len(df)), df["mean_prob"], color=colors_bar, height=0.7,
            edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(0.5, color=COLORS["dark_gray"], linewidth=0.5, linestyle=":", zorder=0)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(sector_names, fontsize=5.5)
    ax.set_xlabel(L("mean_prob", lang))
    ax.set_xlim(0, 0.75)
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_automation_risk_by_sector", lang)


# 13. Automation risk by education
def fig_automation_risk_by_education(lang="en"):
    df = load("automation_analysis_dataset.csv", usecols=["education_level", "automation_prob", "high_risk"])
    df = df.dropna(subset=["education_level"])

    edu_order = ["Ninguno", "Primaria", "Secundaria", "Media", "Tecnico/Tecnologico", "Universitario", "Posgrado"]
    edu_labels_en = ["None", "Primary", "Secondary", "Upper sec.", "Technical", "University", "Postgrad."]
    edu_labels_es = ["Ninguno", "Primaria", "Secundaria", "Media", "Técnico", "Universitario", "Posgrado"]
    edu_labels = edu_labels_en if lang == "en" else edu_labels_es

    means = [df[df["education_level"] == e]["automation_prob"].mean() if len(df[df["education_level"] == e]) > 0 else 0
             for e in edu_order]

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


# 14. Heatmap sector x formality -- FIXED: axis label and translations
def fig_heatmap_sector_formality(lang="en"):
    df = load("automation_cross_tabulation.csv")
    df["label"] = df["formal_label"] + " / " + df["high_risk_label"]
    piv = df.pivot_table(index="sector", columns="label", values="mean_prob", aggfunc="first")
    piv = piv.sort_index()

    # Translate column labels for EN
    if lang == "en":
        piv.columns = [FORMALITY_RISK_EN.get(c, c) for c in piv.columns]
        piv.index = translate_sectors_list(piv.index.tolist(), lang)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    cmap = plt.cm.YlOrRd
    sns.heatmap(piv, cmap=cmap, annot=True, fmt=".2f", annot_kws={"size": 5},
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.6, "label": L("mean_prob", lang)},
                ax=ax)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.tick_params(axis="x", labelsize=5.5)
    ax.set_xlabel(L("formality_risk", lang))  # FIXED: was showing "label"
    ax.set_ylabel(L("sector", lang))
    plt.setp(ax.get_xticklabels(), ha="right", rotation=35, rotation_mode="anchor")
    fig.tight_layout()
    savefig(fig, "fig_heatmap_sector_formality", lang)


# 15. Coefficient plots (firm models) -- LARGER panels
def fig_coefficient_plots(lang="en"):
    df = load("table_regression_results.csv")
    models = ["Model_A", "Model_B", "Model_C"]
    model_labels = [L("model_a", lang), L("model_b", lang), L("model_c", lang)]
    vars_a = ["log_ulc", "log_lp", "log_emp"]
    vars_b = ["labor_share_va", "log_lp", "log_emp"]
    vars_c = ["log_lcpw", "log_emp"]
    var_lists = [vars_a, vars_b, vars_c]

    vl_en = {"log_ulc": "log(Unit labor cost)", "log_lp": "log(Labor prod.)",
             "log_emp": "log(Employment)", "labor_share_va": "Labor share (VA)",
             "log_lcpw": "log(Labor cost/wkr)"}
    vl_es = {"log_ulc": "log(CLU)", "log_lp": "log(Prod. laboral)",
             "log_emp": "log(Empleo)", "labor_share_va": "Partic. laboral (VA)",
             "log_lcpw": "log(Costo lab./trab.)"}
    vl = vl_en if lang == "en" else vl_es

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=False)
    for idx, (model, var_list, mlabel) in enumerate(zip(models, var_lists, model_labels)):
        ax = axes[idx]
        mdata = df[(df["Model"] == model) & (df["Variable"].isin(var_list))].copy()
        mdata = mdata.dropna(subset=["Coefficient", "Std_Error"])
        y_pos = np.arange(len(mdata))
        colors_dots = [COLORS["secondary"] if p < 0.05 else COLORS["light_gray"] for p in mdata["p_value"]]
        ax.errorbar(mdata["Coefficient"], y_pos, xerr=1.96 * mdata["Std_Error"],
                    fmt="none", ecolor=COLORS["dark_gray"], elinewidth=0.6, capsize=2.5, capthick=0.5)
        ax.scatter(mdata["Coefficient"], y_pos, c=colors_dots, s=25, zorder=3,
                   edgecolors="white", linewidths=0.3)
        ax.axvline(0, color=COLORS["light_gray"], linewidth=0.5, linestyle="--", zorder=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([vl.get(v, v) for v in mdata["Variable"]], fontsize=6.5)
        ax.set_xlabel(L("coefficient", lang), fontsize=7)
        ax.set_title(mlabel, fontsize=7.5, fontweight="bold")
        despine(ax)

    fig.tight_layout()
    savefig(fig, "fig_coefficient_plots", lang)


# 16. Scatter labor cost vs investment -- FIXED overplotting with hexbin
def fig_scatter_lcpw_vs_invrate(lang="en"):
    df = load("firm_level_merged_dataset.csv")
    df = df.dropna(subset=["labor_cost_per_worker", "investment_rate"])
    df = df[(df["investment_rate"] >= 0) & (df["investment_rate"] < 1)]
    df = df[df["labor_cost_per_worker"] > 0]

    x = np.log(df["labor_cost_per_worker"])
    y = df["investment_rate"]

    fig, ax = plt.subplots(figsize=SINGLE)
    # Use scatter with very low alpha to show density
    ax.scatter(x, y, s=2, alpha=0.08, color=COLORS["primary"], rasterized=True)

    # Fit line + confidence band
    mask = np.isfinite(x) & np.isfinite(y)
    z = np.polyfit(x[mask], y[mask], 1)
    p = np.poly1d(z)
    xr = np.linspace(x.min(), x.max(), 100)
    y_hat = p(xr)
    residuals = y[mask] - p(x[mask])
    se = np.std(residuals)
    ax.plot(xr, y_hat, color=COLORS["secondary"], linewidth=1.0, linestyle="-")
    ax.fill_between(xr, y_hat - 1.96 * se, y_hat + 1.96 * se,
                     alpha=0.1, color=COLORS["secondary"], linewidth=0)

    ax.set_xlabel(L("log_lcpw", lang))
    ax.set_ylabel(L("investment_rate", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_scatter_lcpw_vs_invrate", lang)


# 17. Bubble chart -- FIXED overlapping labels with adjustText
def fig_sector_bubble_lcost_auto(lang="en"):
    df = load("table_sectoral_averages.csv")
    df = df.dropna(subset=["avg_labor_cost_pw", "avg_automation_proxy", "total_employment"])
    df = df[df["avg_automation_proxy"] > 0]

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    sizes = df["total_employment"] / df["total_employment"].max() * 300
    ax.scatter(df["avg_labor_cost_pw"] / 1e3, df["avg_automation_proxy"] / 1e6,
               s=sizes, alpha=0.6, color=COLORS["tertiary"],
               edgecolors=COLORS["primary"], linewidths=0.5, zorder=3)

    texts = []
    for _, row in df.iterrows():
        name = row["sector_name"][:14]
        if lang == "en":
            name = MFG_SUBSECTOR_EN.get(name, MFG_SUBSECTOR_EN.get(row["sector_name"][:12], name))
        t = ax.text(row["avg_labor_cost_pw"] / 1e3, row["avg_automation_proxy"] / 1e6,
                    name, fontsize=4.5, ha="center")
        texts.append(t)

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.25, color=COLORS["light_gray"]),
                force_text=(0.5, 0.5), force_points=(0.3, 0.3))

    ax.set_xlabel(L("labor_cost_pw", lang))
    ax.set_ylabel(L("automation_proxy", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_sector_bubble_lcost_auto", lang)


# 18. Simulation trajectories
def fig_sim_formal_employment_trajectories(lang="en"):
    df = load("simulation_full_results.csv")
    agg = df.groupby(["year", "scenario"])["formal_employment"].sum().reset_index()

    fig, ax = plt.subplots(figsize=SINGLE)
    scenarios = ["Status Quo", "Labor Reform", "Parafiscal Reform", "AI Acceleration", "Combined Worst"]
    for i, s in enumerate(scenarios):
        d = agg[agg["scenario"] == s].sort_values("year")
        label = SCENARIO_NAMES[lang].get(s, s)
        ls = "--" if s == "Combined Worst" else "-"
        lw = 1.2 if s in ("Combined Worst", "Status Quo") else 0.9
        ax.plot(d["year"], d["formal_employment"], label=label,
                color=PALETTE[i], linewidth=lw, linestyle=ls)

    ax.set_xlabel(L("year", lang))
    ax.set_ylabel(L("formal_employment", lang))
    ax.legend(fontsize=5.5, loc="lower left", frameon=False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_sim_formal_employment_trajectories", lang)


# 19. Waterfall (labor reform decomposition)
def fig_sim_waterfall_labor_reform(lang="en"):
    sc = load("simulation_scenario_comparison.csv")
    sq = sc[sc["Scenario"] == "Status Quo"].iloc[0]
    lr = sc[sc["Scenario"] == "Labor Reform"].iloc[0]

    values = [sq["Cumulative Displaced (M)"],
              lr["Cumulative Displaced (M)"] - sq["Cumulative Displaced (M)"],
              lr["Cumulative Informalized (M)"]]
    labels_en = ["Base displacement\n(Status Quo)", "Additional\ndisplacement", "Informalized"]
    labels_es = ["Desplazamiento\nbase (Statu Quo)", "Desplazamiento\nadicional", "Informalizado"]
    labels = labels_en if lang == "en" else labels_es

    cumulative = [0]
    for v in values[:-1]:
        cumulative.append(cumulative[-1] + v)

    fig, ax = plt.subplots(figsize=SINGLE)
    bar_colors = [COLORS["primary"], COLORS["secondary"], COLORS["quinary"]]
    for i, (val, bottom) in enumerate(zip(values, cumulative)):
        ax.bar(i, val, bottom=bottom, color=bar_colors[i], width=0.5,
               edgecolor="white", linewidth=0.3)
        ax.text(i, bottom + val / 2, f"{val:.2f}M", ha="center", va="center", fontsize=6, color="white")

    total = sum(values)
    ax.plot([-0.4, len(values) - 0.6], [total, total], color=COLORS["dark_gray"],
            linewidth=0.5, linestyle="--")
    ax.text(len(values) - 1, total + 0.1, f"Total: {total:.2f}M", fontsize=6,
            ha="center", color=COLORS["dark_gray"])

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(L("displacement", lang))
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_sim_waterfall_labor_reform", lang)


# 20. Displacement heatmap -- FIXED: colorblind-friendly palette (viridis-based)
def fig_sim_heatmap_displacement(lang="en"):
    df = load("simulation_sectoral_breakdown.csv")
    piv = df.pivot_table(index="Sector", columns="Scenario", values="Pct Change (%)", aggfunc="first")
    scenario_order = ["Status Quo", "Parafiscal Reform", "AI Acceleration", "Labor Reform", "Combined Worst"]
    piv = piv[[s for s in scenario_order if s in piv.columns]]

    # Translate
    sector_map = SIM_SECTOR_ES if lang == "es" else SIM_SECTOR_EN
    piv.index = [sector_map.get(s, s) for s in piv.index]
    if lang == "es":
        piv.columns = [SCENARIO_NAMES["es"].get(c, c) for c in piv.columns]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    # Colorblind-friendly: RdYlBu reversed (red=bad, blue=good)
    cmap = plt.cm.RdYlBu
    sns.heatmap(piv, cmap=cmap, annot=True, fmt=".1f", annot_kws={"size": 5.5},
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.7, "label": L("pct_change", lang)},
                ax=ax, center=-25)
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="x", labelsize=6)
    plt.setp(ax.get_xticklabels(), ha="right", rotation=30, rotation_mode="anchor")
    fig.tight_layout()
    savefig(fig, "fig_sim_heatmap_displacement", lang)


# 21. Tornado sensitivity
def fig_sim_tornado_sensitivity(lang="en"):
    df = load("simulation_sensitivity_analysis.csv")
    df = df.sort_values("Range (M)", ascending=True)

    pl_en = {"Automation risk (+/-10pp)": "Automation risk (±10pp)",
             "Elasticity (sigma)": "Elasticity (σ)",
             "Tech cost decline": "Tech. cost decline",
             "Reinstatement rate": "Reinstatement rate",
             "Min wage growth": "Min. wage growth"}
    pl_es = {"Automation risk (+/-10pp)": "Riesgo autom. (±10pp)",
             "Elasticity (sigma)": "Elasticidad (σ)",
             "Tech cost decline": "Caída costo tecnol.",
             "Reinstatement rate": "Tasa de reinstalación",
             "Min wage growth": "Crecimiento sal. mín."}
    pl = pl_en if lang == "en" else pl_es
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
    legend_elements = [Patch(facecolor=COLORS["tertiary"], label=L("low_value", lang)),
                       Patch(facecolor=COLORS["secondary"], label=L("high_value", lang))]
    ax.legend(handles=legend_elements, fontsize=6, loc="lower right", frameon=False)
    despine(ax)
    fig.tight_layout()
    savefig(fig, "fig_sim_tornado_sensitivity", lang)


# 22. Fan chart
def fig_sim_fan_chart_baseline(lang="en"):
    mc = load("simulation_montecarlo_summary.csv")
    full = load("simulation_full_results.csv")
    sq = full[full["scenario"] == "Status Quo"]
    agg = sq.groupby("year")["formal_employment"].sum().reset_index().sort_values("year")
    sq_mc = mc[mc["Scenario"] == "Status Quo"].iloc[0]
    std_dev = sq_mc["Std Dev (M)"]
    initial_formal = agg["formal_employment"].iloc[0]
    years = agg["year"].values
    formal = agg["formal_employment"].values
    displacement_profile = initial_formal - formal
    max_displacement = displacement_profile[-1]
    frac = np.where(max_displacement > 0, displacement_profile / max_displacement, 0)
    sd_at_year = frac * std_dev

    fig, ax = plt.subplots(figsize=SINGLE)
    bands = [(1.645, 0.08), (1.28, 0.12), (0.674, 0.18)]
    for z_val, alpha_val in bands:
        upper = formal + z_val * sd_at_year
        lower = formal - z_val * sd_at_year
        ax.fill_between(years, lower, upper, alpha=alpha_val, color=COLORS["tertiary"], linewidth=0)

    ax.plot(years, formal, color=COLORS["primary"], linewidth=1.0, label=L("median", lang))
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
    print("Regenerating ALL figures (improved Q1 version)")
    print(f"Output: {IMG_DIR}/en/ and {IMG_DIR}/es/")
    print(f"{'='*60}\n")

    success, failures = 0, []
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
