#!/usr/bin/env python3
"""
12_publication_figures.py
=========================
Regenerates ALL 8 figures for the Technovation paper with a unified
Nature/Q1 journal aesthetic. Outputs PNG (300 DPI) + PDF (vector) to
images/en/.

Figures:
  F1  fig_did_event_study          — Event study (investment per worker)
  F2  fig_did_parallel_trends      — Parallel trends (high vs low labor cost)
  F3  fig_did_forest_plot          — Forest/coefficient plot (5 DiD outcomes)
  F4  fig5_robot_density_comparison — Robot density horizontal bar chart
  F5  fig_iva_robustness_rankings  — IVA robustness box plot (231 weight specs)
  F6  fig_iva_sensitivity_heatmap  — Triangular Kendall-tau heatmap
  F7  fig_did_treatment_distribution — Histogram of pre-treatment labor cost share
  F8  fig_roc_curve                — ROC curve for automation risk logit model

Author: Cristian Espinal  |  Date: 2026-03-21
"""

import os
import sys
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
from scipy import stats

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────
BASE = Path("/Users/cristianespinal/Claude Code/Projects/Research/automatizacion_colombia")
DATA = BASE / "data"
OUT = BASE / "images" / "en"
OUT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# FONT DETECTION
# ──────────────────────────────────────────────────────────────────────
_font_set = False
for font in ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"]:
    try:
        fm.findfont(font, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = font
        _font_set = True
        print(f"[font] Using: {font}")
        break
    except Exception:
        continue
if not _font_set:
    matplotlib.rcParams["font.family"] = "sans-serif"
    print("[font] Fallback to generic sans-serif")

# ──────────────────────────────────────────────────────────────────────
# GLOBAL RCPARAMS — Nature / Q1 journal style
# ──────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "axes.axisbelow": True,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "patch.linewidth": 0.5,
})

# ──────────────────────────────────────────────────────────────────────
# COLOR PALETTE — Nature-style, colorblind-safe
# ──────────────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#2C3E50",
    "secondary": "#E74C3C",
    "tertiary": "#3498DB",
    "accent": "#27AE60",
    "gray": "#95A5A6",
    "light_gray": "#ECF0F1",
    "ci_fill": "#2C3E50",
    "highlight": "#E74C3C",
    "nonsig": "#BDC3C7",
}

FULL_WIDTH = (7.0, 4.33)  # golden ratio, 2-column

# ──────────────────────────────────────────────────────────────────────
# HELPER: horizontal-only grid
# ──────────────────────────────────────────────────────────────────────
def set_hgrid(ax):
    """Enable only horizontal grid lines."""
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.3)
    ax.xaxis.grid(False)


def save_fig(fig, name):
    """Save figure as PNG + PDF and close."""
    for ext in (".png", ".pdf"):
        fig.savefig(OUT / (name + ext))
    plt.close(fig)
    print(f"  [saved] {name}.png / .pdf")


# ──────────────────────────────────────────────────────────────────────
# DATA LOADING — defensive checks
# ──────────────────────────────────────────────────────────────────────
data_files = {
    "did_results": DATA / "did_regression_results.csv",
    "eam_panel": DATA / "eam_panel_constructed.csv",
    "iva_validation": DATA / "iva_validation_results.csv",
    "iva_pca": DATA / "iva_pca_weights.csv",
    "automation": DATA / "automation_analysis_dataset.csv",
}

for label, fp in data_files.items():
    if not fp.exists():
        print(f"[ERROR] Missing data file: {fp}")
        sys.exit(1)
    print(f"[data] {label:20s} OK  ({fp.stat().st_size / 1024:.0f} KB)")

did = pd.read_csv(data_files["did_results"])
panel = pd.read_csv(data_files["eam_panel"])
iva = pd.read_csv(data_files["iva_validation"])
pca_w = pd.read_csv(data_files["iva_pca"])
auto = pd.read_csv(data_files["automation"])

print(f"\n{'='*60}")
print("Generating 8 publication figures …")
print(f"{'='*60}\n")

# ======================================================================
# F1: EVENT STUDY — Investment per Worker
# ======================================================================
print("[F1] Event Study — Investment per Worker")

es_row = did[did["label"].str.contains("Event Study.*Investment per Worker", na=False)].iloc[0]

years_pre = [2016, 2017, 2018, 2019, 2020, 2021]
years_post = [2023, 2024]
years_all = years_pre + [2022] + years_post

betas, ci_los, ci_his = [], [], []
for y in years_pre + years_post:
    betas.append(es_row[f"beta_high_cost_x_{y}"])
    ci_los.append(es_row[f"ci_lo_high_cost_x_{y}"])
    ci_his.append(es_row[f"ci_hi_high_cost_x_{y}"])

# Insert reference year 2022 = 0
idx_ref = len(years_pre)  # position 6
betas.insert(idx_ref, 0.0)
ci_los.insert(idx_ref, 0.0)
ci_his.insert(idx_ref, 0.0)

betas = np.array(betas, dtype=float)
ci_los = np.array(ci_los, dtype=float)
ci_his = np.array(ci_his, dtype=float)
years_arr = np.array(years_all)

fig, ax = plt.subplots(figsize=FULL_WIDTH)

# Light background for post-treatment
ax.axvspan(2022.5, 2024.6, color=COLORS["light_gray"], alpha=0.35, zorder=0)

# Zero line
ax.axhline(0, color=COLORS["gray"], linewidth=0.6, linestyle="-", zorder=1)

# Vertical treatment onset
ax.axvline(2022.5, color=COLORS["gray"], linewidth=0.7, linestyle="--", zorder=1)

# Pre-treatment points (navy)
pre_mask = years_arr < 2022.5
post_mask = years_arr > 2022.5
ref_mask = years_arr == 2022

# Error bars — pre
yerr_pre = np.array([betas[pre_mask] - ci_los[pre_mask],
                     ci_his[pre_mask] - betas[pre_mask]])
ax.errorbar(years_arr[pre_mask], betas[pre_mask], yerr=yerr_pre,
            fmt="o", color=COLORS["primary"], markersize=5,
            markerfacecolor=COLORS["primary"], markeredgecolor="white",
            markeredgewidth=0.5, capsize=3, capthick=0.8, linewidth=0.8,
            ecolor=COLORS["primary"], zorder=3, label="Pre-treatment")

# Error bars — post (red)
yerr_post = np.array([betas[post_mask] - ci_los[post_mask],
                      ci_his[post_mask] - betas[post_mask]])
ax.errorbar(years_arr[post_mask], betas[post_mask], yerr=yerr_post,
            fmt="o", color=COLORS["secondary"], markersize=5,
            markerfacecolor=COLORS["secondary"], markeredgecolor="white",
            markeredgewidth=0.5, capsize=3, capthick=0.8, linewidth=0.8,
            ecolor=COLORS["secondary"], zorder=3, label="Post-treatment")

# Reference year — open diamond
ax.plot(2022, 0, marker="D", markersize=6, markerfacecolor="white",
        markeredgecolor=COLORS["primary"], markeredgewidth=1.0, zorder=4,
        label="Reference (2022)")

# Re-place treatment onset text after axis limits are set
ax.set_xlim(2015.4, 2024.6)
ymin, ymax = ax.get_ylim()
ax.text(2022.6, ymin + 0.05 * (ymax - ymin), "Treatment onset",
        fontsize=6, color=COLORS["gray"], rotation=90, va="bottom", ha="left")

ax.set_xlabel("Year")
ax.set_ylabel("Coefficient (relative to 2022)")
ax.set_xticks(years_all)
ax.legend(frameon=False, loc="upper left")
set_hgrid(ax)
fig.tight_layout()
save_fig(fig, "fig_did_event_study")


# ======================================================================
# F2: PARALLEL TRENDS
# ======================================================================
print("[F2] Parallel Trends")

trends = (panel.groupby(["periodo", "high_cost"])["log_automation_proxy"]
          .mean().unstack())

fig, ax = plt.subplots(figsize=FULL_WIDTH)

ax.axvline(2022.5, color=COLORS["gray"], linewidth=0.7, linestyle="--", zorder=1)
ax.text(2022.6, trends.values.min() + 0.02 * (trends.values.max() - trends.values.min()),
        "Treatment", fontsize=6, color=COLORS["gray"], va="bottom", ha="left")

ax.plot(trends.index, trends[1.0], marker="o", color=COLORS["primary"],
        markersize=5, markerfacecolor=COLORS["primary"], markeredgecolor="white",
        markeredgewidth=0.5, label="High labor cost", zorder=3)
ax.plot(trends.index, trends[0.0], marker="o", color=COLORS["secondary"],
        markersize=5, markerfacecolor=COLORS["secondary"], markeredgecolor="white",
        markeredgewidth=0.5, label="Low labor cost", zorder=3)

ax.set_xlabel("Year")
ax.set_ylabel("Log investment per worker")
ax.set_xticks(sorted(panel["periodo"].unique()))
ax.legend(frameon=False, loc="best")
set_hgrid(ax)
fig.tight_layout()
save_fig(fig, "fig_did_parallel_trends")


# ======================================================================
# F3: FOREST PLOT — Coefficient plot for 5 Binary DiD outcomes
# ======================================================================
print("[F3] Forest Plot — Binary DiD coefficients")

binary_rows = did[did["label"].str.startswith("Binary DiD")].copy()
binary_rows = binary_rows.reset_index(drop=True)

# Extract short outcome names
outcome_map = {
    "Investment per Worker": "Investment per worker",
    "Capital Intensity": "Capital intensity",
    "Investment Rate": "Investment rate",
    "Unit Labor Cost": "Unit labor cost",
    "Labor Productivity": "Labor productivity",
}

labels = []
for lbl in binary_rows["label"]:
    short = lbl.replace("Binary DiD — ", "").replace("Binary DiD - ", "")
    labels.append(outcome_map.get(short, short))

betas_f = binary_rows["beta"].values.astype(float)
ci_lo_f = binary_rows["ci_lo"].values.astype(float)
ci_hi_f = binary_rows["ci_hi"].values.astype(float)
pvals_f = binary_rows["pval"].values.astype(float)

fig, ax = plt.subplots(figsize=FULL_WIDTH)

y_pos = np.arange(len(labels))

# Vertical zero line
ax.axvline(0, color=COLORS["gray"], linewidth=0.6, linestyle="--", zorder=1)

for i in range(len(labels)):
    sig = pvals_f[i] < 0.05
    color = COLORS["primary"] if sig else COLORS["nonsig"]
    face = color if sig else "white"
    edge = color

    ax.plot(betas_f[i], y_pos[i], "o", markersize=6,
            markerfacecolor=face, markeredgecolor=edge,
            markeredgewidth=0.8, zorder=3)
    ax.hlines(y_pos[i], ci_lo_f[i], ci_hi_f[i],
              color=color, linewidth=1.0, zorder=2)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Coefficient estimate (95% CI)")
ax.invert_yaxis()
set_hgrid(ax)

# Add significance annotation
for i in range(len(labels)):
    sig = pvals_f[i] < 0.05
    star = "**" if pvals_f[i] < 0.01 else ("*" if sig else "")
    if star:
        ax.text(ci_hi_f[i] + 0.005, y_pos[i], star, fontsize=7,
                va="center", ha="left", color=COLORS["primary"])

fig.tight_layout()
save_fig(fig, "fig_did_forest_plot")


# ======================================================================
# F4: ROBOT DENSITY COMPARISON — horizontal bar chart
# ======================================================================
print("[F4] Robot Density Comparison")

robot_data = pd.DataFrame({
    "country": ["South Korea", "Singapore", "Germany", "Japan", "China",
                 "Sweden", "USA", "Italy", "France", "Spain",
                 "Mexico", "Brazil", "Colombia"],
    "density": [1012, 770, 429, 419, 406, 321, 295, 237, 180, 160, 40, 18, 5],
})
robot_data = robot_data.sort_values("density", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=FULL_WIDTH)

bar_colors = [COLORS["highlight"] if c == "Colombia" else COLORS["nonsig"]
              for c in robot_data["country"]]

bars = ax.barh(robot_data["country"], robot_data["density"], color=bar_colors,
               edgecolor="white", linewidth=0.3, height=0.65, zorder=2)

# Value labels
for bar, val in zip(bars, robot_data["density"]):
    ax.text(bar.get_width() + 12, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", fontsize=7,
            color=COLORS["primary"], fontweight="bold" if val == 5 else "normal")

ax.set_xlabel("Robots per 10,000 manufacturing workers")
ax.set_xlim(0, robot_data["density"].max() * 1.12)
set_hgrid(ax)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
fig.tight_layout()
save_fig(fig, "fig5_robot_density_comparison")


# ======================================================================
# F5: IVA ROBUSTNESS RANKINGS — box plot from 231 weight specs
# ======================================================================
print("[F5] IVA Robustness Rankings (231 weight specs)")

# Generate 231 weight combinations (w1 + w2 + w3 = 1, step = 0.05)
step = 0.05
weight_combos = []
w1_range = np.arange(0, 1 + step / 2, step)
for w1 in w1_range:
    for w2 in np.arange(0, 1 - w1 + step / 2, step):
        w3 = 1.0 - w1 - w2
        if w3 >= -1e-9:
            weight_combos.append((round(w1, 2), round(w2, 2), round(max(w3, 0), 2)))

print(f"  {len(weight_combos)} weight combinations generated")

# Compute IVA for each weight combo and rank sectors
sectors = iva["sector"].tolist()
c1 = iva["C1_norm"].values
c2 = iva["C2_norm"].values
c3 = iva["C3_norm"].values

all_ranks = {s: [] for s in sectors}

for w1, w2, w3 in weight_combos:
    iva_scores = w1 * c1 + w2 * c2 + w3 * c3
    # Rank: 1 = highest IVA = most vulnerable
    order = np.argsort(-iva_scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    for j, s in enumerate(sectors):
        all_ranks[s].append(ranks[j])

# Baseline ranking (original 3-component: w1=0.40, w2=0.35, w3=0.25)
baseline_ranks = dict(zip(iva["sector"], iva["rank_original_3c"].astype(int)))

# Sort by median rank
sector_medians = {s: np.median(r) for s, r in all_ranks.items()}
sorted_sectors = sorted(sector_medians.keys(), key=lambda s: sector_medians[s])

fig, ax = plt.subplots(figsize=FULL_WIDTH)

box_data = [all_ranks[s] for s in sorted_sectors]
n_sectors = len(sorted_sectors)

# Colormap: darker navy for low median rank (most vulnerable), lighter for high
norm = plt.Normalize(1, n_sectors)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "navy_gray", [COLORS["primary"], COLORS["nonsig"]])

bp = ax.boxplot(box_data, vert=False, patch_artist=True,
                widths=0.55, showfliers=False,
                medianprops=dict(color="white", linewidth=1.0),
                whiskerprops=dict(color=COLORS["gray"], linewidth=0.6),
                capprops=dict(color=COLORS["gray"], linewidth=0.6))

for i, (patch, sec) in enumerate(zip(bp["boxes"], sorted_sectors)):
    median_rank = sector_medians[sec]
    color = cmap(norm(median_rank))
    patch.set_facecolor(color)
    patch.set_edgecolor(COLORS["primary"])
    patch.set_linewidth(0.4)
    # Baseline ranking: star
    bl_rank = baseline_ranks.get(sec, None)
    if bl_rank is not None:
        ax.plot(bl_rank, i + 1, marker="*", markersize=7,
                color="black", zorder=4)

ax.set_yticklabels(sorted_sectors)
ax.set_xlabel("Rank (1 = most vulnerable)")
ax.set_xlim(0.5, n_sectors + 0.5)
ax.set_xticks(range(1, n_sectors + 1))
set_hgrid(ax)
ax.yaxis.grid(False)

# Legend for star
ax.plot([], [], marker="*", markersize=7, color="black", linestyle="None",
        label="Baseline rank")
ax.legend(frameon=False, loc="lower right")

fig.tight_layout()
save_fig(fig, "fig_iva_robustness_rankings")


# ======================================================================
# F6: IVA SENSITIVITY HEATMAP — triangular Kendall tau
# ======================================================================
print("[F6] IVA Sensitivity Heatmap (Kendall tau)")

# Baseline weights
w1_base, w2_base, w3_base = 0.40, 0.35, 0.25

# PCA weights (3-component)
pca_row = pca_w[["Component", "PCA_3c"]].set_index("Component")
w1_pca = float(pca_row.loc["C1_auto_potential", "PCA_3c"])
w2_pca = float(pca_row.loc["C2_labor_cost", "PCA_3c"])
w3_pca = float(pca_row.loc["C3_formality", "PCA_3c"])

# Baseline ranking vector
baseline_scores = w1_base * c1 + w2_base * c2 + w3_base * c3
baseline_order = np.argsort(-baseline_scores)
baseline_rank_vec = np.empty_like(baseline_order)
baseline_rank_vec[baseline_order] = np.arange(1, len(baseline_order) + 1)

# Grid for heatmap (finer resolution)
grid_step = 0.02
w1_grid = np.arange(0, 1 + grid_step / 2, grid_step)
w2_grid = np.arange(0, 1 + grid_step / 2, grid_step)
tau_matrix = np.full((len(w1_grid), len(w2_grid)), np.nan)

for i, w1 in enumerate(w1_grid):
    for j, w2 in enumerate(w2_grid):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9:
            continue
        w3 = max(w3, 0)
        scores = w1 * c1 + w2 * c2 + w3 * c3
        order = np.argsort(-scores)
        rank_vec = np.empty_like(order)
        rank_vec[order] = np.arange(1, len(order) + 1)
        tau, _ = stats.kendalltau(baseline_rank_vec, rank_vec)
        tau_matrix[i, j] = tau

fig, ax = plt.subplots(figsize=FULL_WIDTH)

# Mask upper triangle (w1 + w2 > 1)
masked = np.ma.masked_invalid(tau_matrix.T)  # transpose so x=w1, y=w2

im = ax.pcolormesh(w1_grid, w2_grid, masked, cmap="RdBu_r",
                   vmin=-1, vmax=1, shading="nearest", rasterized=True)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Kendall's τ (vs. baseline)", fontsize=7)
cbar.ax.tick_params(labelsize=6)

# Mark original weights
ax.plot(w1_base, w2_base, marker="*", markersize=10,
        color="black", markeredgecolor="white", markeredgewidth=0.5, zorder=5)
ax.annotate("Baseline", (w1_base, w2_base), textcoords="offset points",
            xytext=(6, 6), fontsize=6, color="black", fontweight="bold")

# Mark PCA weights
ax.plot(w1_pca, w2_pca, marker="D", markersize=7,
        color="black", markeredgecolor="white", markeredgewidth=0.5, zorder=5)
ax.annotate("PCA", (w1_pca, w2_pca), textcoords="offset points",
            xytext=(6, -8), fontsize=6, color="black", fontweight="bold")

# Diagonal boundary line (w1 + w2 = 1)
ax.plot([0, 1], [1, 0], color="black", linewidth=0.6, linestyle="-", zorder=4)

# Note
ax.text(0.70, 0.70, "$w_3 = 1 - w_1 - w_2$", fontsize=6,
        color=COLORS["gray"], style="italic", ha="center",
        transform=ax.transData)

ax.set_xlabel("$w_1$ (Automation Potential)")
ax.set_ylabel("$w_2$ (Labor Cost Incentive)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_aspect("equal")

# Re-enable spines for heatmap
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.xaxis.grid(False)
ax.yaxis.grid(False)

fig.tight_layout()
save_fig(fig, "fig_iva_sensitivity_heatmap")


# ======================================================================
# F7: TREATMENT DISTRIBUTION — histogram of labor cost share
# ======================================================================
print("[F7] Treatment Distribution — labor cost share (2022)")

# Get unique firm-level labor_cost_share_2022 (one per firm)
lcs = panel.dropna(subset=["labor_cost_share_2022"])
# Use one observation per firm (they should be constant across years)
lcs_unique = lcs.drop_duplicates(subset=["nordemp"])["labor_cost_share_2022"]

# Clip to 99th percentile to remove extreme outliers for visualization
p99 = lcs_unique.quantile(0.99)
lcs_plot = lcs_unique[lcs_unique <= p99]

median_val = lcs_unique.median()
q25 = lcs_unique.quantile(0.25)
q75 = lcs_unique.quantile(0.75)

fig, ax = plt.subplots(figsize=FULL_WIDTH)

ax.hist(lcs_plot, bins=40, density=True, color=COLORS["light_gray"],
        edgecolor=COLORS["primary"], linewidth=0.4, zorder=2)

# Median line
ax.axvline(median_val, color=COLORS["primary"], linewidth=1.0,
           linestyle="--", zorder=3)
ymax_hist = ax.get_ylim()[1]
ax.text(median_val + 0.01, ymax_hist * 0.92,
        f"Median = {median_val:.3f}", fontsize=7,
        color=COLORS["primary"], va="top", ha="left")

# Quartile lines
ax.axvline(q25, color=COLORS["gray"], linewidth=0.7, linestyle=":", zorder=2)
ax.axvline(q75, color=COLORS["gray"], linewidth=0.7, linestyle=":", zorder=2)
ax.text(q25 - 0.01, ymax_hist * 0.80, "Q25", fontsize=6,
        color=COLORS["gray"], va="top", ha="right")
ax.text(q75 + 0.01, ymax_hist * 0.80, "Q75", fontsize=6,
        color=COLORS["gray"], va="top", ha="left")

ax.set_xlabel("Pre-treatment labor cost share (2022)")
ax.set_ylabel("Density")
set_hgrid(ax)
fig.tight_layout()
save_fig(fig, "fig_did_treatment_distribution")


# ======================================================================
# F8: ROC CURVE — logistic model for automation risk
# ======================================================================
print("[F8] ROC Curve — Automation risk logit")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Prepare data
roc_df = auto.copy()

# Binary outcome
y_col = "high_risk"
# Predictor columns
x_cols = ["formal", "female", "age", "education_num", "log_income",
          "hours_worked"]

# Also try firm_size as categorical
firm_size_map = {
    "Micro (1)": 1, "Micro (2-10)": 2, "Pequena (11-50)": 3,
    "Mediana (51-200)": 4, "Grande (201+)": 5, "Desconocido": np.nan,
}
roc_df["firm_size_num"] = roc_df["firm_size"].map(firm_size_map)

pred_cols = x_cols + ["firm_size_num"]

# Drop missing
roc_clean = roc_df.dropna(subset=pred_cols + [y_col]).copy()
print(f"  ROC data: {len(roc_clean)} obs, {roc_clean[y_col].mean():.3f} positive rate")

X = roc_clean[pred_cols].values.astype(float)
y = roc_clean[y_col].values.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

y_prob = model.predict_proba(X_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_prob)
auc_val = roc_auc_score(y, y_prob)

# Optimal threshold (Youden's J)
j_scores = tpr - fpr
opt_idx = np.argmax(j_scores)
opt_fpr, opt_tpr = fpr[opt_idx], tpr[opt_idx]

print(f"  AUC = {auc_val:.3f}")

fig, ax = plt.subplots(figsize=FULL_WIDTH)

# Diagonal reference
ax.plot([0, 1], [0, 1], color=COLORS["gray"], linewidth=0.7,
        linestyle="--", zorder=1, label="Random classifier")

# ROC curve
ax.plot(fpr, tpr, color=COLORS["primary"], linewidth=1.2,
        zorder=3, label=f"Logit (AUC = {auc_val:.3f})")

# Optimal threshold point
ax.plot(opt_fpr, opt_tpr, "o", markersize=6,
        markerfacecolor=COLORS["secondary"], markeredgecolor="white",
        markeredgewidth=0.5, zorder=4, label="Optimal threshold")

ax.set_xlabel("False positive rate (1 − Specificity)")
ax.set_ylabel("True positive rate (Sensitivity)")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.legend(frameon=False, loc="lower right")
set_hgrid(ax)
fig.tight_layout()
save_fig(fig, "fig_roc_curve")


# ======================================================================
# SUMMARY TABLE
# ======================================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Figure':<40s} {'PNG KB':>8s}  {'PDF KB':>8s}")
print("-" * 60)

figure_names = [
    "fig_did_event_study",
    "fig_did_parallel_trends",
    "fig_did_forest_plot",
    "fig5_robot_density_comparison",
    "fig_iva_robustness_rankings",
    "fig_iva_sensitivity_heatmap",
    "fig_did_treatment_distribution",
    "fig_roc_curve",
]

for name in figure_names:
    png_path = OUT / f"{name}.png"
    pdf_path = OUT / f"{name}.pdf"
    png_kb = png_path.stat().st_size / 1024 if png_path.exists() else 0
    pdf_kb = pdf_path.stat().st_size / 1024 if pdf_path.exists() else 0
    print(f"  {name:<38s} {png_kb:>7.0f}  {pdf_kb:>7.0f}")

print(f"\nAll 8 figures saved to: {OUT}")
print("Done.")
