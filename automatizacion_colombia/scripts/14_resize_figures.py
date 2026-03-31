#!/usr/bin/env python3
"""
14_resize_figures.py
=====================
Regenerates ALL 10 publication figures at journal-appropriate sizes for
Technovation's 2-column layout. Preserves the exact same visual style
(Helvetica, Nature/Q1 aesthetic, same colors) from 12_publication_figures.py.

Figure sizes (Technovation column width = 3.39 in, page width = 7.08 in):
  SINGLE column: 3.39 x 2.5 in   (most figures)
  DOUBLE column: 7.08 x 2.5-3.0 in (complex figures)

Figures:
  F1   fig_did_event_study          — SINGLE (3.39 x 2.5)
  F2   fig_did_parallel_trends      — SINGLE (3.39 x 2.5)
  F3   fig_did_forest_plot          — DOUBLE (7.08 x 2.5)
  F4   fig5_robot_density_comparison — SINGLE (3.39 x 2.8)
  F5   fig_iva_robustness_rankings  — DOUBLE (7.08 x 3.0)
  F6   fig_iva_sensitivity_heatmap  — SINGLE (3.39 x 3.39)
  F7   fig_did_treatment_distribution — SINGLE (3.39 x 2.5)
  F8   fig_roc_curve                — SINGLE (3.39 x 2.5)
  F9   fig_did_event_formality      — SINGLE (3.39 x 2.5)
  F10  fig_did_event_automation     — SINGLE (3.39 x 2.5)

Author: Cristian Espinal  |  Date: 2026-03-25
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
# GLOBAL RCPARAMS — Nature / Q1 journal style (identical to script 12)
# ──────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
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
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "patch.linewidth": 0.5,
})

# ──────────────────────────────────────────────────────────────────────
# COLOR PALETTE — Nature-style, colorblind-safe (identical to script 12)
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

# ──────────────────────────────────────────────────────────────────────
# FIGURE SIZES — Technovation journal dimensions
# ──────────────────────────────────────────────────────────────────────
SINGLE_COL = (3.39, 2.5)       # single column width
SINGLE_TALL = (3.39, 2.8)      # single column, taller
SINGLE_SQUARE = (3.39, 3.39)   # single column, square
DOUBLE_COL = (7.08, 2.5)       # double column width
DOUBLE_TALL = (7.08, 3.0)      # double column, taller

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
    "geih_cells": DATA / "geih_did_dataset.csv",
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
geih_cells = pd.read_csv(data_files["geih_cells"])

print(f"\n{'='*60}")
print("Generating 10 publication figures (journal-sized) ...")
print(f"{'='*60}\n")

# ======================================================================
# F1: EVENT STUDY — Investment per Worker — SINGLE COLUMN
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

fig, ax = plt.subplots(figsize=SINGLE_COL)

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
            fmt="o", color=COLORS["primary"], markersize=4,
            markerfacecolor=COLORS["primary"], markeredgecolor="white",
            markeredgewidth=0.4, capsize=2, capthick=0.6, linewidth=0.6,
            ecolor=COLORS["primary"], zorder=3, label="Pre-treatment")

# Error bars — post (red)
yerr_post = np.array([betas[post_mask] - ci_los[post_mask],
                      ci_his[post_mask] - betas[post_mask]])
ax.errorbar(years_arr[post_mask], betas[post_mask], yerr=yerr_post,
            fmt="o", color=COLORS["secondary"], markersize=4,
            markerfacecolor=COLORS["secondary"], markeredgecolor="white",
            markeredgewidth=0.4, capsize=2, capthick=0.6, linewidth=0.6,
            ecolor=COLORS["secondary"], zorder=3, label="Post-treatment")

# Reference year — open diamond
ax.plot(2022, 0, marker="D", markersize=5, markerfacecolor="white",
        markeredgecolor=COLORS["primary"], markeredgewidth=0.8, zorder=4,
        label="Reference (2022)")

# Re-place treatment onset text after axis limits are set
ax.set_xlim(2015.4, 2024.6)
ymin, ymax = ax.get_ylim()
ax.text(2022.6, ymin + 0.05 * (ymax - ymin), "Treatment onset",
        fontsize=5, color=COLORS["gray"], rotation=90, va="bottom", ha="left")

ax.set_xlabel("Year")
ax.set_ylabel("Coefficient (relative to 2022)")
ax.set_xticks(years_all)
ax.set_xticklabels([str(y)[2:] for y in years_all])  # short labels: '16, '17...
ax.legend(frameon=False, loc="upper left", fontsize=5)
set_hgrid(ax)
fig.tight_layout()
save_fig(fig, "fig_did_event_study")


# ======================================================================
# F2: PARALLEL TRENDS — SINGLE COLUMN
# ======================================================================
print("[F2] Parallel Trends")

trends = (panel.groupby(["periodo", "high_cost"])["log_automation_proxy"]
          .mean().unstack())

fig, ax = plt.subplots(figsize=SINGLE_COL)

ax.axvline(2022.5, color=COLORS["gray"], linewidth=0.7, linestyle="--", zorder=1)
ax.text(2022.6, trends.values.min() + 0.02 * (trends.values.max() - trends.values.min()),
        "Treatment", fontsize=5, color=COLORS["gray"], va="bottom", ha="left")

ax.plot(trends.index, trends[1.0], marker="o", color=COLORS["primary"],
        markersize=4, markerfacecolor=COLORS["primary"], markeredgecolor="white",
        markeredgewidth=0.4, label="High labor cost", zorder=3)
ax.plot(trends.index, trends[0.0], marker="o", color=COLORS["secondary"],
        markersize=4, markerfacecolor=COLORS["secondary"], markeredgecolor="white",
        markeredgewidth=0.4, label="Low labor cost", zorder=3)

ax.set_xlabel("Year")
ax.set_ylabel("Log investment per worker")
ax.set_xticks(sorted(panel["periodo"].unique()))
ax.set_xticklabels([str(y)[2:] for y in sorted(panel["periodo"].unique())])
ax.legend(frameon=False, loc="best", fontsize=5)
set_hgrid(ax)
fig.tight_layout()
save_fig(fig, "fig_did_parallel_trends")


# ======================================================================
# F3: FOREST PLOT — DOUBLE COLUMN (needs width for labels + CI bars)
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

fig, ax = plt.subplots(figsize=DOUBLE_COL)

y_pos = np.arange(len(labels))

# Vertical zero line
ax.axvline(0, color=COLORS["gray"], linewidth=0.6, linestyle="--", zorder=1)

for i in range(len(labels)):
    sig = pvals_f[i] < 0.05
    color = COLORS["primary"] if sig else COLORS["nonsig"]
    face = color if sig else "white"
    edge = color

    ax.plot(betas_f[i], y_pos[i], "o", markersize=5,
            markerfacecolor=face, markeredgecolor=edge,
            markeredgewidth=0.6, zorder=3)
    ax.hlines(y_pos[i], ci_lo_f[i], ci_hi_f[i],
              color=color, linewidth=0.8, zorder=2)

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
        ax.text(ci_hi_f[i] + 0.005, y_pos[i], star, fontsize=6,
                va="center", ha="left", color=COLORS["primary"])

fig.tight_layout()
save_fig(fig, "fig_did_forest_plot")


# ======================================================================
# F4: ROBOT DENSITY COMPARISON — SINGLE COLUMN (taller)
# ======================================================================
print("[F4] Robot Density Comparison")

robot_data = pd.DataFrame({
    "country": ["South Korea", "Singapore", "Germany", "Japan", "China",
                 "Sweden", "USA", "Italy", "France", "Spain",
                 "Mexico", "Brazil", "Colombia"],
    "density": [1012, 770, 429, 419, 406, 321, 295, 237, 180, 160, 40, 18, 5],
})
robot_data = robot_data.sort_values("density", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=SINGLE_TALL)

bar_colors = [COLORS["highlight"] if c == "Colombia" else COLORS["nonsig"]
              for c in robot_data["country"]]

bars = ax.barh(robot_data["country"], robot_data["density"], color=bar_colors,
               edgecolor="white", linewidth=0.3, height=0.65, zorder=2)

# Value labels
for bar, val in zip(bars, robot_data["density"]):
    ax.text(bar.get_width() + 12, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", fontsize=5,
            color=COLORS["primary"], fontweight="bold" if val == 5 else "normal")

ax.set_xlabel("Robots per 10,000 mfg. workers")
ax.set_xlim(0, robot_data["density"].max() * 1.12)
set_hgrid(ax)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
fig.tight_layout()
save_fig(fig, "fig5_robot_density_comparison")


# ======================================================================
# F5: IVA ROBUSTNESS RANKINGS — DOUBLE COLUMN (taller)
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

fig, ax = plt.subplots(figsize=DOUBLE_TALL)

box_data = [all_ranks[s] for s in sorted_sectors]
n_sectors = len(sorted_sectors)

# Colormap: darker navy for low median rank (most vulnerable), lighter for high
norm = plt.Normalize(1, n_sectors)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "navy_gray", [COLORS["primary"], COLORS["nonsig"]])

bp = ax.boxplot(box_data, vert=False, patch_artist=True,
                widths=0.55, showfliers=False,
                medianprops=dict(color="white", linewidth=0.8),
                whiskerprops=dict(color=COLORS["gray"], linewidth=0.5),
                capprops=dict(color=COLORS["gray"], linewidth=0.5))

for i, (patch, sec) in enumerate(zip(bp["boxes"], sorted_sectors)):
    median_rank = sector_medians[sec]
    color = cmap(norm(median_rank))
    patch.set_facecolor(color)
    patch.set_edgecolor(COLORS["primary"])
    patch.set_linewidth(0.3)
    # Baseline ranking: star
    bl_rank = baseline_ranks.get(sec, None)
    if bl_rank is not None:
        ax.plot(bl_rank, i + 1, marker="*", markersize=6,
                color="black", zorder=4)

ax.set_yticklabels(sorted_sectors)
ax.set_xlabel("Rank (1 = most vulnerable)")
ax.set_xlim(0.5, n_sectors + 0.5)
ax.set_xticks(range(1, n_sectors + 1))
set_hgrid(ax)
ax.yaxis.grid(False)

# Legend for star
ax.plot([], [], marker="*", markersize=6, color="black", linestyle="None",
        label="Baseline rank")
ax.legend(frameon=False, loc="lower right", fontsize=5)

fig.tight_layout()
save_fig(fig, "fig_iva_robustness_rankings")


# ======================================================================
# F6: IVA SENSITIVITY HEATMAP — SINGLE COLUMN (square)
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

fig, ax = plt.subplots(figsize=SINGLE_SQUARE)

# Mask upper triangle (w1 + w2 > 1)
masked = np.ma.masked_invalid(tau_matrix.T)  # transpose so x=w1, y=w2

im = ax.pcolormesh(w1_grid, w2_grid, masked, cmap="RdBu_r",
                   vmin=-1, vmax=1, shading="nearest", rasterized=True)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Kendall's $\\tau$ (vs. baseline)", fontsize=6)
cbar.ax.tick_params(labelsize=5)

# Mark original weights
ax.plot(w1_base, w2_base, marker="*", markersize=8,
        color="black", markeredgecolor="white", markeredgewidth=0.4, zorder=5)
ax.annotate("Baseline", (w1_base, w2_base), textcoords="offset points",
            xytext=(5, 5), fontsize=5, color="black", fontweight="bold")

# Mark PCA weights
ax.plot(w1_pca, w2_pca, marker="D", markersize=5,
        color="black", markeredgecolor="white", markeredgewidth=0.4, zorder=5)
ax.annotate("PCA", (w1_pca, w2_pca), textcoords="offset points",
            xytext=(5, -7), fontsize=5, color="black", fontweight="bold")

# Diagonal boundary line (w1 + w2 = 1)
ax.plot([0, 1], [1, 0], color="black", linewidth=0.6, linestyle="-", zorder=4)

# Note
ax.text(0.70, 0.70, "$w_3 = 1 - w_1 - w_2$", fontsize=5,
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
# F7: TREATMENT DISTRIBUTION — SINGLE COLUMN
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

fig, ax = plt.subplots(figsize=SINGLE_COL)

ax.hist(lcs_plot, bins=40, density=True, color=COLORS["light_gray"],
        edgecolor=COLORS["primary"], linewidth=0.3, zorder=2)

# Median line
ax.axvline(median_val, color=COLORS["primary"], linewidth=0.8,
           linestyle="--", zorder=3)
ymax_hist = ax.get_ylim()[1]
ax.text(median_val + 0.01, ymax_hist * 0.92,
        f"Median = {median_val:.3f}", fontsize=5,
        color=COLORS["primary"], va="top", ha="left")

# Quartile lines
ax.axvline(q25, color=COLORS["gray"], linewidth=0.5, linestyle=":", zorder=2)
ax.axvline(q75, color=COLORS["gray"], linewidth=0.5, linestyle=":", zorder=2)
ax.text(q25 - 0.01, ymax_hist * 0.80, "Q25", fontsize=5,
        color=COLORS["gray"], va="top", ha="right")
ax.text(q75 + 0.01, ymax_hist * 0.80, "Q75", fontsize=5,
        color=COLORS["gray"], va="top", ha="left")

ax.set_xlabel("Pre-treatment labor cost share (2022)")
ax.set_ylabel("Density")
set_hgrid(ax)
fig.tight_layout()
save_fig(fig, "fig_did_treatment_distribution")


# ======================================================================
# F8: ROC CURVE — SINGLE COLUMN
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

fig, ax = plt.subplots(figsize=SINGLE_COL)

# Diagonal reference
ax.plot([0, 1], [0, 1], color=COLORS["gray"], linewidth=0.7,
        linestyle="--", zorder=1, label="Random classifier")

# ROC curve
ax.plot(fpr, tpr, color=COLORS["primary"], linewidth=1.0,
        zorder=3, label=f"Logit (AUC = {auc_val:.3f})")

# Optimal threshold point
ax.plot(opt_fpr, opt_tpr, "o", markersize=5,
        markerfacecolor=COLORS["secondary"], markeredgecolor="white",
        markeredgewidth=0.4, zorder=4, label="Optimal threshold")

ax.set_xlabel("False positive rate (1 - Specificity)")
ax.set_ylabel("True positive rate (Sensitivity)")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.legend(frameon=False, loc="lower right", fontsize=5)
set_hgrid(ax)
fig.tight_layout()
save_fig(fig, "fig_roc_curve")


# ======================================================================
# F9 & F10: GEIH EVENT STUDY FIGURES — SINGLE COLUMN
# ======================================================================
print("[F9/F10] GEIH Event Study Figures (formality + automation)")

import statsmodels.api as sm

def run_geih_event_study(cells, outcome, outcome_label):
    """Run event study regression at the cell level and return coefficients."""
    data = cells.dropna(subset=[outcome, "sector_mw_bite_2022"]).copy()
    data["sector_fe"] = data["sector_2d"].astype(str)

    ref_idx = 3  # 2022q4
    time_indices = sorted(data["time_idx"].unique())
    time_indices_no_ref = [t for t in time_indices if t != ref_idx]

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

    y_vals = data[outcome].astype(float)
    w = data["total_employment"].astype(float)

    model_wls = sm.WLS(y_vals, X, weights=w)
    results = model_wls.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["sector_fe"].values},
    )

    event_coefs = []
    for t in time_indices:
        col = f"bite_x_t{t}"
        if t == ref_idx:
            event_coefs.append({
                "time_idx": t, "beta": 0.0, "se": 0.0,
                "ci_low": 0.0, "ci_high": 0.0,
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
    idx_to_label = {}
    for yr in [2022, 2023, 2024]:
        for q_i, q in enumerate([1, 2, 3, 4], start=0):
            idx = (yr - 2022) * 4 + q_i
            idx_to_label[idx] = f"{yr}q{q}"
    event_df["year_quarter"] = event_df["time_idx"].map(idx_to_label)

    print(f"  Event study: {outcome_label}")
    print(event_df[["year_quarter", "beta", "ci_low", "ci_high"]].to_string(index=False))

    return event_df


def plot_geih_event_study(event_df, outcome_label, filename_stem):
    """Plot GEIH event study with journal-appropriate style."""
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    x = event_df["time_idx"].values
    y_vals = event_df["beta"].values
    ci_lo = event_df["ci_low"].values
    ci_hi = event_df["ci_high"].values

    # Shade post-treatment period
    ax.axvspan(3.5, x.max() + 0.5, alpha=0.08, color=COLORS["secondary"],
               zorder=0)

    # Reference line
    ax.axhline(y=0, color=COLORS["gray"], linewidth=0.6, linestyle="--", zorder=1)

    # Vertical line at treatment onset
    ax.axvline(x=3.5, color=COLORS["gray"], linewidth=0.7, linestyle=":", zorder=1)

    # CI band
    ax.fill_between(x, ci_lo, ci_hi, alpha=0.20, color=COLORS["primary"])

    # Point estimates
    ax.plot(x, y_vals, "o-", color=COLORS["primary"], markersize=4, linewidth=1.0, zorder=5)

    # Mark reference period
    ref_idx = 3
    if ref_idx in x:
        ax.plot(ref_idx, 0, "D", color=COLORS["secondary"], markersize=5, zorder=6)

    # Labels
    labels_q = event_df["year_quarter"].values
    # Shorten labels
    short_labels = [l.replace("20", "'") for l in labels_q]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Coefficient (MW bite interaction)")
    set_hgrid(ax)
    fig.tight_layout()
    save_fig(fig, filename_stem)


# Run the event studies
try:
    ev_form = run_geih_event_study(geih_cells, "formality_rate", "Formality Rate")
    plot_geih_event_study(ev_form, "Formality Rate", "fig_did_event_formality")
except Exception as e:
    print(f"  ERROR generating formality event study: {e}")

try:
    ev_auto = run_geih_event_study(geih_cells, "mean_automation_risk", "Mean Automation Risk")
    plot_geih_event_study(ev_auto, "Mean Automation Risk", "fig_did_event_automation")
except Exception as e:
    print(f"  ERROR generating automation event study: {e}")


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
    "fig_did_event_formality",
    "fig_did_event_automation",
]

for name in figure_names:
    png_path = OUT / f"{name}.png"
    pdf_path = OUT / f"{name}.pdf"
    png_kb = png_path.stat().st_size / 1024 if png_path.exists() else 0
    pdf_kb = pdf_path.stat().st_size / 1024 if pdf_path.exists() else 0
    print(f"  {name:<38s} {png_kb:>7.0f}  {pdf_kb:>7.0f}")

print(f"\nAll 10 figures saved to: {OUT}")
print("Done.")
