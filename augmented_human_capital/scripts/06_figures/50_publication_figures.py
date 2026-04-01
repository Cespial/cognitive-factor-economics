#!/usr/bin/env python3
"""
Sprint 4 — Publication-quality figures for the AHC paper.
Generates 6 figures in PDF format for LaTeX inclusion.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDICES_DIR = PROJECT_ROOT / "data" / "indices"
FIG_DIR = PROJECT_ROOT / "paper" / "arxiv_submission" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#16a34a"
GRAY = "#6b7280"
ORANGE = "#ea580c"


def load_data():
    est = pd.read_parquet(PROCESSED_DIR / "estimation_sample.parquet")
    est = est[est["log_income"] > 0].copy()
    ahc_v2 = pd.read_parquet(INDICES_DIR / "ahc_index_v2_improved_crosswalk.parquet")
    est = est.merge(ahc_v2[["CIUO_code", "AHC_score", "SUB_score"]],
                    left_on="CIUO_4d", right_on="CIUO_code", how="left",
                    suffixes=("_v1", "_v2"))
    return est


# ============================================================
# FIG 2: AHC Distribution by Sector
# ============================================================
def fig2_ahc_by_sector(est):
    print("  Fig 2: AHC by sector...")
    sector_ahc = est.groupby("sector").agg(
        ahc_mean=("AHC_score_v2", "mean"),
        ahc_std=("AHC_score_v2", "std"),
        n=("weight", "sum"),
    ).reset_index().sort_values("ahc_mean", ascending=True)

    # Top 15 sectors
    sector_ahc = sector_ahc.tail(15)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [BLUE if v > sector_ahc["ahc_mean"].median() else GRAY
              for v in sector_ahc["ahc_mean"]]

    bars = ax.barh(range(len(sector_ahc)), sector_ahc["ahc_mean"],
                   xerr=sector_ahc["ahc_std"] * 0.3, color=colors,
                   edgecolor="white", linewidth=0.5, capsize=2, height=0.7)

    ax.set_yticks(range(len(sector_ahc)))
    ax.set_yticklabels(sector_ahc["sector"].values, fontsize=8)
    ax.set_xlabel("AHC Score (Augmentable Human Capital)")
    ax.set_title("Augmentable Human Capital by Sector")
    ax.axvline(x=sector_ahc["ahc_mean"].median(), color=RED, linestyle="--",
               linewidth=0.8, alpha=0.7, label=f"Median = {sector_ahc['ahc_mean'].median():.1f}")
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_ahc_by_sector.pdf")
    plt.close()


# ============================================================
# FIG 3: AHC vs Frey-Osborne Scatter
# ============================================================
def fig3_ahc_vs_fo(est):
    print("  Fig 3: AHC vs Frey-Osborne...")
    occ = est.groupby("CIUO_4d").agg(
        AHC=("AHC_score_v2", "mean"),
        FO=("automation_prob", "mean"),
        n=("weight", "sum"),
    ).reset_index().dropna()

    fig, ax = plt.subplots(figsize=(6, 5))
    sizes = np.sqrt(occ["n"]) * 0.3
    sizes = sizes.clip(upper=100)

    ax.scatter(occ["FO"], occ["AHC"], s=sizes, alpha=0.4,
               c=BLUE, edgecolors="white", linewidth=0.3)

    # Fit line
    z = np.polyfit(occ["FO"], occ["AHC"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(occ["FO"].min(), occ["FO"].max(), 100)
    ax.plot(x_line, p(x_line), color=RED, linewidth=1.5, linestyle="--")

    cor = occ["AHC"].corr(occ["FO"])
    ax.text(0.05, 0.95, f"$r = {cor:.2f}$", transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Frey--Osborne Automation Probability")
    ax.set_ylabel("AHC Score (Augmentation Potential)")
    ax.set_title("Augmentation vs. Automation:\nTwo Distinct Dimensions")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_ahc_vs_fo.pdf")
    plt.close()


# ============================================================
# FIG 5: Heterogeneity Bar Chart
# ============================================================
def fig5_heterogeneity(est):
    print("  Fig 5: Heterogeneity...")
    # Pre-computed β₂ values from Sprint 3
    data = {
        "Formal":    {"beta": 0.164, "sig": True},
        "Informal":  {"beta": -0.042, "sig": True},
        "Male":      {"beta": 0.028, "sig": True},
        "Female":    {"beta": -0.006, "sig": False},
        "Age 18-30": {"beta": 0.006, "sig": False},
        "Age 31-45": {"beta": 0.020, "sig": True},
        "Age 46-65": {"beta": 0.072, "sig": True},
        "Health":    {"beta": 0.608, "sig": True},
        "Education": {"beta": 0.203, "sig": True},
        "Manuf.":    {"beta": -0.051, "sig": False},
        "Agric.":    {"beta": -0.046, "sig": True},
    }

    labels = list(data.keys())
    betas = [data[k]["beta"] for k in labels]
    sigs = [data[k]["sig"] for k in labels]
    colors = [BLUE if b > 0 and s else RED if b < 0 and s else GRAY
              for b, s in zip(betas, sigs)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(labels)), betas, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.65)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$\beta_2$ (AHC $\times$ D interaction coefficient)")
    ax.set_title(r"Heterogeneity of the Augmentation Premium ($\beta_2$)")

    # Add significance markers
    for i, (b, s) in enumerate(zip(betas, sigs)):
        marker = "***" if s and abs(b) > 0.05 else "**" if s else "n.s."
        offset = 0.01 if b >= 0 else -0.03
        ax.text(b + offset, i, marker, va="center", fontsize=7, color="black")

    # Divider lines
    for y in [1.5, 3.5, 6.5]:
        ax.axhline(y=y, color=GRAY, linewidth=0.3, linestyle=":")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_heterogeneity.pdf")
    plt.close()


# ============================================================
# FIG 6: Quantile Regression Coefficients
# ============================================================
def fig6_quantile(est):
    print("  Fig 6: Quantile regression...")
    # Pre-computed from Sprint 3
    taus = [0.10, 0.25, 0.50, 0.75, 0.90]
    ahc_coefs = [-0.1058, -0.0608, 0.0306, 0.1378, 0.2404]
    ahc_d_coefs = [0.0138, 0.0210, 0.1179, 0.2225, 0.2687]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: AHC level
    ax1.plot(taus, ahc_coefs, "o-", color=BLUE, linewidth=2, markersize=6, label=r"$\beta_1$ (AHC level)")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.fill_between(taus, [c - 0.03 for c in ahc_coefs], [c + 0.03 for c in ahc_coefs],
                     alpha=0.15, color=BLUE)
    ax1.set_xlabel(r"Quantile ($\tau$)")
    ax1.set_ylabel("Coefficient")
    ax1.set_title(r"AHC Level Effect ($\beta_1$)")
    ax1.set_xticks(taus)

    # Right: AHC×D interaction
    ax2.plot(taus, ahc_d_coefs, "s-", color=RED, linewidth=2, markersize=6, label=r"$\beta_2$ (AHC $\times$ D)")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.fill_between(taus, [c - 0.03 for c in ahc_d_coefs], [c + 0.03 for c in ahc_d_coefs],
                     alpha=0.15, color=RED)
    ax2.set_xlabel(r"Quantile ($\tau$)")
    ax2.set_ylabel("Coefficient")
    ax2.set_title(r"Augmentation Premium ($\beta_2 = $ AHC $\times$ D)")
    ax2.set_xticks(taus)

    ax2.annotate(r"$19\times$ larger at $\tau=0.90$",
                 xy=(0.90, 0.2687), xytext=(0.60, 0.30),
                 fontsize=8, arrowprops=dict(arrowstyle="->", color=RED),
                 color=RED)

    fig.suptitle("Augmentation Premium Across the Wage Distribution", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_quantile_regression.pdf")
    plt.close()


# ============================================================
# FIG 7: Validation — AHC vs existing indices
# ============================================================
def fig7_validation():
    print("  Fig 7: Validation matrix...")
    indices = ["Felten\nAIOE", "Eloundou\nGPT-β", "Eloundou\nHuman-α",
               "Frey-\nOsborne", "Webb\nRobots", "Webb\nAI Patents", "Webb\nSoftware"]
    correlations = [0.86, 0.79, 0.72, -0.79, -0.89, -0.24, -0.68]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_bar = [GREEN if c > 0.5 else BLUE if c > 0 else ORANGE if c > -0.5 else RED
                  for c in correlations]

    bars = ax.bar(range(len(indices)), correlations, color=colors_bar,
                  edgecolor="white", linewidth=0.5, width=0.7)

    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(indices, fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Correlation with AHC Index")
    ax.set_title("External Validation: AHC vs. Existing AI Exposure Indices")
    ax.set_ylim(-1.05, 1.05)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, correlations)):
        offset = 0.05 if val >= 0 else -0.08
        ax.text(i, val + offset, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8, fontweight="bold")

    # Add zones
    ax.axhspan(0.5, 1.0, alpha=0.05, color=GREEN)
    ax.axhspan(-1.0, -0.5, alpha=0.05, color=RED)
    ax.text(6.5, 0.85, "Convergent\nvalidity", fontsize=7, color=GREEN, ha="right")
    ax.text(6.5, -0.85, "Discriminant\nvalidity", fontsize=7, color=RED, ha="right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_validation.pdf")
    plt.close()


def main():
    print("=" * 60)
    print("SPRINT 4 — Publication Figures")
    print("=" * 60)

    est = load_data()
    print(f"Data loaded: {len(est):,} observations")

    fig2_ahc_by_sector(est)
    fig3_ahc_vs_fo(est)
    fig5_heterogeneity(est)
    fig6_quantile(est)
    fig7_validation()

    print(f"\n  All figures saved to {FIG_DIR}/")
    for f in sorted(FIG_DIR.glob("*.pdf")):
        print(f"    {f.name} ({f.stat().st_size/1024:.0f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
