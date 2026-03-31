#!/usr/bin/env python3
"""
Phase 4 — S4.1.6: Produce descriptive statistics and correlation analysis.
Generates Table 1, correlation heatmap, and AHC distribution figures.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_sample() -> pd.DataFrame:
    path = PROCESSED_DIR / "estimation_sample.parquet"
    if not path.exists():
        print("[ERROR] Run 30_merge_geih_ahc.py first")
        sys.exit(1)
    return pd.read_parquet(path)


def table_descriptive_by_ahc(df: pd.DataFrame):
    """Table: Compare workers in high vs low AHC occupations."""
    if "AHC_score" not in df.columns or df["AHC_score"].isna().all():
        print("  [SKIP] AHC scores not available")
        return

    median_ahc = df["AHC_score"].median()
    df["high_ahc"] = (df["AHC_score"] >= median_ahc).astype(int)

    vars_compare = ["income", "log_income", "age", "education_years", "experience",
                     "hours_worked", "formal", "female", "automation_prob"]
    vars_available = [v for v in vars_compare if v in df.columns]

    records = []
    for var in vars_available:
        high = df.loc[df["high_ahc"] == 1, var].dropna()
        low = df.loc[df["high_ahc"] == 0, var].dropna()

        records.append({
            "variable": var,
            "high_ahc_mean": high.mean(),
            "high_ahc_std": high.std(),
            "high_ahc_n": len(high),
            "low_ahc_mean": low.mean(),
            "low_ahc_std": low.std(),
            "low_ahc_n": len(low),
            "diff": high.mean() - low.mean(),
        })

    table = pd.DataFrame(records)
    path = TABLE_DIR / "descriptive_by_ahc_level.csv"
    table.to_csv(path, index=False)
    print(f"  [SAVED] {path.name}")


def correlation_matrix(df: pd.DataFrame):
    """Compute correlation matrix of key indices."""
    index_vars = ["AHC_score", "SUB_score", "PHY_score", "ROU_score", "automation_prob"]
    available = [v for v in index_vars if v in df.columns and df[v].notna().any()]

    if len(available) < 2:
        print("  [SKIP] Not enough index variables for correlation matrix")
        return

    corr = df[available].corr()
    path = TABLE_DIR / "index_correlation_matrix.csv"
    corr.to_csv(path)
    print(f"  [SAVED] {path.name}")

    print(f"\n  Correlation Matrix:")
    print(corr.to_string(float_format="{:.3f}".format))


def sector_ahc_summary(df: pd.DataFrame):
    """AHC by sector summary."""
    if "AHC_score" not in df.columns or df["AHC_score"].isna().all():
        return

    sector_col = "sector" if "sector" in df.columns else None
    if not sector_col:
        return

    sector_stats = df.groupby(sector_col).agg(
        mean_ahc=("AHC_score", "mean"),
        mean_sub=("SUB_score", "mean"),
        mean_income=("income", "mean"),
        mean_formal=("formal", "mean"),
        n_workers=("weight", "sum"),
    ).reset_index().sort_values("mean_ahc", ascending=False)

    path = TABLE_DIR / "sector_ahc_summary.csv"
    sector_stats.to_csv(path, index=False)
    print(f"  [SAVED] {path.name}")

    print(f"\n  Sector AHC Summary:")
    print(sector_stats.to_string(index=False, float_format="{:.3f}".format))


def generate_figures(df: pd.DataFrame):
    """Generate key figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", font_scale=1.1)
    except ImportError:
        print("  [SKIP] matplotlib/seaborn not available")
        return

    # Figure 2: AHC distribution
    if "AHC_score" in df.columns and df["AHC_score"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df["AHC_score"].dropna(), bins=50, edgecolor="white", alpha=0.8, color="#2563eb")
        ax.set_xlabel("AHC Score (Augmentable Human Capital)")
        ax.set_ylabel("Number of Workers")
        ax.set_title("Distribution of Augmentable Human Capital Across Colombian Workers")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "fig2_ahc_distribution.png", dpi=200)
        plt.close()
        print(f"  [SAVED] fig2_ahc_distribution.png")

    # Figure 3: AHC vs Frey-Osborne scatter
    if all(c in df.columns for c in ["AHC_score", "automation_prob"]) and df["AHC_score"].notna().any():
        occ_level = df.groupby("CIUO_4d").agg(
            AHC=("AHC_score", "mean"),
            FO=("automation_prob", "mean"),
            n=("weight", "sum"),
        ).reset_index().dropna()

        if len(occ_level) > 5:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                occ_level["FO"], occ_level["AHC"],
                s=np.sqrt(occ_level["n"]) * 0.5, alpha=0.5, c="#2563eb", edgecolors="white", linewidth=0.3
            )
            ax.set_xlabel("Frey-Osborne Automation Probability")
            ax.set_ylabel("AHC Score (Augmentation Potential)")
            ax.set_title("Augmentation vs. Substitution: Two Distinct Dimensions")
            # Add correlation text
            corr = occ_level["AHC"].corr(occ_level["FO"])
            ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=12,
                    verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / "fig3_ahc_vs_fo_scatter.png", dpi=200)
            plt.close()
            print(f"  [SAVED] fig3_ahc_vs_fo_scatter.png")

    # Correlation heatmap
    index_vars = ["AHC_score", "SUB_score", "PHY_score", "ROU_score", "automation_prob"]
    available = [v for v in index_vars if v in df.columns and df[v].notna().any()]
    if len(available) >= 3:
        corr = df[available].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                    square=True, linewidths=0.5)
        ax.set_title("Correlation Between Human Capital Indices")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=200)
        plt.close()
        print(f"  [SAVED] correlation_heatmap.png")


def main():
    print("=" * 60)
    print("Descriptive Statistics & Figures — Phase 4")
    print("=" * 60)

    df = load_sample()
    print(f"  Sample: {len(df):,} observations")

    table_descriptive_by_ahc(df)
    correlation_matrix(df)
    sector_ahc_summary(df)
    generate_figures(df)

    print("\n[DONE] Descriptive analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
