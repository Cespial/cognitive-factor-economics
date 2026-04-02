#!/usr/bin/env python3
"""
Paper 0 — Sprint 5: Publication figures (8 figures).
"""

import sys
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 12, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3,
})

PROJECT = Path(__file__).resolve().parent.parent
P1 = PROJECT / "data" / "paper1"
FIG = PROJECT / "paper" / "arxiv_submission" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
OUTPUT = PROJECT / "output" / "tables"

BLUE, RED, GREEN, GRAY, ORANGE, PURPLE = "#2563eb", "#dc2626", "#16a34a", "#6b7280", "#ea580c", "#7c3aed"


def load_scores(path, key="augmentation_score"):
    scores = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if "error" not in r and key in r:
                    scores[r["task_id"]] = r
            except: pass
    return scores


# ============================================================
# FIG 2: Correlation heatmap (all indices)
# ============================================================
def fig2_correlation_heatmap():
    print("  Fig 2: Correlation heatmap...")
    corr = pd.read_csv(OUTPUT / "correlation_pearson.csv", index_col=0)

    # Shorten names
    rename = {
        "ahc_haiku": "AHC\n(Haiku)", "ahc_sonnet": "AHC\n(Sonnet)",
        "sub_haiku": "SUB\n(Haiku)", "sub_sonnet": "SUB\n(Sonnet)",
        "felten_aioe": "Felten\nAIOE",
        "eloundou_dv_rating_alpha": "Eloundou\nα (GPT)",
        "eloundou_dv_rating_beta": "Eloundou\nβ (GPT)",
        "eloundou_dv_rating_gamma": "Eloundou\nγ (GPT)",
        "eloundou_human_rating_alpha": "Eloundou\nα (Human)",
        "eloundou_human_rating_beta": "Eloundou\nβ (Human)",
        "eloundou_human_rating_gamma": "Eloundou\nγ (Human)",
    }
    corr = corr.rename(index=rename, columns=rename)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(corr.index, fontsize=7)

    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Pairwise Correlations: 11 AI Exposure Indices (207 SOC Occupations)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_correlation_heatmap.pdf")
    plt.close()


# ============================================================
# FIG 3: PCA biplot
# ============================================================
def fig3_pca_biplot():
    print("  Fig 3: PCA biplot...")
    loadings = pd.read_csv(OUTPUT / "pca_loadings.csv", index_col=0)

    rename = {
        "ahc_haiku": "AHC (Haiku)", "ahc_sonnet": "AHC (Sonnet)",
        "sub_haiku": "SUB (Haiku)", "sub_sonnet": "SUB (Sonnet)",
        "felten_aioe": "Felten AIOE",
        "eloundou_dv_rating_alpha": "Eloundou α",
        "eloundou_dv_rating_beta": "Eloundou β",
        "eloundou_dv_rating_gamma": "Eloundou γ",
        "eloundou_human_rating_alpha": "Human α",
        "eloundou_human_rating_beta": "Human β",
        "eloundou_human_rating_gamma": "Human γ",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    colors_map = {
        "AHC (Haiku)": BLUE, "AHC (Sonnet)": BLUE,
        "SUB (Haiku)": RED, "SUB (Sonnet)": RED,
        "Felten AIOE": GREEN,
        "Eloundou α": ORANGE, "Eloundou β": ORANGE, "Eloundou γ": ORANGE,
        "Human α": PURPLE, "Human β": PURPLE, "Human γ": PURPLE,
    }

    for idx in loadings.index:
        name = rename.get(idx, idx)
        x, y = loadings.loc[idx, "PC1"], loadings.loc[idx, "PC2"]
        color = colors_map.get(name, GRAY)
        ax.arrow(0, 0, x, y, head_width=0.015, head_length=0.01, fc=color, ec=color, alpha=0.8)
        ax.text(x * 1.12, y * 1.12, name, fontsize=7, ha="center", va="center", color=color)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    circle = plt.Circle((0, 0), 0.4, fill=False, color=GRAY, linestyle="--", linewidth=0.5)
    ax.add_patch(circle)

    ax.set_xlabel("PC1 (62% variance) — General AI Cognitive Relevance")
    ax.set_ylabel("PC2 (15% variance) — Substitution vs. Augmentation")
    ax.set_title("PCA Biplot: AI Exposure Indices Span Two Dimensions")
    ax.set_xlim(-0.05, 0.45)
    ax.set_ylim(-0.6, 0.6)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_pca_biplot.pdf")
    plt.close()


# ============================================================
# FIG 4: Bland-Altman (Haiku vs Sonnet)
# ============================================================
def fig4_bland_altman():
    print("  Fig 4: Bland-Altman...")
    haiku = load_scores(P1 / "indices" / "raw_llm_scores.jsonl")
    sonnet = load_scores(P1 / "indices" / "sonnet_validation_scores.jsonl")

    common = set(haiku) & set(sonnet)
    h = np.array([haiku[t]["augmentation_score"] for t in common])
    s = np.array([sonnet[t]["augmentation_score"] for t in common])

    mean_hs = (h + s) / 2
    diff_hs = s - h  # Sonnet - Haiku

    mean_diff = diff_hs.mean()
    std_diff = diff_hs.std()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean_hs, diff_hs, alpha=0.15, s=8, c=BLUE, edgecolors="none")
    ax.axhline(mean_diff, color=RED, linewidth=1.5, label=f"Mean bias = {mean_diff:+.1f}")
    ax.axhline(mean_diff + 1.96 * std_diff, color=RED, linewidth=0.8, linestyle="--",
               label=f"+1.96 SD = {mean_diff + 1.96 * std_diff:+.1f}")
    ax.axhline(mean_diff - 1.96 * std_diff, color=RED, linewidth=0.8, linestyle="--",
               label=f"$-$1.96 SD = {mean_diff - 1.96 * std_diff:+.1f}")
    ax.axhline(0, color="black", linewidth=0.5)

    ax.set_xlabel("Mean of Haiku and Sonnet Scores")
    ax.set_ylabel("Difference (Sonnet $-$ Haiku)")
    ax.set_title(f"Bland--Altman Agreement Plot ($n = {len(common):,}$ tasks)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_bland_altman.pdf")
    plt.close()


# ============================================================
# FIG 5: Prompt sensitivity
# ============================================================
def fig5_prompt_sensitivity():
    print("  Fig 5: Prompt sensitivity...")
    prompts = {}
    for pname in ["A_baseline", "B_behavioral", "C_counterfactual", "D_negative"]:
        path = OUTPUT / f"prompt_{pname}_scores.jsonl"
        if not path.exists():
            continue
        scores = {}
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    scores[r["task_id"]] = r["a"]
                except: pass
        prompts[pname] = scores

    if len(prompts) < 2:
        print("    [SKIP] Not enough prompts")
        return

    names = list(prompts.keys())
    common = set.intersection(*[set(s.keys()) for s in prompts.values()])
    n = len(names)

    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v1 = [prompts[names[i]][t] for t in common]
            v2 = [prompts[names[j]][t] for t in common]
            rho, _ = spearmanr(v1, v2)
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    short = ["A: Baseline", "B: Behavioral", "C: Counterfactual", "D: Negative"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: correlation matrix
    im = ax1.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(short, fontsize=8, rotation=30, ha="right")
    ax1.set_yticklabels(short, fontsize=8)
    for i in range(n):
        for j in range(n):
            color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            ax1.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center", fontsize=9, color=color)
    ax1.set_title("Spearman $\\rho$ Across Prompts")
    fig.colorbar(im, ax=ax1, shrink=0.8)

    # Right: mean scores by prompt
    means = [np.mean([prompts[names[i]][t] for t in common]) for i in range(n)]
    colors = [BLUE, GREEN, ORANGE, RED]
    ax2.bar(range(n), means, color=colors, edgecolor="white", width=0.6)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(short, fontsize=8, rotation=30, ha="right")
    ax2.set_ylabel("Mean Augmentation Score")
    ax2.set_title("Absolute Scale Varies by Prompt")
    for i, m in enumerate(means):
        ax2.text(i, m + 1, f"{m:.0f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(f"Prompt Sensitivity Analysis ($n = {len(common)}$ tasks, 4 prompt variants)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_prompt_sensitivity.pdf")
    plt.close()


# ============================================================
# FIG 6: Horse race R² (incremental contribution)
# ============================================================
def fig6_horse_race():
    print("  Fig 6: Horse race R²...")
    path = OUTPUT / "oriv_horse_race_results.json"
    if not path.exists():
        print("    [SKIP] No horse race results")
        return

    hr = json.loads(path.read_text())["horse_race"]

    models = ["Controls\nonly", "+ F&O", "+ AHC", "+ AHC\n+ F&O", "+ AHC×D\n+ F&O"]
    r2s = [hr["r2_controls"], hr["r2_fo"], hr["r2_ahc"], hr["r2_both"], hr["r2_full"]]
    colors = [GRAY, RED, BLUE, PURPLE, GREEN]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(models)), r2s, color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("$R^2$")
    ax.set_title("Horse Race: Incremental $R^2$ of AHC vs. Frey--Osborne")
    ax.set_ylim(0.41, 0.44)

    for i, (bar, val) in enumerate(zip(bars, r2s)):
        ax.text(i, val + 0.0005, f"{val:.4f}", ha="center", fontsize=8, fontweight="bold")

    # Annotations
    ax.annotate("", xy=(2, r2s[2]), xytext=(0, r2s[0]),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    ax.text(1, (r2s[0] + r2s[2]) / 2 + 0.001, f"AHC\n+{(r2s[2]-r2s[0])*100:.2f}pp",
            ha="center", fontsize=7, color=BLUE)

    fig.tight_layout()
    fig.savefig(FIG / "fig6_horse_race.pdf")
    plt.close()


# ============================================================
# FIG 7: ORIV comparison (OLS vs ORIV vs External IV)
# ============================================================
def fig7_oriv_comparison():
    print("  Fig 7: ORIV comparison...")
    path = OUTPUT / "oriv_horse_race_results.json"
    if not path.exists():
        return

    data = json.loads(path.read_text())["oriv"]

    methods = ["OLS\n(Haiku)", "OLS\n(Sonnet)", "ORIV\n(Sonnet→Haiku)", "External IV\n(Paper 1)"]
    values = [data["ols_haiku"], data["ols_sonnet"], data["oriv"], 0.234]
    colors = [BLUE, ORANGE, GREEN, RED]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(methods)), values, color=colors, edgecolor="white", width=0.55)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("$\\beta_1$ (AHC level coefficient)")
    ax.set_title("Measurement Error Correction: OLS → ORIV → IV")
    ax.axhline(0, color="black", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(i, val + 0.003, f"{val:+.3f}", ha="center", fontsize=9, fontweight="bold")

    # Annotation: attenuation
    ax.annotate("Attenuation\nbias", xy=(0, values[0]), xytext=(1.5, 0.06),
                fontsize=8, color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.annotate("ORIV\ncorrects", xy=(2, values[2]), xytext=(1.5, 0.06),
                fontsize=8, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))

    fig.tight_layout()
    fig.savefig(FIG / "fig7_oriv_comparison.pdf")
    plt.close()


# ============================================================
# FIG 8: Three-way reliability
# ============================================================
def fig8_three_way():
    print("  Fig 8: Three-way reliability...")
    haiku = load_scores(P1 / "indices" / "raw_llm_scores.jsonl")
    sonnet = load_scores(P1 / "indices" / "sonnet_validation_scores.jsonl")
    gpt = load_scores(PROJECT / "output" / "tables" / "gpt4o_validation_scores.jsonl")

    pairs = [
        ("Haiku ↔ Sonnet", haiku, sonnet, BLUE),
        ("Haiku ↔ GPT-4o", haiku, gpt, GREEN),
        ("Sonnet ↔ GPT-4o", sonnet, gpt, ORANGE),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (title, d1, d2, color) in zip(axes, pairs):
        common = set(d1) & set(d2)
        if len(common) < 10:
            ax.set_title(f"{title}\n(n={len(common)}, insufficient)")
            continue
        v1 = np.array([d1[t]["augmentation_score"] for t in common])
        v2 = np.array([d2[t]["augmentation_score"] for t in common])
        r = np.corrcoef(v1, v2)[0, 1]
        rho, _ = spearmanr(v1, v2)

        ax.scatter(v1, v2, alpha=0.2, s=10, c=color, edgecolors="none")
        # Fit line
        z = np.polyfit(v1, v2, 1)
        x_line = np.linspace(v1.min(), v1.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color=RED, linewidth=1.5, linestyle="--")
        ax.plot([0, 100], [0, 100], color=GRAY, linewidth=0.5, linestyle=":")

        ax.set_xlabel(title.split(" ↔ ")[0])
        ax.set_ylabel(title.split(" ↔ ")[1])
        ax.set_title(f"{title}\n$r = {r:.2f}$, $\\rho = {rho:.2f}$, $n = {len(common)}$")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    fig.suptitle("Three-Way Inter-Model Agreement on Augmentation Scores", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_three_way_reliability.pdf")
    plt.close()


def main():
    print("=" * 60)
    print("PAPER 0 — Publication Figures")
    print("=" * 60)

    fig2_correlation_heatmap()
    fig3_pca_biplot()
    fig4_bland_altman()
    fig5_prompt_sensitivity()
    fig6_horse_race()
    fig7_oriv_comparison()
    fig8_three_way()

    print(f"\n  Figures saved to {FIG}/")
    for f in sorted(FIG.glob("*.pdf")):
        print(f"    {f.name} ({f.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
