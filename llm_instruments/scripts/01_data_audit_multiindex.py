#!/usr/bin/env python3
"""
Paper 0 — Phase 0+2: Data audit and multi-index comparison.
Builds master dataset at 6-digit SOC level with all available indices.
Computes pairwise correlations, PCA, and factor analysis.
"""

import sys
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
P1_DATA = PROJECT / "data" / "paper1"
OUTPUT = PROJECT / "output"
OUTPUT_T = OUTPUT / "tables"
OUTPUT_F = OUTPUT / "figures"
for d in [OUTPUT_T, OUTPUT_F]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Load all indices
# ============================================================

def load_ahc_haiku():
    """Load our AHC scores aggregated to SOC level."""
    scores = []
    with open(P1_DATA / "indices" / "raw_llm_scores.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if "error" not in r and "augmentation_score" in r:
                    scores.append(r)
            except:
                pass
    df = pd.DataFrame(scores)
    # Aggregate to unique task_id (already unique, but ensure)
    agg = df.groupby("soc_code").agg(
        ahc_haiku=("augmentation_score", "mean"),
        sub_haiku=("substitution_score", "mean"),
        n_tasks_haiku=("task_id", "nunique"),
    ).reset_index()
    agg = agg.rename(columns={"soc_code": "SOC"})
    print(f"  AHC Haiku: {len(agg)} SOC codes")
    return agg


def load_ahc_sonnet():
    """Load Sonnet validation scores."""
    scores = []
    with open(P1_DATA / "indices" / "sonnet_validation_scores.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if "error" not in r and "augmentation_score" in r:
                    scores.append(r)
            except:
                pass
    df = pd.DataFrame(scores)
    # Need SOC code — get from haiku scores
    haiku = []
    with open(P1_DATA / "indices" / "raw_llm_scores.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if "error" not in r:
                    haiku.append({"task_id": r["task_id"], "soc_code": r.get("soc_code", "")})
            except:
                pass
    haiku_map = pd.DataFrame(haiku).drop_duplicates("task_id")
    df = df.merge(haiku_map, on="task_id", how="left")

    agg = df.groupby("soc_code").agg(
        ahc_sonnet=("augmentation_score", "mean"),
        sub_sonnet=("substitution_score", "mean"),
        n_tasks_sonnet=("task_id", "nunique"),
    ).reset_index()
    agg = agg.rename(columns={"soc_code": "SOC"})
    print(f"  AHC Sonnet: {len(agg)} SOC codes")
    return agg


def load_felten():
    """Load Felten AIOE at 6-digit SOC."""
    path = P1_DATA / "raw" / "felten" / "AIOE_DataAppendix.xlsx"
    if not path.exists():
        print("  [SKIP] Felten AIOE not found")
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="Appendix A")
    df.columns = ["SOC", "title_felten", "felten_aioe"]
    df["SOC"] = df["SOC"].astype(str).str.strip()
    # Standardize SOC format: remove dots, ensure consistent
    df["SOC_clean"] = df["SOC"].str.replace("-", "").str[:6]
    print(f"  Felten AIOE: {len(df)} occupations")
    return df[["SOC", "SOC_clean", "felten_aioe"]]


def load_webb():
    """Load Webb AI/robot/software exposure + Felten + F&O from autoScores."""
    path = P1_DATA / "raw" / "webb" / "autoScores.csv"
    if not path.exists():
        print("  [SKIP] Webb autoScores not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Get latest year per occupation
    if "year" in df.columns:
        latest = df.sort_values("year").groupby("simpleOcc").last().reset_index()
    else:
        latest = df.copy()

    occ_col = "simpleOcc"
    latest["SOC"] = latest[occ_col].astype(str)

    cols = ["SOC"]
    for c in ["pct_ai", "pct_robot", "pct_software", "felten_raj_seamans", "freyOsborne"]:
        if c in latest.columns:
            cols.append(c)
            latest[c] = pd.to_numeric(latest[c], errors="coerce")

    print(f"  Webb: {len(latest)} occupations, cols: {[c for c in cols if c != 'SOC']}")
    return latest[cols]


def load_eloundou():
    """Load Eloundou GPT exposure (human + model ratings)."""
    path = P1_DATA / "raw" / "eloundou" / "occ_level.csv"
    if not path.exists():
        print("  [SKIP] Eloundou not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["SOC"] = df["O*NET-SOC Code"].str.replace(r"\.\d+$", "", regex=True)

    score_cols = [c for c in df.columns if "rating" in c.lower()]
    agg = df.groupby("SOC")[score_cols].mean().reset_index()
    # Rename
    rename = {}
    for c in score_cols:
        rename[c] = f"eloundou_{c}"
    agg = agg.rename(columns=rename)
    print(f"  Eloundou: {len(agg)} SOC codes, scores: {list(rename.values())}")
    return agg


def load_eisfeldt():
    """Load Eisfeldt GenAI exposure."""
    path = P1_DATA / "raw" / "eisfeldt" / "genaiexp_scores.csv"
    if not path.exists():
        print("  [SKIP] Eisfeldt not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
    soc_col = [c for c in df.columns if "soc" in c.lower()][0] if any("soc" in c.lower() for c in df.columns) else None
    if not soc_col:
        print("  [SKIP] Eisfeldt: no SOC column found")
        return pd.DataFrame()
    score_cols = [c for c in df.columns if "genai" in c.lower() or "exp" in c.lower()]
    df["SOC"] = df[soc_col].astype(str)
    agg = df.groupby("SOC")[score_cols[:3]].mean().reset_index()
    rename = {c: f"eisfeldt_{c}" for c in score_cols[:3]}
    agg = agg.rename(columns=rename)
    print(f"  Eisfeldt: {len(agg)} occupations")
    return agg


# ============================================================
# 2. Build master dataset and compute correlations
# ============================================================

def build_master():
    """Merge all indices into master dataset at SOC level."""
    print("\n=== Loading all indices ===")
    haiku = load_ahc_haiku()
    sonnet = load_ahc_sonnet()
    felten = load_felten()
    webb = load_webb()
    eloundou = load_eloundou()
    eisfeldt = load_eisfeldt()

    # Start with Haiku (most complete)
    master = haiku.copy()

    # Merge Sonnet
    if not sonnet.empty:
        master = master.merge(sonnet, on="SOC", how="left")

    # Merge Eloundou (same SOC format)
    if not eloundou.empty:
        master = master.merge(eloundou, on="SOC", how="left")

    # Felten and Webb use different SOC formats — try merge
    if not felten.empty:
        # Try direct merge
        m1 = master.merge(felten[["SOC", "felten_aioe"]], on="SOC", how="left")
        if m1["felten_aioe"].notna().sum() > 10:
            master = m1
        else:
            # Try with cleaned SOC
            master["SOC_clean"] = master["SOC"].str.replace("-", "").str[:6]
            master = master.merge(felten[["SOC_clean", "felten_aioe"]], on="SOC_clean", how="left")

    print(f"\n  Master dataset: {len(master)} occupations")
    for c in master.columns:
        if c != "SOC" and c != "SOC_clean":
            n = master[c].notna().sum()
            if n > 0:
                print(f"    {c}: {n} non-null ({n/len(master)*100:.0f}%)")

    return master


def compute_correlations(master):
    """Compute pairwise Pearson and Spearman correlations."""
    print("\n=== Pairwise Correlations ===")

    score_cols = [c for c in master.columns if any(x in c for x in
                  ["ahc_", "sub_", "felten", "pct_", "frey", "eloundou", "eisfeldt"])
                  and "n_tasks" not in c]

    # Pearson
    pearson = master[score_cols].corr(method="pearson")
    # Spearman
    spearman = master[score_cols].corr(method="spearman")

    print("\nPearson correlations:")
    print(pearson.to_string(float_format="{:.3f}".format))

    print("\nSpearman correlations:")
    print(spearman.to_string(float_format="{:.3f}".format))

    # Save
    pearson.to_csv(OUTPUT_T / "correlation_pearson.csv")
    spearman.to_csv(OUTPUT_T / "correlation_spearman.csv")
    print(f"\n  [SAVED] correlation_pearson.csv, correlation_spearman.csv")

    return pearson, spearman, score_cols


def run_pca(master, score_cols):
    """PCA to identify dimensionality."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("\n=== Principal Component Analysis ===")
    data = master[score_cols].dropna()
    if len(data) < 10:
        print("  [SKIP] Insufficient data for PCA")
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    pca = PCA()
    pca.fit(X)

    print(f"  N observations: {len(data)}")
    print(f"  N components: {len(score_cols)}")
    print(f"  Explained variance ratio:")
    cumvar = 0
    for i, v in enumerate(pca.explained_variance_ratio_):
        cumvar += v
        print(f"    PC{i+1}: {v:.3f} (cumulative: {cumvar:.3f})")

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(score_cols))],
        index=score_cols
    )
    print(f"\n  Loadings (first 3 PCs):")
    print(loadings.iloc[:, :3].to_string(float_format="{:.3f}".format))

    loadings.to_csv(OUTPUT_T / "pca_loadings.csv")
    print(f"\n  [SAVED] pca_loadings.csv")

    # Key insight
    if pca.explained_variance_ratio_[0] > 0.7:
        print("\n  >>> 1 PC explains >70%: indices measure ONE dimension")
    elif sum(pca.explained_variance_ratio_[:2]) > 0.7:
        print("\n  >>> 2 PCs explain >70%: TWO distinct dimensions (likely augmentation vs substitution)")


def main():
    print("=" * 70)
    print("PAPER 0 — Data Audit & Multi-Index Comparison")
    print("=" * 70)

    master = build_master()
    master.to_csv(OUTPUT_T / "master_multiindex.csv", index=False)

    pearson, spearman, score_cols = compute_correlations(master)
    run_pca(master, score_cols)

    # Summary stats
    print("\n=== Summary Statistics ===")
    desc = master[score_cols].describe().T[["count", "mean", "std", "min", "max"]]
    print(desc.to_string(float_format="{:.2f}".format))
    desc.to_csv(OUTPUT_T / "index_summary_stats.csv")

    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
