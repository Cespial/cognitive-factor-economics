#!/usr/bin/env python3
"""
Phase 2 — S2.3: Aggregate LLM scores into occupation-level AHC index.
AHC_o = weighted average of augmentation scores across tasks within each occupation.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDICES_DIR = PROJECT_ROOT / "data" / "indices"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT = INDICES_DIR / "ahc_index_by_occupation.parquet"


def load_llm_scores() -> pd.DataFrame:
    """Load raw LLM scores from JSONL."""
    jsonl_path = INDICES_DIR / "raw_llm_scores.jsonl"
    if not jsonl_path.exists():
        print("[ERROR] No LLM scores found. Run 20_score_tasks_llm.py first.")
        sys.exit(1)

    records = []
    errors = 0
    with open(jsonl_path) as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if "error" not in record and "augmentation_score" in record:
                    records.append(record)
                else:
                    errors += 1
            except json.JSONDecodeError:
                errors += 1

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} valid scores ({errors} errors skipped)")
    return df


def load_task_weights() -> pd.DataFrame:
    """Load task importance/frequency weights from O*NET."""
    task_path = PROCESSED_DIR / "occupation_task_matrix.parquet"
    if not task_path.exists():
        return pd.DataFrame()

    tasks = pd.read_parquet(task_path)
    # Keep importance and frequency per task-occupation pair
    weights = tasks[["Task ID", "SOC_code", "CIUO_code", "importance", "frequency",
                      "weighted_importance", "occupation_title"]].copy()
    weights = weights.rename(columns={"Task ID": "task_id", "SOC_code": "soc_code"})
    return weights


def aggregate_index(scores: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Compute AHC index per CIUO occupation."""
    # Merge scores with weights
    if not weights.empty:
        merged = scores.merge(weights, on=["task_id", "soc_code"], how="left", suffixes=("", "_w"))
        # Use importance × frequency as weight
        merged["task_weight"] = merged["importance"].fillna(3.0) * merged["frequency"].fillna(3.0)
    else:
        merged = scores.copy()
        merged["task_weight"] = 1.0
        merged["CIUO_code"] = merged.get("ciuo_code", "")

    # Use CIUO_code from weights if available, else from scores
    if "CIUO_code" not in merged.columns:
        merged["CIUO_code"] = merged.get("ciuo_code", "")

    # Winsorize scores at 1st/99th percentile
    for col in ["augmentation_score", "substitution_score"]:
        if col in merged.columns:
            p01, p99 = merged[col].quantile([0.01, 0.99])
            merged[col] = merged[col].clip(p01, p99)

    # Aggregate per CIUO occupation
    def weighted_mean(group, value_col, weight_col="task_weight"):
        w = group[weight_col]
        v = group[value_col]
        mask = v.notna() & w.notna() & (w > 0)
        if mask.sum() == 0:
            return np.nan
        return np.average(v[mask], weights=w[mask])

    records = []
    for ciuo, group in merged.groupby("CIUO_code"):
        if not ciuo or ciuo == "":
            continue

        ahc = weighted_mean(group, "augmentation_score")
        sub = weighted_mean(group, "substitution_score")

        # Count augmentation types
        type_counts = group["augmentation_type"].value_counts().to_dict() if "augmentation_type" in group.columns else {}

        # Confidence distribution
        conf_counts = group["confidence"].value_counts().to_dict() if "confidence" in group.columns else {}

        records.append({
            "CIUO_code": ciuo,
            "AHC_score": ahc,
            "SUB_score": sub,
            "AHC_normalized": None,  # Will normalize below
            "n_tasks": len(group),
            "n_unique_tasks": group["task_id"].nunique(),
            "primary_aug_type": max(type_counts, key=type_counts.get) if type_counts else "N/A",
            "aug_type_distribution": json.dumps(type_counts),
            "confidence_distribution": json.dumps(conf_counts),
            "mean_task_weight": group["task_weight"].mean(),
            "occupation_title": group.get("occupation_title", pd.Series()).dropna().iloc[0] if "occupation_title" in group.columns and group["occupation_title"].notna().any() else "",
        })

    result = pd.DataFrame(records)

    # Normalize AHC to [0, 1]
    if not result.empty and result["AHC_score"].notna().any():
        min_ahc = result["AHC_score"].min()
        max_ahc = result["AHC_score"].max()
        if max_ahc > min_ahc:
            result["AHC_normalized"] = (result["AHC_score"] - min_ahc) / (max_ahc - min_ahc)
        else:
            result["AHC_normalized"] = 0.5

    return result


def merge_classification_scores(index: pd.DataFrame) -> pd.DataFrame:
    """Add H^P, H^C, H^A classification shares from Phase 1."""
    class_path = PROCESSED_DIR / "task_classification.parquet"
    if not class_path.exists():
        return index

    classified = pd.read_parquet(class_path)

    # Aggregate classification shares per CIUO
    class_agg = classified.groupby("CIUO_code").agg(
        PHY_score=("final_hp", "mean"),
        ROU_score=("final_hc", "mean"),
        AUG_rules=("final_ha", "mean"),
    ).reset_index()

    index = index.merge(class_agg, on="CIUO_code", how="left")
    return index


def main():
    print("=" * 60)
    print("AHC Index Aggregation — Phase 2, Sprint 2.3")
    print("=" * 60)

    scores = load_llm_scores()
    weights = load_task_weights()

    if scores.empty:
        print("[ERROR] No valid LLM scores to aggregate")
        return 1

    index = aggregate_index(scores, weights)
    index = merge_classification_scores(index)

    # Save
    index.to_parquet(OUTPUT, index=False)
    print(f"\n  [SAVED] {OUTPUT.name}: {len(index)} occupations")

    # Also save CSV for inspection
    csv_path = OUTPUT.with_suffix(".csv")
    index.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path.name}")

    # Summary statistics
    print(f"\n--- AHC Index Summary ---")
    print(f"  Occupations: {len(index)}")
    for col in ["AHC_score", "SUB_score", "AHC_normalized", "PHY_score", "ROU_score"]:
        if col in index.columns and index[col].notna().any():
            print(f"  {col}: mean={index[col].mean():.2f}, std={index[col].std():.2f}, "
                  f"min={index[col].min():.2f}, max={index[col].max():.2f}")

    # Orthogonality check
    if all(c in index.columns for c in ["AHC_score", "SUB_score", "PHY_score", "ROU_score"]):
        valid = index.dropna(subset=["AHC_score", "SUB_score"])
        if len(valid) > 5:
            corr_ahc_sub = valid["AHC_score"].corr(valid["SUB_score"])
            print(f"\n  Orthogonality checks:")
            print(f"    cor(AHC, SUB) = {corr_ahc_sub:.3f} (target: < 0.3)")
            if "PHY_score" in valid.columns:
                print(f"    cor(AHC, PHY) = {valid['AHC_score'].corr(valid['PHY_score']):.3f} (target: ~0)")
            if "ROU_score" in valid.columns:
                print(f"    cor(AHC, ROU) = {valid['AHC_score'].corr(valid['ROU_score']):.3f} (target: negative)")

    # Top/bottom occupations
    if "AHC_score" in index.columns and index["AHC_score"].notna().any():
        print(f"\n  Top 10 AHC occupations:")
        for _, row in index.nlargest(10, "AHC_score").iterrows():
            print(f"    {row['CIUO_code']} ({row.get('occupation_title', '')[:40]}): AHC={row['AHC_score']:.1f}")

        print(f"\n  Bottom 10 AHC occupations:")
        for _, row in index.nsmallest(10, "AHC_score").iterrows():
            print(f"    {row['CIUO_code']} ({row.get('occupation_title', '')[:40]}): AHC={row['AHC_score']:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
