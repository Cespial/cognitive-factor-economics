#!/usr/bin/env python3
"""
Phase 1 — S1.3 (continued): Build occupation-task matrix.
Maps O*NET tasks to Colombian CIUO-08 occupations via chained crosswalk.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ONET_DIR = PROJECT_ROOT / "data" / "raw" / "onet"
CROSSWALK_DIR = PROJECT_ROOT / "data" / "crosswalks"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "occupation_task_matrix.parquet"


def load_onet_tasks() -> pd.DataFrame:
    """Load O*NET task statements with importance/frequency ratings."""
    tasks = pd.read_csv(ONET_DIR / "Task Statements.txt", sep="\t", encoding="utf-8", on_bad_lines="skip")
    ratings = pd.read_csv(ONET_DIR / "Task Ratings.txt", sep="\t", encoding="utf-8", on_bad_lines="skip")

    # Tasks has: O*NET-SOC Code, Task ID, Task, Task Type, Incumbents Responding
    # Ratings has: O*NET-SOC Code, Task ID, Scale ID, Data Value, N, etc.
    # Scale IDs: IM = Importance, FQ = Frequency, RT = Relevance

    # Pivot ratings to get importance and frequency per task
    ratings_pivot = ratings.pivot_table(
        index=["O*NET-SOC Code", "Task ID"],
        columns="Scale ID",
        values="Data Value",
        aggfunc="mean"
    ).reset_index()

    # Rename scale columns (FT = Frequency, IM = Importance, RT = Relevance)
    scale_rename = {"IM": "importance", "FT": "frequency", "RT": "relevance"}
    ratings_pivot = ratings_pivot.rename(columns=scale_rename)

    # Merge tasks with ratings
    tasks = tasks.merge(ratings_pivot, on=["O*NET-SOC Code", "Task ID"], how="left")

    # Extract SOC code (remove O*NET suffix)
    tasks["SOC_code"] = tasks["O*NET-SOC Code"].str.replace(r"\.\d+$", "", regex=True)

    print(f"  O*NET tasks loaded: {len(tasks)} task-occupation pairs")
    print(f"  Unique occupations: {tasks['SOC_code'].nunique()}")
    print(f"  Unique task statements: {tasks['Task ID'].nunique()}")

    return tasks


def load_onet_abilities() -> pd.DataFrame:
    """Load O*NET ability ratings per occupation (used for H^P/H^C/H^A classification)."""
    path = ONET_DIR / "Abilities.txt"
    if not path.exists():
        print("  [WARN] Abilities.txt not found")
        return pd.DataFrame()

    abilities = pd.read_csv(path, sep="\t", encoding="utf-8", on_bad_lines="skip")
    abilities["SOC_code"] = abilities["O*NET-SOC Code"].str.replace(r"\.\d+$", "", regex=True)

    # Pivot: importance (IM) and level (LV)
    abilities_pivot = abilities.pivot_table(
        index=["SOC_code", "Element Name"],
        columns="Scale ID",
        values="Data Value",
        aggfunc="mean"
    ).reset_index()

    scale_rename = {"IM": "ability_importance", "LV": "ability_level"}
    abilities_pivot = abilities_pivot.rename(columns=scale_rename)

    print(f"  O*NET abilities loaded: {len(abilities_pivot)} ability-occupation pairs")
    return abilities_pivot


def map_tasks_to_ciuo(tasks: pd.DataFrame) -> pd.DataFrame:
    """Map O*NET tasks to CIUO-08 codes using chained crosswalk."""
    crosswalk = pd.read_parquet(CROSSWALK_DIR / "soc_ciuo_chained.parquet")

    # Merge tasks with crosswalk
    merged = tasks.merge(
        crosswalk[["SOC_code", "CIUO_code", "weight", "confidence", "occupation_title"]],
        on="SOC_code",
        how="inner"
    )

    print(f"  Tasks mapped to CIUO: {len(merged)} task-CIUO pairs")
    print(f"  CIUO occupations with tasks: {merged['CIUO_code'].nunique()}")

    # Adjust importance by crosswalk weight
    if "importance" in merged.columns:
        merged["weighted_importance"] = merged["importance"] * merged["weight"]
    else:
        merged["weighted_importance"] = merged["weight"]

    return merged


def compute_occupation_task_summary(mapped: pd.DataFrame) -> pd.DataFrame:
    """Summarize task composition per CIUO occupation."""
    agg_dict = {
        "n_tasks": ("Task ID", "nunique"),
        "n_source_soc": ("SOC_code", "nunique"),
    }
    if "importance" in mapped.columns:
        agg_dict["avg_importance"] = ("importance", "mean")
    if "frequency" in mapped.columns:
        agg_dict["avg_frequency"] = ("frequency", "mean")
    if "weighted_importance" in mapped.columns:
        agg_dict["total_weighted_importance"] = ("weighted_importance", "sum")

    summary = mapped.groupby("CIUO_code").agg(**agg_dict).reset_index()

    # Add representative title (from most common SOC mapping)
    titles = mapped.groupby("CIUO_code")["occupation_title"].first().reset_index()
    summary = summary.merge(titles, on="CIUO_code", how="left")

    return summary


def main():
    print("=" * 60)
    print("Occupation-Task Matrix — Phase 1, Sprint 1.3")
    print("=" * 60)

    # Load O*NET data
    tasks = load_onet_tasks()
    abilities = load_onet_abilities()

    # Map to CIUO
    mapped = map_tasks_to_ciuo(tasks)

    # Save full matrix
    cols_to_save = [
        "CIUO_code", "SOC_code", "Task ID", "Task",
        "importance", "frequency", "relevance",
        "weight", "weighted_importance", "confidence",
        "occupation_title",
    ]
    cols_available = [c for c in cols_to_save if c in mapped.columns]
    mapped[cols_available].to_parquet(OUTPUT, index=False)
    print(f"\n  [SAVED] {OUTPUT.name}: {len(mapped)} rows")

    # Save summary
    summary = compute_occupation_task_summary(mapped)
    summary_path = PROJECT_ROOT / "data" / "processed" / "occupation_task_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    print(f"  [SAVED] {summary_path.name}: {len(summary)} occupations")

    # Save abilities if available
    if not abilities.empty:
        abilities_path = PROJECT_ROOT / "data" / "processed" / "occupation_abilities.parquet"
        abilities.to_parquet(abilities_path, index=False)
        print(f"  [SAVED] {abilities_path.name}: {len(abilities)} ability-occupation pairs")

    # Report coverage
    print(f"\n--- Coverage Report ---")
    print(f"  Total CIUO occupations with tasks: {mapped['CIUO_code'].nunique()}")
    print(f"  Total unique tasks mapped: {mapped['Task ID'].nunique()}")
    print(f"  Average tasks per occupation: {summary['n_tasks'].mean():.1f}")
    print(f"  Median tasks per occupation: {summary['n_tasks'].median():.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
