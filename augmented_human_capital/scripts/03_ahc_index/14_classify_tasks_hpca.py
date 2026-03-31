#!/usr/bin/env python3
"""
Phase 1 — S1.4: Classify O*NET tasks into H^P, H^C, H^A components.
Uses O*NET abilities and work activities to build rules-based initial classification.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ONET_DIR = PROJECT_ROOT / "data" / "raw" / "onet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT = PROCESSED_DIR / "task_classification.parquet"

# ============================================================
# THEORETICAL CLASSIFICATION RUBRIC
# ============================================================
# Based on O*NET ability taxonomy, dual-process theory, and ALM (2003)

# H^P — Physical-Manual Component
# Tasks requiring physical strength, dexterity, motor coordination, sensory acuity
HP_ABILITIES = {
    "Manual Dexterity", "Arm-Hand Steadiness", "Finger Dexterity",
    "Control Precision", "Multilimb Coordination", "Response Orientation",
    "Rate Control", "Reaction Time", "Wrist-Finger Speed",
    "Speed of Limb Movement", "Static Strength", "Explosive Strength",
    "Dynamic Strength", "Trunk Strength", "Stamina",
    "Extent Flexibility", "Dynamic Flexibility", "Gross Body Coordination",
    "Gross Body Equilibrium",
    # Sensory (physical)
    "Night Vision", "Peripheral Vision", "Depth Perception",
    "Glare Sensitivity", "Hearing Sensitivity", "Auditory Attention",
    "Sound Localization",
}

# H^C — Cognitive-Routine Component
# Tasks requiring rule-following, data processing, pattern matching, memory recall
HC_ABILITIES = {
    "Mathematical Reasoning", "Number Facility",
    "Perceptual Speed", "Memorization", "Flexibility of Closure",
    "Speed of Closure", "Selective Attention", "Time Sharing",
    "Near Vision", "Far Vision", "Visual Color Discrimination",
    "Category Flexibility",
}

# H^A — Cognitive-Augmentable Component
# Tasks requiring judgment, creativity, strategic thinking, complex communication
HA_ABILITIES = {
    "Oral Comprehension", "Written Comprehension", "Oral Expression",
    "Written Expression", "Fluency of Ideas", "Originality",
    "Problem Sensitivity", "Deductive Reasoning", "Inductive Reasoning",
    "Information Ordering", "Visualization", "Speech Recognition",
    "Speech Clarity",
}

# Work activity keywords for classification
HP_KEYWORDS = [
    "operat", "handl", "lift", "carry", "assembl", "install",
    "repair", "driv", "load", "clean", "construct", "physical",
    "manual", "machin", "equip", "tool", "material", "build",
    "maintain", "inspect physically",
]

HC_KEYWORDS = [
    "record", "enter data", "process", "file", "sort", "calculat",
    "verify", "check", "monitor routine", "follow procedure",
    "schedule", "inventory", "tabulat", "classif", "code",
    "transcrib", "proofread", "bookkeep", "invoice", "payroll",
]

HA_KEYWORDS = [
    "analyz", "design", "develop", "plan", "strateg", "creat",
    "negotiat", "counsel", "advise", "research", "evaluat",
    "interpret", "diagnos", "coordinat", "supervis", "teach",
    "train", "mentor", "communicat", "present", "write",
    "collaborat", "innovate", "problem-solv", "decision",
    "judgment", "critical think", "consult",
]


def classify_by_abilities(abilities_df: pd.DataFrame) -> pd.DataFrame:
    """Classify occupations using O*NET ability profiles."""
    if abilities_df.empty:
        return pd.DataFrame()

    # Calculate H^P, H^C, H^A scores based on ability importance
    records = []
    for soc, group in abilities_df.groupby("SOC_code"):
        hp_score = group[group["Element Name"].isin(HP_ABILITIES)]["ability_importance"].mean()
        hc_score = group[group["Element Name"].isin(HC_ABILITIES)]["ability_importance"].mean()
        ha_score = group[group["Element Name"].isin(HA_ABILITIES)]["ability_importance"].mean()

        # Handle NaN
        hp_score = hp_score if pd.notna(hp_score) else 0
        hc_score = hc_score if pd.notna(hc_score) else 0
        ha_score = ha_score if pd.notna(ha_score) else 0

        # Normalize to proportions
        total = hp_score + hc_score + ha_score
        if total > 0:
            hp_share = hp_score / total
            hc_share = hc_score / total
            ha_share = ha_score / total
        else:
            hp_share = hc_share = ha_share = 1/3

        records.append({
            "SOC_code": soc,
            "hp_ability_score": hp_score,
            "hc_ability_score": hc_score,
            "ha_ability_score": ha_score,
            "hp_share_ability": hp_share,
            "hc_share_ability": hc_share,
            "ha_share_ability": ha_share,
            "primary_ability": max(
                ("HP", hp_share), ("HC", hc_share), ("HA", ha_share),
                key=lambda x: x[1]
            )[0],
        })

    return pd.DataFrame(records)


def classify_task_by_keywords(task_text: str) -> dict:
    """Classify a single task using keyword matching."""
    text = task_text.lower()

    hp_matches = sum(1 for kw in HP_KEYWORDS if kw in text)
    hc_matches = sum(1 for kw in HC_KEYWORDS if kw in text)
    ha_matches = sum(1 for kw in HA_KEYWORDS if kw in text)

    total = hp_matches + hc_matches + ha_matches
    if total == 0:
        return {"hp_kw": 0, "hc_kw": 0, "ha_kw": 0, "primary_kw": "UNCLASSIFIED"}

    return {
        "hp_kw": hp_matches / total,
        "hc_kw": hc_matches / total,
        "ha_kw": ha_matches / total,
        "primary_kw": max(
            ("HP", hp_matches), ("HC", hc_matches), ("HA", ha_matches),
            key=lambda x: x[1]
        )[0],
    }


def classify_all_tasks(task_matrix: pd.DataFrame) -> pd.DataFrame:
    """Classify all tasks in the occupation-task matrix."""
    print("  Classifying tasks by keywords...")
    task_texts = task_matrix[["Task ID", "Task"]].drop_duplicates()

    kw_results = []
    for _, row in task_texts.iterrows():
        result = classify_task_by_keywords(row["Task"])
        result["Task ID"] = row["Task ID"]
        kw_results.append(result)

    kw_df = pd.DataFrame(kw_results)
    return task_matrix.merge(kw_df, on="Task ID", how="left")


def combine_classifications(task_df: pd.DataFrame, ability_df: pd.DataFrame) -> pd.DataFrame:
    """Combine keyword-based and ability-based classifications."""
    if ability_df.empty:
        # Use keyword-only classification
        task_df["final_hp"] = task_df.get("hp_kw", 1/3)
        task_df["final_hc"] = task_df.get("hc_kw", 1/3)
        task_df["final_ha"] = task_df.get("ha_kw", 1/3)
        task_df["final_class"] = task_df.get("primary_kw", "UNCLASSIFIED")
        return task_df

    # Merge ability scores at occupation level
    merged = task_df.merge(ability_df, on="SOC_code", how="left")

    # Combine: 50% keyword, 50% ability
    merged["final_hp"] = 0.5 * merged.get("hp_kw", 0).fillna(0) + 0.5 * merged.get("hp_share_ability", 1/3).fillna(1/3)
    merged["final_hc"] = 0.5 * merged.get("hc_kw", 0).fillna(0) + 0.5 * merged.get("hc_share_ability", 1/3).fillna(1/3)
    merged["final_ha"] = 0.5 * merged.get("ha_kw", 0).fillna(0) + 0.5 * merged.get("ha_share_ability", 1/3).fillna(1/3)

    # Normalize
    row_sum = merged["final_hp"] + merged["final_hc"] + merged["final_ha"]
    row_sum = row_sum.replace(0, 1)
    merged["final_hp"] /= row_sum
    merged["final_hc"] /= row_sum
    merged["final_ha"] /= row_sum

    # Primary classification
    merged["final_class"] = merged[["final_hp", "final_hc", "final_ha"]].idxmax(axis=1).map({
        "final_hp": "HP", "final_hc": "HC", "final_ha": "HA"
    })

    return merged


def main():
    print("=" * 60)
    print("Task Classification (H^P / H^C / H^A) — Phase 1, Sprint 1.4")
    print("=" * 60)

    # Load task matrix
    task_path = PROCESSED_DIR / "occupation_task_matrix.parquet"
    if not task_path.exists():
        print("[ERROR] Run 13_build_task_matrix.py first")
        sys.exit(1)

    task_matrix = pd.read_parquet(task_path)
    print(f"  Task matrix: {len(task_matrix)} rows")

    # Load abilities
    abilities_path = PROCESSED_DIR / "occupation_abilities.parquet"
    if abilities_path.exists():
        abilities = pd.read_parquet(abilities_path)
        ability_scores = classify_by_abilities(abilities)
        print(f"  Ability-based classification: {len(ability_scores)} occupations")
    else:
        ability_scores = pd.DataFrame()
        print("  [WARN] No ability data; using keyword-only classification")

    # Classify tasks by keywords
    classified = classify_all_tasks(task_matrix)

    # Combine classifications
    final = combine_classifications(classified, ability_scores)

    # Save
    final.to_parquet(OUTPUT, index=False)
    print(f"\n  [SAVED] {OUTPUT.name}: {len(final)} rows")

    # Summary
    print(f"\n--- Classification Summary ---")
    if "final_class" in final.columns:
        class_dist = final["final_class"].value_counts()
        print(f"  Tasks by primary class:")
        for cls, count in class_dist.items():
            pct = count / len(final) * 100
            print(f"    {cls}: {count:,} ({pct:.1f}%)")

    # Average scores
    for col in ["final_hp", "final_hc", "final_ha"]:
        if col in final.columns:
            print(f"  Mean {col}: {final[col].mean():.3f}")

    # Save classification rules for documentation
    rules = {
        "HP_abilities": sorted(HP_ABILITIES),
        "HC_abilities": sorted(HC_ABILITIES),
        "HA_abilities": sorted(HA_ABILITIES),
        "HP_keywords": HP_KEYWORDS,
        "HC_keywords": HC_KEYWORDS,
        "HA_keywords": HA_KEYWORDS,
        "combination_method": "50% keyword + 50% ability scores",
        "n_tasks_classified": len(final),
    }
    rules_path = PROCESSED_DIR / "task_classification_rules.json"
    rules_path.write_text(json.dumps(rules, indent=2))
    print(f"  [SAVED] {rules_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
