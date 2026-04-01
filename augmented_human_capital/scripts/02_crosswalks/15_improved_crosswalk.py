#!/usr/bin/env python3
"""
Sprint 2 — Improved SOC → ISCO-08 crosswalk at 2-digit level.
Uses the well-established SOC-to-ISCO correspondence from ILO/BLS.

The key improvement: instead of mapping all occupations within a SOC major
group to the same ISCO major group (1-digit), we map at the 2-digit level
which gives ~40 occupation groups instead of 10.

Then we rebuild the task matrix and AHC index with this finer crosswalk.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ONET_DIR = PROJECT_ROOT / "data" / "raw" / "onet"
CROSSWALK_DIR = PROJECT_ROOT / "data" / "crosswalks"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDICES_DIR = PROJECT_ROOT / "data" / "indices"

# =============================================================
# SOC 2-digit → ISCO-08 2-digit mapping
# Based on BLS/ILO correspondence tables
# =============================================================
SOC_TO_ISCO_2D = {
    # Management
    "11": ["11", "12", "13", "14"],  # SOC Management → ISCO Chief executives, Admin managers, Production managers, Hospitality managers
    # Business and Financial
    "13": ["24", "33", "24"],  # → ISCO Business/admin professionals, Business associate professionals
    # Computer and Mathematical
    "15": ["25", "35"],  # → ISCO ICT professionals, ICT technicians
    # Architecture and Engineering
    "17": ["21", "31"],  # → ISCO Science/engineering professionals, Sci/eng technicians
    # Life, Physical, Social Science
    "19": ["21", "22", "26"],  # → ISCO Science, Health, Social/Cultural professionals
    # Community and Social Service
    "21": ["26", "34"],  # → ISCO Social/Cultural professionals, Legal/social associate
    # Legal
    "23": ["26"],  # → ISCO Legal professionals
    # Educational Instruction
    "25": ["23", "23"],  # → ISCO Teaching professionals
    # Arts, Design, Entertainment, Sports, Media
    "27": ["26", "34", "35"],  # → ISCO Arts/cultural + associate
    # Healthcare Practitioners
    "29": ["22", "32"],  # → ISCO Health professionals, Health associate professionals
    # Healthcare Support
    "31": ["32", "53"],  # → ISCO Health associate, Personal care
    # Protective Service
    "33": ["03", "54"],  # → ISCO Armed forces, Protective services
    # Food Preparation and Serving
    "35": ["51", "94"],  # → ISCO Personal service, Food preparation
    # Building and Grounds Cleaning
    "37": ["91"],  # → ISCO Cleaners and helpers
    # Personal Care and Service
    "39": ["51", "53"],  # → ISCO Personal service, Personal care
    # Sales
    "41": ["33", "52"],  # → ISCO Business associate, Sales workers
    # Office and Administrative Support
    "43": ["41", "42", "43", "44"],  # → ISCO General clerks, Secretaries, Numerical, Other clerical
    # Farming, Fishing, Forestry
    "45": ["61", "62", "63", "92"],  # → ISCO Agriculture/forestry/fishery workers
    # Construction and Extraction
    "47": ["71"],  # → ISCO Building/related trades
    # Installation, Maintenance, Repair
    "49": ["72", "73", "74"],  # → ISCO Metal/machinery, Craft/printing, Electrical
    # Production
    "51": ["75", "81", "82"],  # → ISCO Food/garment, Stationary plant, Assemblers
    # Transportation
    "53": ["83", "93"],  # → ISCO Drivers, Labourers in transport
    # Military
    "55": ["01", "02", "03"],  # → ISCO Armed forces
}


def build_2digit_crosswalk() -> pd.DataFrame:
    """Build SOC-2d → ISCO-2d crosswalk with weights."""
    rows = []
    for soc_2d, isco_list in SOC_TO_ISCO_2D.items():
        # Remove duplicates
        unique_isco = list(dict.fromkeys(isco_list))
        weight = 1.0 / len(unique_isco)
        for isco_2d in unique_isco:
            rows.append({
                "SOC_2d": soc_2d,
                "ISCO_2d": isco_2d,
                "weight": weight,
                "confidence": "2digit",
            })
    return pd.DataFrame(rows)


def map_onet_to_isco(crosswalk_2d: pd.DataFrame) -> pd.DataFrame:
    """Map all O*NET occupations to ISCO-08 2-digit codes."""
    onet = pd.read_csv(ONET_DIR / "Occupation Data.txt", sep="\t")
    onet["SOC_6d"] = onet["O*NET-SOC Code"].str.replace(r"\.\d+$", "", regex=True)
    onet["SOC_2d"] = onet["SOC_6d"].str[:2]

    # Merge with crosswalk
    mapped = onet.merge(crosswalk_2d, on="SOC_2d", how="left")

    print(f"  O*NET occupations: {len(onet)}")
    print(f"  Mapped to ISCO-2d: {len(mapped)}")
    print(f"  Unique ISCO-2d: {mapped['ISCO_2d'].nunique()}")

    return mapped


def map_isco_to_ciuo(mapped: pd.DataFrame) -> pd.DataFrame:
    """Map ISCO-08 2-digit to CIUO-08 2-digit (direct 1:1 for Colombia)."""
    # CIUO-08 AC is identical to ISCO-08 at 2-digit level
    mapped["CIUO_2d"] = mapped["ISCO_2d"]

    # Load GEIH to get actual 4-digit CIUO codes per 2-digit group
    est = pd.read_parquet(PROCESSED_DIR / "estimation_sample.parquet")
    est["CIUO_2d_geih"] = est["CIUO_4d"].str[:2]

    # Get employment distribution within each 2-digit group
    ciuo_dist = est.groupby(["CIUO_2d_geih", "CIUO_4d"]).agg(
        n_workers=("weight", "sum")
    ).reset_index()
    ciuo_dist.columns = ["CIUO_2d", "CIUO_4d_target", "n_workers"]

    # For each ISCO-2d → distribute across actual CIUO 4-digit codes
    chained = mapped.merge(ciuo_dist, left_on="CIUO_2d", right_on="CIUO_2d", how="inner")

    # Adjust weight by employment share within 2-digit group
    total_by_2d = chained.groupby(["O*NET-SOC Code", "CIUO_2d"])["n_workers"].transform("sum")
    chained["final_weight"] = chained["weight"] * chained["n_workers"] / total_by_2d.clip(lower=1)

    print(f"  Chained SOC → CIUO-4d: {len(chained)}")
    print(f"  Unique CIUO-4d: {chained['CIUO_4d_target'].nunique()}")

    return chained


def rebuild_ahc_with_improved_crosswalk(chained: pd.DataFrame):
    """Rebuild AHC index using improved crosswalk."""
    import json

    # Load LLM scores
    scores_path = INDICES_DIR / "raw_llm_scores.jsonl"
    scores = {}
    with open(scores_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                scores[r["task_id"]] = r

    # Load task matrix
    tasks = pd.read_csv(ONET_DIR / "Task Statements.txt", sep="\t")
    tasks["SOC_6d"] = tasks["O*NET-SOC Code"].str.replace(r"\.\d+$", "", regex=True)

    # Load task ratings for importance
    ratings = pd.read_csv(ONET_DIR / "Task Ratings.txt", sep="\t")
    importance = ratings[ratings["Scale ID"] == "IM"].groupby(
        ["O*NET-SOC Code", "Task ID"]
    )["Data Value"].mean().reset_index()
    importance.columns = ["O*NET-SOC Code", "Task ID", "importance"]

    tasks = tasks.merge(importance, on=["O*NET-SOC Code", "Task ID"], how="left")
    tasks["importance"] = tasks["importance"].fillna(3.0)

    # Merge tasks with scores
    tasks["aug_score"] = tasks["Task ID"].map(lambda tid: scores.get(tid, {}).get("augmentation_score", np.nan))
    tasks["sub_score"] = tasks["Task ID"].map(lambda tid: scores.get(tid, {}).get("substitution_score", np.nan))
    tasks = tasks.dropna(subset=["aug_score"])

    # Merge with improved crosswalk (SOC → CIUO-4d)
    tasks["SOC_6d"] = tasks["O*NET-SOC Code"].str.replace(r"\.\d+$", "", regex=True)
    chained_slim = chained[["SOC_6d", "CIUO_4d_target", "final_weight"]].drop_duplicates()

    # Aggregate: for each CIUO-4d, compute weighted AHC
    merged = tasks.merge(chained_slim, on="SOC_6d", how="inner")
    merged["weighted_aug"] = merged["importance"] * merged["final_weight"] * merged["aug_score"]
    merged["weighted_sub"] = merged["importance"] * merged["final_weight"] * merged["sub_score"]
    merged["weight_sum"] = merged["importance"] * merged["final_weight"]

    ahc_v2 = merged.groupby("CIUO_4d_target").agg(
        AHC_score=("weighted_aug", "sum"),
        SUB_score=("weighted_sub", "sum"),
        weight_total=("weight_sum", "sum"),
        n_tasks=("Task ID", "nunique"),
        n_soc=("SOC_6d", "nunique"),
    ).reset_index()

    ahc_v2["AHC_score"] = ahc_v2["AHC_score"] / ahc_v2["weight_total"].clip(lower=0.001)
    ahc_v2["SUB_score"] = ahc_v2["SUB_score"] / ahc_v2["weight_total"].clip(lower=0.001)
    ahc_v2 = ahc_v2.rename(columns={"CIUO_4d_target": "CIUO_code"})

    # Normalize
    ahc_v2["AHC_normalized"] = (ahc_v2["AHC_score"] - ahc_v2["AHC_score"].min()) / \
                                (ahc_v2["AHC_score"].max() - ahc_v2["AHC_score"].min())

    # Save
    output = INDICES_DIR / "ahc_index_v2_improved_crosswalk.parquet"
    ahc_v2.to_parquet(output, index=False)
    ahc_v2.to_csv(output.with_suffix(".csv"), index=False)

    print(f"\n  [SAVED] ahc_index_v2: {len(ahc_v2)} occupations")
    print(f"  AHC: mean={ahc_v2['AHC_score'].mean():.1f}, std={ahc_v2['AHC_score'].std():.1f}, "
          f"min={ahc_v2['AHC_score'].min():.1f}, max={ahc_v2['AHC_score'].max():.1f}")
    print(f"  SUB: mean={ahc_v2['SUB_score'].mean():.1f}, std={ahc_v2['SUB_score'].std():.1f}")
    print(f"  cor(AHC, SUB) = {ahc_v2['AHC_score'].corr(ahc_v2['SUB_score']):.3f}")
    print(f"  Avg tasks per occupation: {ahc_v2['n_tasks'].mean():.1f}")
    print(f"  Avg SOC sources per occupation: {ahc_v2['n_soc'].mean():.1f}")

    return ahc_v2


def main():
    print("=" * 60)
    print("IMPROVED CROSSWALK — Sprint 2")
    print("=" * 60)

    # Build 2-digit crosswalk
    crosswalk_2d = build_2digit_crosswalk()
    print(f"  2-digit crosswalk: {len(crosswalk_2d)} SOC-ISCO pairs")

    # Map O*NET to ISCO
    mapped = map_onet_to_isco(crosswalk_2d)

    # Chain to CIUO
    chained = map_isco_to_ciuo(mapped)

    # Save crosswalk
    crosswalk_path = CROSSWALK_DIR / "soc_ciuo_improved_2digit.parquet"
    chained.to_parquet(crosswalk_path, index=False)
    print(f"  [SAVED] {crosswalk_path.name}")

    # Rebuild AHC index
    print("\n--- Rebuilding AHC Index with Improved Crosswalk ---")
    ahc_v2 = rebuild_ahc_with_improved_crosswalk(chained)

    # Compare v1 vs v2
    v1_path = INDICES_DIR / "ahc_index_by_occupation.parquet"
    if v1_path.exists():
        v1 = pd.read_parquet(v1_path)
        # Merge on CIUO code
        comparison = v1.merge(ahc_v2, on="CIUO_code", suffixes=("_v1", "_v2"), how="inner")
        if len(comparison) > 5:
            cor = comparison["AHC_score_v1"].corr(comparison["AHC_score_v2"])
            print(f"\n  v1 vs v2 correlation: {cor:.3f}")
            print(f"  v1 occupations: {len(v1)}, v2 occupations: {len(ahc_v2)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
