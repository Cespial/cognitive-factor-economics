#!/usr/bin/env python3
"""
Phase 1 — S1.3: Chain SOC → ISCO → CIUO crosswalk.
Produces the complete mapping from O*NET occupations to Colombian CIUO-08 codes.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CROSSWALK_DIR = PROJECT_ROOT / "data" / "crosswalks"
ONET_DIR = PROJECT_ROOT / "data" / "raw" / "onet"
OUTPUT = CROSSWALK_DIR / "soc_ciuo_chained.parquet"


def load_crosswalks():
    soc_isco = pd.read_parquet(CROSSWALK_DIR / "soc_isco.parquet")
    isco_ciuo = pd.read_parquet(CROSSWALK_DIR / "isco_ciuo.parquet")
    print(f"  SOC-ISCO: {len(soc_isco)} mappings")
    print(f"  ISCO-CIUO: {len(isco_ciuo)} mappings")
    return soc_isco, isco_ciuo


def chain_crosswalks(soc_isco: pd.DataFrame, isco_ciuo: pd.DataFrame) -> pd.DataFrame:
    """Chain SOC → ISCO → CIUO with probabilistic weights."""
    # Normalize code formats
    soc_isco["SOC_code"] = soc_isco["SOC_code"].astype(str).str.strip()
    soc_isco["ISCO_code"] = soc_isco["ISCO_code"].astype(str).str.strip()
    isco_ciuo["ISCO_code"] = isco_ciuo["ISCO_code"].astype(str).str.strip()
    isco_ciuo["CIUO_code"] = isco_ciuo["CIUO_code"].astype(str).str.strip()

    # Determine matching level: try 4-digit first, fall back to 2-digit, then 1-digit
    chained_rows = []

    for _, soc_row in soc_isco.iterrows():
        soc = soc_row["SOC_code"]
        isco = soc_row["ISCO_code"]
        w1 = soc_row.get("weight", 1.0)

        # Try exact ISCO match
        exact = isco_ciuo[isco_ciuo["ISCO_code"] == isco]
        if not exact.empty:
            for _, ciuo_row in exact.iterrows():
                chained_rows.append({
                    "SOC_code": soc,
                    "ISCO_code": isco,
                    "CIUO_code": ciuo_row["CIUO_code"],
                    "weight": w1 * ciuo_row.get("weight", 1.0),
                    "match_level": "exact",
                    "confidence": "high",
                })
            continue

        # Try 2-digit ISCO prefix match
        isco_2d = isco[:2] if len(isco) >= 2 else isco
        prefix_2d = isco_ciuo[isco_ciuo["ISCO_code"].str[:2] == isco_2d]
        if not prefix_2d.empty:
            # Distribute weight evenly across matches
            n = len(prefix_2d)
            for _, ciuo_row in prefix_2d.iterrows():
                chained_rows.append({
                    "SOC_code": soc,
                    "ISCO_code": isco,
                    "CIUO_code": ciuo_row["CIUO_code"],
                    "weight": w1 / n,
                    "match_level": "2digit",
                    "confidence": "medium",
                })
            continue

        # Try 1-digit ISCO prefix match
        isco_1d = isco[0]
        prefix_1d = isco_ciuo[isco_ciuo["ISCO_code"].str[0] == isco_1d]
        if not prefix_1d.empty:
            n = len(prefix_1d)
            for _, ciuo_row in prefix_1d.iterrows():
                chained_rows.append({
                    "SOC_code": soc,
                    "ISCO_code": isco,
                    "CIUO_code": ciuo_row["CIUO_code"],
                    "weight": w1 / n,
                    "match_level": "1digit",
                    "confidence": "low",
                })
            continue

        # No match at all
        chained_rows.append({
            "SOC_code": soc,
            "ISCO_code": isco,
            "CIUO_code": None,
            "weight": 0.0,
            "match_level": "unmatched",
            "confidence": "none",
        })

    return pd.DataFrame(chained_rows)


def merge_onet_titles(chained: pd.DataFrame) -> pd.DataFrame:
    """Add O*NET occupation titles to chained crosswalk."""
    occ_path = ONET_DIR / "Occupation Data.txt"
    if not occ_path.exists():
        return chained

    onet = pd.read_csv(occ_path, sep="\t", encoding="utf-8", on_bad_lines="skip")
    onet = onet.rename(columns={"O*NET-SOC Code": "ONET_SOC", "Title": "occupation_title"})
    onet["SOC_code"] = onet["ONET_SOC"].str.replace(r"\.\d+$", "", regex=True)

    # Keep first title per SOC (some have multiple O*NET-SOC variants like .00, .01)
    titles = onet.groupby("SOC_code")["occupation_title"].first().reset_index()
    chained = chained.merge(titles, on="SOC_code", how="left")
    return chained


def main():
    print("=" * 60)
    print("Chained Crosswalk SOC → ISCO → CIUO — Phase 1, Sprint 1.3")
    print("=" * 60)

    soc_isco, isco_ciuo = load_crosswalks()
    chained = chain_crosswalks(soc_isco, isco_ciuo)
    chained = merge_onet_titles(chained)

    # Remove unmatched
    matched = chained[chained["CIUO_code"].notna()].copy()
    unmatched = chained[chained["CIUO_code"].isna()]

    print(f"\n--- Chaining Results ---")
    print(f"  Total mappings: {len(chained)}")
    print(f"  Matched: {len(matched)} ({len(matched)/len(chained):.1%})")
    print(f"  Unmatched: {len(unmatched)} ({len(unmatched)/len(chained):.1%})")
    print(f"  By match level:")
    print(matched["match_level"].value_counts().to_string(header=False))
    print(f"  By confidence:")
    print(matched["confidence"].value_counts().to_string(header=False))

    # Save
    matched.to_parquet(OUTPUT, index=False)
    print(f"\n  [SAVED] {OUTPUT.name}: {len(matched)} rows")

    # Save unmatched for debugging
    if not unmatched.empty:
        unmatched_path = CROSSWALK_DIR / "unmatched_soc_codes.csv"
        unmatched.to_csv(unmatched_path, index=False)
        print(f"  [SAVED] {unmatched_path.name}: {len(unmatched)} unmatched codes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
