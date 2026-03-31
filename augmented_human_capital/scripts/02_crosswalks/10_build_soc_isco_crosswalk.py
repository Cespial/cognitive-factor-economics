#!/usr/bin/env python3
"""
Phase 1 — S1.1: Build SOC → ISCO-08 crosswalk.
Parses BLS crosswalk and resolves many-to-many mappings with probabilistic weights.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CROSSWALK_DIR = PROJECT_ROOT / "data" / "crosswalks"
ONET_DIR = PROJECT_ROOT / "data" / "raw" / "onet"
OUTPUT = CROSSWALK_DIR / "soc_isco.parquet"


def load_bls_crosswalk() -> pd.DataFrame:
    """Load BLS SOC-ISCO crosswalk from XLS file."""
    xls_path = CROSSWALK_DIR / "soc_isco_crosswalk.xls"

    if not xls_path.exists():
        print("[WARN] BLS crosswalk not found (download may have failed). Using fallback.")
        return pd.DataFrame()

    # Try different sheet names and header rows (BLS format varies)
    for sheet in [0, "SOC-ISCO", "Sheet1"]:
        for header_row in [0, 1, 2, 3]:
            try:
                df = pd.read_excel(xls_path, sheet_name=sheet, header=header_row)
                cols = [str(c).lower().strip() for c in df.columns]
                # Look for SOC and ISCO columns
                soc_col = next((c for c in df.columns if "soc" in str(c).lower() and "code" in str(c).lower()), None)
                isco_col = next((c for c in df.columns if "isco" in str(c).lower() and "code" in str(c).lower()), None)
                if soc_col and isco_col:
                    print(f"  Found crosswalk: sheet={sheet}, header_row={header_row}")
                    print(f"  SOC col: {soc_col}, ISCO col: {isco_col}")
                    return df.rename(columns={soc_col: "SOC_code", isco_col: "ISCO_code"})
            except Exception:
                continue

    # Fallback: try reading with flexible column detection
    df = pd.read_excel(xls_path, header=None)
    print(f"  Raw shape: {df.shape}")
    print(f"  First rows:\n{df.head()}")

    # Try to identify columns by content pattern
    for i, col in enumerate(df.columns):
        sample = df[col].dropna().astype(str).head(20)
        if sample.str.match(r"^\d{2}-\d{4}").any():
            soc_idx = i
        elif sample.str.match(r"^\d{4}$").any():
            isco_idx = i

    if "soc_idx" in dir() and "isco_idx" in dir():
        result = df[[df.columns[soc_idx], df.columns[isco_idx]]].copy()
        result.columns = ["SOC_code", "ISCO_code"]
        return result.dropna()

    print("[ERROR] Could not parse BLS crosswalk format")
    sys.exit(1)


def build_manual_crosswalk() -> pd.DataFrame:
    """Build a manual SOC-6-digit to ISCO-08-4-digit crosswalk for key occupations."""
    # Major group mapping (always works as fallback)
    major_group_map = {
        "11": "1",    # Management -> Managers
        "13": "2",    # Business/Financial -> Professionals
        "15": "2",    # Computer/Math -> Professionals
        "17": "2",    # Architecture/Engineering -> Professionals
        "19": "2",    # Life/Physical/Social Science -> Professionals
        "21": "2",    # Community/Social Service -> Professionals
        "23": "2",    # Legal -> Professionals
        "25": "2",    # Education -> Professionals
        "27": "2",    # Arts/Design/Media -> Professionals
        "29": "2",    # Healthcare Practitioners -> Professionals
        "31": "3",    # Healthcare Support -> Technicians
        "33": "5",    # Protective Service -> Service/Sales
        "35": "5",    # Food Preparation -> Service/Sales
        "37": "9",    # Building/Grounds -> Elementary
        "39": "5",    # Personal Care -> Service/Sales
        "41": "5",    # Sales -> Service/Sales
        "43": "4",    # Office/Admin -> Clerical
        "45": "6",    # Farming/Fishing -> Agriculture
        "47": "7",    # Construction -> Craft
        "49": "7",    # Installation/Maintenance -> Craft
        "51": "8",    # Production -> Operators
        "53": "8",    # Transportation -> Operators
        "55": "0",    # Military -> Armed Forces
    }
    rows = [{"SOC_2d": k, "ISCO_1d": v, "confidence": "major_group"} for k, v in major_group_map.items()]
    return pd.DataFrame(rows)


def load_onet_occupations() -> pd.DataFrame:
    """Load O*NET occupation list to get all SOC codes."""
    occ_path = ONET_DIR / "Occupation Data.txt"
    if not occ_path.exists():
        print("[WARN] O*NET Occupation Data not found")
        return pd.DataFrame()

    df = pd.read_csv(occ_path, sep="\t", encoding="utf-8", on_bad_lines="skip")
    df = df.rename(columns={"O*NET-SOC Code": "ONET_SOC"})
    # Extract 6-digit SOC (remove .XX suffix)
    df["SOC_6d"] = df["ONET_SOC"].str.replace(r"\.\d+$", "", regex=True)
    df["SOC_2d"] = df["SOC_6d"].str[:2]
    return df


def resolve_weights(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Assign probabilistic weights to many-to-many mappings."""
    # Count how many ISCO codes each SOC maps to
    soc_counts = crosswalk.groupby("SOC_code")["ISCO_code"].nunique().reset_index()
    soc_counts.columns = ["SOC_code", "n_isco"]

    crosswalk = crosswalk.merge(soc_counts, on="SOC_code")
    crosswalk["weight"] = 1.0 / crosswalk["n_isco"]

    return crosswalk


def main():
    print("=" * 60)
    print("SOC → ISCO-08 Crosswalk — Phase 1, Sprint 1.1")
    print("=" * 60)

    # Try BLS crosswalk first
    try:
        bls = load_bls_crosswalk()
        bls["SOC_code"] = bls["SOC_code"].astype(str).str.strip()
        bls["ISCO_code"] = bls["ISCO_code"].astype(str).str.strip().str.zfill(4)
        bls = bls.dropna(subset=["SOC_code", "ISCO_code"])
        bls = bls[bls["SOC_code"].str.len() > 2]

        print(f"\n  BLS crosswalk: {len(bls)} mappings")
        print(f"  Unique SOC: {bls['SOC_code'].nunique()}")
        print(f"  Unique ISCO: {bls['ISCO_code'].nunique()}")

        bls = resolve_weights(bls)
        bls["source"] = "bls_official"
        bls["confidence"] = "high"

    except Exception as e:
        print(f"  [WARN] BLS crosswalk parsing failed: {e}")
        bls = pd.DataFrame()

    # Build major group fallback
    fallback = build_manual_crosswalk()
    print(f"  Major group fallback: {len(fallback)} mappings")

    # Load O*NET occupations
    onet_occ = load_onet_occupations()
    if not onet_occ.empty:
        print(f"  O*NET occupations: {len(onet_occ)}")

        # For occupations without BLS mapping, use major group fallback
        if not bls.empty:
            mapped_soc = set(bls["SOC_code"].str[:2].unique())
        else:
            mapped_soc = set()

        unmapped = onet_occ[~onet_occ["SOC_2d"].isin(mapped_soc)]
        if not unmapped.empty:
            unmapped_with_isco = unmapped.merge(fallback, on="SOC_2d", how="left")
            print(f"  Unmapped occupations resolved via major group: {len(unmapped_with_isco)}")

    # Combine and save
    if not bls.empty:
        result = bls[["SOC_code", "ISCO_code", "weight", "source", "confidence"]].copy()
    else:
        # Use major group mapping with O*NET occupations
        result = onet_occ.merge(fallback, on="SOC_2d", how="left")
        result = result.rename(columns={"ISCO_1d": "ISCO_code"})
        result["weight"] = 1.0
        result["source"] = "major_group_fallback"
        result = result[["SOC_6d", "ISCO_code", "weight", "source", "confidence"]].rename(
            columns={"SOC_6d": "SOC_code"}
        )

    result.to_parquet(OUTPUT, index=False)
    print(f"\n  [SAVED] {OUTPUT.name}: {len(result)} rows")

    # Summary stats
    print(f"\n--- Crosswalk Summary ---")
    print(f"  Total mappings: {len(result)}")
    print(f"  Unique SOC codes: {result['SOC_code'].nunique()}")
    print(f"  Unique ISCO codes: {result['ISCO_code'].nunique()}")
    if "confidence" in result.columns:
        print(f"  By confidence: {result['confidence'].value_counts().to_dict()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
