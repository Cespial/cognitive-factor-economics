#!/usr/bin/env python3
"""
Sprint 2 — Compute Ω̇_o: occupation-specific AI advancement rate.
Maps AI benchmark growth rates to occupations via O*NET knowledge areas.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw" / "benchmarks"
P1 = PROJECT / "data" / "paper1"
OUTPUT = PROJECT / "output" / "tables"
PROCESSED = PROJECT / "data" / "processed"
for d in [OUTPUT, PROCESSED]:
    d.mkdir(parents=True, exist_ok=True)


def compute_benchmark_growth_rates():
    """Compute annual frontier growth rate per benchmark."""
    frontier = pd.read_csv(RAW / "ai_frontier.csv")
    frontier["date"] = pd.to_datetime(frontier["date"])
    frontier["year"] = frontier["date"].dt.year

    # Use 2022-2024 period (most relevant, post-ChatGPT)
    growth_rates = {}
    for bench in frontier["benchmark"].unique():
        sub = frontier[frontier["benchmark"] == bench]
        by_year = sub.groupby("year")["frontier_score"].max()

        # Average annual growth over available years
        rates = []
        years = sorted(by_year.index)
        for i in range(1, len(years)):
            prev = by_year[years[i - 1]]
            curr = by_year[years[i]]
            if prev > 0:
                rate = (curr - prev) / prev
                rates.append(rate)

        growth_rates[bench] = {
            "mean_annual_growth": np.mean(rates) if rates else 0,
            "recent_growth_2023_2024": rates[-1] if len(rates) >= 1 else 0,
            "peak_growth": max(rates) if rates else 0,
            "latest_score": float(by_year.max()),
        }

    print("Benchmark annual growth rates (frontier):")
    for bench, info in growth_rates.items():
        print(f"  {bench:12s}: mean={info['mean_annual_growth']:.1%}, "
              f"recent={info['recent_growth_2023_2024']:.1%}, "
              f"peak={info['peak_growth']:.1%}")

    return growth_rates


def load_onet_knowledge():
    """Load O*NET knowledge area importance by occupation."""
    path = P1 / "raw" / "onet" / "Knowledge.txt"
    if not path.exists():
        print("[WARN] Knowledge.txt not found, using ability-based proxy")
        return pd.DataFrame()

    know = pd.read_csv(path, sep="\t", on_bad_lines="skip")
    know["SOC"] = know["O*NET-SOC Code"].str.replace(r"\.\d+$", "", regex=True)

    # Get importance scores (Scale ID = IM)
    importance = know[know["Scale ID"] == "IM"].copy()
    importance = importance.groupby(["SOC", "Element Name"])["Data Value"].mean().reset_index()
    importance.columns = ["SOC", "knowledge_area", "importance"]

    print(f"O*NET Knowledge: {importance['SOC'].nunique()} occupations, "
          f"{importance['knowledge_area'].nunique()} knowledge areas")

    return importance


def compute_omega_dot(growth_rates, onet_knowledge):
    """
    Compute Ω̇_o for each occupation:
    Ω̇_o = Σ_b w_{ob} · growth_rate_b

    where w_{ob} = (importance of knowledge area b in occupation o) ×
                   (mapping weight from benchmark to knowledge area)
    """
    # Load benchmark-to-knowledge mapping
    mapping = pd.read_csv(RAW / "benchmark_to_onet_mapping.csv")

    if onet_knowledge.empty:
        # Fallback: use AHC scores as proxy for Ω̇
        print("[FALLBACK] Using AHC-based proxy for Ω̇")
        ahc = pd.read_csv(P1 / "indices" / "ahc_index_v2_improved_crosswalk.csv")
        ahc["omega_dot"] = ahc["AHC_score"] / 100 * 0.30  # Scale to ~30% annual growth
        return ahc[["CIUO_code", "omega_dot", "AHC_score"]]

    # For each occupation, compute weighted average of benchmark growth rates
    records = []
    for soc in onet_knowledge["SOC"].unique():
        occ_know = onet_knowledge[onet_knowledge["SOC"] == soc]

        omega_dot = 0
        total_weight = 0

        for _, m_row in mapping.iterrows():
            bench = m_row["benchmark"]
            know_area = m_row["onet_knowledge_area"]
            bench_weight = m_row["weight"]

            # Find this knowledge area's importance for this occupation
            match = occ_know[occ_know["knowledge_area"] == know_area]
            if not match.empty:
                occ_importance = match.iloc[0]["importance"] / 5.0  # Normalize 0-5 → 0-1
                bench_growth = growth_rates.get(bench, {}).get("mean_annual_growth", 0)

                w = bench_weight * occ_importance
                omega_dot += w * bench_growth
                total_weight += w

        if total_weight > 0:
            omega_dot /= total_weight  # Normalize

        records.append({
            "SOC": soc,
            "omega_dot": omega_dot,
        })

    result = pd.DataFrame(records)
    print(f"\nΩ̇ computed for {len(result)} occupations")
    print(f"  Mean Ω̇: {result['omega_dot'].mean():.4f}")
    print(f"  Std Ω̇:  {result['omega_dot'].std():.4f}")
    print(f"  Min Ω̇:  {result['omega_dot'].min():.4f}")
    print(f"  Max Ω̇:  {result['omega_dot'].max():.4f}")

    # Top and bottom occupations
    occ_data = pd.read_csv(P1 / "raw" / "onet" / "Occupation Data.txt", sep="\t")
    occ_titles = dict(zip(
        occ_data["O*NET-SOC Code"].str.replace(r"\.\d+$", "", regex=True),
        occ_data["Title"]
    ))
    result["title"] = result["SOC"].map(occ_titles)

    print(f"\n  Top 10 Ω̇ (fastest AI advancement):")
    for _, row in result.nlargest(10, "omega_dot").iterrows():
        print(f"    {row['SOC']} ({row['title'][:40]}): Ω̇={row['omega_dot']:.4f}")

    print(f"\n  Bottom 10 Ω̇ (slowest AI advancement):")
    for _, row in result.nsmallest(10, "omega_dot").iterrows():
        print(f"    {row['SOC']} ({row['title'][:40]}): Ω̇={row['omega_dot']:.4f}")

    result.to_csv(PROCESSED / "omega_dot_by_occupation.csv", index=False)
    return result


def compute_skill_halflife(omega_dot_df, delta_0=0.015):
    """
    Compute theoretical skill half-life per occupation:
    t_{1/2} = ln(2) / (δ₀ + λ · Ω̇)

    Using λ = 1 as baseline (will be calibrated later).
    """
    lambda_param = 1.0  # Will be estimated/calibrated

    omega_dot_df["delta_total"] = delta_0 + lambda_param * omega_dot_df["omega_dot"]
    omega_dot_df["half_life_years"] = np.log(2) / omega_dot_df["delta_total"]

    print(f"\n=== Skill Half-Lives (δ₀={delta_0}, λ={lambda_param}) ===")
    print(f"  Mean half-life: {omega_dot_df['half_life_years'].mean():.1f} years")
    print(f"  Median:         {omega_dot_df['half_life_years'].median():.1f} years")
    print(f"  Min:            {omega_dot_df['half_life_years'].min():.1f} years")
    print(f"  Max:            {omega_dot_df['half_life_years'].max():.1f} years")

    print(f"\n  Shortest half-lives (fastest depreciation):")
    for _, row in omega_dot_df.nsmallest(10, "half_life_years").iterrows():
        print(f"    {row.get('title', row['SOC'])[:40]}: {row['half_life_years']:.1f} years")

    print(f"\n  Longest half-lives (slowest depreciation):")
    for _, row in omega_dot_df.nlargest(10, "half_life_years").iterrows():
        print(f"    {row.get('title', row['SOC'])[:40]}: {row['half_life_years']:.1f} years")

    omega_dot_df.to_csv(PROCESSED / "skill_halflife_by_occupation.csv", index=False)
    return omega_dot_df


def main():
    print("=" * 60)
    print("SPRINT 2 — Compute Ω̇ and Skill Half-Lives")
    print("=" * 60)

    growth_rates = compute_benchmark_growth_rates()
    onet_knowledge = load_onet_knowledge()
    omega_dot = compute_omega_dot(growth_rates, onet_knowledge)
    halflife = compute_skill_halflife(omega_dot)

    print(f"\n[DONE] Saved to data/processed/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
