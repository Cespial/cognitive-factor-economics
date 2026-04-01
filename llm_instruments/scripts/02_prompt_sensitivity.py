#!/usr/bin/env python3
"""
Paper 0 — Sprint 2.3: Prompt sensitivity analysis.
Score 10% subsample (1,880 tasks) with 3 alternative prompts using Haiku.
Then decompose variance: prompt vs model vs task (ANOVA).
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
P1_DATA = PROJECT / "data" / "paper1"
OUTPUT = PROJECT / "output" / "tables"
OUTPUT.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT / ".env")

# ============================================================
# 4 PROMPT VARIANTS
# ============================================================

PROMPTS = {
    "A_baseline": """You are an expert labor economist. OCCUPATION: {title}. TASK: {task}.
Rate: 1) Augmentation potential (0-100): How much can GenAI help as complementary tool?
2) Substitution risk (0-100): How much can GenAI fully replace the human?
JSON only: {{"a":<int>,"s":<int>}}""",

    "B_behavioral": """You are an expert labor economist. OCCUPATION: {title}. TASK: {task}.
If a worker performing this task had unlimited access to generative AI tools (like ChatGPT, Copilot, Claude), estimate:
1) Productivity gain (0-100): How much more output could they produce? 0=none, 100=10x more
2) Replacement feasibility (0-100): Could AI do this task alone without the human? 0=impossible, 100=fully
JSON only: {{"a":<int>,"s":<int>}}""",

    "C_counterfactual": """You are an expert labor economist. OCCUPATION: {title}. TASK: {task}.
Compare two worlds: World A has no generative AI. World B has widespread GenAI access.
1) Value change (0-100): How much MORE valuable is a human doing this task in World B vs A? 0=less valuable (AI replaces), 50=same, 100=much more valuable (AI amplifies)
2) Automation in World B (0-100): What fraction of this task can AI handle alone in World B? 0=none, 100=all
JSON only: {{"a":<int>,"s":<int>}}""",

    "D_negative": """You are an expert labor economist. OCCUPATION: {title}. TASK: {task}.
1) AI resistance (0-100): How RESISTANT is this task to being improved by AI assistance? 0=AI helps enormously, 100=AI provides zero benefit
2) AI automation resistance (0-100): How RESISTANT is this task to full AI automation? 0=easily automated, 100=impossible to automate
JSON only: {{"a":<int>,"s":<int>}}""",
}


def load_task_sample():
    """Load 10% subsample of tasks."""
    scores = []
    with open(P1_DATA / "indices" / "raw_llm_scores.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if "error" not in r and "augmentation_score" in r:
                    scores.append(r)
            except:
                pass

    # Load task texts
    tasks_df = pd.read_csv(P1_DATA / "raw" / "onet" / "Task Statements.txt", sep="\t")
    occ = pd.read_csv(P1_DATA / "raw" / "onet" / "Occupation Data.txt", sep="\t")
    occ_titles = dict(zip(
        occ["O*NET-SOC Code"].str.replace(r"\.\d+$", "", regex=True),
        occ["Title"]
    ))
    task_map = {}
    for _, row in tasks_df.iterrows():
        soc = row["O*NET-SOC Code"].split(".")[0]
        task_map[row["Task ID"]] = {"task": row["Task"], "title": occ_titles.get(soc, "Unknown")}

    # 10% sample
    random.seed(123)  # Different seed from Sonnet validation
    sample = random.sample(scores, min(int(len(scores) * 0.10), 1880))
    print(f"Prompt sensitivity sample: {len(sample)} tasks")

    return sample, task_map


def score_with_prompt(sample, task_map, prompt_name, prompt_template, client, model):
    """Score sample with a specific prompt variant."""
    outfile = OUTPUT / f"prompt_{prompt_name}_scores.jsonl"

    # Check existing
    existing = set()
    if outfile.exists():
        with open(outfile) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if "error" not in r:
                        existing.add(r["task_id"])
                except:
                    pass
    remaining = [s for s in sample if s["task_id"] not in existing]
    print(f"  {prompt_name}: {len(existing)} done, {len(remaining)} remaining")

    if not remaining:
        return

    with open(outfile, "a") as f:
        for i, s in enumerate(remaining):
            tid = s["task_id"]
            info = task_map.get(tid, {"task": "Unknown", "title": "Unknown"})

            prompt = prompt_template.format(title=info["title"], task=info["task"])

            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=60,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                text = resp.content[0].text.strip()
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
                parsed = json.loads(text)

                result = {
                    "task_id": tid,
                    "prompt": prompt_name,
                    "a": parsed.get("a", 50),
                    "s": parsed.get("s", 50),
                }
                f.write(json.dumps(result) + "\n")
            except:
                time.sleep(2)
                continue

            time.sleep(0.15)
            if (i + 1) % 500 == 0:
                f.flush()
                print(f"    {prompt_name}: {i+1}/{len(remaining)}")


def analyze_prompt_sensitivity():
    """Analyze variance decomposition across prompts."""
    print("\n=== Prompt Sensitivity Analysis ===")

    all_scores = {}
    for pname in PROMPTS.keys():
        path = OUTPUT / f"prompt_{pname}_scores.jsonl"
        if not path.exists():
            continue
        scores = {}
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if "error" not in r:
                        scores[r["task_id"]] = r["a"]
                except:
                    pass
        all_scores[pname] = scores
        print(f"  {pname}: {len(scores)} scores")

    if len(all_scores) < 2:
        print("  [SKIP] Need at least 2 prompt variants")
        return

    # Find common tasks
    common = set.intersection(*[set(s.keys()) for s in all_scores.values()])
    print(f"  Common tasks: {len(common)}")

    if len(common) < 50:
        print("  [SKIP] Insufficient common tasks")
        return

    # Build matrix: tasks × prompts
    prompts_list = list(all_scores.keys())
    matrix = np.array([[all_scores[p].get(tid, np.nan) for p in prompts_list] for tid in common])

    # Pairwise correlations between prompts
    print(f"\n  Pairwise Spearman correlations:")
    from scipy.stats import spearmanr
    for i in range(len(prompts_list)):
        for j in range(i + 1, len(prompts_list)):
            rho, p = spearmanr(matrix[:, i], matrix[:, j])
            print(f"    {prompts_list[i]} vs {prompts_list[j]}: ρ={rho:.3f} (p={p:.2e})")

    # ANOVA-style variance decomposition
    # Total variance = between-task variance + between-prompt variance + residual
    grand_mean = np.nanmean(matrix)
    task_means = np.nanmean(matrix, axis=1)  # Mean across prompts per task
    prompt_means = np.nanmean(matrix, axis=0)  # Mean across tasks per prompt

    SS_total = np.nansum((matrix - grand_mean) ** 2)
    SS_task = len(prompts_list) * np.sum((task_means - grand_mean) ** 2)
    SS_prompt = len(common) * np.sum((prompt_means - grand_mean) ** 2)
    SS_residual = SS_total - SS_task - SS_prompt

    print(f"\n  Variance Decomposition:")
    print(f"    Total SS:    {SS_total:.0f} (100%)")
    print(f"    Task SS:     {SS_task:.0f} ({SS_task/SS_total*100:.1f}%) — TRUE signal")
    print(f"    Prompt SS:   {SS_prompt:.0f} ({SS_prompt/SS_total*100:.1f}%) — Prompt sensitivity")
    print(f"    Residual SS: {SS_residual:.0f} ({SS_residual/SS_total*100:.1f}%) — Random noise")

    # Key finding
    if SS_task / SS_total > 0.7:
        print(f"\n  >>> Task variance dominates ({SS_task/SS_total*100:.0f}%): ROBUST to prompt choice")
    else:
        print(f"\n  >>> Task variance = {SS_task/SS_total*100:.0f}%: prompt sensitivity is significant")

    # Save
    results = {
        "n_tasks": int(len(common)),
        "n_prompts": len(prompts_list),
        "prompts": prompts_list,
        "prompt_means": {p: float(m) for p, m in zip(prompts_list, prompt_means)},
        "SS_total": float(SS_total),
        "SS_task": float(SS_task),
        "SS_task_pct": float(SS_task / SS_total * 100),
        "SS_prompt": float(SS_prompt),
        "SS_prompt_pct": float(SS_prompt / SS_total * 100),
        "SS_residual": float(SS_residual),
        "SS_residual_pct": float(SS_residual / SS_total * 100),
    }

    with open(OUTPUT / "prompt_sensitivity_anova.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [SAVED] prompt_sensitivity_anova.json")


def main():
    print("=" * 70)
    print("PAPER 0 — Prompt Sensitivity Analysis")
    print("=" * 70)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[ERROR] ANTHROPIC_API_KEY not set")
        return 1

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    model = "claude-haiku-4-5-20251001"

    sample, task_map = load_task_sample()

    # Score with each prompt variant
    for pname, ptemplate in PROMPTS.items():
        print(f"\n--- Scoring with prompt {pname} ---")
        score_with_prompt(sample, task_map, pname, ptemplate, client, model)

    # Analyze
    analyze_prompt_sensitivity()

    return 0


if __name__ == "__main__":
    sys.exit(main())
