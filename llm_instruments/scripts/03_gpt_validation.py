#!/usr/bin/env python3
"""
Paper 0 — Sprint 2.4: Third model validation with GPT-4o-mini.
Score 20% subsample (~3,760 tasks) for three-way reliability analysis.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT / ".env")

P1_DATA = PROJECT / "data" / "paper1"
OUTPUT = PROJECT / "output" / "tables"
OUTPUT.mkdir(parents=True, exist_ok=True)

OUTFILE = OUTPUT / "gpt4o_validation_scores.jsonl"

PROMPT = """You are an expert labor economist evaluating how generative AI affects specific occupational tasks.

OCCUPATION: {title}
TASK: {task}

Rate:
1) Augmentation potential (0-100): How much can GenAI help as complementary tool?
2) Substitution risk (0-100): How much can GenAI fully replace the human?

JSON only: {{"a":<int>,"s":<int>}}"""


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set")
        return 1

    pip_install = False
    try:
        from openai import OpenAI
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai", "-q"])
        from openai import OpenAI

    client = OpenAI(api_key=api_key)

    # Test
    print("Testing GPT-4o-mini...")
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=10,
        )
        print(f"  Test: {r.choices[0].message.content}")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return 1

    # Load Haiku scores
    scores = []
    with open(P1_DATA / "indices" / "raw_llm_scores.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if "error" not in r and "augmentation_score" in r:
                    scores.append(r)
            except:
                pass
    print(f"Haiku scores: {len(scores)}")

    # Load task texts
    import pandas as pd
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

    # 20% sample (same seed as Sonnet for comparability)
    random.seed(42)
    sample = random.sample(scores, min(int(len(scores) * 0.20), 3760))
    print(f"Sample: {len(sample)} tasks")

    # Check existing
    existing = set()
    if OUTFILE.exists():
        with open(OUTFILE) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if "error" not in r:
                        existing.add(r["task_id"])
                except:
                    pass
    remaining = [s for s in sample if s["task_id"] not in existing]
    print(f"Already scored: {len(existing)}, Remaining: {len(remaining)}")

    # Score
    scored = 0
    with open(OUTFILE, "a") as f:
        for i, s in enumerate(remaining):
            tid = s["task_id"]
            info = task_map.get(tid, {"task": "Unknown", "title": "Unknown"})
            prompt = PROMPT.format(title=info["title"], task=info["task"])

            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.0,
                )
                text = resp.choices[0].message.content.strip()
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
                parsed = json.loads(text)

                result = {
                    "task_id": tid,
                    "augmentation_score": parsed.get("a", 50),
                    "substitution_score": parsed.get("s", 50),
                    "llm": "gpt-4o-mini",
                }
                f.write(json.dumps(result) + "\n")
                scored += 1

            except Exception as e:
                if "rate_limit" in str(e).lower():
                    time.sleep(30)
                else:
                    time.sleep(1)
                continue

            time.sleep(0.1)  # GPT-4o-mini has high rate limits
            if (i + 1) % 500 == 0:
                f.flush()
                print(f"  {i+1}/{len(remaining)} scored")

    print(f"\nTotal GPT-4o-mini scores: {len(existing) + scored}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
