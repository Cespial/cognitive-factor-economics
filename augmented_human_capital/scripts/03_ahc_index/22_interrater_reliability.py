#!/usr/bin/env python3
"""
Sprint 2 — Inter-rater reliability: re-score 20% of tasks with Claude Sonnet.
Computes Krippendorff's alpha between Haiku and Sonnet scores.
Target: alpha > 0.7 for augmentation scores.
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

INDICES_DIR = PROJECT_ROOT / "data" / "indices"
LOG_DIR = PROJECT_ROOT / "output" / "logs"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
for d in [INDICES_DIR, LOG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SONNET_SCORES_FILE = INDICES_DIR / "sonnet_validation_scores.jsonl"
SAMPLE_FRACTION = 0.20  # 20% of tasks
RATE_LIMIT_DELAY = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"interrater_{datetime.now():%Y%m%d_%H%M}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

SCORING_PROMPT = """You are an expert labor economist evaluating how generative AI affects specific occupational tasks.

OCCUPATION: {occupation_title}
TASK: {task_description}

Evaluate on three dimensions:
1. AUGMENTATION POTENTIAL (0-100): How much can GenAI help as a complementary tool?
2. SUBSTITUTION RISK (0-100): How much can GenAI fully replace the human?
3. AUGMENTATION TYPE: A)Info synthesis B)Creative C)Communication D)Decision support E)QA F)None G)Pure substitution
4. CONFIDENCE: LOW/MEDIUM/HIGH

Respond ONLY with JSON: {{"a": <0-100>, "s": <0-100>, "t": "<A-G>", "c": "<LOW|MEDIUM|HIGH>"}}"""


def load_haiku_scores() -> pd.DataFrame:
    """Load original Haiku scores."""
    path = INDICES_DIR / "raw_llm_scores.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            if "error" not in r and "augmentation_score" in r:
                records.append(r)
    df = pd.DataFrame(records)
    logger.info(f"Haiku scores: {len(df):,}")
    return df


def select_validation_sample(haiku_df: pd.DataFrame) -> pd.DataFrame:
    """Select stratified 20% sample for Sonnet validation."""
    random.seed(42)
    n_sample = int(len(haiku_df) * SAMPLE_FRACTION)

    # Stratify by augmentation score quartiles
    haiku_df["aug_quartile"] = pd.qcut(haiku_df["augmentation_score"], 4, labels=False)
    sample = haiku_df.groupby("aug_quartile", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), n_sample // 4), random_state=42)
    )

    logger.info(f"Validation sample: {len(sample):,} tasks ({len(sample)/len(haiku_df):.1%})")
    return sample


def load_task_texts() -> dict:
    """Load task descriptions from classification file."""
    path = PROJECT_ROOT / "data" / "processed" / "task_classification.parquet"
    df = pd.read_parquet(path)
    # Get unique task text + occupation title per task_id + soc_code
    tasks = df.groupby(["Task ID", "SOC_code"]).agg(
        Task=("Task", "first"),
        occupation_title=("occupation_title", "first"),
    ).reset_index()
    return {(str(r["Task ID"]), r["SOC_code"]): {
        "Task": r["Task"],
        "occupation_title": r["occupation_title"]
    } for _, r in tasks.iterrows()}


def score_with_sonnet(sample: pd.DataFrame, task_texts: dict) -> list:
    """Re-score sample with Claude Sonnet."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return []

    client = anthropic.Anthropic(api_key=api_key)
    results = []

    # Check for existing scores
    existing_ids = set()
    if SONNET_SCORES_FILE.exists():
        with open(SONNET_SCORES_FILE) as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r:
                    existing_ids.add(r.get("task_id"))
        logger.info(f"Existing Sonnet scores: {len(existing_ids)}")

    remaining = sample[~sample["task_id"].isin(existing_ids)]
    logger.info(f"Remaining to score: {len(remaining)}")

    with open(SONNET_SCORES_FILE, "a") as f:
        for i, (_, row) in enumerate(remaining.iterrows()):
            task_id = row["task_id"]
            soc = row["soc_code"]

            # Get task text
            key = (str(task_id), soc)
            if key in task_texts:
                text_info = task_texts[key]
            else:
                # Try just task_id
                matching = {k: v for k, v in task_texts.items() if k[0] == str(task_id)}
                if matching:
                    text_info = list(matching.values())[0]
                else:
                    continue

            prompt = SCORING_PROMPT.format(
                occupation_title=text_info.get("occupation_title", "Unknown"),
                task_description=text_info.get("Task", ""),
            )

            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                text = response.content[0].text.strip()

                # Extract JSON
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]

                parsed = json.loads(text)
                result = {
                    "task_id": task_id,
                    "soc_code": soc,
                    "augmentation_score": parsed.get("a", 50),
                    "substitution_score": parsed.get("s", 50),
                    "augmentation_type": parsed.get("t", "F"),
                    "confidence": parsed.get("c", "MEDIUM"),
                    "llm": "claude-sonnet-4",
                    "timestamp": datetime.now().isoformat(),
                }
                f.write(json.dumps(result) + "\n")
                results.append(result)

            except Exception as e:
                logger.warning(f"Error task {task_id}: {str(e)[:100]}")
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    time.sleep(30)
                else:
                    time.sleep(2)
                continue

            time.sleep(RATE_LIMIT_DELAY)

            if (i + 1) % 100 == 0:
                f.flush()
                logger.info(f"  Scored {i+1}/{len(remaining)} with Sonnet")

    return results


def compute_krippendorff_alpha(haiku_scores: list, sonnet_scores: list) -> float:
    """Compute Krippendorff's alpha for inter-rater reliability."""
    # Build paired arrays
    haiku_dict = {s["task_id"]: s["augmentation_score"] for s in haiku_scores}
    sonnet_dict = {s["task_id"]: s["augmentation_score"] for s in sonnet_scores}

    common_ids = set(haiku_dict.keys()) & set(sonnet_dict.keys())
    if len(common_ids) < 10:
        return np.nan

    rater1 = np.array([haiku_dict[tid] for tid in common_ids])
    rater2 = np.array([sonnet_dict[tid] for tid in common_ids])

    # Krippendorff's alpha for interval data
    # Using the formula: alpha = 1 - D_observed / D_expected
    n = len(common_ids)

    # All values pooled
    all_values = np.concatenate([rater1, rater2])
    grand_mean = all_values.mean()

    # Observed disagreement
    D_o = np.mean((rater1 - rater2) ** 2)

    # Expected disagreement (variance of all values)
    D_e = np.var(all_values, ddof=0) * 2  # Factor of 2 for 2 raters

    if D_e == 0:
        return 1.0 if D_o == 0 else 0.0

    alpha = 1.0 - D_o / D_e
    return alpha


def compute_reliability_stats(haiku_df: pd.DataFrame, sonnet_results: list) -> dict:
    """Compute comprehensive reliability statistics."""
    sonnet_dict_aug = {r["task_id"]: r["augmentation_score"] for r in sonnet_results}
    sonnet_dict_sub = {r["task_id"]: r["substitution_score"] for r in sonnet_results}
    sonnet_dict_type = {r["task_id"]: r["augmentation_type"] for r in sonnet_results}

    # Match with Haiku
    paired = haiku_df[haiku_df["task_id"].isin(sonnet_dict_aug.keys())].copy()
    paired["sonnet_aug"] = paired["task_id"].map(sonnet_dict_aug)
    paired["sonnet_sub"] = paired["task_id"].map(sonnet_dict_sub)
    paired["sonnet_type"] = paired["task_id"].map(sonnet_dict_type)

    n = len(paired)
    if n < 10:
        return {"n_paired": n, "error": "insufficient pairs"}

    # Correlations
    cor_aug = paired["augmentation_score"].corr(paired["sonnet_aug"])
    cor_sub = paired["substitution_score"].corr(paired["sonnet_sub"])

    # Mean absolute difference
    mad_aug = (paired["augmentation_score"] - paired["sonnet_aug"]).abs().mean()
    mad_sub = (paired["substitution_score"] - paired["sonnet_sub"]).abs().mean()

    # Type agreement (exact match rate)
    type_agree = (paired["augmentation_type"] == paired["sonnet_type"]).mean()

    # Krippendorff's alpha
    haiku_list = [{"task_id": r["task_id"], "augmentation_score": r["augmentation_score"]}
                  for _, r in paired.iterrows()]
    alpha_aug = compute_krippendorff_alpha(haiku_list, sonnet_results)

    return {
        "n_paired": n,
        "cor_augmentation": round(cor_aug, 3),
        "cor_substitution": round(cor_sub, 3),
        "mad_augmentation": round(mad_aug, 1),
        "mad_substitution": round(mad_sub, 1),
        "type_agreement": round(type_agree, 3),
        "krippendorff_alpha_aug": round(alpha_aug, 3),
        "haiku_aug_mean": round(paired["augmentation_score"].mean(), 1),
        "sonnet_aug_mean": round(paired["sonnet_aug"].mean(), 1),
        "haiku_sub_mean": round(paired["substitution_score"].mean(), 1),
        "sonnet_sub_mean": round(paired["sonnet_sub"].mean(), 1),
    }


def main():
    logger.info("=" * 60)
    logger.info("INTER-RATER RELIABILITY — Sprint 2")
    logger.info("=" * 60)

    haiku_df = load_haiku_scores()
    sample = select_validation_sample(haiku_df)
    task_texts = load_task_texts()

    logger.info(f"Task texts loaded: {len(task_texts):,}")

    # Score with Sonnet
    logger.info("\n--- Scoring with Claude Sonnet ---")
    sonnet_results = score_with_sonnet(sample, task_texts)

    # Also load any existing sonnet scores
    all_sonnet = []
    if SONNET_SCORES_FILE.exists():
        with open(SONNET_SCORES_FILE) as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r:
                    all_sonnet.append(r)

    logger.info(f"\nTotal Sonnet scores: {len(all_sonnet)}")

    # Compute reliability
    logger.info("\n--- Computing Reliability Statistics ---")
    stats = compute_reliability_stats(haiku_df, all_sonnet)

    logger.info(f"\n{'='*60}")
    logger.info("INTER-RATER RELIABILITY RESULTS")
    logger.info(f"{'='*60}")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Check threshold
    alpha = stats.get("krippendorff_alpha_aug", 0)
    if alpha > 0.7:
        logger.info(f"\n  >>> PASSES threshold: alpha = {alpha} > 0.7")
    elif alpha > 0.6:
        logger.info(f"\n  >>> MARGINAL: alpha = {alpha} (0.6-0.7 range)")
    else:
        logger.info(f"\n  >>> BELOW threshold: alpha = {alpha} < 0.6")

    # Save results
    results_path = TABLE_DIR / "interrater_reliability.json"
    results_path.write_text(json.dumps(stats, indent=2))
    logger.info(f"\n  [SAVED] {results_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
