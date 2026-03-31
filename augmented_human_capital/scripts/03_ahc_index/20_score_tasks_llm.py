#!/usr/bin/env python3
"""
Phase 2 — S2.2: LLM-based scoring of augmentation potential for each task.
Scores ~19K unique O*NET tasks using Claude Sonnet for:
  1) Augmentation potential (0-100)
  2) Substitution risk (0-100)
  3) Augmentation type (A-G)

Scores per unique Task ID (not per occupation-task pair).
Each task gets a representative occupation title for context.

OVERNIGHT SCRIPT — designed for unattended batch processing with checkpointing.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDICES_DIR = PROJECT_ROOT / "data" / "indices"
LOG_DIR = PROJECT_ROOT / "output" / "logs"
INDICES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSONL = INDICES_DIR / "raw_llm_scores.jsonl"
CHECKPOINT_FILE = INDICES_DIR / "scoring_checkpoint.json"

BATCH_SIZE = 10
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 0.15  # seconds between calls (Haiku has high rate limits)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"llm_scoring_{datetime.now():%Y%m%d_%H%M}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

SCORING_PROMPT = """You are an expert labor economist evaluating how generative AI affects specific occupational tasks.

OCCUPATION: {occupation_title}
TASK: {task_description}

Evaluate on three dimensions. Be precise and calibrated.

1. AUGMENTATION POTENTIAL (0-100): How much can a worker increase productivity by using GenAI as a complementary tool (not replacement)?
0=no benefit, 25=marginal, 50=moderate, 75=substantial (2-3x), 100=massive amplification

2. SUBSTITUTION RISK (0-100): How much can GenAI fully replace the human in this task?
0=impossible, 25=very difficult, 50=partial, 75=largely replaceable, 100=fully automatable

3. AUGMENTATION TYPE (A-G):
A) Information synthesis  B) Creative amplification  C) Communication enhancement
D) Decision support  E) Quality assurance  F) No meaningful augmentation  G) Pure substitution

4. CONFIDENCE: LOW, MEDIUM, or HIGH

Respond ONLY with this JSON (no markdown, no explanation):
{{"a": <int 0-100>, "s": <int 0-100>, "t": "<A-G>", "c": "<LOW|MEDIUM|HIGH>"}}"""


def load_unique_tasks() -> pd.DataFrame:
    """Load unique tasks with a representative occupation title."""
    path = PROCESSED_DIR / "task_classification.parquet"
    if not path.exists():
        logger.error("Run 14_classify_tasks_hpca.py first")
        sys.exit(1)

    df = pd.read_parquet(path)
    # Get one representative row per unique Task ID
    tasks = df.groupby("Task ID").agg(
        Task=("Task", "first"),
        occupation_title=("occupation_title", "first"),
        SOC_code=("SOC_code", "first"),
        CIUO_code=("CIUO_code", "first"),
        final_class=("final_class", "first"),
    ).reset_index()

    logger.info(f"Unique tasks to score: {len(tasks):,}")
    return tasks


def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        scored = set(data.get("scored_ids", []))
        logger.info(f"Checkpoint: {len(scored)} already scored")
        return scored
    return set()


def save_checkpoint(scored_ids: set):
    CHECKPOINT_FILE.write_text(json.dumps({
        "scored_ids": list(scored_ids),
        "last_updated": datetime.now().isoformat(),
        "total_scored": len(scored_ids),
    }))


def score_batch_anthropic(tasks_batch: list[dict], client) -> list[dict]:
    """Score a batch of tasks using Anthropic Claude."""
    results = []
    for task in tasks_batch:
        prompt = SCORING_PROMPT.format(
            occupation_title=task.get("occupation_title", "Unknown occupation"),
            task_description=task.get("Task", ""),
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
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
                results.append({
                    "task_id": task["Task ID"],
                    "soc_code": task["SOC_code"],
                    "ciuo_code": task.get("CIUO_code", ""),
                    "augmentation_score": parsed.get("a", parsed.get("augmentation_score", 50)),
                    "substitution_score": parsed.get("s", parsed.get("substitution_score", 50)),
                    "augmentation_type": parsed.get("t", parsed.get("augmentation_type", "F")),
                    "confidence": parsed.get("c", parsed.get("confidence", "MEDIUM")),
                    "llm": "claude-haiku-4-5",
                    "timestamp": datetime.now().isoformat(),
                })
                time.sleep(RATE_LIMIT_DELAY)
                break

            except json.JSONDecodeError:
                logger.warning(f"JSON parse error task {task['Task ID']}, attempt {attempt+1}, raw: {text[:200]}")
                if attempt == MAX_RETRIES - 1:
                    results.append({
                        "task_id": task["Task ID"],
                        "soc_code": task["SOC_code"],
                        "ciuo_code": task.get("CIUO_code", ""),
                        "error": "json_parse_error",
                        "raw": text[:300],
                        "llm": "claude-haiku-4-5",
                        "timestamp": datetime.now().isoformat(),
                    })
                time.sleep(1)

            except Exception as e:
                err_str = str(e)
                logger.warning(f"API error task {task['Task ID']}: {err_str[:100]}, attempt {attempt+1}")
                if "rate_limit" in err_str.lower() or "429" in err_str:
                    wait = 30 * (attempt + 1)
                    logger.info(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                elif "overloaded" in err_str.lower():
                    time.sleep(60)
                else:
                    time.sleep(5 * (attempt + 1))

                if attempt == MAX_RETRIES - 1:
                    results.append({
                        "task_id": task["Task ID"],
                        "soc_code": task["SOC_code"],
                        "error": err_str[:200],
                        "llm": "claude-haiku-4-5",
                        "timestamp": datetime.now().isoformat(),
                    })

    return results


def main():
    logger.info("=" * 60)
    logger.info("LLM Task Scoring — Phase 2, Sprint 2.2")
    logger.info("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set. Add to .env file.")
        return 1

    try:
        import anthropic
    except ImportError:
        logger.info("Installing anthropic package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic", "-q"])
        import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    tasks = load_unique_tasks()
    scored_ids = load_checkpoint()

    remaining = tasks[~tasks["Task ID"].isin(scored_ids)]
    logger.info(f"Remaining: {len(remaining):,} tasks")

    if remaining.empty:
        logger.info("All tasks already scored!")
        return 0

    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    total_scored = len(scored_ids)
    total_errors = 0
    start_time = time.time()

    with open(OUTPUT_JSONL, "a") as f:
        for batch_idx in range(0, len(remaining), BATCH_SIZE):
            batch = remaining.iloc[batch_idx:batch_idx + BATCH_SIZE]
            batch_dicts = batch.to_dict("records")
            batch_num = batch_idx // BATCH_SIZE + 1

            results = score_batch_anthropic(batch_dicts, client)

            for result in results:
                f.write(json.dumps(result) + "\n")
                if "error" not in result:
                    scored_ids.add(result["task_id"])
                    total_scored += 1
                else:
                    total_errors += 1

            f.flush()

            # Progress report every 10 batches
            if batch_num % 10 == 0 or batch_num == 1:
                elapsed = time.time() - start_time
                rate = (batch_num * BATCH_SIZE) / elapsed if elapsed > 0 else 0
                eta_min = (len(remaining) - batch_idx - BATCH_SIZE) / rate / 60 if rate > 0 else 0
                logger.info(
                    f"Batch {batch_num}/{total_batches} | "
                    f"Scored: {total_scored:,} | Errors: {total_errors} | "
                    f"Rate: {rate:.1f} tasks/s | ETA: {eta_min:.0f} min"
                )

            # Checkpoint every 50 batches
            if batch_num % 50 == 0:
                save_checkpoint(scored_ids)

    save_checkpoint(scored_ids)

    elapsed = time.time() - start_time
    logger.info(f"\nScoring complete in {elapsed/60:.1f} min")
    logger.info(f"Total scored: {total_scored:,}, Errors: {total_errors}")
    logger.info(f"Output: {OUTPUT_JSONL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
