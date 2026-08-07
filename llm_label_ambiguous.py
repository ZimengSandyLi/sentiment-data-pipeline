"""
LLM Labelling for Ambiguous Reviews
=====================================
Identifies reviews where the rating signal and VADER text signal conflict,
then uses Claude as a third signal to produce a final label via majority voting.

Target reviews (two categories):
  1. Rating-text disagreement: vader_label != combined_label (~660 reviews)
  2. 3-star reviews: no reliable anchor signal from either rating or VADER (~1,452 reviews)

Voting logic:
  - 3 signals: rating_signal, vader_label, llm_label
  - If 2+ agree → final_label = majority
  - If all 3 disagree → final_label = "ambiguous"
  - label_confidence tracks how many signals agreed (0.67 = 2/3, 1.0 = 3/3)

Output:
  - Updates features.db with final_label, label_confidence, llm_sentiment_label columns
  - Saves llm_labelling_results.csv for inspection

Usage:
    python llm_label_ambiguous.py               # run on all ambiguous reviews
    python llm_label_ambiguous.py --limit 50    # test on small batch first
    python llm_label_ambiguous.py --dry-run     # preview without API calls
"""

import sqlite3
import pandas as pd
import anthropic
import json
import time
import logging
import argparse
import os
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

FEATURES_DB  = "features.db"
OUTPUT_CSV   = "llm_labelling_results.csv"
LLM_MODEL    = "claude-haiku-4-5-20251001"
LLM_DELAY    = 0.3   # seconds between API calls


# ── Helpers ───────────────────────────────────────────────────

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def rating_to_signal(rating: int) -> str:
    """Convert star rating to sentiment signal."""
    if rating <= 2:
        return "negative"
    if rating >= 4:
        return "positive"
    return "neutral"


def majority_vote(signal_a: str, signal_b: str, signal_c: str) -> tuple[str, float]:
    """
    Three-way majority vote across rating, VADER, and LLM signals.
    Returns (final_label, confidence).
    confidence = 1.0 if all agree, 0.67 if 2/3 agree, 0.0 if all disagree.
    """
    votes = [signal_a, signal_b, signal_c]
    valid = [v for v in votes if v is not None]

    if not valid:
        return "ambiguous", 0.0

    from collections import Counter
    counts = Counter(valid)
    top_label, top_count = counts.most_common(1)[0]

    if top_count >= 2:
        confidence = round(top_count / len(valid), 2)
        return top_label, confidence
    else:
        return "ambiguous", 0.0


# ── LLM call ──────────────────────────────────────────────────

def call_llm(text: str, client) -> tuple[str, float]:
    """
    Asks Claude to classify the sentiment of a single review.
    Returns (label, confidence).
    """
    if not text or not text.strip():
        return None, None

    prompt = f"""Classify the sentiment of this app store review.

Review: "{text[:500]}"

Respond with JSON only, no other text:
{{"label": "positive" | "negative" | "neutral", "confidence": 0.0-1.0}}"""

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # strip markdown code block if present (e.g. ```json ... ```)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        result = json.loads(raw)
        return result.get("label"), result.get("confidence")
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return None, None


# ── Data loading ──────────────────────────────────────────────

def load_ambiguous_reviews(db_path: str) -> pd.DataFrame:
    """
    Loads two types of ambiguous reviews:
    1. Rating-text disagreements: where vader_label != combined_label
    2. 3-star reviews: no strong anchor signal from either source
    """
    conn = sqlite3.connect(db_path)

    df = pd.read_sql("""
        SELECT review_id, app_name, rating, text,
               vader_label, vader_compound,
               combined_label, combined_score,
               llm_sentiment_label
        FROM features
        WHERE text IS NOT NULL
          AND (
              -- category 1: rating and text signal conflict
              vader_label != combined_label
              OR
              -- category 2: 3-star reviews with no reliable anchor
              rating = 3
          )
        ORDER BY
            -- prioritise true conflicts first: where vader and combined disagree
            CASE WHEN vader_label != combined_label THEN 0 ELSE 1 END,
            -- within conflicts, prioritise by how extreme the disagreement is
            ABS(vader_compound) DESC
    """, conn)
    conn.close()

    # remove duplicates (a review can fall in both categories)
    df = df.drop_duplicates(subset=['review_id'])
    log.info(f"Found {len(df):,} ambiguous reviews to label")

    # breakdown
    disagreements = df[df['vader_label'] != df['combined_label']]
    three_star    = df[df['rating'] == 3]
    log.info(f"  Rating-text disagreements : {len(disagreements):,}")
    log.info(f"  3-star reviews            : {len(three_star):,}")

    return df


# ── Schema update ─────────────────────────────────────────────

def ensure_columns(conn: sqlite3.Connection):
    """Add final_label and label_confidence columns if they don't exist."""
    existing = [row[1] for row in conn.execute("PRAGMA table_info(features)").fetchall()]

    for col, col_type in [
        ("llm_sentiment_label",      "TEXT"),
        ("llm_sentiment_confidence", "REAL"),
        ("final_label",              "TEXT"),
        ("label_confidence",         "REAL"),
        ("label_source",             "TEXT"),
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE features ADD COLUMN {col} {col_type}")
            log.info(f"Added column: {col}")

    conn.commit()


# ── Main labelling loop ────────────────────────────────────────

def label_reviews(df: pd.DataFrame, client, dry_run: bool = False) -> pd.DataFrame:
    """
    For each ambiguous review:
    1. Get LLM sentiment label
    2. Run three-way majority vote: rating_signal + vader_label + llm_label
    3. Store final_label and confidence
    """
    results = []
    total   = len(df)

    for i, row in df.iterrows():
        if i % 50 == 0:
            log.info(f"  Progress: {results.__len__()} / {total}")

        text          = row['text']
        rating        = int(row['rating'])
        vader_label   = row['vader_label']
        rating_signal = rating_to_signal(rating)

        # get LLM label (skip if dry run or already labelled)
        if dry_run:
            llm_label      = "positive"   # placeholder
            llm_confidence = 0.9
        elif pd.notna(row.get('llm_sentiment_label')):
            # already has LLM label from previous run — reuse it
            llm_label      = row['llm_sentiment_label']
            llm_confidence = None
        else:
            llm_label, llm_confidence = call_llm(text, client)
            time.sleep(LLM_DELAY)

        # three-way majority vote
        final_label, label_confidence = majority_vote(
            rating_signal, vader_label, llm_label
        )

        # track which signals agreed for auditability
        agreed = [s for s in [rating_signal, vader_label, llm_label]
                  if s == final_label]
        label_source = "+".join(sorted(set(agreed))) if final_label != "ambiguous" else "none"

        results.append({
            "review_id"             : row['review_id'],
            "app_name"              : row['app_name'],
            "rating"                : rating,
            "text_preview"          : str(text)[:100],
            "rating_signal"         : rating_signal,
            "vader_label"           : vader_label,
            "llm_label"             : llm_label,
            "llm_confidence"        : llm_confidence,
            "final_label"           : final_label,
            "label_confidence"      : label_confidence,
            "label_source"          : label_source,
        })

    return pd.DataFrame(results)


# ── Write back to DB ──────────────────────────────────────────

def write_results(results: pd.DataFrame, db_path: str):
    """Updates features.db with final labels and LLM outputs."""
    conn = sqlite3.connect(db_path)
    ensure_columns(conn)

    updated = 0
    for _, row in results.iterrows():
        conn.execute("""
            UPDATE features
            SET llm_sentiment_label      = ?,
                llm_sentiment_confidence = ?,
                final_label              = ?,
                label_confidence         = ?,
                label_source             = ?
            WHERE review_id = ?
        """, (
            row['llm_label'],
            row['llm_confidence'],
            row['final_label'],
            row['label_confidence'],
            row['label_source'],
            row['review_id'],
        ))
        updated += 1

    conn.commit()
    conn.close()
    log.info(f"Updated {updated:,} rows in {db_path}")


# ── Summary ───────────────────────────────────────────────────

def print_summary(results: pd.DataFrame):
    log.info("=" * 50)
    log.info("LABELLING SUMMARY")
    log.info("=" * 50)

    total = len(results)
    log.info(f"Total reviews labelled : {total:,}")

    log.info("\nFinal label distribution:")
    for label, count in results['final_label'].value_counts().items():
        log.info(f"  {label:<15} {count:,} ({count/total*100:.1f}%)")

    log.info("\nLabel source breakdown:")
    for source, count in results['label_source'].value_counts().items():
        log.info(f"  {source:<30} {count:,}")

    # show interesting cases where all three disagreed
    ambiguous = results[results['final_label'] == 'ambiguous']
    if len(ambiguous) > 0:
        log.info(f"\nAmbiguous (all 3 signals disagree): {len(ambiguous):,}")
        log.info("Sample cases:")
        for _, row in ambiguous.head(3).iterrows():
            log.info(f"  [{row['app_name']}] rating={row['rating']}★ "
                     f"vader={row['vader_label']} llm={row['llm_label']}")
            log.info(f"  text: {row['text_preview']}")

    log.info("=" * 50)


# ── Main ──────────────────────────────────────────────────────

def main(limit: int, dry_run: bool):
    if not os.path.exists(FEATURES_DB):
        raise FileNotFoundError(f"{FEATURES_DB} not found. Run feature_pipeline.py first.")

    log.info("LLM labelling started.")
    if dry_run:
        log.info("DRY RUN mode — no API calls will be made.")

    # load ambiguous reviews
    df = load_ambiguous_reviews(FEATURES_DB)
    if limit:
        df = df.head(limit)
        log.info(f"Limited to {limit} reviews for testing.")

    # set up Claude client
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not dry_run:
        raise ValueError("ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=your_key")
    client = anthropic.Anthropic(api_key=api_key) if not dry_run else None

    # run labelling
    log.info("Running LLM labelling + three-way voting...")
    results = label_reviews(df, client, dry_run=dry_run)

    # save results
    results.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Results saved → {OUTPUT_CSV}")

    if not dry_run:
        write_results(results, FEATURES_DB)

    print_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM labelling for ambiguous reviews.")
    parser.add_argument("--limit",   type=int, default=None,  help="Process only first N reviews")
    parser.add_argument("--dry-run", action="store_true",     help="Preview without API calls")
    args = parser.parse_args()
    main(args.limit, args.dry_run)
