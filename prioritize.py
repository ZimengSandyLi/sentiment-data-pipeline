"""
Issue Prioritization Framework
================================
Scores and ranks product issues by combining three signals:
  1. Sentiment severity   — how negative are the reviews mentioning this issue
  2. Review volume        — how many users are talking about it
  3. Recency              — are complaints getting worse or better lately

Output is a ranked backlog of product issues that helps the team
focus on the most business-critical problems first.

Usage:
    python prioritize.py                    # rank all aspects
    python prioritize.py --app "WhatsApp"   # filter by app
    python prioritize.py --top 20           # show top N issues
    python prioritize.py --output issues.csv

How scoring works:
    priority_score = (
        w_sentiment * sentiment_score   +   # how negative
        w_volume    * volume_score      +   # how many reviews
        w_recency   * recency_score         # trending up recently?
    )
    Each component is normalised 0-1 before weighting.
"""

import sqlite3
import pandas as pd
import numpy as np
import argparse
import json
import os
import logging
import spacy
from datetime import datetime, timedelta, timezone

# load spaCy model for named entity recognition
# used to filter out person names, locations, and organisations
# from the aspect list — these are campaign-style noise, not product features
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    NLP = None
    print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PIPELINE_DB = "pipeline.db"
FEATURES_DB = "features.db"

# ── Scoring weights ────────────────────────────────────────────
# Adjust these to change what the team cares about most.
# Must sum to 1.0.
W_SENTIMENT = 0.40   # how severe / negative the feedback is
W_VOLUME    = 0.35   # how many users are affected
W_RECENCY   = 0.25   # is this getting worse recently?

# Recency window: reviews in the last N days count as "recent"
RECENCY_DAYS = 30


# ── Data loading ───────────────────────────────────────────────

def load_data(app_filter: str = None) -> pd.DataFrame:
    """
    Joins features.db (sentiment, aspects) with pipeline.db (dates).
    Returns one row per review with all the fields we need for scoring.
    """
    if not os.path.exists(FEATURES_DB):
        raise FileNotFoundError("features.db not found. Run feature_pipeline.py first.")
    if not os.path.exists(PIPELINE_DB):
        raise FileNotFoundError("pipeline.db not found. Run pipeline.py first.")

    # load features
    conn_f = sqlite3.connect(FEATURES_DB)
    feat = pd.read_sql("""
        SELECT review_id, app_name, rating,
               combined_label, combined_score,
               spacy_aspects, spacy_aspect_count
        FROM features
        WHERE spacy_aspects IS NOT NULL
    """, conn_f)
    conn_f.close()

    # load dates from pipeline.db
    conn_p = sqlite3.connect(PIPELINE_DB)
    dates = pd.read_sql("SELECT review_id, date FROM reviews", conn_p)
    conn_p.close()

    df = feat.merge(dates, on='review_id', how='left')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if app_filter:
        df = df[df['app_name'] == app_filter]
        log.info(f"Filtered to app: {app_filter} ({len(df):,} reviews)")
    else:
        log.info(f"Loaded {len(df):,} reviews across {df['app_name'].nunique()} apps")

    return df


# ── Aspect expansion ───────────────────────────────────────────

# normalise near-duplicate aspects to a canonical form
# e.g. "the search results", "search results", "the search" → "search"
# this prevents the same underlying issue from being split across multiple ranks
ASPECT_NORMALISATION = {
    "the search results" : "search",
    "the search"         : "search",
    "search results"     : "search",
    "searches"           : "search",
    "the new ui"         : "ui",
    "new ui"             : "ui",
    "this update"        : "update",
    "the latest update"  : "update",
    "latest update"      : "update",
    "no driver"          : "driver availability",
    "worst experience"   : "user experience",
    "the keyboard"       : "keyboard",
    "the app"            : None,   # too generic, drop entirely
    "this app"           : None,
    "the platform"       : None,
    "the day"            : None,   # not a product feature
    "worst app"          : None,   # sentiment expression, not a feature
    "1 star"             : None,   # rating behaviour, not a feature
    "years"              : None,   # too vague
}


def normalise_aspect(aspect: str) -> str | None:
    """
    Maps near-duplicate or overly generic aspects to a canonical form.
    Returns None if the aspect should be dropped entirely.
    """
    lower = aspect.lower().strip()
    if lower in ASPECT_NORMALISATION:
        return ASPECT_NORMALISATION[lower]
    return aspect


def expand_aspects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explodes the spacy_aspects JSON list so each row is one (review, aspect) pair.
    This lets us aggregate statistics per aspect across all reviews that mention it.
    """
    rows = []
    for _, row in df.iterrows():
        try:
            aspects = json.loads(row['spacy_aspects'])
        except Exception:
            continue
        for aspect in aspects:
            normalised = normalise_aspect(aspect)
            if normalised is None:
                continue
            rows.append({
                'review_id'     : row['review_id'],
                'app_name'      : row['app_name'],
                'rating'        : row['rating'],
                'combined_label': row['combined_label'],
                'combined_score': row['combined_score'],
                'date'          : row['date'],
                'aspect'        : normalised,
            })

    expanded = pd.DataFrame(rows)
    log.info(f"Expanded to {len(expanded):,} (review, aspect) pairs")
    return expanded


# ── Scoring ────────────────────────────────────────────────────

# entity types that indicate noise rather than product features
NOISE_ENTITY_TYPES = {"PERSON", "GPE", "LOC", "ORG", "NORP", "FAC", "EVENT"}

# keyword patterns that signal noise — either campaign terms or
# aspects too vague to be actionable product feedback
NOISE_PATTERNS = {
    # campaign/event noise
    "yogesh", "rawat", "lockup", "biased", "unfair",
    # too vague to be a useful product issue
    "this point", "the platform", "everything", "something",
    "nothing", "anything", "a lot", "a bit", "the way",
    "this app",  # normalised earlier but catch any remaining
    # rating/sentiment expressions, not product features
    "star", "stars", "rating", "review", "reviews",
    # time references
    "the day", "the week", "the month", "the year",
    "yesterday", "today", "tonight",
}

# aspects that are just the app/brand name — not actionable
# e.g. "amazon" in Amazon Shopping reviews is noise
APP_BRAND_TOKENS = {
    "amazon", "netflix", "spotify", "instagram", "whatsapp",
    "youtube", "uber", "twitter", "duolingo", "chatgpt", "teams",
}

def is_product_feature(aspect: str) -> bool:
    """
    Returns True if the aspect looks like a real product feature,
    False if it looks like a person name, place, campaign keyword,
    or other noise that shouldn't appear in a product backlog.

    Two-stage check:
      1. Keyword blocklist — fast check for known noise patterns
      2. spaCy NER — detects person names, locations, organisations
    """
    aspect_lower = aspect.lower()

    # stage 1a: exact match on app brand names
    # "amazon" in Amazon Shopping reviews adds no signal
    if aspect_lower.strip() in APP_BRAND_TOKENS:
        return False

    # stage 1b: keyword blocklist
    for pattern in NOISE_PATTERNS:
        if pattern in aspect_lower:
            return False

    # stage 2: NER — if the whole aspect is a named entity, it's noise
    if NLP is not None:
        doc = NLP(aspect)
        # if any token is a named entity of a noise type, filter it out
        for ent in doc.ents:
            if ent.label_ in NOISE_ENTITY_TYPES:
                return False
        # also check if all tokens are proper nouns (PROPN) with no common nouns
        # e.g. "Yogesh Rawat" — all PROPN, likely a person name
        tokens = [t for t in doc if not t.is_space and not t.is_punct]
        if tokens and all(t.pos_ == "PROPN" for t in tokens):
            return False

    return True


def score_aspects(expanded: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per-aspect statistics and computes a priority score.

    Sentiment score:
        % of reviews mentioning this aspect that are negative.
        Higher = more severe problem.

    Volume score:
        Total number of reviews mentioning this aspect.
        Higher = more users affected.

    Recency score:
        % of negative mentions that fall in the last RECENCY_DAYS.
        Higher = problem is getting worse recently (trending up).

    All three are normalised 0-1 then combined with weights.
    """
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=RECENCY_DAYS)
    # make date tz-aware for comparison
    expanded['date_aware'] = expanded['date'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')

    records = []
    for (app_name, aspect), group in expanded.groupby(['app_name', 'aspect']):
        total         = len(group)
        n_negative    = (group['combined_label'] == 'negative').sum()
        n_positive    = (group['combined_label'] == 'positive').sum()
        avg_rating    = group['rating'].mean()

        # sentiment severity: negative rate
        sentiment_raw = n_negative / total if total > 0 else 0

        # recency: among negative reviews, what % are recent?
        neg_reviews   = group[group['combined_label'] == 'negative']
        if len(neg_reviews) > 0:
            recent_neg = neg_reviews['date_aware'].gt(cutoff).sum()
            recency_raw = recent_neg / len(neg_reviews)
        else:
            recency_raw = 0

        records.append({
            'app_name'      : app_name,
            'aspect'        : aspect,
            'mention_count' : total,
            'negative_count': int(n_negative),
            'positive_count': int(n_positive),
            'negative_rate' : round(sentiment_raw, 3),
            'avg_rating'    : round(avg_rating, 2),
            'recency_score_raw': round(recency_raw, 3),
        })

    scores = pd.DataFrame(records)

    # filter out very low-volume aspects — not enough signal
    min_mentions = 10
    scores = scores[scores['mention_count'] >= min_mentions].copy()
    log.info(f"Aspects with >= {min_mentions} mentions: {len(scores):,}")

    # apply product feature filter — remove person names, locations,
    # campaign keywords and other noise that isn't a real product issue
    before = len(scores)
    scores = scores[scores['aspect'].apply(is_product_feature)].copy()
    log.info(f"After NER filter: {len(scores):,} aspects (removed {before - len(scores):,} noise aspects)")

    # normalise each raw score to 0-1
    def normalise(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    scores['sentiment_norm'] = normalise(scores['negative_rate'])
    scores['volume_norm']    = normalise(scores['mention_count'])
    scores['recency_norm']   = normalise(scores['recency_score_raw'])

    # weighted priority score
    scores['priority_score'] = (
        W_SENTIMENT * scores['sentiment_norm'] +
        W_VOLUME    * scores['volume_norm']    +
        W_RECENCY   * scores['recency_norm']
    ).round(4)

    scores = scores.sort_values('priority_score', ascending=False).reset_index(drop=True)
    scores.index += 1   # rank starts at 1
    scores.index.name = 'rank'

    return scores


# ── Output ─────────────────────────────────────────────────────

def print_summary(scores: pd.DataFrame, top_n: int):
    """Prints a human-readable ranked backlog to terminal."""
    print()
    print("=" * 70)
    print(f"  PRODUCT ISSUE PRIORITY BACKLOG  (top {top_n})")
    print(f"  Weights: sentiment {W_SENTIMENT:.0%}  |  "
          f"volume {W_VOLUME:.0%}  |  recency {W_RECENCY:.0%}")
    print("=" * 70)
    print(f"  {'Rank':<6} {'App':<22} {'Aspect':<22} "
          f"{'Score':>6} {'Neg%':>6} {'Count':>7} {'Avg★':>6}")
    print("-" * 70)

    for rank, row in scores.head(top_n).iterrows():
        print(
            f"  {rank:<6} {row['app_name']:<22} {row['aspect']:<22} "
            f"{row['priority_score']:>6.3f} "
            f"{row['negative_rate']*100:>5.1f}% "
            f"{row['mention_count']:>7,} "
            f"{row['avg_rating']:>6.2f}"
        )

    print("=" * 70)
    print()


# ── Main ───────────────────────────────────────────────────────

def main(app_filter, top_n, output_path):
    df       = load_data(app_filter)
    expanded = expand_aspects(df)
    scores   = score_aspects(expanded)

    print_summary(scores, top_n)

    if output_path:
        scores.reset_index().to_csv(output_path, index=False)
        log.info(f"Full ranked backlog saved → {output_path}")
    else:
        scores.reset_index().to_csv("issue_priority.csv", index=False)
        log.info("Full ranked backlog saved → issue_priority.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Issue prioritization framework.")
    parser.add_argument("--app",    default=None,  help="Filter to a single app name")
    parser.add_argument("--top",    default=20, type=int, help="Number of top issues to show")
    parser.add_argument("--output", default=None,  help="Path to save full CSV output")
    args = parser.parse_args()
    main(args.app, args.top, args.output)
