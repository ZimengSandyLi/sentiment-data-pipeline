"""
Fake / Low-Quality Review Classifier
======================================
Trains a binary classifier to detect low-quality reviews using weak supervision.
Labels are generated via heuristic rules (no manual annotation needed for training),
then a logistic regression model learns to generalise beyond the rules themselves.

What counts as "low quality" in this dataset:
  - Very short text (<10 chars): single words, emoji-only, punctuation
  - Duplicate text: same review appearing multiple times in the same app
  - Purely non-alphabetic: only numbers, emoji, or special characters
  - Single generic word with zero thumbs up (e.g. "good", "bad", "ok")

Why train a model on top of rules?
  The model can catch low-quality reviews that don't exactly match the rules,
  for example "gooood" or "this app" — borderline cases the rules miss.
  It also produces a continuous confidence score, not just a binary flag.

Usage:
    python fake_review_classifier.py              # train + save model
    python fake_review_classifier.py --validate   # run on validation set
    python fake_review_classifier.py --predict    # score all reviews in features.db

Output:
    fake_review_model.pkl     — trained classifier
    fake_review_scores.csv    — all reviews with quality scores
    validation_results.csv    — results on manually labeled validation set (if exists)
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import argparse
import logging
import re
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PIPELINE_DB  = "pipeline.db"
FEATURES_DB  = "features.db"
MODEL_PATH   = "fake_review_model.pkl"
SCORES_PATH  = "fake_review_scores.csv"
VALIDATION_PATH = "validation_set.csv"   # created by generate_validation_set()

# ── Heuristic label generation ─────────────────────────────────

GENERIC_WORDS = {
    "good", "bad", "ok", "okay", "great", "nice", "love", "hate",
    "best", "worst", "fine", "awesome", "terrible", "poor", "excellent",
    "amazing", "horrible", "perfect", "trash", "rubbish", "garbage",
    "wow", "yes", "no", "meh", "cool", "lol", "hmm",
}

def is_low_quality(text: str, app_name: str, thumbs_up: int,
                   dup_texts: set) -> int:
    """
    Generate a weak label for a review.
    Returns 1 (low quality) or 0 (acceptable quality).

    Rules (any one is sufficient):
      1. Text under 10 characters
      2. Exact duplicate within same app
      3. No alphabetic characters at all (pure emoji/numbers/symbols)
      4. Single generic word with zero helpful votes
    """
    if not text or not isinstance(text, str):
        return 1

    text_clean = text.strip()
    text_len   = len(text_clean)

    # Rule 1: very short
    if text_len < 10:
        return 1

    # Rule 2: duplicate within same app
    key = f"{app_name}|||{text_clean.lower()}"
    if key in dup_texts:
        return 1

    # Rule 3: no alphabetic characters
    if not re.search(r'[a-zA-Z]', text_clean):
        return 1

    # Rule 4: single generic word + no helpful votes
    words = text_clean.lower().split()
    if len(words) == 1 and words[0] in GENERIC_WORDS and thumbs_up == 0:
        return 1

    return 0


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply heuristic rules to generate weak labels for the full dataset.
    """
    log.info("Generating weak labels via heuristics...")

    # build duplicate text set: app_name + text combinations that appear > 1 time
    dup_mask = df.duplicated(subset=['app_name', 'text'], keep=False)
    dup_texts = set(
        df[dup_mask].apply(
            lambda r: f"{r['app_name']}|||{str(r['text']).strip().lower()}", axis=1
        )
    )
    log.info(f"  Found {len(dup_texts):,} duplicate text instances.")

    df = df.copy()
    df['low_quality'] = df.apply(
        lambda r: is_low_quality(
            r['text'], r['app_name'],
            r.get('thumbs_up', 0) or 0,
            dup_texts
        ), axis=1
    )

    n_lq = df['low_quality'].sum()
    log.info(f"  Labelled {n_lq:,} / {len(df):,} reviews as low quality ({n_lq/len(df)*100:.1f}%)")
    return df


# ── Feature engineering ─────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numerical features for the classifier.
    These are designed to capture patterns beyond what the hard rules check,
    so the model can generalise to borderline cases.
    """
    df = df.copy()
    text = df['text'].fillna('').astype(str)

    # basic length features
    df['f_text_len']       = text.apply(len)
    df['f_word_count']     = text.apply(lambda t: len(t.split()))
    df['f_avg_word_len']   = text.apply(
        lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0
    )
    df['f_char_per_word']  = df['f_text_len'] / (df['f_word_count'] + 1)

    # character type ratios
    df['f_alpha_ratio']    = text.apply(
        lambda t: sum(c.isalpha() for c in t) / max(len(t), 1)
    )
    df['f_digit_ratio']    = text.apply(
        lambda t: sum(c.isdigit() for c in t) / max(len(t), 1)
    )
    df['f_upper_ratio']    = text.apply(
        lambda t: sum(c.isupper() for c in t) / max(sum(c.isalpha() for c in t), 1)
    )
    df['f_punct_ratio']    = text.apply(
        lambda t: sum(not c.isalnum() and not c.isspace() for c in t) / max(len(t), 1)
    )

    # content signals
    df['f_unique_words']   = text.apply(lambda t: len(set(t.lower().split())))
    df['f_type_token_ratio'] = df['f_unique_words'] / (df['f_word_count'] + 1)
    df['f_has_generic_word'] = text.apply(
        lambda t: int(any(w in GENERIC_WORDS for w in t.lower().split()))
    )
    df['f_is_single_word'] = (df['f_word_count'] == 1).astype(int)

    # engagement signal
    df['f_thumbs_up']      = df.get('thumbs_up', pd.Series(0, index=df.index)).fillna(0).astype(int)
    df['f_has_votes']      = (df['f_thumbs_up'] > 0).astype(int)

    # sentence structure
    df['f_has_punctuation'] = text.apply(lambda t: int(bool(re.search(r'[.!?,;]', t))))
    df['f_sentence_count']  = text.apply(lambda t: max(1, len(re.split(r'[.!?]+', t))))

    return df

FEATURE_COLS = [
    'f_text_len', 'f_word_count', 'f_avg_word_len', 'f_char_per_word',
    'f_alpha_ratio', 'f_digit_ratio', 'f_upper_ratio', 'f_punct_ratio',
    'f_unique_words', 'f_type_token_ratio', 'f_has_generic_word', 'f_is_single_word',
    'f_thumbs_up', 'f_has_votes', 'f_has_punctuation', 'f_sentence_count',
]


# ── Data loading ────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load reviews with thumbs_up from pipeline.db."""
    if not os.path.exists(PIPELINE_DB):
        raise FileNotFoundError(f"{PIPELINE_DB} not found.")
    conn = sqlite3.connect(PIPELINE_DB)
    df = pd.read_sql("""
        SELECT r.review_id, a.app_name, r.rating,
               r.text, r.thumbs_up
        FROM reviews r
        JOIN apps a ON r.app_id = a.app_id
    """, conn)
    conn.close()
    log.info(f"Loaded {len(df):,} reviews from {PIPELINE_DB}")
    return df


# ── Training ────────────────────────────────────────────────────

def train(df: pd.DataFrame):
    """
    Train a logistic regression classifier on the weakly labelled data.
    Uses cross-validation to estimate generalisation performance.
    """
    log.info("Extracting features...")
    df = extract_features(df)

    X = df[FEATURE_COLS].values
    y = df['low_quality'].values

    log.info(f"Training set: {len(X):,} samples, {y.mean()*100:.1f}% low quality")

    # pipeline: scale → logistic regression
    # LogisticRegression is interpretable and fast — good for a first classifier
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(
            C=1.0,
            class_weight='balanced',   # compensate for class imbalance
            max_iter=500,
            random_state=42,
        )),
    ])

    # 5-fold cross-validation to estimate real performance
    log.info("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    log.info(f"  CV F1 scores: {cv_scores.round(3)}")
    log.info(f"  Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # train on full dataset
    model.fit(X, y)

    # save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    log.info(f"Model saved → {MODEL_PATH}")

    # feature importance (logistic regression coefficients)
    coefs = model.named_steps['clf'].coef_[0]
    importance = sorted(zip(FEATURE_COLS, coefs), key=lambda x: abs(x[1]), reverse=True)
    log.info("Top feature importances:")
    for feat, coef in importance[:8]:
        direction = "↑ low quality" if coef > 0 else "↓ high quality"
        log.info(f"  {feat:<30} {coef:+.3f}  {direction}")

    return model


# ── Prediction ──────────────────────────────────────────────────

def predict_all(df: pd.DataFrame, model) -> pd.DataFrame:
    """Score all reviews and save to CSV."""
    df = extract_features(df)
    X  = df[FEATURE_COLS].values

    df['quality_score']     = model.predict_proba(X)[:, 0]   # prob of being HIGH quality
    df['low_quality_pred']  = model.predict(X)
    df['low_quality_label'] = df['low_quality']               # heuristic label for comparison

    output = df[['review_id','app_name','rating','text','thumbs_up',
                 'low_quality_label','low_quality_pred','quality_score']].copy()
    output = output.sort_values('quality_score')   # worst quality first
    output.to_csv(SCORES_PATH, index=False)
    log.info(f"Scores saved → {SCORES_PATH}")

    n_flagged = df['low_quality_pred'].sum()
    log.info(f"Reviews flagged as low quality: {n_flagged:,} ({n_flagged/len(df)*100:.1f}%)")
    return output


# ── Validation set generation ────────────────────────────────────

def generate_validation_set(df: pd.DataFrame, n: int = 100):
    """
    Generates a CSV with n randomly sampled reviews for manual labelling.
    Stratified: 50% from predicted low-quality, 50% from predicted high-quality
    so you get a balanced view of both ends.

    Instructions:
      Open validation_set.csv and fill in the 'manual_label' column:
        1 = low quality / fake / not useful
        0 = acceptable quality
      Then run: python fake_review_classifier.py --validate
    """
    if not os.path.exists(SCORES_PATH):
        log.error("Run --predict first to generate scores.")
        return

    scores = pd.read_csv(SCORES_PATH)

    # stratified sample: half from each end of the quality spectrum
    low_q  = scores[scores['low_quality_pred'] == 1].sample(
        min(n//2, len(scores[scores['low_quality_pred']==1])), random_state=42)
    high_q = scores[scores['low_quality_pred'] == 0].sample(
        min(n//2, len(scores[scores['low_quality_pred']==0])), random_state=42)

    val_set = pd.concat([low_q, high_q]).sample(frac=1, random_state=42)
    val_set['manual_label'] = ''   # to be filled in manually

    val_set[['review_id','app_name','rating','text',
             'low_quality_pred','quality_score','manual_label']].to_csv(
        VALIDATION_PATH, index=False
    )
    log.info(f"Validation set saved → {VALIDATION_PATH}")
    log.info(f"  Open the file and fill in the 'manual_label' column:")
    log.info(f"  1 = low quality / not useful")
    log.info(f"  0 = acceptable quality")
    log.info(f"  Then run: python fake_review_classifier.py --validate")


# ── Validation evaluation ────────────────────────────────────────

def evaluate_on_validation():
    """
    Compare model predictions against manual labels in validation_set.csv.
    Produces precision, recall, and F1 against human ground truth.
    """
    if not os.path.exists(VALIDATION_PATH):
        log.error(f"{VALIDATION_PATH} not found. Run --predict first, then label the file.")
        return

    val = pd.read_csv(VALIDATION_PATH)
    val = val[val['manual_label'].notna() & (val['manual_label'] != '')]

    if len(val) == 0:
        log.error("No manual labels found. Fill in the 'manual_label' column first.")
        return

    val['manual_label']    = val['manual_label'].astype(int)
    val['low_quality_pred'] = val['low_quality_pred'].astype(int)

    log.info(f"Evaluating on {len(val)} manually labelled reviews...")
    log.info("\n" + classification_report(
        val['manual_label'], val['low_quality_pred'],
        target_names=['high quality', 'low quality']
    ))

    p = precision_score(val['manual_label'], val['low_quality_pred'])
    r = recall_score(val['manual_label'], val['low_quality_pred'])
    f = f1_score(val['manual_label'], val['low_quality_pred'])

    log.info(f"Against manual labels:")
    log.info(f"  Precision : {p:.3f}  (of reviews flagged as low quality, how many actually are?)")
    log.info(f"  Recall    : {r:.3f}  (of actual low quality reviews, how many did we catch?)")
    log.info(f"  F1        : {f:.3f}")

    # save results
    val.to_csv('validation_results.csv', index=False)
    log.info("Results saved → validation_results.csv")


# ── Main ────────────────────────────────────────────────────────

def main(args):
    df = load_data()
    df = generate_labels(df)

    if args.validate:
        evaluate_on_validation()
        return

    # train
    model = train(df)

    if args.predict or not args.validate:
        predict_all(df, model)
        generate_validation_set(df)
        log.info("")
        log.info("Next step: open validation_set.csv, fill in 'manual_label' column,")
        log.info("then run: python fake_review_classifier.py --validate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake review classifier.")
    parser.add_argument("--validate", action="store_true",
                        help="Evaluate model against manual labels in validation_set.csv")
    parser.add_argument("--predict",  action="store_true",
                        help="Score all reviews (runs automatically after training)")
    args = parser.parse_args()
    main(args)
