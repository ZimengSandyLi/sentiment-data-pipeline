"""
Feature Engineering Pipeline
==============================
Transforms raw app review text into structured, model-ready features.

For each review, this pipeline extracts:
  - Sentiment polarity   (VADER vs LLM zero-shot)
  - Subjectivity score   (TextBlob vs LLM scoring)
  - Aspects              (spaCy noun chunks vs LLM extraction)
  - Semantic embeddings  (TF-IDF vs sentence-transformers)

The goal is to compare traditional NLP methods against LLM-based approaches,
and understand which produces more useful signal for downstream ML tasks.

Usage:
    python feature_pipeline.py                  # process all reviews in DB
    python feature_pipeline.py --limit 500      # process first N reviews
    python feature_pipeline.py --skip-llm       # skip LLM modules (no API cost)

Output:
    features.csv   — one row per review with all extracted features
    features.db    — same data in SQLite for easy querying

Dependencies:
    pip install vaderSentiment textblob sentence-transformers anthropic scikit-learn umap-learn
    python -m spacy download en_core_web_sm
    export ANTHROPIC_API_KEY=your_key
"""

import sqlite3
import os
import csv
import json
import time
import logging
import argparse
from datetime import datetime, timezone

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline as SklearnPipeline
import anthropic

# ── Configuration ─────────────────────────────────────────────

PIPELINE_DB   = "pipeline.db"      # source database from Phase I
FEATURES_DB   = "features.db"      # output database for engineered features
FEATURES_CSV  = "features.csv"     # output CSV for easy inspection

# sentence-transformers model — lightweight and good quality, runs locally
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM model for aspect extraction and sentiment scoring
LLM_MODEL = "claude-haiku-4-5-20251001"  # fast and cheap, good for structured extraction

# how long to wait between LLM API calls to avoid rate limiting
LLM_DELAY = 0.3   # seconds

LOG_LEVEL = logging.INFO

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("feature_pipeline.log"),
    ],
)
log = logging.getLogger(__name__)


# ── Setup ─────────────────────────────────────────────────────

def load_models(reviews: list[dict] = None):
    """
    Load all the NLP models we need upfront.
    Doing this once at startup is much faster than loading per review.

    Also fits the TF-IDF vectorizer on the full corpus if reviews are provided —
    TF-IDF needs to see all documents before it can weight word frequencies correctly.
    """
    log.info("Loading NLP models...")

    # spaCy — for POS tagging and noun chunk extraction
    nlp = spacy.load("en_core_web_sm")

    # VADER — rule-based sentiment analyser, good for short informal text
    vader = SentimentIntensityAnalyzer()

    # sentence-transformers — LLM-based embeddings, runs locally
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # TF-IDF + TruncatedSVD (LSA) — traditional baseline embedding method
    # optimised with four improvements over a naive TF-IDF:
    #   1. Lemmatization via spaCy — "crashes" and "crash" become the same token
    #   2. App name filtering — removes app brand names that dominate their own
    #      reviews and cause the embedding to encode app identity, not sentiment
    #   3. Char-level n-grams (2-4 chars) added alongside word n-grams — helps
    #      with spelling variants like "awsome" vs "awesome"
    #   4. 200 SVD dimensions instead of 100 — retains more semantic structure
    tfidf_vectorizer = None
    if reviews:
        log.info("Fitting optimised TF-IDF vectorizer on full corpus...")
        texts = [r["text"] for r in reviews if r.get("text")]

        # preprocessing: lemmatize and remove app brand names
        # doing this before TF-IDF so the vocabulary is cleaner
        log.info("  Lemmatizing texts for TF-IDF...")
        processed_texts = []
        app_name_tokens = {
            "spotify", "whatsapp", "instagram", "netflix", "amazon",
            "duolingo", "uber", "youtube", "teams", "twitter", "chatgpt",
            "google", "apple", "microsoft", "meta",
        }
        for text in texts:
            doc = nlp(text.lower())
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop          # remove stopwords
                and not token.is_punct        # remove punctuation
                and not token.is_space        # remove whitespace tokens
                and token.lemma_ not in app_name_tokens   # remove app names
                and len(token.lemma_) > 2     # remove very short tokens
            ]
            processed_texts.append(" ".join(tokens))
        log.info("  Lemmatization done.")

        # word-level TF-IDF with bigrams
        word_tfidf = TfidfVectorizer(
            max_features = 15000,
            ngram_range  = (1, 2),    # unigrams + bigrams
            min_df       = 3,
            sublinear_tf = True,      # log-scale TF to reduce impact of very frequent words
            analyzer     = "word",
        )

        # char-level TF-IDF to catch spelling variants and word fragments
        char_tfidf = TfidfVectorizer(
            max_features = 5000,
            ngram_range  = (2, 4),    # 2-4 character n-grams
            min_df       = 5,
            sublinear_tf = True,
            analyzer     = "char_wb", # char_wb respects word boundaries
        )

        # fit both vectorizers and concatenate their outputs before SVD
        from scipy.sparse import hstack as sparse_hstack
        word_matrix = word_tfidf.fit_transform(processed_texts)
        char_matrix = char_tfidf.fit_transform(processed_texts)
        combined    = sparse_hstack([word_matrix, char_matrix])

        # SVD to compress to 200 dense dimensions (up from 100)
        svd = TruncatedSVD(n_components=200, random_state=42)
        svd.fit(combined)

        # store everything needed to transform new texts later
        tfidf_vectorizer = {
            "word_tfidf"       : word_tfidf,
            "char_tfidf"       : char_tfidf,
            "svd"              : svd,
            "app_name_tokens"  : app_name_tokens,
            "nlp"              : nlp,
        }
        log.info("TF-IDF vectorizer fitted.")

    # Anthropic client — for LLM-based aspect extraction and sentiment scoring
    # will be None if API key is not set
    try:
        llm_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        log.info("Anthropic client ready.")
    except Exception as e:
        log.warning(f"Anthropic client not available: {e}. LLM features will be skipped.")
        llm_client = None

    log.info("All models loaded.")
    return nlp, vader, embedder, tfidf_vectorizer, llm_client


# ── Module 1: Sentiment Polarity ──────────────────────────────

def sentiment_vader(text: str, vader, rating: int = None) -> dict:
    """
    Traditional method: VADER sentiment analysis, with three improvements:

    1. Rating-assisted labeling — star rating is the most reliable sentiment
       signal we have. For 1-2 and 4-5 star reviews we use the rating directly,
       and only fall back to VADER text analysis for ambiguous 3-star reviews.

    2. Adjusted thresholds — widened from +/-0.05 to +/-0.1 to reduce the number
       of borderline reviews being misclassified as neutral.

    3. Text truncation — VADER performs better on short text. We only feed it
       the first 150 characters, which avoids long reviews where sentiment
       words get diluted across many sentences.
    """
    if not text or not text.strip():
        return {"vader_compound": None, "vader_label": None, "vader_method": None}

    # truncate to first 150 chars — VADER is most reliable on short text
    truncated = text[:150]
    scores = vader.polarity_scores(truncated)
    compound = scores["compound"]

    # use rating as primary signal where it is unambiguous
    if rating is not None:
        if rating <= 2:
            label = "negative"
            method = "rating"
        elif rating >= 4:
            label = "positive"
            method = "rating"
        else:
            # 3-star reviews are genuinely ambiguous — use VADER text score
            if compound >= 0.1:
                label = "positive"
            elif compound <= -0.1:
                label = "negative"
            else:
                label = "neutral"
            method = "vader_text"
    else:
        # no rating available — fall back to VADER with widened thresholds
        if compound >= 0.1:
            label = "positive"
        elif compound <= -0.1:
            label = "negative"
        else:
            label = "neutral"
        method = "vader_text"

    return {
        "vader_compound": round(compound, 4),
        "vader_label": label,
        "vader_method": method,   # tracks whether label came from rating or text
    }


def combined_sentiment(rating: int, vader_compound: float, text_len: int) -> dict:
    """
    Combines star rating and VADER text score into a single sentiment label.

    The key insight is that neither signal is perfect on its own:
    - Rating alone misses cases where the user's text contradicts their stars
      (e.g. "good app, low stars to get dev attention" with rating=2)
    - VADER alone struggles with long reviews and complex language

    So we blend them with weights that depend on text length:
    - Short reviews (<20 chars): rating gets 90% weight — text has almost no signal
    - Medium reviews (<100 chars): rating gets 70%, VADER gets 30%
    - Long reviews (100+ chars): equal 50/50 — VADER is more reliable on long text

    The rating is normalised from 1-5 stars to a -1 to +1 scale to match
    VADER's compound score range before blending.
    """
    # normalise rating from 1-5 scale to -1 to +1
    # e.g. 1 star → -1.0, 3 stars → 0.0, 5 stars → +1.0
    rating_signal = (rating - 3) / 2

    # set blend weights based on how much text there is to analyse
    if text_len < 20:
        weight_rating = 0.9
        weight_vader  = 0.1
    elif text_len < 100:
        weight_rating = 0.7
        weight_vader  = 0.3
    else:
        weight_rating = 0.5
        weight_vader  = 0.5

    vader_score = vader_compound if vader_compound is not None else 0.0
    combined_score = weight_rating * rating_signal + weight_vader * vader_score

    # classify using same ±0.1 threshold as the improved VADER
    if combined_score >= 0.1:
        label = "positive"
    elif combined_score <= -0.1:
        label = "negative"
    else:
        label = "neutral"

    return {
        "combined_label"       : label,
        "combined_score"       : round(combined_score, 4),
        "combined_weight_rating": weight_rating,   # logged so we can audit the decision
    }


def sentiment_llm(text: str, client) -> dict:
    """
    LLM-based method: ask Claude to classify sentiment.
    Returns a label and a confidence score (0-1).
    More accurate than VADER on nuanced text, but slower and has API cost.
    """
    if not text or not text.strip() or client is None:
        return {"llm_sentiment_label": None, "llm_sentiment_confidence": None}

    prompt = f"""Classify the sentiment of this app review.

Review: "{text}"

Respond with JSON only, no other text:
{{"label": "positive" | "negative" | "neutral", "confidence": 0.0-1.0}}"""

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        result = json.loads(raw)
        return {
            "llm_sentiment_label": result.get("label"),
            "llm_sentiment_confidence": result.get("confidence"),
        }
    except Exception as e:
        log.warning(f"LLM sentiment failed: {e}")
        return {"llm_sentiment_label": None, "llm_sentiment_confidence": None}


# ── Module 2: Subjectivity ────────────────────────────────────

def subjectivity_textblob(text: str) -> dict:
    """
    Traditional method: TextBlob subjectivity scoring.
    Returns a score from 0 (fully objective) to 1 (fully subjective).
    Example: "the app crashes on login" is objective; "I hate this app" is subjective.
    """
    if not text or not text.strip():
        return {"textblob_subjectivity": None, "textblob_polarity": None}

    blob = TextBlob(text)
    return {
        "textblob_subjectivity": round(blob.sentiment.subjectivity, 4),
        "textblob_polarity": round(blob.sentiment.polarity, 4),
    }


def subjectivity_llm(text: str, client) -> dict:
    """
    LLM-based method: ask Claude to score subjectivity.
    More accurate than TextBlob on edge cases, especially mixed reviews.
    """
    if not text or not text.strip() or client is None:
        return {"llm_subjectivity": None}

    prompt = f"""Rate how subjective this app review is.

Review: "{text}"

Subjectivity scale: 0.0 = purely factual/objective, 1.0 = purely opinion/emotional.

Respond with JSON only:
{{"subjectivity": 0.0-1.0, "reasoning": "one sentence"}}"""

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        result = json.loads(raw)
        return {
            "llm_subjectivity": result.get("subjectivity"),
            "llm_subjectivity_reasoning": result.get("reasoning"),
        }
    except Exception as e:
        log.warning(f"LLM subjectivity failed: {e}")
        return {"llm_subjectivity": None, "llm_subjectivity_reasoning": None}


# ── Module 3: Aspect Extraction ───────────────────────────────

# words that look like noun chunks but carry no useful aspect signal
ASPECT_BLACKLIST = {
    # pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
    # generic filler words that spaCy picks up as noun chunks
    "everything", "something", "anything", "nothing", "someone", "anyone",
    "everyone", "nobody", "somebody", "whoever", "whatever",
    # time/quantity words that show up as chunks but aren't aspects
    "time", "way", "lot", "bit", "all", "one", "two", "more",
    # too generic to be a useful aspect
    "people", "money", "a lot", "english", "reason", "stuff", "thing", "things",
    "issue", "issues", "problem", "problems", "experience", "feature", "features",
}

# normalise these surface forms to a single canonical aspect
# e.g. "this app", "the app", "your app" all mean the same thing
ASPECT_NORMALISATION = {
    "this app"  : "app",
    "the app"   : "app",
    "your app"  : "app",
    "my app"    : "app",
    "an app"    : "app",
    "this application": "app",
    "the application" : "app",
    # plural → singular for cleaner aggregation
    "videos"    : "video",
    "messages"  : "message",
    "updates"   : "update",
    "ads"       : "ad",
    "crashes"   : "crash",
    "bugs"      : "bug",
    "reviews"   : "review",
    "features"  : "feature",
    "accounts"  : "account",
    "payments"  : "payment",
    "notifications" : "notification",
    "subscriptions" : "subscription",
}

# app-domain vocabulary — these terms are almost always meaningful aspects
# having this list helps us keep them even if they're single short words
APP_DOMAIN_TERMS = {
    "ui", "ux", "app", "bug", "ads", "api", "otp", "pin",
    "login", "signup", "logout", "password", "account", "profile",
    "notification", "notifications", "update", "updates", "crash", "crashes",
    "battery", "storage", "memory", "speed", "performance", "loading",
    "camera", "audio", "video", "screen", "button", "menu", "settings",
    "search", "filter", "feed", "chat", "message", "call", "payment",
    "subscription", "premium", "customer service", "support", "refund",
}


def aspects_spacy(text: str, nlp) -> dict:
    """
    Traditional method: spaCy noun chunk extraction, with three improvements:

    1. Blacklist filter — removes pronouns, generic words, and determiners
       that spaCy picks up as noun chunks but carry no aspect signal.

    2. Domain vocabulary boost — app-specific terms like "ui", "crash", "login"
       are kept even if they are short, since they are always meaningful.

    3. Noun-only filter — only keeps chunks where the head word is a noun
       (POS tag NN or NNS), which eliminates most of the noise.
    """
    if not text or not text.strip():
        return {"spacy_aspects": None, "spacy_aspect_count": 0}

    doc = nlp(text)
    aspects = set()

    for chunk in doc.noun_chunks:
        clean = chunk.text.lower().strip()

        # skip if it's in the blacklist
        if clean in ASPECT_BLACKLIST:
            continue

        # skip very short chunks unless they're in our domain vocabulary
        if len(clean) < 4 and clean not in APP_DOMAIN_TERMS:
            continue

        # only keep chunks where the head word is a noun
        if chunk.root.pos_ not in ("NOUN", "PROPN"):
            continue

        # if it's a multi-word chunk, also check it's not just a pronoun + noun
        # e.g. "my money" — strip possessive pronouns from the front
        tokens = clean.split()
        if tokens[0] in {"my", "your", "his", "her", "its", "our", "their"}:
            clean = " ".join(tokens[1:])

        if clean and len(clean) >= 3:
            aspects.add(clean)

    # also check against domain vocabulary directly — catch single-word aspects
    # that might have been filtered out by the noun chunk logic
    words = text.lower().split()
    for word in words:
        if word in APP_DOMAIN_TERMS:
            aspects.add(word)

    # apply normalisation — merge variant forms into canonical terms
    normalised = set()
    for a in aspects:
        normalised.add(ASPECT_NORMALISATION.get(a, a))

    # remove anything that ended up in the blacklist after normalisation
    normalised = {a for a in normalised if a not in ASPECT_BLACKLIST}

    aspects_list = sorted(normalised)
    return {
        "spacy_aspects": json.dumps(aspects_list),
        "spacy_aspect_count": len(aspects_list),
    }


def aspects_llm(text: str, client) -> dict:
    """
    LLM-based method: ask Claude to extract aspects and their sentiment.
    Can identify implicit aspects that spaCy misses, e.g. "it takes forever"
    maps to the aspect "performance" even though the word doesn't appear.
    """
    if not text or not text.strip() or client is None:
        return {"llm_aspects": None, "llm_aspect_count": 0}

    prompt = f"""Extract the app features or aspects being discussed in this review,
and the sentiment toward each one.

Review: "{text}"

Respond with JSON only:
{{"aspects": [{{"aspect": "feature name", "sentiment": "positive"|"negative"|"neutral"}}]}}

If no specific aspects are mentioned, return {{"aspects": []}}"""

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        result = json.loads(raw)
        aspects = result.get("aspects", [])
        return {
            "llm_aspects": json.dumps(aspects),
            "llm_aspect_count": len(aspects),
        }
    except Exception as e:
        log.warning(f"LLM aspect extraction failed: {e}")
        return {"llm_aspects": None, "llm_aspect_count": 0}


# ── Module 4: Embeddings ──────────────────────────────────────

def embed_sentence_transformer(texts: list[str], embedder) -> list:
    """
    LLM-based method: sentence-transformers embeddings.
    Converts text into dense vectors where semantically similar reviews
    are close together in vector space.
    "laggy", "slow", and "freezing" will have similar vectors.
    Runs locally — no API cost.
    Processes in batches for efficiency.
    """
    # replace empty/None with a placeholder so the batch size stays consistent
    cleaned = [t if t and t.strip() else "[empty]" for t in texts]
    embeddings = embedder.encode(cleaned, show_progress_bar=False)
    return embeddings.tolist()


def embed_tfidf(texts: list[str], vectorizer) -> list:
    """
    Optimised TF-IDF + TruncatedSVD (LSA) embeddings.

    Applies the same preprocessing used during fitting:
    lemmatization, app name removal, then word + char TF-IDF
    concatenated and compressed via SVD to 200 dimensions.

    Key limitation vs sentence-transformers (even after optimisation):
    - "slow" and "laggy" are still unrelated tokens at the word level
      (char n-grams help a little but don't bridge this gap fully)
    - no understanding of negation: "not good" and "good" are similar
    - but it's fully transparent — every dimension traces back to real words
    """
    from scipy.sparse import hstack as sparse_hstack

    nlp_model      = vectorizer["nlp"]
    word_tfidf     = vectorizer["word_tfidf"]
    char_tfidf     = vectorizer["char_tfidf"]
    svd            = vectorizer["svd"]
    app_name_tokens= vectorizer["app_name_tokens"]

    # apply the same lemmatization preprocessing as during fitting
    processed = []
    for text in texts:
        t = text if text and text.strip() else "empty"
        doc = nlp_model(t.lower())
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.lemma_ not in app_name_tokens
            and len(token.lemma_) > 2
        ]
        processed.append(" ".join(tokens) if tokens else "empty")

    word_matrix = word_tfidf.transform(processed)
    char_matrix = char_tfidf.transform(processed)
    combined    = sparse_hstack([word_matrix, char_matrix])
    vectors     = svd.transform(combined)
    return vectors.tolist()


# ── Database output ───────────────────────────────────────────

def init_features_db(conn: sqlite3.Connection):
    """
    Creates the features table in features.db.
    One row per review, with columns for every feature we extract.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS features (
            review_id                   TEXT PRIMARY KEY,
            app_id                      TEXT,
            app_name                    TEXT,
            rating                      INTEGER,
            text                        TEXT,

            -- Module 1: Sentiment Polarity
            vader_compound              REAL,
            vader_label                 TEXT,
            vader_method                TEXT,   -- "rating" or "vader_text"
            llm_sentiment_label         TEXT,
            llm_sentiment_confidence    REAL,

            -- Module 2: Subjectivity
            textblob_subjectivity       REAL,
            textblob_polarity           REAL,
            llm_subjectivity            REAL,
            llm_subjectivity_reasoning  TEXT,

            -- Module 3: Aspects
            spacy_aspects               TEXT,   -- JSON array
            spacy_aspect_count          INTEGER,
            llm_aspects                 TEXT,   -- JSON array
            llm_aspect_count            INTEGER,

            -- Combined sentiment (rating + VADER blend)
            combined_label              TEXT,    -- final blended label
            combined_score              REAL,    -- blended score -1 to +1
            combined_weight_rating      REAL,    -- how much weight rating got (0.5-0.9)

            -- Module 4: Embeddings
            -- sentence-transformer: 384 dimensions (all-MiniLM-L6-v2)
            -- tfidf: 100 dimensions (TF-IDF + TruncatedSVD/LSA)
            embedding                   TEXT,
            tfidf_embedding             TEXT,

            processed_at                TIMESTAMP
        );
    """)
    conn.commit()


# ── Main pipeline ─────────────────────────────────────────────

def load_reviews(db_path: str, limit: int = None) -> list[dict]:
    """Load reviews from the Phase I pipeline database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT r.review_id, r.app_id, a.app_name, r.rating, r.text
        FROM reviews r
        JOIN apps a ON r.app_id = a.app_id
        WHERE r.text IS NOT NULL AND LENGTH(r.text) >= 10
    """
    if limit:
        query += f" LIMIT {limit}"

    rows = conn.execute(query).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def process_reviews(
    reviews: list[dict],
    nlp, vader, embedder, tfidf_vectorizer, llm_client,
    skip_llm: bool = False,
) -> list[dict]:
    """
    Runs all four feature modules on each review.
    LLM modules can be skipped with --skip-llm to avoid API cost during testing.
    Both embedding methods are computed in batch upfront for efficiency.
    """
    results = []
    total = len(reviews)

    log.info(f"Processing {total:,} reviews...")

    texts = [r["text"] for r in reviews]

    # compute sentence-transformer embeddings in one batch
    log.info("Computing sentence-transformer embeddings (batch)...")
    st_embeddings = embed_sentence_transformer(texts, embedder)
    log.info("Sentence-transformer embeddings done.")

    # compute TF-IDF embeddings in one batch
    tfidf_embeddings = None
    if tfidf_vectorizer is not None:
        log.info("Computing TF-IDF embeddings (batch)...")
        tfidf_embeddings = embed_tfidf(texts, tfidf_vectorizer)
        log.info("TF-IDF embeddings done.")

    for i, review in enumerate(reviews):
        if i % 100 == 0:
            log.info(f"  Progress: {i:,} / {total:,}")

        text = review["text"]
        features = {
            "review_id" : review["review_id"],
            "app_id"    : review["app_id"],
            "app_name"  : review["app_name"],
            "rating"    : review["rating"],
            "text"      : text,
        }

        # Module 1: Sentiment Polarity
        vader_result = sentiment_vader(text, vader, rating=review["rating"])
        features.update(vader_result)

        # Combined sentiment — blends rating signal and VADER compound score
        features.update(combined_sentiment(
            rating        = review["rating"],
            vader_compound= vader_result["vader_compound"],
            text_len      = len(text),
        ))

        if not skip_llm:
            features.update(sentiment_llm(text, llm_client))
            time.sleep(LLM_DELAY)

        # Module 2: Subjectivity
        features.update(subjectivity_textblob(text))
        if not skip_llm:
            features.update(subjectivity_llm(text, llm_client))
            time.sleep(LLM_DELAY)

        # Module 3: Aspect Extraction
        features.update(aspects_spacy(text, nlp))
        if not skip_llm:
            features.update(aspects_llm(text, llm_client))
            time.sleep(LLM_DELAY)

        # Module 4: Embeddings (both computed in batch above)
        features["embedding"] = json.dumps(st_embeddings[i])
        if tfidf_embeddings is not None:
            features["tfidf_embedding"] = json.dumps(
                [round(v, 6) for v in tfidf_embeddings[i]]
            )

        features["processed_at"] = datetime.now(timezone.utc).isoformat()
        results.append(features)

    return results


def save_results(results: list[dict], features_db_path: str, csv_path: str):
    """Saves extracted features to both SQLite and CSV."""

    # save to SQLite
    conn = sqlite3.connect(features_db_path)
    init_features_db(conn)

    for r in results:
        cols = ", ".join(r.keys())
        placeholders = ", ".join(["?"] * len(r))
        conn.execute(
            f"INSERT OR REPLACE INTO features ({cols}) VALUES ({placeholders})",
            list(r.values()),
        )
    conn.commit()
    conn.close()
    log.info(f"Saved {len(results):,} rows to {features_db_path}")

    # save to CSV — easier to inspect in Excel or pandas
    if results:
        # don't write the full embedding vector to CSV — it's 384 numbers per row
        # and makes the file huge and unreadable. write the first 5 dims as a preview.
        csv_results = []
        for r in results:
            # exclude full embedding vectors from CSV — too large to be readable
            row = {k: v for k, v in r.items()
                   if k not in ("embedding", "tfidf_embedding")}
            try:
                emb = json.loads(r.get("embedding", "[]"))
                row["embedding_preview"] = str(emb[:5]) + "..."
            except Exception:
                row["embedding_preview"] = None
            try:
                tfidf = json.loads(r.get("tfidf_embedding", "[]"))
                row["tfidf_embedding_preview"] = str(tfidf[:5]) + "..."
            except Exception:
                row["tfidf_embedding_preview"] = None
            csv_results.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
            writer.writeheader()
            writer.writerows(csv_results)
        log.info(f"Saved CSV → {csv_path}")


# ── Entry point ───────────────────────────────────────────────

def main(limit: int, skip_llm: bool):
    log.info("Feature engineering pipeline started.")

    # load source data from Phase I database
    if not os.path.exists(PIPELINE_DB):
        raise FileNotFoundError(
            f"{PIPELINE_DB} not found. Make sure you're in the right folder."
        )

    reviews = load_reviews(PIPELINE_DB, limit=limit)
    log.info(f"Loaded {len(reviews):,} reviews from {PIPELINE_DB}")

    if skip_llm:
        log.info("--skip-llm flag set: LLM modules will be skipped.")

    # load models — pass reviews so TF-IDF can be fitted on the full corpus
    nlp, vader, embedder, tfidf_vectorizer, llm_client = load_models(reviews)

    # run pipeline
    start = time.time()
    results = process_reviews(reviews, nlp, vader, embedder, tfidf_vectorizer, llm_client, skip_llm)
    elapsed = round(time.time() - start, 1)

    # save output
    save_results(results, FEATURES_DB, FEATURES_CSV)

    log.info(f"Done in {elapsed}s — {len(results):,} reviews processed.")
    log.info(f"Output: {FEATURES_DB}, {FEATURES_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering pipeline.")
    parser.add_argument("--limit",    type=int, default=None, help="Process only first N reviews")
    parser.add_argument("--skip-llm", action="store_true",    help="Skip LLM modules (no API cost)")
    args = parser.parse_args()
    main(args.limit, args.skip_llm)
