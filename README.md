# Review Intelligence Pipeline

A full-stack data and ML pipeline that collects Google Play app reviews, extracts NLP-based signals, and surfaces actionable product insights through an interactive dashboard and AI assistant.

**Live demo:** https://comment-sentiment-data-pipeline.streamlit.app/

---

## What it does

The pipeline automatically collects user reviews from Google Play, processes them through a feature engineering system, and delivers structured insights to a product team — without anyone manually reading thousands of reviews. It includes a RAG-based AI assistant that lets team members ask questions about the data in plain English.

---

## Architecture

```
Google Play Store
       ↓
  pipeline.py          ← scrapes reviews, loads into SQLite + Supabase
       ↓
  feature_pipeline.py  ← extracts sentiment, subjectivity, aspects, embeddings
       ↓
  prioritize.py        ← ranks product issues by severity, volume, recency
       ↓
  dashboard.py         ← Streamlit dashboard + RAG AI assistant
```

Automated daily scheduling via Airflow (`review_pipeline_dag.py`).

---

## Files

| File | Description |
|------|-------------|
| `pipeline.py` | Scrapes reviews from Google Play and loads into SQLite. Supports incremental updates. |
| `schema.sql` | Defines the three-table database schema (apps, reviews, ingestion_runs). |
| `monitor.py` | Generates a pipeline health report with data quality alerts. |
| `stress_test.py` | Tests pipeline operational limits: rate limits, volume ceiling, delay sensitivity. |
| `review_pipeline_dag.py` | Airflow DAG for daily automated scheduling. |
| `feature_pipeline.py` | NLP feature engineering: sentiment, subjectivity, aspect extraction, embeddings. |
| `visualise_embeddings.py` | UMAP comparison of TF-IDF vs sentence-transformer embeddings. |
| `fake_review_classifier.py` | Logistic regression classifier to detect low-quality reviews. |
| `prioritize.py` | Issue prioritization framework combining sentiment, volume, and recency. |
| `dashboard.py` | Streamlit dashboard with five pages and a RAG-based AI assistant. |
| `migrate_to_supabase.py` | Migrates reviews from local SQLite to Supabase PostgreSQL. |
| `migrate_features_to_supabase.py` | Migrates features and issue priority data to Supabase. |
| `requirements.txt` | Python dependencies for Streamlit Cloud deployment. |

---

## Setup & Usage

### Requirements

```bash
pip install gplay-scraper curl-cffi vaderSentiment textblob sentence-transformers \
            scikit-learn spacy streamlit plotly anthropic psycopg2-binary
python -m spacy download en_core_web_sm
```

### Run the pipeline

```bash
python pipeline.py
```

Scrapes new reviews for all apps in `TARGET_APPS` and loads them into `pipeline.db`. Safe to re-run — incremental updates only.

### Extract NLP features

```bash
python feature_pipeline.py --skip-llm
```

Processes reviews through four modules: sentiment scoring, subjectivity, aspect extraction, and embeddings. Outputs to `features.db`.

### Generate issue priority backlog

```bash
python prioritize.py --top 20
```

Ranks product issues by a weighted score combining sentiment severity (40%), review volume (35%), and recency (25%).

### Run the dashboard locally

```bash
streamlit run dashboard.py
```

### Run the health monitor

```bash
python monitor.py
```

### Automate with Airflow

```bash
cp review_pipeline_dag.py ~/airflow/dags/
airflow standalone
```

---

## Database Schema

### Local SQLite (`pipeline.db`)

Three tables: `apps`, `reviews`, `ingestion_runs`.

The `ingestion_runs` table tracks every pipeline execution — when it started, how many reviews were collected, and whether it succeeded. This provides full data lineage and makes debugging straightforward.

### Cloud PostgreSQL (Supabase)

Same schema as SQLite, plus two additional tables:

- `features` — NLP-extracted signals per review (sentiment, subjectivity, aspects)
- `issue_priority` — ranked product issue backlog

The dashboard reads from Supabase when `DATABASE_URL` is set, falling back to local SQLite for development.

---

## Feature Engineering

Four modules, each comparing a traditional NLP method against an LLM-based approach:

| Module | Traditional | LLM-based |
|--------|------------|-----------|
| Sentiment polarity | VADER + rating-weighted blending | Claude zero-shot classification |
| Subjectivity | TextBlob | Claude scoring |
| Aspect extraction | spaCy noun chunks + domain vocabulary | Claude extraction |
| Embeddings | TF-IDF + SVD (LSA) | Sentence-transformers |

UMAP visualisation confirmed that sentence-transformer embeddings produce significantly better sentiment separation than TF-IDF, even after optimisation (lemmatization, app name filtering, char-level n-grams, 200-dim SVD).

---

## Fake Review Classifier

A logistic regression classifier trained on weakly labelled data to detect low-quality reviews (spam, single-word submissions, duplicates, emoji-only).

- Labels generated automatically via heuristic rules (no manual annotation for training)
- Validated against 100 manually labelled reviews
- **Precision: 96% | Recall: 84%**

---

## Issue Prioritization

Ranks product issues using a weighted score:

```
priority_score = 0.40 x sentiment_severity
              + 0.35 x review_volume
              + 0.25 x recency_score
```

Includes a Named Entity Recognition filter (spaCy) to remove person names, campaign keywords, and other noise from the aspect list.

---

## Dashboard

Five-page Streamlit dashboard connected to Supabase:

- **Overview** — total reviews, rating distribution, sentiment trend over time
- **Sentiment Analysis** — positive/negative breakdown by app, disagreement cases, subjectivity
- **Aspect Explorer** — most-mentioned product features with sentiment breakdown
- **Issue Priority** — ranked product issue backlog with priority scores
- **Pipeline Health** — ingestion run history and data quality metrics

### AI Assistant

A RAG-based AI assistant in the sidebar powered by Claude. When a user asks a question:

1. Aggregated statistics are pulled from the database as context
2. Relevant negative reviews are retrieved via keyword matching
3. Claude generates an answer grounded in real data

---

## Dataset

| Metric | Value |
|--------|-------|
| Total reviews | ~95,000 |
| Apps covered | 11 |
| Language | English (en-US) |
| Cloud storage | Supabase PostgreSQL |
| Local storage | SQLite |

**Apps:** Spotify, WhatsApp, Instagram, Netflix, Amazon Shopping, Duolingo, Uber, YouTube, Microsoft Teams, X (Twitter), ChatGPT

---

## Deployment

The dashboard is deployed on Streamlit Cloud, connected to Supabase PostgreSQL.

To deploy your own instance:

1. Fork this repo
2. Create a Supabase project and run the schema SQL
3. Set secrets in Streamlit Cloud:
```toml
DATABASE_URL = "postgresql://..."
ANTHROPIC_API_KEY = "sk-ant-..."
```
4. Deploy via https://share.streamlit.io
