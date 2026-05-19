# Sentiment Data Pipeline — Phase I: Data Ingestion & Infrastructure

## Overview

This repository contains the foundational data pipeline for an AI-powered sentiment analysis system. The goal of Phase I is to build a reliable, automated infrastructure that collects user-generated app reviews from Google Play, structures the data, and stores it in a queryable relational database — ready for downstream labelling, model training, and experimentation.

The pipeline is designed to be modular and extensible, supporting multiple apps, incremental updates, and future integration with additional data sources.

---

## Pipeline Flow

```
Google Play Store
       ↓
  pipeline.py        ← scrapes reviews via GPlay Scraper
       ↓
  schema.sql         ← defines the database structure
       ↓
  pipeline.db        ← SQLite database (local, not tracked in git)
```

One script handles the full flow: scrape → validate → load into database. No intermediate files are generated.

---

## Files

| File | Description |
|------|-------------|
| `pipeline.py` | Main script. Scrapes reviews from Google Play and loads them directly into SQLite. |
| `schema.sql` | SQL schema defining the three database tables. Run automatically on first pipeline execution. |
| `.gitignore` | Excludes the database file, output folder, logs, and Python cache from version control. |

---

## Database Schema

The database consists of three tables:

### `apps`
Stores metadata for each app being tracked. Keeping app information separate avoids repeating it across every review row.

| Column | Type | Description |
|--------|------|-------------|
| `app_id` | TEXT (PK) | Google Play package name (e.g. `com.spotify.music`) |
| `app_name` | TEXT | Human-readable app name |
| `category` | TEXT | App category (e.g. `music`, `productivity`) |
| `created_at` | TIMESTAMP | When the app was first added to the database |

### `reviews`
Main table. One row per review.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK) | Auto-incremented internal ID |
| `review_id` | TEXT (UNIQUE) | Google Play's own review ID — used to prevent duplicate ingestion |
| `app_id` | TEXT (FK) | References `apps.app_id` |
| `ingestion_run_id` | INTEGER (FK) | References `ingestion_runs.id` — tracks which run collected this review |
| `rating` | INTEGER | Star rating (1–5) |
| `text` | TEXT | Review body text |
| `date` | TIMESTAMP | When the user wrote the review |
| `app_version` | TEXT | App version at time of review (currently unavailable from endpoint) |
| `thumbs_up` | INTEGER | Number of helpful votes |
| `reply` | TEXT | Developer reply (currently unavailable from endpoint) |
| `lang` | TEXT | Language code (e.g. `en`) |
| `country` | TEXT | Country code (e.g. `us`) |
| `scraped_at` | TIMESTAMP | When this review was collected by the pipeline |

### `ingestion_runs`
Tracks every pipeline execution. Each app gets one row per run, recording when it started, how many reviews were collected, and whether it succeeded.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK) | Auto-incremented run ID |
| `app_id` | TEXT (FK) | References `apps.app_id` |
| `started_at` | TIMESTAMP | When the run began |
| `completed_at` | TIMESTAMP | When the run finished (NULL if still in progress) |
| `reviews_collected` | INTEGER | Number of new reviews inserted in this run |
| `status` | TEXT | `in_progress`, `success`, or `failed` |

### Design Decisions
- **`review_id` as unique key** — prevents duplicate rows if the pipeline is re-run, making incremental updates safe by default.
- **`ingestion_runs` table** — provides full traceability of when data was collected, useful for debugging and for building scheduled/incremental update logic in future iterations.
- **`app_version` and `reply` retained despite being 100% null** — these fields are logically meaningful and reserved for future use if the data becomes available.
- **SQLite for prototyping** — lightweight, zero-configuration, and sufficient for the current data volume (~40k reviews). Can be migrated to PostgreSQL when the pipeline scales.

---

## Setup & Usage

### Requirements
```bash
pip install gplay-scraper curl-cffi
```

### Run the pipeline
```bash
python pipeline.py
```

This will:
1. Create `pipeline.db` and initialise the schema (first run only)
2. Scrape reviews for all apps defined in `TARGET_APPS`
3. Load new reviews into the database, skipping any already present

### Optional: specify a custom database path
```bash
python pipeline.py --db my_database.db
```

### Incremental updates
Re-running the pipeline is safe. The script checks existing `review_id` values in the database and only inserts reviews that aren't already there. No duplicates will be created.

---

## Current Dataset

| Metric | Value |
|--------|-------|
| Total reviews | ~40,800 |
| Apps covered | 11 |
| Language | English (en-US) |
| Collection method | NEWEST sort — most recent reviews per app |

**Apps included:** Spotify, WhatsApp, Instagram, Netflix, Amazon Shopping, Duolingo, Uber, YouTube, Microsoft Teams, X (Twitter), ChatGPT

---

## Notes

- `app_version` and `reply` are currently 100% null. Google Play does not expose these fields via the unofficial endpoint used for scraping. These columns are retained in the schema for potential future use.
- The pipeline uses `NEWEST` sort order, so collected reviews reflect recent activity. Historical coverage can be expanded in future iterations.
- Both `app_version` and `reply` fields should be excluded from any modelling pipeline until data becomes available.
