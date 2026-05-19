-- ============================================================
-- Google Play Reviews Pipeline — Database Schema
-- Database: SQLite (prototype)
-- ============================================================

-- ── 1. apps ──────────────────────────────────────────────────
-- One row per app. Keeps app metadata out of the reviews table.
CREATE TABLE IF NOT EXISTS apps (
    app_id      TEXT        PRIMARY KEY,        -- e.g. "com.spotify.music"
    app_name    TEXT        NOT NULL,            -- e.g. "Spotify"
    category    TEXT,                            -- e.g. "music", "productivity"
    created_at  TIMESTAMP   DEFAULT (datetime('now'))
);

-- ── 2. ingestion_runs ────────────────────────────────────────
-- One row per scraping session. Tracks when data was collected,
-- how many reviews were fetched, and whether the run succeeded.
-- Essential for incremental updates and debugging.
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id                  INTEGER     PRIMARY KEY AUTOINCREMENT,
    app_id              TEXT        NOT NULL REFERENCES apps(app_id),
    started_at          TIMESTAMP   NOT NULL,
    completed_at        TIMESTAMP,              -- NULL if run is still in progress
    reviews_collected   INTEGER     DEFAULT 0,
    status              TEXT        DEFAULT 'in_progress'
                                    CHECK(status IN ('in_progress', 'success', 'failed'))
);

-- ── 3. reviews ───────────────────────────────────────────────
-- Main table. One row per review.
CREATE TABLE IF NOT EXISTS reviews (
    id                  INTEGER     PRIMARY KEY AUTOINCREMENT,
    review_id           TEXT        NOT NULL UNIQUE,    -- Google Play's own ID, prevents duplicates
    app_id              TEXT        NOT NULL REFERENCES apps(app_id),
    ingestion_run_id    INTEGER     NOT NULL REFERENCES ingestion_runs(id),
    rating              INTEGER     NOT NULL CHECK(rating BETWEEN 1 AND 5),
    text                TEXT,
    date                TIMESTAMP,                      -- when the user wrote the review
    app_version         TEXT,                           -- currently 100% null, reserved for future
    thumbs_up           INTEGER     DEFAULT 0,
    reply               TEXT,                           -- developer reply, currently 100% null
    lang                TEXT,                           -- e.g. "en"
    country             TEXT,                           -- e.g. "us"
    scraped_at          TIMESTAMP   NOT NULL
);

-- ── Indexes ──────────────────────────────────────────────────
-- Speed up the most common query patterns
CREATE INDEX IF NOT EXISTS idx_reviews_app_id  ON reviews(app_id);
CREATE INDEX IF NOT EXISTS idx_reviews_rating  ON reviews(rating);
CREATE INDEX IF NOT EXISTS idx_reviews_date    ON reviews(date);
