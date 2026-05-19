"""
Google Play Reviews Pipeline
=============================
This script handles the full ingestion flow in one go:
scrape reviews from Google Play → load directly into SQLite.

I originally had this split into two separate scripts (scraper.py and load_data.py)
but merged them so the pipeline runs end-to-end without generating intermediate CSV files.

Usage:
    python pipeline.py                        # scrape all apps in TARGET_APPS
    python pipeline.py --db my_pipeline.db   # use a custom database path

Dependencies:
    pip install gplay-scraper curl-cffi
"""

import sqlite3
import logging
import time
import argparse
import os
from datetime import datetime, timezone
from gplay_scraper import GPlayScraper


#Configuration
#add or remove apps here to update the next scraping result. Format is:
#"google_play_package_name": ("Display Name", "category")
#the package name is the id= parameter in the Play Store URL.
TARGET_APPS = {
    "com.spotify.music":                 ("Spotify",          "music"),
    "com.whatsapp":                      ("WhatsApp",         "messaging"),
    "com.instagram.android":             ("Instagram",        "social"),
    "com.netflix.mediaclient":           ("Netflix",          "entertainment"),
    "com.amazon.mShop.android.shopping": ("Amazon Shopping",  "ecommerce"),
    "com.duolingo":                      ("Duolingo",         "education"),
    "com.ubercab":                       ("Uber",             "transportation"),
    "com.google.android.youtube":        ("YouTube",          "entertainment"),
    "com.microsoft.teams":               ("Microsoft Teams",  "productivity"),
    "com.twitter.android":              ("X (Twitter)",      "social"),
    "com.openai.chatgpt":                ("ChatGPT",          "ai"),
}

REVIEWS_PER_APP    = 2500   # how many reviews to fetch per app per run
LANGUAGE           = "en"
COUNTRY            = "us"

#NEWEST: using MOST_RELEVANT (the default) causes Google to
#silently cap results at a few hundred, which I ran into during testing.
SORT               = "NEWEST"

#A short wait between apps to avoid hitting Google's rate limits.
#10 seconds has been reliable in testing.
DELAY_BETWEEN_APPS = 10

DEFAULT_DB         = "pipeline.db"
SCHEMA_FILE        = "schema.sql"
LOG_LEVEL          = logging.INFO


#Logging
#Logs go both to the terminal and to pipeline.log so I can check
#what happened after the fact without re-running anything.
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
log = logging.getLogger(__name__)


#Helpers
def utcnow() -> str:
    """Returns current UTC time as an ISO string. Used for all timestamps in the DB."""
    return datetime.now(timezone.utc).isoformat()


#Database setup

def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Opens a connection to the SQLite database.
    foreign_keys = ON enforces referential integrity between tables.
    row_factory = sqlite3.Row lets us access columns by name instead of index.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_database(conn: sqlite3.Connection):
    """
    Reads schema.sql and creates the tables if they don't exist yet.
    Safe to run on every startup — CREATE TABLE IF NOT EXISTS means it won't
    overwrite anything if the DB is already set up.
    """
    #check schema file exists before trying to read it
    if not os.path.exists(SCHEMA_FILE):
        raise FileNotFoundError(
            f"schema.sql not found. Make sure it's in the same folder as pipeline.py."
        )

    #executescript runs all the CREATE TABLE statements in one go
    with open(SCHEMA_FILE) as f:
        conn.executescript(f.read())
    conn.commit()
    log.info("Database ready.")


#App tracking

def upsert_app(conn: sqlite3.Connection, app_id: str, app_name: str, category: str):
    """
    Adds the app to the apps table if it isn't there already.
    INSERT OR IGNORE means re-running the pipeline won't cause duplicate app rows.
    """
    conn.execute(
        "INSERT OR IGNORE INTO apps (app_id, app_name, category) VALUES (?, ?, ?)",
        (app_id, app_name, category),
    )
    conn.commit()


#Ingestion run tracking
#Each time the pipeline runs for an app, it creates a row in ingestion_runs.
#This gives a full history of when data was collected, how much was fetched, and whether the run succeeded
# useful for debugging and future scheduling.

def create_ingestion_run(conn: sqlite3.Connection, app_id: str) -> int:
    """Opens a new ingestion run record and returns its id."""
    #status starts as 'in_progress' and gets updated to 'success' or 'failed'
    #by complete_ingestion_run() once the scraping is done
    cursor = conn.execute(
        "INSERT INTO ingestion_runs (app_id, started_at, status) VALUES (?, ?, 'in_progress')",
        (app_id, utcnow()),
    )
    conn.commit()
    return cursor.lastrowid  #need this id to link reviews back to this run


def complete_ingestion_run(
    conn: sqlite3.Connection, run_id: int, reviews_collected: int, success: bool
):
    """Updates the ingestion run record with the final result."""
    conn.execute(
        """
        UPDATE ingestion_runs
        SET completed_at      = ?,
            reviews_collected = ?,
            status            = ?
        WHERE id = ?
        """,
        (utcnow(), reviews_collected, "success" if success else "failed", run_id),
    )
    conn.commit()


#Incremental update logic

def get_existing_review_ids(conn: sqlite3.Connection, app_id: str) -> set:
    """
    Fetches all review_ids already stored for this app.
    Used to skip reviews we've already collected so re-running the pipeline
    is safe and doesn't create duplicates.
    """
    #pull just the ids — no need to fetch the full review rows
    rows = conn.execute(
        "SELECT review_id FROM reviews WHERE app_id = ?", (app_id,)
    ).fetchall()

    #return as a set for O(1) lookup in insert_reviews
    return {row["review_id"] for row in rows}


#Review insertion

def insert_reviews(
    conn: sqlite3.Connection,
    reviews: list[dict],
    app_id: str,
    run_id: int,
    existing_ids: set,
) -> tuple[int, int]:
    """
    Inserts new reviews into the database, skipping any already present.

    Two layers of deduplication:
    1. Check against existing_ids (reviews already in DB from previous runs)
    2. INSERT OR IGNORE as a safety net in case of any edge cases

    Returns (inserted, skipped) counts.
    """
    inserted = 0
    skipped  = 0

    for r in reviews:
        rid = r.get("reviewId", "")

        #layer 1: skip if we've seen this review_id before
        if rid in existing_ids:
            skipped += 1
            continue

        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO reviews (
                    review_id, app_id, ingestion_run_id,
                    rating, text, date,
                    app_version, thumbs_up, reply,
                    lang, country, scraped_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    app_id,
                    run_id,
                    int(r.get("score", 0)),
                    r.get("content") or None,
                    str(r["at"]) if r.get("at") else None,
                    #app_version and replyContent are currently always null —
                    #Google doesn't expose these via the unofficial endpoint.
                    #Keeping the columns in the schema for potential future use.
                    r.get("reviewCreatedVersion") or None,
                    int(r.get("thumbsUpCount") or 0),
                    r.get("replyContent") or None,
                    LANGUAGE,
                    COUNTRY,
                    utcnow(),
                ),
            )
            inserted += 1
            #layer 2: add to the set so we don't insert the same id twice
            #within the same run (edge case but worth handling)
            existing_ids.add(rid)

        except Exception as e:
            log.warning(f"Failed to insert review {rid}: {e}")
            skipped += 1

    #commit once after all inserts rather than per-row for better performance
    conn.commit()
    return inserted, skipped


#Main scraping loop

def scrape_and_load(conn: sqlite3.Connection, scraper: GPlayScraper):
    """
    Iterates through TARGET_APPS, scrapes reviews for each one,
    and loads them directly into the database.
    """
    total_inserted = 0
    total_skipped  = 0

    for i, (app_id, (app_name, category)) in enumerate(TARGET_APPS.items()):

        log.info(f"── {app_name} ({app_id}) ──")

        #step 1: make sure this app exists in the apps table
        upsert_app(conn, app_id, app_name, category)

        #step 2: fetch existing review ids so we can skip duplicates later
        existing_ids = get_existing_review_ids(conn, app_id)
        if existing_ids:
            log.info(f"  {len(existing_ids):,} reviews already in DB — will skip duplicates.")

        #step 3: open an ingestion run to track this session
        run_id = create_ingestion_run(conn, app_id)

        #step 4: scrape reviews from Google Play
        try:
            raw = scraper.reviews_analyze(
                app_id,
                count=REVIEWS_PER_APP,
                sort=SORT,
                lang=LANGUAGE,
                country=COUNTRY,
            )
        except Exception as e:
            #if scraping fails, mark the run as failed and move on to the next app
            log.error(f"  Scraping failed: {e}")
            complete_ingestion_run(conn, run_id, 0, success=False)
            continue

        if not raw:
            log.warning(f"  No reviews returned.")
            complete_ingestion_run(conn, run_id, 0, success=False)
            continue

        log.info(f"  Fetched {len(raw)} reviews from Google Play.")

        #step 5: insert new reviews into the database
        inserted, skipped = insert_reviews(conn, raw, app_id, run_id, existing_ids)

        #step 6: close the ingestion run with final counts
        complete_ingestion_run(conn, run_id, inserted, success=True)

        log.info(f"  Inserted: {inserted} | Skipped: {skipped}")
        total_inserted += inserted
        total_skipped  += skipped

        #wait between apps to be respectful of rate limits
        if i < len(TARGET_APPS) - 1:
            log.info(f"  Waiting {DELAY_BETWEEN_APPS}s before next app...")
            time.sleep(DELAY_BETWEEN_APPS)

    return total_inserted, total_skipped


#Post-run summary

def print_summary(conn: sqlite3.Connection):
    """Prints a quick summary after the pipeline finishes — good for a sanity check."""
    log.info("=" * 50)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 50)

    #total reviews across all apps
    total = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    log.info(f"Total reviews in DB : {total:,}")

    #breakdown per app
    log.info("Per app:")
    for row in conn.execute("""
        SELECT a.app_name, COUNT(*) as n
        FROM reviews r JOIN apps a ON r.app_id = a.app_id
        GROUP BY a.app_name ORDER BY n DESC
    """):
        log.info(f"  {row['app_name']:<30} {row['n']:,}")

    #last 10 ingestion runs so I can see what happened most recently
    log.info("Last ingestion runs:")
    for row in conn.execute("""
        SELECT a.app_name, i.completed_at, i.reviews_collected, i.status
        FROM ingestion_runs i JOIN apps a ON i.app_id = a.app_id
        ORDER BY i.started_at DESC LIMIT 10
    """):
        log.info(f"  [{row['status']}] {row['app_name']}: {row['reviews_collected']} reviews")

    log.info("=" * 50)


#main entrance 
def main(db_path: str):
    log.info("Pipeline started.")
    log.info(f"Database: {db_path}")

    #set up DB connection and initialise schema
    conn = get_connection(db_path)
    init_database(conn)

    #curl_cffi handles bot detection better than plain requests
    scraper = GPlayScraper(http_client="curl_cffi")

    start = time.time()
    inserted, skipped = scrape_and_load(conn, scraper)
    elapsed = round(time.time() - start, 1)

    log.info(f"Done in {elapsed}s — {inserted:,} inserted, {skipped:,} skipped.")
    print_summary(conn)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Play reviews pipeline.")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    args = parser.parse_args()
    main(args.db)
