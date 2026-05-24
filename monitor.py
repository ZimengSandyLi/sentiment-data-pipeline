"""
Pipeline Monitor
=================
Queries the pipeline database and produces a health report covering:
- ingestion run success/failure rates
- collection volume trends
- data quality indicators (duplicates, missing fields, short text)
- simple alerts when metrics fall outside expected ranges

Usage:
    python monitor.py                   # uses pipeline.db by default
    python monitor.py --db my.db        # custom database path
    python monitor.py --runs 20         # check last N ingestion runs (default 10)

Output is printed to terminal and saved to monitor_report.txt.
"""

import sqlite3
import argparse
import os
from datetime import datetime, timezone
from collections import defaultdict


# Configuration
DEFAULT_DB        = "pipeline.db"
DEFAULT_RUNS      = 10        # how many recent ingestion runs to show
REPORT_FILE       = "monitor_report.txt"

# Alert thresholds — adjust these as the pipeline matures
ALERT_FAILURE_RATE      = 0.2    # alert if >20% of runs failed
ALERT_DUPLICATE_RATE    = 0.25   # alert if >25% of reviews are duplicate texts
ALERT_SHORT_TEXT_RATE   = 0.3    # alert if >30% of reviews are under 20 chars
ALERT_MIN_REVIEWS       = 100    # alert if any successful run collected fewer than this


#Helpers
def utcnow() -> str:
    #returns current UTC time as a string
    #using UTC so timestamps are consistent regardless of where the script is run
    return datetime.now(timezone.utc).isoformat()

def get_connection(db_path: str) -> sqlite3.Connection:
    #check the database file exists before trying to open it
    #if it doesn't, it probably means pipeline.py hasn't been run yet
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found: {db_path}. Run pipeline.py first."
        )
    conn = sqlite3.connect(db_path)
    #row_factory lets us access query results by column name (e.g. row["app_name"])
    #instead of by index (e.g. row[0]), which is much easier to read
    conn.row_factory = sqlite3.Row
    return conn

def divider(char="─", width=55):
    #generates a divider line to make the report easier to read
    #char and width can be changed — e.g. divider("═") for a heavier line
    return char * width


#Report sections 

def section_overview(conn: sqlite3.Connection) -> tuple[list[str], list[str]]:
    """
    pulls the top-level numbers from the database — total reviews, apps, runs,
    success/failure counts, and the date range of collected reviews
    if the failure rate is above the threshold, adds an alert to flag it

    High-level numbers: total reviews, apps, and ingestion runs.
    Returns (lines, alerts).
    """
    lines  = []
    alerts = []

    total_reviews = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    total_apps    = conn.execute("SELECT COUNT(*) FROM apps").fetchone()[0]
    total_runs    = conn.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()[0]
    success_runs  = conn.execute("SELECT COUNT(*) FROM ingestion_runs WHERE status='success'").fetchone()[0]
    failed_runs   = conn.execute("SELECT COUNT(*) FROM ingestion_runs WHERE status='failed'").fetchone()[0]

    # earliest and most recent review dates
    date_range = conn.execute(
        "SELECT MIN(date), MAX(date) FROM reviews"
    ).fetchone()

    lines.append(f"  Total reviews in DB  : {total_reviews:,}")
    lines.append(f"  Apps tracked         : {total_apps}")
    lines.append(f"  Total ingestion runs : {total_runs}")
    lines.append(f"  Successful runs      : {success_runs}")
    lines.append(f"  Failed runs          : {failed_runs}")
    lines.append(f"  Review date range    : {date_range[0]} → {date_range[1]}")

    # alert if failure rate is too high
    if total_runs > 0:
        failure_rate = failed_runs / total_runs
        if failure_rate > ALERT_FAILURE_RATE:
            alerts.append(
                f"HIGH FAILURE RATE: {failure_rate*100:.1f}% of runs failed "
                f"(threshold: {ALERT_FAILURE_RATE*100:.0f}%)"
            )

    return lines, alerts


def section_run_history(conn: sqlite3.Connection, n: int) -> tuple[list[str], list[str]]:
    """
    shows the most recent N ingestion runs with their status, review count, and timestamp
    useful for quickly checking whether the pipeline has been running cleanly
    only alerts on low volume if it's the first run for that app
    incremental runs returning 0 is expected and not a problem
    """
    lines  = []
    alerts = []

    rows = conn.execute("""
        SELECT a.app_name, i.started_at, i.completed_at,
               i.reviews_collected, i.status
        FROM ingestion_runs i
        JOIN apps a ON i.app_id = a.app_id
        ORDER BY i.started_at DESC
        LIMIT ?
    """, (n,)).fetchall()

    if not rows:
        lines.append("  No ingestion runs found.")
        return lines, alerts

    for row in rows:
        # format status with a simple indicator
        status_icon = "✓" if row["status"] == "success" else "✗" if row["status"] == "failed" else "…"
        started = row["started_at"][:16] if row["started_at"] else "—"
        lines.append(
            f"  [{status_icon}] {row['app_name']:<22} "
            f"{row['reviews_collected']:>5} reviews   {started}"
        )

        # alert if a successful run collected very few reviews —
        # but only on the first run for this app. incremental runs legitimately
        # return 0 if nothing new has been posted since the last scrape, so
        # flagging those as warnings would just be noise.
        is_first_run = conn.execute("""
            SELECT COUNT(*) FROM ingestion_runs i
            JOIN apps a ON i.app_id = a.app_id
            WHERE a.app_name = ? AND i.started_at < ?
        """, (row["app_name"], row["started_at"])).fetchone()[0] == 0

        if (
            is_first_run
            and row["status"] == "success"
            and row["reviews_collected"] is not None
            and row["reviews_collected"] < ALERT_MIN_REVIEWS
        ):
            alerts.append(
                f"LOW VOLUME: {row['app_name']} collected only "
                f"{row['reviews_collected']} reviews on first run "
                f"(threshold: {ALERT_MIN_REVIEWS})"
            )

    return lines, alerts


def section_volume_per_app(conn: sqlite3.Connection) -> tuple[list[str], list[str]]:
    """
    Shows total reviews stored per app, and how many came from each ingestion run.
    Useful for spotting which apps have less coverage.
    """
    lines  = []
    alerts = []

    rows = conn.execute("""
        SELECT a.app_name, COUNT(*) as total
        FROM reviews r
        JOIN apps a ON r.app_id = a.app_id
        GROUP BY a.app_name
        ORDER BY total DESC
    """).fetchall()

    for row in rows:
        lines.append(f"  {row['app_name']:<28} {row['total']:>6,} reviews")

    return lines, alerts


def section_data_quality(conn: sqlite3.Connection) -> tuple[list[str], list[str]]:
    """
    Checks for common data quality issues:
    - missing fields (app_version, reply)
    - short / low-signal reviews
    - duplicate texts within the same app
    """
    lines  = []
    alerts = []

    total = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    if total == 0:
        lines.append("  No reviews to analyse.")
        return lines, alerts

    # missing fields
    missing_version = conn.execute(
        "SELECT COUNT(*) FROM reviews WHERE app_version IS NULL"
    ).fetchone()[0]
    missing_reply = conn.execute(
        "SELECT COUNT(*) FROM reviews WHERE reply IS NULL"
    ).fetchone()[0]

    lines.append("  Missing fields:")
    lines.append(f"    app_version  : {missing_version:,} / {total:,} ({missing_version/total*100:.1f}%)")
    lines.append(f"    reply        : {missing_reply:,} / {total:,} ({missing_reply/total*100:.1f}%)")

    # short text
    short_10 = conn.execute(
        "SELECT COUNT(*) FROM reviews WHERE LENGTH(text) < 10"
    ).fetchone()[0]
    short_20 = conn.execute(
        "SELECT COUNT(*) FROM reviews WHERE LENGTH(text) < 20"
    ).fetchone()[0]

    short_rate = short_20 / total
    lines.append("")
    lines.append("  Short / low-signal reviews:")
    lines.append(f"    < 10 chars   : {short_10:,} ({short_10/total*100:.1f}%)")
    lines.append(f"    < 20 chars   : {short_20:,} ({short_20/total*100:.1f}%)")

    if short_rate > ALERT_SHORT_TEXT_RATE:
        alerts.append(
            f"HIGH SHORT-TEXT RATE: {short_rate*100:.1f}% of reviews are under 20 chars "
            f"(threshold: {ALERT_SHORT_TEXT_RATE*100:.0f}%)"
        )

    # duplicate texts within same app
    dup_count = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT app_id, text, COUNT(*) as n
            FROM reviews
            WHERE text IS NOT NULL
            GROUP BY app_id, text
            HAVING n > 1
        )
    """).fetchone()[0]

    # total reviews that are part of a duplicate group
    dup_reviews = conn.execute("""
        SELECT SUM(n) FROM (
            SELECT COUNT(*) as n
            FROM reviews
            WHERE text IS NOT NULL
            GROUP BY app_id, text
            HAVING n > 1
        )
    """).fetchone()[0] or 0

    dup_rate = dup_reviews / total
    lines.append("")
    lines.append("  Duplicate texts (within same app):")
    lines.append(f"    Unique duplicate phrases : {dup_count:,}")
    lines.append(f"    Reviews affected         : {dup_reviews:,} ({dup_rate*100:.1f}%)")

    if dup_rate > ALERT_DUPLICATE_RATE:
        alerts.append(
            f"HIGH DUPLICATE RATE: {dup_rate*100:.1f}% of reviews are duplicate texts "
            f"(threshold: {ALERT_DUPLICATE_RATE*100:.0f}%)"
        )

    # most common duplicate phrases — useful to spot noise patterns
    lines.append("")
    lines.append("  Most repeated review texts:")
    top_dups = conn.execute("""
        SELECT a.app_name, r.text, COUNT(*) as n
        FROM reviews r
        JOIN apps a ON r.app_id = a.app_id
        WHERE r.text IS NOT NULL
        GROUP BY r.app_id, r.text
        HAVING n > 1
        ORDER BY n DESC
        LIMIT 5
    """).fetchall()
    for row in top_dups:
        preview = row["text"][:40].replace("\n", " ")
        lines.append(f"    [{row['app_name']}] \"{preview}\" × {row['n']}")

    return lines, alerts


def section_rating_distribution(conn: sqlite3.Connection) -> tuple[list[str], list[str]]:
    """
    shows the 1-5 star breakdown as a simple ASCII bar chart
    makes it easy to see class imbalance immediately
    matters for modelling later since 1★ and 5★ tend to dominate
    """
    lines  = []
    alerts = []

    total = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    rows  = conn.execute(
        "SELECT rating, COUNT(*) as n FROM reviews GROUP BY rating ORDER BY rating"
    ).fetchall()

    for row in rows:
        pct  = row["n"] / total * 100
        bar  = "█" * int(pct / 2)   # simple ascii bar chart
        lines.append(f"  {row['rating']}★  {bar:<25} {row['n']:>6,}  ({pct:.1f}%)")

    return lines, alerts


# ── Health summary ────────────────────────────────────────────

def health_summary(all_alerts: list[str]) -> list[str]:
    """
    Prints a final status based on whether any alerts were triggered.
    WARNING with a list of issues if anything triggered
    prints at the bottom of the report so the overall status is easy to find
    """
    lines = []
    if not all_alerts:
        lines.append("  Status  : HEALTHY — no issues detected")
    else:
        lines.append(f"  Status  : WARNING — {len(all_alerts)} issue(s) detected")
        lines.append("")
        for alert in all_alerts:
            lines.append(f"{alert}")
    return lines


# Report builder
def build_report(conn: sqlite3.Connection, n_runs: int) -> str:
    """
    Runs all sections and assembles the full report as a string.
    """
    all_alerts = []
    report     = []

    timestamp = utcnow()
    report.append(divider("═"))
    report.append("  PIPELINE HEALTH REPORT")
    report.append(f"  Generated : {timestamp}")
    report.append(divider("═"))

    #overview
    report.append("")
    report.append("  OVERVIEW")
    report.append(divider())
    lines, alerts = section_overview(conn)
    report.extend(lines)
    all_alerts.extend(alerts)

    #ingestion run history
    report.append("")
    report.append(f"  LAST {n_runs} INGESTION RUNS")
    report.append(divider())
    lines, alerts = section_run_history(conn, n_runs)
    report.extend(lines)
    all_alerts.extend(alerts)

    #volume per app
    report.append("")
    report.append("  COLLECTION VOLUME PER APP")
    report.append(divider())
    lines, alerts = section_volume_per_app(conn)
    report.extend(lines)
    all_alerts.extend(alerts)

    # rating distribution
    report.append("")
    report.append("  RATING DISTRIBUTION")
    report.append(divider())
    lines, alerts = section_rating_distribution(conn)
    report.extend(lines)
    all_alerts.extend(alerts)

    #data quality
    report.append("")
    report.append("  DATA QUALITY")
    report.append(divider())
    lines, alerts = section_data_quality(conn)
    report.extend(lines)
    all_alerts.extend(alerts)

    #health summary
    report.append("")
    report.append("  HEALTH SUMMARY")
    report.append(divider("═"))
    report.extend(health_summary(all_alerts))
    report.append(divider("═"))

    return "\n".join(report)


#main entrance
def main(db_path: str, n_runs: int):
    conn   = get_connection(db_path)
    report = build_report(conn, n_runs)
    conn.close()

    # print to terminal
    print(report)

    # save to file so there's a record of each time the monitor was run
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline health monitor.")
    parser.add_argument("--db",   default=DEFAULT_DB,   help="SQLite database path")
    parser.add_argument("--runs", default=DEFAULT_RUNS, type=int, help="Number of recent runs to show")
    args = parser.parse_args()
    main(args.db, args.runs)
