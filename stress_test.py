"""
Pipeline Stress Test
=====================
Tests the operational limits of the Google Play scraping pipeline across three dimensions:

  Test 1 — Rate limit:     how quickly consecutive runs trigger throttling or blocks
  Test 2 — Volume ceiling: how many reviews can realistically be fetched in one run
  Test 3 — Delay sensitivity: how short the inter-app delay can be before errors appear

Results are printed to terminal and saved to stress_test_report.txt.

Usage:
    python stress_test.py               # runs all three tests
    python stress_test.py --test 1      # run a specific test only (1, 2, or 3)
    python stress_test.py --app com.whatsapp   # use a different test app

Note: this will make real requests to Google Play. Run from your own machine
(residential IP), not a server or VPN, to avoid immediate blocks.
"""

import time
import argparse
import logging
from datetime import datetime, timezone
from gplay_scraper import GPlayScraper

#Configuration

# Use a high-volume app for testing — more data available means less chance
# of hitting an app-level ceiling rather than a rate limit.
DEFAULT_TEST_APP  = "com.whatsapp"
DEFAULT_TEST_APP_NAME = "WhatsApp"

REPORT_FILE = "stress_test_report.txt"

# Test 1 — Rate limit
# Run the same app this many times back-to-back with a short delay between runs.
T1_RUNS          = 5
T1_REVIEWS       = 500     # small count so each run finishes quickly
T1_DELAY         = 5       # seconds between runs

# Test 2 — Volume ceiling
# Try fetching progressively more reviews in a single run.
T2_VOLUMES       = [1000, 2500, 5000, 10000]
T2_DELAY         = 30      # wait between volume tests to avoid rate limiting

# Test 3 — Delay sensitivity
# Run the same app twice with different delays, and see what happens.
T3_DELAYS        = [0, 2, 5, 10]   # seconds — 0 means fire immediately
T3_REVIEWS       = 500


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# Helpers

def utcnow() -> str:
    #similarly, returns current UTC time as a string to keep consistent UTC timestamp
    return datetime.now(timezone.utc).isoformat()

def divider(char="─", width=58):
    #generates a divider line
    return char * width

def scrape(scraper: GPlayScraper, app_id: str, count: int) -> dict:
    """
    Runs a single scrape and returns a result dict with timing and outcome.
    Catches exceptions so a failure doesn't crash the whole test suite.
    """
    start = time.time()
    result = {
        "requested" : count,
        "returned"  : 0,
        "elapsed_s" : 0,
        "status"    : "unknown",
        "error"     : None,
    }
    try:
        raw = scraper.reviews_analyze(
            app_id,
            count=count,
            sort="NEWEST",
            lang="en",
            country="us",
        )
        result["returned"]  = len(raw) if raw else 0
        result["elapsed_s"] = round(time.time() - start, 1)
        result["status"]    = "success" if raw else "empty"

    except Exception as e:
        result["elapsed_s"] = round(time.time() - start, 1)
        result["status"]    = "error"
        result["error"]     = str(e)

    return result


# Test 1: Rate limit

def test_rate_limit(scraper: GPlayScraper, app_id: str) -> list[str]:
    """
    Fires consecutive scraping runs with a short delay between each,
    and records how many reviews each run returns.
    A significant drop in returned reviews suggests rate limiting is kicking in.
    """
    lines = []
    lines.append(f"  App       : {app_id}")
    lines.append(f"  Runs      : {T1_RUNS}")
    lines.append(f"  Reviews/run : {T1_REVIEWS}")
    lines.append(f"  Delay between runs : {T1_DELAY}s")
    lines.append("")

    first_count = None
    for i in range(1, T1_RUNS + 1):
        log.info(f"  T1 run {i}/{T1_RUNS}...")
        result = scrape(scraper, app_id, T1_REVIEWS)

        # compare against first run to spot degradation
        if first_count is None and result["status"] == "success":
            first_count = result["returned"]

        drop = ""
        if first_count and result["returned"] < first_count * 0.8:
            drop = "  ← possible throttling"

        lines.append(
            f"  Run {i}:  status={result['status']:<8}  "
            f"returned={result['returned']:>5}  "
            f"time={result['elapsed_s']}s"
            f"{drop}"
        )
        if result["error"]:
            lines.append(f"    Error: {result['error']}")

        if i < T1_RUNS:
            time.sleep(T1_DELAY)

    return lines


#Test 2: Volume ceiling

def test_volume_ceiling(scraper: GPlayScraper, app_id: str) -> list[str]:
    """
    Tries fetching increasing numbers of reviews in a single call.
    Tracks how many are actually returned vs requested, and how long each takes.
    A big gap between requested and returned signals a practical ceiling.
    """
    lines = []
    lines.append(f"  App    : {app_id}")
    lines.append(f"  Volumes tested : {T2_VOLUMES}")
    lines.append(f"  Delay between tests : {T2_DELAY}s")
    lines.append("")
    lines.append(f"  {'Requested':>10}  {'Returned':>10}  {'Fill rate':>10}  {'Time':>8}")
    lines.append(f"  {divider('-', 46)}")

    for count in T2_VOLUMES:
        log.info(f"  T2 volume test: requesting {count} reviews...")
        result = scrape(scraper, app_id, count)

        fill_rate = (
            f"{result['returned']/count*100:.1f}%"
            if count > 0 and result["status"] == "success"
            else "—"
        )
        flag = ""
        if result["status"] == "success" and result["returned"] < count * 0.5:
            flag = "  ← significant shortfall"

        lines.append(
            f"  {count:>10,}  {result['returned']:>10,}  {fill_rate:>10}  "
            f"{result['elapsed_s']:>6}s{flag}"
        )
        if result["error"]:
            lines.append(f"    Error: {result['error']}")
        if result["status"] == "error":
            lines.append("    Stopping volume test early due to error.")
            break

        time.sleep(T2_DELAY)

    return lines


#Test 3: Delay sensitivity

def test_delay_sensitivity(scraper: GPlayScraper, app_id: str) -> list[str]:
    """
    Runs pairs of back-to-back scrapes with different delays between them.
    Helps identify the minimum safe delay before throttling or errors appear.
    """
    lines = []
    lines.append(f"  App     : {app_id}")
    lines.append(f"  Reviews per scrape : {T3_REVIEWS}")
    lines.append(f"  Delays tested : {T3_DELAYS}s")
    lines.append("")

    for delay in T3_DELAYS:
        log.info(f"  T3 delay test: {delay}s gap between two back-to-back scrapes...")

        # first scrape
        r1 = scrape(scraper, app_id, T3_REVIEWS)
        lines.append(f"  Delay {delay:>3}s — scrape 1: status={r1['status']:<8} returned={r1['returned']:>5}")

        time.sleep(delay)

        #second scrape immediately after the delay
        r2 = scrape(scraper, app_id, T3_REVIEWS)

        flag = ""
        if r1["status"] == "success" and r2["returned"] < r1["returned"] * 0.8:
            flag = "  ← possible degradation vs scrape 1"

        lines.append(
            f"  Delay {delay:>3}s — scrape 2: status={r2['status']:<8} "
            f"returned={r2['returned']:>5}{flag}"
        )
        if r2["error"]:
            lines.append(f"    Error: {r2['error']}")
        lines.append("")

        #cool down between delay tests so they don't bleed into each other
        log.info("  Cooling down 30s before next delay test...")
        time.sleep(30)

    return lines


#Report builder
def build_report(scraper: GPlayScraper, app_id: str, tests: list[int]) -> str:
    lines = []
    timestamp = utcnow()

    lines.append(divider("═"))
    lines.append("  PIPELINE STRESS TEST REPORT")
    lines.append(f"  Generated : {timestamp}")
    lines.append(f"  Test app  : {app_id}")
    lines.append(divider("═"))

    if 1 in tests:
        lines.append("")
        lines.append("  TEST 1 — RATE LIMIT")
        lines.append(f"  {divider()}")
        lines.append("  Fires consecutive runs to check for throttling.")
        lines.append("  A drop in returned reviews suggests rate limiting.")
        lines.append("")
        log.info("Starting Test 1: Rate limit...")
        lines.extend(test_rate_limit(scraper, app_id))

    if 2 in tests:
        lines.append("")
        lines.append("  TEST 2 — VOLUME CEILING")
        lines.append(f"  {divider()}")
        lines.append("  Requests increasing review counts in a single call.")
        lines.append("  A big gap between requested vs returned = practical ceiling.")
        lines.append("")
        log.info("Starting Test 2: Volume ceiling...")
        lines.extend(test_volume_ceiling(scraper, app_id))

    if 3 in tests:
        lines.append("")
        lines.append("  TEST 3 — DELAY SENSITIVITY")
        lines.append(f"  {divider()}")
        lines.append("  Tests back-to-back scrapes with different gaps.")
        lines.append("  Identifies the minimum safe delay between runs.")
        lines.append("")
        log.info("Starting Test 3: Delay sensitivity...")
        lines.extend(test_delay_sensitivity(scraper, app_id))

    lines.append("")
    lines.append(divider("═"))
    lines.append("  END OF REPORT")
    lines.append(divider("═"))

    return "\n".join(lines)


#main entrance

def main(app_id: str, tests: list[int]):
    log.info("Stress test started.")
    log.info(f"App: {app_id} | Tests: {tests}")

    #curl_cffi is more resilient to bot detection — important for stress testing
    scraper = GPlayScraper(http_client="curl_cffi")

    report = build_report(scraper, app_id, tests)

    print(report)
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline stress test.")
    parser.add_argument(
        "--app", default=DEFAULT_TEST_APP,
        help=f"Google Play app ID to test (default: {DEFAULT_TEST_APP})"
    )
    parser.add_argument(
        "--test", type=int, choices=[1, 2, 3], default=None,
        help="Run a specific test only (1, 2, or 3). Omit to run all."
    )
    args = parser.parse_args()

    tests_to_run = [args.test] if args.test else [1, 2, 3]
    main(args.app, tests_to_run)
