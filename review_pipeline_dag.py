"""
Review Pipeline DAG
====================
Airflow DAG that schedules the full review intelligence pipeline to run daily.

Task order:
    1. scrape_and_load   — runs pipeline.py to collect new reviews
    2. run_features      — runs feature_pipeline.py to extract NLP features
    3. health_check      — runs monitor.py to generate the health report

Each task only starts if the previous one succeeded.

Setup:
    1. Copy this file to ~/airflow/dags/review_pipeline_dag.py
    2. Update PROJECT_DIR below to point to your project folder
    3. airflow scheduler &
    4. airflow webserver &
    5. Visit http://localhost:8080 to see the DAG
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import subprocess
import sys
import os

# ── Config ─────────────────────────────────────────────────────
# Update this to your actual project folder path
PROJECT_DIR = os.path.expanduser("~/Documents/Job Apply/MLE_Intern")

# Use the same Python interpreter that has all the dependencies installed
PYTHON = sys.executable

# Default args applied to every task in the DAG
default_args = {
    "owner"           : "zimeng",
    "retries"         : 1,                        # retry once if a task fails
    "retry_delay"     : timedelta(minutes=5),     # wait 5 mins before retry
    "email_on_failure": False,                    # set to True and add email if you want alerts
}

# ── Task functions ─────────────────────────────────────────────

def run_pipeline():
    """
    Step 1: scrape new reviews from Google Play and load into pipeline.db.
    Calls pipeline.py in the project directory.
    Incremental update — only new reviews will be inserted.
    """
    result = subprocess.run(
        [PYTHON, "pipeline.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception(f"pipeline.py failed with return code {result.returncode}")
    print("pipeline.py completed successfully.")


def run_feature_pipeline():
    """
    Step 2: extract NLP features from newly ingested reviews.
    Calls feature_pipeline.py with --skip-llm (no API cost).
    Only processes reviews not already in features.db.
    """
    result = subprocess.run(
        [PYTHON, "feature_pipeline.py", "--skip-llm"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception(f"feature_pipeline.py failed with return code {result.returncode}")
    print("feature_pipeline.py completed successfully.")


def run_monitor():
    """
    Step 3: run the health monitor and save the report.
    Calls monitor.py to generate monitor_report.txt.
    If any alerts are triggered, they will appear in the Airflow task logs.
    """
    result = subprocess.run(
        [PYTHON, "monitor.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception(f"monitor.py failed with return code {result.returncode}")
    print("monitor.py completed successfully.")


# ── DAG definition ─────────────────────────────────────────────

with DAG(
    dag_id            = "review_intelligence_pipeline",
    description       = "Daily pipeline: scrape Google Play reviews → extract features → health check",
    default_args      = default_args,
    start_date        = datetime(2026, 7, 1),
    schedule_interval = "@daily",    # runs once per day at midnight UTC
    catchup           = False,       # don't backfill missed runs
    tags              = ["reviews", "nlp", "sentiment"],
) as dag:

    # Task 1 — scrape and load new reviews
    task_scrape = PythonOperator(
        task_id         = "scrape_and_load",
        python_callable = run_pipeline,
        doc_md          = "Scrapes new reviews from Google Play and loads them into pipeline.db.",
    )

    # Task 2 — extract NLP features
    task_features = PythonOperator(
        task_id         = "extract_features",
        python_callable = run_feature_pipeline,
        doc_md          = "Runs NLP feature pipeline on new reviews.",
    )

    # Task 3 — health check
    task_monitor = PythonOperator(
        task_id         = "health_check",
        python_callable = run_monitor,
        doc_md          = "Generates pipeline health report and flags any data quality issues.",
    )

    # Define execution order: scrape → features → monitor
    task_scrape >> task_features >> task_monitor
