"""
UMAP Embedding Visualisation
==============================
Compares sentence-transformer embeddings vs TF-IDF+SVD embeddings
by reducing both to 2D with UMAP and plotting side by side.

If the sentence-transformer embeddings are capturing more meaningful
semantic signal, its plots should show cleaner separation by sentiment,
rating, and app compared to TF-IDF.

Usage:
    python visualise_embeddings.py              # uses features.db
    python visualise_embeddings.py --limit 5000 # subsample for speed
    python visualise_embeddings.py --db my.db

Output:
    umap_by_sentiment.png  — side-by-side sentiment comparison
    umap_by_rating.png     — side-by-side rating comparison
    umap_by_app.png        — side-by-side app comparison
"""

import sqlite3
import json
import argparse
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["UMAP_NO_TF"] = "1"
from umap import UMAP

DEFAULT_DB    = "features.db"
DEFAULT_LIMIT = None

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST    = 0.1
UMAP_METRIC      = "cosine"
RANDOM_STATE     = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────

def load_embeddings(db_path: str, limit: int = None) -> dict:
    """
    Loads both embedding types and metadata from features.db.
    Returns a dict with arrays for each field.
    Only includes rows where both embeddings are present.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT app_name, rating, combined_label,
               textblob_subjectivity, embedding, tfidf_embedding
        FROM features
        WHERE embedding IS NOT NULL
          AND tfidf_embedding IS NOT NULL
    """
    if limit:
        query += f" ORDER BY RANDOM() LIMIT {limit}"

    rows = conn.execute(query).fetchall()
    conn.close()

    st_embeddings   = []
    tfidf_embeddings = []
    labels   = []
    ratings  = []
    apps     = []

    for row in rows:
        try:
            st  = json.loads(row["embedding"])
            tfi = json.loads(row["tfidf_embedding"])
            if len(st) < 10 or len(tfi) < 10:
                continue
            st_embeddings.append(st)
            tfidf_embeddings.append(tfi)
            labels.append(row["combined_label"] or "unknown")
            ratings.append(row["rating"])
            apps.append(row["app_name"])
        except Exception:
            continue

    log.info(f"Loaded {len(st_embeddings):,} rows with both embeddings.")
    return {
        "st"     : np.array(st_embeddings),
        "tfidf"  : np.array(tfidf_embeddings),
        "labels" : np.array(labels),
        "ratings": np.array(ratings),
        "apps"   : np.array(apps),
    }


# ── UMAP reduction ────────────────────────────────────────────

def reduce(embeddings: np.ndarray, label: str) -> np.ndarray:
    """Runs UMAP on an embedding matrix and returns 2D coordinates."""
    log.info(f"Running UMAP on {label} ({embeddings.shape})...")
    reducer = UMAP(
        n_neighbors  = UMAP_N_NEIGHBORS,
        min_dist     = UMAP_MIN_DIST,
        metric       = UMAP_METRIC,
        random_state = RANDOM_STATE,
        low_memory   = True,
    )
    result = reducer.fit_transform(embeddings)
    log.info(f"UMAP done for {label}.")
    return result


# ── Plotting helpers ──────────────────────────────────────────

def setup_ax(ax, title: str):
    """Applies consistent styling to a subplot axis."""
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("UMAP dim 1", fontsize=8)
    ax.set_ylabel("UMAP dim 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_facecolor("#FAFAFA")


def plot_sentiment(ax, reduced: np.ndarray, labels: np.ndarray, title: str):
    """Plots sentiment labels on a given axis."""
    COLOURS = {"positive": "#4CAF50", "negative": "#F44336",
               "neutral": "#9E9E9E", "unknown": "#BDBDBD"}
    for label in ["neutral", "positive", "negative"]:
        mask = labels == label
        if mask.sum() == 0:
            continue
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   c=COLOURS[label], s=2, alpha=0.35, linewidths=0,
                   label=f"{label} ({mask.sum():,})")
    ax.legend(markerscale=3, fontsize=7, framealpha=0.8)
    setup_ax(ax, title)


def plot_rating(ax, reduced: np.ndarray, ratings: np.ndarray, title: str):
    """Plots star ratings on a given axis."""
    COLOURS = {1: "#D32F2F", 2: "#FF7043", 3: "#FFC107",
               4: "#66BB6A", 5: "#1B5E20"}
    for r in [3, 2, 4, 1, 5]:
        mask = ratings == r
        if mask.sum() == 0:
            continue
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   c=COLOURS[r], s=2, alpha=0.35, linewidths=0,
                   label=f"{r}★ ({mask.sum():,})")
    ax.legend(markerscale=3, fontsize=7, framealpha=0.8, title="Rating",
              title_fontsize=7)
    setup_ax(ax, title)


def plot_app(ax, reduced: np.ndarray, apps: np.ndarray, title: str):
    """Plots app clusters on a given axis."""
    unique_apps = sorted(set(apps))
    cmap = plt.colormaps["tab10"]
    colours = {a: cmap(i / max(len(unique_apps) - 1, 1))
               for i, a in enumerate(unique_apps)}
    for app in unique_apps:
        mask = apps == app
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   c=[colours[app]], s=2, alpha=0.3, linewidths=0,
                   label=f"{app} ({mask.sum():,})")
    ax.legend(markerscale=3, fontsize=6, framealpha=0.8,
              loc="upper left", bbox_to_anchor=(1.01, 1))
    setup_ax(ax, title)


# ── Side-by-side comparison plots ────────────────────────────

def make_comparison_plot(
    st_2d, tfidf_2d, colour_data, colour_fn,
    suptitle: str, filename: str,
    wide: bool = False,
):
    """
    Creates a 1×2 subplot comparing sentence-transformer (left)
    vs TF-IDF (right) for the same colour scheme.
    wide=True gives more horizontal space for the app legend.
    """
    figw = 18 if wide else 14
    fig, axes = plt.subplots(1, 2, figsize=(figw, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle(suptitle, fontsize=13, y=1.01)

    colour_fn(axes[0], st_2d,    colour_data, "Sentence-Transformer\n(all-MiniLM-L6-v2)")
    colour_fn(axes[1], tfidf_2d, colour_data, "TF-IDF + SVD (LSA)\n(100 dimensions)")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {filename}")


# ── Main ──────────────────────────────────────────────────────

def main(db_path: str, limit: int):
    log.info("Embedding visualisation started.")

    data = load_embeddings(db_path, limit=limit)
    if len(data["st"]) == 0:
        log.error("No embeddings found. Run feature_pipeline.py first.")
        return

    # reduce both embedding types to 2D
    st_2d    = reduce(data["st"],    "sentence-transformer")
    tfidf_2d = reduce(data["tfidf"], "TF-IDF")

    log.info("Generating comparison plots...")

    make_comparison_plot(
        st_2d, tfidf_2d,
        colour_data = data["labels"],
        colour_fn   = plot_sentiment,
        suptitle    = "Embedding Comparison — Coloured by Sentiment\n"
                      "Left: Sentence-Transformer  |  Right: TF-IDF + SVD",
        filename    = "umap_by_sentiment.png",
    )

    make_comparison_plot(
        st_2d, tfidf_2d,
        colour_data = data["ratings"],
        colour_fn   = plot_rating,
        suptitle    = "Embedding Comparison — Coloured by Star Rating\n"
                      "Left: Sentence-Transformer  |  Right: TF-IDF + SVD",
        filename    = "umap_by_rating.png",
    )

    make_comparison_plot(
        st_2d, tfidf_2d,
        colour_data = data["apps"],
        colour_fn   = plot_app,
        suptitle    = "Embedding Comparison — Coloured by App\n"
                      "Left: Sentence-Transformer  |  Right: TF-IDF + SVD",
        filename    = "umap_by_app.png",
        wide        = True,
    )

    log.info("Done.")
    log.info("  umap_by_sentiment.png")
    log.info("  umap_by_rating.png")
    log.info("  umap_by_app.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",    default=DEFAULT_DB)
    parser.add_argument("--limit", default=DEFAULT_LIMIT, type=int)
    args = parser.parse_args()
    main(args.db, args.limit)
