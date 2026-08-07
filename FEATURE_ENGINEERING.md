# Feature Engineering Pipeline

## Overview

This pipeline transforms raw app reviews from the Phase I database into structured, model-ready features. Each review is processed through four modules, each implemented with both a traditional NLP method and an LLM-based method for comparison.

The primary goal is to understand what kind of signal can be extracted from raw review text, and whether LLM-based approaches produce meaningfully better features than traditional methods for downstream sentiment modelling.

---

## Pipeline Flow

```
pipeline.db (Phase I)
       ↓
feature_pipeline.py
       ├── Module 1: Sentiment Polarity   (VADER + combined scoring)
       ├── Module 2: Subjectivity         (TextBlob)
       ├── Module 3: Aspect Extraction    (spaCy noun chunks)
       └── Module 4: Embeddings           (TF-IDF + SVD  vs  sentence-transformers)
       ↓
features.db / features.csv
       ↓
visualise_embeddings.py  →  umap_by_sentiment.png
                             umap_by_rating.png
                             umap_by_app.png
```

---

## Files

| File | Description |
|------|-------------|
| `feature_pipeline.py` | Main pipeline. Reads from `pipeline.db`, extracts all features, writes to `features.db` and `features.csv`. |
| `visualise_embeddings.py` | Loads both embedding types from `features.db`, runs UMAP, and generates side-by-side comparison plots. |
| `features.db` | Output SQLite database with one row per review and all extracted features. Not tracked in git. |
| `features.csv` | Same data in CSV. Embedding vectors are truncated to a 5-value preview. Not tracked in git. |

---

## Usage

```bash
# install dependencies
pip install vaderSentiment textblob sentence-transformers anthropic scikit-learn umap-learn
python -m spacy download en_core_web_sm

# run feature extraction (skip LLM to avoid API cost)
python feature_pipeline.py --skip-llm

# run on a subset for testing
python feature_pipeline.py --limit 500 --skip-llm

# generate UMAP visualisation
python visualise_embeddings.py
```

---

## Module 1: Sentiment Polarity

### Traditional method: VADER

VADER is a rule-based sentiment analyser designed for short informal text. It returns a compound score from -1 to +1.

**Known limitations on this dataset:**
- Long reviews: sentiment words get diluted across many sentences, pulling the compound score toward neutral
- Rating-text conflicts: some users intentionally give low stars to get developer attention while writing positive text (e.g. *"good app, low stars to get dev attention"* — 2 stars, VADER compound 0.82)

**Improvements applied:**
- Text truncation to 150 characters before scoring — VADER is most reliable on short text
- Widened classification thresholds from ±0.05 to ±0.1 to reduce neutral misclassification

### Combined sentiment scoring

To handle rating-text conflicts, a blended label is computed that combines the star rating and VADER compound score using dynamic weights based on review length:

| Review length | Rating weight | VADER weight | Rationale |
|--------------|---------------|--------------|-----------|
| < 20 chars | 90% | 10% | Text has almost no signal at this length |
| 20–100 chars | 70% | 30% | Rating is primary, text is supplementary |
| 100+ chars | 50% | 50% | VADER is more reliable on longer text |

The star rating is normalised from 1–5 to a −1 to +1 scale before blending.

**Results (28,653 reviews processed):**

| Label | Count | Share |
|-------|-------|-------|
| Positive | 16,581 | 57.9% |
| Negative | 11,053 | 38.6% |
| Neutral | 1,019 | 3.6% |

660 reviews (2.3%) show disagreement between the VADER text score and the combined label. These are cases where the user's written sentiment and their star rating conflict, and are likely the most ambiguous examples in the dataset.

| Disagreement type | Count |
|-------------------|-------|
| VADER negative → Combined neutral | 329 |
| VADER positive → Combined neutral | 260 |
| VADER negative → Combined positive | 56 |
| VADER positive → Combined negative | 15 |

**Remaining concern:** The combined label is still primarily driven by star rating (95% of reviews use the `rating` method). 3-star reviews (1,452 total) rely entirely on VADER text analysis, which has no reliable ground truth to validate against. LLM-based labelling is proposed as a next step to address this.

### LLM method: Claude zero-shot (proposed)

Claude would classify each review as positive / negative / neutral with a confidence score. Combined with the rating signal and VADER, a three-way majority vote would determine the final label. Reviews where all three disagree would be flagged as `ambiguous`. *Not yet implemented — pending API access.*

---

## Module 2: Subjectivity Scoring

### Traditional method: TextBlob

TextBlob returns a subjectivity score from 0 (fully objective) to 1 (fully subjective). This distinguishes factual bug reports from emotional opinions, which may have different downstream utility.

**Results:**

| Metric | Value |
|--------|-------|
| Mean subjectivity | 0.445 |
| Median subjectivity | 0.500 |
| Std deviation | 0.322 |

The roughly symmetric distribution around 0.5 indicates a balanced mix of objective and subjective reviews in the dataset.

### LLM method: Claude subjectivity scoring (proposed)

Claude would score subjectivity 0–1 and provide a one-sentence reasoning. More accurate on edge cases. *Not yet implemented — pending API access.*

---

## Module 3: Aspect Extraction

### Traditional method: spaCy noun chunks

spaCy extracts noun phrases from each review, identifying which product features are being discussed (e.g. *"login screen"*, *"customer service"*, *"battery life"*).

**Optimisations applied over naive noun chunk extraction:**

1. **Pronoun and generic word blacklist** — removes *"they"*, *"it"*, *"someone"*, *"everything"*, *"people"*, *"money"*, etc.
2. **Noun-only filter** — only keeps chunks where the head word has POS tag `NOUN` or `PROPN`
3. **Domain vocabulary boost** — app-specific terms like `"ui"`, `"crash"`, `"login"`, `"notification"` are always retained even if short
4. **Normalisation map** — merges variant forms: `"this app"`, `"the app"`, `"your app"` → `"app"`; plurals: `"videos"` → `"video"`, `"crashes"` → `"crash"`, etc.
5. **Possessive stripping** — removes leading possessives: `"my money"` → `"money"`

**Results:**

| Metric | Value |
|--------|-------|
| Mean aspects per review | 3.2 |
| Reviews with 0 aspects | 3,902 (13.6%) |
| Max aspects in one review | 27 |

Top 10 aspects across all reviews: `app`, `account`, `update`, `video`, `ad`, `login`, `support`, `music`, `screen`, `subscription`

### LLM method: Claude aspect + sentiment extraction (proposed)

Claude extracts aspects as `(aspect, sentiment)` pairs and can identify implicit aspects that spaCy misses (e.g. *"it takes forever to load"* → aspect: `performance`). *Not yet implemented — pending API access.*

---

## Module 4: Semantic Embeddings

Two embedding methods were implemented and compared using UMAP visualisation.

### Method A: TF-IDF + TruncatedSVD (LSA)

TF-IDF converts each review into a weighted word frequency vector. TruncatedSVD (LSA) then compresses the sparse matrix to dense vectors.

**Baseline issues identified:**
- Dense central ball in UMAP — most reviews collapsed to similar vectors
- App name tokens dominated the vocabulary, causing app-level clustering rather than sentiment clustering
- Plural/variant forms treated as unrelated words

**Optimisations applied (v2):**

| Optimisation | Details |
|-------------|---------|
| Lemmatization | spaCy lemmatizes all tokens before TF-IDF (e.g. `"crashes"` → `"crash"`) |
| App name filtering | Removes: `spotify`, `instagram`, `youtube`, `amazon`, `duolingo`, `uber`, `whatsapp`, `teams`, `twitter`, `chatgpt`, `netflix`, `google`, `apple`, `microsoft`, `meta` |
| Char-level n-grams | `analyzer="char_wb"`, 2–4 chars, merged with word n-grams — handles spelling variants |
| Dimensionality | Expanded from 100 → 200 SVD components |
| Stopword removal | spaCy stopwords removed during lemmatization |

**Effect of optimisation:** The dense central ball structure was partially resolved. App clustering reduced significantly after brand name filtering. Sentiment and rating separation improved but remained weaker than sentence-transformers.

### Method B: Sentence-Transformers (all-MiniLM-L6-v2)

Pre-trained transformer model that produces 384-dimensional dense vectors where semantically similar texts are geometrically close.

**Advantages over TF-IDF:**
- Understands that `"slow"`, `"laggy"`, and `"freezing"` describe the same concept
- Handles negation and context: `"not bad"` and `"bad"` produce different vectors
- Runs locally — no API cost

### Comparison results (UMAP 2D)

| Dimension | Sentence-Transformer | TF-IDF (optimised) |
|-----------|---------------------|---------------------|
| Sentiment separation | Clear regional separation | Partial separation; significant overlap |
| Rating separation | Visible 1★ / 5★ gradient | Improved over baseline; less clear than ST |
| App clustering | Moderate — some app-specific regions | Reduced after brand name filtering |

**Key finding:** Even after full optimisation, TF-IDF shows weaker sentiment and rating separation than sentence-transformers. This gap is structural — TF-IDF cannot understand semantic similarity between words — not an engineering problem. The optimisations were still valuable to confirm this and to reduce app bias in the vocabulary.

**Concern (both methods):** Both embedding spaces show some app-level clustering, meaning a downstream model trained on these embeddings may learn to distinguish apps rather than true sentiment. This should be addressed through app-stratified training or by including app identity as an explicit feature.

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total reviews processed | 28,653 |
| Source apps | 11 |
| Reviews with 0 aspects | 3,902 (13.6%) |
| Sentiment: positive | 16,581 (57.9%) |
| Sentiment: negative | 11,053 (38.6%) |
| Sentiment: neutral | 1,019 (3.6%) |
| Rating-text disagreements | 660 (2.3%) |
| Mean subjectivity | 0.445 |
| Embedding dimensions (ST) | 384 |
| Embedding dimensions (TF-IDF) | 200 |

---

## Module Summary & Tradeoffs

| Module | Signal | Trust level | Best used for | Tends to fail when |
|--------|--------|-------------|---------------|--------------------|
| Sentiment (rating) | Star rating | High | Scalable weak labels across the full dataset | User deliberately misuses stars (e.g. low stars to get attention, test reviews) |
| Sentiment (VADER) | Text compound score | Medium | Enriching or challenging the rating signal; long reviews where text is informative | Short reviews (<20 chars); sarcasm; domain-specific language; rating-text conflicts |
| Sentiment (combined) | Rating + VADER blend | High for extremes, lower for 3★ | General-purpose sentiment label for downstream modelling | 3-star reviews, which have no strong anchor signal from either source |
| Subjectivity (TextBlob) | Objectivity 0–1 | Medium | Distinguishing bug reports from emotional opinions | Mixed reviews; short reviews |
| Aspect extraction (spaCy) | Noun chunk phrases | Medium | Identifying which product features are mentioned; topic frequency analysis | Implicit aspects; short reviews; slang |
| Embeddings (TF-IDF) | Weighted word frequencies | Low-medium | Lightweight baseline; interpretable | Semantic similarity across word variants; very short reviews |
| Embeddings (sentence-transformers) | Dense semantic vectors | High | Semantic clustering; similarity search; downstream classification | App-level bias if not corrected |

---

## Signal Failure Modes

Understanding where each signal breaks down is as important as knowing where it works. These are the failure patterns observed on this dataset.

### Star rating as sentiment proxy
- **Intentional rating manipulation:** users give 1 star specifically to attract developer attention while the review text is positive. 56 such cases were identified in this dataset.
- **Generic reviews:** ~25% of reviews are under 10 characters (e.g. "good", "👍", "TRASH"). For these, the rating is essentially the only signal — the text adds nothing.
- **Cultural variance:** different user populations may interpret the star scale differently.

### VADER text sentiment
- **Long reviews:** sentiment signal dilutes across many sentences. A 300-word review that starts with a complaint and ends with praise may score near neutral.
- **Sarcasm and irony:** "Oh great, another update that breaks everything" scores as positive because of "great."
- **Domain-specific negativity:** terms like "crash", "bug", "freeze" carry strong negative signal in the app context but may not be weighted heavily in VADER's general-purpose lexicon.
- **Short reviews:** VADER on a single word like "informative" returns compound 0.0 — completely uninformative.

### TextBlob subjectivity
- **Mixed reviews:** "The app crashes every login but the UI is beautiful" — TextBlob scores this as medium subjectivity, which is accurate but not actionable.
- **Short reviews:** single-word reviews score 0.0 subjectivity regardless of content.

### spaCy aspect extraction
- **Implicit aspects:** "it takes forever to load" contains no explicit aspect noun, but clearly refers to performance. spaCy returns nothing useful here.
- **Coverage gaps:** 13.6% of reviews yield zero meaningful aspects, mostly because the text is too short or generic.

### Embeddings (both methods)
- **App-level clustering:** both methods show app-specific clustering in UMAP. A downstream model trained on these embeddings may learn to recognise the app rather than the sentiment — a form of shortcut learning that would generalise poorly to new apps.
- **Short review collapse:** very short reviews cluster tightly regardless of meaning because there is almost no semantic content to differentiate them.

---

## Recommended v1 Production Direction

If designing a practical first version of a production sentiment pipeline, the recommendation based on this work is as follows.

### What to keep

**Star rating as the primary sentiment anchor.** Ratings are available at scale, require no computation, and are correct for the clear majority of cases. This alone covers ~95% of the data with high confidence.

**VADER as a challenge signal, not a replacement.** Use VADER to identify cases where the text contradicts the rating. These disagreement cases (2.3% of this dataset) are worth flagging explicitly rather than forcing a single label — they represent genuine ambiguity in the data.

**Sentence-transformer embeddings.** TF-IDF was a useful baseline to confirm that the performance gap with sentence-transformers is structural rather than tunable. For production, sentence-transformers (all-MiniLM-L6-v2 or similar) should be the default. They run locally with no API cost and produce meaningfully better semantic structure.

**spaCy aspect extraction with domain vocabulary.** Sufficient for identifying which product features appear most frequently in reviews. Should be combined with a maintained domain vocabulary list that evolves as the product changes.

**Explicit ambiguity flagging.** Rather than forcing every review into a binary label, the production pipeline should maintain an `ambiguous` category for reviews where signals conflict. These reviews should be treated cautiously in downstream modelling — either excluded from training, downweighted, or reserved for a manually-reviewed validation set.

### What to treat cautiously

**TextBlob subjectivity as a standalone signal.** Useful as a secondary filter, but not reliable enough to use as a primary feature without validation.

**TF-IDF embeddings in production.** Useful for lightweight search or keyword-frequency analysis, but not recommended as the primary embedding method for sentiment modelling.

### What to address before scaling

**App-level clustering bias.** Both embedding methods show app-specific clustering. Before training any downstream model, app-stratified sampling or explicit app identity features should be incorporated to prevent shortcut learning.

**LLM-based labelling for ambiguous cases.** For the ~5% of genuinely ambiguous reviews, an LLM classifier would provide a third semantic signal. The recommended approach is not majority voting to force a label, but to use LLM output as additional evidence — if rating, VADER, and LLM all disagree, flag the review as ambiguous rather than forcing a label.

### Summary

| Component | Production recommendation |
|-----------|--------------------------|
| Sentiment label | Rating as primary anchor; VADER as challenge signal; flag disagreements explicitly |
| Subjectivity | Include as secondary feature; do not rely on as primary signal |
| Aspect extraction | spaCy with domain vocabulary; plan for ongoing maintenance |
| Embeddings | Sentence-transformers; address app bias before model training |
| Ambiguous reviews | Flag explicitly; do not force into a label |
| LLM integration | Valuable for edge cases; not a wholesale replacement for rating-based labelling |

---

## Proposed Next Steps

1. **LLM labelling for ambiguous cases** — run Claude zero-shot sentiment classification on the 660 disagreement cases and 1,452 three-star reviews. Use as a third signal to understand disagreement patterns rather than to override existing labels. Requires API access.
2. **LLM aspect extraction comparison** — compare Claude's implicit aspect extraction against spaCy on a sample of 200–300 reviews to quantify the coverage gap.
3. **Address app clustering bias** — explore app-stratified sampling or explicit app identity features to prevent downstream models from learning app shortcuts.
4. **Manually review a small seed set** — hand-label 100–200 ambiguous reviews to understand what the disagreement patterns actually represent. This would provide ground truth to validate all three signal sources against.

---

## LLM Labelling Experiment (Ambiguous Reviews)

### Background

The combined sentiment scoring (rating + VADER blend) leaves two categories of reviews with uncertain labels:

- **Rating-text disagreements** (~1,326 reviews): where the VADER text score and the star rating point in different directions
- **3-star reviews** (~2,937 reviews): no strong anchor signal from either source

To address this, we ran an experiment using Claude as a third signal and implemented three-way majority voting (rating signal + VADER + LLM) to produce a final label.

### Method

For each ambiguous review:
1. Call Claude to classify sentiment as positive / negative / neutral
2. Run majority vote across three signals — if 2+ agree, that becomes the final label
3. If all three disagree, mark as `ambiguous` rather than forcing a label

### Results (50-review pilot)

| Final label | Count | % |
|-------------|-------|---|
| negative | 33 | 66% |
| positive | 17 | 34% |

17 out of 50 reviews (34%) were reclassified compared to the combined label alone, suggesting the three-way vote adds meaningful signal on true conflict cases.

### Limitations Observed

**LLM bias toward negative language:** Claude tended to classify high-star reviews as negative when they contained complaint language, even when the user's overall sentiment was positive (e.g. a 5-star review mentioning a specific bug was classified as negative). This suggests Claude is sensitive to the presence of negative words rather than judging overall satisfaction level.

**Non-standard labels:** Claude occasionally returned "mixed" instead of the required positive/negative/neutral, requiring additional handling.

**Fundamental ambiguity:** Some disagreement between signals reflects genuine ambiguity in the data rather than a system error — users who give high stars while complaining, or low stars while writing positive text, are inherently difficult to label reliably.

### Decision

Rather than running the full 3,661 reviews through LLM labelling, we chose to:

- Keep the combined label (rating + VADER blend) as the primary label for downstream use
- Explicitly flag disagreement cases as `ambiguous` in the dataset rather than forcing a corrected label
- Treat the LLM experiment as a validation step rather than a production labelling pipeline

This aligns with the principle that disagreement between signals often reflects real data ambiguity, and that forcing a label can introduce more noise than it removes. A small manually reviewed seed set would be a more reliable next step for resolving these cases.
