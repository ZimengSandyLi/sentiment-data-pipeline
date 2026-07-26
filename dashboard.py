"""
Review Sentiment Dashboard
===========================
Streamlit dashboard for the Google Play Review Intelligence Pipeline.
Connects to pipeline.db and features.db to visualise sentiment trends,
aspect frequency, data quality, and ingestion health.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from collections import Counter
import os

# ── Config ─────────────────────────────────────────────────────
PIPELINE_DB  = "pipeline.db"
FEATURES_DB  = "features.db"

st.set_page_config(
    page_title  = "Review Intelligence Dashboard",
    page_icon   = "📱",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Helpers ────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # cache for 5 minutes
def load_reviews():
    if not os.path.exists(PIPELINE_DB):
        return pd.DataFrame()
    conn = sqlite3.connect(PIPELINE_DB)
    df = pd.read_sql("""
        SELECT r.review_id, a.app_name, r.rating, r.text,
               r.thumbs_up, r.date, r.scraped_at
        FROM reviews r
        JOIN apps a ON r.app_id = a.app_id
    """, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


@st.cache_data(ttl=300)
def load_features():
    if not os.path.exists(FEATURES_DB):
        return pd.DataFrame()
    conn = sqlite3.connect(FEATURES_DB)
    df = pd.read_sql("""
        SELECT review_id, app_name, rating, text,
               vader_label, vader_compound,
               combined_label, combined_score,
               textblob_subjectivity,
               spacy_aspects, spacy_aspect_count,
               processed_at
        FROM features
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_priority_scores():
    """Load pre-computed issue priority scores if available."""
    score_path = "issue_priority.csv"
    if not os.path.exists(score_path):
        return pd.DataFrame()
    df = pd.read_csv(score_path)
    return df


@st.cache_data(ttl=300)
def load_ingestion_runs():
    if not os.path.exists(PIPELINE_DB):
        return pd.DataFrame()
    conn = sqlite3.connect(PIPELINE_DB)
    df = pd.read_sql("""
        SELECT i.id, a.app_name, i.started_at,
               i.completed_at, i.reviews_collected, i.status
        FROM ingestion_runs i
        JOIN apps a ON i.app_id = a.app_id
        ORDER BY i.started_at DESC
    """, conn)
    conn.close()
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    return df


def extract_all_aspects(features_df):
    """Flatten all spacy_aspects JSON arrays into a list."""
    all_aspects = []
    for val in features_df['spacy_aspects'].dropna():
        try:
            all_aspects.extend(json.loads(val))
        except Exception:
            pass
    return all_aspects


SENTIMENT_COLORS = {
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral":  "#9E9E9E",
}

# ── Sidebar ────────────────────────────────────────────────────

st.sidebar.title("📱 Review Intelligence")
st.sidebar.caption("Google Play Feedback Analytics")

reviews_df  = load_reviews()
features_df = load_features()
runs_df     = load_ingestion_runs()
priority_df = load_priority_scores()

if reviews_df.empty:
    st.error("pipeline.db not found. Run pipeline.py first.")
    st.stop()

all_apps = sorted(reviews_df['app_name'].unique())
selected_apps = st.sidebar.multiselect(
    "Filter by app", all_apps, default=all_apps
)

page = st.sidebar.radio("View", [
    "Overview",
    "Sentiment Analysis",
    "Aspect Explorer",
    "Issue Priority",
    "Data Quality",
    "Pipeline Health",
])

# apply app filter
rev = reviews_df[reviews_df['app_name'].isin(selected_apps)]
feat = features_df[features_df['app_name'].isin(selected_apps)] if not features_df.empty else pd.DataFrame()


# ── Overview ──────────────────────────────────────────────────

if page == "Overview":
    st.title("Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{len(rev):,}")
    col2.metric("Apps", len(selected_apps))
    col3.metric("Ingestion Runs", len(runs_df))

    if not feat.empty:
        neg_pct = (feat['combined_label'] == 'negative').mean() * 100
        col4.metric("Negative Reviews", f"{neg_pct:.1f}%")
    else:
        col4.metric("Negative Reviews", "Run pipeline")

    st.subheader("Reviews per App")
    app_counts = rev['app_name'].value_counts().reset_index()
    app_counts.columns = ['app_name', 'count']
    fig = px.bar(app_counts, x='app_name', y='count',
                 color='count', color_continuous_scale='Blues',
                 labels={'app_name': 'App', 'count': 'Reviews'})
    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                      plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rating Distribution")
    rating_counts = rev['rating'].value_counts().sort_index()
    fig2 = px.bar(x=rating_counts.index, y=rating_counts.values,
                  labels={'x': 'Stars', 'y': 'Count'},
                  color=rating_counts.index,
                  color_continuous_scale=['#D32F2F','#FF7043','#FFC107','#66BB6A','#1B5E20'])
    fig2.update_layout(coloraxis_showscale=False,
                       plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig2, use_container_width=True)

    # ── Time trend ──────────────────────────────────────────────
    st.subheader("Sentiment Trend Over Time")
    st.caption("Monthly breakdown of positive vs negative reviews — useful for spotting whether a version update improved or hurt user perception.")

    if not feat.empty:
        # merge date from reviews into features
        rev_dates = rev[['review_id', 'date']].dropna(subset=['date'])
        feat_dated = feat.merge(rev_dates, on='review_id', how='left')
        feat_dated = feat_dated.dropna(subset=['date'])

        if len(feat_dated) > 0:
            feat_dated['month'] = feat_dated['date'].dt.to_period('M').astype(str)

            granularity = st.radio(
                "Granularity", ["Monthly", "Weekly"], horizontal=True, key="trend_gran"
            )
            if granularity == "Weekly":
                feat_dated['period'] = feat_dated['date'].dt.to_period('W').astype(str)
            else:
                feat_dated['period'] = feat_dated['month']

            trend = (feat_dated
                     .groupby(['period', 'combined_label'])
                     .size()
                     .reset_index(name='count'))
            trend_total = trend.groupby('period')['count'].transform('sum')
            trend['pct'] = trend['count'] / trend_total * 100

            view_mode = st.radio(
                "View as", ["Percentage", "Count"], horizontal=True, key="trend_mode"
            )
            y_col = 'pct' if view_mode == "Percentage" else 'count'
            y_label = "% of reviews" if view_mode == "Percentage" else "Review count"

            fig_trend = px.bar(
                trend, x='period', y=y_col, color='combined_label',
                color_discrete_map=SENTIMENT_COLORS,
                barmode='stack',
                labels={'period': 'Period', y_col: y_label, 'combined_label': 'Sentiment'},
            )
            fig_trend.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                xaxis_tickangle=-30, legend_title='Sentiment',
                height=380,
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # Negative rate line chart — easier to spot trend direction
            st.caption("Negative review rate over time — a rising line after a release date may indicate a UX regression.")
            neg_trend = (feat_dated[feat_dated['combined_label'] == 'negative']
                         .groupby('period')
                         .size()
                         .reset_index(name='negative'))
            total_trend = feat_dated.groupby('period').size().reset_index(name='total')
            neg_rate = neg_trend.merge(total_trend, on='period')
            neg_rate['negative_pct'] = neg_rate['negative'] / neg_rate['total'] * 100

            fig_line = px.line(
                neg_rate, x='period', y='negative_pct',
                labels={'period': 'Period', 'negative_pct': '% Negative'},
                markers=True,
                color_discrete_sequence=['#F44336'],
            )
            fig_line.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                xaxis_tickangle=-30, height=300,
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No dated reviews found in features.db. Re-run feature_pipeline.py to include date data.")
    else:
        st.info("Run feature_pipeline.py to see sentiment trends.")


# ── Sentiment Analysis ─────────────────────────────────────────

elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis")

    if feat.empty:
        st.warning("features.db not found. Run feature_pipeline.py first.")
        st.stop()

    # Overall sentiment breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall Sentiment")
        label_counts = feat['combined_label'].value_counts().reset_index()
        label_counts.columns = ['label', 'count']
        fig = px.pie(label_counts, names='label', values='count',
                     color='label', color_discrete_map=SENTIMENT_COLORS,
                     hole=0.4)
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sentiment by App")
        if not feat.empty:
            pivot = feat.groupby(['app_name','combined_label']).size().unstack(fill_value=0)
            pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
            fig2 = px.bar(pivot_pct.reset_index(), x='app_name',
                          y=[c for c in ['positive','neutral','negative'] if c in pivot_pct.columns],
                          color_discrete_map=SENTIMENT_COLORS,
                          labels={'value':'%', 'app_name':'App', 'variable':'Sentiment'},
                          barmode='stack')
            fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                               legend_title='Sentiment')
            st.plotly_chart(fig2, use_container_width=True)

    # Disagreement cases
    st.subheader("Rating-Text Disagreements")
    st.caption("Reviews where the star rating and VADER text score point in different directions — these are the most ambiguous examples in the dataset.")

    if 'vader_label' in feat.columns:
        def rating_to_sentiment(r):
            if r <= 2: return 'negative'
            if r >= 4: return 'positive'
            return 'neutral'
        feat['rating_sentiment'] = feat['rating'].apply(rating_to_sentiment)
        disagree = feat[feat['vader_label'] != feat['combined_label']]
        col1, col2, col3 = st.columns(3)
        col1.metric("Disagreements", f"{len(disagree):,}")
        col2.metric("% of Dataset", f"{len(disagree)/len(feat)*100:.1f}%")
        col3.metric("Rating-text conflicts", f"{len(disagree[disagree['combined_label'] != disagree['rating_sentiment']]):,}")

        if len(disagree) > 0:
            st.dataframe(
                disagree[['app_name','rating','vader_label','combined_label','text']]
                .head(20)
                .rename(columns={
                    'app_name':'App', 'rating':'Stars',
                    'vader_label':'VADER', 'combined_label':'Combined',
                    'text':'Review text'
                }),
                use_container_width=True,
            )

    # Subjectivity
    st.subheader("Subjectivity Distribution")
    st.caption("0 = objective fact (e.g. 'app crashes on login')  |  1 = subjective opinion (e.g. 'I love this app')")
    fig3 = px.histogram(feat, x='textblob_subjectivity', nbins=40,
                        color_discrete_sequence=['#4C9BE8'],
                        labels={'textblob_subjectivity': 'Subjectivity score'})
    fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig3, use_container_width=True)


# ── Aspect Explorer ────────────────────────────────────────────

elif page == "Aspect Explorer":
    st.title("Aspect Explorer")
    st.caption("Which product features are users talking about most?")

    if feat.empty:
        st.warning("features.db not found. Run feature_pipeline.py first.")
        st.stop()

    all_aspects = extract_all_aspects(feat)
    aspect_counts = Counter(all_aspects)
    top_n = st.slider("Show top N aspects", 10, 50, 20)

    top_aspects = pd.DataFrame(
        aspect_counts.most_common(top_n),
        columns=['Aspect', 'Mentions']
    )

    fig = px.bar(top_aspects, x='Mentions', y='Aspect', orientation='h',
                 color='Mentions', color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder':'total ascending'},
                      plot_bgcolor='white', paper_bgcolor='white',
                      coloraxis_showscale=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Aspect x sentiment heatmap
    st.subheader("Top Aspects by Sentiment")
    st.caption("For each frequently-mentioned aspect, what proportion of those reviews are negative?")

    top_aspect_names = [a for a, _ in aspect_counts.most_common(15)]
    rows = []
    for aspect in top_aspect_names:
        mask = feat['spacy_aspects'].apply(
            lambda x: aspect in (json.loads(x) if isinstance(x, str) else [])
        )
        subset = feat[mask]
        if len(subset) == 0:
            continue
        for label in ['positive', 'neutral', 'negative']:
            rows.append({
                'Aspect': aspect,
                'Sentiment': label,
                'Count': (subset['combined_label'] == label).sum(),
                'Pct': (subset['combined_label'] == label).mean() * 100,
            })

    if rows:
        aspect_sent_df = pd.DataFrame(rows)
        fig2 = px.bar(aspect_sent_df, x='Aspect', y='Pct', color='Sentiment',
                      color_discrete_map=SENTIMENT_COLORS, barmode='stack',
                      labels={'Pct': '%', 'Aspect': 'Product Feature'},
                      height=400)
        fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                           xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)


# ── Issue Priority ─────────────────────────────────────────────

elif page == "Issue Priority":
    st.title("Issue Priority Backlog")
    st.caption(
        "Ranked product issues combining sentiment severity (40%), "
        "review volume (35%), and recency (25%). "
        "Run `python prioritize.py` to refresh."
    )

    if priority_df.empty:
        st.warning("issue_priority.csv not found. Run: `python prioritize.py` first.")
        st.stop()

    # filter by selected apps
    prio = priority_df.copy()
    if selected_apps:
        prio = prio[prio['app_name'].isin(selected_apps)]

    if prio.empty:
        st.info("No priority data for selected apps.")
        st.stop()

    # top-level metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Issues tracked", f"{len(prio):,}")
    top_app = prio.groupby('app_name')['priority_score'].mean().idxmax()
    col2.metric("Highest-risk app", top_app)
    col3.metric("Avg negative rate", f"{prio['negative_rate'].mean()*100:.1f}%")

    # top issues table
    st.subheader("Top Issues")
    top_n = st.slider("Show top N", 10, 50, 20, key="prio_top_n")

    display = prio.head(top_n)[['rank','app_name','aspect','priority_score',
                                 'negative_rate','mention_count','avg_rating']].copy()
    display['negative_rate'] = (display['negative_rate'] * 100).round(1).astype(str) + '%'
    display['priority_score'] = display['priority_score'].round(3)
    display['avg_rating'] = display['avg_rating'].round(2)
    display = display.rename(columns={
        'rank': 'Rank', 'app_name': 'App', 'aspect': 'Issue',
        'priority_score': 'Score', 'negative_rate': 'Neg%',
        'mention_count': 'Mentions', 'avg_rating': 'Avg ★'
    })
    st.dataframe(display.set_index('Rank'), use_container_width=True)

    # priority score bar chart
    st.subheader("Priority Score by Issue")
    top_issues = prio.head(top_n).copy()
    top_issues['label'] = top_issues['app_name'] + ' / ' + top_issues['aspect']
    fig = px.bar(
        top_issues.sort_values('priority_score'),
        x='priority_score', y='label', orientation='h',
        color='negative_rate',
        color_continuous_scale='RdYlGn_r',
        labels={'priority_score': 'Priority Score', 'label': '',
                'negative_rate': 'Neg rate'},
        height=max(400, top_n * 22),
    )
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                      yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # per-app breakdown
    st.subheader("Top Issue per App")
    top_per_app = (prio.groupby('app_name')
                   .apply(lambda g: g.nlargest(1, 'priority_score'))
                   .reset_index(drop=True)
                   [['app_name','aspect','priority_score','negative_rate','mention_count']])
    top_per_app['negative_rate'] = (top_per_app['negative_rate']*100).round(1).astype(str) + '%'
    top_per_app['priority_score'] = top_per_app['priority_score'].round(3)
    st.dataframe(
        top_per_app.rename(columns={
            'app_name':'App','aspect':'Top Issue',
            'priority_score':'Score','negative_rate':'Neg%',
            'mention_count':'Mentions'
        }).set_index('App'),
        use_container_width=True,
    )


# ── Data Quality ───────────────────────────────────────────────

elif page == "Data Quality":
    st.title("Data Quality")

    col1, col2, col3, col4 = st.columns(4)

    short_10 = (rev['text'].fillna('').apply(len) < 10).sum()
    short_20 = (rev['text'].fillna('').apply(len) < 20).sum()
    dup_text = rev.duplicated(subset=['app_name', 'text']).sum()
    empty    = rev['text'].isna().sum()

    col1.metric("Reviews < 10 chars", f"{short_10:,}", f"{short_10/len(rev)*100:.1f}%")
    col2.metric("Reviews < 20 chars", f"{short_20:,}", f"{short_20/len(rev)*100:.1f}%")
    col3.metric("Duplicate texts",    f"{dup_text:,}", f"{dup_text/len(rev)*100:.1f}%")
    col4.metric("Missing text",       f"{empty:,}")

    st.subheader("Review Text Length Distribution")
    rev['text_len'] = rev['text'].fillna('').apply(len)
    fig = px.histogram(rev, x='text_len', nbins=60,
                       range_x=[0, 300],
                       labels={'text_len': 'Characters (clipped at 300)'},
                       color_discrete_sequence=['#4C9BE8'])
    fig.add_vline(x=rev['text_len'].median(), line_dash='dash',
                  line_color='orange',
                  annotation_text=f"Median: {int(rev['text_len'].median())} chars")
    fig.add_vline(x=10, line_dash='dot', line_color='red',
                  annotation_text='10-char threshold')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most Common Duplicate Texts")
    dup_df = (rev[rev.duplicated(subset=['app_name','text'], keep=False)]
              .groupby(['app_name','text'])
              .size()
              .reset_index(name='count')
              .sort_values('count', ascending=False)
              .head(10))
    if len(dup_df):
        st.dataframe(dup_df.rename(columns={'app_name':'App','text':'Review text','count':'Occurrences'}),
                     use_container_width=True)


# ── Pipeline Health ────────────────────────────────────────────

elif page == "Pipeline Health":
    st.title("Pipeline Health")

    col1, col2, col3 = st.columns(3)
    success = (runs_df['status'] == 'success').sum()
    failed  = (runs_df['status'] == 'failed').sum()
    col1.metric("Total runs",     len(runs_df))
    col2.metric("Successful",     success)
    col3.metric("Failed",         failed,
                delta=f"{failed/len(runs_df)*100:.1f}% failure rate" if len(runs_df) else None,
                delta_color="inverse")

    st.subheader("Recent Ingestion Runs")
    display_runs = runs_df.head(30).copy()
    display_runs['started_at'] = display_runs['started_at'].dt.strftime('%Y-%m-%d %H:%M')
    display_runs['status'] = display_runs['status'].map({
        'success': '✅ success',
        'failed':  '❌ failed',
        'in_progress': '⏳ in progress',
    }).fillna(display_runs['status'])
    st.dataframe(
        display_runs[['app_name','started_at','reviews_collected','status']]
        .rename(columns={'app_name':'App','started_at':'Started at',
                         'reviews_collected':'Reviews collected','status':'Status'}),
        use_container_width=True,
    )

    st.subheader("Reviews Collected per Run")
    fig = px.bar(runs_df.sort_values('started_at').tail(50),
                 x='started_at', y='reviews_collected', color='app_name',
                 labels={'started_at':'Run time','reviews_collected':'Reviews collected',
                         'app_name':'App'},
                 height=350)
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                      xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
