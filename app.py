"""
=====================================================================
  T9.3  ·  Daily Tech News Briefing  —  Streamlit deployment (v3)
=====================================================================

  Dark glassmorphism rewrite.  Forces dark theme via .streamlit/config.toml
  so it looks identical regardless of the user's OS preference.
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime

import streamlit as st
import pandas as pd

warnings.filterwarnings("ignore")

from main import (
    CATEGORIES, ALL_FEEDS, INDIAN_NEWS_FEEDS, TECH_FEEDS, N_ARTICLES,
    fetch_rss_articles, classify_batch, summarize_article,
)

st.set_page_config(
    page_title="Daily Tech News Briefing",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

CAT_COLOR = {
    "Politics":      "#FF5577", 
    "Sports":        "#22D3A0",   
    "Technology":    "#00D4FF",   
    "Business":      "#FFB347",   
    "Entertainment": "#C77DFF",  
}
CAT_ICON = {
    "Politics":      "▣",
    "Sports":        "◆",
    "Technology":    "▲",
    "Business":      "●",
    "Entertainment": "★",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ===== ROOT ===== */
html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #E2E8F0;
}
.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(0,212,255,0.08) 0, transparent 40%),
        radial-gradient(circle at 100% 0%, rgba(34,211,160,0.06) 0, transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(199,125,255,0.05) 0, transparent 50%),
        #0A0F1E;
}

/* hide default chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 1rem !important;
    max-width: 1400px;
}

/* ===== HERO ===== */
.hero {
    background: linear-gradient(120deg, rgba(0,102,204,0.85) 0%, rgba(0,212,255,0.7) 50%, rgba(34,211,160,0.7) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
    color: white;
    box-shadow: 0 20px 60px -20px rgba(0,212,255,0.4);
    position: relative; overflow: hidden;
    backdrop-filter: blur(20px);
}
.hero::before {
    content:""; position:absolute; right:-80px; top:-80px;
    width:280px; height:280px; border-radius:50%;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
}
.hero::after {
    content:""; position:absolute; left:30%; bottom:-100px;
    width:200px; height:200px; border-radius:50%;
    background: radial-gradient(circle, rgba(199,125,255,0.2) 0%, transparent 70%);
}
.hero h1 {
    font-size: 2.0rem; font-weight: 800; margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
    background: linear-gradient(180deg, #ffffff 0%, #cce4ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { margin: 0; opacity: 0.92; font-size: 0.95rem; font-weight: 400; }
.hero p b { color: #fff; }
.hero .badge {
    display:inline-block; padding:3px 12px; border-radius:20px;
    background: rgba(255,255,255,0.15); font-size:0.68rem; font-weight:700;
    letter-spacing: 1.2px; margin-bottom: 0.55rem;
    backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.2);
}

/* ===== METRIC TILES ===== */
.metric-row { display: flex; gap: 12px; margin-bottom: 1rem; flex-wrap: wrap; }
.metric-tile {
    flex: 1; min-width: 180px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 18px;
    backdrop-filter: blur(10px);
    transition: all 0.2s ease;
    position: relative; overflow: hidden;
}
.metric-tile::before {
    content:""; position:absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
}
.metric-tile:hover {
    transform: translateY(-2px);
    background: rgba(255,255,255,0.06);
    border-color: rgba(0,212,255,0.3);
    box-shadow: 0 12px 24px -8px rgba(0,212,255,0.2);
}
.metric-tile .label {
    font-size: 0.68rem; color: #94A3B8; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px;
}
.metric-tile .value {
    font-size: 1.6rem; font-weight: 800; color: #F1F5F9;
    margin-top: 4px; line-height: 1.1;
}
.metric-tile .sub { font-size: 0.74rem; color: #64748B; margin-top: 3px; }

/* ===== ARTICLE CARDS ===== */
.article-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid var(--accent, #00D4FF);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.65rem;
    transition: all 0.18s ease;
    backdrop-filter: blur(10px);
}
.article-card:hover {
    transform: translateX(3px);
    background: rgba(255,255,255,0.05);
    border-color: var(--accent, #00D4FF);
    box-shadow: 0 8px 24px -8px var(--accent-glow, rgba(0,212,255,0.3));
}
.article-title {
    font-size: 1.0rem; font-weight: 600; color: #F1F5F9;
    line-height: 1.4; margin-bottom: 0.4rem;
}
.article-title a { color: inherit; text-decoration: none; }
.article-title a:hover { color: var(--accent, #00D4FF); }
.meta-row {
    display:flex; gap:6px; flex-wrap:wrap; align-items:center;
    font-size: 0.72rem; color:#64748B; margin-top: 0.3rem;
}
.pill {
    display:inline-flex; align-items:center; gap:4px;
    padding: 3px 10px; border-radius: 999px;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.2px;
    border: 1px solid transparent;
}
.pill-cat {
    background: var(--accent-soft, rgba(0,212,255,0.12));
    color: var(--accent, #00D4FF);
    border-color: var(--accent-border, rgba(0,212,255,0.25));
}
.pill-src {
    background: rgba(148,163,184,0.1);
    color: #CBD5E1;
    border-color: rgba(148,163,184,0.2);
}
.pill-conf {
    background: rgba(34,211,160,0.12); color: #4ADE80;
    border-color: rgba(34,211,160,0.25);
}
.pill-conf-low {
    background: rgba(255,179,71,0.12); color: #FFB347;
    border-color: rgba(255,179,71,0.25);
}
.pill-time { color: #64748B; font-size: 0.7rem; }

.summary-box {
    background: rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid var(--accent, #00D4FF);
    border-radius: 8px;
    padding: 12px 16px;
    color: #CBD5E1;
    font-size: 0.92rem; line-height: 1.6;
    margin-top: 0.5rem;
}

/* ===== SECTION HEADERS ===== */
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #F1F5F9;
    margin: 1.2rem 0 0.6rem 0;
    display: flex; align-items: center; gap: 10px;
}
.section-header .accent-bar {
    width: 4px; height: 20px; border-radius: 2px;
    background: linear-gradient(180deg, #00D4FF, #22D3A0);
    box-shadow: 0 0 12px rgba(0,212,255,0.6);
}
.section-header .count {
    color:#64748B; font-weight:500; font-size:0.85rem;
    margin-left: auto;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab-list"] button {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #94A3B8 !important;
    transition: all 0.18s ease !important;
    border: none !important;
}
.stTabs [data-baseweb="tab-list"] button:hover {
    color: #F1F5F9 !important;
    background: rgba(255,255,255,0.04) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(34,211,160,0.18)) !important;
    color: #F1F5F9 !important;
    box-shadow: 0 0 0 1px rgba(0,212,255,0.3) inset;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem !important; }

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1729 0%, #0A0F1E 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] > div {
    background: transparent !important;
}
section[data-testid="stSidebar"] h2 {
    font-size: 1.0rem !important; color: #F1F5F9 !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] h3 {
    font-size: 0.85rem !important; color: #94A3B8 !important;
    text-transform: uppercase; letter-spacing: 1px; font-weight: 700 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div[data-testid="stMarkdown"] {
    color: #CBD5E1 !important;
}
/* sidebar inputs */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="input"] {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #F1F5F9 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.04) !important;
}
/* sidebar caption */
section[data-testid="stSidebar"] .stCaption {
    color: #64748B !important;
}

/* radio + multiselect + slider tweaks for dark */
.stRadio label, .stMultiSelect label, .stSlider label, .stTextInput label,
.stToggle label { color: #CBD5E1 !important; font-weight: 500 !important; }

/* expander */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    color: #CBD5E1 !important;
    font-weight: 500 !important;
}
.streamlit-expanderHeader:hover { background: rgba(255,255,255,0.05) !important; }

/* dataframe styling */
.stDataFrame { border-radius: 10px; overflow: hidden;
               border: 1px solid rgba(255,255,255,0.08); }

/* button */
.stButton button {
    background: linear-gradient(135deg, #0066CC, #00D4FF) !important;
    border: none !important; border-radius: 10px !important;
    color: white !important; font-weight: 600 !important;
    box-shadow: 0 4px 16px -4px rgba(0,212,255,0.5) !important;
    transition: all 0.2s ease !important;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px -4px rgba(0,212,255,0.7) !important;
}

/* spinner */
.stSpinner > div { border-top-color: #00D4FF !important; }

/* alerts */
.stAlert { background: rgba(255,255,255,0.04) !important;
           border: 1px solid rgba(255,255,255,0.1) !important;
           color: #CBD5E1 !important; }

/* divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* footer caption */
.footer-line { text-align:center; color:#475569; font-size:0.74rem;
               margin-top: 1.5rem; padding: 1rem 0;
               border-top: 1px solid rgba(255,255,255,0.05); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <span class="badge">PROJECT T9.3 · TIER 1 · LIVE</span>
    <h1>📰 Daily Tech News Briefing</h1>
    <p>Latest 30 articles from Indian + Tech RSS feeds · classified with
       <b>BART-MNLI</b> zero-shot · summarised with <b>Gemini 1.5 Flash</b></p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️  Controls")

    api_key = st.text_input(
        "Gemini API key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Free key from aistudio.google.com",
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    feed_choice = st.radio(
        "Feed pack",
        options=["All (Indian + Tech)", "Indian only", "Tech only"],
        index=0,
    )
    if feed_choice == "Indian only":
        active_feeds = INDIAN_NEWS_FEEDS
    elif feed_choice == "Tech only":
        active_feeds = TECH_FEEDS
    else:
        active_feeds = ALL_FEEDS

    n_articles = st.slider("Articles to fetch", 10, 60, N_ARTICLES, step=5)

    st.markdown("---")
    st.markdown("### Filter")
    search_q = st.text_input("Search", "",
                             placeholder="title or source…")
    selected_cats = st.multiselect("Categories", CATEGORIES, default=CATEGORIES)

    st.markdown("---")
    show_summary = st.toggle("3-line AI summaries", value=bool(api_key))
    view_mode    = st.radio("View", ["Category tabs", "Grouped by source", "Flat feed"])

    st.markdown("---")
    if st.button("🔄  Refresh", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"⏱ {datetime.now():%d %b · %H:%M}")
    st.caption(f"📡 {len(active_feeds)} feeds active")

@st.cache_resource(show_spinner="🤖  Loading BART-MNLI…")
def warmup_classifier():
    from main import get_classifier
    get_classifier()
    return True

@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch(feed_keys: tuple, n: int):
    feeds = {k: ALL_FEEDS[k] for k in feed_keys}
    return fetch_rss_articles(feeds, limit_per_feed=max(3, n // len(feeds) + 2))[:n]

@st.cache_data(ttl=1800, show_spinner=False)
def cached_classify(texts: tuple):
    return classify_batch(list(texts))

@st.cache_data(ttl=1800)
def cached_summary(text: str) -> str:
    return summarize_article(text)

warmup_classifier()

with st.spinner("Pulling latest RSS feeds & classifying…"):
    articles = cached_fetch(tuple(active_feeds.keys()), n_articles)
    if not articles:
        st.error("No articles could be fetched — check your network / RSS URLs.")
        st.stop()
    texts = tuple(f"{a['title']}. {a['summary']}" for a in articles)
    cls   = cached_classify(texts)

df = pd.DataFrame(articles)
df["category"]   = [c["label"] for c in cls]
df["confidence"] = [c["score"] for c in cls]
df["text_full"]  = list(texts)

flt = df.copy()
if selected_cats:
    flt = flt[flt["category"].isin(selected_cats)]
if search_q:
    q = search_q.lower().strip()
    flt = flt[flt["title"].str.lower().str.contains(q, na=False) |
              flt["source"].str.lower().str.contains(q, na=False)]

top_cat = flt["category"].mode().iloc[0] if not flt.empty else "—"
avg_conf = flt["confidence"].mean() if not flt.empty else 0
sources_preview = ', '.join(flt['source'].unique()[:3]) if not flt.empty else '—'
sources_more = '…' if (not flt.empty and flt['source'].nunique() > 3) else ''
top_color = CAT_COLOR.get(top_cat, '#00D4FF')

st.markdown(f"""
<div class="metric-row">
    <div class="metric-tile">
        <div class="label">Articles shown</div>
        <div class="value">{len(flt)}</div>
        <div class="sub">{len(df)} fetched · {len(df)-len(flt)} filtered</div>
    </div>
    <div class="metric-tile">
        <div class="label">Sources</div>
        <div class="value">{flt['source'].nunique() if not flt.empty else 0}</div>
        <div class="sub">{sources_preview}{sources_more}</div>
    </div>
    <div class="metric-tile">
        <div class="label">Avg. confidence</div>
        <div class="value">{avg_conf:.2f}</div>
        <div class="sub">{'high quality' if avg_conf>0.7 else 'mixed'}</div>
    </div>
    <div class="metric-tile">
        <div class="label">Top category</div>
        <div class="value" style="color:{top_color}">
            {CAT_ICON.get(top_cat,'')} {top_cat}
        </div>
        <div class="sub">{(flt['category']==top_cat).sum()} articles</div>
    </div>
</div>
""", unsafe_allow_html=True)

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def render_card(row, with_summary: bool):
    color  = CAT_COLOR.get(row["category"], "#00D4FF")
    soft   = hex_to_rgba(color, 0.12)
    border = hex_to_rgba(color, 0.30)
    glow   = hex_to_rgba(color, 0.30)
    icon   = CAT_ICON.get(row["category"], "•")
    conf   = row["confidence"]
    conf_class = "pill-conf" if conf >= 0.65 else "pill-conf-low"

    st.markdown(f"""
    <div class="article-card" style="--accent:{color}; --accent-soft:{soft}; --accent-border:{border}; --accent-glow:{glow};">
        <div class="article-title">
            <a href="{row['link']}" target="_blank">{row['title']}</a>
        </div>
        <div class="meta-row">
            <span class="pill pill-cat">{icon} {row['category']}</span>
            <span class="pill pill-src">{row['source']}</span>
            <span class="pill {conf_class}">conf {conf:.2f}</span>
            <span class="pill-time">{row.get('published','')[:25]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if with_summary:
        with st.expander("✨ 3-line summary", expanded=False):
            with st.spinner(""):
                summ = cached_summary(row["text_full"])
            st.markdown(
                f"<div class='summary-box' style='--accent:{color};'>{summ}</div>",
                unsafe_allow_html=True
            )


if flt.empty:
    st.info("No articles match your current filters.")
    st.stop()

if view_mode == "Category tabs":
    counts = flt["category"].value_counts()
    tabs = st.tabs([f"{CAT_ICON[c]}  {c}  ({counts.get(c,0)})" for c in CATEGORIES])
    for tab, cat in zip(tabs, CATEGORIES):
        with tab:
            sub = flt[flt["category"] == cat].sort_values("confidence", ascending=False)
            if sub.empty:
                st.info(f"No articles classified as **{cat}** in this batch.")
                continue
            for _, row in sub.iterrows():
                render_card(row, show_summary)

elif view_mode == "Grouped by source":
    for src in sorted(flt["source"].unique()):
        sub = flt[flt["source"] == src].sort_values("confidence", ascending=False)
        st.markdown(f"""
        <div class="section-header">
            <span class="accent-bar"></span> {src}
            <span class="count">· {len(sub)} articles</span>
        </div>""", unsafe_allow_html=True)
        for _, row in sub.iterrows():
            render_card(row, show_summary)

else:    
    st.markdown('<div class="section-header"><span class="accent-bar"></span>'
                'All articles <span class="count">· sorted by confidence</span></div>',
                unsafe_allow_html=True)
    for _, row in flt.sort_values("confidence", ascending=False).iterrows():
        render_card(row, show_summary)

with st.expander(" Raw classified table", expanded=False):
    st.dataframe(
        flt[["source", "category", "confidence", "title", "link"]]
            .sort_values("confidence", ascending=False)
            .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0, max_value=1, format="%.2f"),
            "link": st.column_config.LinkColumn("link", display_text="open →"),
        },
    )

st.markdown(
    "<div class='footer-line'>Built for T9.3 · Tech News Tracker · "
    "BART-MNLI + Gemini 1.5 Flash + Streamlit</div>",
    unsafe_allow_html=True
)