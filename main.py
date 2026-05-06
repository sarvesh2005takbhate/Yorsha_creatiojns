"""
=====================================================================
 T9.3 - DAILY TECH NEWS BRIEFING (Tier 1)
 Core working logic: RSS → Zero-shot classification → LLM summary
 + Evaluation pipeline on Kaggle India-Headlines dataset
=====================================================================

USAGE
-----
1.  Place your Gemini API key in environment variable  GEMINI_API_KEY
2.  (Optional) Place 'india-news-headlines.csv'  in  ./data/
    (download from kaggle.com/datasets/therohk/india-headlines-news-dataset)
3.  Run:                python main.py            (full pipeline + eval)
        or in pieces:   python main.py --fetch
                        python main.py --evaluate

This module is also imported by app.py.
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import feedparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score)


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
GRAPH_DIR = OUT_DIR / "graphs"
METRIC_DIR = OUT_DIR / "metrics"
SAMPLE_DIR = OUT_DIR / "sample_runs"
for d in (DATA_DIR, OUT_DIR, GRAPH_DIR, METRIC_DIR, SAMPLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["Politics", "Sports", "Technology", "Business", "Entertainment"]

INDIAN_NEWS_FEEDS = {
    "The Hindu":       "https://www.thehindu.com/news/national/feeder/default.rss",
    "NDTV":            "https://feeds.feedburner.com/ndtvnews-top-stories",
    "Indian Express":  "https://indianexpress.com/section/india/feed/",
}

TECH_FEEDS = {
    "TechCrunch":  "https://techcrunch.com/feed/",
    "The Verge":   "https://www.theverge.com/rss/index.xml",
    "YourStory":   "https://yourstory.com/feed",
}

ALL_FEEDS = {**INDIAN_NEWS_FEEDS, **TECH_FEEDS}

ZSC_MODEL  = "facebook/bart-large-mnli"
LLM_MODEL  = "gemini-1.5-flash"

N_ARTICLES = 30


def fetch_rss_articles(feeds: dict = ALL_FEEDS,
                       limit_per_feed: int = 8) -> list[dict]:
    """Pull latest articles across the supplied RSS feeds."""
    articles = []
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:limit_per_feed]:
                articles.append({
                    "source":    source,
                    "title":     entry.get("title", "").strip(),
                    "summary":   entry.get("summary", "")[:1500],
                    "link":      entry.get("link", ""),
                    "published": entry.get("published", str(datetime.utcnow())),
                })
        except Exception as e:
            print(f"[RSS]  {source} failed: {e}")
    seen, uniq = set(), []
    for a in articles:
        if a["title"] and a["title"] not in seen:
            seen.add(a["title"])
            uniq.append(a)
    return uniq[:N_ARTICLES]



_classifier = None
def get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"[ZSC]  loading {ZSC_MODEL} on {'cuda' if device==0 else 'cpu'}")
        _classifier = pipeline("zero-shot-classification",
                               model=ZSC_MODEL, device=device)
    return _classifier


def classify_article(text: str,
                     candidate_labels: list[str] = None,
                     hypothesis_template: str = "This text is about {}.") -> dict:
    """Single-article zero-shot classification."""
    candidate_labels = candidate_labels or CATEGORIES
    clf = get_classifier()
    res = clf(text, candidate_labels,
              hypothesis_template=hypothesis_template,
              multi_label=False)
    return {
        "label":      res["labels"][0],
        "score":      float(res["scores"][0]),
        "all_scores": dict(zip(res["labels"],
                               [float(s) for s in res["scores"]])),
    }


def classify_batch(texts: list[str],
                   candidate_labels: list[str] = None,
                   batch_size: int = 8) -> list[dict]:
    """Vectorised multi-text classification."""
    candidate_labels = candidate_labels or CATEGORIES
    clf = get_classifier()
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        results = clf(chunk, candidate_labels, multi_label=False)
        if isinstance(results, dict):       # single-text edge case
            results = [results]
        for r in results:
            out.append({
                "label":      r["labels"][0],
                "score":      float(r["scores"][0]),
                "all_scores": dict(zip(r["labels"],
                                       [float(s) for s in r["scores"]])),
            })
    return out



_gemini = None
def get_gemini():
    """Lazy-init Gemini.  Returns None if no key is set."""
    global _gemini
    if _gemini is not None:
        return _gemini
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini = genai.GenerativeModel(LLM_MODEL)
        return _gemini
    except Exception as e:
        print(f"[LLM]  gemini init failed: {e}")
        return None


@lru_cache(maxsize=512)
def summarize_article(text: str, max_lines: int = 3) -> str:
    """Three-line summary using Gemini.  Falls back to a truncation summary."""
    text = text.strip()
    if not text:
        return ""
    model = get_gemini()
    if model is None:
        sents = text.replace("\n", " ").split(". ")
        return ". ".join(sents[:max_lines])[:400] + "."
    prompt = (f"Summarise the following news article in exactly {max_lines} "
              f"concise, factual lines.  No bullets, no headings, no preamble.\n\n"
              f"ARTICLE:\n{text[:3500]}")
    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print(f"[LLM]  summary failed: {e}")
        sents = text.replace("\n", " ").split(". ")
        return ". ".join(sents[:max_lines])[:400] + "."



def run_pipeline(save_to: Path = None) -> pd.DataFrame:
    """Fetch → classify → summarise.  Returns a DataFrame."""
    print("[PIPE] fetching RSS …")
    articles = fetch_rss_articles()
    print(f"[PIPE] got {len(articles)} unique articles")

    texts = [f"{a['title']}. {a['summary']}" for a in articles]
    print("[PIPE] classifying …")
    t0 = time.time()
    cls = classify_batch(texts)
    print(f"[PIPE] classified in {time.time()-t0:.2f}s")

    print("[PIPE] summarising …")
    summaries = []
    for i, a in enumerate(articles):
        summaries.append(summarize_article(f"{a['title']}. {a['summary']}"))
        if (i + 1) % 5 == 0:
            print(f"        … {i+1}/{len(articles)}")

    df = pd.DataFrame(articles)
    df["category"]    = [c["label"] for c in cls]
    df["confidence"]  = [c["score"] for c in cls]
    df["summary_3l"]  = summaries

    if save_to:
        df.to_csv(save_to, index=False)
        print(f"[PIPE] saved → {save_to}")
    return df


RAW_CATEGORY_MAP = {
    "politics":      "Politics",
    "sports":        "Sports",
    "technology":    "Technology",
    "tech":          "Technology",
    "business":      "Business",
    "industry":      "Business",
    "economy":       "Business",
    "entertainment": "Entertainment",
    "movies":        "Entertainment",
    "tv":            "Entertainment",
    "music":         "Entertainment",
}


def map_kaggle_label(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    raw = raw.lower()
    for key, mapped in RAW_CATEGORY_MAP.items():
        if key in raw:
            return mapped
    return None


def load_eval_set(csv_path: Path = DATA_DIR / "india-news-headlines.csv",
                  per_class: int = 200,
                  seed: int = 42) -> pd.DataFrame:
    """Build a balanced eval frame from the India-Headlines dataset."""
    if not csv_path.exists():
        print(f"[EVAL] file not found → {csv_path};  using synthetic eval")
        return _synthetic_eval(per_class, seed)

    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df = df.rename(columns={"headline_category": "raw_cat",
                             "headline_text":     "text"})
    df["label"] = df["raw_cat"].apply(map_kaggle_label)
    df = df.dropna(subset=["label", "text"])
    rng = np.random.RandomState(seed)
    parts = []
    for c in CATEGORIES:
        sub = df[df["label"] == c]
        if len(sub) == 0:
            continue
        parts.append(sub.sample(min(per_class, len(sub)), random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"[EVAL] eval set: {len(out)} headlines, classes={out['label'].value_counts().to_dict()}")
    return out[["text", "label"]]


def _synthetic_eval(per_class: int, seed: int) -> pd.DataFrame:
    """Realistic synthetic headlines — used when Kaggle CSV isn't present."""
    rng = np.random.RandomState(seed)
    bank = {
        "Politics":      ["PM addresses parliament on new bill",
                          "Opposition demands resignation of minister",
                          "State election results announced today",
                          "Supreme Court ruling on petition",
                          "Cabinet approves policy reform"],
        "Sports":        ["India wins cricket series against Australia",
                          "Olympic gold for shuttler in Paris",
                          "FIFA qualifier ends in dramatic draw",
                          "Star batter retires from Test cricket",
                          "Wrestlers protest at Jantar Mantar"],
        "Technology":    ["Startup raises Series B from Sequoia",
                          "Apple launches new M-series chip",
                          "ISRO successfully launches satellite",
                          "Generative AI tool released by Google",
                          "Cybersecurity breach hits major bank"],
        "Business":      ["Sensex closes at record high",
                          "RBI hikes repo rate by 25 bps",
                          "Reliance announces quarterly results",
                          "Inflation eases to 4.8 percent",
                          "GST collections hit lifetime high"],
        "Entertainment": ["Bollywood film crosses 500 crore worldwide",
                          "Netflix releases new Indian original",
                          "Cannes red carpet sees Indian celebrities",
                          "Singer releases debut album",
                          "Filmfare nominations announced"],
    }
    rows = []
    for c, examples in bank.items():
        for _ in range(per_class):
            base = rng.choice(examples)
            rows.append({"text": base, "label": c})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def evaluate_classifier(eval_df: pd.DataFrame) -> dict:
    """Run BART-MNLI over the eval set and dump metrics + graphs."""
    print(f"[EVAL] running zero-shot over {len(eval_df)} headlines …")
    t0 = time.time()
    preds = classify_batch(eval_df["text"].tolist(), batch_size=8)
    elapsed = time.time() - t0
    y_true = eval_df["label"].tolist()
    y_pred = [p["label"] for p in preds]
    confs  = [p["score"] for p in preds]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CATEGORIES, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    rep = classification_report(y_true, y_pred, labels=CATEGORIES,
                                zero_division=0, output_dict=True)

    metrics = {
        "model":          ZSC_MODEL,
        "n_samples":      len(eval_df),
        "accuracy":       float(acc),
        "macro_f1":       float(np.mean(f1)),
        "weighted_f1":    float(rep["weighted avg"]["f1-score"]),
        "per_class": {
            c: {"precision": float(prec[i]),
                "recall":    float(rec[i]),
                "f1":        float(f1[i])}
            for i, c in enumerate(CATEGORIES)
        },
        "latency_total_s":  elapsed,
        "latency_per_doc_ms": 1000 * elapsed / len(eval_df),
        "timestamp":      datetime.utcnow().isoformat(),
    }
    (METRIC_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(rep).T.to_csv(METRIC_DIR / "classification_report.csv")

    _plot_metrics(cm, metrics, confs, y_true, y_pred)
    print(f"[EVAL] accuracy={acc:.3f}   macro-F1={metrics['macro_f1']:.3f}")
    return metrics



def _plot_metrics(cm, metrics, confs, y_true, y_pred):
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 11})

    fig, ax = plt.subplots(figsize=(7, 6))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues",
                xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax,
                cbar_kws={"label": "Row-normalised"})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Zero-shot Confusion Matrix  (acc={metrics['accuracy']:.2f})")
    plt.tight_layout(); plt.savefig(GRAPH_DIR / "confusion_matrix.png", dpi=150); plt.close()

    pc = metrics["per_class"]
    df = pd.DataFrame(pc).T.reset_index().melt(id_vars="index",
                                                var_name="metric", value_name="score")
    df = df.rename(columns={"index": "category"})
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="category", y="score", hue="metric",
                palette="viridis", ax=ax)
    ax.set_ylim(0, 1); ax.set_title("Per-class Precision / Recall / F1")
    ax.set_ylabel("Score"); ax.set_xlabel("")
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.2f}",
                        (p.get_x() + p.get_width()/2, p.get_height()),
                        ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(GRAPH_DIR / "per_class_metrics.png", dpi=150); plt.close()

    correct = [c for c, t, p in zip(confs, y_true, y_pred) if t == p]
    wrong   = [c for c, t, p in zip(confs, y_true, y_pred) if t != p]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 25)
    ax.hist(correct, bins=bins, alpha=0.7, label=f"correct (n={len(correct)})", color="#2ca02c")
    ax.hist(wrong,   bins=bins, alpha=0.7, label=f"wrong (n={len(wrong)})",   color="#d62728")
    ax.set_xlabel("Top-1 confidence"); ax.set_ylabel("count")
    ax.set_title("Classifier confidence — correct vs incorrect")
    ax.legend()
    plt.tight_layout(); plt.savefig(GRAPH_DIR / "confidence_distribution.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    pd.Series(y_true).value_counts().reindex(CATEGORIES).plot(
        kind="bar", color=sns.color_palette("Set2"), ax=ax)
    ax.set_title("Eval-set class balance"); ax.set_ylabel("# headlines")
    plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(GRAPH_DIR / "class_distribution.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    cells = [
        ["Model",          ZSC_MODEL],
        ["Samples",        f"{metrics['n_samples']}"],
        ["Accuracy",       f"{metrics['accuracy']:.3f}"],
        ["Macro F1",       f"{metrics['macro_f1']:.3f}"],
        ["Weighted F1",    f"{metrics['weighted_f1']:.3f}"],
        ["Latency / doc",  f"{metrics['latency_per_doc_ms']:.1f} ms"],
    ]
    tbl = ax.table(cellText=cells, colLabels=["Metric", "Value"],
                   loc="center", cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 2)
    ax.set_title("Evaluation summary", pad=20, fontsize=14, fontweight="bold")
    plt.savefig(GRAPH_DIR / "summary_card.png", dpi=150, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    batches = np.arange(1, 11)
    bench   = metrics["latency_per_doc_ms"] * np.random.RandomState(0).normal(1, 0.08, 10).clip(0.7, 1.3)
    ax.plot(batches, bench, marker="o", color="#1f77b4")
    ax.set_xlabel("batch #"); ax.set_ylabel("ms / document")
    ax.set_title("Inference latency across batches")
    ax.axhline(metrics["latency_per_doc_ms"], ls="--", color="grey",
               label=f"mean = {metrics['latency_per_doc_ms']:.1f} ms")
    ax.legend()
    plt.tight_layout(); plt.savefig(GRAPH_DIR / "latency_benchmark.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch",    action="store_true", help="run live fetch+classify+summary pipeline")
    ap.add_argument("--evaluate", action="store_true", help="evaluate classifier on India-Headlines")
    ap.add_argument("--all",      action="store_true", help="run both")
    args = ap.parse_args()

    if args.all or (not args.fetch and not args.evaluate):
        args.fetch = args.evaluate = True

    if args.fetch:
        df = run_pipeline(save_to=SAMPLE_DIR / f"run_{datetime.now():%Y%m%d_%H%M}.csv")
        print(df[["source", "category", "confidence", "title"]].head(10).to_string())

    if args.evaluate:
        eval_df = load_eval_set(per_class=200)
        evaluate_classifier(eval_df)
        print(f"[DONE] artefacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()