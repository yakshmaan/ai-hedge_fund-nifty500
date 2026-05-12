"""
finbert_sentiment_agent.py
--------------------------
FinBERT Sentiment Agent — replaces lexicon-based sentiment with
a finance-specific BERT transformer model.

FinBERT is trained on 10,000+ financial news articles and understands
context that simple word matching misses:

  "Revenue beat expectations" → POSITIVE (lexicon: beat=bullish ✓)
  "Beat the competition by cutting prices" → NEUTRAL (lexicon: beat=bullish ✗)
  "Not a bad quarter" → POSITIVE (lexicon: bad=bearish ✗)

FinBERT handles all of these correctly because it reads full context.

Setup (one time):
    pip install transformers torch --break-system-packages
    # OR if torch is too heavy:
    pip install transformers --break-system-packages

Model downloads automatically on first run (~500MB, cached after).

Output: score between -1.0 and +1.0
"""

import numpy as np
import pandas as pd
import urllib.request
import xml.etree.ElementTree as ET
import re
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/nifty50.db"


def fetch_yahoo_news(symbol, max_items=20):
    """Fetch recent headlines from Yahoo Finance RSS."""
    yahoo_symbol = symbol + ".NS"
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={yahoo_symbol}&region=IN&lang=en-IN"
    headlines = []
    try:
        req  = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        tree = ET.parse(resp)
        root = tree.getroot()
        for item in root.findall(".//item")[:max_items]:
            title = item.findtext("title", "")
            if title:
                headlines.append(title)
    except Exception:
        pass
    return headlines


def load_finbert():
    """
    Load FinBERT model from HuggingFace.
    Downloads on first run, cached locally after.
    Falls back to lexicon sentiment if transformers not installed.
    """
    try:
        from transformers import pipeline
        print("  Loading FinBERT (first run downloads ~500MB)...")
        classifier = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
        print("  FinBERT loaded.")
        return classifier
    except ImportError:
        print("  transformers not installed. Run: pip install transformers --break-system-packages")
        return None
    except Exception as e:
        print(f"  FinBERT load failed: {e}")
        return None


# Cache model globally so we don't reload every call
_finbert_model = None


def get_finbert_score(headline, model):
    """
    Score one headline using FinBERT.
    Returns score between -1 (negative) and +1 (positive).
    """
    try:
        result = model(headline)[0]
        scores = {r["label"].lower(): r["score"] for r in result}
        positive = scores.get("positive", 0)
        negative = scores.get("negative", 0)
        neutral  = scores.get("neutral", 0)
        # Weighted score: positive pulls up, negative pulls down, neutral is anchor
        score = positive - negative
        # Reduce score when neutral is very high (uncertain news)
        confidence = 1 - (neutral * 0.5)
        return float(np.clip(score * confidence, -1, 1))
    except Exception:
        return 0.0


# Fallback lexicon (same as sentiment_agent.py)
BULLISH_WORDS = {
    "beat":2,"beats":2,"record":2,"profit":1,"growth":1,"surge":2,"soar":2,
    "rally":1,"gain":1,"rise":1,"strong":1,"robust":1,"outperform":2,
    "upgrade":2,"upgraded":2,"positive":1,"exceed":2,"acquisition":1,
    "order":1,"dividend":1,"buyback":1,"recovery":1,"bullish":2,
}
BEARISH_WORDS = {
    "miss":2,"misses":2,"loss":2,"decline":1,"fall":1,"drop":1,"weak":1,
    "underperform":2,"downgrade":2,"negative":1,"disappoint":2,"concern":1,
    "debt":1,"default":2,"fraud":3,"probe":2,"penalty":2,"layoff":2,
    "lawsuit":1,"delay":1,"bearish":2,"slowdown":1,"headwind":1,
}

def lexicon_score(headline):
    words = re.findall(r'\b[a-zA-Z]+\b', headline.lower())
    score = 0.0
    for w in words:
        if w in BULLISH_WORDS: score += BULLISH_WORDS[w]
        if w in BEARISH_WORDS: score -= BEARISH_WORDS[w]
    return float(np.clip(score / 5.0, -1, 1))


def get_latest_signal(symbol):
    global _finbert_model

    headlines = fetch_yahoo_news(symbol, max_items=20)

    if not headlines:
        return {
            "symbol": symbol, "agent": "finbert_sentiment",
            "score": 0.0, "interpretation": "no recent news",
            "n_headlines": 0, "model_used": "none",
        }

    # Try FinBERT first
    if _finbert_model is None:
        _finbert_model = load_finbert()

    if _finbert_model is not None:
        # Use FinBERT
        scores     = [get_finbert_score(h, _finbert_model) for h in headlines]
        model_used = "FinBERT (ProsusAI)"
    else:
        # Fallback to lexicon
        scores     = [lexicon_score(h) for h in headlines]
        model_used = "lexicon (fallback)"

    # Exponential decay weighting — recent news matters more
    weights = np.exp(-0.1 * np.arange(len(scores)))
    weights /= weights.sum()
    weighted = float(np.average(scores, weights=weights))
    score    = round(float(np.clip(weighted, -1, 1)), 4)

    if score > 0.3:    interp = f"strongly positive news ({model_used})"
    elif score > 0.1:  interp = f"mildly positive news ({model_used})"
    elif score < -0.3: interp = f"strongly negative news ({model_used})"
    elif score < -0.1: interp = f"mildly negative news ({model_used})"
    else:              interp = f"neutral news ({model_used})"

    return {
        "symbol":         symbol,
        "agent":          "finbert_sentiment",
        "score":          score,
        "interpretation": interp,
        "n_headlines":    len(headlines),
        "model_used":     model_used,
        "top_headlines":  headlines[:3],
    }


if __name__ == "__main__":
    print("FinBERT Sentiment Agent test\n")
    for sym in ["RELIANCE", "TCS", "HDFCBANK"]:
        r = get_latest_signal(sym)
        print(f"{sym:15} score={r['score']:+.4f}  {r['interpretation']}")
        if r.get("top_headlines"):
            print(f"  Top: {r['top_headlines'][0][:70]}")
        print()
