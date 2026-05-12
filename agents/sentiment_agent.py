"""
sentiment_agent.py
------------------
Sentiment Agent — analyzes news headlines for a stock.

No HuggingFace needed — uses a lexicon-based approach with
a finance-specific word dictionary. Same principle as FinBERT
but runs instantly with zero dependencies.

How it works:
  1. Fetch recent news headlines from Yahoo Finance RSS feed
  2. Score each headline using finance sentiment lexicon
     Bullish words: growth, profit, beat, surge, upgrade, acquisition...
     Bearish words: loss, miss, decline, downgrade, debt, layoff...
  3. Weight recent headlines more than older ones
  4. Combine into a single sentiment score

Why sentiment matters:
  All other agents use price/volume data.
  Sentiment captures information NOT yet in the price —
  an earnings beat announced at 9 AM moves the stock by 9:15 AM.
  Agents using only price data miss the first and biggest move.

Output: score between -1.0 and +1.0
"""

import numpy as np
import pandas as pd
import sqlite3
import urllib.request
import xml.etree.ElementTree as ET
import re
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/nifty50.db"

# ── FINANCE SENTIMENT LEXICON ─────────────────────────────────────────────────
# Carefully curated for Indian stock market context

BULLISH_WORDS = {
    # Earnings / Performance
    "beat": 2, "beats": 2, "record": 2, "profit": 1, "growth": 1,
    "surge": 2, "soar": 2, "rally": 1, "gain": 1, "rise": 1, "rises": 1,
    "up": 0.5, "high": 0.5, "strong": 1, "robust": 1, "outperform": 2,
    "upgrade": 2, "upgraded": 2, "buy": 1, "overweight": 1,
    "positive": 1, "exceed": 2, "exceeds": 2, "exceeded": 2,
    "expansion": 1, "acquisition": 1, "merger": 0.5, "deal": 0.5,
    "contract": 0.5, "order": 1, "orders": 1, "revenue": 0.5,
    "dividend": 1, "buyback": 1, "bonus": 1, "split": 0.5,
    "launch": 1, "partnership": 1, "investment": 0.5, "capex": 0.5,
    "margin": 0.5, "margins": 0.5, "ebitda": 0.5, "roe": 0.5,
    "recovery": 1, "rebound": 1, "turnaround": 1, "momentum": 1,
    "optimistic": 1, "confident": 1, "bullish": 2, "upbeat": 1,
    "approved": 1, "approval": 1, "wins": 1, "won": 1, "award": 1,
}

BEARISH_WORDS = {
    # Earnings / Performance
    "miss": 2, "misses": 2, "missed": 2, "loss": 2, "losses": 2,
    "decline": 1, "declines": 1, "fall": 1, "falls": 1, "drop": 1,
    "drops": 1, "down": 0.5, "low": 0.5, "weak": 1, "poor": 1,
    "underperform": 2, "downgrade": 2, "downgraded": 2, "sell": 1,
    "underweight": 1, "negative": 1, "disappoint": 2, "disappoints": 2,
    "disappointed": 2, "disappointing": 2, "concern": 1, "concerns": 1,
    "risk": 0.5, "risks": 0.5, "warning": 1, "warns": 1, "warned": 1,
    "debt": 1, "leverage": 0.5, "default": 2, "bankruptcy": 3,
    "fraud": 3, "scam": 3, "probe": 2, "investigation": 2, "penalty": 2,
    "fine": 1, "lawsuit": 1, "litigation": 1, "recall": 1,
    "layoff": 2, "layoffs": 2, "restructure": 1, "closure": 2,
    "shutdown": 2, "delay": 1, "delays": 1, "postpone": 1,
    "bearish": 2, "pessimistic": 1, "cautious": 0.5, "slowdown": 1,
    "headwind": 1, "headwinds": 1, "pressure": 1, "squeeze": 1,
    "rejected": 1, "reject": 1, "cancelled": 1, "cancel": 1,
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "without", "lack"}
INTENSIFIER_WORDS = {"very", "highly", "significantly", "sharply", "substantially"}


def score_headline(headline):
    """
    Score a single headline using the sentiment lexicon.
    Handles negation (not good = bad) and intensifiers (very good = more good).
    Returns score between -1 and +1.
    """
    words  = re.findall(r'\b[a-zA-Z]+\b', headline.lower())
    score  = 0.0
    i      = 0

    while i < len(words):
        word = words[i]
        multiplier = 1.0

        # Check for negation in previous word
        if i > 0 and words[i-1] in NEGATION_WORDS:
            multiplier = -1.0

        # Check for intensifier in previous word
        if i > 0 and words[i-1] in INTENSIFIER_WORDS:
            multiplier *= 1.5

        if word in BULLISH_WORDS:
            score += BULLISH_WORDS[word] * multiplier
        elif word in BEARISH_WORDS:
            score -= BEARISH_WORDS[word] * multiplier

        i += 1

    # Normalize to -1 to +1
    return float(np.clip(score / 5.0, -1, 1))


def fetch_yahoo_news(symbol, max_items=20):
    """
    Fetch recent news headlines from Yahoo Finance RSS.
    Returns list of (headline, age_days) tuples.
    """
    # Convert NSE symbol to Yahoo Finance format
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
        # If fetch fails, return empty — system continues without sentiment
        pass

    return headlines


def get_latest_signal(symbol):
    """
    Fetch news and compute sentiment signal for a stock.
    Falls back to neutral (0.0) if no news available.
    """
    headlines = fetch_yahoo_news(symbol, max_items=20)

    if not headlines:
        # No news available — return neutral with low confidence
        return {
            "symbol":         symbol,
            "agent":          "sentiment",
            "score":          0.0,
            "interpretation": "no recent news available — neutral",
            "n_headlines":    0,
            "headlines":      [],
        }

    # Score each headline
    scores = [score_headline(h) for h in headlines]

    # Weight recent headlines more (exponential decay)
    weights = np.exp(-0.1 * np.arange(len(scores)))
    weights /= weights.sum()

    weighted_score = float(np.average(scores, weights=weights))
    weighted_score = round(np.clip(weighted_score, -1, 1), 4)

    # Interpretation
    if weighted_score > 0.3:    interp = f"positive news sentiment ({len(headlines)} headlines)"
    elif weighted_score > 0.1:  interp = f"mildly positive news sentiment"
    elif weighted_score < -0.3: interp = f"negative news sentiment ({len(headlines)} headlines)"
    elif weighted_score < -0.1: interp = f"mildly negative news sentiment"
    else:                       interp = f"neutral news sentiment ({len(headlines)} headlines)"

    return {
        "symbol":         symbol,
        "agent":          "sentiment",
        "score":          weighted_score,
        "interpretation": interp,
        "n_headlines":    len(headlines),
        "headlines":      headlines[:5],  # top 5 for display
    }


if __name__ == "__main__":
    print("Sentiment Agent test\n")
    for sym in ["RELIANCE", "TCS", "HDFCBANK"]:
        r = get_latest_signal(sym)
        print(f"{sym:15} score={r['score']:+.4f}  {r['interpretation']}")
        if r["headlines"]:
            print(f"  Top headline: {r['headlines'][0][:80]}")
        print()
