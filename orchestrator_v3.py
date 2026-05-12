"""
orchestrator_v3.py
------------------
V3 Orchestrator — 7 agents total:
  1. HMM Regime Detection
  2. Kalman Momentum
  3. Advanced Mean Reversion (ADF + Hurst + Cointegration)
  4. LSTM Sequential Pattern
  5. Heston Stochastic Volatility
  6. GBM Monte Carlo
  7. Sentiment (News NLP)

Usage:
    python orchestrator_v3.py RELIANCE
    python orchestrator_v3.py RELIANCE TCS INFY
    python orchestrator_v3.py --all
"""

import sys
import os
import json
import sqlite3
import requests
import pandas as pd
import numpy as np
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.regime_agent                  import get_regime
from agents.kalman_momentum_agent         import get_latest_signal as kalman_sig
from agents.advanced_mean_reversion_agent import get_latest_signal as adv_mr_sig
from agents.lstm_agent                    import get_latest_signal as lstm_sig
from agents.heston_agent                  import get_latest_signal as heston_sig
from agents.gbm_monte_carlo_agent         import get_latest_signal as mc_sig
from agents.sentiment_agent               import get_latest_signal as sentiment_sig
from risk.risk_engine                     import evaluate_trade, Portfolio, RiskConfig

DB_PATH      = "data/nifty50.db"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def get_current_price(symbol):
    conn   = sqlite3.connect(DB_PATH)
    result = conn.execute(
        "SELECT Close FROM prices WHERE Symbol=? ORDER BY Date DESC LIMIT 1", (symbol,)
    ).fetchone()
    conn.close()
    if result is None:
        raise ValueError(f"No price data for {symbol}")
    return float(result[0])


def get_price_context(symbol, days=10):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"""
        SELECT Date, Close, Daily_Return FROM prices
        WHERE Symbol='{symbol}' ORDER BY Date DESC LIMIT {days}
    """, conn)
    conn.close()
    df = df.sort_values("Date")
    lines = []
    for _, r in df.iterrows():
        ret = f"{r['Daily_Return']:+.2%}" if pd.notna(r['Daily_Return']) else "N/A"
        lines.append(f"  {r['Date']}: ₹{r['Close']:.2f} ({ret})")
    return "\n".join(lines)


def call_groq(prompt):
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 800,
            "messages": [
                {
                    "role": "system",
                    "content": """You are a senior quantitative analyst at an AI hedge fund.
You receive outputs from 7 mathematical agents. Synthesize into one decision.
Respond ONLY in valid JSON:
{
  "final_score": <float -1 to 1>,
  "recommendation": "<STRONG BUY|BUY|HOLD|SELL|STRONG SELL>",
  "thesis": "<2-3 sentences>",
  "risks": "<key risks>",
  "confidence": "<HIGH|MEDIUM|LOW>",
  "key_driver": "<which agent drove the decision most>"
}"""
                },
                {"role": "user", "content": prompt}
            ]
        },
        timeout=20,
    )
    if resp.status_code != 200:
        raise Exception(f"Groq error {resp.status_code}")
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


def build_prompt(symbol, regime, signals, price_ctx):
    mc   = signals["monte_carlo"]
    mr   = signals["adv_mr"]
    sent = signals["sentiment"]
    return f"""Analyze {symbol} for a trade decision.

MARKET REGIME: {regime['regime_name'].upper()} (regime {regime['regime']})
{regime['description']}
Probabilities: Bull={regime['probabilities'][0]:.0%} HighVol={regime['probabilities'][1]:.0%} Bear={regime['probabilities'][2]:.0%}

AGENT SIGNALS (7 agents):
  Kalman Momentum    : {signals['kalman']['score']:+.4f} | {signals['kalman']['interpretation']}
  Adv Mean Reversion : {mr['score']:+.4f} | H={mr.get('hurst','?')} | {mr['interpretation']}
  LSTM Sequential    : {signals['lstm']['score']:+.4f} | {signals['lstm']['interpretation']}
  Heston Stoch Vol   : {signals['heston']['score']:+.4f} | {signals['heston']['interpretation']}
  GBM Monte Carlo    : {mc['score']:+.4f} | P(gain)={mc.get('prob_gain',0):.1%} | CVaR={mc.get('cvar_5pct',0):.2f}%
  Sentiment (News)   : {sent['score']:+.4f} | {sent['interpretation']}

WEIGHTED COMBINED: {signals['combined']:+.4f}

RECENT PRICE ACTION:
{price_ctx}

Provide analysis."""


def run_orchestrator_v3(symbol, portfolio, config):
    print(f"\n{'═'*65}")
    print(f"  Analyzing: {symbol} [V3 — 7 Agents]")
    print(f"{'═'*65}")

    # Step 1: Regime
    print("  [1/7] HMM Regime Detection...")
    try:
        regime  = get_regime()
        weights = regime["agent_weights"]
        print(f"         {regime['regime_name']} — {regime['description']}")
    except Exception as e:
        regime  = {"regime": 0, "regime_name": "unknown", "probabilities": [0.33,0.33,0.34],
                   "description": "fallback", "agent_weights": {"momentum":0.35,"mean_reversion":0.30,"ml_classifier":0.35}}
        weights = regime["agent_weights"]

    signals = {}

    # Step 2: All agents
    print("  [2/7] Kalman Momentum...")
    try:
        signals["kalman"] = kalman_sig(symbol)
        print(f"         {signals['kalman']['score']:+.4f}")
    except Exception as e:
        signals["kalman"] = {"score": 0.0, "interpretation": f"failed: {e}"}

    print("  [3/7] Advanced Mean Reversion...")
    try:
        signals["adv_mr"] = adv_mr_sig(symbol)
        print(f"         {signals['adv_mr']['score']:+.4f}  H={signals['adv_mr'].get('hurst','?')}")
    except Exception as e:
        signals["adv_mr"] = {"score": 0.0, "interpretation": f"failed: {e}", "hurst": 0.5}

    print("  [4/7] LSTM Sequential Pattern...")
    try:
        signals["lstm"] = lstm_sig(symbol)
        print(f"         {signals['lstm']['score']:+.4f}")
    except Exception as e:
        signals["lstm"] = {"score": 0.0, "interpretation": f"failed: {e}"}

    print("  [5/7] Heston Stochastic Volatility...")
    try:
        signals["heston"] = heston_sig(symbol)
        print(f"         {signals['heston']['score']:+.4f}  vol_ratio={signals['heston'].get('vol_ratio','?')}")
    except Exception as e:
        signals["heston"] = {"score": 0.0, "interpretation": f"failed: {e}"}

    print("  [6/7] GBM Monte Carlo...")
    try:
        signals["monte_carlo"] = mc_sig(symbol)
        print(f"         {signals['monte_carlo']['score']:+.4f}  P(gain)={signals['monte_carlo'].get('prob_gain',0):.1%}")
    except Exception as e:
        signals["monte_carlo"] = {"score": 0.0, "interpretation": f"failed: {e}",
                                   "prob_gain": 0.5, "expected_ret": 0.0, "cvar_5pct": 0.0}

    print("  [7/7] Sentiment (News NLP)...")
    try:
        signals["sentiment"] = sentiment_sig(symbol)
        print(f"         {signals['sentiment']['score']:+.4f}  {signals['sentiment'].get('n_headlines',0)} headlines")
    except Exception as e:
        signals["sentiment"] = {"score": 0.0, "interpretation": f"failed: {e}"}

    # Step 3: Weighted combination
    # Regime weights cover momentum/mean_reversion/ml
    # New agents get fixed weights, regime weights scaled down
    regime_scale = 0.60   # 60% to regime-weighted agents
    lstm_w       = 0.12
    heston_w     = 0.10
    mc_w         = 0.10
    sentiment_w  = 0.08

    combined = float(np.clip(
        regime_scale * (
            weights["momentum"]       * signals["kalman"]["score"]  +
            weights["mean_reversion"] * signals["adv_mr"]["score"]  +
            weights["ml_classifier"]  * 0.0  # ML agent replaced by LSTM
        ) +
        lstm_w      * signals["lstm"]["score"]        +
        heston_w    * signals["heston"]["score"]      +
        mc_w        * signals["monte_carlo"]["score"] +
        sentiment_w * signals["sentiment"]["score"],
        -1, 1
    ))
    signals["combined"] = round(combined, 4)
    print(f"\n  Combined Score: {combined:+.4f}")

    # Step 4: LLM
    print("  Calling Groq LLM...")
    price_ctx = get_price_context(symbol)
    prompt    = build_prompt(symbol, regime, signals, price_ctx)
    try:
        llm = call_groq(prompt)
    except Exception as e:
        rec = "BUY" if combined > 0.1 else ("SELL" if combined < -0.1 else "HOLD")
        llm = {"final_score": combined, "recommendation": rec,
               "thesis": f"LLM unavailable. Combined={combined:+.4f}",
               "risks": "N/A", "confidence": "MEDIUM", "key_driver": "combined score"}

    print(f"  LLM: {llm['recommendation']} score={llm['final_score']:+.4f} conf={llm.get('confidence','?')}")
    print(f"  Driver: {llm.get('key_driver','?')}")
    print(f"  Thesis: {llm['thesis'][:90]}...")

    # Step 5: Risk engine
    current_price = get_current_price(symbol)
    trade = evaluate_trade(
        symbol=symbol, signal_score=llm["final_score"],
        current_price=current_price, portfolio=portfolio, config=config,
        regime=regime["regime"], confidence=llm.get("confidence","MEDIUM"),
        hurst=float(signals["adv_mr"].get("hurst", 0.5)),
        cvar_mc=float(signals["monte_carlo"].get("cvar_5pct", 0.0)),
    )

    if trade.approved:
        print(f"  ✓ APPROVED → {trade.action} {trade.shares} shares @ ₹{current_price:.2f}")
    else:
        print(f"  ✗ BLOCKED  → {trade.rejection_reason}")

    return {"symbol": symbol, "regime": regime, "signals": signals,
            "llm": llm, "trade": asdict(trade), "price": current_price}


def run_all(portfolio, config, top_n=5):
    conn    = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT Symbol FROM prices").fetchall()]
    conn.close()
    results = []
    for sym in symbols[:top_n]:
        results.append(run_orchestrator_v3(sym, portfolio, config))
    return results


if __name__ == "__main__":
    portfolio = Portfolio(cash=100_000)
    config    = RiskConfig(total_capital=100_000)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python orchestrator_v3.py RELIANCE")
        print("  python orchestrator_v3.py RELIANCE TCS")
        print("  python orchestrator_v3.py --all")
        sys.exit(0)

    if sys.argv[1] == "--all":
        results = run_all(portfolio, config, top_n=5)
    else:
        results = [run_orchestrator_v3(s.upper(), portfolio, config) for s in sys.argv[1:]]

    print(f"\n{'═'*70}")
    print("V3 SUMMARY")
    print(f"{'═'*70}")
    print(f"{'Stock':<12} {'Score':>7} {'Rec':<14} {'Conf':<8} {'Driver':<20} {'Action'}")
    print("─" * 70)
    for r in results:
        if "error" in r: continue
        print(
            f"{r['symbol']:<12} "
            f"{r['llm']['final_score']:>+7.4f} "
            f"{r['llm']['recommendation']:<14} "
            f"{r['llm'].get('confidence','?'):<8} "
            f"{r['llm'].get('key_driver','?')[:18]:<20} "
            f"{'✓ '+r['trade']['action'] if r['trade']['approved'] else '✗ BLOCKED'}"
        )
