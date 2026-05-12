"""
orchestrator.py
---------------
The brain of the system. Collects signals from all 3 agents,
runs them through the risk engine, then uses Claude (via Anthropic API)
to synthesize everything into a final trade decision with reasoning.

Flow for each stock:
  1. Get signal from momentum_agent     → score + interpretation
  2. Get signal from mean_reversion_agent → score + interpretation
  3. Get signal from ml_agent           → score + interpretation
  4. Compute weighted combined score
  5. Send everything to Claude API      → get trade thesis + decision
  6. Pass decision through risk_engine  → get approved/rejected trade
  7. Print final decision

This is Phase 4. In Phase 5, step 7 will actually execute the trade.

Usage:
    python orchestrator.py RELIANCE
    python orchestrator.py RELIANCE TCS INFY
    python orchestrator.py --all
"""

import sys
import os
import json
import sqlite3
import requests
import pandas as pd
from dataclasses import asdict

# Import our agents and risk engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.momentum_agent      import get_latest_signal as momentum_signal
from agents.mean_reversion_agent import get_latest_signal as mean_reversion_signal
from agents.ml_agent            import get_latest_signal as ml_signal
from risk.risk_engine           import evaluate_trade, Portfolio, RiskConfig

DB_PATH = "data/nifty50.db"

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Agent weights — how much to trust each agent's score
# These add up to 1.0
AGENT_WEIGHTS = {
    "momentum":       0.35,
    "mean_reversion": 0.30,
    "ml_classifier":  0.35,
}

# ── FETCH CURRENT PRICE ───────────────────────────────────────────────────────

def get_current_price(symbol: str) -> float:
    """Get the most recent closing price for a stock from the database."""
    conn = sqlite3.connect(DB_PATH)
    result = conn.execute(
        "SELECT Close FROM prices WHERE Symbol = ? ORDER BY Date DESC LIMIT 1",
        (symbol,)
    ).fetchone()
    conn.close()

    if result is None:
        raise ValueError(f"No price data found for {symbol}")
    return float(result[0])


def get_recent_price_context(symbol: str, days: int = 10) -> str:
    """Get last N days of price action as a text summary for the LLM."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"""
        SELECT Date, Close, Daily_Return
        FROM prices
        WHERE Symbol = '{symbol}'
        ORDER BY Date DESC
        LIMIT {days}
    """, conn)
    conn.close()

    if df.empty:
        return "No recent price data available."

    df = df.sort_values("Date")
    lines = []
    for _, row in df.iterrows():
        ret = row["Daily_Return"]
        ret_str = f"{ret:+.2%}" if pd.notna(ret) else "N/A"
        lines.append(f"  {row['Date']}: ₹{row['Close']:.2f} ({ret_str})")

    return "\n".join(lines)


# ── LLM ORCHESTRATION ─────────────────────────────────────────────────────────

def call_claude(prompt: str) -> str:
    """
    Call Groq API (free tier) instead of Anthropic.
    Using llama-3.3-70b — fast, free, good enough for this.
    """
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "system",
                    "content": """You are a quantitative analyst at an AI-powered hedge fund focused on Nifty 50 stocks.

You receive signal scores from three independent agents:
- Momentum agent: detects price trends (positive = uptrend, negative = downtrend)
- Mean reversion agent: detects overbought/oversold (positive = oversold/buy, negative = overbought/sell)
- ML classifier: Random Forest model predicting next-day direction

Your job is to synthesize these signals and provide:
1. A final combined score from -1.0 to +1.0
2. A clear trade recommendation (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)
3. A brief trade thesis (2-3 sentences max) explaining your reasoning
4. Key risk factors to watch

Respond ONLY in valid JSON. No extra text before or after.
JSON format:
{
  "final_score": <float between -1 and 1>,
  "recommendation": "<STRONG BUY | BUY | HOLD | SELL | STRONG SELL>",
  "thesis": "<2-3 sentence reasoning>",
  "risks": "<key risk factors>"
}"""
                },
                {"role": "user", "content": prompt}
            ]
        }
    )

    if response.status_code != 200:
        raise Exception(f"Groq API error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def build_prompt(symbol: str, signals: dict, price_context: str) -> str:
    """Build the prompt we send to Claude."""
    return f"""Analyze {symbol} and provide a trade decision.

AGENT SIGNALS:
- Momentum Agent:       score={signals['momentum']['score']:+.4f} | {signals['momentum']['interpretation']}
- Mean Reversion Agent: score={signals['mean_reversion']['score']:+.4f} | {signals['mean_reversion']['interpretation']}
- ML Classifier:        score={signals['ml']['score']:+.4f} | {signals['ml']['interpretation']}

WEIGHTED COMBINED SCORE: {signals['combined']:+.4f}
(weights: momentum=35%, mean_reversion=30%, ml=35%)

RECENT PRICE ACTION (last 10 days):
{price_context}

Based on these signals and price context, provide your analysis in the required JSON format."""


# ── MAIN ORCHESTRATOR ─────────────────────────────────────────────────────────

def run_orchestrator(symbol: str, portfolio: Portfolio, config: RiskConfig) -> dict:
    """
    Full pipeline for one stock:
      agents → combine → LLM synthesis → risk check → decision
    """
    print(f"\n{'═'*60}")
    print(f"  Analyzing: {symbol}")
    print(f"{'═'*60}")

    # Step 1: Collect signals from all agents
    print("  [1/4] Collecting agent signals...")
    try:
        mom_sig = momentum_signal(symbol)
        mr_sig  = mean_reversion_signal(symbol)
        ml_sig  = ml_signal(symbol)
    except Exception as e:
        print(f"  [ERROR] Agent failed: {e}")
        return {"symbol": symbol, "error": str(e)}

    # Step 2: Compute weighted combined score
    combined = (
        AGENT_WEIGHTS["momentum"]       * mom_sig["score"] +
        AGENT_WEIGHTS["mean_reversion"] * mr_sig["score"] +
        AGENT_WEIGHTS["ml_classifier"]  * ml_sig["score"]
    )

    signals = {
        "momentum":       mom_sig,
        "mean_reversion": mr_sig,
        "ml":             ml_sig,
        "combined":       round(combined, 4),
    }

    print(f"  Momentum:       {mom_sig['score']:+.4f} ({mom_sig['interpretation']})")
    print(f"  Mean Reversion: {mr_sig['score']:+.4f} ({mr_sig['interpretation']})")
    print(f"  ML Classifier:  {ml_sig['score']:+.4f} ({ml_sig['interpretation']})")
    print(f"  Combined Score: {combined:+.4f}")

    # Step 3: Get price context and call Claude
    print("  [2/4] Calling Claude for trade thesis...")
    price_context = get_recent_price_context(symbol)
    prompt        = build_prompt(symbol, signals, price_context)

    try:
        llm_response_raw = call_claude(prompt)
        llm_decision     = json.loads(llm_response_raw)
    except Exception as e:
        print(f"  [ERROR] Claude API call failed: {e}")
        # Fallback: use combined score directly without LLM
        llm_decision = {
            "final_score":     combined,
            "recommendation":  "BUY" if combined > 0.1 else ("SELL" if combined < -0.1 else "HOLD"),
            "thesis":          "LLM unavailable — using raw combined score.",
            "risks":           "Unable to assess via LLM.",
        }

    print(f"  Claude says:    {llm_decision['recommendation']} (score: {llm_decision['final_score']:+.4f})")
    print(f"  Thesis:         {llm_decision['thesis']}")

    # Step 4: Run through risk engine
    print("  [3/4] Running risk checks...")
    current_price = get_current_price(symbol)

    trade = evaluate_trade(
        symbol        = symbol,
        signal_score  = llm_decision["final_score"],
        current_price = current_price,
        portfolio     = portfolio,
        config        = config,
    )

    # Step 5: Report final decision
    print(f"  [4/4] Final Decision:")
    if trade.approved:
        print(f"  ✓ APPROVED  → {trade.action} {trade.shares} shares @ ₹{current_price:.2f}")
        print(f"               Capital: ₹{trade.capital_allocated:,.0f} ({trade.position_size_pct:.1%} of portfolio)")
    else:
        print(f"  ✗ BLOCKED   → {trade.rejection_reason}")

    for note in trade.risk_notes:
        print(f"    Risk: {note}")

    return {
        "symbol":       symbol,
        "signals":      signals,
        "llm_decision": llm_decision,
        "trade":        asdict(trade),
        "price":        current_price,
    }


def run_all(portfolio: Portfolio, config: RiskConfig, top_n: int = 5) -> list:
    """
    Run orchestrator on all Nifty 50 stocks.
    Returns results sorted by signal strength.
    Only analyzes top_n by default to save time.
    """
    conn = sqlite3.connect(DB_PATH)
    symbols = [
        row[0] for row in
        conn.execute("SELECT DISTINCT Symbol FROM prices").fetchall()
    ]
    conn.close()

    print(f"Running orchestrator on {len(symbols)} stocks...")
    print(f"(Showing top {top_n} by signal strength)\n")

    results = []
    for symbol in symbols[:top_n]:
        result = run_orchestrator(symbol, portfolio, config)
        results.append(result)

    return results


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    portfolio = Portfolio(cash=100_000)
    config    = RiskConfig(total_capital=100_000)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python orchestrator.py RELIANCE")
        print("  python orchestrator.py RELIANCE TCS INFY")
        print("  python orchestrator.py --all")
        sys.exit(0)

    if sys.argv[1] == "--all":
        results = run_all(portfolio, config, top_n=5)
    else:
        symbols = [s.upper() for s in sys.argv[1:]]
        results = []
        for sym in symbols:
            result = run_orchestrator(sym, portfolio, config)
            results.append(result)

    # Summary table
    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'═'*60}")
    print(f"{'Stock':<15} {'Score':>8} {'Recommendation':<18} {'Action':<8} {'Shares':>7}")
    print("─" * 60)
    for r in results:
        if "error" in r:
            continue
        rec    = r["llm_decision"]["recommendation"]
        score  = r["llm_decision"]["final_score"]
        action = r["trade"]["action"] if r["trade"]["approved"] else "BLOCKED"
        shares = r["trade"]["shares"] if r["trade"]["approved"] else 0
        print(f"{r['symbol']:<15} {score:>+8.4f} {rec:<18} {action:<8} {shares:>7}")