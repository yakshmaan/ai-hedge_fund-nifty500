"""
orchestrator_v2.py
------------------
Final orchestrator — wired to all 6 agents + risk engine v2.
 
Flow:
  1. HMM Regime Agent        → market state + dynamic weights
  2. Kalman Momentum         → trend signal
  3. Advanced Mean Reversion → ADF + Hurst + pairs signal
  4. ML Classifier           → Random Forest signal
  5. GBM Monte Carlo         → probability distribution signal
  6. Groq LLM                → synthesize into trade thesis
  7. Risk Engine v2          → regime + confidence + hurst + cvar gated decision
 
Usage:
    python orchestrator_v2.py RELIANCE
    python orchestrator_v2.py RELIANCE TCS INFY
    python orchestrator_v2.py --all
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
from agents.kalman_momentum_agent         import get_latest_signal as kalman_signal
from agents.advanced_mean_reversion_agent import get_latest_signal as adv_mr_signal
from agents.ml_agent                      import get_latest_signal as ml_signal
from agents.gbm_monte_carlo_agent         import get_latest_signal as mc_signal
from risk.risk_engine                     import evaluate_trade, Portfolio, RiskConfig
 
DB_PATH = "data/nifty50.db"
 
 
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
        SELECT Date, Close, Daily_Return
        FROM prices WHERE Symbol='{symbol}'
        ORDER BY Date DESC LIMIT {days}
    """, conn)
    conn.close()
    df = df.sort_values("Date")
    lines = []
    for _, r in df.iterrows():
        ret = f"{r['Daily_Return']:+.2%}" if pd.notna(r['Daily_Return']) else "N/A"
        lines.append(f"  {r['Date']}: ₹{r['Close']:.2f} ({ret})")
    return "\n".join(lines)
 
 
def call_groq(prompt):
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
                    "content": """You are a senior quantitative analyst at an AI hedge fund.
 
You receive outputs from 5 mathematical agents plus market regime:
  - Kalman Momentum: trend via Kalman filter + ADX strength
  - Advanced Mean Reversion: ADF-tested + Hurst exponent + pairs cointegration
  - ML Classifier: Random Forest probability
  - GBM Monte Carlo: 10,000 simulation probability distribution
  - Market Regime: HMM-detected state (bull/high-vol/bear)
 
Synthesize into one decision. Respond ONLY in valid JSON:
{
  "final_score": <float -1 to 1>,
  "recommendation": "<STRONG BUY | BUY | HOLD | SELL | STRONG SELL>",
  "thesis": "<2-3 sentences>",
  "risks": "<key risks>",
  "confidence": "<HIGH | MEDIUM | LOW>"
}"""
                },
                {"role": "user", "content": prompt}
            ]
        }
    )
    if response.status_code != 200:
        raise Exception(f"Groq error {response.status_code}: {response.text}")
    return response.json()["choices"][0]["message"]["content"]
 
 
def build_prompt(symbol, regime, signals, price_ctx):
    mc = signals["monte_carlo"]
    mr = signals["adv_mr"]
    return f"""Analyze {symbol} for a trade decision.
 
MARKET REGIME: {regime['regime_name'].upper()} (regime {regime['regime']})
  {regime['description']}
  Probabilities: Bull={regime['probabilities'][0]:.0%}  HighVol={regime['probabilities'][1]:.0%}  Bear={regime['probabilities'][2]:.0%}
 
AGENT SIGNALS:
  Kalman Momentum    : score={signals['kalman']['score']:+.4f} | {signals['kalman']['interpretation']}
  Adv Mean Reversion : score={mr['score']:+.4f} | Hurst={mr.get('hurst','?')} | {mr['interpretation']}
  ML Classifier      : score={signals['ml']['score']:+.4f} | {signals['ml']['interpretation']}
  GBM Monte Carlo    : score={mc['score']:+.4f} | P(gain)={mc.get('prob_gain',0):.1%} | E[ret]={mc.get('expected_ret',0):+.2f}% | CVaR={mc.get('cvar_5pct',0):.2f}%
 
REGIME-WEIGHTED COMBINED: {signals['combined']:+.4f}
 
RECENT PRICE ACTION:
{price_ctx}"""
 
 
def run_orchestrator_v2(symbol, portfolio, config):
    print(f"\n{'═'*65}")
    print(f"  Analyzing: {symbol}")
    print(f"{'═'*65}")
 
    # Step 1: Market regime
    print("  [1/5] Detecting market regime (HMM)...")
    try:
        regime  = get_regime()
        weights = regime["agent_weights"]
        print(f"         Regime: {regime['regime_name']} — {regime['description']}")
    except Exception as e:
        print(f"         Regime failed: {e} — using defaults")
        regime  = {
            "regime": 0, "regime_name": "unknown",
            "probabilities": [0.33, 0.33, 0.34],
            "description": "fallback",
            "agent_weights": {"momentum": 0.35, "mean_reversion": 0.30, "ml_classifier": 0.35}
        }
        weights = regime["agent_weights"]
 
    # Step 2: All signal agents
    print("  [2/5] Running signal agents...")
 
    try:
        k_sig = kalman_signal(symbol)
        print(f"         Kalman momentum : {k_sig['score']:+.4f}")
    except Exception as e:
        print(f"         Kalman failed: {e}")
        k_sig = {"score": 0.0, "interpretation": "failed", "agent": "kalman_momentum"}
 
    try:
        mr_sig = adv_mr_signal(symbol)
        print(f"         Adv mean rev    : {mr_sig['score']:+.4f}  H={mr_sig.get('hurst','?')}")
    except Exception as e:
        print(f"         Mean rev failed: {e}")
        mr_sig = {"score": 0.0, "interpretation": "failed", "hurst": 0.5,
                  "pairs_score": 0.0, "agent": "advanced_mean_reversion"}
 
    try:
        ml_sig = ml_signal(symbol)
        print(f"         ML classifier   : {ml_sig['score']:+.4f}")
    except Exception as e:
        print(f"         ML failed: {e}")
        ml_sig = {"score": 0.0, "interpretation": "failed", "agent": "ml_classifier"}
 
    try:
        mc_sig = mc_signal(symbol)
        print(f"         GBM Monte Carlo : {mc_sig['score']:+.4f}  P(gain)={mc_sig.get('prob_gain',0):.1%}")
    except Exception as e:
        print(f"         Monte Carlo failed: {e}")
        mc_sig = {"score": 0.0, "interpretation": "failed", "agent": "gbm_monte_carlo",
                  "prob_gain": 0.5, "expected_ret": 0.0, "cvar_5pct": 0.0}
 
    # Step 3: Regime-weighted combined score
    mc_weight    = 0.20
    regime_scale = 1.0 - mc_weight
 
    combined = float(np.clip(
        weights["momentum"]       * regime_scale * k_sig["score"]  +
        weights["mean_reversion"] * regime_scale * mr_sig["score"] +
        weights["ml_classifier"]  * regime_scale * ml_sig["score"] +
        mc_weight * mc_sig["score"],
        -1, 1
    ))
 
    signals = {
        "kalman":      k_sig,
        "adv_mr":      mr_sig,
        "ml":          ml_sig,
        "monte_carlo": mc_sig,
        "combined":    round(combined, 4),
        "weights":     weights,
    }
    print(f"         Combined score  : {combined:+.4f}")
 
    # Step 4: LLM synthesis
    print("  [3/5] Calling Groq LLM...")
    price_ctx = get_price_context(symbol)
    prompt    = build_prompt(symbol, regime, signals, price_ctx)
 
    try:
        raw          = call_groq(prompt)
        llm_decision = json.loads(raw)
    except Exception as e:
        print(f"         LLM failed: {e} — using combined score")
        rec = "BUY" if combined > 0.1 else ("SELL" if combined < -0.1 else "HOLD")
        llm_decision = {
            "final_score": combined, "recommendation": rec,
            "thesis": "LLM unavailable.", "risks": "N/A", "confidence": "LOW",
        }
 
    print(f"         LLM: {llm_decision['recommendation']} "
          f"score={llm_decision['final_score']:+.4f} "
          f"confidence={llm_decision.get('confidence','?')}")
    print(f"         Thesis: {llm_decision['thesis'][:90]}...")
 
    # Step 5: Risk engine v2 — fully wired
    print("  [4/5] Risk engine v2...")
    current_price = get_current_price(symbol)
 
    trade = evaluate_trade(
        symbol        = symbol,
        signal_score  = llm_decision["final_score"],
        current_price = current_price,
        portfolio     = portfolio,
        config        = config,
        regime        = regime["regime"],
        confidence    = llm_decision.get("confidence", "MEDIUM"),
        hurst         = float(mr_sig.get("hurst", 0.5)),
        cvar_mc       = float(mc_sig.get("cvar_5pct", 0.0)),
    )
 
    # Step 6: Final output
    print("  [5/5] Final decision:")
    if trade.approved:
        print(f"         ✓ APPROVED → {trade.action} {trade.shares} shares "
              f"@ ₹{current_price:.2f} | ₹{trade.capital_allocated:,.0f}")
    else:
        print(f"         ✗ BLOCKED  → {trade.rejection_reason}")
 
    return {
        "symbol":       symbol,
        "regime":       regime,
        "signals":      signals,
        "llm_decision": llm_decision,
        "trade":        asdict(trade),
        "price":        current_price,
    }
 
 
def run_all(portfolio, config, top_n=5):
    conn    = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT Symbol FROM prices").fetchall()]
    conn.close()
    results = []
    for sym in symbols[:top_n]:
        results.append(run_orchestrator_v2(sym, portfolio, config))
    return results
 
 
if __name__ == "__main__":
    portfolio = Portfolio(cash=100_000)
    config    = RiskConfig(total_capital=100_000)
 
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python orchestrator_v2.py RELIANCE")
        print("  python orchestrator_v2.py RELIANCE TCS INFY")
        print("  python orchestrator_v2.py --all")
        sys.exit(0)
 
    if sys.argv[1] == "--all":
        results = run_all(portfolio, config, top_n=5)
    else:
        results = [run_orchestrator_v2(s.upper(), portfolio, config) for s in sys.argv[1:]]
 
    print(f"\n{'═'*70}")
    print("FINAL SUMMARY")
    print(f"{'═'*70}")
    print(f"{'Stock':<12} {'Regime':<20} {'Score':>7} {'Rec':<12} {'Conf':<8} {'Action'}")
    print("─" * 70)
    for r in results:
        if "error" in r: continue
        print(
            f"{r['symbol']:<12} "
            f"{r['regime']['regime_name'][:18]:<20} "
            f"{r['llm_decision']['final_score']:>+7.4f} "
            f"{r['llm_decision']['recommendation']:<12} "
            f"{r['llm_decision'].get('confidence','?'):<8} "
            f"{'✓ '+r['trade']['action'] if r['trade']['approved'] else '✗ BLOCKED'}"
        )