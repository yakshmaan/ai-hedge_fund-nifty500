"""
orchestrator_v4.py
------------------
V4 Orchestrator — 9 agents total:
  1. HMM Regime Detection
  2. Kalman Momentum
  3. Advanced Mean Reversion
  4. LSTM Sequential
  5. Transformer Attention
  6. Heston Stochastic Vol
  7. GBM Monte Carlo
  8. FinBERT Sentiment
  9. RL DQN (optimizes weights dynamically)

Usage:
    python orchestrator_v4.py RELIANCE
    python orchestrator_v4.py --all
"""

import sys, os, json, sqlite3, requests
import pandas as pd
import numpy as np
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.regime_agent                  import get_regime
from agents.kalman_momentum_agent         import get_latest_signal as kalman_sig
from agents.advanced_mean_reversion_agent import get_latest_signal as adv_mr_sig
from agents.lstm_agent                    import get_latest_signal as lstm_sig
from agents.transformer_attention_agent   import get_latest_signal as transformer_sig
from agents.heston_agent                  import get_latest_signal as heston_sig
from agents.gbm_monte_carlo_agent         import get_latest_signal as mc_sig
from agents.finbert_sentiment_agent       import get_latest_signal as sentiment_sig
from agents.rl_agent                      import get_latest_signal as rl_sig, get_optimal_weights
from risk.risk_engine_v3                  import evaluate_trade_v3, Portfolio, RiskConfig

DB_PATH      = "data/nifty50.db"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def get_current_price(symbol):
    conn   = sqlite3.connect(DB_PATH)
    result = conn.execute(
        "SELECT Close FROM prices WHERE Symbol=? ORDER BY Date DESC LIMIT 1",(symbol,)
    ).fetchone()
    conn.close()
    if result is None: raise ValueError(f"No price data for {symbol}")
    return float(result[0])


def get_price_context(symbol, days=10):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"""
        SELECT Date, Close, Daily_Return FROM prices
        WHERE Symbol='{symbol}' ORDER BY Date DESC LIMIT {days}
    """, conn)
    conn.close()
    df = df.sort_values("Date")
    return "\n".join(
        f"  {r['Date']}: ₹{r['Close']:.2f} ({r['Daily_Return']:+.2%})"
        if pd.notna(r['Daily_Return']) else f"  {r['Date']}: ₹{r['Close']:.2f}"
        for _, r in df.iterrows()
    )


def call_groq(prompt):
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Content-Type":"application/json",
                 "Authorization":f"Bearer {GROQ_API_KEY}"},
        json={"model":"llama-3.3-70b-versatile","max_tokens":800,
              "messages":[
                  {"role":"system","content":"""You are a senior quant analyst.
You receive outputs from 9 mathematical agents including RL-optimized weights.
Respond ONLY in valid JSON:
{"final_score":float,"recommendation":"STRONG BUY|BUY|HOLD|SELL|STRONG SELL",
 "thesis":"2-3 sentences","risks":"key risks",
 "confidence":"HIGH|MEDIUM|LOW","key_driver":"agent name"}"""},
                  {"role":"user","content":prompt}
              ]},
        timeout=20,
    )
    if resp.status_code != 200: raise Exception(f"Groq {resp.status_code}")
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"): content = content[4:]
    return json.loads(content.strip())


def run_orchestrator_v4(symbol, portfolio, config):
    print(f"\n{'═'*65}")
    print(f"  {symbol} [V4 — 9 Agents + RL Weighting]")
    print(f"{'═'*65}")

    # Regime
    print("  [1/9] HMM Regime...")
    try:
        regime  = get_regime()
        weights = regime["agent_weights"]
        print(f"         {regime['regime_name']}")
    except:
        regime  = {"regime":0,"regime_name":"unknown","probabilities":[0.33,0.33,0.34],
                   "description":"fallback","agent_weights":{"momentum":0.35,"mean_reversion":0.30,"ml_classifier":0.35}}
        weights = regime["agent_weights"]

    signals = {}

    # All agents
    for step, (name, func, label) in enumerate([
        ("kalman",      kalman_sig,      "Kalman Momentum"),
        ("adv_mr",      adv_mr_sig,      "Advanced Mean Rev"),
        ("lstm",        lstm_sig,        "LSTM Sequential"),
        ("transformer", transformer_sig, "Transformer Attn"),
        ("heston",      heston_sig,      "Heston Stoch Vol"),
        ("monte_carlo", mc_sig,          "GBM Monte Carlo"),
        ("sentiment",   sentiment_sig,   "FinBERT Sentiment"),
    ], start=2):
        print(f"  [{step}/9] {label}...")
        try:
            signals[name] = func(symbol)
            score = signals[name]["score"]
            print(f"         {score:+.4f}")
        except Exception as e:
            signals[name] = {"score":0.0,"interpretation":f"failed:{e}","hurst":0.5}
            print(f"         failed: {e}")

    # RL agent — uses all other scores to determine optimal weights
    print("  [9/9] RL DQN Weight Optimizer...")
    try:
        agent_score_map = {
            "kalman":      signals["kalman"]["score"],
            "mean_rev":    signals["adv_mr"]["score"],
            "lstm":        signals["lstm"]["score"],
            "heston":      signals["heston"]["score"],
            "monte_carlo": signals["monte_carlo"]["score"],
            "sentiment":   signals["sentiment"]["score"],
        }

        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql(
            f"SELECT Daily_Return FROM prices WHERE Symbol='{symbol}' ORDER BY Date DESC LIMIT 60",
            conn
        )
        conn.close()
        ret = df["Daily_Return"].dropna()
        market_features = {
            "vol":      float(np.clip(ret.std()*50, 0, 1)),
            "trend":    float(np.clip(ret.mean()*100, -1, 1)),
            "regime":   float(regime["regime"]) / 2.0,
            "momentum": float(np.clip((ret.iloc[-5:].mean()-ret.mean())*100, -1, 1)),
        }

        rl_result = rl_sig(symbol, agent_score_map, market_features)
        signals["rl"] = rl_result
        rl_weights    = rl_result.get("weights", {})
        print(f"         RL weights: {rl_weights}")
        print(f"         RL Q-value: {rl_result.get('q_value',0):.4f}")
    except Exception as e:
        print(f"         RL failed: {e}")
        rl_weights = {}
        signals["rl"] = {"score":0.0,"interpretation":f"failed:{e}","weights":{}}

    # Combined score using RL weights if available, else fixed
    if rl_weights:
        combined = float(np.clip(
            rl_weights.get("kalman",0.25)      * signals["kalman"]["score"]      +
            rl_weights.get("mean_rev",0.20)    * signals["adv_mr"]["score"]      +
            rl_weights.get("lstm",0.18)        * signals["lstm"]["score"]        +
            rl_weights.get("heston",0.12)      * signals["heston"]["score"]      +
            rl_weights.get("monte_carlo",0.13) * signals["monte_carlo"]["score"] +
            rl_weights.get("sentiment",0.12)   * signals["sentiment"]["score"],
            -1, 1
        ))
    else:
        combined = float(np.clip(
            0.25*signals["kalman"]["score"]      +
            0.20*signals["adv_mr"]["score"]      +
            0.15*signals["lstm"]["score"]        +
            0.10*signals["transformer"]["score"] +
            0.12*signals["heston"]["score"]      +
            0.10*signals["monte_carlo"]["score"] +
            0.08*signals["sentiment"]["score"],
            -1, 1
        ))

    signals["combined"] = round(combined, 4)
    print(f"\n  Combined (RL-weighted): {combined:+.4f}")

    # LLM
    print("  Calling Groq LLM...")
    mc   = signals["monte_carlo"]
    mr   = signals["adv_mr"]
    sent = signals["sentiment"]
    prompt = f"""Analyze {symbol} — V4 System (9 agents + RL weighting).

REGIME: {regime['regime_name']} | {regime['description']}

AGENT SIGNALS:
  Kalman Momentum    : {signals['kalman']['score']:+.4f} | {signals['kalman']['interpretation']}
  Adv Mean Reversion : {mr['score']:+.4f} | H={mr.get('hurst','?')} | {mr['interpretation']}
  LSTM               : {signals['lstm']['score']:+.4f} | {signals['lstm']['interpretation']}
  Transformer Attn   : {signals['transformer']['score']:+.4f} | {signals['transformer']['interpretation']}
  Heston Stoch Vol   : {signals['heston']['score']:+.4f} | {signals['heston']['interpretation']}
  GBM Monte Carlo    : {mc['score']:+.4f} | P(gain)={mc.get('prob_gain',0):.1%} | CVaR={mc.get('cvar_5pct',0):.2f}%
  FinBERT Sentiment  : {sent['score']:+.4f} | {sent['interpretation']}
  RL DQN             : Q={signals['rl'].get('q_value',0):.4f} | {signals['rl']['interpretation']}

RL OPTIMAL WEIGHTS: {rl_weights}
COMBINED SCORE: {combined:+.4f}

RECENT PRICE:
{get_price_context(symbol)}"""

    try:
        llm = call_groq(prompt)
    except Exception as e:
        rec = "BUY" if combined>0.1 else ("SELL" if combined<-0.1 else "HOLD")
        mr  = signals["adv_mr"]
        llm = {"final_score":combined,"recommendation":rec,
               "thesis":f"V4 combined={combined:+.4f} using RL-optimized weights.",
               "risks":f"Hurst={mr.get('hurst','?')}. CVaR={signals['monte_carlo'].get('cvar_5pct',0):.2f}%.",
               "confidence":"MEDIUM","key_driver":"combined"}

    print(f"  LLM: {llm['recommendation']} score={llm['final_score']:+.4f} conf={llm.get('confidence','?')}")

    # Risk engine v3
    current_price = get_current_price(symbol)
    all_scores = {k: signals[k]["score"] for k in ["kalman","adv_mr","lstm","heston","monte_carlo","sentiment"]}

    trade = evaluate_trade_v3(
        symbol=symbol, signal_score=llm["final_score"],
        current_price=current_price, portfolio=portfolio, config=config,
        regime=regime["regime"], confidence=llm.get("confidence","MEDIUM"),
        hurst=float(signals["adv_mr"].get("hurst",0.5)),
        cvar_mc=float(signals["monte_carlo"].get("cvar_5pct",0.0)),
        heston_score=float(signals["heston"]["score"]),
        heston_vol_ratio=float(signals["heston"].get("vol_ratio",1.0)),
        lstm_score=float(signals["lstm"]["score"]),
        sentiment_score=float(signals["sentiment"]["score"]),
        all_agent_scores=all_scores,
    )

    if trade.approved:
        print(f"  ✓ APPROVED → {trade.action} {trade.shares} shares @ ₹{current_price:.2f}")
        print(f"    Consensus={trade.consensus_score:.2f} Heston={trade.heston_regime}")
    else:
        print(f"  ✗ BLOCKED  → {trade.rejection_reason}")

    return {"symbol":symbol,"regime":regime,"signals":signals,
            "llm":llm,"trade":asdict(trade),"price":current_price}


def run_all(portfolio, config, top_n=5):
    conn    = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT Symbol FROM prices").fetchall()]
    conn.close()
    return [run_orchestrator_v4(s, portfolio, config) for s in symbols[:top_n]]


if __name__ == "__main__":
    portfolio = Portfolio(cash=100_000)
    config    = RiskConfig(total_capital=100_000)

    if len(sys.argv) < 2:
        print("Usage: python orchestrator_v4.py RELIANCE")
        print("       python orchestrator_v4.py --all")
        sys.exit(0)

    if sys.argv[1] == "--all":
        results = run_all(portfolio, config, top_n=5)
    else:
        results = [run_orchestrator_v4(s.upper(), portfolio, config) for s in sys.argv[1:]]

    print(f"\n{'═'*70}\nV4 SUMMARY\n{'═'*70}")
    print(f"{'Stock':<12}{'Score':>8}{'Rec':<14}{'Conf':<8}{'Consensus':>10}{'Action':>12}")
    print("─"*70)
    for r in results:
        if "error" in r: continue
        cons = r["trade"].get("consensus_score", 1.0)
        print(f"{r['symbol']:<12}{r['llm']['final_score']:>+8.4f}"
              f"{r['llm']['recommendation']:<14}{r['llm'].get('confidence','?'):<8}"
              f"{cons:>10.2f}"
              f"{'✓ '+r['trade']['action'] if r['trade']['approved'] else '✗ BLOCKED':>12}")
