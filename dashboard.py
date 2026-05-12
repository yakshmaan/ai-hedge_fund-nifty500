"""
dashboard.py - AI Hedge Fund Dashboard V2
Full system: 9 Agents + Portfolio Allocator V3 + Nifty 500 + SIP Calculator
"""

import os
import sys
import json
import sqlite3
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dataclasses import asdict


GROQ_API_KEY = ""

DB_PATH_50  = "data/nifty50.db"
DB_PATH_500 = "data/nifty500.db"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="AI Hedge Fund — Nifty 50 & 500",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
.block-container{padding-top:1rem}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {padding: 8px 20px; border-radius: 8px;}
</style>""", unsafe_allow_html=True)


# ── DATA LOADERS ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_symbols_50():
    if not os.path.exists(DB_PATH_50): return []
    conn = sqlite3.connect(DB_PATH_50)
    syms = [r[0] for r in conn.execute("SELECT DISTINCT Symbol FROM prices ORDER BY Symbol").fetchall()]
    conn.close()
    return syms


@st.cache_data(ttl=300)
def get_symbols_500():
    if not os.path.exists(DB_PATH_500): return []
    conn = sqlite3.connect(DB_PATH_500)
    syms = [r[0] for r in conn.execute("SELECT DISTINCT Symbol FROM prices ORDER BY Symbol").fetchall()]
    conn.close()
    return syms


@st.cache_data(ttl=60)
def get_price_history(symbol, db_path, days=180):
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql(f"""
        SELECT Date, Open, High, Low, Close, Volume, Daily_Return
        FROM prices WHERE Symbol='{symbol}'
        ORDER BY Date DESC LIMIT {days}
    """, conn)
    conn.close()
    return df.sort_values("Date").reset_index(drop=True)


def load_equity_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def load_trade_log(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def load_sip_log(path):
    return pd.read_csv(path) if os.path.exists(path) else None


# ── CHART HELPERS ─────────────────────────────────────────────────────────────

def candlestick_chart(df, symbol):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00c853", decreasing_line_color="#ff1744", name=symbol,
    ))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], line=dict(color="#2196f3", width=1.5), name="MA20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], line=dict(color="#ff9800", width=1.5), name="MA50"))
    fig.update_layout(
        title=f"{symbol} — Price Chart", xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e0e1a", plot_bgcolor="#0e0e1a",
        font=dict(color="#ccc"), height=400, margin=dict(t=40,b=20,l=20,r=20),
    )
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e")
    return fig


def equity_chart(df, value_col="portfolio_value", title="Portfolio Equity Curve", color="#00c853"):
    fig = go.Figure()
    date_col = "date" if "date" in df.columns else df.columns[0]
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[value_col],
        fill="tozeroy", line=dict(color=color, width=2),
        fillcolor=f"rgba(0,200,83,0.1)", name="Portfolio",
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor="#0e0e1a", plot_bgcolor="#0e0e1a",
        font=dict(color="#ccc"), height=300, margin=dict(t=40,b=20,l=20,r=20),
    )
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e", tickprefix="₹")
    return fig


def multi_equity_chart(curves, title="Strategy Comparison"):
    colors = ["#00c853", "#2196f3", "#ff9800", "#e040fb", "#ff1744"]
    fig    = go.Figure()
    for i, (label, df, col) in enumerate(curves):
        if df is None: continue
        date_col = "date" if "date" in df.columns else df.columns[0]
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[col],
            line=dict(color=colors[i % len(colors)], width=2),
            name=label,
        ))
    fig.update_layout(
        title=title,
        paper_bgcolor="#0e0e1a", plot_bgcolor="#0e0e1a",
        font=dict(color="#ccc"), height=400, margin=dict(t=40,b=20,l=20,r=20),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e", tickprefix="₹")
    return fig


def signal_bar(score, label, note=""):
    color = "#00c853" if score >= 0 else "#ff1744"
    width = f"{abs(score)*100:.0f}%"
    left  = "50%" if score >= 0 else f"{(0.5+score/2)*100:.0f}%"
    st.markdown(f"""
    <div style='margin-bottom:10px'>
        <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
            <span style='font-size:13px;color:#aaa'>{label}</span>
            <span style='font-size:13px;font-weight:bold;color:{color}'>{score:+.4f}</span>
        </div>
        <div style='background:#2a2a3e;border-radius:4px;height:8px;position:relative'>
            <div style='position:absolute;left:50%;width:1px;height:8px;background:#555'></div>
            <div style='position:absolute;left:{left};width:{width};height:8px;background:{color};border-radius:4px'></div>
        </div>
        <div style='font-size:11px;color:#666;margin-top:2px'>{note}</div>
    </div>""", unsafe_allow_html=True)


# ── GROQ + ANALYSIS ───────────────────────────────────────────────────────────

def call_groq(prompt):
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY not set. Run: export GROQ_API_KEY=your_key")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Content-Type":"application/json","Authorization":f"Bearer {GROQ_API_KEY}"},
        json={"model":"llama-3.3-70b-versatile","max_tokens":600,
              "messages":[
                  {"role":"system","content":"You are a quant analyst. Respond ONLY in valid JSON: {\"final_score\":float,\"recommendation\":\"STRONG BUY|BUY|HOLD|SELL|STRONG SELL\",\"thesis\":\"2-3 sentences\",\"risks\":\"1-2 sentences\",\"confidence\":\"HIGH|MEDIUM|LOW\"}"},
                  {"role":"user","content":prompt}
              ]},
        timeout=15,
    )
    if resp.status_code != 200: raise Exception(f"Groq {resp.status_code}")
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"): content = content[4:]
    return json.loads(content.strip())


def run_analysis(symbol):
    try:
        from agents.regime_agent                  import get_regime
        from agents.kalman_momentum_agent         import get_latest_signal as kalman_sig
        from agents.advanced_mean_reversion_agent import get_latest_signal as adv_mr_sig
        from agents.lstm_agent                    import get_latest_signal as lstm_sig
        from agents.transformer_attention_agent   import get_latest_signal as transformer_sig
        from agents.heston_agent                  import get_latest_signal as heston_sig
        from agents.gbm_monte_carlo_agent         import get_latest_signal as mc_sig
        from agents.finbert_sentiment_agent       import get_latest_signal as sentiment_sig
        from agents.rl_agent                      import get_latest_signal as rl_sig
        from risk.risk_engine_v4                  import evaluate_trade_v4, Portfolio, RiskConfig

        with st.spinner("HMM Regime Detection..."):
            try:
                regime  = get_regime()
                weights = regime["agent_weights"]
            except:
                regime  = {"regime":0,"regime_name":"low volatility bull","probabilities":[1.0,0.0,0.0],
                           "description":"Default","agent_weights":{"momentum":0.35,"mean_reversion":0.30,"ml_classifier":0.35}}
                weights = regime["agent_weights"]

        signals = {}
        with st.spinner("Kalman Momentum..."):
            try: signals["kalman"] = kalman_sig(symbol)
            except: signals["kalman"] = {"score":0.0,"interpretation":"failed"}

        with st.spinner("Advanced Mean Reversion..."):
            try: signals["adv_mr"] = adv_mr_sig(symbol)
            except: signals["adv_mr"] = {"score":0.0,"interpretation":"failed","hurst":0.5}

        with st.spinner("LSTM Sequential..."):
            try: signals["lstm"] = lstm_sig(symbol)
            except: signals["lstm"] = {"score":0.0,"interpretation":"failed"}

        with st.spinner("Transformer Attention..."):
            try: signals["transformer"] = transformer_sig(symbol)
            except: signals["transformer"] = {"score":0.0,"interpretation":"failed"}

        with st.spinner("Heston Stochastic Volatility..."):
            try: signals["heston"] = heston_sig(symbol)
            except: signals["heston"] = {"score":0.0,"interpretation":"failed","vol_ratio":1.0}

        with st.spinner("GBM Monte Carlo (10,000 sims)..."):
            try: signals["monte_carlo"] = mc_sig(symbol)
            except: signals["monte_carlo"] = {"score":0.0,"interpretation":"failed","prob_gain":0.5,"expected_ret":0.0,"cvar_5pct":0.0}

        with st.spinner("FinBERT Sentiment..."):
            try: signals["sentiment"] = sentiment_sig(symbol)
            except: signals["sentiment"] = {"score":0.0,"interpretation":"failed","n_headlines":0}

        with st.spinner("RL DQN Weight Optimizer..."):
            try:
                agent_score_map = {
                    "kalman":      signals["kalman"]["score"],
                    "mean_rev":    signals["adv_mr"]["score"],
                    "lstm":        signals["lstm"]["score"],
                    "heston":      signals["heston"]["score"],
                    "monte_carlo": signals["monte_carlo"]["score"],
                    "sentiment":   signals["sentiment"]["score"],
                }
                conn = sqlite3.connect(DB_PATH_50)
                ret  = pd.read_sql(f"SELECT Daily_Return FROM prices WHERE Symbol='{symbol}' ORDER BY Date DESC LIMIT 60",conn)["Daily_Return"].dropna()
                conn.close()
                market_features = {
                    "vol":      float(np.clip(ret.std()*50,0,1)),
                    "trend":    float(np.clip(ret.mean()*100,-1,1)),
                    "regime":   float(regime["regime"])/2.0,
                    "momentum": float(np.clip((ret.iloc[-5:].mean()-ret.mean())*100,-1,1)),
                }
                signals["rl"] = rl_sig(symbol, agent_score_map, market_features)
                rl_weights     = signals["rl"].get("weights",{})
            except Exception as e:
                signals["rl"] = {"score":0.0,"interpretation":f"failed:{e}","weights":{},"q_value":0.0}
                rl_weights = {}

        if rl_weights:
            combined = float(np.clip(
                rl_weights.get("kalman",0.25)      * signals["kalman"]["score"]      +
                rl_weights.get("mean_rev",0.20)    * signals["adv_mr"]["score"]      +
                rl_weights.get("lstm",0.18)        * signals["lstm"]["score"]        +
                rl_weights.get("heston",0.12)      * signals["heston"]["score"]      +
                rl_weights.get("monte_carlo",0.13) * signals["monte_carlo"]["score"] +
                rl_weights.get("sentiment",0.12)   * signals["sentiment"]["score"]   +
                0.05                               * signals["transformer"]["score"],
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

        with st.spinner("Groq LLM synthesis..."):
            try:
                mc   = signals["monte_carlo"]
                mr   = signals["adv_mr"]
                sent = signals["sentiment"]
                prompt = f"""Analyze {symbol} — V4 System (9 agents).
REGIME: {regime['regime_name']} | {regime['description']}
Kalman Momentum: {signals['kalman']['score']:+.4f} | {signals['kalman']['interpretation']}
Adv Mean Rev: {mr['score']:+.4f} | H={mr.get('hurst','?')} | {mr['interpretation']}
LSTM: {signals['lstm']['score']:+.4f} | {signals['lstm']['interpretation']}
Transformer: {signals['transformer']['score']:+.4f} | {signals['transformer']['interpretation']}
Heston: {signals['heston']['score']:+.4f} | {signals['heston']['interpretation']}
GBM Monte Carlo: {mc['score']:+.4f} | P(gain)={mc.get('prob_gain',0):.1%} | CVaR={mc.get('cvar_5pct',0):.2f}%
FinBERT Sentiment: {sent['score']:+.4f} | {sent['interpretation']}
RL DQN: Q={signals['rl'].get('q_value',0):.4f} | weights={rl_weights}
Combined: {combined:+.4f}"""
                llm = call_groq(prompt)
            except Exception as e:
                rec = "BUY" if combined>0.1 else ("SELL" if combined<-0.1 else "HOLD")
                llm = {
                    "final_score":combined,"recommendation":rec,"confidence":"MEDIUM",
                    "thesis":f"V4 combined={combined:+.4f}. Kalman={signals['kalman']['score']:+.3f}, LSTM={signals['lstm']['score']:+.3f}, MC P(gain)={signals['monte_carlo'].get('prob_gain',0):.1%}.",
                    "risks":f"Hurst={mr.get('hurst','?')}. CVaR={signals['monte_carlo'].get('cvar_5pct',0):.2f}%.",
                }

        conn  = sqlite3.connect(DB_PATH_50)
        price = float(conn.execute("SELECT Close FROM prices WHERE Symbol=? ORDER BY Date DESC LIMIT 1",(symbol,)).fetchone()[0])
        conn.close()

        portfolio = Portfolio(cash=100_000)
        config    = RiskConfig(total_capital=100_000)
        all_scores = {k: signals[k]["score"] for k in ["kalman","adv_mr","lstm","heston","monte_carlo","sentiment","transformer"]}

        trade = evaluate_trade_v4(
            symbol=symbol, signal_score=llm["final_score"],
            current_price=price, portfolio=portfolio, config=config,
            regime=regime["regime"], confidence=llm.get("confidence","MEDIUM"),
            hurst=float(signals["adv_mr"].get("hurst",0.5)),
            cvar_mc=float(signals["monte_carlo"].get("cvar_5pct",0.0)),
            heston_score=float(signals["heston"]["score"]),
            heston_vol_ratio=float(signals["heston"].get("vol_ratio",1.0)),
            lstm_score=float(signals["lstm"]["score"]),
            sentiment_score=float(signals["sentiment"]["score"]),
            transformer_score=float(signals["transformer"]["score"]),
            rl_q_value=float(signals["rl"].get("q_value",0.0)),
            all_agent_scores=all_scores,
        )

        return {"regime":regime,"signals":signals,"llm":llm,"trade":asdict(trade),"price":price}

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def run_forecast(symbol):
    try:
        from forecaster import run_forecast as _forecast
        return _forecast(symbol)
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        return None


def forecast_chart(fc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["hist_dates"],y=fc["hist_close"],
        line=dict(color="#aaaaaa",width=1.5),name="Historical"))
    fig.add_trace(go.Scatter(x=fc["hist_dates"],y=fc["smoothed_hist"],
        line=dict(color="#2196f3",width=2,dash="dot"),name="Kalman Smoothed"))
    fd = fc["future_dates"]
    p5=fc["p5"][1:]; p25=fc["p25"][1:]; p50=fc["p50"][1:]; p75=fc["p75"][1:]; p95=fc["p95"][1:]
    fig.add_trace(go.Scatter(x=fd+fd[::-1],y=p95+p5[::-1],fill="toself",
        fillcolor="rgba(33,150,243,0.08)",line=dict(color="rgba(0,0,0,0)"),name="90% Confidence"))
    fig.add_trace(go.Scatter(x=fd+fd[::-1],y=p75+p25[::-1],fill="toself",
        fillcolor="rgba(33,150,243,0.18)",line=dict(color="rgba(0,0,0,0)"),name="50% Confidence"))
    fig.add_trace(go.Scatter(x=fd,y=p50,line=dict(color="#00c853",width=2.5,dash="dash"),name="Base Case"))
    fig.add_trace(go.Scatter(x=fd,y=p75,line=dict(color="#69f0ae",width=1.5,dash="dot"),name=f"Bull ({fc['bull_ret']:+.1f}%)"))
    fig.add_trace(go.Scatter(x=fd,y=p25,line=dict(color="#ff5252",width=1.5,dash="dot"),name=f"Bear ({fc['bear_ret']:+.1f}%)"))
    fig.add_hline(y=fc["resistance"],line_dash="dash",line_color="#ff9800",opacity=0.7,
        annotation_text=f"Resistance ₹{fc['resistance']}",annotation_position="right")
    fig.add_hline(y=fc["support"],line_dash="dash",line_color="#2196f3",opacity=0.7,
        annotation_text=f"Support ₹{fc['support']}",annotation_position="right")
    fig.update_layout(
        title=f"{fc['symbol']} — 30-Day GBM Forecast ({fc['n_sims']:,} simulations)",
        paper_bgcolor="#0e0e1a",plot_bgcolor="#0e0e1a",
        font=dict(color="#ccc"),height=500,margin=dict(t=50,b=20,l=20,r=80),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e",tickprefix="₹")
    return fig


# ── SIP CALCULATOR ────────────────────────────────────────────────────────────

def sip_calculator(initial, monthly_bull, monthly_bear, annual_return, years):
    """
    Simulates Flexi-SIP returns.
    Assumes 80% of months are bull, 20% bear (based on our backtest).
    """
    monthly_rate = annual_return / 12
    portfolio    = initial
    total_invested = initial
    monthly_values = [initial]
    invested_values = [initial]

    bull_months = int(years * 12 * 0.79)
    bear_months = int(years * 12 * 0.21)
    schedule    = ([monthly_bull] * bull_months + [monthly_bear] * bear_months)

    for i, sip in enumerate(schedule):
        portfolio      = portfolio * (1 + monthly_rate) + sip
        total_invested += sip
        monthly_values.append(round(portfolio, 2))
        invested_values.append(round(total_invested, 2))

    return {
        "final_value":    round(portfolio, 2),
        "total_invested": round(total_invested, 2),
        "total_profit":   round(portfolio - total_invested, 2),
        "return_pct":     round((portfolio - total_invested) / total_invested * 100, 2),
        "monthly_values": monthly_values,
        "invested_values": invested_values,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # Sidebar
    st.sidebar.title("📈 AI Hedge Fund")
    st.sidebar.markdown("**Nifty 50 + Nifty 500 · Dual DB**")
    st.sidebar.markdown("---")

    symbols_50  = get_symbols_50()
    symbols_500 = get_symbols_500()

    st.sidebar.markdown("**Stock Analysis**")
    db_choice      = st.sidebar.radio("Database", ["Nifty 50", "Nifty 500"], horizontal=True)
    symbols        = symbols_50 if db_choice == "Nifty 50" else symbols_500
    db_path        = DB_PATH_50 if db_choice == "Nifty 50" else DB_PATH_500
    selected_stock = st.sidebar.selectbox("Select Stock", symbols) if symbols else None

    analysis_btn = st.sidebar.button("🔍 Run Full Analysis (Nifty 50)", type="primary", use_container_width=True)
    forecast_btn = st.sidebar.button("🔮 Run Forecasting",              type="secondary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**V4 Agent Pipeline**")
    st.sidebar.markdown("""
1. HMM Regime Detection
2. Kalman Filter Momentum
3. ADF + Hurst Mean Reversion
4. LSTM Sequential
5. Transformer Attention
6. Heston Stochastic Vol
7. GBM Monte Carlo
8. FinBERT Sentiment
9. RL DQN Weight Optimizer
    """)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    # Main title
    st.title("AI Hedge Fund Dashboard")
    st.caption("Nifty 50 + Nifty 500 · 9-Agent V4 System · Portfolio Allocator V3 · Flexi-SIP")
    st.markdown("---")

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Analysis",
        "📈 Portfolio Results",
        "🌐 Nifty 500 Passive",
        "💰 SIP Calculator",
        "🏆 System Comparison",
    ])

    # ── TAB 1: LIVE ANALYSIS ──────────────────────────────────────────────────
    with tab1:
        if selected_stock:
            col_chart, col_info = st.columns([2,1])
            with col_chart:
                price_df = get_price_history(selected_stock, db_path)
                if not price_df.empty:
                    st.plotly_chart(candlestick_chart(price_df, selected_stock), use_container_width=True)
            with col_info:
                if not price_df.empty:
                    latest = price_df.iloc[-1]; prev = price_df.iloc[-2] if len(price_df)>1 else latest
                    change = (latest["Close"]-prev["Close"])/prev["Close"]*100
                    st.metric(f"{selected_stock}", f"₹{latest['Close']:.2f}", f"{change:+.2f}%")
                    st.metric("52W High", f"₹{price_df['High'].max():.2f}")
                    st.metric("52W Low",  f"₹{price_df['Low'].min():.2f}")
                    st.metric("Avg Volume", f"{price_df['Volume'].mean()/1e6:.1f}M")

            if forecast_btn:
                st.markdown("---")
                st.markdown("## 🔮 30-Day Price Forecast")
                with st.spinner(f"Running 1,000 GBM simulations for {selected_stock}..."):
                    fc = run_forecast(selected_stock)
                if fc:
                    st.plotly_chart(forecast_chart(fc), use_container_width=True)
                    t1,t2,t3,t4 = st.columns(4)
                    t1.markdown(f"<div style='background:#1e1e2e;border-radius:10px;padding:1rem;text-align:center'><div style='color:#aaa;font-size:0.85rem'>Trend</div><div style='color:{fc['trend_color']};font-weight:bold'>{fc['trend']}</div></div>",unsafe_allow_html=True)
                    t2.markdown(f"<div style='background:#1e1e2e;border-radius:10px;padding:1rem;text-align:center'><div style='color:#aaa;font-size:0.85rem'>🐂 Bull</div><div style='color:#00c853;font-weight:bold'>₹{fc['bull_target']} ({fc['bull_ret']:+.1f}%)</div></div>",unsafe_allow_html=True)
                    t3.markdown(f"<div style='background:#1e1e2e;border-radius:10px;padding:1rem;text-align:center'><div style='color:#aaa;font-size:0.85rem'>📊 Base</div><div style='color:#ffa000;font-weight:bold'>₹{fc['base_target']} ({fc['base_ret']:+.1f}%)</div></div>",unsafe_allow_html=True)
                    t4.markdown(f"<div style='background:#1e1e2e;border-radius:10px;padding:1rem;text-align:center'><div style='color:#aaa;font-size:0.85rem'>🐻 Bear</div><div style='color:#ff1744;font-weight:bold'>₹{fc['bear_target']} ({fc['bear_ret']:+.1f}%)</div></div>",unsafe_allow_html=True)

            if analysis_btn and db_choice == "Nifty 50":
                st.markdown("---")
                st.markdown("## 🔍 V4 Full Analysis — 9 Agents")
                result = run_analysis(selected_stock)

                if result:
                    regime = result["regime"]
                    icons  = {"low volatility bull":"🟢","high volatility bull":"🟡","bear / crisis":"🔴"}
                    icon   = icons.get(regime["regime_name"],"⚪")
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Market Regime", f"{icon} {regime['regime_name'].title()}")
                    c2.metric("Bull Prob",  f"{regime['probabilities'][0]:.0%}")
                    c3.metric("Bear Prob",  f"{regime['probabilities'][2]:.0%}")
                    c4.metric("Combined Score", f"{result['signals']['combined']:+.4f}")
                    st.caption(regime["description"])
                    st.markdown("---")

                    sig_col, dec_col = st.columns([1,1])

                    with sig_col:
                        st.markdown("### 9 Agent Signals")
                        s = result["signals"]
                        signal_bar(s["kalman"]["score"],      "1. Kalman Momentum",      s["kalman"]["interpretation"])
                        signal_bar(s["adv_mr"]["score"],      "2. Advanced Mean Rev",     s["adv_mr"]["interpretation"])
                        signal_bar(s["lstm"]["score"],        "3. LSTM Sequential",       s["lstm"]["interpretation"])
                        signal_bar(s["transformer"]["score"], "4. Transformer Attention", s["transformer"]["interpretation"])
                        signal_bar(s["heston"]["score"],      "5. Heston Stoch Vol",      s["heston"]["interpretation"])
                        signal_bar(s["monte_carlo"]["score"], "6. GBM Monte Carlo",       s["monte_carlo"]["interpretation"])
                        signal_bar(s["sentiment"]["score"],   "7. FinBERT Sentiment",     s["sentiment"]["interpretation"])
                        signal_bar(s["rl"]["score"],          "8. RL DQN Optimizer",      s["rl"]["interpretation"])
                        st.markdown("---")
                        signal_bar(s["combined"],             "COMBINED (RL-weighted)",   "")

                        st.markdown("**Monte Carlo Stats**")
                        mc = s["monte_carlo"]
                        m1,m2,m3 = st.columns(3)
                        m1.metric("P(Gain)",   f"{mc.get('prob_gain',0):.1%}")
                        m2.metric("E[Return]", f"{mc.get('expected_ret',0):+.2f}%")
                        m3.metric("CVaR 5%",   f"{mc.get('cvar_5pct',0):.2f}%")

                        st.markdown("**RL Optimal Weights**")
                        rl_w = s["rl"].get("weights",{})
                        if rl_w:
                            for k,v in rl_w.items():
                                st.caption(f"{k}: {v:.3f}")
                        st.metric("RL Q-Value", f"{s['rl'].get('q_value',0):.4f}")

                    with dec_col:
                        st.markdown("### Trade Decision")
                        llm   = result["llm"]
                        trade = result["trade"]

                        rec_colors = {"STRONG BUY":"#00c853","BUY":"#69f0ae","HOLD":"#ffa000","SELL":"#ff5252","STRONG SELL":"#ff1744"}
                        color = rec_colors.get(llm["recommendation"],"#aaa")

                        st.markdown(f"""
                        <div style='background:#1e1e2e;border-radius:12px;padding:1rem;margin-bottom:1rem;border-left:4px solid {color}'>
                            <div style='font-size:1.4rem;font-weight:bold;color:{color}'>{llm['recommendation']}</div>
                            <div style='font-size:0.85rem;color:#aaa;margin-top:4px'>
                                Score: {llm['final_score']:+.4f} | Confidence: {llm.get('confidence','?')}
                            </div>
                        </div>""",unsafe_allow_html=True)

                        st.markdown("**Trade Thesis**")
                        st.info(llm.get("thesis","N/A"))
                        st.markdown("**Key Risks**")
                        st.warning(llm.get("risks","N/A"))

                        st.markdown("**Risk Engine V4 Decision**")
                        if trade["approved"]:
                            st.success(
                                f"✓ APPROVED — {trade['action']} {trade['shares']} shares @ ₹{result['price']:.2f}\n\n"
                                f"Capital: ₹{trade['capital_allocated']:,.0f} ({trade['position_size_pct']:.1%} of portfolio)"
                            )
                        else:
                            st.error(f"✗ BLOCKED — {trade['rejection_reason']}")

                        st.metric("Consensus Score", f"{trade.get('consensus_score',1.0):.2f}")
                        st.metric("Heston Regime",   trade.get('heston_regime','normal_vol'))

                        if trade["risk_notes"]:
                            with st.expander("Full Risk Engine Notes"):
                                for note in trade["risk_notes"]:
                                    st.caption(f"• {note}")

                        H  = result["signals"]["adv_mr"].get("hurst",0.5)
                        hc = "#00c853" if H<0.5 else ("#ffa000" if H<0.6 else "#ff1744")
                        lb = "mean-reverting" if H<0.5 else ("random walk" if H<0.6 else "trending")
                        st.markdown("**Hurst Exponent**")
                        st.markdown(f"<div style='background:#1e1e2e;border-radius:8px;padding:0.75rem'><span style='font-size:1.2rem;font-weight:bold;color:{hc}'>{H:.3f}</span><span style='color:#aaa;margin-left:8px'>{lb}</span></div>",unsafe_allow_html=True)

            elif analysis_btn and db_choice == "Nifty 500":
                st.warning("Live analysis only available for Nifty 50 stocks. Switch database to Nifty 50.")

    # ── TAB 2: PORTFOLIO RESULTS ──────────────────────────────────────────────
    with tab2:
        st.markdown("## Portfolio Allocator V3 — Dual DB + Flexi-SIP")
        st.caption("80% V5 Passive Nifty 500 | 20% V4 Active Nifty 50 | ₹10k bull SIP | ₹5k bear SIP")

        # Key metrics
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Initial Capital",  "₹1,00,000")
        c2.metric("Total Invested",   "₹9,95,000")
        c3.metric("Final Value",      "₹38,51,713", "+₹28,56,713 profit")
        c4.metric("Return",           "+287.11%",   "+17.73% annual")
        c5.metric("Max Drawdown",     "-30.01%",    "vs -34.82% benchmark")

        st.markdown("---")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Sharpe Ratio",  "1.760", "Excellent (>1.5)")
        c2.metric("Sortino Ratio", "2.327", "Very strong")
        c3.metric("Calmar Ratio",  "0.715")
        c4.metric("Alpha vs N50",  "+39.54%", "Outperformed benchmark")

        st.markdown("---")

        # SIP breakdown
        st.markdown("### Flexi-SIP Breakdown")
        s1,s2,s3 = st.columns(3)
        s1.metric("Bull Months (₹10k)", "79 months", "₹7,90,000 added")
        s2.metric("Bear Months (₹5k)",  "21 months", "₹1,05,000 added")
        s3.metric("Total SIP",          "100 months", "₹8,95,000 added")

        # Equity curves
        st.markdown("---")
        st.markdown("### Equity Curves")

        eq_v3 = load_equity_csv("data/equity_curve_allocator_v3.csv")
        eq_v2 = load_equity_csv("data/equity_curve_allocator_v2.csv")
        eq_v4 = load_equity_csv("data/equity_curve.csv")

        if eq_v3 is not None:
            st.plotly_chart(equity_chart(eq_v3, "total", "Portfolio Allocator V3 — Flexi-SIP"), use_container_width=True)

            # Show V5 and V4 sub-curves
            sub1, sub2 = st.columns(2)
            with sub1:
                if "v5_value" in eq_v3.columns:
                    st.plotly_chart(equity_chart(eq_v3, "v5_value", "V5 Passive Sub-Portfolio (80%)", "#2196f3"), use_container_width=True)
            with sub2:
                if "v4_value" in eq_v3.columns:
                    st.plotly_chart(equity_chart(eq_v3, "v4_value", "V4 Active Sub-Portfolio (20%)", "#ff9800"), use_container_width=True)
        else:
            st.info("Run `python3 portfolio_allocator_v3.py` first to generate results.")

        # Trade log
        st.markdown("---")
        st.markdown("### Recent Trades")
        tl = load_trade_log("data/trade_log_allocator_v3.csv")
        if tl is not None:
            sells = tl[tl["action"] == "SELL"].tail(20) if "action" in tl.columns else tl.tail(20)
            st.dataframe(sells, use_container_width=True, height=300)

        # SIP log
        st.markdown("### SIP Log")
        sip = load_sip_log("data/sip_log_v3.csv")
        if sip is not None:
            st.dataframe(sip.tail(20), use_container_width=True, height=300)

    # ── TAB 3: NIFTY 500 PASSIVE ──────────────────────────────────────────────
    with tab3:
        st.markdown("## Nifty 500 Passive — Market Cap Weighted")
        st.caption("Top N stocks by market cap proxy (price × volume) | Buy and hold 2018-2026")

        # Results table
        results_data = {
            "Basket": ["Top 50", "Top 100", "Top 200"],
            "Return": ["+284.85%", "+259.43%", "+161.69%"],
            "Max DD":  ["-30.09%", "-25.62%", "-17.66%"],
            "Sharpe":  ["0.638",   "0.668",   "0.497"],
            "Sortino": ["0.755",   "0.796",   "0.579"],
            "Calmar":  ["0.583",   "0.648",   "0.693"],
            "Alpha vs N500 bench": ["+176.82%", "+151.39%", "+53.65%"],
        }
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)

        st.markdown("---")
        st.markdown("### Key Finding")
        st.success("""
        **Top 100 Nifty 500 stocks by market cap (passive buy & hold) returned +259% over 2018-2026 
        with only -25.62% max drawdown** — outperforming both the Nifty 50 benchmark (+176%) and 
        our V4 active system (+66%) on full period returns.
        
        This proves Indian mid-cap equities offered significant passive alpha during this period.
        The key insight: use price × volume as selection criterion, NOT raw volume 
        (which picks junk penny stocks with terrible returns).
        """)

        st.markdown("### Benchmark Comparison")
        b1,b2,b3 = st.columns(3)
        b1.metric("Nifty 50 Benchmark",      "+176.48%", "Max DD: -34.82%")
        b2.metric("Nifty 500 Avg Benchmark", "+108.04%", "Max DD: -35.23%")
        b3.metric("Top 100 N500 Passive",    "+259.43%", "Max DD: -25.62%")

        # Equity curve
        eq_n500 = load_equity_csv("data/equity_curve_v5_n500.csv")
        if eq_n500 is not None:
            st.markdown("---")
            st.plotly_chart(equity_chart(eq_n500, "portfolio_value", "Nifty 500 Top 200 Passive Equity Curve"), use_container_width=True)
        else:
            st.info("Run `python3 backtester_v5_nifty500.py` to generate equity curve.")

    # ── TAB 4: SIP CALCULATOR ─────────────────────────────────────────────────
    with tab4:
        st.markdown("## Flexi-SIP Calculator")
        st.caption("Simulate your investment with our Portfolio Allocator V3 system")

        col_inp, col_out = st.columns([1,1])

        with col_inp:
            st.markdown("### Your Investment Parameters")
            initial    = st.number_input("Initial Investment (₹)", min_value=10_000, max_value=10_000_000, value=100_000, step=10_000)
            bull_sip   = st.number_input("Bull Month SIP (₹)", min_value=1_000, max_value=100_000, value=10_000, step=1_000)
            bear_sip   = st.number_input("Bear Month SIP (₹)", min_value=500,   max_value=50_000,  value=5_000,  step=500)
            years      = st.slider("Investment Period (Years)", min_value=1, max_value=20, value=8)
            annual_ret = st.slider("Expected Annual Return (%)", min_value=5.0, max_value=30.0, value=17.73, step=0.5)

            st.caption("Based on our backtest: 17.73% annual return, 79% bull months, 21% bear months")

        result_sip = sip_calculator(initial, bull_sip, bear_sip, annual_ret/100, years)

        with col_out:
            st.markdown("### Projected Results")
            st.metric("Total Invested",  f"₹{result_sip['total_invested']:,.0f}")
            st.metric("Final Value",     f"₹{result_sip['final_value']:,.0f}",
                      f"+₹{result_sip['total_profit']:,.0f} profit")
            st.metric("Return on Invested", f"+{result_sip['return_pct']:.2f}%")

            # Compare with FD
            fd_rate   = 0.07
            fd_val    = initial * (1 + fd_rate) ** years
            fd_sip    = (bull_sip * 0.79 + bear_sip * 0.21)
            fd_months = years * 12
            fd_total  = fd_val + fd_sip * (((1 + fd_rate/12) ** fd_months - 1) / (fd_rate/12))

            st.markdown("---")
            st.markdown("### vs Fixed Deposit (7%)")
            st.metric("FD Final Value",   f"₹{fd_total:,.0f}")
            st.metric("Extra vs FD",      f"₹{result_sip['final_value'] - fd_total:,.0f}",
                      f"{((result_sip['final_value']/fd_total)-1)*100:.1f}% better than FD")

        # Chart
        months = list(range(len(result_sip["monthly_values"])))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=result_sip["monthly_values"],
            fill="tozeroy", line=dict(color="#00c853", width=2),
            fillcolor="rgba(0,200,83,0.1)", name="Portfolio Value",
        ))
        fig.add_trace(go.Scatter(
            x=months, y=result_sip["invested_values"],
            line=dict(color="#2196f3", width=2, dash="dash"),
            name="Total Invested",
        ))
        fig.update_layout(
            title="SIP Growth Projection",
            paper_bgcolor="#0e0e1a", plot_bgcolor="#0e0e1a",
            font=dict(color="#ccc"), height=350, margin=dict(t=40,b=20,l=20,r=20),
            xaxis_title="Months", yaxis_tickprefix="₹",
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        )
        fig.update_xaxes(gridcolor="#1e1e2e")
        fig.update_yaxes(gridcolor="#1e1e2e")
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5: SYSTEM COMPARISON ──────────────────────────────────────────────
    with tab5:
        st.markdown("## Complete System Comparison")
        st.caption("All systems tested on 2018-2026 | ₹1,00,000 initial capital (except Allocator V3)")

        comparison = {
            "System": [
                "V1 Simple MA (2020-24)",
                "V2 Kalman+HMM+GBM (2020-24)",
                "V3 +LSTM+Heston (2020-24)",
                "V4 Full 9-Agent (2020-24)",
                "V4 Full Period (2018-26)",
                "V5 N50 Passive Top 15",
                "V6 Hybrid 50/50",
                "CC2 Early Warning",
                "V5 N500 Top 50",
                "V5 N500 Top 100",
                "V5 N500 Top 200",
                "Allocator V2 80/20",
                "Allocator V3 + Flexi-SIP ⭐",
                "Nifty 50 Benchmark",
            ],
            "Return": [
                "+44.77%", "+123.95%", "+136.19%", "+182.27%",
                "+66.34%", "+204.47%", "+110.40%", "+145.21%",
                "+284.85%", "+259.43%", "+161.69%",
                "+216.02%", "+287.11%",
                "+176.48%",
            ],
            "Max DD": [
                "-10.55%", "-15.77%", "-12.49%", "-9.38%",
                "-16.19%", "-40.63%", "-21.74%", "-11.19%",
                "-30.09%", "-25.62%", "-17.66%",
                "-20.81%", "-30.01%",
                "-34.82%",
            ],
            "Sharpe": [
                "0.184", "1.002", "1.100", "1.429",
                "0.027", "0.472", "0.464", "0.533",
                "0.638", "0.668", "0.497",
                "0.629", "1.760",
                "—",
            ],
            "Universe": [
                "N50","N50","N50","N50",
                "N50","N50","N50","N50",
                "N500","N500","N500",
                "N50+N500","N50+N500",
                "N50",
            ],
        }

        df_comp = pd.DataFrame(comparison)
        st.dataframe(df_comp, use_container_width=True, height=500)

        st.markdown("---")
        st.markdown("### Key Findings")

        f1,f2 = st.columns(2)
        with f1:
            st.success("""
            **What Works:**
            - Progressive math (V1→V4) consistently improves performance
            - V4 generates +43% alpha in bull markets (2020-2024)
            - V4 protects capital in crises (-8.57% DD vs -18.27% benchmark)
            - Nifty 500 top 100 passive beats all systems on raw returns
            - Dual DB portfolio allocator with Flexi-SIP is best overall
            - Sharpe of 1.76 is institutional grade
            """)
        with f2:
            st.error("""
            **What Doesn't Work:**
            - V4 fails on Nifty 500 mid/small caps (-11% return, 22% win rate)
            - Switching systems (CC1, CC2, CC3) bleeds transaction costs
            - Volume-based selection picks junk penny stocks
            - Simple MA crossover vs buy-and-hold: -91% alpha
            - Sustained bull markets → passive always beats active
            - CC2 emergency DD bug causes false triggers
            """)

        st.markdown("---")
        st.markdown("### Limitations")
        st.warning("""
        **Honest Limitations (must disclose in paper):**
        1. Survivorship bias — Nifty 500 DB missing delisted/bankrupt companies
        2. Look-ahead bias — stock selection used future market cap data
        3. Commission too low — used 0.1%, real cost 0.3-0.5% for small caps
        4. All results in-sample except 2018-2019 V4 out-of-sample test
        5. FinBERT neutral in backtest — no historical news data
        6. Liquidity assumption — instant execution at close price
        7. No corporate actions — dividends, splits distort prices
        """)

        st.markdown("---")
        st.markdown("### Research Papers")
        c1,c2 = st.columns(2)
        with c1:
            st.info("""
            **Paper 1 (Published)**
            
            "Why Machine Learning Trading Strategies Fail: 
            An Empirical Analysis of Nifty 50"
            
            📄 SSRN | 🔗 GitHub: yakshmaan/ai-hedge-fund-nifty50
            """)
        with c2:
            st.info("""
            **Paper 2 (In Progress)**
            
            "From Simple Signals to Multi-Agent Intelligence: 
            Building and Testing a Progressive Quantitative 
            Trading System on Indian Equity Markets (Nifty 50 
            and Nifty 500), 2018-2026"
            
            📄 SSRN submission pending
            """)


if __name__ == "__main__":
    main()
