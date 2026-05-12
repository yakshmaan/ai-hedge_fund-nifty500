# AI Hedge Fund — Nifty 50 & Nifty 500

> A progressive multi-agent quantitative trading system built and tested on Indian equity markets (2018-2026). Published research on SSRN. Built by Yaksh, 2nd semester BTech CS student.

---

## 🏆 Flagship Result

**Portfolio Allocator V3 — Dual Database + Fixed SIP**

| Metric | Value |
|--------|-------|
| Initial Capital | ₹1,00,000 |
| Monthly SIP | ₹5,000/month |
| Total Invested | ₹6,00,000 |
| Final Value | ₹24,97,067 |
| Total Profit | ₹18,97,067 |
| Return on Invested | **+316.18%** |
| Annual Return | **+18.76%** |
| Sharpe Ratio | **1.600** |
| Max Drawdown | **-29.83%** |
| Benchmark Max DD | -34.82% |
| Nifty 50 Benchmark | +176.48% |

> A normal salaried person investing ₹1 lakh + ₹5,000/month for 8 years turns ₹6 lakh into ₹24.97 lakh — beating the Nifty 50 benchmark with lower drawdown.

---

## 📚 Research Papers

1. **Published:** ["Why Machine Learning Trading Strategies Fail: An Empirical Analysis of Nifty 50"](https://ssrn.com) — SSRN
2. **In Progress:** "From Simple Signals to Multi-Agent Intelligence: Building and Testing a Progressive Quantitative Trading System on Indian Equity Markets (Nifty 50 and Nifty 500), 2018-2026"

---

## 🧠 System Architecture — 9 Agents

| Agent | Method | Purpose |
|-------|--------|---------|
| 1. HMM Regime Detection | Baum-Welch Hidden Markov Model | Detects bull/high-vol/bear regime |
| 2. Kalman Filter Momentum | State space model + ADX gating | Tracks price velocity and trend |
| 3. Advanced Mean Reversion | ADF test + Hurst Exponent + Z-score | Detects mean-reverting opportunities |
| 4. LSTM Sequential | Momentum acceleration + streak | Captures sequential price patterns |
| 5. Transformer Attention | Cosine similarity + softmax attention | Pattern matching on historical windows |
| 6. Heston Stochastic Volatility | dv = κ(θ-v)dt + ξ√vdW | Models time-varying volatility regimes |
| 7. GBM Monte Carlo | 10,000 simulations | P(gain) and CVaR estimation |
| 8. FinBERT Sentiment | Transformer NLP | News headline sentiment analysis |
| 9. RL DQN Weight Optimizer | Deep Q-Network + Bellman equation | Dynamic agent weight optimization |

**LLM Orchestrator:** Llama 3.3 70B via Groq API synthesizes all signals into trade thesis

---

## 📊 Risk Engine V4

- **Kelly Criterion** position sizing
- **Consensus scoring** — 85%+ agreement → 1.30× size boost
- **Heston vol multiplier** — high vol → 0.70× position size
- **FinBERT sentiment gate** — strongly negative news blocks all buys
- **Dynamic VaR threshold** — high vol → 8%, low vol → 15%
- **Multi-agent stop loss voting** — 4+ bearish agents → early exit
- **Drawdown circuit breaker** at 15%
- **Hurst-adjusted stop losses** — trending stocks get wider stops

---

## 📈 Complete Results

### Nifty 50 — Progressive System (V1 → V4)

| Version | Period | Return | Benchmark | Alpha | Sharpe | Max DD |
|---------|--------|--------|-----------|-------|--------|--------|
| V1 Simple MA | 2020-24 | +44.77% | +136.06% | -91.29% | 0.184 | -10.55% |
| V2 Kalman+HMM+GBM | 2020-24 | +123.95% | +136.06% | -12.11% | 1.002 | -15.77% |
| V3 +LSTM+Heston | 2020-24 | +136.19% | +136.06% | +0.13% | 1.100 | -12.49% |
| V4 Full 9-Agent | 2020-24 | +182.27% | +139.39% | **+42.88%** | **1.429** | **-9.38%** |
| V4 Full Period | 2018-26 | +66.34% | +176.48% | -110.15% | 0.027 | -16.19% |

### V4 Crisis Performance

| Period | Event | V4 Return | Benchmark | V4 Max DD | Benchmark Max DD |
|--------|-------|-----------|-----------|-----------|-----------------|
| 2018-2019 | IL&FS + NBFC Crisis | +9.67% | +5.73% | -8.57% | **-18.27%** |
| Feb-Apr 2020 | COVID Crash | **0.00%** (cash) | -19.94% | 0.00% | -19.94% |
| 2019-2020 | COVID + Recovery | +44.15% | +25.71% | -4.11% | — |

### Nifty 500 — Passive Systems

| Basket | Return | Max DD | Sharpe | Calmar |
|--------|--------|--------|--------|--------|
| Top 50 by market cap | +284.85% | -30.09% | 0.638 | 0.583 |
| **Top 100 by market cap** | **+259.43%** | **-25.62%** | **0.668** | **0.648** |
| Top 200 by market cap | +161.69% | -17.66% | 0.497 | 0.693 |
| Nifty 500 Benchmark | +108.04% | -35.23% | — | — |

### Portfolio Allocators — Dual Database

| System | Return | Max DD | Sharpe | Notes |
|--------|--------|--------|--------|-------|
| Allocator V2 40/40/20 | +141.19% | -12.35% | 0.506 | First dual DB |
| Allocator V2 80/20 | +216.02% | -20.81% | 0.629 | Best single lump sum |
| **Allocator V3 Fixed SIP** | **+316.18%** | **-29.83%** | **1.600** | **Best overall** |

---

## 🗂️ File Structure

```
ai-hedge-fund-nifty500/
│
├── data/
│   ├── fetch_data_nifty500.py    # Downloads 407 Nifty 500 stocks from Yahoo Finance
│   ├── clean_data.py             # Cleans raw CSVs, forward fills, removes outliers
│   ├── store_data_nifty500.py    # Stores into SQLite nifty500.db
│   ├── fetch_data.py             # Downloads Nifty 50 stocks
│   └── store_data.py             # Stores into SQLite nifty50.db
│
├── agents/
│   ├── regime_agent.py           # HMM regime detection
│   ├── kalman_momentum_agent.py  # Kalman filter momentum
│   ├── advanced_mean_reversion_agent.py  # ADF + Hurst
│   ├── lstm_agent.py             # LSTM sequential
│   ├── transformer_attention_agent.py    # Transformer proxy
│   ├── heston_agent.py           # Heston stochastic vol
│   ├── gbm_monte_carlo_agent.py  # GBM Monte Carlo
│   ├── finbert_sentiment_agent.py # FinBERT NLP
│   └── rl_agent.py               # RL DQN weight optimizer
│
├── risk/
│   ├── risk_engine_v3.py         # V3 risk engine
│   └── risk_engine_v4.py         # V4 risk engine (full)
│
├── backtester.py                 # V1 simple backtester
├── backtester_v2.py              # V2 Kalman + HMM + GBM
├── backtester_v3.py              # V3 + LSTM + Heston
├── backtester_v4.py              # V4 full 9-agent system
├── backtester_v5.py              # V5 passive buy & hold
├── backtester_v5_nifty500.py     # V5 Nifty 500 market cap selection
├── backtester_v6.py              # V6 hybrid 50/50
│
├── central_command_v1.py         # V5 default + V4 on bear
├── central_command_v2.py         # Early warning exit to cash
├── central_command_v3.py         # Failed — emergency DD bug
│
├── portfolio_allocator.py        # V1 single DB allocator
├── portfolio_allocator_v2.py     # V2 dual DB (80% N500 + 20% N50)
├── portfolio_allocator_v3.py     # V3 dual DB + Flexi-SIP ⭐ FLAGSHIP
│
├── orchestrator.py               # Live trading orchestrator
├── orchestrator_v2.py            # V2 with regime awareness
├── orchestrator_v3.py            # V3 with Heston + LSTM
├── orchestrator_v4.py            # V4 full + LLM thesis via Groq
│
├── dashboard.py                  # Streamlit dashboard (5 tabs)
├── forecaster.py                 # 30-day GBM forecaster
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install yfinance pandas numpy streamlit plotly requests transformers torch
```

### 2. Download data
```bash
python3 data/fetch_data_nifty500.py    # ~45 mins, downloads 407 stocks
python3 data/fetch_data.py             # ~5 mins, downloads 48 Nifty 50 stocks
python3 data/clean_data.py
python3 data/store_data_nifty500.py
python3 data/store_data.py
```

### 3. Run backtests
```bash
python3 backtester_v4.py               # V4 active system
python3 backtester_v5_nifty500.py      # Nifty 500 passive
python3 portfolio_allocator_v3.py      # Flagship system
```

### 4. Run dashboard
```bash
export GROQ_API_KEY="your_groq_api_key"
streamlit run dashboard.py
```

---

## 🔑 Key Findings

1. **Progressive math works** — V1 to V4 shows clear improvement (-91% → +43% alpha)
2. **V4 beats benchmark in bull markets** — +42.88% alpha in 2020-2024
3. **V4 protects capital in crises** — only -8.57% drawdown vs -18.27% benchmark in IL&FS crisis
4. **V4 goes to cash during COVID crash** — 0% loss vs -20% benchmark
5. **V4 fails on Nifty 500** — proves active systems are market-specific
6. **Nifty 500 top 100 passive beats everything** — +259% return vs +176% Nifty 50 benchmark
7. **Dual DB portfolio allocator is best** — combines strengths of both universes
8. **Fixed SIP beats timed SIP** — rupee cost averaging outperforms market timing
9. **Switching systems bleed transaction costs** — CC1, CC2, CC3 all failed
10. **Indian mid-caps offer passive alpha** — market cap selection critical

---

## ⚠️ Limitations

| Limitation | Impact |
|-----------|--------|
| Survivorship bias | Nifty 500 missing delisted companies — inflates returns |
| Look-ahead bias | Stock selection used future market cap data |
| Commission too low | Used 0.1%, real cost 0.3-0.5% for small caps |
| In-sample results | Only 2018-2019 is true out-of-sample for V4 |
| No corporate actions | Dividends, splits distort prices |
| Liquidity assumption | Assumes instant execution at close price |
| FinBERT neutral | No historical news data in backtest |

---

## 📐 Mathematical Concepts Used

`Geometric Brownian Motion` · `Kalman Filter` · `Hidden Markov Model (Baum-Welch)` · `ADF Test` · `Hurst Exponent R/S Analysis` · `Engle-Granger Cointegration` · `Kelly Criterion` · `VaR/CVaR` · `Heston Stochastic Volatility` · `LSTM Gates` · `Transformer Multi-Head Self-Attention` · `Deep Q-Network` · `Bellman Equation` · `Experience Replay`

---

## 👤 Author

**Yaksh** — 2nd Semester BTech Computer Science, India

- 📄 SSRN: [Paper 1 — Why ML Trading Strategies Fail](https://ssrn.com)
- 🔗 GitHub: [github.com/yakshmaan](https://github.com/yakshmaan)
- 💼 LinkedIn: [Connect on LinkedIn](https://linkedin.com)

---

## 📄 License

MIT License — free to use for research and educational purposes.

> ⚠️ This is a research project. Not financial advice. Past backtest performance does not guarantee future results. Always consult a SEBI registered advisor before investing.
# ai-hedge_fund-nifty500
