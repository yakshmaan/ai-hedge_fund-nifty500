"""
risk_engine_v3.py
-----------------
Risk Engine V3 — wired to all 7 agents.

New in V3 vs V2:
  1. Heston vol regime awareness
     - When Heston detects high volatility regime → tighten all limits
     - When low vol regime → can be slightly more aggressive
  2. Sentiment-adjusted position sizing
     - Strong positive sentiment → small size boost
     - Strong negative sentiment → reduce size
  3. LSTM confidence integration
     - LSTM agreement with other agents → boost size
     - LSTM disagreement → reduce size (conflicting signals)
  4. Dynamic stop loss based on Heston volatility
     - High vol → wider stop (stock needs room to breathe)
     - Low vol  → tighter stop (less excuse for big moves)
  5. Cross-agent consensus score
     - If 5+ agents agree → higher conviction → larger size
     - If agents split → lower conviction → smaller size

Six core checks still run in sequence.
All new features layer on top without breaking existing logic.
"""

import pandas as pd
import numpy as np
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

DB_PATH = "data/nifty50.db"


@dataclass
class RiskConfig:
    total_capital:           float = 100_000.0
    max_position_pct:        float = 0.10
    max_portfolio_var_pct:   float = 0.12
    max_drawdown_pct:        float = 0.15
    stop_loss_pct:           float = 0.07
    kelly_fraction:          float = 1.0
    min_signal_strength:     float = 0.10


REGIME_RISK_MULTIPLIERS = {
    0: {"max_position_pct":1.0,  "max_drawdown_pct":1.0,  "stop_loss_pct":1.0,  "kelly_fraction":1.0,  "description":"Normal limits"},
    1: {"max_position_pct":0.80, "max_drawdown_pct":0.85, "stop_loss_pct":0.85, "kelly_fraction":0.80, "description":"Tighter — high volatility"},
    2: {"max_position_pct":0.50, "max_drawdown_pct":0.70, "stop_loss_pct":0.70, "kelly_fraction":0.50, "description":"Defensive — bear market"},
}

# Heston vol regime multipliers (on top of market regime)
HESTON_VOL_MULTIPLIERS = {
    "high_vol":   {"position": 0.70, "stop": 1.30, "description": "High vol — reduce size, widen stop"},
    "normal_vol": {"position": 1.00, "stop": 1.00, "description": "Normal vol — standard limits"},
    "low_vol":    {"position": 1.15, "stop": 0.85, "description": "Low vol — slight size boost, tighter stop"},
}

CONFIDENCE_MULTIPLIERS = {
    "HIGH":   1.0,
    "MEDIUM": 1.0,
    "LOW":    1.0,
}


@dataclass
class Portfolio:
    cash:          float = 100_000.0
    positions:     dict  = field(default_factory=dict)
    peak_value:    float = 100_000.0
    trade_history: list  = field(default_factory=list)

    def total_value(self):
        return self.cash + sum(
            p["shares"] * p["current_price"]
            for p in self.positions.values()
        )

    def drawdown(self):
        current = self.total_value()
        if current > self.peak_value:
            self.peak_value = current
        return (self.peak_value - current) / self.peak_value

    def position_weight(self, symbol):
        if symbol not in self.positions:
            return 0.0
        val = self.positions[symbol]["shares"] * self.positions[symbol]["current_price"]
        return val / max(self.total_value(), 1.0)


def compute_var(returns, confidence=0.95):
    if len(returns) < 30:
        return 0.0
    return abs(float(np.percentile(returns.dropna(), (1-confidence)*100)))


def compute_consensus_score(all_agent_scores: dict) -> float:
    """
    Measures how much agents agree with each other.

    If all agents point the same direction → high consensus → bigger size
    If agents are split → low consensus → smaller size

    Returns multiplier between 0.5 and 1.3
    """
    scores = [v for v in all_agent_scores.values()
              if isinstance(v, (int, float)) and not np.isnan(v)]

    if len(scores) < 3:
        return 1.0

    scores = np.array(scores)
    # Count agents agreeing with the majority direction
    majority_dir = np.sign(np.mean(scores))
    agreeing     = np.sum(np.sign(scores) == majority_dir)
    total        = len(scores)

    consensus_ratio = agreeing / total

    if consensus_ratio >= 0.85:    return 1.30   # 85%+ agree — high conviction
    elif consensus_ratio >= 0.70:  return 1.15
    elif consensus_ratio >= 0.57:  return 1.00
    elif consensus_ratio >= 0.43:  return 0.80   # agents split — reduce size
    else:                          return 0.60   # mostly disagreeing — very small


def get_heston_vol_regime(heston_score: float, vol_ratio: float = 1.0) -> str:
    """
    Determine Heston volatility regime from agent output.
    vol_ratio = current_vol / long_run_vol
    """
    if vol_ratio > 1.3 or heston_score < -0.15:
        return "high_vol"
    elif vol_ratio < 0.8 or heston_score > 0.10:
        return "low_vol"
    else:
        return "normal_vol"


def kelly_position_size(signal_score, confidence, win_rate, avg_win, avg_loss, config, regime=0):
    if signal_score <= 0:
        return 0.0
    q = 1 - win_rate
    if avg_loss == 0:
        return 0.0
    b       = avg_win / avg_loss
    kelly_f = (win_rate * b - q) / b
    if kelly_f <= 0:
        return 0.0
    kelly_f = max(kelly_f, 0.02)
    size        = kelly_f * config.kelly_fraction
    conf_mult   = CONFIDENCE_MULTIPLIERS.get(confidence, 1.0)
    size       *= conf_mult
    regime_mult = REGIME_RISK_MULTIPLIERS.get(regime, REGIME_RISK_MULTIPLIERS[0])
    size       *= regime_mult["kelly_fraction"]
    max_pos     = config.max_position_pct * regime_mult["max_position_pct"]
    return min(size, max_pos)


def compute_portfolio_var_historical(portfolio):
    if not portfolio.positions:
        return 0.0
    conn    = sqlite3.connect(DB_PATH)
    symbols = list(portfolio.positions.keys())
    placeholders = ",".join([f"'{s}'" for s in symbols])
    df = pd.read_sql(f"""
        SELECT Date, Symbol, Daily_Return FROM prices
        WHERE Symbol IN ({placeholders}) ORDER BY Date
    """, conn)
    conn.close()
    if df.empty or len(df) < 30:
        return 0.0
    matrix  = df.pivot(index="Date", columns="Symbol", values="Daily_Return").dropna()
    if len(matrix) < 30:
        return 0.0
    weights = np.array([portfolio.position_weight(s) for s in matrix.columns])
    return compute_var(pd.Series(matrix.values @ weights))


@dataclass
class TradeDecision:
    approved:          bool
    symbol:            str
    action:            str
    shares:            int
    capital_allocated: float
    rejection_reason:  Optional[str] = None
    position_size_pct: float         = 0.0
    risk_notes:        list          = field(default_factory=list)
    consensus_score:   float         = 1.0
    heston_regime:     str           = "normal_vol"


def evaluate_trade_v3(
    symbol,
    signal_score,
    current_price,
    portfolio,
    config,
    regime=0,
    confidence="MEDIUM",
    hurst=0.5,
    cvar_mc=0.0,
    heston_score=0.0,
    heston_vol_ratio=1.0,
    lstm_score=0.0,
    sentiment_score=0.0,
    all_agent_scores=None,
):
    """
    V3 evaluate_trade — uses all 7 agent outputs for smarter decisions.

    New parameters vs V2:
      heston_score:     from heston_agent (vol regime signal)
      heston_vol_ratio: v0/theta from heston_agent (current/long-run vol)
      lstm_score:       from lstm_agent (sequential pattern)
      sentiment_score:  from sentiment_agent (news NLP)
      all_agent_scores: dict of all agent scores for consensus calculation
    """
    risk_notes  = []
    regime_mult = REGIME_RISK_MULTIPLIERS.get(regime, REGIME_RISK_MULTIPLIERS[0])
    risk_notes.append(f"Regime {regime}: {regime_mult['description']}")

    # Determine Heston vol regime
    heston_regime    = get_heston_vol_regime(heston_score, heston_vol_ratio)
    heston_vol_mult  = HESTON_VOL_MULTIPLIERS[heston_regime]
    risk_notes.append(f"Heston: {heston_regime} — {heston_vol_mult['description']}")

    # Compute consensus across all agents
    if all_agent_scores is None:
        all_agent_scores = {}
    consensus_mult = compute_consensus_score(all_agent_scores)
    n_agreeing     = sum(1 for v in all_agent_scores.values()
                        if isinstance(v,(int,float)) and np.sign(v)==np.sign(signal_score))
    risk_notes.append(f"Consensus: {n_agreeing}/{len(all_agent_scores)} agents agree (mult={consensus_mult:.2f})")

    # Sentiment adjustment
    sent_mult = 1.0
    if sentiment_score > 0.3:    sent_mult = 1.10
    elif sentiment_score > 0.1:  sent_mult = 1.05
    elif sentiment_score < -0.3: sent_mult = 0.85
    elif sentiment_score < -0.1: sent_mult = 0.92
    if sent_mult != 1.0:
        risk_notes.append(f"Sentiment adjustment: {sent_mult:.2f}x (score={sentiment_score:+.3f})")

    # LSTM agreement check
    lstm_agreement = np.sign(lstm_score) == np.sign(signal_score)
    lstm_mult      = 1.05 if lstm_agreement and abs(lstm_score) > 0.1 else (0.92 if not lstm_agreement and abs(lstm_score) > 0.2 else 1.0)
    if lstm_mult != 1.0:
        risk_notes.append(f"LSTM {'agrees' if lstm_agreement else 'disagrees'}: {lstm_mult:.2f}x")

    # ── CHECK 1: Signal strength ──────────────────────────────────────────────
    if abs(signal_score) < config.min_signal_strength:
        return TradeDecision(
            approved=False, symbol=symbol, action="HOLD",
            shares=0, capital_allocated=0,
            rejection_reason=f"Signal too weak ({signal_score:.3f} < {config.min_signal_strength})",
            risk_notes=risk_notes, consensus_score=consensus_mult, heston_regime=heston_regime,
        )

    action = "BUY" if signal_score > 0 else "SELL"

    # ── CHECK 2: Drawdown circuit breaker ─────────────────────────────────────
    current_drawdown = portfolio.drawdown()
    max_dd = config.max_drawdown_pct * regime_mult["max_drawdown_pct"]
    if current_drawdown > max_dd:
        return TradeDecision(
            approved=False, symbol=symbol, action="HOLD",
            shares=0, capital_allocated=0,
            rejection_reason=f"Drawdown circuit breaker: {current_drawdown:.1%} > {max_dd:.1%}",
            risk_notes=risk_notes, consensus_score=consensus_mult, heston_regime=heston_regime,
        )
    risk_notes.append(f"Drawdown OK: {current_drawdown:.2%} (limit {max_dd:.2%})")

    # ── CHECK 3: Concentration ────────────────────────────────────────────────
    max_pos = (config.max_position_pct
               * regime_mult["max_position_pct"]
               * heston_vol_mult["position"])
    if action == "BUY":
        current_weight = portfolio.position_weight(symbol)
        if current_weight >= max_pos:
            return TradeDecision(
                approved=False, symbol=symbol, action="HOLD",
                shares=0, capital_allocated=0,
                rejection_reason=f"Concentration limit: {current_weight:.1%} >= {max_pos:.1%}",
                risk_notes=risk_notes, consensus_score=consensus_mult, heston_regime=heston_regime,
            )
        risk_notes.append(f"Concentration OK: {current_weight:.2%} (limit {max_pos:.2%})")

    # ── CHECK 4: Stop loss — Hurst + Heston adjusted ──────────────────────────
    hurst_stop_adj  = 1.3 if hurst > 0.6 else (0.8 if hurst < 0.4 else 1.0)
    heston_stop_adj = heston_vol_mult["stop"]
    stop_pct = (config.stop_loss_pct
                * regime_mult["stop_loss_pct"]
                * hurst_stop_adj
                * heston_stop_adj)

    if symbol in portfolio.positions:
        entry_price = portfolio.positions[symbol]["entry_price"]
        loss_pct    = (current_price - entry_price) / entry_price
        if loss_pct < -stop_pct:
            return TradeDecision(
                approved=True, symbol=symbol, action="SELL",
                shares=portfolio.positions[symbol]["shares"],
                capital_allocated=0,
                risk_notes=[f"STOP LOSS: {loss_pct:.2%} (limit -{stop_pct:.2%}, H={hurst:.2f}, Heston={heston_regime})"],
                consensus_score=consensus_mult, heston_regime=heston_regime,
            )
        risk_notes.append(f"Stop loss OK: {loss_pct:+.2%} (limit -{stop_pct:.2%})")

    # ── CHECK 5: Kelly + all V3 multipliers ───────────────────────────────────
    size_fraction = kelly_position_size(
        signal_score=abs(signal_score),
        confidence=confidence,
        win_rate=0.54,
        avg_win=0.018,
        avg_loss=0.012,
        config=config,
        regime=regime,
    )

    # Apply V3 multipliers
    size_fraction *= heston_vol_mult["position"]
    size_fraction *= consensus_mult
    size_fraction *= sent_mult
    size_fraction *= lstm_mult
    size_fraction  = min(size_fraction, max_pos)

    capital_to_allocate = portfolio.total_value() * size_fraction
    shares = int(capital_to_allocate / current_price)

    if shares == 0:
        return TradeDecision(
            approved=False, symbol=symbol, action="HOLD",
            shares=0, capital_allocated=0,
            rejection_reason=f"Kelly sizing = 0 shares (signal={signal_score:.3f}, size_frac={size_fraction:.3f})",
            risk_notes=risk_notes, consensus_score=consensus_mult, heston_regime=heston_regime,
        )
    risk_notes.append(
        f"V3 Kelly: {size_fraction:.2%} = {shares} shares "
        f"(consensus={consensus_mult:.2f}, sentiment={sent_mult:.2f}, lstm={lstm_mult:.2f})"
    )

    # ── CHECK 6: VaR / CVaR ───────────────────────────────────────────────────
    if cvar_mc != 0.0:
        portfolio_var = abs(cvar_mc) / 100
        var_source    = "Monte Carlo CVaR"
    else:
        portfolio_var = compute_portfolio_var_historical(portfolio)
        var_source    = "historical VaR"

    if portfolio_var > config.max_portfolio_var_pct:
        return TradeDecision(
            approved=False, symbol=symbol, action="HOLD",
            shares=0, capital_allocated=0,
            rejection_reason=f"{var_source} too high: {portfolio_var:.2%} > {config.max_portfolio_var_pct:.2%}",
            risk_notes=risk_notes, consensus_score=consensus_mult, heston_regime=heston_regime,
        )
    risk_notes.append(f"{var_source} OK: {portfolio_var:.2%}")

    return TradeDecision(
        approved=True, symbol=symbol, action=action,
        shares=shares, capital_allocated=shares*current_price,
        position_size_pct=size_fraction, risk_notes=risk_notes,
        consensus_score=consensus_mult, heston_regime=heston_regime,
    )


# Keep backward-compatible evaluate_trade for V2 orchestrator
def evaluate_trade(
    symbol, signal_score, current_price, portfolio, config,
    regime=0, confidence="MEDIUM", hurst=0.5, cvar_mc=0.0,
):
    """V2-compatible wrapper — calls V3 with default V3 params."""
    return evaluate_trade_v3(
        symbol=symbol, signal_score=signal_score,
        current_price=current_price, portfolio=portfolio, config=config,
        regime=regime, confidence=confidence, hurst=hurst, cvar_mc=cvar_mc,
    )


if __name__ == "__main__":
    config    = RiskConfig(total_capital=100_000)
    portfolio = Portfolio(cash=100_000)

    print("Risk Engine V3 Test\n" + "─"*55)

    all_scores = {
        "kalman":      0.72,
        "adv_mr":      -0.15,
        "lstm":        0.30,
        "heston":      0.10,
        "monte_carlo": 0.08,
        "sentiment":   0.25,
    }

    d = evaluate_trade_v3(
        symbol="AXISBANK", signal_score=0.327, current_price=1050.0,
        portfolio=portfolio, config=config,
        regime=0, confidence="MEDIUM",
        hurst=0.977, cvar_mc=-9.01,
        heston_score=0.10, heston_vol_ratio=0.85,
        lstm_score=0.30, sentiment_score=0.25,
        all_agent_scores=all_scores,
    )

    print(f"Approved      : {d.approved}")
    print(f"Action        : {d.action}")
    print(f"Shares        : {d.shares}")
    print(f"Capital       : ₹{d.capital_allocated:,.0f}")
    print(f"Heston Regime : {d.heston_regime}")
    print(f"Consensus     : {d.consensus_score:.2f}")
    for n in d.risk_notes:
        print(f"  [{n}]")
