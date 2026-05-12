"""
risk_engine.py v2
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
    0: {"max_position_pct": 1.0,  "max_drawdown_pct": 1.0,  "stop_loss_pct": 1.0,  "kelly_fraction": 1.0,  "description": "Normal limits"},
    1: {"max_position_pct": 0.80, "max_drawdown_pct": 0.85, "stop_loss_pct": 0.85, "kelly_fraction": 0.80, "description": "Tighter — high volatility"},
    2: {"max_position_pct": 0.50, "max_drawdown_pct": 0.70, "stop_loss_pct": 0.70, "kelly_fraction": 0.50, "description": "Defensive — bear market"},
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
    return abs(float(np.percentile(returns.dropna(), (1 - confidence) * 100)))
 
 
def kelly_position_size(signal_score, confidence, win_rate, avg_win, avg_loss, config, regime=0):
    if signal_score <= 0:
        return 0.0
    adjusted_win_rate = win_rate
    q = 1 - adjusted_win_rate
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    kelly_f = (adjusted_win_rate * b - q) / b
    if kelly_f <= 0:
        return 0.0
    kelly_f = max(kelly_f, 0.02)
    size = kelly_f * config.kelly_fraction
    conf_mult = CONFIDENCE_MULTIPLIERS.get(confidence, 1.0)
    size *= conf_mult
    regime_mult = REGIME_RISK_MULTIPLIERS.get(regime, REGIME_RISK_MULTIPLIERS[0])
    size *= regime_mult["kelly_fraction"]
    max_pos = config.max_position_pct * regime_mult["max_position_pct"]
    return min(size, max_pos)
 
 
def compute_portfolio_var_historical(portfolio):
    if not portfolio.positions:
        return 0.0
    conn = sqlite3.connect(DB_PATH)
    symbols = list(portfolio.positions.keys())
    placeholders = ",".join([f"'{s}'" for s in symbols])
    df = pd.read_sql(f"""
        SELECT Date, Symbol, Daily_Return FROM prices
        WHERE Symbol IN ({placeholders}) ORDER BY Date
    """, conn)
    conn.close()
    if df.empty or len(df) < 30:
        return 0.0
    matrix = df.pivot(index="Date", columns="Symbol", values="Daily_Return").dropna()
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
 
 
def evaluate_trade(
    symbol, signal_score, current_price, portfolio, config,
    regime=0, confidence="MEDIUM", hurst=0.5, cvar_mc=0.0,
):
    risk_notes = []
    regime_mult = REGIME_RISK_MULTIPLIERS.get(regime, REGIME_RISK_MULTIPLIERS[0])
    risk_notes.append(f"Regime {regime}: {regime_mult['description']}")
 
    if abs(signal_score) < config.min_signal_strength:
        return TradeDecision(
            approved=False, symbol=symbol, action="HOLD",
            shares=0, capital_allocated=0,
            rejection_reason=f"Signal too weak ({signal_score:.3f} < {config.min_signal_strength})",
            risk_notes=risk_notes,
        )
 
    action = "BUY" if signal_score > 0 else "SELL"
 
    current_drawdown = portfolio.drawdown()
    max_dd = config.max_drawdown_pct * regime_mult["max_drawdown_pct"]
    if current_drawdown > max_dd:
        return TradeDecision(
            approved=False, symbol=symbol, action="HOLD",
            shares=0, capital_allocated=0,
            rejection_reason=f"Drawdown circuit breaker: {current_drawdown:.1%} > {max_dd:.1%}",
            risk_notes=risk_notes,
        )
    risk_notes.append(f"Drawdown OK: {current_drawdown:.2%} (limit {max_dd:.2%})")
 
    max_pos = config.max_position_pct * regime_mult["max_position_pct"]
    if action == "BUY":
        current_weight = portfolio.position_weight(symbol)
        if current_weight >= max_pos:
            return TradeDecision(
                approved=False, symbol=symbol, action="HOLD",
                shares=0, capital_allocated=0,
                rejection_reason=f"Concentration limit: {current_weight:.1%} >= {max_pos:.1%}",
                risk_notes=risk_notes,
            )
        risk_notes.append(f"Concentration OK: {current_weight:.2%} (limit {max_pos:.2%})")
 
    hurst_stop_adj = 1.3 if hurst > 0.6 else (0.8 if hurst < 0.4 else 1.0)
    stop_pct = config.stop_loss_pct * regime_mult["stop_loss_pct"] * hurst_stop_adj
 
    if symbol in portfolio.positions:
        entry_price = portfolio.positions[symbol]["entry_price"]
        loss_pct = (current_price - entry_price) / entry_price
        if loss_pct < -stop_pct:
            return TradeDecision(
                approved=True, symbol=symbol, action="SELL",
                shares=portfolio.positions[symbol]["shares"],
                capital_allocated=0,
                risk_notes=[f"STOP LOSS: {loss_pct:.2%} loss (limit -{stop_pct:.2%})"],
            )
        risk_notes.append(f"Stop loss OK: {loss_pct:+.2%} from entry (limit -{stop_pct:.2%})")
 
    size_fraction = kelly_position_size(
        signal_score=abs(signal_score),
        confidence=confidence,
        win_rate=0.54,
        avg_win=0.018,
        avg_loss=0.012,
        config=config,
        regime=regime,
    )
 
    capital_to_allocate = portfolio.total_value() * size_fraction
    shares = int(capital_to_allocate / current_price)
 
    if shares == 0:
        return TradeDecision(
            approved=False, symbol=symbol, action="HOLD",
            shares=0, capital_allocated=0,
            rejection_reason=f"Kelly sizing = 0 shares (signal={signal_score:.3f}, confidence={confidence})",
            risk_notes=risk_notes,
        )
    risk_notes.append(f"Kelly size: {size_fraction:.2%} = {shares} shares (confidence={confidence})")
 
    if cvar_mc != 0.0:
        portfolio_var = abs(cvar_mc) / 100
        var_source = "Monte Carlo CVaR"
    else:
        portfolio_var = compute_portfolio_var_historical(portfolio)
        var_source = "historical VaR"
 
    if portfolio_var > config.max_portfolio_var_pct:
        return TradeDecision(
            approved=False, symbol=symbol, action="HOLD",
            shares=0, capital_allocated=0,
            rejection_reason=f"{var_source} too high: {portfolio_var:.2%} > {config.max_portfolio_var_pct:.2%}",
            risk_notes=risk_notes,
        )
    risk_notes.append(f"{var_source} OK: {portfolio_var:.2%}")
 
    return TradeDecision(
        approved=True, symbol=symbol, action=action,
        shares=shares, capital_allocated=shares * current_price,
        position_size_pct=size_fraction, risk_notes=risk_notes,
    )
 
 
if __name__ == "__main__":
    config    = RiskConfig(total_capital=100_000)
    portfolio = Portfolio(cash=100_000)
 
    print("Risk Engine v2 Test\n" + "─" * 50)
    d = evaluate_trade(
        symbol="AXISBANK", signal_score=0.327, current_price=1050.0,
        portfolio=portfolio, config=config,
        regime=0, confidence="LOW", hurst=0.977, cvar_mc=-9.01,
    )
    print(f"Approved : {d.approved}")
    print(f"Action   : {d.action}")
    print(f"Shares   : {d.shares}")
    print(f"Capital  : ₹{d.capital_allocated:,.0f}")
    for n in d.risk_notes:
        print(f"  [{n}]")
 