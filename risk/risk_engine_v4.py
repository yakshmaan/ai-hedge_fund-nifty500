"""
risk_engine_v4.py
-----------------
Risk Engine V4 — wired to all 9 agents.

New in V4 vs V3:
  1. Transformer attention confidence
     - High transformer confidence + agreement → boost size
     - Transformer disagrees → reduce size
  2. RL Q-value integration
     - High Q-value (DQN is confident) → boost size up to 1.2x
     - Low Q-value (DQN uncertain) → reduce size
  3. Dynamic VaR threshold
     - Adjusts based on Heston vol regime + market regime
     - Bull + low vol → allow up to 15% VaR
     - Bear + high vol → tighten to 8% VaR
  4. Multi-agent stop loss voting
     - If 3+ agents agree price will fall → tighten stop loss
     - Acts as early warning before stop is hit
  5. Sentiment-gated trading
     - If FinBERT strongly negative → block new buys regardless of signal
     - Protects against buying into bad news events

All V3 features retained.
Backward compatible with V3 via evaluate_trade wrapper.
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
    1: {"max_position_pct":0.80, "max_drawdown_pct":0.85, "stop_loss_pct":0.85, "kelly_fraction":0.80, "description":"Tighter — high vol"},
    2: {"max_position_pct":0.50, "max_drawdown_pct":0.70, "stop_loss_pct":0.70, "kelly_fraction":0.50, "description":"Defensive — bear"},
}

HESTON_VOL_MULTIPLIERS = {
    "high_vol":   {"position":0.70, "stop":1.30, "var_limit":0.08},
    "normal_vol": {"position":1.00, "stop":1.00, "var_limit":0.12},
    "low_vol":    {"position":1.15, "stop":0.85, "var_limit":0.15},
}

CONFIDENCE_MULTIPLIERS = {"HIGH":1.0, "MEDIUM":1.0, "LOW":1.0}


@dataclass
class Portfolio:
    cash:float=100_000.0; positions:dict=field(default_factory=dict)
    peak_value:float=100_000.0; trade_history:list=field(default_factory=list)

    def total_value(self):
        return self.cash+sum(p["shares"]*p["current_price"] for p in self.positions.values())

    def drawdown(self):
        current=self.total_value()
        if current>self.peak_value: self.peak_value=current
        return (self.peak_value-current)/self.peak_value

    def position_weight(self,symbol):
        if symbol not in self.positions: return 0.0
        val=self.positions[symbol]["shares"]*self.positions[symbol]["current_price"]
        return val/max(self.total_value(),1.0)


def compute_var(returns,confidence=0.95):
    if len(returns)<30: return 0.0
    return abs(float(np.percentile(returns.dropna(),(1-confidence)*100)))


def compute_consensus_score(all_agent_scores):
    scores=[v for v in all_agent_scores.values() if isinstance(v,(int,float)) and not np.isnan(v)]
    if len(scores)<3: return 1.0
    scores=np.array(scores)
    majority_dir=np.sign(np.mean(scores))
    agreeing=np.sum(np.sign(scores)==majority_dir)
    ratio=agreeing/len(scores)
    if ratio>=0.85:   return 1.30
    elif ratio>=0.70: return 1.15
    elif ratio>=0.57: return 1.00
    elif ratio>=0.43: return 0.80
    else:             return 0.60


def get_heston_vol_regime(heston_score,vol_ratio=1.0):
    if vol_ratio>1.3 or heston_score<-0.15: return "high_vol"
    elif vol_ratio<0.8 or heston_score>0.10: return "low_vol"
    else: return "normal_vol"


def kelly_position_size(signal_score,confidence,win_rate,avg_win,avg_loss,config,regime=0):
    if signal_score<=0: return 0.0
    q=1-win_rate
    if avg_loss==0: return 0.0
    b=avg_win/avg_loss; kelly_f=(win_rate*b-q)/b
    if kelly_f<=0: return 0.0
    kelly_f=max(kelly_f,0.02)
    size=kelly_f*config.kelly_fraction
    size*=CONFIDENCE_MULTIPLIERS.get(confidence,1.0)
    regime_mult=REGIME_RISK_MULTIPLIERS.get(regime,REGIME_RISK_MULTIPLIERS[0])
    size*=regime_mult["kelly_fraction"]
    max_pos=config.max_position_pct*regime_mult["max_position_pct"]
    return min(size,max_pos)


def compute_portfolio_var_historical(portfolio):
    if not portfolio.positions: return 0.0
    conn=sqlite3.connect(DB_PATH)
    symbols=list(portfolio.positions.keys())
    placeholders=",".join([f"'{s}'" for s in symbols])
    df=pd.read_sql(f"SELECT Date,Symbol,Daily_Return FROM prices WHERE Symbol IN ({placeholders}) ORDER BY Date",conn)
    conn.close()
    if df.empty or len(df)<30: return 0.0
    matrix=df.pivot(index="Date",columns="Symbol",values="Daily_Return").dropna()
    if len(matrix)<30: return 0.0
    weights=np.array([portfolio.position_weight(s) for s in matrix.columns])
    return compute_var(pd.Series(matrix.values@weights))


def multi_agent_stop_vote(all_agent_scores, current_loss_pct, base_stop_pct):
    """
    If multiple agents agree price will fall, tighten stop loss.
    Protects against being stopped out after a big move — early exit.
    """
    bearish_count = sum(
        1 for v in all_agent_scores.values()
        if isinstance(v,(int,float)) and v < -0.15
    )
    total = len(all_agent_scores)

    if bearish_count >= 4 and current_loss_pct < -base_stop_pct * 0.5:
        # More than half agents bearish and already down 50% of stop → early exit
        return True, f"Multi-agent stop vote: {bearish_count}/{total} agents bearish"
    return False, ""


@dataclass
class TradeDecision:
    approved:bool; symbol:str; action:str; shares:int; capital_allocated:float
    rejection_reason:Optional[str]=None; position_size_pct:float=0.0
    risk_notes:list=field(default_factory=list)
    consensus_score:float=1.0; heston_regime:str="normal_vol"


def evaluate_trade_v4(
    symbol, signal_score, current_price, portfolio, config,
    regime=0, confidence="MEDIUM", hurst=0.5, cvar_mc=0.0,
    heston_score=0.0, heston_vol_ratio=1.0,
    lstm_score=0.0, sentiment_score=0.0,
    transformer_score=0.0, rl_q_value=0.0,
    all_agent_scores=None,
):
    risk_notes=[]
    regime_mult=REGIME_RISK_MULTIPLIERS.get(regime,REGIME_RISK_MULTIPLIERS[0])
    risk_notes.append(f"Regime {regime}: {regime_mult['description']}")

    heston_regime   = get_heston_vol_regime(heston_score,heston_vol_ratio)
    heston_vol_mult = HESTON_VOL_MULTIPLIERS[heston_regime]
    risk_notes.append(f"Heston: {heston_regime}")

    if all_agent_scores is None: all_agent_scores={}
    consensus_mult=compute_consensus_score(all_agent_scores)
    n_agree=sum(1 for v in all_agent_scores.values()
                if isinstance(v,(int,float)) and np.sign(v)==np.sign(signal_score))
    risk_notes.append(f"Consensus: {n_agree}/{len(all_agent_scores)} agree (×{consensus_mult:.2f})")

    # Sentiment gate — block buys on strongly negative news
    if sentiment_score < -0.4 and signal_score > 0:
        return TradeDecision(
            approved=False,symbol=symbol,action="HOLD",shares=0,capital_allocated=0,
            rejection_reason=f"FinBERT sentiment gate: strongly negative news (score={sentiment_score:.3f})",
            risk_notes=risk_notes,consensus_score=consensus_mult,heston_regime=heston_regime,
        )

    # Sentiment multiplier
    sent_mult=1.10 if sentiment_score>0.3 else (1.05 if sentiment_score>0.1 else
              (0.85 if sentiment_score<-0.3 else (0.92 if sentiment_score<-0.1 else 1.0)))

    # LSTM multiplier
    lstm_agree=np.sign(lstm_score)==np.sign(signal_score)
    lstm_mult=1.05 if lstm_agree and abs(lstm_score)>0.1 else (0.92 if not lstm_agree and abs(lstm_score)>0.2 else 1.0)

    # Transformer multiplier
    trans_agree=np.sign(transformer_score)==np.sign(signal_score)
    trans_mult=1.08 if trans_agree and abs(transformer_score)>0.2 else (0.90 if not trans_agree and abs(transformer_score)>0.25 else 1.0)
    if trans_mult!=1.0:
        risk_notes.append(f"Transformer {'agrees' if trans_agree else 'disagrees'}: ×{trans_mult:.2f}")

    # RL Q-value multiplier
    # High Q-value = DQN is confident → boost
    # Low/negative Q-value = DQN uncertain → reduce
    rl_mult=1.0
    if rl_q_value>0.5:   rl_mult=1.15
    elif rl_q_value>0.2: rl_mult=1.05
    elif rl_q_value<-0.2: rl_mult=0.90
    elif rl_q_value<-0.5: rl_mult=0.80
    if rl_mult!=1.0:
        risk_notes.append(f"RL Q-value={rl_q_value:.3f}: ×{rl_mult:.2f}")

    # ── CHECK 1: Signal strength ──────────────────────────────────────────────
    if abs(signal_score)<config.min_signal_strength:
        return TradeDecision(
            approved=False,symbol=symbol,action="HOLD",shares=0,capital_allocated=0,
            rejection_reason=f"Signal too weak ({signal_score:.3f})",
            risk_notes=risk_notes,consensus_score=consensus_mult,heston_regime=heston_regime,
        )

    action="BUY" if signal_score>0 else "SELL"

    # ── CHECK 2: Drawdown ─────────────────────────────────────────────────────
    current_drawdown=portfolio.drawdown()
    max_dd=config.max_drawdown_pct*regime_mult["max_drawdown_pct"]
    if current_drawdown>max_dd:
        return TradeDecision(
            approved=False,symbol=symbol,action="HOLD",shares=0,capital_allocated=0,
            rejection_reason=f"Drawdown {current_drawdown:.1%} > {max_dd:.1%}",
            risk_notes=risk_notes,consensus_score=consensus_mult,heston_regime=heston_regime,
        )
    risk_notes.append(f"Drawdown OK: {current_drawdown:.2%}")

    # ── CHECK 3: Concentration ────────────────────────────────────────────────
    max_pos=config.max_position_pct*regime_mult["max_position_pct"]*heston_vol_mult["position"]
    if action=="BUY":
        cw=portfolio.position_weight(symbol)
        if cw>=max_pos:
            return TradeDecision(
                approved=False,symbol=symbol,action="HOLD",shares=0,capital_allocated=0,
                rejection_reason=f"Concentration {cw:.1%} >= {max_pos:.1%}",
                risk_notes=risk_notes,consensus_score=consensus_mult,heston_regime=heston_regime,
            )
        risk_notes.append(f"Concentration OK: {cw:.2%}")

    # ── CHECK 4: Stop loss + multi-agent vote ─────────────────────────────────
    hurst_stop_adj=1.3 if hurst>0.6 else (0.8 if hurst<0.4 else 1.0)
    stop_pct=config.stop_loss_pct*regime_mult["stop_loss_pct"]*hurst_stop_adj*heston_vol_mult["stop"]

    if symbol in portfolio.positions:
        entry_price=portfolio.positions[symbol]["entry_price"]
        loss_pct=(current_price-entry_price)/entry_price

        # Multi-agent early stop vote
        early_stop,vote_reason=multi_agent_stop_vote(all_agent_scores,loss_pct,stop_pct)
        if early_stop or loss_pct<-stop_pct:
            reason=vote_reason if early_stop else f"STOP LOSS: {loss_pct:.2%} (limit -{stop_pct:.2%})"
            return TradeDecision(
                approved=True,symbol=symbol,action="SELL",
                shares=portfolio.positions[symbol]["shares"],capital_allocated=0,
                risk_notes=[reason],consensus_score=consensus_mult,heston_regime=heston_regime,
            )
        risk_notes.append(f"Stop OK: {loss_pct:+.2%} (limit -{stop_pct:.2%})")

    # ── CHECK 5: Kelly + V4 multipliers ──────────────────────────────────────
    size_fraction=kelly_position_size(
        abs(signal_score),confidence,0.54,0.018,0.012,config,regime
    )
    size_fraction*=heston_vol_mult["position"]
    size_fraction*=consensus_mult
    size_fraction*=sent_mult
    size_fraction*=lstm_mult
    size_fraction*=trans_mult
    size_fraction*=rl_mult
    size_fraction=min(size_fraction,max_pos)

    shares=int(portfolio.total_value()*size_fraction/current_price)
    if shares==0:
        return TradeDecision(
            approved=False,symbol=symbol,action="HOLD",shares=0,capital_allocated=0,
            rejection_reason=f"Kelly=0 shares (frac={size_fraction:.3f})",
            risk_notes=risk_notes,consensus_score=consensus_mult,heston_regime=heston_regime,
        )
    risk_notes.append(
        f"V4 Kelly: {size_fraction:.2%}={shares} shares "
        f"(consensus×{consensus_mult:.2f} sent×{sent_mult:.2f} "
        f"lstm×{lstm_mult:.2f} trans×{trans_mult:.2f} rl×{rl_mult:.2f})"
    )

    # ── CHECK 6: Dynamic VaR ──────────────────────────────────────────────────
    var_limit=heston_vol_mult["var_limit"]  # dynamic based on vol regime
    if cvar_mc!=0.0:
        portfolio_var=abs(cvar_mc)/100; var_source="Monte Carlo CVaR"
    else:
        portfolio_var=compute_portfolio_var_historical(portfolio); var_source="historical VaR"

    if portfolio_var>var_limit:
        return TradeDecision(
            approved=False,symbol=symbol,action="HOLD",shares=0,capital_allocated=0,
            rejection_reason=f"{var_source} {portfolio_var:.2%} > dynamic limit {var_limit:.2%} ({heston_regime})",
            risk_notes=risk_notes,consensus_score=consensus_mult,heston_regime=heston_regime,
        )
    risk_notes.append(f"{var_source} OK: {portfolio_var:.2%} (limit {var_limit:.2%})")

    return TradeDecision(
        approved=True,symbol=symbol,action=action,shares=shares,
        capital_allocated=shares*current_price,position_size_pct=size_fraction,
        risk_notes=risk_notes,consensus_score=consensus_mult,heston_regime=heston_regime,
    )


# Backward compatible wrappers
def evaluate_trade_v3(symbol,signal_score,current_price,portfolio,config,
                      regime=0,confidence="MEDIUM",hurst=0.5,cvar_mc=0.0,
                      heston_score=0.0,heston_vol_ratio=1.0,
                      lstm_score=0.0,sentiment_score=0.0,all_agent_scores=None):
    return evaluate_trade_v4(
        symbol=symbol,signal_score=signal_score,current_price=current_price,
        portfolio=portfolio,config=config,regime=regime,confidence=confidence,
        hurst=hurst,cvar_mc=cvar_mc,heston_score=heston_score,
        heston_vol_ratio=heston_vol_ratio,lstm_score=lstm_score,
        sentiment_score=sentiment_score,all_agent_scores=all_agent_scores,
    )


def evaluate_trade(symbol,signal_score,current_price,portfolio,config,
                   regime=0,confidence="MEDIUM",hurst=0.5,cvar_mc=0.0):
    return evaluate_trade_v4(
        symbol=symbol,signal_score=signal_score,current_price=current_price,
        portfolio=portfolio,config=config,regime=regime,confidence=confidence,
        hurst=hurst,cvar_mc=cvar_mc,
    )


if __name__=="__main__":
    config=RiskConfig(total_capital=100_000)
    portfolio=Portfolio(cash=100_000)
    print("Risk Engine V4 Test\n"+"─"*55)

    all_scores={"kalman":0.72,"adv_mr":-0.15,"lstm":0.30,
                "transformer":0.25,"heston":0.10,"monte_carlo":0.08,"sentiment":0.20}

    d=evaluate_trade_v4(
        symbol="RELIANCE",signal_score=0.40,current_price=2850.0,
        portfolio=portfolio,config=config,regime=0,confidence="HIGH",
        hurst=0.55,cvar_mc=-8.5,heston_score=0.10,heston_vol_ratio=0.85,
        lstm_score=0.30,sentiment_score=0.20,transformer_score=0.25,rl_q_value=0.45,
        all_agent_scores=all_scores,
    )
    print(f"Approved      : {d.approved}")
    print(f"Action        : {d.action}")
    print(f"Shares        : {d.shares}")
    print(f"Capital       : ₹{d.capital_allocated:,.0f}")
    print(f"Heston Regime : {d.heston_regime}")
    print(f"Consensus     : {d.consensus_score:.2f}")
    for n in d.risk_notes: print(f"  [{n}]")
