"""
regime_agent.py
---------------
Regime Detection Agent using Hidden Markov Model (HMM).
 
Markets don't behave the same way all the time. They switch between:
  - Regime 0: Low volatility bull  (slow uptrend, buy momentum)
  - Regime 1: High volatility bull (fast uptrend, reduce size)
  - Regime 2: Bear / crisis        (downtrend, go defensive)
 
Why HMM?
  A Hidden Markov Model assumes the market is always in one of N hidden
  states. We can't observe the state directly — we only observe returns
  and volatility. HMM learns the most likely state sequence from data
  and gives us the probability of being in each state right now.
 
Output:
  - current_regime: 0, 1, or 2
  - regime_probs: [p0, p1, p2] probabilities for each regime
  - agent_weights: how to reweight other agents given this regime
 
Usage:
    python agents/regime_agent.py
"""
 
import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings("ignore")
 
DB_PATH = "data/nifty50.db"
 
# ── HMM IMPLEMENTATION (from scratch — no hmmlearn needed) ───────────────────
 
class GaussianHMM:
    """
    Gaussian Hidden Markov Model trained with Baum-Welch algorithm.
 
    States emit observations drawn from Gaussian distributions.
    We use returns and volatility as observations.
 
    Parameters:
        n_states: number of hidden market regimes
        n_iter:   number of EM iterations
    """
 
    def __init__(self, n_states: int = 3, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.fitted   = False
 
    def _init_params(self, X: np.ndarray):
        n, d = X.shape
 
        # Start probabilities — equal chance of starting in any state
        self.pi = np.ones(self.n_states) / self.n_states
 
        # Transition matrix — mostly stay in same state
        self.A = np.full((self.n_states, self.n_states), 0.1)
        np.fill_diagonal(self.A, 0.7)
        self.A /= self.A.sum(axis=1, keepdims=True)
 
        # Emission parameters: mean and variance per state
        # Initialize by splitting data into n_states chunks
        chunk = n // self.n_states
        self.means = np.array([
            X[i * chunk:(i + 1) * chunk].mean(axis=0)
            for i in range(self.n_states)
        ])
        self.covars = np.array([
            np.cov(X[i * chunk:(i + 1) * chunk].T) + np.eye(d) * 1e-4
            for i in range(self.n_states)
        ])
 
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        d = len(mean)
        diff = x - mean
        try:
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            if det_cov <= 0:
                det_cov = 1e-10
            exponent = -0.5 * diff @ inv_cov @ diff
            coeff = 1.0 / (np.sqrt((2 * np.pi) ** d * det_cov))
            return max(coeff * np.exp(exponent), 1e-300)
        except Exception:
            return 1e-300
 
    def _emission_probs(self, X: np.ndarray) -> np.ndarray:
        """Compute emission probabilities B[t, s] for all t and s."""
        n = len(X)
        B = np.zeros((n, self.n_states))
        for t in range(n):
            for s in range(self.n_states):
                B[t, s] = self._gaussian_pdf(X[t], self.means[s], self.covars[s])
        return B
 
    def _forward(self, B: np.ndarray):
        n = len(B)
        alpha = np.zeros((n, self.n_states))
        alpha[0] = self.pi * B[0]
        alpha[0] /= alpha[0].sum() + 1e-300
 
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            scale = alpha[t].sum()
            alpha[t] /= scale + 1e-300
 
        return alpha
 
    def _backward(self, B: np.ndarray):
        n = len(B)
        beta = np.zeros((n, self.n_states))
        beta[-1] = 1.0
 
        for t in range(n - 2, -1, -1):
            beta[t] = self.A @ (B[t + 1] * beta[t + 1])
            scale = beta[t].sum()
            beta[t] /= scale + 1e-300
 
        return beta
 
    def fit(self, X: np.ndarray):
        self._init_params(X)
        n = len(X)
 
        for iteration in range(self.n_iter):
            B     = self._emission_probs(X)
            alpha = self._forward(B)
            beta  = self._backward(B)
 
            # Gamma: probability of being in state s at time t
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
 
            # Xi: probability of transitioning s->s' at time t
            xi = np.zeros((n - 1, self.n_states, self.n_states))
            for t in range(n - 1):
                xi[t] = (
                    alpha[t][:, None] *
                    self.A *
                    B[t + 1][None, :] *
                    beta[t + 1][None, :]
                )
                xi[t] /= xi[t].sum() + 1e-300
 
            # M-step: update parameters
            self.pi = gamma[0] / gamma[0].sum()
            self.A  = xi.sum(axis=0) / xi.sum(axis=0).sum(axis=1, keepdims=True)
 
            for s in range(self.n_states):
                w = gamma[:, s]
                self.means[s]  = (w[:, None] * X).sum(axis=0) / (w.sum() + 1e-300)
                diff = X - self.means[s]
                self.covars[s] = (
                    (w[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
                    / (w.sum() + 1e-300)
                ) + np.eye(X.shape[1]) * 1e-4
 
        self.fitted = True
        return self
 
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return state probabilities for each observation."""
        B     = self._emission_probs(X)
        alpha = self._forward(B)
        beta  = self._backward(B)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
        return gamma
 
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return most likely state sequence (Viterbi)."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
 
 
# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
 
def build_regime_features(close: pd.Series) -> np.ndarray:
    """
    Build observation matrix for HMM.
    Features: rolling return, rolling volatility, rolling skew.
    All normalized so HMM can learn meaningful Gaussian distributions.
    """
    returns  = close.pct_change().fillna(0)
    vol_10   = returns.rolling(10).std().fillna(returns.std())
    vol_20   = returns.rolling(20).std().fillna(returns.std())
    ret_5    = close.pct_change(5).fillna(0)
 
    X = np.column_stack([
        returns.values,
        vol_10.values,
        vol_20.values,
        ret_5.values,
    ])
 
    # Normalize each feature to zero mean unit variance
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X
 
 
# ── REGIME LABELING ───────────────────────────────────────────────────────────
 
def label_regimes(model: GaussianHMM, X: np.ndarray, close: pd.Series) -> dict:
    """
    After fitting, figure out which state index corresponds to which regime.
    We label states by their average return: highest = bull, lowest = bear.
    """
    states    = model.predict(X)
    returns   = close.pct_change().fillna(0)
 
    state_returns = {}
    for s in range(model.n_states):
        mask = states == s
        if mask.sum() > 0:
            state_returns[s] = returns[mask].mean()
 
    sorted_states = sorted(state_returns, key=state_returns.get)
 
    # Map: worst avg return → bear, middle → high vol bull, best → low vol bull
    label_map = {}
    if len(sorted_states) >= 3:
        label_map[sorted_states[0]] = 2   # bear
        label_map[sorted_states[1]] = 1   # high vol
        label_map[sorted_states[2]] = 0   # low vol bull
    elif len(sorted_states) == 2:
        label_map[sorted_states[0]] = 2
        label_map[sorted_states[1]] = 0
 
    return label_map
 
 
# ── AGENT WEIGHTS PER REGIME ──────────────────────────────────────────────────
 
REGIME_WEIGHTS = {
    0: {  # Low volatility bull — trust momentum most
        "momentum":       0.50,
        "mean_reversion": 0.20,
        "ml_classifier":  0.30,
        "description":    "Low vol bull market — momentum dominant",
    },
    1: {  # High volatility bull — balance all agents
        "momentum":       0.30,
        "mean_reversion": 0.35,
        "ml_classifier":  0.35,
        "description":    "High vol market — balanced signals",
    },
    2: {  # Bear / crisis — trust mean reversion, reduce momentum
        "momentum":       0.15,
        "mean_reversion": 0.50,
        "ml_classifier":  0.35,
        "description":    "Bear market — mean reversion dominant, reduce exposure",
    },
}
 
 
# ── MAIN AGENT FUNCTION ───────────────────────────────────────────────────────
 
def get_regime(symbol: str = "NIFTY50_AVG") -> dict:
    """
    Detect current market regime using HMM trained on Nifty 50 average.
 
    Returns regime info + recommended agent weights for orchestrator.
    """
    conn = sqlite3.connect(DB_PATH)
 
    # Use average of all stocks as market proxy
    df = pd.read_sql("""
        SELECT Date, AVG(Close) as Close
        FROM prices
        GROUP BY Date
        ORDER BY Date
    """, conn)
    conn.close()
 
    if len(df) < 100:
        return {
            "regime": 0,
            "regime_name": "unknown",
            "probabilities": [0.33, 0.33, 0.34],
            "agent_weights": REGIME_WEIGHTS[0],
            "description": "Not enough data",
        }
 
    close = df.set_index("Date")["Close"]
    X     = build_regime_features(close)
 
    # Train HMM
    model     = GaussianHMM(n_states=3, n_iter=80)
    model.fit(X)
 
    # Get current regime (last observation)
    probs      = model.predict_proba(X)
    label_map  = label_regimes(model, X, close)
 
    raw_regime      = int(np.argmax(probs[-1]))
    current_regime  = label_map.get(raw_regime, 0)
 
    # Remap probabilities to labeled regimes
    regime_probs = [0.0, 0.0, 0.0]
    for raw, labeled in label_map.items():
        if labeled < 3:
            regime_probs[labeled] = float(probs[-1][raw])
 
    regime_names = {
        0: "low volatility bull",
        1: "high volatility bull",
        2: "bear / crisis",
    }
 
    return {
        "regime":       current_regime,
        "regime_name":  regime_names.get(current_regime, "unknown"),
        "probabilities": [round(p, 4) for p in regime_probs],
        "agent_weights": REGIME_WEIGHTS[current_regime],
        "description":  REGIME_WEIGHTS[current_regime]["description"],
    }
 
 
if __name__ == "__main__":
    print("Detecting market regime...\n")
    result = get_regime()
    print(f"Current Regime  : {result['regime']} — {result['regime_name']}")
    print(f"Probabilities   : Bull={result['probabilities'][0]:.2%}  "
          f"HighVol={result['probabilities'][1]:.2%}  "
          f"Bear={result['probabilities'][2]:.2%}")
    print(f"Description     : {result['description']}")
    print(f"\nRecommended agent weights:")
    w = result["agent_weights"]
    print(f"  Momentum       : {w['momentum']:.0%}")
    print(f"  Mean Reversion : {w['mean_reversion']:.0%}")
    print(f"  ML Classifier  : {w['ml_classifier']:.0%}")
 