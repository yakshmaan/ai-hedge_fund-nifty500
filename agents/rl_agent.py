"""
rl_agent.py
-----------
Reinforcement Learning Agent — Deep Q-Network (DQN) for dynamic
agent weight optimization.

Instead of fixed regime-based weights, the RL agent LEARNS
the optimal weights through trial and error.

How it works:
  State:  current scores from all 6 signal agents + market features
  Action: one of 27 discrete weight combinations to try
  Reward: portfolio return over next 5 days using those weights
  
  The DQN learns: "given these agent scores and market conditions,
  which weighting scheme has historically produced the best returns?"

Q-Network architecture:
  Input:  state vector (agent scores + market features) = 14 dims
  Hidden: 64 → 32 neurons with ReLU
  Output: Q-values for each action (27 possible weight combos)

Training:
  Experience replay buffer — stores (state, action, reward, next_state)
  Mini-batch updates from replay buffer prevent correlated learning
  Epsilon-greedy exploration — starts random, becomes more confident

Output:
  Optimal weight vector for current market conditions
  Score: weighted combination using RL-selected weights
"""

import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/nifty50.db"


# ── Q-NETWORK ─────────────────────────────────────────────────────────────────

class QNetwork:
    """
    Simple feedforward Q-network implemented in numpy.
    Maps state → Q-values for each action.
    """

    def __init__(self, state_size, action_size, hidden1=64, hidden2=32, seed=42):
        rng    = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / state_size)
        scale2 = np.sqrt(2.0 / hidden1)
        scale3 = np.sqrt(2.0 / hidden2)

        self.W1 = rng.normal(0, scale1, (state_size, hidden1))
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.normal(0, scale2, (hidden1, hidden2))
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.normal(0, scale3, (hidden2, action_size))
        self.b3 = np.zeros(action_size)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, state):
        """state: (state_size,) → Q-values: (action_size,)"""
        h1 = self.relu(state @ self.W1 + self.b1)
        h2 = self.relu(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def update(self, state, action, target_q, lr=0.001):
        """
        Simple gradient update for one (state, action, target) tuple.
        Uses mean squared error: loss = (Q(s,a) - target)^2
        """
        # Forward pass
        h1    = self.relu(state @ self.W1 + self.b1)
        h2    = self.relu(h1 @ self.W2 + self.b2)
        q_out = h2 @ self.W3 + self.b3

        # Error at output
        error       = np.zeros_like(q_out)
        error[action] = q_out[action] - target_q

        # Backprop through output layer
        dW3 = np.outer(h2, error)
        db3 = error

        # Backprop through hidden layer 2
        dh2 = self.W3 @ error
        dh2 *= (h2 > 0)  # ReLU gradient
        dW2  = np.outer(h1, dh2)
        db2  = dh2

        # Backprop through hidden layer 1
        dh1 = self.W2 @ dh2
        dh1 *= (h1 > 0)
        dW1  = np.outer(state, dh1)
        db1  = dh1

        # Update weights
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


# ── ACTION SPACE ──────────────────────────────────────────────────────────────

def build_action_space():
    """
    27 discrete weight combinations for 6 agents.
    Each agent gets low/medium/high weight.
    We normalize so weights sum to 1.

    Agents: kalman, mean_rev, lstm, heston, monte_carlo, sentiment
    """
    actions = []
    # 3 levels per agent, simplified to 27 key combinations
    weight_levels = [0.05, 0.15, 0.30]

    # Sample representative combinations
    for w_mom in weight_levels:
        for w_mr in weight_levels:
            for w_ml in weight_levels:
                w_rest = (1.0 - w_mom - w_mr - w_ml)
                if w_rest < 0.1:
                    continue
                w_each = w_rest / 3
                actions.append({
                    "kalman":       w_mom,
                    "mean_rev":     w_mr,
                    "lstm":         w_ml,
                    "heston":       w_each,
                    "monte_carlo":  w_each,
                    "sentiment":    w_each,
                })

    # Normalize each action
    normalized = []
    for a in actions:
        total = sum(a.values())
        normalized.append({k: v/total for k, v in a.items()})

    return normalized


ACTION_SPACE = build_action_space()
N_ACTIONS    = len(ACTION_SPACE)
STATE_SIZE   = 10  # 6 agent scores + 4 market features


# ── EXPERIENCE REPLAY ─────────────────────────────────────────────────────────

class ReplayBuffer:
    """Stores past experiences for mini-batch training."""

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer   = []
        self.pos      = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, rng):
        idx     = rng.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch   = [self.buffer[i] for i in idx]
        states  = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        nstates = np.array([b[3] for b in batch])
        return states, actions, rewards, nstates

    def __len__(self):
        return len(self.buffer)


# ── DQN TRAINING ──────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network agent that learns optimal agent weights.
    """

    def __init__(self, state_size=STATE_SIZE, action_size=N_ACTIONS, seed=42):
        self.q_net     = QNetwork(state_size, action_size, seed=seed)
        self.target_net = QNetwork(state_size, action_size, seed=seed+1)
        self.buffer    = ReplayBuffer(capacity=2000)
        self.epsilon   = 1.0    # exploration rate
        self.eps_min   = 0.05
        self.eps_decay = 0.995
        self.gamma     = 0.95   # discount factor
        self.rng       = np.random.default_rng(seed)
        self.step_count = 0

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_ACTIONS))
        q_vals = self.q_net.forward(state)
        return int(np.argmax(q_vals))

    def train_step(self, batch_size=32, lr=0.001):
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states = self.buffer.sample(batch_size, self.rng)
        gamma = self.gamma

        for i in range(len(states)):
            # TD target: r + γ * max Q(s', a')
            next_q  = self.target_net.forward(next_states[i])
            target  = rewards[i] + gamma * np.max(next_q)
            self.q_net.update(states[i], actions[i], target, lr=lr)

        # Decay epsilon
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        self.step_count += 1

        # Sync target network every 50 steps
        if self.step_count % 50 == 0:
            self.target_net.W1 = self.q_net.W1.copy()
            self.target_net.W2 = self.q_net.W2.copy()
            self.target_net.W3 = self.q_net.W3.copy()

    def get_optimal_weights(self, state):
        """Return optimal weight combination for current state."""
        q_vals = self.q_net.forward(state)
        action = int(np.argmax(q_vals))
        return ACTION_SPACE[action], action, float(q_vals[action])


# ── STATE BUILDING ────────────────────────────────────────────────────────────

def build_state(agent_scores, market_features):
    """
    Build state vector from agent scores and market features.

    agent_scores: dict with keys kalman, mean_rev, lstm, heston, monte_carlo, sentiment
    market_features: dict with vol, trend, regime, momentum
    """
    state = np.array([
        agent_scores.get("kalman",       0.0),
        agent_scores.get("mean_rev",     0.0),
        agent_scores.get("lstm",         0.0),
        agent_scores.get("heston",       0.0),
        agent_scores.get("monte_carlo",  0.0),
        agent_scores.get("sentiment",    0.0),
        market_features.get("vol",       0.0),
        market_features.get("trend",     0.0),
        market_features.get("regime",    0.0),
        market_features.get("momentum",  0.0),
    ], dtype=float)

    # Normalize state
    norm = np.linalg.norm(state)
    if norm > 0:
        state = state / norm

    return state


# ── TRAIN RL AGENT ON HISTORICAL DATA ────────────────────────────────────────

def train_rl_agent(symbol, n_episodes=50):
    """
    Train the DQN on historical data for one stock.
    Each episode = one trading period.
    Reward = return achieved using selected weights.
    """
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(
        f"SELECT Date, Close, Daily_Return, High, Low FROM prices "
        f"WHERE Symbol='{symbol}' ORDER BY Date",
        conn, index_col="Date"
    )
    conn.close()

    if len(df) < 200:
        return None

    agent   = DQNAgent(state_size=STATE_SIZE, action_size=N_ACTIONS)
    returns = df["Daily_Return"].fillna(0)
    close   = df["Close"]

    print(f"  Training DQN for {symbol} ({n_episodes} episodes)...")

    for episode in range(n_episodes):
        # Random starting point in history
        start = agent.rng.integers(100, max(101, len(df) - 10))
        end   = min(start + 5, len(df) - 1)

        # Build state from recent window
        window_ret = returns.iloc[start-60:start]
        vol        = float(window_ret.std())
        trend      = float(window_ret.mean())
        momentum   = float(window_ret.iloc[-5:].mean() - window_ret.mean())
        regime_val = 0.0 if trend > 0 else (2.0 if trend < -0.001 else 1.0)

        # Simple agent score proxies from price data
        ma20    = float(close.rolling(20).mean().iloc[start])
        ma50    = float(close.rolling(50).mean().iloc[start])
        current = float(close.iloc[start])

        agent_scores = {
            "kalman":      float(np.clip((current - ma20) / (ma20 + 1e-8) * 10, -1, 1)),
            "mean_rev":    float(np.clip(-(current - ma20) / (ma20 + 1e-8) * 5, -1, 1)),
            "lstm":        float(np.clip(trend * 100, -1, 1)),
            "heston":      float(np.clip(-vol * 20, -1, 1)),
            "monte_carlo": float(np.clip(trend * 50, -1, 1)),
            "sentiment":   0.0,
        }

        market_features = {
            "vol":      float(np.clip(vol * 50, 0, 1)),
            "trend":    float(np.clip(trend * 100, -1, 1)),
            "regime":   regime_val / 2.0,
            "momentum": float(np.clip(momentum * 100, -1, 1)),
        }

        state  = build_state(agent_scores, market_features)
        action = agent.select_action(state)
        weights = ACTION_SPACE[action]

        # Compute reward: portfolio return using selected weights
        combined_signal = sum(
            weights.get(k, 0) * v for k, v in agent_scores.items()
        )
        future_return = float(returns.iloc[start:end].mean())

        # Reward: signal-return alignment
        reward = combined_signal * future_return * 100

        # Next state
        next_scores = {k: v * 0.95 + agent.rng.normal(0, 0.05) for k, v in agent_scores.items()}
        next_state  = build_state(next_scores, market_features)

        agent.buffer.push(state, action, reward, next_state)
        agent.train_step(batch_size=16, lr=0.001)

    return agent


# ── AGENT INTERFACE ───────────────────────────────────────────────────────────

# Cache trained agents
_rl_agents = {}


def get_optimal_weights(symbol, agent_scores, market_features):
    """
    Get RL-optimized weights for current market conditions.
    Trains on historical data if not already trained.
    """
    global _rl_agents

    if symbol not in _rl_agents:
        agent = train_rl_agent(symbol, n_episodes=50)
        if agent is None:
            # Return default weights if training failed
            return {
                "kalman":0.28,"mean_rev":0.22,"lstm":0.20,
                "heston":0.12,"monte_carlo":0.10,"sentiment":0.08
            }, 0, 0.0
        _rl_agents[symbol] = agent

    agent = _rl_agents[symbol]
    state = build_state(agent_scores, market_features)
    weights, action, q_value = agent.get_optimal_weights(state)

    return weights, action, q_value


def get_latest_signal(symbol, agent_scores=None, market_features=None):
    """
    Get RL-optimized combined signal for a stock.
    If agent_scores not provided, uses proxy signals from price data.
    """
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(
        f"SELECT Date, Close, Daily_Return FROM prices WHERE Symbol='{symbol}' ORDER BY Date",
        conn, index_col="Date"
    )
    conn.close()

    if len(df) < 200:
        return {"symbol":symbol,"agent":"rl_dqn","score":0.0,
                "interpretation":"insufficient data","weights":{}}

    returns = df["Daily_Return"].fillna(0)
    close   = df["Close"]
    ma20    = float(close.rolling(20).mean().iloc[-1])
    ma50    = float(close.rolling(50).mean().iloc[-1])
    current = float(close.iloc[-1])
    vol     = float(returns.iloc[-60:].std())
    trend   = float(returns.iloc[-60:].mean())

    if agent_scores is None:
        agent_scores = {
            "kalman":      float(np.clip((current-ma20)/(ma20+1e-8)*10,-1,1)),
            "mean_rev":    float(np.clip(-(current-ma20)/(ma20+1e-8)*5,-1,1)),
            "lstm":        float(np.clip(trend*100,-1,1)),
            "heston":      float(np.clip(-vol*20,-1,1)),
            "monte_carlo": float(np.clip(trend*50,-1,1)),
            "sentiment":   0.0,
        }

    if market_features is None:
        market_features = {
            "vol":      float(np.clip(vol*50,0,1)),
            "trend":    float(np.clip(trend*100,-1,1)),
            "regime":   0.0 if trend > 0 else 1.0,
            "momentum": float(np.clip((returns.iloc[-5:].mean()-trend)*100,-1,1)),
        }

    weights, action, q_val = get_optimal_weights(symbol, agent_scores, market_features)

    combined = sum(weights.get(k,0) * agent_scores.get(k,0) for k in weights)
    combined = float(np.clip(combined, -1, 1))
    score    = round(combined, 4)

    if score > 0.2:    interp = f"RL-DQN: bullish weighting (Q={q_val:.3f})"
    elif score < -0.2: interp = f"RL-DQN: bearish weighting (Q={q_val:.3f})"
    else:              interp = f"RL-DQN: neutral/uncertain (Q={q_val:.3f})"

    return {
        "symbol":         symbol,
        "agent":          "rl_dqn",
        "score":          score,
        "interpretation": interp,
        "weights":        {k: round(v,3) for k,v in weights.items()},
        "q_value":        round(q_val, 4),
        "action_idx":     action,
    }


if __name__ == "__main__":
    print("RL DQN Agent test\n")
    print("Training takes ~30s per stock\n")
    for sym in ["RELIANCE", "TCS"]:
        r = get_latest_signal(sym)
        print(f"{sym:15} score={r['score']:+.4f}  {r['interpretation']}")
        print(f"  Optimal weights: {r['weights']}\n")
