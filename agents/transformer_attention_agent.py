"""
transformer_attention_agent.py
-------------------------------
Transformer Attention Agent — self-attention over price sequences.

Why attention over LSTM?
  LSTM processes sequences left-to-right and can forget distant patterns.
  Attention can directly connect any two time steps regardless of distance.
  "What happened 45 days ago that looks like today?" — attention finds it.
  LSTM struggles to look that far back. Attention does it in one step.

Architecture (implemented from scratch in numpy):
  Input:  60-day sequence × 8 features
  ↓
  Linear projection to d_model=32 dimensions
  ↓
  Multi-head self-attention (4 heads, d_k=8)
    Q = query matrix  (what am I looking for?)
    K = key matrix    (what do I contain?)
    V = value matrix  (what do I output?)
    Attention(Q,K,V) = softmax(QK^T / √d_k) × V
  ↓
  Mean pool over sequence
  ↓
  Feed-forward: 32 → 16 → 1 → sigmoid
  ↓
  Output: probability price goes up

Training: same random hill-climbing as LSTM (no backprop needed)
"""

import numpy as np
import pandas as pd
import sqlite3
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/nifty50.db"


class MultiHeadAttention:
    """
    Multi-head self-attention implemented in numpy.

    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

    Multiple heads learn different types of relationships:
      Head 1 might focus on short-term momentum patterns
      Head 2 might focus on volatility clustering
      Head 3 might focus on seasonal patterns
      Head 4 might focus on trend reversals
    """

    def __init__(self, d_model=32, n_heads=4, seed=42):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)

        # Weight matrices for Q, K, V projections per head
        self.W_q = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_k = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_v = rng.normal(0, scale, (n_heads, d_model, self.d_k))

        # Output projection
        self.W_o = rng.normal(0, scale, (n_heads * self.d_k, d_model))

    def softmax(self, x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-8)

    def forward(self, X):
        """
        X: shape (seq_len, d_model)
        Returns: shape (seq_len, d_model)
        """
        seq_len = X.shape[0]
        head_outputs = []

        for h in range(self.n_heads):
            Q = X @ self.W_q[h]   # (seq_len, d_k)
            K = X @ self.W_k[h]   # (seq_len, d_k)
            V = X @ self.W_v[h]   # (seq_len, d_k)

            # Scaled dot-product attention
            scores = Q @ K.T / np.sqrt(self.d_k)  # (seq_len, seq_len)
            attn   = self.softmax(scores)           # (seq_len, seq_len)
            head_out = attn @ V                     # (seq_len, d_k)
            head_outputs.append(head_out)

        # Concatenate heads
        concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)
        return concat @ self.W_o                         # (seq_len, d_model)


class TransformerBlock:
    """
    One transformer block:
      Multi-head attention → Add & Norm → Feed-forward → Add & Norm
    """

    def __init__(self, d_model=32, n_heads=4, d_ff=64, seed=42):
        self.attention = MultiHeadAttention(d_model, n_heads, seed)
        rng = np.random.default_rng(seed + 10)
        scale = np.sqrt(2.0 / d_model)
        self.W1 = rng.normal(0, scale, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, scale, (d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def layer_norm(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std  = x.std(axis=-1, keepdims=True) + 1e-8
        return (x - mean) / std

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        # Self-attention with residual
        attn_out = self.attention.forward(X)
        X = self.layer_norm(X + attn_out)

        # Feed-forward with residual
        ff_out = self.relu(X @ self.W1 + self.b1) @ self.W2 + self.b2
        X = self.layer_norm(X + ff_out)

        return X


class PriceTransformer:
    """
    Full transformer model for price prediction.
    Input projection → Transformer blocks → Mean pool → Output
    """

    def __init__(self, input_size, d_model=32, n_heads=4, n_layers=2, seed=42):
        rng = np.random.default_rng(seed)

        # Input projection
        self.W_in = rng.normal(0, 0.1, (input_size, d_model))
        self.b_in = np.zeros(d_model)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_model*2, seed+i)
            for i in range(n_layers)
        ]

        # Output head
        self.W_out = rng.normal(0, 0.1, (d_model, 1))
        self.b_out = np.zeros(1)

    def forward(self, X_seq):
        """
        X_seq: (seq_len, input_size)
        Returns: probability 0-1
        """
        # Project input
        X = X_seq @ self.W_in + self.b_in  # (seq_len, d_model)

        # Add positional encoding (simple sinusoidal)
        seq_len, d_model = X.shape
        pos = np.arange(seq_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000) / d_model))
        pe  = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d_model//2])
        X = X + pe * 0.1

        # Transformer blocks
        for block in self.blocks:
            X = block.forward(X)

        # Mean pool over sequence
        pooled = X.mean(axis=0)  # (d_model,)

        # Output
        logit = float((pooled @ self.W_out + self.b_out)[0])
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500))))

    def fit(self, X_sequences, y, n_iter=150, lr=0.02):
        """Train using random perturbation (gradient-free)."""
        best_acc  = self._evaluate(X_sequences, y)
        best_Wout = self.W_out.copy()
        best_bout = self.b_out.copy()

        rng = np.random.default_rng(99)

        for i in range(n_iter):
            dW = rng.normal(0, lr, self.W_out.shape)
            db = rng.normal(0, lr, self.b_out.shape)
            self.W_out += dW
            self.b_out += db

            acc = self._evaluate(X_sequences, y)
            if acc > best_acc:
                best_acc  = acc
                best_Wout = self.W_out.copy()
                best_bout = self.b_out.copy()
            else:
                self.W_out = best_Wout.copy()
                self.b_out = best_bout.copy()

            if i % 50 == 49:
                lr *= 0.8

        self.W_out = best_Wout
        self.b_out = best_bout
        return best_acc

    def _evaluate(self, X_sequences, y):
        correct = sum(
            1 for i, seq in enumerate(X_sequences)
            if (1 if self.forward(seq) > 0.5 else 0) == y[i]
        )
        return correct / len(y) if y else 0


def build_features(df, seq_len=60):
    """Same feature engineering as LSTM agent."""
    close   = df["Close"]
    volume  = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)
    returns = close.pct_change().fillna(0)

    feat = pd.DataFrame(index=df.index)
    feat["ret_1"]     = returns
    feat["ret_5"]     = close.pct_change(5).fillna(0)
    feat["ret_10"]    = close.pct_change(10).fillna(0)
    feat["vol_10"]    = returns.rolling(10).std().fillna(returns.std())
    feat["vol_ratio"] = (volume / volume.rolling(20).mean()).fillna(1)

    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    feat["rsi"]      = ((100 - 100/(1+rs))/100).fillna(0.5)
    feat["ma20_gap"] = ((close - close.rolling(20).mean()) / close.rolling(20).mean()).fillna(0)
    feat["ma50_gap"] = ((close - close.rolling(50).mean()) / close.rolling(50).mean()).fillna(0)

    for col in feat.columns:
        std = feat[col].std()
        if std > 0:
            feat[col] = (feat[col] - feat[col].mean()) / std
    feat = feat.fillna(0)

    target = (close.shift(-1) > close).astype(int)
    X_seqs, y_labels, dates = [], [], []

    for i in range(seq_len, len(feat) - 1):
        X_seqs.append(feat.values[i-seq_len:i])
        y_labels.append(int(target.iloc[i]))
        dates.append(df.index[i])

    return X_seqs, y_labels, dates


def walk_forward_transformer(df, seq_len=60, train_window=400, retrain_every=60):
    """Walk-forward training — same approach as LSTM."""
    X_seqs, y_labels, dates = build_features(df, seq_len)

    if len(X_seqs) < train_window + 10:
        return pd.Series(dtype=float)

    input_size = X_seqs[0].shape[1]
    scores     = {}
    model      = None
    last_train = -retrain_every

    for i in range(train_window, len(X_seqs)):
        if i - last_train >= retrain_every or model is None:
            train_start = max(0, i - train_window)
            X_train     = X_seqs[train_start:i]
            y_train     = y_labels[train_start:i]
            model = PriceTransformer(input_size, d_model=32, n_heads=4, n_layers=2)
            model.fit(X_train, y_train, n_iter=100, lr=0.02)
            last_train = i

        prob  = model.forward(X_seqs[i])
        score = (prob - 0.5) * 2
        scores[dates[i]] = round(score, 4)

    return pd.Series(scores)


def get_latest_signal(symbol):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(
        f"SELECT Date, Close, Volume FROM prices WHERE Symbol='{symbol}' ORDER BY Date",
        conn, index_col="Date"
    )
    conn.close()

    if len(df) < 300:
        return {"symbol":symbol,"agent":"transformer","score":0.0,
                "interpretation":"insufficient data"}

    print(f"  Training Transformer for {symbol} (~45s)...")
    scores = walk_forward_transformer(df, seq_len=60, train_window=400, retrain_every=60)

    if scores.empty:
        return {"symbol":symbol,"agent":"transformer","score":0.0,
                "interpretation":"training failed"}

    score = round(float(scores.iloc[-1]), 4)

    if score > 0.3:    interp = "Transformer: strong upward attention pattern"
    elif score > 0.1:  interp = "Transformer: mild upward bias"
    elif score < -0.3: interp = "Transformer: strong downward attention pattern"
    elif score < -0.1: interp = "Transformer: mild downward bias"
    else:              interp = "Transformer: no clear attention pattern"

    return {"symbol":symbol,"agent":"transformer","score":score,
            "interpretation":interp,"date":str(scores.index[-1])}


if __name__ == "__main__":
    print("Transformer Attention Agent test\n")
    print("Note: ~45s per stock\n")
    for sym in ["RELIANCE", "TCS"]:
        r = get_latest_signal(sym)
        print(f"{sym:15} score={r['score']:+.4f}  {r['interpretation']}\n")
