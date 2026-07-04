"""
LESSON 4: Multi-head attention
===============================

One attention pattern per layer is limiting: maybe token i should attend to
its subject AND to the previous word AND to matching quotes — different
relationships at once. Multi-head attention runs h independent attention
operations ("heads") in parallel, each in a smaller subspace, then
concatenates the results.

With d_model = 64 and n_heads = 4:

  x (T, 64)
    -> project to Q, K, V, each (T, 64)
    -> split each into 4 heads of size 16:  (4, T, 16)
    -> scaled dot-product attention independently per head -> (4, T, 16)
    -> concatenate heads back -> (T, 64)
    -> final output projection W_o -> (T, 64)

Same total compute as one big head, but 4 different attention patterns.

Note the trick: we don't create separate weight matrices per head. One big
(d_model, d_model) projection is computed, then RESHAPED into heads. That's
also exactly how real implementations (PyTorch, GPT) do it.

Run me:  python3 04_multi_head_attention.py
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int, seed: int = 0):
        assert d_model % n_heads == 0, "d_model must divide evenly into heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        rng = np.random.default_rng(seed)
        s = 1 / np.sqrt(d_model)  # keep activations at a sane scale
        self.W_q = rng.normal(0, s, (d_model, d_model))
        self.W_k = rng.normal(0, s, (d_model, d_model))
        self.W_v = rng.normal(0, s, (d_model, d_model))
        self.W_o = rng.normal(0, s, (d_model, d_model))

    def __call__(self, x: np.ndarray, causal: bool = True):
        """x: (T, d_model) -> (output (T, d_model), weights (n_heads, T, T))"""
        T, D = x.shape
        H, d_h = self.n_heads, self.d_head

        # 1) One big projection each, then split into heads:
        #    (T, D) -> (T, H, d_h) -> (H, T, d_h)
        Q = (x @ self.W_q).reshape(T, H, d_h).transpose(1, 0, 2)
        K = (x @ self.W_k).reshape(T, H, d_h).transpose(1, 0, 2)
        V = (x @ self.W_v).reshape(T, H, d_h).transpose(1, 0, 2)

        # 2) Attention per head, all at once via batched matmul.
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_h)   # (H, T, T)
        if causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores = np.where(mask, -np.inf, scores)
        weights = softmax(scores, axis=-1)                  # (H, T, T)
        heads = weights @ V                                 # (H, T, d_h)

        # 3) Concatenate heads and apply the output projection.
        concat = heads.transpose(1, 0, 2).reshape(T, D)     # (T, D)
        return concat @ self.W_o, weights


def demo():
    np.set_printoptions(precision=2, suppress=True)
    rng = np.random.default_rng(1)

    T, d_model, n_heads = 6, 32, 4
    x = rng.normal(size=(T, d_model))

    mha = MultiHeadAttention(d_model, n_heads)
    out, weights = mha(x, causal=True)

    print(f"input  x       : {x.shape}")
    print(f"output         : {out.shape}   (same shape — attention is shape-preserving)")
    print(f"attention maps : {weights.shape} = (n_heads, T, T)\n")

    # Each head learned (well, here: randomly initialized) a DIFFERENT pattern.
    print("Each head attends differently. Row 5's weights per head:")
    for h in range(n_heads):
        print(f"  head {h}: {weights[h, 5]}")
    print("\nAll rows are valid probability distributions, futures masked:")
    assert np.allclose(weights.sum(-1), 1.0)
    assert np.allclose(np.triu(weights, k=1), 0.0)
    print("  weights.sum(-1) == 1  and  upper triangle == 0  ✓")

    # Show that head-splitting really is just a reshape of one projection.
    Q_big = x @ mha.W_q                       # (T, 32)
    Q_head0_manual = Q_big[:, : mha.d_head]   # first 8 dims = head 0
    Q_heads = Q_big.reshape(T, n_heads, mha.d_head).transpose(1, 0, 2)
    assert np.allclose(Q_head0_manual, Q_heads[0])
    print("\nHead 0's query is literally columns 0..7 of the big Q projection —")
    print("multi-head = one matmul + a reshape. No extra math.")


if __name__ == "__main__":
    demo()
