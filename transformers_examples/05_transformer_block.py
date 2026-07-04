"""
LESSON 5: The transformer block — attention + FFN + the plumbing
=================================================================

Attention alone isn't a transformer. A transformer BLOCK (GPT-style,
"pre-norm") is:

    x = x + Attention(LayerNorm(x))     # tokens TALK to each other
    x = x + FeedForward(LayerNorm(x))   # each token THINKS by itself

Three new ingredients:

1) RESIDUAL CONNECTIONS ("x + ..."): the block computes a small UPDATE to x
   instead of replacing it. Gradients flow straight through the "+", which
   is what makes 100-layer transformers trainable.

2) LAYERNORM: re-centers and re-scales each token's vector to mean 0,
   variance 1 (then applies a learned gain and bias). Keeps activations in
   a healthy range no matter how deep the stack gets.

3) FEED-FORWARD NETWORK (FFN/MLP): a two-layer MLP applied to EACH TOKEN
   INDEPENDENTLY, usually expanding to 4*d_model in the middle:
       FFN(x) = ReLU(x W1 + b1) W2 + b2      # GPT uses GELU; ReLU is simpler
   Attention moves information BETWEEN positions; the FFN transforms it AT
   each position. Most of a transformer's parameters live here.

Stack N of these blocks and you have a transformer.

Run me:  python3 05_transformer_block.py
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, gain, bias, eps=1e-5):
    """Normalize each row (token vector) to mean 0 / var 1, then scale+shift."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


class TransformerBlock:
    def __init__(self, d_model, n_heads, seed=0):
        rng = np.random.default_rng(seed)
        s = 1 / np.sqrt(d_model)
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        # attention weights
        self.W_q, self.W_k, self.W_v, self.W_o = (
            rng.normal(0, s, (d_model, d_model)) for _ in range(4))
        # feed-forward weights (expand 4x, contract back)
        self.W1 = rng.normal(0, s, (d_model, 4 * d_model))
        self.b1 = np.zeros(4 * d_model)
        self.W2 = rng.normal(0, 1 / np.sqrt(4 * d_model), (4 * d_model, d_model))
        self.b2 = np.zeros(d_model)
        # layernorm parameters (learned; init = identity transform)
        self.ln1_g, self.ln1_b = np.ones(d_model), np.zeros(d_model)
        self.ln2_g, self.ln2_b = np.ones(d_model), np.zeros(d_model)

    def attention(self, x):
        T, D = x.shape
        H, d_h = self.n_heads, self.d_head
        Q = (x @ self.W_q).reshape(T, H, d_h).transpose(1, 0, 2)
        K = (x @ self.W_k).reshape(T, H, d_h).transpose(1, 0, 2)
        V = (x @ self.W_v).reshape(T, H, d_h).transpose(1, 0, 2)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_h)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
        heads = softmax(scores) @ V
        return heads.transpose(1, 0, 2).reshape(T, D) @ self.W_o

    def ffn(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)   # ReLU
        return h @ self.W2 + self.b2

    def __call__(self, x):
        # Pre-norm residual wiring — read this as "x plus a correction".
        x = x + self.attention(layer_norm(x, self.ln1_g, self.ln1_b))
        x = x + self.ffn(layer_norm(x, self.ln2_g, self.ln2_b))
        return x


def demo():
    np.set_printoptions(precision=3, suppress=True)
    rng = np.random.default_rng(7)
    T, d_model = 8, 32

    x = rng.normal(size=(T, d_model))
    block = TransformerBlock(d_model, n_heads=4)
    y = block(x)
    print(f"block: {x.shape} -> {y.shape}  (shape-preserving, so blocks stack)")

    # LayerNorm really does normalize each token vector:
    ln = layer_norm(x, np.ones(d_model), np.zeros(d_model))
    print(f"\nlayer_norm per-token mean ~ {ln.mean(-1).max():.1e}, "
          f"std ~ {ln.std(-1).mean():.3f}  (0 and 1 as promised)")

    # The residual path really matters. Stack 20 blocks with and without it:
    x_res, x_plain = x.copy(), x.copy()
    blocks = [TransformerBlock(d_model, 4, seed=i) for i in range(20)]
    for b in blocks:
        x_res = b(x_res)                                     # with residuals
        x_plain = b.ffn(layer_norm(x_plain, b.ln2_g, b.ln2_b))  # replace, don't add
    print(f"\nAfter 20 stacked blocks:")
    print(f"  with residuals    : activation std = {x_res.std():8.3f}  (stable)")
    print(f"  without residuals : activation std = {x_plain.std():8.3f}  (signal dying/exploding)")

    # FFN is per-token: shuffling token order shuffles outputs identically.
    perm = rng.permutation(T)
    assert np.allclose(block.ffn(x)[perm], block.ffn(x[perm]))
    print("\nFFN(x)[perm] == FFN(x[perm]) — the FFN never mixes positions;")
    print("only attention moves information between tokens.")


if __name__ == "__main__":
    demo()
