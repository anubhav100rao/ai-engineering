"""
TinyGPT: a complete GPT-style decoder-only transformer in pure numpy,
including HAND-WRITTEN BACKPROPAGATION and an Adam optimizer.

This is the "real" artifact of the course: lessons 01-05 explain the pieces,
06 walks through this model's forward pass, 07 trains it on text, and
tests/test_transformer.py verifies the gradients against finite differences.

Architecture (same shape as GPT-2, just tiny):

    ids -> token_embedding + position_embedding
        -> [ LN -> multi-head causal attention -> +residual
             LN -> MLP (ReLU)                  -> +residual ]  x n_layer
        -> final LN -> linear head -> logits over vocab

Everything is float64 for clean gradient checks; speed is not the point.
"""

import numpy as np


# ----------------------------------------------------------------------------
# small differentiable pieces: each has forward(...) -> (out, cache)
# and backward(d_out, cache) -> gradients
# ----------------------------------------------------------------------------

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def linear_forward(x, W, b):
    """x: (..., in), W: (in, out), b: (out,)"""
    return x @ W + b, (x, W)


def linear_backward(dout, cache):
    x, W = cache
    x2d = x.reshape(-1, x.shape[-1])
    d2d = dout.reshape(-1, dout.shape[-1])
    dx = dout @ W.T
    dW = x2d.T @ d2d
    db = d2d.sum(axis=0)
    return dx, dW, db


def layernorm_forward(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def layernorm_backward(dout, cache):
    xhat, rstd, g = cache
    D = xhat.shape[-1]
    dg = (dout * xhat).reshape(-1, D).sum(axis=0)
    db = dout.reshape(-1, D).sum(axis=0)
    dxhat = dout * g
    # standard layernorm gradient (derive it once in your life — it's worth it)
    dx = rstd / D * (
        D * dxhat
        - dxhat.sum(-1, keepdims=True)
        - xhat * (dxhat * xhat).sum(-1, keepdims=True)
    )
    return dx, dg, db


def attention_forward(x, p, prefix, n_head):
    """Multi-head causal self-attention. x: (B, T, D)."""
    B, T, D = x.shape
    hd = D // n_head

    qkv, lin1_cache = linear_forward(x, p[f"{prefix}.Wqkv"], p[f"{prefix}.bqkv"])
    q, k, v = np.split(qkv, 3, axis=-1)                       # each (B, T, D)
    # split heads: (B, T, D) -> (B, n_head, T, hd)
    def heads(t):
        return t.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)
    q, k, v = heads(q), heads(k), heads(v)

    att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)           # (B, nh, T, T)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    att = np.where(mask, -np.inf, att)
    A = softmax(att)                                          # attention weights
    y = A @ v                                                 # (B, nh, T, hd)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, D)              # merge heads
    out, lin2_cache = linear_forward(y, p[f"{prefix}.Wproj"], p[f"{prefix}.bproj"])

    cache = (lin1_cache, lin2_cache, q, k, v, A, y, n_head)
    return out, cache


def attention_backward(dout, cache, grads, prefix):
    lin1_cache, lin2_cache, q, k, v, A, y, n_head = cache
    B, nh, T, hd = q.shape
    D = nh * hd

    dy, dWproj, dbproj = linear_backward(dout, lin2_cache)
    grads[f"{prefix}.Wproj"] += dWproj
    grads[f"{prefix}.bproj"] += dbproj

    dy = dy.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)       # (B, nh, T, hd)
    dA = dy @ v.transpose(0, 1, 3, 2)                         # (B, nh, T, T)
    dv = A.transpose(0, 1, 3, 2) @ dy                         # (B, nh, T, hd)
    # softmax backward per row; masked positions have A == 0, so datt == 0 there
    datt = A * (dA - (dA * A).sum(-1, keepdims=True))
    dq = datt @ k / np.sqrt(hd)
    dk = datt.transpose(0, 1, 3, 2) @ q / np.sqrt(hd)

    def unheads(t):  # (B, nh, T, hd) -> (B, T, D)
        return t.transpose(0, 2, 1, 3).reshape(B, T, D)
    dqkv = np.concatenate([unheads(dq), unheads(dk), unheads(dv)], axis=-1)

    dx, dWqkv, dbqkv = linear_backward(dqkv, lin1_cache)
    grads[f"{prefix}.Wqkv"] += dWqkv
    grads[f"{prefix}.bqkv"] += dbqkv
    return dx


# ----------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------

class TinyGPT:
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4,
                 n_layer=2, seed=0):
        self.vocab_size, self.block_size = vocab_size, block_size
        self.n_embd, self.n_head, self.n_layer = n_embd, n_head, n_layer
        rng = np.random.default_rng(seed)

        def W(shape, std=0.02):
            return rng.normal(0, std, shape)

        p = {
            "wte": W((vocab_size, n_embd)),        # token embedding table
            "wpe": W((block_size, n_embd)),        # learned position table
            "lnf.g": np.ones(n_embd), "lnf.b": np.zeros(n_embd),
            "head.W": W((n_embd, vocab_size)), "head.b": np.zeros(vocab_size),
        }
        for i in range(n_layer):
            b = f"block{i}"
            p[f"{b}.ln1.g"] = np.ones(n_embd)
            p[f"{b}.ln1.b"] = np.zeros(n_embd)
            p[f"{b}.attn.Wqkv"] = W((n_embd, 3 * n_embd))
            p[f"{b}.attn.bqkv"] = np.zeros(3 * n_embd)
            p[f"{b}.attn.Wproj"] = W((n_embd, n_embd))
            p[f"{b}.attn.bproj"] = np.zeros(n_embd)
            p[f"{b}.ln2.g"] = np.ones(n_embd)
            p[f"{b}.ln2.b"] = np.zeros(n_embd)
            p[f"{b}.mlp.W1"] = W((n_embd, 4 * n_embd))
            p[f"{b}.mlp.b1"] = np.zeros(4 * n_embd)
            p[f"{b}.mlp.W2"] = W((4 * n_embd, n_embd))
            p[f"{b}.mlp.b2"] = np.zeros(n_embd)
        self.params = p

    def num_params(self):
        return sum(v.size for v in self.params.values())

    # ---------------- forward ----------------

    def forward(self, idx, targets=None):
        """idx: (B, T) int token ids. Returns (logits, loss, cache).

        If targets (B, T) is given, loss = mean cross-entropy of predicting
        targets[b, t] from idx[b, :t+1].
        """
        p = self.params
        B, T = idx.shape
        assert T <= self.block_size

        x = p["wte"][idx] + p["wpe"][:T]                       # (B, T, D)
        caches = []
        for i in range(self.n_layer):
            b = f"block{i}"
            ln1, c_ln1 = layernorm_forward(x, p[f"{b}.ln1.g"], p[f"{b}.ln1.b"])
            att, c_att = attention_forward(ln1, p, f"{b}.attn", self.n_head)
            x = x + att                                        # residual 1
            ln2, c_ln2 = layernorm_forward(x, p[f"{b}.ln2.g"], p[f"{b}.ln2.b"])
            h, c_fc1 = linear_forward(ln2, p[f"{b}.mlp.W1"], p[f"{b}.mlp.b1"])
            hr = np.maximum(0, h)                              # ReLU
            mlp, c_fc2 = linear_forward(hr, p[f"{b}.mlp.W2"], p[f"{b}.mlp.b2"])
            x = x + mlp                                        # residual 2
            caches.append((c_ln1, c_att, c_ln2, c_fc1, h, c_fc2))

        lnf, c_lnf = layernorm_forward(x, p["lnf.g"], p["lnf.b"])
        logits, c_head = linear_forward(lnf, p["head.W"], p["head.b"])

        loss = None
        probs = None
        if targets is not None:
            probs = softmax(logits)                            # (B, T, V)
            picked = probs[np.arange(B)[:, None], np.arange(T)[None, :], targets]
            loss = -np.log(picked + 1e-12).mean()

        cache = (idx, caches, c_lnf, c_head, probs, targets)
        return logits, loss, cache

    # ---------------- backward ----------------

    def backward(self, cache):
        """Returns grads: dict with the same keys/shapes as self.params."""
        p = self.params
        idx, caches, c_lnf, c_head, probs, targets = cache
        B, T = idx.shape
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        # d(mean CE)/d(logits) = (softmax - onehot) / (B*T)
        dlogits = probs.copy()
        dlogits[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1.0
        dlogits /= B * T

        dlnf, dWh, dbh = linear_backward(dlogits, c_head)
        grads["head.W"] += dWh
        grads["head.b"] += dbh
        dx, dg, db = layernorm_backward(dlnf, c_lnf)
        grads["lnf.g"] += dg
        grads["lnf.b"] += db

        for i in reversed(range(self.n_layer)):
            b = f"block{i}"
            c_ln1, c_att, c_ln2, c_fc1, h_pre, c_fc2 = caches[i]

            # ---- MLP branch: x = x + mlp(ln2(x)) ----
            dmlp = dx                                # gradient into the branch
            dhr, dW2, db2 = linear_backward(dmlp, c_fc2)
            grads[f"{b}.mlp.W2"] += dW2
            grads[f"{b}.mlp.b2"] += db2
            dh = dhr * (h_pre > 0)                   # ReLU gate
            dln2, dW1, db1 = linear_backward(dh, c_fc1)
            grads[f"{b}.mlp.W1"] += dW1
            grads[f"{b}.mlp.b1"] += db1
            dx2, dg2, db2_ = layernorm_backward(dln2, c_ln2)
            grads[f"{b}.ln2.g"] += dg2
            grads[f"{b}.ln2.b"] += db2_
            dx = dx + dx2                            # residual: gradients ADD

            # ---- attention branch: x = x + attn(ln1(x)) ----
            datt = attention_backward(dx, c_att, grads, f"{b}.attn")
            dx1, dg1, db1_ = layernorm_backward(datt, c_ln1)
            grads[f"{b}.ln1.g"] += dg1
            grads[f"{b}.ln1.b"] += db1_
            dx = dx + dx1                            # residual again

        # ---- embeddings ----
        np.add.at(grads["wte"], idx, dx)             # scatter-add per token id
        grads["wpe"][:T] += dx.sum(axis=0)           # summed over the batch
        return grads

    # ---------------- generation ----------------

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 rng=None):
        """idx: (1, T) prompt ids -> extended with sampled tokens."""
        rng = rng or np.random.default_rng()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.block_size:]
            logits, _, _ = self.forward(ctx)
            logits = logits[0, -1] / temperature       # last position only
            if top_k is not None:
                cutoff = np.sort(logits)[-top_k]
                logits = np.where(logits < cutoff, -np.inf, logits)
            probs = softmax(logits)
            nxt = rng.choice(len(probs), p=probs)
            idx = np.concatenate([idx, [[nxt]]], axis=1)
        return idx


class Adam:
    """The optimizer GPTs are actually trained with, in ~15 lines."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
