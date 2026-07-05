"""TinyGPT: a GPT-style decoder-only transformer in pure numpy.

This is the "real" artifact of the course: lessons 01-05 explain the pieces,
06 walks through this model's forward pass, 07 trains it on text, and
tests/test_transformer.py verifies the hand-written gradients against finite
differences.

Architecture (identical in shape to GPT-2, just tiny):

    ids -> token_embedding + position_embedding
        -> [ LN -> multi-head causal attention -> +residual
             LN -> MLP (ReLU)                  -> +residual ]  x n_layer
        -> final LN -> linear head -> logits over vocab

What "production grade" means here:
  * ``GPTConfig`` dataclass with validation — one place to define a model.
  * Every forward/backward primitive is a small, documented, testable pair.
  * Strict input validation with actionable error messages.
  * Checkpointing (``save``/``load``) that round-trips config + weights + metadata.
  * AdamW-style optimizer (decoupled weight decay) and gradient clipping.

Everything runs in float64: gradient checks are then exact to ~1e-8, and
speed is explicitly not the point of this codebase.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPTConfig:
    """Hyperparameters that define a TinyGPT instance."""

    vocab_size: int          # number of distinct token ids
    block_size: int          # maximum context length T
    n_embd: int = 64         # model width D
    n_head: int = 4          # attention heads (must divide n_embd)
    n_layer: int = 2         # transformer blocks
    seed: int = 0            # rng seed for weight initialization

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by "
                f"n_head ({self.n_head}) so heads get equal slices"
            )


# ---------------------------------------------------------------------------
# differentiable primitives: forward(...) -> (out, cache),
# backward(d_out, cache) -> input/parameter gradients
# ---------------------------------------------------------------------------

def softmax(x: Array, axis: int = -1) -> Array:
    """Numerically stable softmax (max-subtraction changes nothing)."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def linear_forward(x: Array, W: Array, b: Array) -> tuple[Array, tuple]:
    """Affine map. x: (..., d_in), W: (d_in, d_out), b: (d_out,)."""
    return x @ W + b, (x, W)


def linear_backward(dout: Array, cache: tuple) -> tuple[Array, Array, Array]:
    """Returns (dx, dW, db) for :func:`linear_forward`."""
    x, W = cache
    x2d = x.reshape(-1, x.shape[-1])
    d2d = dout.reshape(-1, dout.shape[-1])
    return dout @ W.T, x2d.T @ d2d, d2d.sum(axis=0)


def layernorm_forward(
    x: Array, g: Array, b: Array, eps: float = 1e-5
) -> tuple[Array, tuple]:
    """Normalize each token vector to mean 0 / var 1, then scale and shift."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)


def layernorm_backward(dout: Array, cache: tuple) -> tuple[Array, Array, Array]:
    """Returns (dx, dg, db) for :func:`layernorm_forward`."""
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


def attention_forward(
    x: Array, p: dict[str, Array], prefix: str, n_head: int
) -> tuple[Array, tuple]:
    """Multi-head causal self-attention. x: (B, T, D) -> (B, T, D)."""
    B, T, D = x.shape
    hd = D // n_head

    qkv, lin1_cache = linear_forward(x, p[f"{prefix}.Wqkv"], p[f"{prefix}.bqkv"])
    q, k, v = np.split(qkv, 3, axis=-1)                       # each (B, T, D)

    def heads(t: Array) -> Array:  # (B, T, D) -> (B, n_head, T, hd)
        return t.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)

    q, k, v = heads(q), heads(k), heads(v)

    att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)           # (B, nh, T, T)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)          # True = future
    att = np.where(mask, -np.inf, att)
    A = softmax(att)                                          # attention weights
    y = A @ v                                                 # (B, nh, T, hd)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, D)              # merge heads
    out, lin2_cache = linear_forward(y, p[f"{prefix}.Wproj"], p[f"{prefix}.bproj"])

    return out, (lin1_cache, lin2_cache, q, k, v, A, y, n_head)


def attention_backward(
    dout: Array, cache: tuple, grads: dict[str, Array], prefix: str
) -> Array:
    """Backward pass for :func:`attention_forward`; accumulates into ``grads``."""
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

    def unheads(t: Array) -> Array:  # (B, nh, T, hd) -> (B, T, D)
        return t.transpose(0, 2, 1, 3).reshape(B, T, D)

    dqkv = np.concatenate([unheads(dq), unheads(dk), unheads(dv)], axis=-1)
    dx, dWqkv, dbqkv = linear_backward(dqkv, lin1_cache)
    grads[f"{prefix}.Wqkv"] += dWqkv
    grads[f"{prefix}.bqkv"] += dbqkv
    return dx


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class TinyGPT:
    """Decoder-only transformer with hand-written forward and backward passes.

    Parameters live in ``self.params``, a flat ``{name: ndarray}`` dict —
    the numpy equivalent of a PyTorch ``state_dict``.
    """

    def __init__(self, config: GPTConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        V, T, D = config.vocab_size, config.block_size, config.n_embd

        def W(shape: tuple[int, ...], std: float = 0.02) -> Array:
            return rng.normal(0.0, std, shape)

        p: dict[str, Array] = {
            "wte": W((V, D)),                       # token embedding table
            "wpe": W((T, D)),                       # learned position table
            "lnf.g": np.ones(D), "lnf.b": np.zeros(D),
            "head.W": W((D, V)), "head.b": np.zeros(V),
        }
        for i in range(config.n_layer):
            b = f"block{i}"
            p[f"{b}.ln1.g"] = np.ones(D)
            p[f"{b}.ln1.b"] = np.zeros(D)
            p[f"{b}.attn.Wqkv"] = W((D, 3 * D))
            p[f"{b}.attn.bqkv"] = np.zeros(3 * D)
            p[f"{b}.attn.Wproj"] = W((D, D))
            p[f"{b}.attn.bproj"] = np.zeros(D)
            p[f"{b}.ln2.g"] = np.ones(D)
            p[f"{b}.ln2.b"] = np.zeros(D)
            p[f"{b}.mlp.W1"] = W((D, 4 * D))
            p[f"{b}.mlp.b1"] = np.zeros(4 * D)
            p[f"{b}.mlp.W2"] = W((4 * D, D))
            p[f"{b}.mlp.b2"] = np.zeros(D)
        self.params = p

    def num_params(self) -> int:
        return sum(v.size for v in self.params.values())

    # ---------------- forward ----------------

    def _validate_ids(self, idx: Array, name: str) -> None:
        if not isinstance(idx, np.ndarray) or idx.ndim != 2:
            raise ValueError(f"{name} must be a 2-D array of shape (B, T), "
                             f"got {getattr(idx, 'shape', type(idx))}")
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"{name} must contain integer token ids, "
                            f"got dtype {idx.dtype}")
        if idx.shape[1] > self.config.block_size:
            raise ValueError(
                f"sequence length {idx.shape[1]} exceeds block_size "
                f"{self.config.block_size}; shorten the input or build the "
                f"model with a larger block_size")
        if idx.size and (idx.min() < 0 or idx.max() >= self.config.vocab_size):
            raise ValueError(
                f"{name} contains ids outside [0, {self.config.vocab_size}); "
                f"min={idx.min()}, max={idx.max()}")

    def forward(
        self, idx: Array, targets: Array | None = None
    ) -> tuple[Array, float | None, tuple]:
        """Compute logits (and loss, if targets are given).

        Args:
            idx: (B, T) integer token ids, T <= block_size.
            targets: optional (B, T) ids; targets[b, t] is the token that
                should follow idx[b, :t+1].

        Returns:
            (logits (B, T, vocab_size), mean cross-entropy loss or None,
             cache for :meth:`backward`).
        """
        self._validate_ids(idx, "idx")
        if targets is not None:
            self._validate_ids(targets, "targets")
            if targets.shape != idx.shape:
                raise ValueError(f"targets shape {targets.shape} must match "
                                 f"idx shape {idx.shape}")

        p = self.params
        B, T = idx.shape

        x = p["wte"][idx] + p["wpe"][:T]                       # (B, T, D)
        caches = []
        for i in range(self.config.n_layer):
            b = f"block{i}"
            ln1, c_ln1 = layernorm_forward(x, p[f"{b}.ln1.g"], p[f"{b}.ln1.b"])
            att, c_att = attention_forward(ln1, p, f"{b}.attn", self.config.n_head)
            x = x + att                                        # residual 1
            ln2, c_ln2 = layernorm_forward(x, p[f"{b}.ln2.g"], p[f"{b}.ln2.b"])
            h, c_fc1 = linear_forward(ln2, p[f"{b}.mlp.W1"], p[f"{b}.mlp.b1"])
            hr = np.maximum(0, h)                              # ReLU
            mlp, c_fc2 = linear_forward(hr, p[f"{b}.mlp.W2"], p[f"{b}.mlp.b2"])
            x = x + mlp                                        # residual 2
            caches.append((c_ln1, c_att, c_ln2, c_fc1, h, c_fc2))

        lnf, c_lnf = layernorm_forward(x, p["lnf.g"], p["lnf.b"])
        logits, c_head = linear_forward(lnf, p["head.W"], p["head.b"])

        loss: float | None = None
        probs: Array | None = None
        if targets is not None:
            probs = softmax(logits)                            # (B, T, V)
            picked = probs[np.arange(B)[:, None], np.arange(T)[None, :], targets]
            loss = float(-np.log(picked + 1e-12).mean())

        cache = (idx, caches, c_lnf, c_head, probs, targets)
        return logits, loss, cache

    # ---------------- backward ----------------

    def backward(self, cache: tuple) -> dict[str, Array]:
        """Backpropagate through the cached forward pass.

        Requires the forward pass to have been called WITH targets.
        Returns a gradient dict with the same keys/shapes as ``self.params``.
        """
        idx, caches, c_lnf, c_head, probs, targets = cache
        if probs is None:
            raise ValueError("backward() needs a cache from forward(idx, targets) "
                             "— no loss was computed, so there is nothing to "
                             "differentiate")

        p = self.params
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

        for i in reversed(range(self.config.n_layer)):
            b = f"block{i}"
            c_ln1, c_att, c_ln2, c_fc1, h_pre, c_fc2 = caches[i]

            # ---- MLP branch: x = x + mlp(ln2(x)) ----
            dhr, dW2, db2 = linear_backward(dx, c_fc2)
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

    def generate(
        self,
        idx: Array,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> Array:
        """Autoregressively extend (B, T) prompt ids by max_new_tokens.

        The context is cropped to the last ``block_size`` tokens each step.
        ``temperature`` < 1 sharpens the distribution, > 1 flattens it;
        ``top_k`` restricts sampling to the k most likely tokens.
        """
        self._validate_ids(idx[:, -self.config.block_size:], "idx")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError(f"top_k must be in [1, {self.config.vocab_size}]")

        rng = rng or np.random.default_rng()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.config.block_size:]
            logits, _, _ = self.forward(ctx)
            logits = logits[:, -1, :] / temperature    # (B, V), last position
            if top_k is not None:
                cutoff = np.sort(logits, axis=-1)[:, [-top_k]]
                logits = np.where(logits < cutoff, -np.inf, logits)
            probs = softmax(logits)
            nxt = np.array([[rng.choice(p.size, p=p)] for p in probs])
            idx = np.concatenate([idx, nxt], axis=1)
        return idx

    # ---------------- checkpointing ----------------

    def save(self, path: str | Path, meta: dict[str, Any] | None = None) -> Path:
        """Write config + weights + optional JSON metadata to a .npz file."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            __config__=json.dumps(asdict(self.config)),
            __meta__=json.dumps(meta or {}),
            **self.params,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> tuple["TinyGPT", dict[str, Any]]:
        """Rebuild a model (and its metadata) from :meth:`save` output."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no checkpoint at {path}")
        with np.load(path) as d:
            if "__config__" not in d:
                raise ValueError(f"{path} is not a TinyGPT checkpoint "
                                 "(missing __config__ entry)")
            config = GPTConfig(**json.loads(str(d["__config__"])))
            meta: dict[str, Any] = json.loads(str(d["__meta__"]))
            model = cls(config)
            for k in model.params:
                if k not in d:
                    raise ValueError(f"checkpoint {path} is missing "
                                     f"parameter {k!r}")
                if d[k].shape != model.params[k].shape:
                    raise ValueError(
                        f"checkpoint parameter {k!r} has shape {d[k].shape}, "
                        f"expected {model.params[k].shape}")
                model.params[k] = d[k]
        return model, meta


# ---------------------------------------------------------------------------
# training utilities
# ---------------------------------------------------------------------------

def clip_grad_norm(grads: dict[str, Array], max_norm: float) -> float:
    """Scale gradients in place so their global L2 norm is <= max_norm.

    Returns the pre-clip norm — worth logging: a spiking gradient norm is
    the classic early warning of training instability.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be > 0")
    total = float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for g in grads.values():
            g *= scale
    return total


class Adam:
    """AdamW-style optimizer: Adam with decoupled weight decay.

    Following standard practice, weight decay is applied only to matrices
    (ndim >= 2) — biases, layernorm gains, and other vectors are exempt.
    """

    def __init__(
        self,
        params: dict[str, Array],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be > 0")
        self.lr, (self.b1, self.b2) = lr, betas
        self.eps, self.weight_decay = eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, Array], grads: dict[str, Array]) -> None:
        self.t += 1
        for k, P in params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            if self.weight_decay and P.ndim >= 2:
                P -= self.lr * self.weight_decay * P
            P -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
