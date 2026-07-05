"""
LESSON 3: Scaled dot-product attention — the heart of the transformer
======================================================================

Big idea: each token builds its output as a WEIGHTED AVERAGE of every
token's information, where the weights are computed from content.
"Attention" = each token deciding which other tokens to copy from.

Every token vector is projected three ways:

    Query (Q) : "what am I looking for?"
    Key   (K) : "what do I contain?"        (matched against queries)
    Value (V) : "what do I hand over if someone attends to me?"

The formula from "Attention Is All You Need":

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Reading it inside-out:
  1. Q K^T          — every query dotted with every key: a (T, T) score
                      matrix. scores[i, j] = how relevant token j is to token i.
  2. / sqrt(d_k)    — dot products grow with dimension; scaling keeps the
                      softmax from saturating (all weight on one token).
  3. softmax(...)   — each row becomes a probability distribution: "how much
                      should token i listen to each token j?"
  4. ... V          — mix the value vectors using those weights.

CAUSAL MASKING: a language model predicts the future, so token i must not
peek at tokens j > i. We set those scores to -inf before the softmax, which
turns them into exactly 0 weight after it.

(This file re-derives attention standalone for teaching; the batched,
multi-head, differentiable version everything trains with is in model.py.)

Run me:  python3 03_attention.py
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtracting the max changes nothing)."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Single-head attention.

    Args:
        Q: (T, d_k) queries.  K: (T, d_k) keys.  V: (T, d_v) values.
        causal: if True, position i may only attend to positions <= i.

    Returns:
        (output (T, d_v), attention weights (T, T)); each weight row is a
        probability distribution over the positions attended to.
    """
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K, V must be 2-D (T, d) arrays")
    if Q.shape[1] != K.shape[1]:
        raise ValueError(f"query dim {Q.shape[1]} != key dim {K.shape[1]}")
    if K.shape[0] != V.shape[0]:
        raise ValueError(f"K has {K.shape[0]} positions but V has {V.shape[0]}")

    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)              # (T, T)
    if causal:
        T = scores.shape[0]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # True above diagonal
        scores = np.where(mask, -np.inf, scores)
    weights = softmax(scores, axis=-1)           # rows sum to 1
    return weights @ V, weights


def demo() -> None:
    np.set_printoptions(precision=3, suppress=True)

    # A tiny hand-built example: 4 tokens, d_k = 4.
    # Pretend the sentence is:  "the  cat  sat  down"
    tokens = ["the", "cat", "sat", "down"]

    # We hand-craft Q and K so the story is readable:
    # give "sat" (a verb) a query that matches "cat"'s key (its subject).
    Q = np.array([
        [0.1, 0.0, 0.0, 0.0],   # "the"  — barely looking for anything
        [0.0, 1.0, 0.0, 0.0],   # "cat"  — looking for determiners
        [2.0, 0.0, 2.0, 0.0],   # "sat"  — strongly looking for its subject
        [0.0, 0.0, 1.0, 0.0],   # "down" — looking for the verb
    ])
    K = np.array([
        [0.0, 1.0, 0.0, 0.0],   # "the"  — I am a determiner
        [1.0, 0.0, 1.0, 0.0],   # "cat"  — I am a noun/subject
        [0.0, 0.0, 1.0, 1.0],   # "sat"  — I am a verb
        [0.0, 0.0, 0.0, 1.0],   # "down" — I am a particle
    ])
    # Values: what each token contributes when attended to. Use one-hot so
    # the output plainly shows "whose information got mixed in".
    V = np.eye(4)

    print("STEP 1 — raw scores Q K^T (before scaling):")
    print(Q @ K.T)

    out, W = scaled_dot_product_attention(Q, K, V)
    print("\nSTEP 2+3 — attention weights, softmax(QK^T/sqrt(d_k)), rows sum to 1:")
    header = "        " + "".join(f"{t:>7}" for t in tokens)
    print(header)
    for t, row in zip(tokens, W):
        print(f"{t:>7} " + "".join(f"{w:7.3f}" for w in row))
    print("\nRead row 'sat': it puts most weight on 'cat' — the verb found "
          "its subject.")

    print("\nSTEP 4 — output = weights @ V (with V = identity, output IS the "
          "weights):")
    print(out)

    # --- causal masking ----------------------------------------------------
    out_c, W_c = scaled_dot_product_attention(Q, K, V, causal=True)
    print("\nWith a CAUSAL mask (upper triangle zeroed):")
    print(header)
    for t, row in zip(tokens, W_c):
        print(f"{t:>7} " + "".join(f"{w:7.3f}" for w in row))
    print("Each token now only attends to itself and the past — required "
          "for next-token prediction.")

    # Sanity checks (the tests file runs these too).
    assert np.allclose(W.sum(axis=-1), 1.0)
    assert np.allclose(np.triu(W_c, k=1), 0.0), "future weights must be zero"

    # --- why the sqrt(d_k) scaling matters ----------------------------------
    rng = np.random.default_rng(0)
    d = 256
    q, k = rng.normal(size=(50, d)), rng.normal(size=(50, d))
    raw = (q @ k.T).std()
    scaled = (q @ k.T / np.sqrt(d)).std()
    print(f"\nWhy scale? With d_k={d}, random dot products have std ~{raw:.1f} "
          f"(softmax saturates);\nafter /sqrt(d_k) the std is ~{scaled:.2f} "
          "(healthy gradients).")


if __name__ == "__main__":
    demo()
