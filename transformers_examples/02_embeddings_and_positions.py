"""
LESSON 2: Embeddings and positional encoding
=============================================

Token ids (lesson 1) are arbitrary labels — id 7 is not "closer" to id 8
than to id 40. The model needs each token as a VECTOR it can do math on.

1) TOKEN EMBEDDING: a lookup table of shape (vocab_size, d_model).
   Token id 7 -> row 7 of the table. The rows start random and are LEARNED:
   during training, tokens that behave similarly drift to similar vectors.

2) POSITIONAL ENCODING: attention (lesson 3) is order-blind — it treats the
   input as a SET of vectors. "dog bites man" and "man bites dog" would look
   identical! So we add a position-dependent vector to each token embedding.

   The classic "Attention Is All You Need" recipe uses sines and cosines of
   different frequencies:

       PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
       PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

   Each position gets a unique fingerprint, and nearby positions get similar
   fingerprints. (GPT-style models instead LEARN a position table — same
   idea, and that's what our trainable model in model.py uses.)

Run me:  python3 02_embeddings_and_positions.py
"""

from __future__ import annotations

import numpy as np


def token_embedding_lookup(ids: np.ndarray, table: np.ndarray) -> np.ndarray:
    """ids: (T,) integer ids. table: (vocab, d_model). Returns (T, d_model).

    An embedding "layer" is literally just fancy indexing.
    """
    if ids.size and (ids.min() < 0 or ids.max() >= table.shape[0]):
        raise ValueError(f"ids must be in [0, {table.shape[0]})")
    return table[ids]


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """Build the (max_len, d_model) sinusoidal position table."""
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")
    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError(f"d_model must be a positive even number, got "
                         f"{d_model} (sin/cos come in pairs)")
    pos = np.arange(max_len)[:, None]            # (max_len, 1)
    i = np.arange(0, d_model, 2)[None, :]        # (1, d_model/2)
    angle = pos / (10000 ** (i / d_model))       # (max_len, d_model/2)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angle)                  # even dims get sin
    pe[:, 1::2] = np.cos(angle)                  # odd dims get cos
    return pe


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def demo() -> None:
    rng = np.random.default_rng(42)
    vocab_size, d_model = 50, 16

    # --- token embeddings -------------------------------------------------
    table = rng.normal(0, 0.02, size=(vocab_size, d_model))
    ids = np.array([3, 17, 3, 8])  # note token 3 appears twice
    emb = token_embedding_lookup(ids, table)
    print(f"ids {ids} -> embeddings of shape {emb.shape}")
    assert np.allclose(emb[0], emb[2]), "same token id => identical vector (before positions!)"
    print("positions 0 and 2 hold the same token -> identical vectors. "
          "The model can't tell them apart yet.\n")

    # --- positional encoding ----------------------------------------------
    pe = sinusoidal_positional_encoding(max_len=32, d_model=d_model)
    x = emb + pe[: len(ids)]  # broadcast add: THIS is the transformer's input
    assert not np.allclose(x[0], x[2]), "after adding positions they differ"
    print("After adding positional encodings, the two copies of token 3 differ.")

    # Nearby positions have similar encodings; distant ones don't.
    print("\nCosine similarity between position vectors:")
    for p, q in [(0, 1), (0, 2), (0, 5), (0, 20)]:
        print(f"  pos {p} vs pos {q:>2}: {cosine_similarity(pe[p], pe[q]):+.3f}")
    print("-> similarity decays with distance: positions carry geometry.")

    # Visualize the table coarsely (rows = positions, cols = dims).
    print("\nPE table sign pattern (+/-), first 12 positions x 16 dims:")
    for row in pe[:12]:
        print("  " + "".join("+" if v >= 0 else "-" for v in row))
    print("-> low dims oscillate fast, high dims slowly: like binary counting.")


if __name__ == "__main__":
    demo()
