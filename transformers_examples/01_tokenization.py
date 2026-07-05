"""
LESSON 1: Tokenization — how text becomes numbers
==================================================

A neural network can't read "hello". It can only do math on numbers.
Tokenization is the bridge: it chops text into pieces (tokens) and maps
each piece to an integer id.

Three common granularities:

  character-level : "hello" -> ['h','e','l','l','o']      tiny vocab, long sequences
  word-level      : "hello world" -> ['hello','world']    huge vocab, can't handle new words
  subword (BPE)   : "unhappiness" -> ['un','happi','ness'] the sweet spot; used by GPT

In these lessons we use CHARACTER-LEVEL tokenization because it's the
simplest thing that actually works, and it lets us train a real model on a
small corpus. The transformer itself doesn't care — it just sees integers.

The implementation lives in char_tokenizer.py (30 lines — read it!); every
other file in this course imports that one class, so there is exactly one
tokenizer in the codebase.

Run me:  python3 01_tokenization.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from char_tokenizer import CharTokenizer

CORPUS_PATH = Path(__file__).resolve().parent / "data" / "tiny_corpus.txt"


def load_corpus(path: Path = CORPUS_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"corpus not found at {path} — did data/tiny_corpus.txt move?")
    return path.read_text(encoding="utf-8")


def demo() -> None:
    corpus = load_corpus()
    tok = CharTokenizer(corpus)

    print(f"Corpus length : {len(corpus):,} characters")
    print(f"Vocab size    : {tok.vocab_size} unique characters")
    print(f"Vocabulary    : {tok.vocab!r}\n")

    sample = "Alice was beginning"
    ids = tok.encode(sample)
    print(f"encode({sample!r})")
    print(f"  -> {ids}")
    print(f"decode(...) -> {tok.decode(ids)!r}")
    assert tok.decode(ids) == sample, "round-trip must be lossless"

    # Encoding is STRICT: unknown characters fail loudly at the boundary
    # instead of corrupting training data silently.
    try:
        tok.encode("Alice 🐇")
    except ValueError as e:
        print(f"\nencode('Alice 🐇') raises: {e}")

    # This is what the model will actually consume: a numpy array of ids.
    data = np.array(tok.encode(corpus), dtype=np.int64)
    print(f"\nAs training data: array of shape {data.shape}, dtype {data.dtype}")
    print(f"First 20 ids: {data[:20]}")

    # Language modeling = predict the NEXT token. So inputs and targets are
    # the same sequence shifted by one:
    block = data[:10]
    print("\nNext-token prediction pairs (input -> target):")
    for i in range(1, 6):
        ctx, tgt = block[:i], block[i]
        print(f"  {tok.decode(ctx.tolist())!r:>12} -> {tok.decode([int(tgt)])!r}")


if __name__ == "__main__":
    demo()
