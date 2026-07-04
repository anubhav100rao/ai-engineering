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

Run me:  python3 01_tokenization.py
"""

import numpy as np


class CharTokenizer:
    """Maps every unique character in a corpus to an integer id."""

    def __init__(self, text: str):
        # The vocabulary is simply the sorted set of unique characters.
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        # stoi: string -> integer, itos: integer -> string
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def demo():
    corpus = open("data/tiny_corpus.txt", encoding="utf-8").read()
    tok = CharTokenizer(corpus)

    print(f"Corpus length : {len(corpus):,} characters")
    print(f"Vocab size    : {tok.vocab_size} unique characters")
    print(f"Vocabulary    : {''.join(tok.chars)!r}\n")

    sample = "Alice was beginning"
    ids = tok.encode(sample)
    print(f"encode({sample!r})")
    print(f"  -> {ids}")
    print(f"decode(...) -> {tok.decode(ids)!r}")
    assert tok.decode(ids) == sample, "round-trip must be lossless"

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
