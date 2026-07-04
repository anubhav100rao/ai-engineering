"""
LESSON 7: Train a tiny GPT on Alice in Wonderland
==================================================

Everything comes together. The training loop is four lines, repeated:

    1. sample a random batch of (input, target) character windows
    2. forward pass  -> loss
    3. backward pass -> gradients   (hand-derived in model.py!)
    4. Adam step     -> update parameters

Watch the loss fall from ln(vocab) ≈ 4.2 ("random guessing") and the
samples evolve from noise -> letter frequencies -> words -> phrases.
Pure numpy on CPU, so we keep it small: ~15-30 seconds for 1200 steps.

Run me:        python3 07_train_tiny_gpt.py
Quick check:   python3 07_train_tiny_gpt.py --steps 200
"""

import argparse
import time

import numpy as np

from model import TinyGPT, Adam


class CharTokenizer:  # (same as lesson 1, inlined so this file stands alone)
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def get_batch(data, block_size, batch_size, rng):
    """Random windows: inputs = data[i:i+T], targets = data[i+1:i+T+1]."""
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s: s + block_size] for s in starts])
    y = np.stack([data[s + 1: s + block_size + 1] for s in starts])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--block-size", type=int, default=64, help="context length")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    text = open("data/tiny_corpus.txt", encoding="utf-8").read()
    tok = CharTokenizer(text)
    data = np.array(tok.encode(text), dtype=np.int64)

    # 90/10 train/validation split so we can detect memorization.
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    model = TinyGPT(vocab_size=tok.vocab_size, block_size=args.block_size,
                    n_embd=64, n_head=4, n_layer=2, seed=0)
    opt = Adam(model.params, lr=args.lr)

    print(f"corpus: {len(data):,} chars, vocab {tok.vocab_size}, "
          f"model {model.num_params():,} params")
    print(f"baseline loss (random guessing) = ln({tok.vocab_size}) "
          f"= {np.log(tok.vocab_size):.3f}\n")

    def sample(n_chars=150):
        prompt = np.array([tok.encode("Alice ")])
        out = model.generate(prompt, n_chars, temperature=0.8, top_k=10,
                             rng=np.random.default_rng(1))
        return tok.decode(out[0])

    t0 = time.time()
    for step in range(args.steps + 1):
        x, y = get_batch(train_data, args.block_size, args.batch_size, rng)
        _, loss, cache = model.forward(x, y)        # forward
        grads = model.backward(cache)               # backward (by hand!)
        opt.step(model.params, grads)               # Adam update

        if step % 100 == 0:
            xv, yv = get_batch(val_data, args.block_size, args.batch_size, rng)
            _, vloss, _ = model.forward(xv, yv)
            print(f"step {step:5d} | train loss {loss:.3f} | "
                  f"val loss {vloss:.3f} | {time.time()-t0:5.1f}s")
        if step in (0, args.steps // 4, args.steps):
            print(f"\n--- sample at step {step} ---")
            print(sample().replace("\n", " "))
            print()

    print("Done. Things to try:")
    print("  - more steps / bigger n_embd, n_layer (edit the TinyGPT(...) call)")
    print("  - temperature 0.3 vs 1.5 in sample() — sharp vs chaotic text")
    print("  - train on your own text: replace data/tiny_corpus.txt")


if __name__ == "__main__":
    main()
