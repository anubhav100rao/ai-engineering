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

Production touches worth noticing:
  * every hyperparameter is a CLI flag (reproducible runs, no magic numbers)
  * validation loss is averaged over several batches, not one noisy sample
  * gradient clipping guards against loss spikes
  * the trained model is CHECKPOINTED with its config and vocabulary,
    so generate.py can reload it without retraining

Run me:        python3 07_train_tiny_gpt.py
Quick check:   python3 07_train_tiny_gpt.py --steps 200
Then sample:   python3 generate.py --prompt "The Queen"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from char_tokenizer import CharTokenizer
from model import Adam, GPTConfig, TinyGPT, clip_grad_norm

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "tiny_corpus.txt"
DEFAULT_CHECKPOINT = HERE / "checkpoints" / "tiny_gpt.npz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="path to a plain-text training corpus")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                    help="where to save the trained model (.npz)")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--block-size", type=int, default=64, help="context length")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="max global gradient norm")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=8,
                    help="batches to average for the validation loss")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def get_batch(
    data: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random windows: inputs = data[i:i+T], targets = data[i+1:i+T+1]."""
    if len(data) < block_size + 2:
        raise ValueError(f"corpus too small ({len(data)} tokens) for "
                         f"block_size={block_size}")
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s: s + block_size] for s in starts])
    y = np.stack([data[s + 1: s + block_size + 1] for s in starts])
    return x, y


def evaluate(
    model: TinyGPT, data: np.ndarray, block_size: int, batch_size: int,
    n_batches: int, rng: np.random.Generator,
) -> float:
    """Mean loss over several batches — one batch is too noisy to trust."""
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, rng)
        _, loss, _ = model.forward(x, y)
        losses.append(loss)
    return float(np.mean(losses))


def sample_text(model: TinyGPT, tok: CharTokenizer, prompt: str = "Alice ",
                n_chars: int = 150, seed: int = 1) -> str:
    ids = np.array([tok.encode(prompt)])
    out = model.generate(ids, n_chars, temperature=0.8, top_k=10,
                         rng=np.random.default_rng(seed))
    return tok.decode(out[0])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    if not args.data.exists():
        raise SystemExit(f"error: corpus not found at {args.data}")
    text = args.data.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    data = np.array(tok.encode(text), dtype=np.int64)

    # 90/10 train/validation split so we can detect memorization.
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    config = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                       n_embd=args.n_embd, n_head=args.n_head,
                       n_layer=args.n_layer, seed=args.seed)
    model = TinyGPT(config)
    opt = Adam(model.params, lr=args.lr, weight_decay=args.weight_decay)

    print(f"corpus: {len(data):,} chars, vocab {tok.vocab_size}, "
          f"model {model.num_params():,} params")
    print(f"baseline loss (random guessing) = ln({tok.vocab_size}) "
          f"= {np.log(tok.vocab_size):.3f}\n")

    t0 = time.time()
    for step in range(args.steps + 1):
        x, y = get_batch(train_data, args.block_size, args.batch_size, rng)
        _, loss, cache = model.forward(x, y)        # forward
        grads = model.backward(cache)               # backward (by hand!)
        gnorm = clip_grad_norm(grads, args.grad_clip)
        opt.step(model.params, grads)               # AdamW update

        if step % args.eval_every == 0:
            vloss = evaluate(model, val_data, args.block_size,
                             args.batch_size, args.eval_batches, rng)
            print(f"step {step:5d} | train loss {loss:.3f} | "
                  f"val loss {vloss:.3f} | grad norm {gnorm:6.2f} | "
                  f"{time.time() - t0:5.1f}s")
        if step in (0, args.steps // 4, args.steps):
            print(f"\n--- sample at step {step} ---")
            print(sample_text(model, tok).replace("\n", " "))
            print()

    # Persist everything generate.py needs: weights, config, and vocab.
    saved = model.save(args.checkpoint, meta={"vocab": tok.vocab,
                                              "corpus": str(args.data),
                                              "steps": args.steps})
    print(f"checkpoint saved to {saved}")
    print("\nThings to try:")
    print("  python3 generate.py --prompt 'The Queen'   # sample without retraining")
    print("  --steps 5000 --n-embd 128 --n-layer 4      # bigger model, better text")
    print("  --data path/to/your_text.txt               # train on your own corpus")


if __name__ == "__main__":
    main()
