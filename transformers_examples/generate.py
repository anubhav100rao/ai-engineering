"""Sample text from a TinyGPT checkpoint produced by 07_train_tiny_gpt.py.

Separating training from inference is the production norm: train once,
save a self-describing checkpoint (weights + config + vocabulary), then
load and sample as many times as you like.

Usage:
    python3 generate.py                                  # defaults
    python3 generate.py --prompt "The Queen" --tokens 300
    python3 generate.py --temperature 0.3                # sharper, safer text
    python3 generate.py --temperature 1.5                # chaotic text
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from char_tokenizer import CharTokenizer
from model import TinyGPT

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "checkpoints" / "tiny_gpt.npz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--prompt", default="Alice ")
    ap.add_argument("--tokens", type=int, default=250,
                    help="number of characters to generate")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="<1 sharpens the distribution, >1 flattens it")
    ap.add_argument("--top-k", type=int, default=10,
                    help="sample only from the k most likely characters")
    ap.add_argument("--seed", type=int, default=None,
                    help="rng seed for reproducible samples")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.checkpoint.exists():
        raise SystemExit(
            f"error: no checkpoint at {args.checkpoint}\n"
            f"train one first:  python3 07_train_tiny_gpt.py")

    model, meta = TinyGPT.load(args.checkpoint)
    if "vocab" not in meta:
        raise SystemExit(f"error: checkpoint {args.checkpoint} has no vocabulary "
                         "metadata; retrain with 07_train_tiny_gpt.py")
    tok = CharTokenizer.from_vocab(meta["vocab"])

    print(f"loaded {args.checkpoint.name}: {model.num_params():,} params, "
          f"trained {meta.get('steps', '?')} steps on "
          f"{Path(str(meta.get('corpus', '?'))).name}\n")

    ids = np.array([tok.encode(args.prompt)])
    rng = np.random.default_rng(args.seed)
    out = model.generate(ids, args.tokens, temperature=args.temperature,
                         top_k=args.top_k, rng=rng)
    print(tok.decode(out[0]))


if __name__ == "__main__":
    main()
