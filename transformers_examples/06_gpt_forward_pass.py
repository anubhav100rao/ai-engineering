"""
LESSON 6: One forward pass through a full GPT, narrated
========================================================

Lessons 1-5 built the pieces. model.py assembles them into TinyGPT — the
same architecture as GPT-2, just tiny. This lesson traces ONE forward pass
end to end and shows the tensor shapes at every stage, then explains how
the loss is computed.

The whole pipeline:

    "Alice was"                                text
      -> [13, 50, 47, ...]                     token ids        (T,)
      -> wte[ids] + wpe[:T]                    vectors          (T, D)
      -> block 0, block 1, ... (attention+MLP) vectors          (T, D)
      -> final LayerNorm                       vectors          (T, D)
      -> linear head                           logits           (T, vocab)
      -> softmax                               P(next token)    (T, vocab)

Note position t's output predicts token t+1 — a single forward pass makes
T predictions at once. That's why transformers train so efficiently.

Run me:  python3 06_gpt_forward_pass.py
"""

import importlib
import numpy as np

from model import TinyGPT, softmax

# lesson files start with digits, so import them the roundabout way
tokenization = importlib.import_module("01_tokenization")


def demo():
    np.set_printoptions(precision=3, suppress=True)

    corpus = open("data/tiny_corpus.txt", encoding="utf-8").read()
    tok = tokenization.CharTokenizer(corpus)

    model = TinyGPT(vocab_size=tok.vocab_size, block_size=32,
                    n_embd=48, n_head=4, n_layer=2, seed=0)
    print(f"TinyGPT: {model.n_layer} layers, {model.n_head} heads, "
          f"d_model={model.n_embd}, vocab={tok.vocab_size}")
    print(f"Total parameters: {model.num_params():,} "
          f"(GPT-2 has 124,000,000; same architecture!)\n")

    prompt = "Alice was b"
    ids = np.array([tok.encode(prompt)])          # (1, T) — batch of one
    B, T = ids.shape
    D, V = model.n_embd, tok.vocab_size

    print(f"prompt {prompt!r} -> ids {ids[0]}   shape (B={B}, T={T})\n")
    print("Shapes through the network:")
    print(f"  token embedding  wte[ids] : ({B}, {T}, {D})")
    print(f"  + pos embedding  wpe[:T]  :     ({T}, {D})  broadcast over batch")
    for i in range(model.n_layer):
        print(f"  block {i} (attn+MLP)        : ({B}, {T}, {D})  shape-preserving")
    print(f"  final layernorm           : ({B}, {T}, {D})")
    print(f"  head projection           : ({B}, {T}, {V})  <- logits\n")

    logits, _, _ = model.forward(ids)
    assert logits.shape == (B, T, V)

    # The last row of logits scores every character as the NEXT one.
    probs = softmax(logits[0, -1])
    top = np.argsort(probs)[::-1][:5]
    print(f"P(next char | {prompt!r}) — top 5 (untrained, so ~uniform noise):")
    for t in top:
        print(f"  {tok.itos[t]!r:>6} : {probs[t]:.4f}")
    print(f"  (uniform would be 1/{V} = {1/V:.4f} — training is what sharpens this)\n")

    # THE LOSS: shift-by-one next-token prediction, all positions at once.
    seq = np.array([tok.encode("Alice was beg")])
    inputs, targets = seq[:, :-1], seq[:, 1:]     # x[t] predicts x[t+1]
    _, loss, _ = model.forward(inputs, targets)
    print("Loss = average cross-entropy over every (context -> next char) pair.")
    print(f"  untrained loss : {loss:.3f}")
    print(f"  ln(vocab) =    : {np.log(V):.3f}  <- 'random guessing' baseline")
    print("Training (lesson 7) is just: nudge all parameters to push this down.")


if __name__ == "__main__":
    demo()
