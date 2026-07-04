# Learn Transformer Architecture from Scratch

A hands-on course in **pure numpy** — no PyTorch, no TensorFlow, nothing hidden.
By the end you'll have trained a real (tiny) GPT on *Alice in Wonderland*,
with every gradient derived by hand.

## The path

Run each lesson in order; each one prints a narrated demo:

| File | What you learn |
|------|----------------|
| `01_tokenization.py` | Text → integer ids; next-token prediction setup |
| `02_embeddings_and_positions.py` | Ids → vectors; why attention needs positional encoding |
| `03_attention.py` | Scaled dot-product attention `softmax(QKᵀ/√d)V`, worked by hand; causal masking |
| `04_multi_head_attention.py` | Many attention patterns in parallel = one matmul + a reshape |
| `05_transformer_block.py` | LayerNorm, residual connections, feed-forward — the plumbing that makes depth work |
| `06_gpt_forward_pass.py` | One narrated forward pass through the full model, shapes at every stage |
| `07_train_tiny_gpt.py` | Train the model end-to-end and watch it learn to write |

Supporting files:

- **`model.py`** — `TinyGPT`: the complete GPT (same architecture as GPT-2, just small),
  with **hand-written backpropagation** and an Adam optimizer. Read it after lesson 5.
- **`data/tiny_corpus.txt`** — 60 KB of *Alice in Wonderland* (public domain).
- **`tests/test_transformer.py`** — 12 tests, including a **numerical gradient check**
  that proves the hand-derived backprop is correct, and a causality test that proves
  the model can't peek at the future.

## Quick start

```bash
cd transformers_examples

python3 01_tokenization.py            # then 02, 03, ... in order
python3 tests/test_transformer.py     # run all tests (no pytest needed)
python3 07_train_tiny_gpt.py          # the finale: ~30s of training on CPU
```

Only requirement: `numpy`.

## The big picture

```
"Alice was"                                  text
  → [13, 50, 47, ...]                        tokenization        (lesson 1)
  → wte[ids] + wpe[:T]                       embeddings          (lesson 2)
  → ┌ x = x + Attention(LayerNorm(x))  ┐     tokens talk         (lessons 3-4)
    └ x = x + FeedForward(LayerNorm(x))┘ ×N  tokens think        (lesson 5)
  → LayerNorm → linear head                  logits              (lesson 6)
  → softmax → sample → repeat                generation          (lesson 7)
```

Three ideas carry the whole architecture:

1. **Attention** lets every token gather information from every earlier token,
   with content-based weights — no recurrence, fully parallel.
2. **Residual streams** mean each block adds a small *correction* rather than
   replacing the signal, so gradients survive dozens of layers.
3. **Next-token prediction** turns any text into `len(text)` training examples,
   all learned simultaneously in one forward pass.

## Where to go next

- Swap `data/tiny_corpus.txt` for your own text and retrain.
- Scale up `n_embd`, `n_layer`, `block_size` in lesson 7 and watch the samples improve.
- Read [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) —
  after these lessons the paper reads easily.
- Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) is this same model in PyTorch, scaled up.
