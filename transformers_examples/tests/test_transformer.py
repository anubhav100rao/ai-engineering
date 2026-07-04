"""
Tests for the transformer lessons.

Run from the transformers_examples directory:

    python3 tests/test_transformer.py        # plain python, no deps
    python3 -m pytest tests/ -v              # or with pytest if you have it

The crown jewel is test_gradient_check: it verifies the hand-written
backprop in model.py against numerical finite-difference gradients. If that
passes, every chain-rule step through attention, layernorm, residuals and
embeddings is correct.
"""

import importlib
import os
import sys

import numpy as np

# make the lesson files importable regardless of where tests run from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model import TinyGPT, Adam, softmax  # noqa: E402

lesson01 = importlib.import_module("01_tokenization")
lesson02 = importlib.import_module("02_embeddings_and_positions")
lesson03 = importlib.import_module("03_attention")
lesson04 = importlib.import_module("04_multi_head_attention")
lesson05 = importlib.import_module("05_transformer_block")


# --------------------------- lesson unit tests ------------------------------

def test_tokenizer_roundtrip():
    tok = lesson01.CharTokenizer("hello world")
    s = "hello hello world"
    assert tok.decode(tok.encode(s)) == s
    assert tok.vocab_size == len(set("hello world"))


def test_positional_encoding_properties():
    pe = lesson02.sinusoidal_positional_encoding(max_len=100, d_model=32)
    assert pe.shape == (100, 32)
    assert np.all(np.abs(pe) <= 1.0)                    # sin/cos bounded
    # every position's fingerprint is unique
    assert len({tuple(np.round(row, 6)) for row in pe}) == 100


def test_attention_weights_are_distributions():
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(3, 7, 8))
    out, W = lesson03.scaled_dot_product_attention(Q, K, V)
    assert out.shape == (7, 8)
    assert np.allclose(W.sum(-1), 1.0)
    assert np.all(W >= 0)


def test_causal_mask_blocks_future():
    rng = np.random.default_rng(1)
    Q, K, V = rng.normal(size=(3, 6, 8))
    out, W = lesson03.scaled_dot_product_attention(Q, K, V, causal=True)
    assert np.allclose(np.triu(W, k=1), 0.0)
    # changing a FUTURE token must not change past outputs
    V2 = V.copy()
    V2[-1] += 100.0
    out2, _ = lesson03.scaled_dot_product_attention(Q, K, V2, causal=True)
    assert np.allclose(out[:-1], out2[:-1]), "future leaked into the past!"


def test_multihead_shapes_and_mask():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(10, 32))
    mha = lesson04.MultiHeadAttention(d_model=32, n_heads=4)
    out, W = mha(x, causal=True)
    assert out.shape == (10, 32)
    assert W.shape == (4, 10, 10)
    assert np.allclose(W.sum(-1), 1.0)
    assert np.allclose(np.triu(W, k=1), 0.0)


def test_block_is_shape_preserving_and_ffn_is_per_token():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(9, 32))
    block = lesson05.TransformerBlock(d_model=32, n_heads=4)
    assert block(x).shape == x.shape
    perm = rng.permutation(9)
    assert np.allclose(block.ffn(x)[perm], block.ffn(x[perm]))


def test_layernorm_normalizes():
    rng = np.random.default_rng(4)
    x = rng.normal(3.0, 5.0, size=(6, 16))              # off-center input
    y = lesson05.layer_norm(x, np.ones(16), np.zeros(16))
    assert np.allclose(y.mean(-1), 0.0, atol=1e-6)
    assert np.allclose(y.std(-1), 1.0, atol=1e-2)


# --------------------------- model.py tests ---------------------------------

def _tiny_model():
    return TinyGPT(vocab_size=11, block_size=6, n_embd=16, n_head=2,
                   n_layer=2, seed=0)


def test_model_forward_shapes_and_loss():
    m = _tiny_model()
    rng = np.random.default_rng(5)
    x = rng.integers(0, 11, size=(3, 6))
    y = rng.integers(0, 11, size=(3, 6))
    logits, loss, _ = m.forward(x, y)
    assert logits.shape == (3, 6, 11)
    # untrained loss should sit near the random-guessing baseline ln(V)
    assert abs(loss - np.log(11)) < 0.2


def test_model_is_causal():
    m = _tiny_model()
    rng = np.random.default_rng(6)
    x = rng.integers(0, 11, size=(1, 6))
    logits1, _, _ = m.forward(x)
    x2 = x.copy()
    x2[0, -1] = (x2[0, -1] + 1) % 11                    # change the LAST token
    logits2, _, _ = m.forward(x2)
    assert np.allclose(logits1[0, :-1], logits2[0, :-1]), \
        "changing token T affected predictions before T — causality broken"


def test_gradient_check():
    """Backprop vs finite differences — the definitive correctness test."""
    m = _tiny_model()
    rng = np.random.default_rng(7)
    x = rng.integers(0, 11, size=(2, 5))
    y = rng.integers(0, 11, size=(2, 5))

    _, _, cache = m.forward(x, y)
    grads = m.backward(cache)

    def loss_at():
        _, loss, _ = m.forward(x, y)
        return loss

    h = 1e-5
    checked = 0
    for name, P in m.params.items():
        flat = P.reshape(-1)
        # probe up to 3 random entries of every parameter tensor
        for j in rng.choice(flat.size, size=min(3, flat.size), replace=False):
            old = flat[j]
            flat[j] = old + h
            lp = loss_at()
            flat[j] = old - h
            lm = loss_at()
            flat[j] = old
            num = (lp - lm) / (2 * h)
            ana = grads[name].reshape(-1)[j]
            denom = max(abs(num), abs(ana), 1e-8)
            rel = abs(num - ana) / denom
            assert rel < 1e-4 or abs(num - ana) < 1e-9, (
                f"gradient mismatch in {name}[{j}]: "
                f"numerical={num:.3e} analytical={ana:.3e} rel={rel:.2e}")
            checked += 1
    print(f"    (gradient check passed on {checked} parameter entries)")


def test_training_reduces_loss():
    """A few Adam steps on a fixed batch must drive the loss down."""
    m = _tiny_model()
    opt = Adam(m.params, lr=1e-2)
    rng = np.random.default_rng(8)
    x = rng.integers(0, 11, size=(4, 6))
    y = rng.integers(0, 11, size=(4, 6))
    _, first, cache = m.forward(x, y)
    for _ in range(30):
        _, loss, cache = m.forward(x, y)
        opt.step(m.params, m.backward(cache))
    _, last, _ = m.forward(x, y)
    assert last < first * 0.5, f"loss barely moved: {first:.3f} -> {last:.3f}"


def test_generate_extends_sequence():
    m = _tiny_model()
    out = m.generate(np.array([[1, 2, 3]]), max_new_tokens=8,
                     rng=np.random.default_rng(9))
    assert out.shape == (1, 11)
    assert np.all((out >= 0) & (out < 11))


# --------------------------- runner -----------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
