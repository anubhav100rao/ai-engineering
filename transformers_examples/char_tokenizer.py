"""Character-level tokenizer shared by the lessons, the trainer, and the tests.

Lesson 01 explains what tokenization is and why it exists; this module is the
single implementation everything else imports. Character-level keeps the vocab
tiny (~70 symbols for English text) at the cost of longer sequences — the
right trade-off for a model we train on CPU.
"""

from __future__ import annotations

from typing import Iterable


class CharTokenizer:
    """Bijective mapping between characters and integer token ids.

    The vocabulary is the sorted set of unique characters in the text the
    tokenizer was built from. Encoding is strict: characters outside the
    vocabulary raise ``ValueError`` rather than being silently dropped or
    remapped, so data problems surface at the boundary instead of as
    mysterious training behavior.
    """

    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError("cannot build a vocabulary from empty text")
        self.chars: list[str] = sorted(set(text))
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: dict[int, str] = dict(enumerate(self.chars))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @property
    def vocab(self) -> str:
        """The vocabulary as one string — handy for storing in checkpoints."""
        return "".join(self.chars)

    @classmethod
    def from_vocab(cls, vocab: str) -> "CharTokenizer":
        """Rebuild a tokenizer from a vocabulary string (see :attr:`vocab`)."""
        return cls(vocab)

    def encode(self, text: str) -> list[int]:
        """Map a string to a list of token ids. Strict on unknown characters."""
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(
                f"character {exc.args[0]!r} is not in the vocabulary "
                f"({self.vocab_size} known characters); build the tokenizer "
                f"from text that covers your inputs"
            ) from None

    def decode(self, ids: Iterable[int]) -> str:
        """Map token ids back to a string. Strict on out-of-range ids."""
        out: list[str] = []
        for i in ids:
            i = int(i)
            if not 0 <= i < self.vocab_size:
                raise ValueError(
                    f"token id {i} is out of range [0, {self.vocab_size})"
                )
            out.append(self.itos[i])
        return "".join(out)

    def __repr__(self) -> str:  # helpful in the REPL and in test failures
        return f"CharTokenizer(vocab_size={self.vocab_size})"
