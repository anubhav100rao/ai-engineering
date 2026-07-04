from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
import json
import urllib.request


Pair = Tuple[int, int]


def get_pair_counts(ids: List[int]) -> Counter[Pair]:
    """
    Count adjacent token pairs.

    Example:
        ids = [10, 20, 10, 20, 30]
        pairs:
            (10, 20)
            (20, 10)
            (10, 20)
            (20, 30)

        counts:
            (10, 20): 2
            (20, 10): 1
            (20, 30): 1
    """
    return Counter(zip(ids, ids[1:]))


def merge_pair(ids: List[int], pair: Pair, new_id: int) -> List[int]:
    """
    Replace every non-overlapping occurrence of `pair` with `new_id`.

    Example:
        ids = [1, 2, 1, 2, 3]
        pair = (1, 2)
        new_id = 99

        result = [99, 99, 3]
    """
    output = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            output.append(new_id)
            i += 2
        else:
            output.append(ids[i])
            i += 1

    return output


class ByteBPETokenizer:
    """
    Minimal byte-level BPE tokenizer.

    Special token IDs:
        <PAD> = 0
        <BOS> = 1
        <EOS> = 2

    Raw bytes are shifted by byte_offset.

    So byte 0 becomes token ID 3.
    Byte 1 becomes token ID 4.
    ...
    Byte 255 becomes token ID 258.

    Learned merge tokens start from 259.
    """

    def __init__(
        self,
        merges: Dict[Pair, int] | None = None,
        special_tokens: Dict[str, int] | None = None,
    ):
        self.special_tokens = special_tokens or {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
        }

        self.byte_offset = len(self.special_tokens)

        self.merges: Dict[Pair, int] = merges or {}

        # Maps token_id -> bytes.
        self.vocab_bytes: Dict[int, bytes] = {
            byte_value + self.byte_offset: bytes([byte_value])
            for byte_value in range(256)
        }

        # Rebuild merged-token byte values if loading from disk.
        for (left_id, right_id), new_id in sorted(
            self.merges.items(),
            key=lambda item: item[1],
        ):
            self.vocab_bytes[new_id] = (
                self.vocab_bytes[left_id] + self.vocab_bytes[right_id]
            )

    @property
    def vocab_size(self) -> int:
        return len(self.special_tokens) + len(self.vocab_bytes)

    def train(
        self,
        text: str,
        target_vocab_size: int = 1000,
        min_pair_freq: int = 2,
        verbose: bool = True,
    ) -> "ByteBPETokenizer":
        """
        Train BPE merges from raw text.

        target_vocab_size includes:
            special tokens
            256 byte tokens
            learned merge tokens

        Example:
            special tokens = 3
            byte tokens = 256
            target_vocab_size = 1000

            learned merges = 1000 - 3 - 256 = 741
        """
        if target_vocab_size <= self.byte_offset + 256:
            raise ValueError(
                f"target_vocab_size must be greater than {self.byte_offset + 256}"
            )

        # Start from UTF-8 bytes.
        ids = [
            byte_value + self.byte_offset
            for byte_value in text.encode("utf-8")
        ]

        next_id = self.byte_offset + 256
        max_merges = target_vocab_size - next_id

        for step in range(max_merges):
            pair_counts = get_pair_counts(ids)

            if not pair_counts:
                break

            best_pair, best_freq = pair_counts.most_common(1)[0]

            if best_freq < min_pair_freq:
                if verbose:
                    print(
                        f"Stopping: best pair frequency {best_freq} "
                        f"is less than min_pair_freq={min_pair_freq}"
                    )
                break

            self.merges[best_pair] = next_id
            self.vocab_bytes[next_id] = (
                self.vocab_bytes[best_pair[0]] + self.vocab_bytes[best_pair[1]]
            )

            ids = merge_pair(ids, best_pair, next_id)

            if verbose and ((step + 1) % 100 == 0 or step == 0):
                token_preview = self.vocab_bytes[next_id].decode(
                    "utf-8",
                    errors="replace",
                )

                print(
                    f"merge={step + 1:4d} | "
                    f"new_id={next_id:5d} | "
                    f"freq={best_freq:5d} | "
                    f"token={token_preview!r}"
                )

            next_id += 1

        return self

    def _encode_bytes(self, raw: bytes) -> List[int]:
        """
        Encode bytes using learned BPE merges.

        Start with individual bytes, then repeatedly apply the learned merge
        with the best rank.

        Earlier merges have lower token IDs, so lower new_id means higher priority.
        """
        ids = [
            byte_value + self.byte_offset
            for byte_value in raw
        ]

        while len(ids) >= 2:
            pairs = list(zip(ids, ids[1:]))

            candidates = [
                (self.merges[pair], pair)
                for pair in pairs
                if pair in self.merges
            ]

            if not candidates:
                break

            _, best_pair = min(candidates)
            new_id = self.merges[best_pair]

            ids = merge_pair(ids, best_pair, new_id)

        return ids

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        ids = self._encode_bytes(text.encode("utf-8"))

        if add_bos:
            ids = [self.special_tokens["<BOS>"]] + ids

        if add_eos:
            ids = ids + [self.special_tokens["<EOS>"]]

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        special_ids = set(self.special_tokens.values())
        byte_chunks = []

        for token_id in ids:
            if token_id in special_ids:
                if skip_special_tokens:
                    continue

                token_name = self.id_to_special_token(token_id)
                byte_chunks.append(token_name.encode("utf-8"))

            elif token_id in self.vocab_bytes:
                byte_chunks.append(self.vocab_bytes[token_id])

            else:
                raise ValueError(f"Unknown token ID: {token_id}")

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def id_to_special_token(self, token_id: int) -> str:
        for token, idx in self.special_tokens.items():
            if idx == token_id:
                return token

        raise ValueError(f"Unknown special token ID: {token_id}")

    def token_to_string(self, token_id: int) -> str:
        """
        Helpful for inspecting tokens.
        """
        if token_id in self.special_tokens.values():
            return self.id_to_special_token(token_id)

        if token_id not in self.vocab_bytes:
            raise ValueError(f"Unknown token ID: {token_id}")

        return self.vocab_bytes[token_id].decode("utf-8", errors="replace")

    def explain_encode(self, text: str) -> None:
        """
        Print token IDs and readable token pieces.
        """
        ids = self.encode(text)

        print(f"Text: {text!r}")
        print(f"Token count: {len(ids)}")
        print()

        for token_id in ids:
            piece = self.token_to_string(token_id)
            print(f"{token_id:5d} -> {piece!r}")

    def save(self, path: str | Path) -> None:
        path = Path(path)

        data = {
            "type": "byte_bpe",
            "special_tokens": self.special_tokens,
            "merges": [
                [left_id, right_id, new_id]
                for (left_id, right_id), new_id in sorted(
                    self.merges.items(),
                    key=lambda item: item[1],
                )
            ],
        }

        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        path = Path(path)

        data = json.loads(path.read_text(encoding="utf-8"))

        merges = {
            (left_id, right_id): new_id
            for left_id, right_id, new_id in data["merges"]
        }

        return cls(
            merges=merges,
            special_tokens=data["special_tokens"],
        )


def download_text(url: str, output_path: str | Path) -> str:
    """
    Download text file from the internet and save it locally.
    """
    output_path = Path(output_path)

    with urllib.request.urlopen(url) as response:
        raw = response.read()

    text = raw.decode("utf-8", errors="replace")
    output_path.write_text(text, encoding="utf-8")

    return text


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Project Gutenberg texts usually contain a header and footer.
    This helper keeps mostly the actual book content.
    """
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_idx = text.find(start_marker)
    if start_idx != -1:
        start_idx = text.find("\n", start_idx)
        text = text[start_idx + 1:]

    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]

    return text.strip()


def compression_stats(tokenizer: ByteBPETokenizer, text: str) -> None:
    raw_bytes = text.encode("utf-8")
    token_ids = tokenizer.encode(text)

    print(f"Characters: {len(text):,}")
    print(f"UTF-8 bytes: {len(raw_bytes):,}")
    print(f"Tokens:     {len(token_ids):,}")
    print(f"Bytes/token: {len(raw_bytes) / max(1, len(token_ids)):.2f}")


if __name__ == "__main__":
    # Plain Text UTF-8 Alice in Wonderland from Project Gutenberg.
    url = "https://www.gutenberg.org/files/11/11-0.txt"

    corpus_path = Path("alice.txt")
    tokenizer_path = Path("tokenizer.json")

    if corpus_path.exists():
        text = corpus_path.read_text(encoding="utf-8")
    else:
        text = download_text(url, corpus_path)

    text = strip_gutenberg_boilerplate(text)

    tokenizer = ByteBPETokenizer()

    tokenizer.train(
        text=text,
        target_vocab_size=1000,
        min_pair_freq=2,
        verbose=True,
    )

    tokenizer.save(tokenizer_path)

    print()
    print(f"Saved tokenizer to {tokenizer_path}")
    print()

    sample = "Alice was beginning to get very tired. मैं Python सीख रहा हूँ 😊"

    ids = tokenizer.encode(sample, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(ids)

    print("Sample text:")
    print(sample)
    print()

    print("Encoded IDs:")
    print(ids)
    print()

    print("Decoded text:")
    print(decoded)
    print()

    print("Token inspection:")
    tokenizer.explain_encode(sample)
    print()

    print("Compression stats on sample:")
    compression_stats(tokenizer, sample)
