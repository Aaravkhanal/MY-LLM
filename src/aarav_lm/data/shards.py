"""
aarav-lm · Token shard files

A shard is a flat array of token ids behind a fixed 1024-byte header, readable with
np.memmap so the OS pages data in lazily (nothing is loaded into RAM up front).

Header: 128 little-endian int64 values
    [0] magic (0x4141524C, "AARL")     [1] format version
    [2] number of tokens               [3] vocab size
    [4] bytes per token (2 or 4)       [5] number of documents
The tokenizer fingerprint lives in the dataset manifest next to the shards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 0x4141524C
VERSION = 1
HEADER_INTS = 128
HEADER_BYTES = HEADER_INTS * 8


class ShardError(ValueError):
    pass


def token_dtype(vocab_size: int) -> np.dtype:
    return np.dtype("<u2") if vocab_size <= 1 << 16 else np.dtype("<u4")


class ShardWriter:
    """Append token arrays, then finalise the header on close."""

    def __init__(self, path: str | Path, vocab_size: int):
        self.path = Path(path)
        self.vocab_size = vocab_size
        self.dtype = token_dtype(vocab_size)
        self.n_tokens = 0
        self.n_docs = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "wb")
        self._f.write(b"\0" * HEADER_BYTES)

    def append(self, tokens: np.ndarray, n_docs: int = 1) -> None:
        if tokens.size and int(tokens.max()) >= self.vocab_size:
            raise ShardError(f"token id {int(tokens.max())} >= vocab size {self.vocab_size}")
        self._f.write(np.ascontiguousarray(tokens, dtype=self.dtype).tobytes())
        self.n_tokens += int(tokens.size)
        self.n_docs += n_docs

    def close(self) -> None:
        if self._f.closed:
            return
        header = np.zeros(HEADER_INTS, dtype="<i8")
        header[:6] = [MAGIC, VERSION, self.n_tokens, self.vocab_size, self.dtype.itemsize, self.n_docs]
        self._f.seek(0)
        self._f.write(header.tobytes())
        self._f.close()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class Shard:
    path: Path
    n_tokens: int
    vocab_size: int
    n_docs: int
    tokens: np.ndarray      # memmap view

    @classmethod
    def open(cls, path: str | Path) -> "Shard":
        path = Path(path)
        size = path.stat().st_size
        if size < HEADER_BYTES:
            raise ShardError(f"{path}: file smaller than the {HEADER_BYTES}-byte header")
        h = np.fromfile(path, dtype="<i8", count=HEADER_INTS)
        if h[0] != MAGIC:
            raise ShardError(f"{path}: bad magic {int(h[0]):#x} (not an aarav-lm shard)")
        if h[1] != VERSION:
            raise ShardError(f"{path}: unsupported shard version {int(h[1])}")
        n_tokens, vocab, itemsize, n_docs = (int(x) for x in h[2:6])
        if itemsize not in (2, 4) or itemsize != token_dtype(vocab).itemsize:
            raise ShardError(f"{path}: inconsistent token width {itemsize} for vocab {vocab}")
        expected = HEADER_BYTES + n_tokens * itemsize
        if size != expected:
            raise ShardError(f"{path}: size {size} != expected {expected} (truncated or corrupt)")
        tokens = np.memmap(path, dtype=token_dtype(vocab), mode="r", offset=HEADER_BYTES, shape=(n_tokens,))
        return cls(path, n_tokens, vocab, n_docs, tokens)
