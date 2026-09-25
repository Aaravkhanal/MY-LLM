"""
aarav-lm · Train / validation / test split

The split is decided by hashing each document's content together with a seed, not
by position or a shuffled index. That makes it deterministic, independent of
document order, and stable when more data is appended later: a document never moves
between splits, so held-out data can't leak into training across dataset versions.
"""

from __future__ import annotations

import hashlib

from aarav_lm.data.dedup import normalized_key


def split_of(text: str, seed: int, val_fraction: float, test_fraction: float) -> str:
    if not (0.0 <= val_fraction and 0.0 <= test_fraction and val_fraction + test_fraction < 1.0):
        raise ValueError("val_fraction + test_fraction must be in [0, 1)")
    h = hashlib.blake2b(normalized_key(text), digest_size=8, salt=seed.to_bytes(8, "little", signed=False)[:8])
    u = int.from_bytes(h.digest(), "little") / 2**64
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "val"
    return "train"
