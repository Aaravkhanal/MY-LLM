"""
aarav-lm · Duplicate removal

Exact de-duplication on a normalised form (case-folded, whitespace-collapsed), so
"Hello  world" and "hello world" count as the same document. Only 8-byte digests are
kept in memory. Near-duplicate detection (MinHash/LSH) is a later upgrade; on
templated corpora like TinyStories it would need careful thresholds to avoid
dropping legitimately similar-but-distinct stories.
"""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")


def normalized_key(text: str) -> bytes:
    """Stable 8-byte digest of the case-folded, whitespace-collapsed text."""
    canon = _WS.sub(" ", text.casefold()).strip()
    return hashlib.blake2b(canon.encode("utf-8"), digest_size=8).digest()


class ExactDeduper:
    def __init__(self) -> None:
        self._seen: set[bytes] = set()

    def is_duplicate(self, text: str) -> bool:
        """True if an equivalent document was already seen; records it otherwise."""
        key = normalized_key(text)
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    def __len__(self) -> int:
        return len(self._seen)
