"""
aarav-lm · API authentication and rate limiting

Keys are shown once at creation and only their SHA-256 is stored, so the key file
leaking does not leak usable keys. Keys are high-entropy random tokens, so a plain
SHA-256 (no salt or slow hash) is appropriate; comparison is constant-time.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

KEY_PREFIX = "aak_"


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class KeyStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> List[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{self.path} is not valid JSON: {e}") from e
        if not isinstance(data, list):
            raise ValueError(f"{self.path}: expected a list of keys")
        return data

    def __len__(self) -> int:
        return len(self._read())

    def create(self, label: str = "") -> str:
        """Generate a key, store its hash, and return the plaintext (the only time it is visible)."""
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        with self._lock:
            entries = self._read()
            entries.append({
                "id": _digest(key)[:8], "sha256": _digest(key), "label": label,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2)
            os.replace(tmp, self.path)
        return key

    def verify(self, token: Optional[str]) -> Optional[str]:
        """Return the key id if `token` is valid, else None. Checks every entry (no early exit)."""
        if not token or not token.startswith(KEY_PREFIX):
            return None
        want = _digest(token)
        found = None
        for e in self._read():
            if hmac.compare_digest(e["sha256"], want):
                found = e["id"]
        return found


class RateLimiter:
    """Per-identity token bucket: `burst` requests at once, refilled at `per_minute`/60 per second."""

    def __init__(self, per_minute: float, burst: int, clock=time.monotonic):
        if per_minute <= 0 or burst < 1:
            raise ValueError("per_minute must be > 0 and burst >= 1")
        self.rate = per_minute / 60.0
        self.burst = burst
        self._clock = clock
        self._buckets: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def allow(self, identity: str) -> tuple[bool, float]:
        """(allowed, retry_after_seconds)."""
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(identity, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[identity] = (tokens - 1.0, now)
                return True, 0.0
            self._buckets[identity] = (tokens, now)
            return False, (1.0 - tokens) / self.rate
