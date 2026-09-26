"""
aarav-lm · Accounts, API keys and usage records (SQLite)

    users      one row per person (an email is the only personal datum kept)
    api_keys   the SHA-256 of each key, never the key; a short prefix to recognise it in a list; limits, expiry, revocation
    usage      one row per generation request: who, which model and endpoint, token counts, time taken, outcome
    models     the models this server has served (name, parameters, checkpoint) so usage rows stay interpretable

Keys are 256-bit random tokens shown once at creation, so a plain SHA-256 is the right hash (nothing to brute-force;
a slow hash would only add latency to every request). Prompts and completions are **not** stored, only counts.
The file is created with mode 0600. SQLite in WAL mode handles one server process; the same schema moves to Postgres
when there is more than one (docs/AUDIT.md, phase 5).

`AccountStore` is a drop-in for the older JSON `KeyStore` (`verify`, `__len__`) and adds `lookup`, which returns the key's
limits so the server can enforce them.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from aarav_lm.api.auth import KEY_PREFIX

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    disabled INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    prefix TEXT NOT NULL,
    key_sha256 TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    expires_at TEXT,
    revoked_at TEXT,
    rotated_from INTEGER REFERENCES api_keys(id),
    rpm_limit REAL,
    tpm_limit INTEGER
);
CREATE INDEX IF NOT EXISTS api_keys_user ON api_keys(user_id);
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    checkpoint TEXT,
    parameters INTEGER,
    context_length INTEGER,
    first_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    user_id INTEGER,
    key_id INTEGER,
    model_id TEXT,
    endpoint TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    status INTEGER NOT NULL,
    error_type TEXT
);
CREATE INDEX IF NOT EXISTS usage_key_ts ON usage(key_id, ts);
"""

LAST_USED_GRANULARITY_S = 60          # writing last_used_at on every request would make every read a write


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


def digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Principal:
    """What the server needs to know about an authenticated key."""
    key_id: int
    user_id: int
    label: str
    rpm_limit: Optional[float]
    tpm_limit: Optional[int]


@dataclass(frozen=True)
class KeyInfo:
    """A key as listed to its owner: never the key itself."""
    id: int
    user_id: int
    prefix: str
    label: str
    created_at: str
    last_used_at: Optional[str]
    expires_at: Optional[str]
    revoked_at: Optional[str]
    rotated_from: Optional[int]
    rpm_limit: Optional[float]
    tpm_limit: Optional[int]

    @property
    def active(self) -> bool:
        return self.revoked_at is None and (self.expires_at is None or self.expires_at > _iso(_now()))


class AccountStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        if new:                                            # create private from the start, not chmod afterwards
            os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o600))
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._last_used_cache: dict = {}

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            else:
                self._db.execute("COMMIT")

    # ── users ───────────────────────────────────────────────────────────────
    def create_user(self, email: str) -> int:
        email = email.strip().lower()
        if "@" not in email or len(email) > 254:
            raise ValueError("a valid email address is required")
        try:
            with self._tx() as db:
                return db.execute("INSERT INTO users(email, created_at) VALUES (?, ?)", (email, _iso(_now()))).lastrowid
        except sqlite3.IntegrityError:
            raise ValueError(f"a user with email {email} already exists") from None

    def get_user(self, email: str) -> Optional[int]:
        row = self._db.execute("SELECT id FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
        return row["id"] if row else None

    def set_user_disabled(self, user_id: int, disabled: bool = True) -> None:
        with self._tx() as db:
            if db.execute("UPDATE users SET disabled = ? WHERE id = ?", (int(disabled), user_id)).rowcount == 0:
                raise KeyError(f"no user {user_id}")

    # ── keys ────────────────────────────────────────────────────────────────
    def create_key(self, user_id: int, label: str = "", rpm_limit: Optional[float] = None,
                   tpm_limit: Optional[int] = None, expires_days: Optional[float] = None,
                   _rotated_from: Optional[int] = None) -> str:
        """Create a key for `user_id` and return the plaintext, which is never stored and cannot be recovered."""
        with self._tx() as db:
            return self._insert_key(db, user_id, label, rpm_limit, tpm_limit, expires_days, _rotated_from)

    def _insert_key(self, db: sqlite3.Connection, user_id: int, label: str, rpm_limit: Optional[float],
                    tpm_limit: Optional[int], expires_days: Optional[float], rotated_from: Optional[int]) -> str:
        """Runs inside the caller's transaction (so `rotate_key` can create and revoke atomically)."""
        if rpm_limit is not None and rpm_limit <= 0:
            raise ValueError("rpm_limit must be positive")
        if tpm_limit is not None and tpm_limit <= 0:
            raise ValueError("tpm_limit must be positive")
        if db.execute("SELECT 1 FROM users WHERE id = ? AND disabled = 0", (user_id,)).fetchone() is None:
            raise KeyError(f"no active user {user_id}")
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        expires = _iso(_now() + timedelta(days=expires_days)) if expires_days else None
        db.execute(
            "INSERT INTO api_keys(user_id, prefix, key_sha256, label, created_at, expires_at, rotated_from, rpm_limit, tpm_limit)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, key[:len(KEY_PREFIX) + 4], digest(key), label, _iso(_now()), expires, rotated_from, rpm_limit, tpm_limit))
        return key

    def _key_row(self, key_id: int) -> sqlite3.Row:
        row = self._db.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
        if row is None:
            raise KeyError(f"no key {key_id}")
        return row

    def list_keys(self, user_id: Optional[int] = None) -> List[KeyInfo]:
        q, args = "SELECT * FROM api_keys", ()
        if user_id is not None:
            q, args = q + " WHERE user_id = ?", (user_id,)
        return [KeyInfo(r["id"], r["user_id"], r["prefix"] + "…", r["label"], r["created_at"], r["last_used_at"],
                        r["expires_at"], r["revoked_at"], r["rotated_from"], r["rpm_limit"], r["tpm_limit"])
                for r in self._db.execute(q + " ORDER BY id", args)]

    def revoke_key(self, key_id: int) -> None:
        with self._tx() as db:
            row = self._key_row(key_id)
            if row["revoked_at"] is None:
                db.execute("UPDATE api_keys SET revoked_at = ? WHERE id = ?", (_iso(_now()), key_id))

    def rotate_key(self, key_id: int) -> str:
        """Issue a replacement with the same owner, label and limits, and revoke the old key, in one transaction."""
        with self._tx() as db:
            old = self._key_row(key_id)
            if old["revoked_at"] is not None:
                raise ValueError(f"key {key_id} is already revoked")
            expires_days = None
            if old["expires_at"]:
                left = (datetime.fromisoformat(old["expires_at"]) - _now()).total_seconds() / 86400
                expires_days = max(left, 1 / 24)
            new = self._insert_key(db, old["user_id"], old["label"], old["rpm_limit"], old["tpm_limit"], expires_days, key_id)
            db.execute("UPDATE api_keys SET revoked_at = ? WHERE id = ?", (_iso(_now()), key_id))
        return new

    # ── authentication (what the server calls per request) ──────────────────
    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM api_keys WHERE revoked_at IS NULL").fetchone()[0]

    def lookup(self, token: Optional[str]) -> Optional[Principal]:
        """The key's owner and limits if `token` is a valid, unrevoked, unexpired key of an enabled user, else None."""
        if not token or not token.startswith(KEY_PREFIX):
            return None
        want = digest(token)
        with self._lock:
            row = self._db.execute(
                "SELECT k.id, k.user_id, k.label, k.key_sha256, k.expires_at, k.revoked_at, k.rpm_limit, k.tpm_limit, u.disabled "
                "FROM api_keys k JOIN users u ON u.id = k.user_id WHERE k.key_sha256 = ?", (want,)).fetchone()
            if row is None or not hmac.compare_digest(row["key_sha256"], want):
                return None
            if row["revoked_at"] is not None or row["disabled"]:
                return None
            if row["expires_at"] is not None and row["expires_at"] <= _iso(_now()):
                return None
            self._touch(row["id"])
            return Principal(row["id"], row["user_id"], row["label"], row["rpm_limit"], row["tpm_limit"])

    def verify(self, token: Optional[str]) -> Optional[str]:
        """JSON-KeyStore-compatible: the key id as a string, or None."""
        p = self.lookup(token)
        return None if p is None else str(p.key_id)

    def _touch(self, key_id: int) -> None:
        now = time.monotonic()
        if now - self._last_used_cache.get(key_id, -1e9) < LAST_USED_GRANULARITY_S:
            return
        self._last_used_cache[key_id] = now
        self._db.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (_iso(_now()), key_id))

    # ── models and usage ────────────────────────────────────────────────────
    def register_model(self, model_id: str, checkpoint: str = "", parameters: int = 0, context_length: int = 0) -> None:
        with self._tx() as db:
            db.execute("INSERT OR IGNORE INTO models(id, checkpoint, parameters, context_length, first_seen) VALUES (?,?,?,?,?)",
                       (model_id, checkpoint, parameters, context_length, _iso(_now())))

    def record_usage(self, *, key_id: Optional[int], user_id: Optional[int], model_id: str, endpoint: str,
                     prompt_tokens: int = 0, completion_tokens: int = 0, duration_ms: int = 0, status: int = 200,
                     error_type: Optional[str] = None) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT INTO usage(ts, user_id, key_id, model_id, endpoint, prompt_tokens, completion_tokens, duration_ms, status, error_type)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (_iso(_now()), user_id, key_id, model_id, endpoint, prompt_tokens, completion_tokens, duration_ms, status, error_type))

    def usage_summary(self, *, key_id: Optional[int] = None, user_id: Optional[int] = None, days: float = 30.0) -> dict:
        since = _iso(_now() - timedelta(days=days))
        where, args = ["ts >= ?"], [since]
        if key_id is not None:
            where.append("key_id = ?"); args.append(key_id)
        if user_id is not None:
            where.append("user_id = ?"); args.append(user_id)
        row = self._db.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(prompt_tokens),0) p, COALESCE(SUM(completion_tokens),0) c, "
            "COALESCE(SUM(status >= 400),0) errors, COALESCE(AVG(duration_ms),0) ms FROM usage WHERE " + " AND ".join(where), args).fetchone()
        return {"days": days, "requests": row["n"], "prompt_tokens": row["p"], "completion_tokens": row["c"],
                "total_tokens": row["p"] + row["c"], "errors": row["errors"], "mean_duration_ms": round(row["ms"], 1)}

    def usage_rows(self, key_id: int, limit: int = 20) -> List[dict]:
        return [dict(r) for r in self._db.execute("SELECT * FROM usage WHERE key_id = ? ORDER BY id DESC LIMIT ?", (key_id, limit))]

    def tokens_used_since(self, key_id: int, seconds: float) -> int:
        since = _iso(_now() - timedelta(seconds=seconds))
        return self._db.execute("SELECT COALESCE(SUM(prompt_tokens + completion_tokens),0) FROM usage WHERE key_id = ? AND ts >= ?",
                                (key_id, since)).fetchone()[0]

    # ── migration from the JSON key file ────────────────────────────────────
    def import_json_keys(self, json_path: str | Path, email: str = "owner@localhost") -> int:
        """Copy the hashes from a legacy `secrets/api_keys.json` under one owner. Keys keep working; returns how many."""
        import json
        entries = json.loads(Path(json_path).read_text(encoding="utf-8"))
        uid = self.get_user(email) or self.create_user(email)
        n = 0
        with self._tx() as db:
            for e in entries:
                if db.execute("SELECT 1 FROM api_keys WHERE key_sha256 = ?", (e["sha256"],)).fetchone():
                    continue
                db.execute("INSERT INTO api_keys(user_id, prefix, key_sha256, label, created_at) VALUES (?,?,?,?,?)",
                           (uid, KEY_PREFIX + "(json)", e["sha256"], e.get("label", ""), e.get("created", _iso(_now()))))
                n += 1
        return n
