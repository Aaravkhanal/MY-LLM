"""Accounts, keys and usage in SQLite: keys are never stored, and every lifecycle rule is enforced."""

import os
import sqlite3
import stat
import time

import pytest

from aarav_lm.api.auth import KEY_PREFIX, KeyStore
from aarav_lm.api.store import AccountStore, digest


@pytest.fixture()
def store(tmp_path):
    s = AccountStore(tmp_path / "accounts.db")
    yield s
    s.close()


@pytest.fixture()
def alice(store):
    return store.create_user("Alice@Example.com")


def test_keys_are_hashed_never_stored_and_the_file_is_private(store, alice, tmp_path):
    key = store.create_key(alice, "laptop")
    assert key.startswith(KEY_PREFIX) and len(key) > 40
    raw = open(tmp_path / "accounts.db", "rb").read() + (open(tmp_path / "accounts.db-wal", "rb").read() if (tmp_path / "accounts.db-wal").exists() else b"")
    assert key.encode() not in raw and key[len(KEY_PREFIX):].encode() not in raw           # the secret is nowhere on disk
    row = sqlite3.connect(tmp_path / "accounts.db").execute("SELECT key_sha256, prefix FROM api_keys").fetchone()
    assert row[0] == digest(key) and row[1] == key[:8] and key.startswith(row[1])
    assert stat.S_IMODE(os.stat(tmp_path / "accounts.db").st_mode) == 0o600
    info = store.list_keys(alice)[0]
    assert info.prefix.endswith("…") and key not in repr(info)                              # listings never contain the key


def test_valid_key_authenticates_and_carries_its_limits(store, alice):
    key = store.create_key(alice, "svc", rpm_limit=30, tpm_limit=5000)
    p = store.lookup(key)
    assert p.user_id == alice and p.label == "svc" and p.rpm_limit == 30 and p.tpm_limit == 5000
    assert store.verify(key) == str(p.key_id)
    for bad in (None, "", "aak_nope", "sk-other", key + "x", key[:-1]):
        assert store.lookup(bad) is None


def test_revoked_expired_and_disabled_keys_stop_working(store, alice):
    k1, k2, k3 = (store.create_key(alice, n) for n in ("a", "b", "c"))
    store.revoke_key(store.lookup(k1).key_id)
    assert store.lookup(k1) is None and store.lookup(k2) is not None
    store.revoke_key(store.lookup(k2).key_id)
    store.revoke_key(store.list_keys()[1].id)                                                # revoking twice is harmless
    exp = store.create_key(alice, "short-lived", expires_days=1e-6)
    time.sleep(1.2)
    assert store.lookup(exp) is None
    store.set_user_disabled(alice)
    assert store.lookup(k3) is None                                                          # disabling a user disables all their keys
    with pytest.raises(KeyError):
        store.create_key(alice)
    store.set_user_disabled(alice, False)
    assert store.lookup(k3) is not None
    with pytest.raises(KeyError):
        store.revoke_key(9999)


def test_rotation_issues_a_replacement_with_the_same_limits_and_kills_the_old_key(store, alice):
    old = store.create_key(alice, "svc", rpm_limit=12, tpm_limit=999, expires_days=30)
    old_id = store.lookup(old).key_id
    new = store.rotate_key(old_id)
    assert new != old and store.lookup(old) is None
    p = store.lookup(new)
    assert (p.label, p.rpm_limit, p.tpm_limit) == ("svc", 12, 999)
    listed = {k.id: k for k in store.list_keys(alice)}
    assert listed[old_id].revoked_at and listed[p.key_id].rotated_from == old_id and listed[p.key_id].expires_at
    with pytest.raises(ValueError, match="already revoked"):
        store.rotate_key(old_id)
    assert len([k for k in store.list_keys(alice) if k.active]) == 1


def test_rotation_is_atomic_when_it_fails_midway(store, alice, monkeypatch):
    key = store.create_key(alice, "svc")
    kid = store.lookup(key).key_id
    monkeypatch.setattr(store, "_insert_key", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    with pytest.raises(RuntimeError):
        store.rotate_key(kid)
    assert store.lookup(key) is not None                                                     # the old key was NOT revoked


def test_user_and_limit_validation(store, alice):
    with pytest.raises(ValueError, match="already exists"):
        store.create_user("alice@example.com")                                               # emails are case-insensitive
    for bad in ("", "no-at-sign", "a" * 300 + "@x.com"):
        with pytest.raises(ValueError):
            store.create_user(bad)
    for kw in ({"rpm_limit": 0}, {"tpm_limit": -5}):
        with pytest.raises(ValueError):
            store.create_key(alice, **kw)
    assert store.get_user("ALICE@example.com") == alice and store.get_user("nobody@x.com") is None


def test_last_used_is_updated_but_not_on_every_request(store, alice):
    key = store.create_key(alice)
    assert store.list_keys()[0].last_used_at is None
    store.lookup(key)
    first = store.list_keys()[0].last_used_at
    assert first is not None
    store.lookup(key); store.lookup(key)
    assert store.list_keys()[0].last_used_at == first                                       # throttled: reads are not writes


def test_usage_is_recorded_and_summarised_per_key_and_user(store, alice):
    bob = store.create_user("bob@example.com")
    ka, kb = store.create_key(alice), store.create_key(bob)
    ia, ib = store.lookup(ka).key_id, store.lookup(kb).key_id
    store.register_model("exp_x", "x.pt", 14_000_000, 512)
    store.register_model("exp_x")                                                            # idempotent
    for _ in range(3):
        store.record_usage(key_id=ia, user_id=alice, model_id="exp_x", endpoint="/v1/chat/completions",
                           prompt_tokens=10, completion_tokens=20, duration_ms=100)
    store.record_usage(key_id=ia, user_id=alice, model_id="exp_x", endpoint="/chat", status=500, error_type="generation_failed")
    store.record_usage(key_id=ib, user_id=bob, model_id="exp_x", endpoint="/generate", prompt_tokens=5, completion_tokens=5)
    a = store.usage_summary(key_id=ia)
    assert (a["requests"], a["prompt_tokens"], a["completion_tokens"], a["total_tokens"], a["errors"]) == (4, 30, 60, 90, 1)
    assert store.usage_summary(user_id=bob)["total_tokens"] == 10 and store.usage_summary()["requests"] == 5
    assert store.tokens_used_since(ia, 60) == 90 and store.tokens_used_since(ib, 60) == 10
    assert store.usage_rows(ia, 2)[0]["error_type"] == "generation_failed"


def test_the_store_is_a_drop_in_for_the_json_keystore_and_can_import_it(store, tmp_path):
    js = KeyStore(tmp_path / "keys.json")
    legacy = js.create("old laptop")
    assert store.import_json_keys(tmp_path / "keys.json") == 1
    assert store.import_json_keys(tmp_path / "keys.json") == 0                               # idempotent
    assert store.lookup(legacy) is not None and store.verify(legacy) is not None            # the old key still works
    assert len(store) == 1
    assert js.verify(legacy) is not None                                                     # and the JSON store is untouched


def test_concurrent_use_from_many_threads_is_safe(store, alice):
    import threading
    key = store.create_key(alice)
    kid = store.lookup(key).key_id
    errors = []

    def work():
        try:
            for _ in range(25):
                assert store.lookup(key) is not None
                store.record_usage(key_id=kid, user_id=alice, model_id="m", endpoint="/x", prompt_tokens=1, completion_tokens=1)
        except Exception as e:                                                               # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert errors == [] and store.usage_summary(key_id=kid)["requests"] == 200


# ── web accounts: passwords and sessions ─────────────────────────────────────

def test_passwords_are_salted_scrypt_hashes_and_never_stored(store, tmp_path):
    from aarav_lm.api.store import hash_password, verify_password
    uid = store.signup("New@Example.com", "correct horse battery")
    row = sqlite3.connect(tmp_path / "accounts.db").execute("SELECT password_hash FROM users WHERE id = ?", (uid,)).fetchone()[0]
    assert row.startswith("scrypt$") and "correct horse" not in row
    assert hash_password("x" * 10) != hash_password("x" * 10)                                # random salt
    assert verify_password("correct horse battery", row) and not verify_password("correct horse batterY", row)
    assert store.check_password("new@example.com", "correct horse battery") == uid
    assert store.check_password("NEW@example.com", "wrong password!!") is None
    assert store.check_password("nobody@example.com", "correct horse battery") is None


def test_signup_validation_and_duplicates(store):
    for bad_email in ("", "nope", "a@b", "a b@c.com", "x" * 260 + "@y.com"):
        with pytest.raises(ValueError, match="email"):
            store.signup(bad_email, "long enough password")
    for bad_pw in ("short", "x" * 129, ""):
        with pytest.raises(ValueError, match="password"):
            store.signup("ok@example.com", bad_pw)
    store.signup("ok@example.com", "long enough password")
    with pytest.raises(ValueError, match="already exists"):
        store.signup("OK@example.com", "another long password")


def test_unknown_email_takes_as_long_as_a_wrong_password(store):
    store.signup("real@example.com", "long enough password")
    def timed(email):
        t = time.perf_counter()
        for _ in range(4):
            store.check_password(email, "wrong wrong wrong")
        return time.perf_counter() - t
    known, unknown = timed("real@example.com"), timed("ghost@example.com")
    assert unknown > 0.5 * known                                                              # both do the full scrypt work


def test_sessions_store_only_a_hash_expire_and_die_with_the_user(store):
    uid = store.signup("s@example.com", "long enough password")
    token, csrf = store.create_session(uid)
    info = store.get_session(token)
    assert (info.user_id, info.email, info.csrf) == (uid, "s@example.com", csrf)
    raw = store._db.execute("SELECT token_sha256 FROM sessions").fetchone()[0]
    assert raw == digest(token) and token not in raw
    assert store.get_session("nope") is None and store.get_session(None) is None
    short, _ = store.create_session(uid, ttl_days=1e-9)
    time.sleep(1.1)
    assert store.get_session(short) is None
    other, _ = store.create_session(uid)
    store.end_sessions(uid, keep_token=token)
    assert store.get_session(token) is not None and store.get_session(other) is None
    store.set_user_disabled(uid)
    assert store.get_session(token) is None and store.check_password("s@example.com", "long enough password") is None
    store.set_user_disabled(uid, False)
    store.end_session(token)
    assert store.get_session(token) is None


def test_password_change_and_a_database_from_before_web_accounts_still_opens(tmp_path):
    db = sqlite3.connect(tmp_path / "old.db")
    db.executescript("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,"
                     " disabled INTEGER NOT NULL DEFAULT 0); INSERT INTO users(email, created_at) VALUES ('old@example.com', 'x');")
    db.commit(); db.close()
    s = AccountStore(tmp_path / "old.db")                                                   # migrates in place
    uid = s.get_user("old@example.com")
    assert s.check_password("old@example.com", "anything long") is None                     # no password yet: cannot log in
    s.set_password(uid, "a brand new password")
    assert s.check_password("old@example.com", "a brand new password") == uid
    s.close()


def test_usage_daily_is_zero_filled_and_per_user(store):
    a, b = store.signup("a@example.com", "long enough password"), store.signup("b@example.com", "long enough password")
    store.record_usage(key_id=None, user_id=a, model_id="m", endpoint="/x", prompt_tokens=3, completion_tokens=4)
    store.record_usage(key_id=None, user_id=b, model_id="m", endpoint="/x", prompt_tokens=100, completion_tokens=100)
    days = store.usage_daily(a, 7)
    assert len(days) == 7 and sum(d["tokens"] for d in days) == 7 and days[-1]["requests"] == 1 and days[0]["requests"] == 0
    assert store.tokens_used_by_user_since(a, 60) == 7 and store.count_active_keys(a) == 0
