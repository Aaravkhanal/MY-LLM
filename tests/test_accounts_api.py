"""The server with the account database: revocation, per-key limits, and one usage row per request."""

import json

import pytest
from fastapi.testclient import TestClient

from aarav_lm.api.app import create_app
from aarav_lm.api.store import AccountStore


@pytest.fixture()
def accounts(tmp_path):
    s = AccountStore(tmp_path / "accounts.db")
    yield s
    s.close()


@pytest.fixture()
def server(engine, accounts):
    uid = accounts.create_user("dev@example.com")
    accounts.create_key(uid, "bootstrap")                    # the server refuses to start with an empty key database
    # A generous queue timeout: these tests are sequential, so a 503 would only mean a slow machine (e.g. while training).
    return TestClient(create_app(engine, accounts, rate_limit_per_min=600, rate_limit_burst=100, queue_timeout=5.0,
                                 model_name="exp_test")), accounts, uid


def auth(key):
    return {"Authorization": f"Bearer {key}"}


def chat(client, key, **kw):
    body = {"model": "exp_test", "messages": [{"role": "user", "content": "Once upon a time"}], "max_tokens": 6,
            "temperature": 0, **kw}
    return client.post("/v1/chat/completions", headers=auth(key), json=body)


def rows(accounts, key):
    return accounts.usage_rows(accounts.lookup(key).key_id, 50)[::-1]


def test_account_keys_authenticate_and_the_model_is_registered(server):
    client, accounts, uid = server
    key = accounts.create_key(uid, "svc")
    assert client.get("/v1/models", headers=auth(key)).status_code == 200
    assert accounts._db.execute("SELECT id, parameters FROM models").fetchone()["id"] == "exp_test"
    assert client.get("/v1/models", headers=auth("aak_wrong")).status_code == 401


def test_revocation_and_rotation_take_effect_on_the_very_next_request(server):
    client, accounts, uid = server
    key = accounts.create_key(uid)
    assert chat(client, key).status_code == 200
    new = accounts.rotate_key(accounts.lookup(key).key_id)
    assert chat(client, key).status_code == 401 and chat(client, new).status_code == 200
    accounts.revoke_key(accounts.lookup(new).key_id)
    assert chat(client, new).status_code == 401
    live = accounts.create_key(uid, "live")
    assert chat(client, live).status_code == 200
    accounts.set_user_disabled(uid)                               # disabling the owner cuts off all of their keys at once
    assert chat(client, live).status_code == 401


def test_one_usage_row_per_request_with_endpoint_tokens_and_status(server):
    client, accounts, uid = server
    key = accounts.create_key(uid)
    r = chat(client, key)
    u = r.json()["usage"]
    responses = [r,
                 client.post("/generate", headers=auth(key), json={"prompt": "Once upon a time", "max_tokens": 5, "temperature": 0}),
                 client.post("/chat", headers=auth(key), json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}),
                 client.post("/v1/completions", headers=auth(key), json={"model": "exp_test", "prompt": "The dog", "max_tokens": 4})]
    assert [x.status_code for x in responses] == [200] * 4, [x.text for x in responses]
    got = rows(accounts, key)
    assert [g["endpoint"] for g in got] == ["/v1/chat/completions", "/generate", "/chat", "/v1/completions"]
    assert all(g["status"] == 200 and g["error_type"] is None and g["model_id"] == "exp_test" for g in got)
    assert (got[0]["prompt_tokens"], got[0]["completion_tokens"]) == (u["prompt_tokens"], u["completion_tokens"])
    for row, resp in zip(got, responses):                    # every row records exactly what its own response reported
        usage = resp.json()["usage"]
        assert (row["prompt_tokens"], row["completion_tokens"]) == (usage["prompt_tokens"], usage["completion_tokens"])
        assert row["duration_ms"] >= 0                       # (a reply may legitimately be empty: unseeded sampling can end at once)


def test_streaming_requests_are_recorded_once_with_their_real_token_counts(server):
    client, accounts, uid = server
    key = accounts.create_key(uid)
    r = chat(client, key, stream=True, max_tokens=8)
    frames = [f for f in r.text.split("\n\n") if f.startswith("data: {")]
    n_chunks = len(frames) - 2                                                # role frame and finish frame
    got = rows(accounts, key)
    assert len(got) == 1 and got[0]["status"] == 200 and got[0]["error_type"] is None
    assert 1 <= got[0]["completion_tokens"] <= 8 and n_chunks >= 1


def test_usage_endpoint_reports_only_the_callers_own_key(server):
    client, accounts, uid = server
    other = accounts.create_user("other@example.com")
    ka, kb = accounts.create_key(uid), accounts.create_key(other)
    for _ in range(3):
        chat(client, ka)
    chat(client, kb)
    mine = client.get("/v1/usage", headers=auth(ka)).json()
    theirs = client.get("/v1/usage?days=7", headers=auth(kb)).json()
    assert mine["object"] == "usage" and mine["requests"] == 3 and theirs["requests"] == 1
    assert mine["total_tokens"] == mine["prompt_tokens"] + mine["completion_tokens"] > 0
    assert client.get("/v1/usage?days=0", headers=auth(ka)).status_code == 400
    assert client.get("/v1/usage").status_code == 401


def test_a_per_key_request_limit_only_throttles_that_key_and_is_recorded(server):
    client, accounts, uid = server
    slow, fast = accounts.create_key(uid, "slow", rpm_limit=0.5), accounts.create_key(uid, "fast")
    limiter_burst_ok = [chat(client, slow).status_code for _ in range(1)]
    assert limiter_burst_ok == [200]
    # the bucket holds `burst` (100) tokens, so exhaust it: a key limited to 0.5/min refills 100x slower than the default
    codes = [chat(client, slow).status_code for _ in range(101)]
    assert 429 in codes and chat(client, fast).status_code == 200
    limited = [r for r in rows(accounts, slow) if r["status"] == 429]
    assert limited and limited[0]["error_type"] == "rate_limited" and limited[0]["completion_tokens"] == 0


def test_a_token_limit_blocks_the_next_request_after_it_is_exceeded(server):
    client, accounts, uid = server
    key = accounts.create_key(uid, "capped", tpm_limit=5)
    first = chat(client, key)
    assert first.status_code == 200 and first.json()["usage"]["total_tokens"] > 5
    second = chat(client, key)
    assert second.status_code == 429 and second.headers["retry-after"] == "30"
    assert second.json()["error"]["type"] == "rate_limit_error" and "token limit" in second.json()["error"]["message"]
    assert [r["error_type"] for r in rows(accounts, key)] == [None, "token_limit"]
    assert chat(client, accounts.create_key(uid, "uncapped")).status_code == 200


def test_a_busy_model_is_a_recorded_503(server, engine):
    client, accounts, uid = server
    key = accounts.create_key(uid)
    busy = TestClient(create_app(engine, accounts, rate_limit_per_min=600, rate_limit_burst=100, queue_timeout=0.2,
                                 model_name="exp_test"))
    engine.lock.acquire()
    try:
        r = chat(busy, key)
    finally:
        engine.lock.release()
    assert r.status_code == 503
    got = rows(accounts, key)
    assert len(got) == 1 and got[0]["status"] == 503 and got[0]["error_type"] == "busy"


def test_a_failing_usage_database_never_fails_the_request_and_is_counted(server, monkeypatch):
    client, accounts, uid = server
    key = accounts.create_key(uid)
    monkeypatch.setattr(accounts, "record_usage", lambda **kw: (_ for _ in ()).throw(RuntimeError("disk full")))
    assert chat(client, key).status_code == 200
    assert client.get("/stats", headers=auth(key)).json()["usage_write_failures"] == 1


def test_the_json_keystore_still_works_and_reports_that_usage_needs_the_database(engine, tmp_path):
    from aarav_lm.api.auth import KeyStore
    store = KeyStore(tmp_path / "keys.json")
    key = store.create("legacy")
    client = TestClient(create_app(engine, store, model_name="exp_test", rate_limit_per_min=600, rate_limit_burst=100))
    assert chat(client, key).status_code == 200
    r = client.get("/v1/usage", headers=auth(key))
    assert r.status_code == 404 and "account database" in r.json()["error"]["message"]


def test_nothing_secret_or_private_is_stored_in_usage_rows(server):
    client, accounts, uid = server
    key = accounts.create_key(uid)
    chat(client, key, messages=[{"role": "user", "content": "my secret prompt text"}])
    dump = json.dumps([dict(r) for r in accounts._db.execute("SELECT * FROM usage")])
    assert "secret prompt" not in dump and key not in dump


def test_a_request_over_its_time_budget_is_cut_off_returned_partial_and_recorded(engine, accounts):
    uid = accounts.create_user("t@example.com")
    key = accounts.create_key(uid)
    client = TestClient(create_app(engine, accounts, model_name="exp_test", request_timeout=1e-9, queue_timeout=5.0,
                                   rate_limit_per_min=600, rate_limit_burst=100))
    native = client.post("/generate", headers=auth(key), json={"prompt": "Once upon a time", "max_tokens": 50, "temperature": 0})
    assert native.status_code == 200 and native.json()["finish_reason"] == "timeout"
    oa = chat(client, key, max_tokens=50)
    assert oa.status_code == 200 and oa.json()["choices"][0]["finish_reason"] == "length"          # a valid OpenAI value
    frames = [json.loads(f[6:]) for f in chat(client, key, stream=True, max_tokens=50).text.split("\n\n") if f.startswith("data: {")]
    assert frames[-1]["choices"][0]["finish_reason"] == "length"
    got = rows(accounts, key)
    assert len(got) == 3 and all(g["status"] == 200 and g["error_type"] == "timeout" for g in got)
    assert client.get("/health").status_code == 200
    # the model slot is free again: an untimed server on the same engine still answers
    ok = TestClient(create_app(engine, accounts, model_name="exp_test", request_timeout=0, rate_limit_per_min=600, rate_limit_burst=100))
    assert chat(ok, key).json()["choices"][0]["finish_reason"] in ("stop", "length")
