"""The server with --batch-size: concurrent users share decode steps, and nothing else about the API changes."""

import http.client
import importlib.util
import json
import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from aarav_lm.api.app import create_app
from aarav_lm.api.auth import KeyStore
from aarav_lm.api.store import AccountStore

spec = importlib.util.spec_from_file_location("loadtest", Path(__file__).resolve().parents[1] / "scripts" / "loadtest.py")
loadtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loadtest)


@pytest.fixture()
def serve(engine, tmp_path):
    import uvicorn
    store = KeyStore(tmp_path / "k.json")
    key = store.create("b")
    started = []

    def start(store_=None, **kw):
        kw.setdefault("queue_timeout", 5.0)
        app = create_app(engine, store_ or store, rate_limit_per_min=600000, rate_limit_burst=10000, model_name="exp_test", **kw)
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        t = threading.Thread(target=srv.run, daemon=True); t.start()
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
            except OSError:
                time.sleep(0.05)
        started.append((srv, t))
        return f"http://127.0.0.1:{port}", port, key
    yield start
    for srv, t in started:
        srv.should_exit = True; t.join(5)


def post(port, key, body, path="/v1/chat/completions"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("POST", path, json.dumps(body), {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, json.loads(r.read()), dict(r.getheaders())


def stats(port, key):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/stats", headers={"Authorization": f"Bearer {key}"})
    return json.loads(c.getresponse().read())


def body(i, **kw):
    return {"model": "exp_test", "max_tokens": 8, "temperature": 0.9, "seed": 40 + i, "messages": [{"role": "user", "content": f"Once upon a time {'the dog ' * i}"}], **kw}


def test_concurrent_users_share_waves_and_get_the_same_answers_as_the_unbatched_server(serve):
    _, plain_port, key = serve()
    want = {i: post(plain_port, key, body(i))[1]["choices"][0]["message"]["content"] for i in range(6)}
    _, port, key = serve(batch_size=8, batch_wait_ms=150)
    got, errors = {}, []

    def user(i):
        try:
            status, data, _ = post(port, key, body(i)); assert status == 200; got[i] = data["choices"][0]["message"]["content"]
        except Exception as e:                                                        # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=user, args=(i,)) for i in range(6)]
    [t.start() for t in threads]; [t.join(30) for t in threads]
    assert errors == [] and got == want
    b = stats(port, key)["batching"]
    assert b["largest_wave"] > 1 and b["waves"] < 6 and b["batched_requests"] == 6


def test_streaming_through_the_batcher_reassembles_to_the_non_streamed_answer(serve):
    _, port, key = serve(batch_size=4, batch_wait_ms=20)
    whole = post(port, key, body(2))[1]["choices"][0]["message"]["content"]
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("POST", "/v1/chat/completions", json.dumps(body(2, stream=True)), {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    text = c.getresponse().read().decode()
    frames = [f for f in text.split("\n\n") if f]
    assert frames[-1] == "data: [DONE]"
    joined = "".join(json.loads(f[6:])["choices"][0]["delta"].get("content") or "" for f in frames[:-1])
    assert joined == whole


def test_a_load_test_against_the_batching_server_serves_everyone_without_faults(serve):
    url, port, key = serve(batch_size=8, batch_wait_ms=10)
    r = loadtest.run(url, key, concurrency=6, requests=24, max_tokens=8, model="exp_test")
    assert r["statuses"] == {"200": 24} and r["faults"] == []
    assert stats(port, key)["batching"]["largest_wave"] > 1


def test_a_full_queue_is_a_clean_503_with_retry_after_and_a_usage_row(serve, engine, tmp_path):
    accounts = AccountStore(tmp_path / "a.db")
    uid = accounts.create_user("q@example.com")
    k = accounts.create_key(uid)
    _, port, _ = serve(accounts, batch_size=2, batch_queue=1, queue_timeout=0.3)
    engine.lock.acquire()                                          # the model is busy: the wave cannot start
    try:
        first = threading.Thread(target=lambda: post(port, k, body(0)))
        first.start(); time.sleep(0.3)                             # taken by the scheduler thread, waiting for the lock
        second = threading.Thread(target=lambda: post(port, k, body(1)))
        second.start(); time.sleep(0.2)                            # fills the queue of one
        status, data, headers = post(port, k, body(2))
        assert status == 503 and headers.get("retry-after") == "5" and "waiting" in data["error"]["message"]
    finally:
        engine.lock.release()
    first.join(20); second.join(20)
    rows = accounts.usage_rows(accounts.lookup(k).key_id, 20)
    assert any(r["status"] == 503 and r["error_type"] == "busy" for r in rows)
    accounts.close()


def test_a_request_that_waits_longer_than_the_queue_timeout_is_refused_and_dropped(serve, engine):
    _, port, key = serve(batch_size=2, queue_timeout=0.3)
    engine.lock.acquire()
    try:
        status, data, headers = post(port, key, body(0))
    finally:
        engine.lock.release()
    assert status == 503 and headers.get("retry-after") == "5"
    time.sleep(0.3)
    st = stats(port, key)
    assert st["waiting"] == 0 and st["in_flight"] == 0                  # nothing leaked
    assert post(port, key, body(0))[0] == 200                           # and the server is fine afterwards


def test_a_client_that_disconnects_mid_stream_frees_its_place(serve):
    _, port, key = serve(batch_size=4, batch_wait_ms=10)
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    payload = json.dumps(body(0, stream=True, max_tokens=30)).encode()
    s.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {key}\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(payload)}\r\n\r\n").encode() + payload)
    s.recv(2048)                                                        # the stream has started
    s.close()                                                           # ...and the client is gone
    for _ in range(100):
        st = stats(port, key)
        if st["in_flight"] == 0 and st["waiting"] == 0:
            break
        time.sleep(0.05)
    assert st["in_flight"] == 0 and st["waiting"] == 0
    assert post(port, key, body(1))[0] == 200


def test_usage_is_recorded_per_request_for_batched_requests(serve, tmp_path):
    accounts = AccountStore(tmp_path / "u.db")
    uid = accounts.create_user("u@example.com")
    k = accounts.create_key(uid)
    _, port, _ = serve(accounts, batch_size=4, batch_wait_ms=100)
    threads = [threading.Thread(target=lambda i=i: post(port, k, body(i))) for i in range(3)]
    [t.start() for t in threads]; [t.join(20) for t in threads]
    rows = accounts.usage_rows(accounts.lookup(k).key_id, 20)
    assert len(rows) == 3 and all(r["status"] == 200 and r["endpoint"] == "/v1/chat/completions" for r in rows)
    accounts.close()


def test_batching_is_off_by_default_and_for_learned_position_models(engine, tmp_path, monkeypatch):
    store = KeyStore(tmp_path / "k.json"); store.create("x")
    from fastapi.testclient import TestClient
    key = store.create("y")
    plain = TestClient(create_app(engine, store))
    assert "batching" not in plain.get("/stats", headers={"Authorization": f"Bearer {key}"}).json()
    monkeypatch.setattr(engine.model.cfg, "arch", "gpt2")
    gpt2 = TestClient(create_app(engine, store, batch_size=8))
    assert "batching" not in gpt2.get("/stats", headers={"Authorization": f"Bearer {key}"}).json()
    monkeypatch.setattr(engine.model.cfg, "arch", "modern")
    on = TestClient(create_app(engine, store, batch_size=8))
    assert on.get("/stats", headers={"Authorization": f"Bearer {key}"}).json()["batching"]["max_batch"] == 8
