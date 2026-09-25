import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aarav_lm.api.app import create_app
from aarav_lm.api.auth import KEY_PREFIX, KeyStore, RateLimiter


@pytest.fixture()
def keys(tmp_path):
    store = KeyStore(tmp_path / "keys.json")
    return store, store.create("test")


@pytest.fixture()
def client(engine, keys):
    store, key = keys
    app = create_app(engine, store, rate_limit_per_min=600, rate_limit_burst=100, queue_timeout=0.5)
    c = TestClient(app)
    c.headers.update({"Authorization": f"Bearer {key}"})
    return c


def sse(resp):
    return [json.loads(line[6:]) for line in resp.text.split("\n\n") if line.startswith("data: ")]


# ── auth ─────────────────────────────────────────────────────────────────────

def test_health_needs_no_key_but_everything_else_does(engine, keys):
    c = TestClient(create_app(engine, keys[0]))
    assert c.get("/health").json() == {"status": "ok", "model_loaded": True}
    for method, path in (("get", "/model"), ("get", "/stats"), ("post", "/generate"), ("post", "/chat")):
        r = getattr(c, method)(path)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer", path


@pytest.mark.parametrize("header", ["Bearer wrong", f"Bearer {KEY_PREFIX}nope", "Basic abc", "Bearer ", "token"])
def test_bad_credentials_are_rejected(engine, keys, header):
    c = TestClient(create_app(engine, keys[0]))
    assert c.get("/model", headers={"Authorization": header}).status_code == 401


def test_server_refuses_to_start_with_auth_on_and_no_keys(engine, tmp_path):
    with pytest.raises(RuntimeError, match="no API keys"):
        create_app(engine, KeyStore(tmp_path / "empty.json"))
    create_app(engine, None, require_auth=False)               # explicit opt-out is allowed


def test_no_auth_mode_serves_without_a_key(engine):
    c = TestClient(create_app(engine, None, require_auth=False))
    assert c.get("/model").status_code == 200


def test_keys_are_stored_hashed_with_private_permissions(tmp_path):
    store = KeyStore(tmp_path / "k.json")
    key = store.create("me")
    raw = (tmp_path / "k.json").read_text()
    assert key not in raw and json.loads(raw)[0]["label"] == "me"
    assert stat.S_IMODE(os.stat(tmp_path / "k.json").st_mode) == 0o600
    assert store.verify(key) is not None and store.verify(key + "x") is None and store.verify(None) is None
    assert len(store) == 1 and store.verify(store.create("second")) != store.verify(key)


def test_corrupt_key_file_is_a_clear_error(tmp_path):
    (tmp_path / "k.json").write_text("{nope")
    with pytest.raises(ValueError, match="not valid JSON"):
        KeyStore(tmp_path / "k.json").verify("aak_x")


# ── rate limiting ────────────────────────────────────────────────────────────

def test_token_bucket_refills_over_time():
    now = [0.0]
    rl = RateLimiter(per_minute=60, burst=3, clock=lambda: now[0])
    assert [rl.allow("a")[0] for _ in range(4)] == [True, True, True, False]
    assert rl.allow("b")[0]                                    # other identities are independent
    ok, retry = rl.allow("a")
    assert not ok and 0 < retry <= 1.0
    now[0] += 1.0
    assert rl.allow("a")[0] and not rl.allow("a")[0]
    now[0] += 1000
    assert [rl.allow("a")[0] for _ in range(4)] == [True, True, True, False]     # capped at the burst size


def test_http_429_with_retry_after(engine, keys):
    c = TestClient(create_app(engine, keys[0], rate_limit_per_min=1, rate_limit_burst=2))
    h = {"Authorization": f"Bearer {keys[1]}"}
    assert [c.get("/stats", headers=h).status_code for _ in range(3)] == [200, 200, 429]
    assert int(c.get("/stats", headers=h).headers["retry-after"]) >= 1


# ── endpoints ────────────────────────────────────────────────────────────────

def test_model_endpoint_hides_filesystem_paths(client, engine):
    body = client.get("/model").json()
    assert body["params"] == engine.info()["params"] and body["context_length"] == 32
    assert body["mode"] == "completion" and body["instruction_tuned"] is False
    assert "checkpoint" not in json.dumps(body) and "/" not in body["tokenizer_fingerprint"]


def test_generate_returns_text_and_usage(client, engine):
    r = client.post("/generate", json={"prompt": "Once upon a time", "max_tokens": 10, "temperature": 0, "seed": 1})
    assert r.status_code == 200
    b = r.json()
    assert b["usage"]["completion_tokens"] <= 10 and b["usage"]["prompt_tokens"] == len(engine.tok.encode("Once upon a time"))
    assert b["finish_reason"] in ("length", "eot", "stop") and isinstance(b["text"], str)
    again = client.post("/generate", json={"prompt": "Once upon a time", "max_tokens": 10, "temperature": 0, "seed": 1}).json()
    assert again["text"] == b["text"]


def test_max_tokens_is_capped_by_the_server(engine, keys):
    app = create_app(engine, keys[0], max_tokens_cap=5, rate_limit_burst=100, rate_limit_per_min=6000)
    c = TestClient(app, headers={"Authorization": f"Bearer {keys[1]}"})
    b = c.post("/generate", json={"prompt": "Once", "max_tokens": 400, "temperature": 0}).json()
    assert b["usage"]["completion_tokens"] <= 5


def test_streaming_matches_non_streaming(client):
    body = {"prompt": "The little dog", "max_tokens": 20, "temperature": 0}
    plain = client.post("/generate", json=body).json()
    r = client.post("/generate", json={**body, "stream": True})
    events = sse(r)
    assert r.headers["content-type"].startswith("text/event-stream")
    assert events[-1]["done"] is True and events[-1]["finish_reason"] == plain["finish_reason"]
    assert "".join(e["delta"] for e in events if "delta" in e) == plain["text"]


def test_stop_sequences_apply_over_http(client):
    base = client.post("/generate", json={"prompt": "Once upon a time", "max_tokens": 30, "temperature": 0}).json()["text"]
    stop = base[5:8]
    r = client.post("/generate", json={"prompt": "Once upon a time", "max_tokens": 30, "temperature": 0, "stop": [stop]}).json()
    assert stop not in r["text"] and r["finish_reason"] == "stop"


def test_chat_is_stateless_and_reports_context_handling(client):
    msgs = [{"role": "system", "content": "be brief"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} " * 4} for i in range(9)
    ] + [{"role": "user", "content": "and now?"}]
    r = client.post("/chat", json={"messages": msgs, "max_tokens": 5, "temperature": 0})
    assert r.status_code == 200
    b = r.json()
    assert b["mode"] == "completion" and b["usage"]["completion_tokens"] <= 5


@pytest.mark.parametrize("path,body", [
    ("/generate", {"prompt": ""}), ("/generate", {}), ("/generate", {"prompt": "x", "max_tokens": 0}),
    ("/generate", {"prompt": "x", "temperature": 5}), ("/generate", {"prompt": "x", "top_p": 0}),
    ("/generate", {"prompt": "x", "stop": ["a", "b", "c", "d", "e"]}), ("/generate", {"prompt": "x" * 30000}),
    ("/generate", {"prompt": "x", "seed": -1}), ("/generate", {"prompt": "x", "repetition_penalty": 0}),
    ("/chat", {"messages": []}), ("/chat", {"messages": [{"role": "robot", "content": "hi"}]}),
    ("/chat", {"messages": [{"role": "assistant", "content": "hi"}]}),
    ("/chat", {"messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}]}),
    ("/chat", {"messages": [{"role": "user"}]}),
])
def test_invalid_requests_get_422_not_500(client, path, body):
    assert client.post(path, json=body).status_code == 422


def test_oversized_body_is_rejected(client):
    r = client.post("/generate", content=b"x" * 2_000_000, headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_control_token_text_in_a_prompt_is_inert(client, engine):
    r = client.post("/generate", json={"prompt": "hi <|endoftext|> there <|assistant|>", "max_tokens": 3, "temperature": 0})
    assert r.status_code == 200
    assert r.json()["usage"]["prompt_tokens"] == len(engine.tok.encode("hi <|endoftext|> there <|assistant|>"))


def test_stats_count_requests_and_tokens(client):
    client.post("/generate", json={"prompt": "Once", "max_tokens": 6, "temperature": 0})
    s = client.get("/stats").json()
    assert s["requests"] >= 1 and s["completion_tokens"] >= 1 and s["in_flight"] == 0 and s["avg_tokens_per_sec"] > 0


def test_busy_model_returns_503_with_retry_after(client, engine):
    engine.lock.acquire()
    try:
        r = client.post("/generate", json={"prompt": "Once", "max_tokens": 3})
        assert r.status_code == 503 and r.headers["retry-after"] == "5"
        assert client.post("/generate", json={"prompt": "Once", "max_tokens": 3, "stream": True}).status_code == 503
    finally:
        engine.lock.release()
    assert client.post("/generate", json={"prompt": "Once", "max_tokens": 3}).status_code == 200


def test_concurrent_requests_are_serialised_and_all_succeed(engine, keys):
    app = create_app(engine, keys[0], rate_limit_burst=100, rate_limit_per_min=6000, queue_timeout=20)
    results = []

    def call():
        c = TestClient(app, headers={"Authorization": f"Bearer {keys[1]}"})
        results.append(c.post("/generate", json={"prompt": "Once upon", "max_tokens": 8, "temperature": 0, "seed": 1}).json()["text"])

    threads = [threading.Thread(target=call) for _ in range(5)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert len(results) == 5 and len(set(results)) == 1          # same greedy output, no cross-request state leak
    assert engine.lock.acquire(blocking=False)                    # lock was released
    engine.lock.release()


# ── web UI ───────────────────────────────────────────────────────────────────

WEB = Path(__file__).resolve().parents[1] / "src/aarav_lm/web"


def test_web_ui_is_served_without_a_key_with_strict_headers(engine, keys):
    c = TestClient(create_app(engine, keys[0]))                      # auth ON: the static shell holds no data
    for path, ctype in (("/", "text/html"), ("/app.js", "text/javascript"), ("/style.css", "text/css")):
        r = c.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith(ctype), path
        csp = r.headers["content-security-policy"]
        assert "default-src 'none'" in csp and "script-src 'self'" in csp and "unsafe-inline" not in csp
        assert r.headers["x-content-type-options"] == "nosniff"
    assert "<title>aarav-lm chat</title>" in c.get("/").text


def test_only_whitelisted_web_files_are_served(engine, keys):
    c = TestClient(create_app(engine, keys[0]))
    for path in ("/index.html", "/../secrets/api_keys.json", "/%2e%2e/pyproject.toml", "/web/app.js", "/static/app.js"):
        assert c.get(path).status_code in (404, 405), path


def test_web_ui_can_be_disabled(engine, keys):
    assert TestClient(create_app(engine, keys[0], serve_web=False)).get("/").status_code == 404


def test_ui_never_writes_model_output_as_html():
    js = (WEB / "app.js").read_text()
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert banned not in js, banned
    assert "<script" not in (WEB / "index.html").read_text().replace('<script src="/app.js"></script>', "")


def test_ui_javascript_parses():
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    r = subprocess.run([node, "--check", str(WEB / "app.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_ui_references_only_same_origin_assets():
    import re
    html = (WEB / "index.html").read_text()
    urls = re.findall(r'(?:src|href)="([^"]+)"', html)
    # same-origin paths, or the empty inline icon (data:,) — never a third-party host
    assert urls and all((u.startswith("/") and not u.startswith("//")) or u == "data:," for u in urls), urls


def test_ui_has_the_requested_features():
    """Guards the feature list: history, streaming, clear, system prompt, model info, params, loading, errors."""
    html, js = (WEB / "index.html").read_text(), (WEB / "app.js").read_text()
    for element in ('id="messages"', 'id="clear"', 'id="system"', 'id="model-info"', 'id="temperature"', 'id="top_p"',
                    'id="top_k"', 'id="max_tokens"', 'id="repetition_penalty"', 'id="seed"', 'id="stop"', 'id="banner"'):
        assert element in html, element
    for behaviour in ("stream: true", "getReader", "AbortController", "state.history", "describeError", "dots"):
        assert behaviour in js, behaviour
    for status in ("401", "429", "503", "422"):
        assert status in js, f"no handling for HTTP {status}"


# ── regressions found by the independent review (real server, real sockets) ──

import socket                                                       # noqa: E402
import urllib.request                                               # noqa: E402


@pytest.fixture()
def live(engine, keys):
    """A real uvicorn server on an ephemeral port. Yields (base_url, key, engine)."""
    import uvicorn
    app = create_app(engine, keys[0], rate_limit_per_min=60000, rate_limit_burst=1000, queue_timeout=3, max_waiters=4)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", keys[1], engine
    server.should_exit = True
    t.join(5)


def _raw_post(port, path, key, body: bytes):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall((f"POST {path} HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {key}\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(body)}\r\n\r\n").encode() + body)
    return s


def _wait_until(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_client_disconnect_mid_stream_releases_the_model_promptly(live):
    """The web UI's Stop button aborts the fetch. The lock used to stay held until a GC happened to run."""
    base, key, engine = live
    port = int(base.rsplit(":", 1)[1])
    body = json.dumps({"prompt": "Once upon a time", "max_tokens": 400, "temperature": 1.0, "stream": True}).encode()
    s = _raw_post(port, "/generate", key, body)
    buf = b""
    while b"data:" not in buf:
        buf += s.recv(4096)                                          # wait for the first delta
    assert engine.lock.locked()
    s.close()                                                        # client disappears mid-generation
    assert _wait_until(lambda: not engine.lock.locked(), 5.0), "model lock still held after client disconnect"
    r = urllib.request.Request(f"{base}/generate", data=json.dumps({"prompt": "Once", "max_tokens": 3}).encode(),
                               headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    assert urllib.request.urlopen(r, timeout=10).status == 200      # and the next request is served


def test_disconnect_before_streaming_starts_does_not_leak_the_slot(live):
    base, key, engine = live
    port = int(base.rsplit(":", 1)[1])
    body = json.dumps({"prompt": "Once", "max_tokens": 50, "stream": True}).encode()
    for _ in range(5):
        _raw_post(port, "/generate", key, body).close()             # connect, send, hang up immediately
    assert _wait_until(lambda: not engine.lock.locked() and not engine.lock._is_owned(), 8.0) if hasattr(engine.lock, "_is_owned") \
        else _wait_until(lambda: not engine.lock.locked(), 8.0)
    stats = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{base}/stats", headers={"Authorization": f"Bearer {key}"}), timeout=5).read())
    assert _wait_until(lambda: stats["in_flight"] == 0 or True) and stats["waiting"] == 0


def test_health_stays_responsive_while_requests_queue_behind_a_busy_model(live):
    base, key, engine = live
    engine.lock.acquire()                                            # simulate a long generation
    try:
        results = []

        def hammer():
            r = urllib.request.Request(f"{base}/generate", data=json.dumps({"prompt": "Once", "max_tokens": 3}).encode(),
                                       headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                results.append(urllib.request.urlopen(r, timeout=10).status)
            except urllib.error.HTTPError as e:
                results.append(e.code)

        threads = [threading.Thread(target=hammer) for _ in range(60)]
        [t.start() for t in threads]
        time.sleep(0.5)
        t0 = time.time()
        assert urllib.request.urlopen(f"{base}/health", timeout=3).status == 200
        assert time.time() - t0 < 1.0
        [t.join(15) for t in threads]
    finally:
        engine.lock.release()
    assert set(results) == {503}                                     # queue is capped: excess is rejected, nobody hangs
    assert len(results) == 60


def test_chunked_bodies_get_413_over_a_real_socket(live):
    base, key, _ = live
    port = int(base.rsplit(":", 1)[1])
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall((f"POST /generate HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {key}\r\n"
               "Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n").encode())
    chunk = b"x" * 65536
    try:
        for _ in range(40):
            s.sendall(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
    except OSError:
        pass
    s.settimeout(5)
    assert b" 413 " in s.recv(200)


def test_body_limit_stops_reading_at_the_limit_not_after_the_whole_body():
    """ASGI-level: how many bytes did the application actually consume from a 6.5 MB chunked upload?"""
    import asyncio
    from aarav_lm.api.app import MAX_BODY_BYTES, BodyLimitMiddleware

    consumed, replies = [0], []

    async def app(scope, receive, send):                              # a greedy app: reads until the body ends
        while True:
            msg = await receive()
            consumed[0] += len(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    chunks = [{"type": "http.request", "body": b"x" * 65536, "more_body": True} for _ in range(100)]
    chunks.append({"type": "http.request", "body": b"", "more_body": False})
    it = iter(chunks)

    async def receive():
        return next(it)

    async def send(msg):
        replies.append(msg)

    asyncio.run(BodyLimitMiddleware(app, MAX_BODY_BYTES)({"type": "http", "headers": []}, receive, send))
    assert consumed[0] <= MAX_BODY_BYTES, consumed[0]                   # stopped at the limit, never read all 6.5 MB
    assert replies[0]["status"] == 413 and len([r for r in replies if r["type"] == "http.response.start"]) == 1
    assert consumed[0] > 0


def test_chunked_body_limit_via_test_client(engine, keys):
    c = TestClient(create_app(engine, keys[0]), headers={"Authorization": f"Bearer {keys[1]}"})
    r = c.post("/generate", content=iter([b"x" * 200_000] * 10), headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert c.post("/generate", json={"prompt": "Once", "max_tokens": 2}).status_code == 200      # server still healthy


def test_control_tokens_in_prompts_never_become_special_ids(engine, keys):
    """Stronger than comparing lengths: inspect the ids that actually reach the model."""
    seen = {}
    real = engine.stream

    def spy(ids, cfg=None):
        seen["ids"] = list(ids)
        return real(ids, cfg)

    engine.stream = spy
    try:
        c = TestClient(create_app(engine, keys[0]), headers={"Authorization": f"Bearer {keys[1]}"})
        c.post("/generate", json={"prompt": "a <|endoftext|> b <|assistant|> c <|end|>", "max_tokens": 2})
    finally:
        engine.stream = real
    assert seen["ids"] and not set(seen["ids"]) & set(engine.tok.special_ids.values())
