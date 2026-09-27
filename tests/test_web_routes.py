"""The web app's routing on the server: page paths, the exact asset map, the public status check and the architecture it reports."""

import json
import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aarav_lm.api.app import create_app
from aarav_lm.api.auth import KeyStore
from aarav_lm.api.store import AccountStore
from aarav_lm.api.web_static import ASSET_TYPES, PAGE_ROUTES, collect_assets

WEB = Path(__file__).resolve().parents[1] / "src" / "aarav_lm" / "web"


def keystore(tmp_path):
    store = KeyStore(tmp_path / "k.json")
    store.create("t")                                   # a server with authentication on refuses to start without a key
    return store


@pytest.fixture()
def client(engine, tmp_path):
    return TestClient(create_app(engine, keystore(tmp_path), model_name="exp_test"))      # auth ON: pages and status are public


# ── pages ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", PAGE_ROUTES)
def test_every_page_path_answers_with_the_shell_and_strict_headers(client, path):
    r = client.get(path)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html"), path
    csp = r.headers["content-security-policy"]
    assert "default-src 'none'" in csp and "script-src 'self'" in csp and "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert r.headers["x-content-type-options"] == "nosniff" and "frame-ancestors 'none'" in csp
    assert r.text == client.get("/").text                                  # one shell; the client draws the page from the path


@pytest.mark.parametrize("path", ["/nope", "/docs/unknown", "/models/other-model", "/dashboard/billing", "/playground/x",
                                  "/models/aarav-lm/evaluations/x", "/index.html", "/%2e%2e/pyproject.toml", "/../secrets/api_keys.json",
                                  "/web/js/main.js", "/static/app.js", "/js/../index.html"])
def test_a_path_that_is_not_a_page_or_an_asset_is_a_real_404(client, path):
    assert client.get(path).status_code in (404, 405), path


def test_the_page_list_has_no_duplicates_and_starts_at_the_root():
    assert len(set(PAGE_ROUTES)) == len(PAGE_ROUTES) and PAGE_ROUTES[0] == "/"
    assert all(p == "/" or (p.startswith("/") and not p.endswith("/")) for p in PAGE_ROUTES)


def test_the_page_routes_do_not_shadow_an_api_route(client):
    api = {r.path for r in client.app.routes if getattr(r, "methods", None) and r.path not in PAGE_ROUTES}
    assert not (set(PAGE_ROUTES) & api)
    for prefix in ("/v1", "/account", "/generate", "/chat", "/stats", "/health", "/model", "/app/config"):
        assert not any(p == prefix or p.startswith(prefix + "/") for p in PAGE_ROUTES if p != "/"), prefix


def test_web_pages_can_be_disabled(engine, tmp_path):
    c = TestClient(create_app(engine, keystore(tmp_path), serve_web=False))
    assert c.get("/").status_code == 404 and c.get("/playground").status_code == 404


# ── assets ───────────────────────────────────────────────────────────────────

def test_only_files_with_allowed_extensions_are_served_and_each_at_its_own_path(tmp_path):
    (tmp_path / "js").mkdir()
    (tmp_path / "js" / "a.js").write_text("1")
    (tmp_path / "style.css").write_text("2")
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "index.html").write_text("<html>")
    (tmp_path / "notes.py").write_text("secret")
    (tmp_path / ".hidden.js").write_text("x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.txt").write_text("x")
    (tmp_path / "link.js").symlink_to(tmp_path / "js" / "a.js")
    got = collect_assets(tmp_path)
    assert set(got) == {"/js/a.js", "/style.css", "/data.json"}
    assert got["/js/a.js"] == (b"1", ASSET_TYPES[".js"])


def test_the_real_web_directory_is_served_completely(client):
    served = collect_assets(WEB)
    assert "/css/tokens.css" in served and "/js/main.js" in served
    for path, (body, media) in served.items():
        r = client.get(path)
        assert r.status_code == 200 and r.content == body and r.headers["content-type"].startswith(media.split(";")[0]), path
        assert "unsafe-inline" not in r.headers["content-security-policy"]


# ── status ───────────────────────────────────────────────────────────────────

def test_status_is_public_and_says_what_each_service_is_doing(client):
    r = client.get("/status.json")
    d = r.json()
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert d["overall"] == "operational" and set(d["services"]) == {"api", "inference", "authentication", "playground"}
    assert all(v["state"] == "operational" and v["detail"] for v in d["services"].values())
    assert d["queue"]["waiting"] == 0 and d["queue"]["limit"] >= 1 and d["uptime_seconds"] >= 0 and d["checked_at"].endswith("+00:00")
    text = json.dumps(d)
    assert "secret" not in text and "@" not in text and "checkpoint" not in text          # counts and states only


def test_status_reports_an_account_database_that_stops_answering(engine, tmp_path):
    store = AccountStore(tmp_path / "a.db")
    c = TestClient(create_app(engine, store, model_name="exp_test", signup_code="invite-123"))
    assert c.get("/status.json").json()["services"]["authentication"]["state"] == "operational"
    store.close()                                                        # a closed connection cannot answer a query
    d = c.get("/status.json").json()
    assert d["services"]["authentication"]["state"] == "outage" and d["overall"] == "outage"
    assert d["services"]["api"]["state"] == "operational"                # the failure is attributed to the right service


def test_status_reports_a_stopped_batch_scheduler_as_an_inference_outage(engine, tmp_path):
    app = create_app(engine, keystore(tmp_path), batch_size=2, model_name="exp_test")
    with TestClient(app) as c:
        d = c.get("/status.json").json()
        assert d["services"]["inference"]["state"] == "operational" and "batching" in d["services"]["inference"]["detail"]
    # leaving the context ran the app's shutdown hooks, which stop the scheduler
    d = TestClient(app).get("/status.json").json()
    assert d["services"]["inference"]["state"] == "outage" and d["overall"] == "outage"


def test_status_reports_a_full_queue_as_degraded_not_operational(engine, tmp_path):
    import uvicorn
    store = KeyStore(tmp_path / "k.json")
    key = store.create("s")
    app = create_app(engine, store, model_name="exp_test", max_waiters=1, queue_timeout=8.0, rate_limit_per_min=60000, rate_limit_burst=1000)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True); t.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/health", timeout=1); break
        except OSError:
            time.sleep(0.05)
    engine.lock.acquire()                                                # the model is busy with someone else
    try:
        def ask():
            req = urllib.request.Request(base + "/v1/chat/completions", method="POST", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                         data=json.dumps({"model": "exp_test", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]}).encode())
            try:
                urllib.request.urlopen(req, timeout=20).read()
            except Exception:                                            # noqa: BLE001 - the test only needs the request to wait in the queue
                pass
        w = threading.Thread(target=ask, daemon=True); w.start()
        d = {}
        for _ in range(100):
            d = json.loads(urllib.request.urlopen(base + "/status.json", timeout=5).read())
            if d["services"]["inference"]["state"] == "degraded":
                break
            time.sleep(0.05)
        assert d["services"]["inference"]["state"] == "degraded" and d["overall"] == "degraded", d
        assert "queue is full" in d["services"]["inference"]["detail"] and d["queue"]["waiting"] == 1
    finally:
        engine.lock.release()
    w.join(15)
    srv.should_exit = True; t.join(5)


# ── architecture and latency ─────────────────────────────────────────────────

def test_the_config_reports_the_real_architecture_of_the_loaded_model(engine, client):
    import aarav_lm.api.app as appmod
    a = appmod.architecture_info(engine)
    c = engine.model.cfg
    assert a["layers"] == c.n_layer == len(engine.model.blocks) and a["width"] == c.n_embd and a["heads"] == c.n_head
    assert a["kv_heads"] == c.n_kv_head and a["head_size"] * a["heads"] == a["width"] and a["context_length"] == c.block_size
    assert a["parameters"] == engine.model.num_params() and a["parameters_body"] == engine.model.num_params(exclude_embeddings=True)
    assert a["ffn_hidden"] == engine.model.blocks[0].mlp.gate.out_features
    assert a["positions"] == "rope" and a["norm"] == c.norm_type and a["tied_embeddings"] == c.tie_embeddings
    json.dumps(a)                                                        # plain JSON types only


def test_app_config_carries_the_architecture_when_accounts_are_enabled(engine, tmp_path):
    store = AccountStore(tmp_path / "a.db")
    c = TestClient(create_app(engine, store, model_name="exp_test", signup_code="invite-123"))
    info = c.get("/app/config").json()["model_info"]
    assert info["architecture"]["layers"] == engine.model.cfg.n_layer and info["parameters"] == engine.model.num_params()
    store.close()


def test_usage_latency_percentiles_count_only_successful_requests(tmp_path):
    s = AccountStore(tmp_path / "a.db")
    uid = s.create_user("a@example.com")
    for ms in range(100, 1100, 100):                                     # 100 .. 1000 ms, all successful
        s.record_usage(key_id=None, user_id=uid, model_id="m", endpoint="/x", duration_ms=ms, status=200)
    s.record_usage(key_id=None, user_id=uid, model_id="m", endpoint="/x", duration_ms=1, status=429, error_type="rate_limited")
    t = s.usage_summary(user_id=uid)
    assert t["requests"] == 11 and t["errors"] == 1
    assert t["latency_p50_ms"] == 600 and t["latency_p95_ms"] == 1000     # the 1 ms refusal does not pull the median down
    empty = s.usage_summary(user_id=uid + 99)
    assert empty["requests"] == 0 and empty["latency_p50_ms"] is None and empty["latency_p95_ms"] is None
    s.close()


# ── the client and the server agree ──────────────────────────────────────────

import re                                                            # noqa: E402

import web_sources                                                   # noqa: E402


def test_the_client_router_and_the_server_page_list_are_the_same_list():
    router = (WEB / "js" / "router.js").read_text()
    client = re.findall(r'path: "(/[^"]*)"', router)
    assert client == list(PAGE_ROUTES), (set(client) ^ set(PAGE_ROUTES))


def test_every_module_import_resolves_to_a_file_the_server_will_serve():
    served = collect_assets(WEB)
    for path in web_sources.JS_FILES:
        src = path.read_text()
        for rel in re.findall(r'(?:from |import\()\s*"(\.{1,2}/[^"]+)"', src):
            target = (path.parent / rel).resolve().relative_to(WEB.resolve()).as_posix()
            assert "/" + target in served, (path.name, rel)


def test_every_internal_link_points_at_a_page_or_a_file():
    served = collect_assets(WEB)
    known = set(PAGE_ROUTES) | set(served)
    links = set()
    for text in (web_sources.JS, web_sources.HTML):
        links |= set(re.findall(r'(?:href: ?|href=|navigate\(|href: `)"(/[^"?#]*)', text))
    links -= {"/v1", "/account"}                                     # API bases, not pages
    unknown = {l for l in links if l.rstrip("/") not in known and l not in known and not l.startswith("/v1") and not l.startswith("/account")}
    assert not unknown, unknown
    assert len(links) > 15                                           # the check is actually looking at the site's links


def test_every_page_module_exists_and_exports_a_page_function():
    router = (WEB / "js" / "router.js").read_text()
    for rel in set(re.findall(r'import\("\./(pages/[a-z]+\.js)"\)', router)):
        src = (WEB / "js" / rel).read_text()
        assert "export default async function page" in src, rel


# ── compression ──────────────────────────────────────────────────────────────

def test_static_files_are_gzipped_only_for_clients_that_accept_it_and_decode_to_the_same_bytes(client):
    import gzip
    plain = client.get("/js/ui.js", headers={"Accept-Encoding": "identity"})
    assert plain.status_code == 200 and "content-encoding" not in plain.headers and plain.headers["vary"] == "Accept-Encoding"
    packed = client.get("/js/ui.js", headers={"Accept-Encoding": "gzip"})
    assert packed.headers["content-encoding"] == "gzip" and packed.headers["vary"] == "Accept-Encoding"
    assert packed.content == plain.content and len(plain.content) > 1000                   # same bytes after decoding
    shell = client.get("/playground", headers={"Accept-Encoding": "gzip"})
    assert shell.text == client.get("/", headers={"Accept-Encoding": "identity"}).text and "unsafe-inline" not in shell.headers["content-security-policy"]


def test_compression_never_touches_api_responses(client):
    r = client.get("/status.json", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in r.headers and r.json()["overall"] == "operational"


def test_gzip_output_is_deterministic_and_smaller():
    from aarav_lm.api.web_static import _gzipped
    body = (WEB / "js" / "ui.js").read_bytes()
    assert _gzipped(body) == _gzipped(body) and len(_gzipped(body)) < 0.5 * len(body)
