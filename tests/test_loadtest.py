"""The load tester, against a real server on a real socket."""

import importlib.util
import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from aarav_lm.api.app import create_app
from aarav_lm.api.auth import KeyStore

spec = importlib.util.spec_from_file_location("loadtest", Path(__file__).resolve().parents[1] / "scripts" / "loadtest.py")
loadtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loadtest)


@pytest.fixture()
def server(engine, tmp_path):
    import uvicorn
    store = KeyStore(tmp_path / "k.json")
    key = store.create("load")

    def start(**kw):
        app = create_app(engine, store, rate_limit_per_min=600000, rate_limit_burst=10000, model_name="exp_test", **kw)
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        t = threading.Thread(target=srv.run, daemon=True); t.start()
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
            except OSError:
                time.sleep(0.05)
        started.append((srv, t))
        return f"http://127.0.0.1:{port}", key
    started = []
    yield start
    for srv, t in started:
        srv.should_exit = True; t.join(5)


def test_a_light_load_is_served_completely_and_measured(server):
    url, key = server(queue_timeout=10)
    r = loadtest.run(url, key, concurrency=2, requests=8, max_tokens=6, model="exp_test")
    assert r["statuses"] == {"200": 8} and r["faults"] == []
    assert r["completion_tokens_per_second"] > 0 and r["served_per_second"] > 0
    lat = r["latency_seconds"]
    assert 0 < lat["p50"] <= lat["p95"] <= lat["max"]


def test_streaming_reports_time_to_first_token_before_the_full_latency(server):
    url, key = server(queue_timeout=10)
    r = loadtest.run(url, key, concurrency=2, requests=6, max_tokens=8, stream=True, model="exp_test")
    assert r["statuses"] == {"200": 6}
    ttft = r["time_to_first_token_seconds"]
    assert ttft["p50"] is not None and 0 < ttft["p50"] <= r["latency_seconds"]["p50"]


def test_overload_is_refused_in_an_orderly_way_and_never_faults(server):
    url, key = server(queue_timeout=0.05, max_waiters=2)                                      # a tiny queue: most requests must be refused
    r = loadtest.run(url, key, concurrency=12, requests=36, max_tokens=40, model="exp_test")
    assert set(r["statuses"]) <= {"200", "503"} and "503" in r["statuses"] and "200" in r["statuses"]
    assert r["faults"] == [] and r["retry_after_seen"] == ["5"]                               # every refusal says when to retry


def test_a_dead_server_is_counted_not_crashed_on():
    r = loadtest.run("http://127.0.0.1:9", "aak_x", concurrency=2, requests=4, max_tokens=4, timeout=1.0)
    assert r["statuses"] == {"connection_error": 4} and len(r["faults"]) == 4
    assert r["latency_seconds"]["p50"] is None


def test_percentiles():
    assert loadtest.pct([1, 2, 3, 4, 5], 50) == 3 and loadtest.pct([1, 2, 3, 4, 5], 95) == 5 and loadtest.pct([], 50) is None
