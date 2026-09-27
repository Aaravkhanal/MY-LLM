"""The public demo: one short prompt, no account, streamed, rate limited per client, off unless the owner turns it on."""

import json

import pytest
from fastapi.testclient import TestClient

from aarav_lm.api.app import create_app
from aarav_lm.api.auth import KeyStore
from aarav_lm.api.store import AccountStore


def keystore(tmp_path):
    s = KeyStore(tmp_path / "k.json")
    s.create("t")
    return s


def frames(text):
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block.startswith("data:") and block[5:].strip() != "[DONE]":
            out.append(json.loads(block[5:]))
    return out


def reply_text(resp_text):
    return "".join((f["choices"][0]["delta"] or {}).get("content", "") for f in frames(resp_text) if f.get("choices"))


def make(engine, tmp_path, **kw):
    kw.setdefault("model_name", "exp_test")
    kw.setdefault("rate_limit_per_min", 600)
    kw.setdefault("rate_limit_burst", 100)
    return TestClient(create_app(engine, keystore(tmp_path), **kw))


def test_the_demo_is_off_by_default_and_the_config_says_so(engine, tmp_path):
    c = make(engine, tmp_path)
    assert c.post("/demo/chat", json={"prompt": "Tell me a story about a fox."}).status_code in (404, 405)


def test_a_visitor_without_any_credentials_gets_a_streamed_reply_when_the_demo_is_on(engine, tmp_path):
    c = make(engine, tmp_path, public_demo=True)
    r = c.post("/demo/chat", json={"prompt": "Tell me a story about a fox named Kip."})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.text.rstrip().endswith("data: [DONE]") and frames(r.text)                # real SSE frames in the OpenAI chunk shape
    assert c.post("/v1/chat/completions", json={"model": "exp_test", "messages": [{"role": "user", "content": "hi"}]}).status_code == 401   # only the demo is open


@pytest.mark.parametrize("body", [{}, {"prompt": ""}, {"prompt": "   "}, {"prompt": "x" * 301}, {"prompt": 5}])
def test_bad_demo_prompts_are_rejected_before_the_model_is_touched(engine, tmp_path, body):
    c = make(engine, tmp_path, public_demo=True)
    r = c.post("/demo/chat", json=body)
    assert r.status_code in (400, 422) and not engine.lock.locked()


def test_the_demo_is_rate_limited_per_client_with_retry_after(engine, tmp_path):
    c = make(engine, tmp_path, public_demo=True, demo_per_minute=1.0)
    codes = [c.post("/demo/chat", json={"prompt": "Once"}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]                    # a burst of three, then one a minute
    r = c.post("/demo/chat", json={"prompt": "Once"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1 and "sign in" in r.json()["detail"]


def test_the_demo_caps_how_much_it_generates(engine, tmp_path, monkeypatch):
    seen = []
    real = engine.stream

    def spy(ids, cfg=None):
        seen.append(cfg)
        return real(ids, cfg)
    monkeypatch.setattr(engine, "stream", spy)
    c = make(engine, tmp_path, public_demo=True, demo_max_tokens=7)
    assert c.post("/demo/chat", json={"prompt": "Once upon a time"}).status_code == 200
    assert seen and seen[0].max_new_tokens == 7 and seen[0].temperature == 0.7           # the client cannot ask for more or change sampling


def test_demo_use_is_not_attributed_to_any_user_or_recorded(engine, tmp_path):
    store = AccountStore(tmp_path / "a.db")
    c = TestClient(create_app(engine, store, model_name="exp_test", signup_code="x", public_demo=True, rate_limit_per_min=600, rate_limit_burst=100))
    assert c.post("/demo/chat", json={"prompt": "Once"}).status_code == 200
    assert store.usage_summary()["requests"] == 0                                        # no user, so no usage row
    assert c.get("/app/config").json()["model_info"]["demo"] == {"enabled": True, "max_tokens": 120, "per_minute": 6.0, "max_chars": 300}
    store.close()


def test_the_demo_reports_a_busy_model_as_503_not_a_hang(engine, tmp_path):
    c = make(engine, tmp_path, public_demo=True, queue_timeout=0.2)
    engine.lock.acquire()
    try:
        r = c.post("/demo/chat", json={"prompt": "Once"})
    finally:
        engine.lock.release()
    assert r.status_code == 503 and "Retry-After" in r.headers


def test_the_closing_stream_frame_carries_real_token_counts(engine, tmp_path):
    """An additive extension: the last chunk of a streamed reply includes usage, so a client can show real counts."""
    c = make(engine, tmp_path, public_demo=True)
    r = c.post("/demo/chat", json={"prompt": "Once upon a time"})
    last = frames(r.text)[-1]
    assert last["choices"][0]["finish_reason"] in ("stop", "length") and set(last["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert last["usage"]["total_tokens"] == last["usage"]["prompt_tokens"] + last["usage"]["completion_tokens"] > 0
    assert all("usage" not in f for f in frames(r.text)[:-1])                            # not on every frame


def test_finish_reasons_are_the_two_values_openai_clients_know(engine, tmp_path):
    """The engine's own reasons include "eot" (the model ended the story); on the OpenAI routes that must read "stop"."""
    from aarav_lm.api.openai_compat import _finish_reason
    assert [_finish_reason({"finish_reason": r}) for r in ("eot", "stop", "length", "timeout", "cancelled")] == ["stop", "stop", "length", "length", "stop"]
