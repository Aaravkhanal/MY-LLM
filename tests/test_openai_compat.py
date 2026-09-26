"""OpenAI-compatible routes, exercised through the official `openai` SDK (what a user actually runs)."""

import json

import pytest
from fastapi.testclient import TestClient

openai = pytest.importorskip("openai")

from aarav_lm.api.app import create_app                      # noqa: E402
from aarav_lm.api.auth import KeyStore                       # noqa: E402


@pytest.fixture()
def served(engine, tmp_path):
    store = KeyStore(tmp_path / "keys.json")
    key = store.create("sdk")
    app = create_app(engine, store, rate_limit_per_min=600, rate_limit_burst=100, queue_timeout=0.5,
                     model_name="exp_test")
    return TestClient(app), key


@pytest.fixture()
def sdk(served):
    http, key = served
    return openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=http, max_retries=0)


# ── models ───────────────────────────────────────────────────────────────────

def test_models_list_and_retrieve(sdk):
    models = sdk.models.list()
    assert [m.id for m in models.data] == ["exp_test"]
    m = sdk.models.retrieve("exp_test")
    assert m.id == "exp_test" and m.owned_by == "aarav-lm"
    assert sdk.models.retrieve("aarav-lm").id == "exp_test"                    # documented alias
    with pytest.raises(openai.NotFoundError) as e:
        sdk.models.retrieve("gpt-4")
    assert e.value.code == "model_not_found"


# ── chat completions ─────────────────────────────────────────────────────────

def test_chat_completion_has_the_openai_shape(sdk):
    r = sdk.chat.completions.create(model="exp_test", messages=[{"role": "user", "content": "Once upon a time"}],
                                    max_tokens=12, temperature=0)
    assert r.object == "chat.completion" and r.id.startswith("chatcmpl-") and r.model == "exp_test"
    c = r.choices[0]
    assert c.index == 0 and c.message.role == "assistant" and c.finish_reason in ("stop", "length")
    assert 1 <= r.usage.completion_tokens <= 12
    assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens


def test_streaming_reassembles_to_exactly_the_non_streaming_answer(sdk):
    kw = dict(model="exp_test", messages=[{"role": "user", "content": "The dog"}], max_tokens=20, temperature=0)
    whole = sdk.chat.completions.create(**kw).choices[0].message.content
    chunks = list(sdk.chat.completions.create(stream=True, **kw))
    assert chunks[0].choices[0].delta.role == "assistant"
    assert all(c.object == "chat.completion.chunk" for c in chunks)
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == whole
    assert chunks[-1].choices[0].finish_reason in ("stop", "length")
    assert all(c.choices[0].finish_reason is None for c in chunks[:-1])


def test_raw_stream_ends_with_the_done_sentinel(served):
    http, key = served
    r = http.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                  json={"model": "exp_test", "messages": [{"role": "user", "content": "Hi"}], "stream": True, "max_tokens": 5})
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = [f for f in r.text.split("\n\n") if f]
    assert frames[-1] == "data: [DONE]" and all(f.startswith("data: ") for f in frames)
    assert json.loads(frames[0][6:])["choices"][0]["delta"]["role"] == "assistant"


def test_seed_makes_sampling_reproducible_and_stop_sequences_apply(sdk):
    kw = dict(model="exp_test", messages=[{"role": "user", "content": "Once"}], max_tokens=25, temperature=1.0, seed=7)
    assert sdk.chat.completions.create(**kw).choices[0].message.content == sdk.chat.completions.create(**kw).choices[0].message.content
    text = sdk.chat.completions.create(model="exp_test", messages=[{"role": "user", "content": "Once"}],
                                       max_tokens=40, temperature=0).choices[0].message.content
    if len(text) > 6:
        cut = text[3:6]
        stopped = sdk.chat.completions.create(model="exp_test", messages=[{"role": "user", "content": "Once"}],
                                              max_tokens=40, temperature=0, stop=cut).choices[0]
        assert cut not in stopped.message.content and stopped.finish_reason == "stop"


def test_content_parts_developer_role_and_history_are_accepted(sdk):
    r = sdk.chat.completions.create(model="exp_test", max_tokens=6, temperature=0, messages=[
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Tell me "}, {"type": "text", "text": "a story"}]},
        {"role": "assistant", "content": "Once upon a time."},
        {"role": "user", "content": "More."}])
    assert r.usage.completion_tokens >= 1


# ── completions ──────────────────────────────────────────────────────────────

def test_text_completion_shape_and_streaming(sdk):
    kw = dict(model="exp_test", prompt="Once upon a time", max_tokens=10, temperature=0)
    r = sdk.completions.create(**kw)
    assert r.object == "text_completion" and r.id.startswith("cmpl-") and r.choices[0].text
    assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens
    streamed = "".join(c.choices[0].text for c in sdk.completions.create(stream=True, **kw))
    assert streamed == r.choices[0].text
    assert sdk.completions.create(model="exp_test", prompt=["Once upon a time"], max_tokens=3).choices[0].text is not None


# ── errors are OpenAI-shaped on /v1 and untouched elsewhere ──────────────────

def test_authentication_error_uses_the_openai_error_object(served):
    http, _ = served
    bad = openai.OpenAI(base_url="http://testserver/v1", api_key="aak_wrong", http_client=http, max_retries=0)
    with pytest.raises(openai.AuthenticationError) as e:
        bad.models.list()
    assert e.value.status_code == 401 and e.value.code == "invalid_api_key"
    raw = http.get("/v1/models")
    assert raw.status_code == 401 and raw.headers["www-authenticate"] == "Bearer"
    assert set(raw.json()["error"]) == {"message", "type", "param", "code"}
    assert "detail" in http.get("/model").json()                                # the native routes keep their own shape


@pytest.mark.parametrize("body,fragment", [
    ({"n": 2}, "n=1"),
    ({"temperature": 5}, "temperature"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"stop": ["a", "b", "c", "d", "e"]}, "stop"),
])
def test_unsupported_or_invalid_parameters_are_a_clear_400(served, body, fragment):
    http, key = served
    r = http.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                  json={"model": "exp_test", "messages": [{"role": "user", "content": "hi"}], **body})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    assert fragment in r.json()["error"]["message"]


def test_bad_bodies_and_unsupported_content_are_rejected(sdk, served):
    http, key = served
    with pytest.raises(openai.BadRequestError):
        sdk.chat.completions.create(model="exp_test", messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}}]}])
    with pytest.raises(openai.BadRequestError):
        sdk.completions.create(model="exp_test", prompt=["a", "b"])
    with pytest.raises(openai.NotFoundError):
        sdk.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
    r = http.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={"model": "exp_test"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "messages"


def test_rate_limit_is_a_429_rate_limit_error(engine, tmp_path):
    store = KeyStore(tmp_path / "k.json")
    key = store.create("x")
    http = TestClient(create_app(engine, store, rate_limit_per_min=1, rate_limit_burst=1, model_name="exp_test"))
    c = openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=http, max_retries=0)
    c.models.list()
    with pytest.raises(openai.RateLimitError) as e:
        c.models.list()
    assert e.value.code == "rate_limit_exceeded" and e.value.response.headers["retry-after"]


def test_usage_from_v1_routes_shows_up_in_the_shared_stats(served, sdk):
    http, key = served
    before = http.get("/stats", headers={"Authorization": f"Bearer {key}"}).json()
    resp = sdk.chat.completions.create(model="exp_test", messages=[{"role": "user", "content": "hi"}], max_tokens=4)
    after = http.get("/stats", headers={"Authorization": f"Bearer {key}"}).json()
    # unseeded sampling may end at once (zero completion tokens), so compare with this response's own usage, not "> 0"
    assert after["requests"] == before["requests"] + 1
    assert after["completion_tokens"] == before["completion_tokens"] + resp.usage.completion_tokens
