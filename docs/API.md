# API

Local HTTP API served by `scripts/serve.py` (FastAPI). It has two front doors over one implementation:
the **OpenAI-compatible** `/v1/...` routes (below, works with the official SDKs) and the original **native**
routes (`/generate`, `/chat`). Accounts, per-key limits and usage records are in the SQLite account database (below).

```bash
# simplest: one owner, keys in a JSON file (only SHA-256 hashes are stored)
python scripts/create_key.py --label "my laptop"
python scripts/serve.py --checkpoint models/tiny-tinystories-v1-sft-int8.pt

# accounts: users, per-key limits, expiry, revoke/rotate, usage records (SQLite)
python scripts/manage_keys.py user add alice@example.com
python scripts/manage_keys.py key create alice@example.com --label laptop --rpm 30 --tpm 20000 --expires-days 90
python scripts/serve.py --db secrets/accounts.db --checkpoint models/tiny-tinystories-v1-sft-int8.pt
```

Defaults: binds to `127.0.0.1:8000`, requires a key, 512-token output cap, 60 requests/minute per key
(burst 10). `--no-auth` is refused unless the host is localhost.

## Authentication

```
Authorization: Bearer aak_...
```

Missing or invalid key → `401` with `WWW-Authenticate: Bearer`. `/health` and the web page need no key.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | no | liveness |
| GET | `/model` | yes | parameters, context length, vocabulary, stage (`pretrain`/`sft`), step, validation loss, tokenizer fingerprint, `max_tokens_cap` |
| GET | `/stats` | yes | uptime, request / reject / error counts, prompt and completion tokens, tokens/s, in-flight and waiting |
| POST | `/generate` | yes | continue raw text |
| POST | `/chat` | yes | stateless chat: the client sends the whole message history |
| GET | `/` | no | bundled web chat UI |
| GET | `/v1/models`, `/v1/models/{id}` | yes | the served model (OpenAI format) |
| POST | `/v1/completions` | yes | text completion (OpenAI format) |
| POST | `/v1/chat/completions` | yes | chat (OpenAI format), `stream: true` supported |

### Sampling fields (both POST endpoints)

| Field | Default | Limits |
|---|---|---|
| `max_tokens` | 128 | ≥ 1; silently capped to the server's `max_tokens_cap` |
| `temperature` | 0.8 | 0 – 2 (0 = greedy) |
| `top_k` | 50 | 1 – 100,000 or `null` |
| `top_p` | 0.95 | (0, 1] or `null` |
| `repetition_penalty` | 1.0 | 0.5 – 2 |
| `no_repeat_ngram_size` | 0 | 0 – 8 |
| `stop` | `[]` | up to 4 strings |
| `seed` | `null` | integer; makes sampling reproducible |
| `stream` | `false` | server-sent events |

`/generate` also takes `prompt` (1 – 20,000 characters). `/chat` takes `messages`: 1 – 64 items of
`{"role": "system"|"user"|"assistant", "content": "..."}` (each ≤ 20,000 characters). If the history
does not fit the model's context window, the oldest turns are dropped and `dropped_messages` says how many.

### Request and response

```bash
curl -s http://127.0.0.1:8000/chat \
  -H "Authorization: Bearer $AARAV_KEY" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me a story about a fox named Kip."}],
       "max_tokens":150, "temperature":0.7}'
```

```json
{
  "text": "Once upon a time, there was a little fox named Kip. ...",
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 21, "completion_tokens": 118},
  "tokens_per_sec": 402.3,
  "model": "exp_002_sft_tinystories",
  "mode": "chat",
  "dropped_messages": 0
}
```

`finish_reason` is `stop` (the model ended, or a stop sequence matched), `length` (hit `max_tokens`) or `timeout` (the
request used up its wall-clock budget, `--request-timeout`, default 60 s; the partial text is returned and the model slot is
freed). On the OpenAI-compatible routes the values are only `stop` (which also covers the model's own end-of-text) and `length` (a timeout is reported as `length`, the closest valid value).

### Streaming

With `"stream": true` the response is `text/event-stream`:

```
data: {"delta": "Once"}
data: {"delta": " upon a time"}
...
data: {"done": true, "finish_reason": "stop", "usage": {...}, "tokens_per_sec": 398.0, "model": "...", "mode": "chat"}
```

An error after the stream has started is sent as `data: {"error": "...", "status": 500}`. Closing the
connection stops generation and frees the model.

### Python

```python
import requests
r = requests.post("http://127.0.0.1:8000/chat",
                  headers={"Authorization": f"Bearer {KEY}"},
                  json={"messages": [{"role": "user", "content": "Tell me a story about a dragon."}], "max_tokens": 120})
print(r.json()["text"])
```

### JavaScript

```javascript
const r = await fetch("http://127.0.0.1:8000/chat", {
  method: "POST",
  headers: { "Authorization": `Bearer ${KEY}`, "Content-Type": "application/json" },
  body: JSON.stringify({ messages: [{ role: "user", content: "Tell me a story about a dragon." }], max_tokens: 120 }),
});
console.log((await r.json()).text);
```

## Errors

| Status | Meaning |
|---|---|
| 401 | missing or invalid key |
| 413 | request body over 1 MB (also enforced for chunked uploads) |
| 422 | invalid field, empty prompt after tokenisation, or an invalid chat history |
| 429 | rate limit; `Retry-After` gives seconds |
| 500 | generation failed (details are not returned to the client) |
| 503 | the model is busy: one generation runs at a time, others wait up to 30 s; `Retry-After: 5` |

Body: `{"detail": "..."}` (or `{"error": "..."}` for 413).

## Limits

One generation at a time; at most a small number of queued waiters; output capped at `max_tokens_cap`; each generation
cut off after `--request-timeout` seconds;
body ≤ 1 MB; text fields ≤ 20,000 characters; rate limit per key (per client IP when auth is off).
The model's context window is 256 tokens for `tiny` and 512 for `small`.

## OpenAI-compatible routes

Point any OpenAI client at the server by changing `base_url`; the key is one made by `scripts/create_key.py`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="aak_...")
r = client.chat.completions.create(
    model="aarav-lm",                                   # or the experiment name that GET /v1/models returns
    messages=[{"role": "user", "content": "Tell me a story about a fox named Kip."}],
    max_tokens=150, temperature=0.7,
)
print(r.choices[0].message.content, r.usage.total_tokens)

for chunk in client.chat.completions.create(model="aarav-lm", stream=True, max_tokens=100,
                                            messages=[{"role": "user", "content": "A story about a dragon."}]):
    print(chunk.choices[0].delta.content or "", end="")
```

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $AARAV_KEY" -H "Content-Type: application/json" \
  -d '{"model":"aarav-lm","messages":[{"role":"user","content":"Tell me a story about a fox."}],"max_tokens":100}'
```

Responses follow OpenAI's objects (`chat.completion`, `chat.completion.chunk`, `text_completion`, `list`/`model`)
with `usage.prompt_tokens / completion_tokens / total_tokens`. A stream is `data: {chunk}` frames, a final chunk
carrying `finish_reason` and the token counts (`usage`, an extension), then `data: [DONE]`. The streamed text reassembles to exactly the non-streamed answer
(tested).

| Request field | Behaviour |
|---|---|
| `model` | must be the served experiment name or the alias `aarav-lm`, else **404** `model_not_found` (omitting it is allowed) |
| `messages` | roles `system`, `developer` (treated as system), `user`, `assistant`; `content` may be a string or a list of `{"type":"text"}` parts; images/audio → 400 |
| `prompt` (completions) | one string, or a list with exactly one string |
| `max_tokens` / `max_completion_tokens` | default 128; capped by the server (`--max-tokens-cap`) |
| `temperature`, `top_p`, `seed`, `stop` (string or ≤ 4 strings), `stream` | as in OpenAI; **if `temperature` is omitted the server default 0.8 applies, not 1.0** |
| `top_k`, `repetition_penalty` | extensions |
| `n` | must be 1 (400 otherwise) |
| `frequency_penalty`, `presence_penalty`, `logit_bias`, `logprobs`, `user`, `tools`, ... | accepted and **ignored** (this model cannot call tools) |

Errors on `/v1/...` paths are OpenAI's object: `{"error": {"message", "type", "param", "code"}}` with types
`invalid_request_error` (400/404/413), `authentication_error` (401, code `invalid_api_key`), `rate_limit_error`
(429, code `rate_limit_exceeded`, with `Retry-After`) and `server_error` (503 busy with `Retry-After`, 500).
The native routes keep their `{"detail": ...}` shape. The SDK tests are in `tests/test_openai_compat.py`.

## Accounts, limits and usage (`--db`)

With `--db` the server uses the account database instead of the JSON key file (`scripts/manage_keys.py --help`;
`import-json` carries existing keys over). Schema and rules are in `src/aarav_lm/api/store.py`:

| Table | Holds |
|---|---|
| `users` | email (the only personal datum), created, disabled |
| `api_keys` | SHA-256 of the key (never the key), a 8-character prefix for recognising it in lists, label, expiry, revoked-at, rotated-from, `rpm_limit`, `tpm_limit`, last-used |
| `usage` | one row per generation request: time, key, user, model, endpoint, prompt tokens, completion tokens, duration, HTTP status, error type. **No prompts or completions.** |
| `models` | models this server has served (name, parameters, context length) |

* **Revocation, rotation, expiry and disabling a user take effect on the very next request** (checked in the database
  on every call). Rotation issues a replacement with the same label and limits and revokes the old key in one transaction.
* **Per-key limits:** `rpm_limit` replaces the server-wide requests-per-minute for that key; `tpm_limit` caps prompt +
  completion tokens per rolling minute, measured from the usage table. A request is refused (429, `Retry-After`) once the
  key is already at its limit, so a key can overshoot by at most one request.
* **Every request leaves exactly one usage row**, whichever way it ends: 200 (`error_type` `timeout` if it was cut off), 429 (`rate_limited` / `token_limit`), 503
  (`busy`), 500 (`generation_failed`), or 499 (`client_disconnected`, a stream the client abandoned). A failing database
  never fails a request; it is counted in `GET /stats` as `usage_write_failures`.
* `GET /v1/usage?days=7` returns the caller's own totals: `{"object": "usage", "key_id", "requests", "prompt_tokens",
  "completion_tokens", "total_tokens", "errors", "mean_duration_ms", "latency_p50_ms", "latency_p95_ms"}` (the percentiles cover successful
  requests only and are `null` when there are none). With the JSON key file it is a 404.

## Account endpoints (the web app's)

Used by the browser app; documented in [PLATFORM.md](PLATFORM.md): `GET /app/config`, `GET /account/session`, `POST
/account/signup|login|logout|password`, `GET|POST /account/keys`, `POST /account/keys/{id}/revoke|rotate`, `GET
/account/usage`. They authenticate with a session cookie and require an `X-CSRF-Token` header on every POST; the API routes
above authenticate with `Authorization: Bearer`, or with the same session cookie from a signed-in browser.

## Public endpoints for the web app

* `GET /status.json` (no key): `{"overall", "services": {"api", "inference", "authentication", "playground"}: {"state", "detail"}, "model",
  "uptime_seconds", "queue": {"waiting", "limit", "in_flight"}, "checked_at"}`; states are `operational`, `degraded`, `outage`. Built from the server's
  own checks (queue against its limit, scheduler thread, database round trip, web files); counts and states only, never user data.
* `GET /app/config` (no key): the sign-in mode, limits, and the loaded model's real `architecture` (layers, width, heads, key/value heads, feed-forward size,
  vocabulary, context, norm, activation, positions, tied embeddings, parameter counts) and whether the demo is on.
* `POST /demo/chat` (only with `--public-demo`; no key): `{"prompt": "..."}` up to 300 characters, streamed in the chat-completion chunk format.
  Rate limited per client address (burst 3, then `--demo-per-minute`, default 6; `429` with `Retry-After`), output capped at 120 tokens, sampling fixed,
  never recorded per user. A busy model answers `503`.

## Not built yet

Email verification and password reset; several concurrent generations (one runs at a time); a shared rate-limit store for
more than one server process.
