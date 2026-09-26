# API

Local HTTP API served by `scripts/serve.py` (FastAPI). This describes what **exists today**. The
OpenAI-style `/v1/...` routes, accounts and usage tracking are **planned, not built** — see
[AUDIT.md](AUDIT.md) (Phase 4).

```bash
python scripts/create_key.py --label "my laptop"        # prints a key once; only its SHA-256 is stored
python scripts/serve.py --checkpoint models/tiny-tinystories-v1-sft-int8.pt
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

`finish_reason` is `stop` (the model ended, or a stop sequence matched) or `length` (hit `max_tokens`).

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

One generation at a time; at most a small number of queued waiters; output capped at `max_tokens_cap`;
body ≤ 1 MB; text fields ≤ 20,000 characters; rate limit per key (per client IP when auth is off).
The model's context window is 256 tokens for `tiny` and 512 for `small`.

## Not built yet

`/v1/models`, `/v1/completions`, `/v1/chat/completions` (OpenAI-compatible request/response shapes,
so common SDKs work by changing `base_url`); accounts; listing, revoking and regenerating keys through
the API; per-user usage records; tokens-per-minute limits.
