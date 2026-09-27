# Security

Status of the controls for exposing the model to other people. **Nothing is deployed publicly.** The
server is safe to run on localhost today; the "Before going public" list is what is still missing.

## What exists

Most of these are covered by `tests/test_api.py` and `tests/test_review_fixes.py`. **Not covered by a
test:** the `--no-auth` refusal on non-local hosts (`scripts/serve.py`), the CORS configuration, the
constant-time comparison, and `weights_only` checkpoint loading in the API path.

| Control | Implementation |
|---|---|
| API keys never stored in plaintext | `api/auth.py`: `aak_` + 256 bits from `secrets.token_urlsafe`; only the SHA-256 is saved; the key is printed once by `scripts/create_key.py` |
| Constant-time key check | `hmac.compare_digest` over every stored entry (no early exit) |
| Key file permissions | created `0600`, written atomically, git-ignored (`secrets/`) |
| Rate limiting | per-key (per-IP without auth) token bucket, `429` + `Retry-After` |
| Input validation | pydantic bounds on every field; text ≤ 20,000 chars; ≤ 64 messages; body ≤ 1 MB, enforced for chunked uploads too |
| Output cap | `max_tokens_cap` (default 512) applied server-side |
| Bounded concurrency | one generation at a time; a bounded waiter queue; `503` after 30 s |
| Client disconnect frees the model | async event generator; regression test against a real socket |
| No code execution from checkpoints | `torch.load(weights_only=True)`; tokenizer text can't inject special tokens |
| Safe-by-default binding | `127.0.0.1`; `--no-auth` refused on any other host |
| Web accounts | scrypt password hashes, equal-time login for unknown emails, sign-in/signup rate limits, invite-gated signup, HttpOnly + SameSite=Strict + Secure cookies with only the session hash stored, CSRF token on every state-changing request, other users' keys look like missing ones, 5 active keys per user (`tests/test_web_accounts.py`) |
| Browser hardening for the bundled UI | strict CSP, `nosniff`, `no-referrer`, `frame-ancestors 'none'`; only files found under `web/` at start-up with an allowed extension are served, and only a fixed list of page paths gets the shell (`api/web_static.py`; tests for traversal and unknown paths); no inline script or style, no third-party host; no `innerHTML` anywhere |
| Public demo (opt-in) | `--public-demo`: one 300-character prompt, output capped, sampling fixed, per-address token bucket (burst 3, then 6 a minute), never attributed to a user, same queue and 503 behaviour as everyone else (`tests/test_demo.py`). Behind a proxy it needs `--trust-proxy` or every visitor shares one address |
| CORS | off unless `--cors-origin` is given; then only those origins, `GET`/`POST`, `Authorization`/`Content-Type` |
| Secrets in logs | keys are only ever in the `Authorization` header, which uvicorn's access log does not print; request bodies are not logged |

## Known gaps (do not expose publicly until these are closed)

1. **No TLS in the server itself.** It speaks plain HTTP; a reverse proxy terminates HTTPS. `deploy/Caddyfile` and
   `docker-compose.yml` do that with automatic certificates, but **they have not been built or run** (no Docker on the
   development machine): see [HOSTING.md](HOSTING.md).
2. ~~Keys are a JSON file.~~ **Closed with `--db`**: users, per-key rpm/tpm limits, expiry, revoke/rotate (effective on
   the next request), last-used, and self-service signup, sign-in and key management over HTTP with cookie sessions and CSRF
   protection ([PLATFORM.md](PLATFORM.md)). Not closed: no email verification or password reset (signup is gated by an invite
   code instead). The JSON key file remains as the simple mode.
3. **Rate limit is per process and in memory**; it resets on restart and does not span replicas.
4. ~~No tokens-per-minute limit~~ **Closed with `--db`** (per-key `tpm_limit`). The per-request wall-clock timeout is also
   closed (`--request-timeout`, default 60 s).
5. ~~No usage records~~ **Closed with `--db`**: one row per request (counts and outcome only; no prompt text). Still
   open: nothing reads them for abuse detection, and there is no retention policy.
6. **`/stats` and `/model` are visible to any valid key** (fine for one owner, not for many users); `/stats` shows
   server-wide counters.
7. **No content policy or output filtering.** The model is a small story model; a chat model exposed to
   the public needs a stated policy.
8. **Unauthenticated `/health`, `/status.json` and the web pages** are intentional (`/status.json` carries states, queue depth and uptime, no user data). The demo, if enabled, is a public way to use the model: cap it low and watch `/stats`.
9. **Dependency and supply-chain checks** are not automated (no lock file audit in CI).
10. **The `--no-auth` local mode** is one flag away; keep it out of any service definition.

## Before going public

Phase 4 in [AUDIT.md](AUDIT.md) closes gaps 2, 4, 5 and 6 (accounts, key management, TPM limits, usage
table). Gaps 1 and 3 belong to deployment (TLS proxy, shared limiter). The rest are decisions or CI work.
