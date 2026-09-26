"""
aarav-lm · Local inference API

    FastAPI  ->  InferenceEngine  ->  our model + tokenizer

    GET  /health    liveness, no auth
    GET  /model     model facts (params, context, tokenizer fingerprint, stage, step)
    GET  /stats     uptime, request and token counters, throughput
    POST /generate  {"prompt": ...}    continue text
    POST /chat      {"messages": [...]} stateless chat; the client sends the history
    both accept "stream": true for server-sent events:
        data: {"delta": "..."}   ...   data: {"done": true, "finish_reason": ..., "usage": {...}}

One generation runs at a time (the model lives on one 8 GB machine); other requests wait
up to `queue_timeout` seconds, then get 503 with Retry-After. Requests are validated,
capped (`max_tokens_cap`), rate-limited per key/IP, and require a bearer key unless
auth is explicitly disabled.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterator, List, Literal, Optional

import anyio.to_thread
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from aarav_lm.api.accounts_web import COOKIE, SAFE_METHODS, add_account_routes, client_ip, csrf_ok
from aarav_lm.api.auth import Caller, KeyStore, RateLimiter
from aarav_lm.api.openai_compat import add_openai_routes
from aarav_lm.inference.chat_format import ChatFormatError, Message
from aarav_lm.inference.engine import GenerationConfig, InferenceEngine
from aarav_lm.inference.scheduler import BatchScheduler, QueueFull
from aarav_lm.inference.prompting import build_prompt, resolve_mode

MAX_BODY_BYTES = 1_000_000
MAX_TEXT_CHARS = 20_000

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
# Only these files are ever served (no directory listing, no path traversal surface).
WEB_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                               "img-src 'self' data:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-cache",
}


class _TooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """Reject request bodies over `max_bytes`, including chunked ones with no Content-Length.

    The Content-Length header is checked up front; otherwise bytes are counted as the app reads
    them and the response is replaced with 413 (FastAPI would otherwise report a 400/422).
    """

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length", b"")
        if length.isdigit() and int(length) > self.max_bytes:
            return await self._reject(send, started=False)
        seen, exceeded, started = 0, False, False

        async def limited_receive():
            nonlocal seen, exceeded
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.max_bytes:
                    exceeded = True
                    raise _TooLarge()
            return msg

        async def guarded_send(msg):
            nonlocal started
            if exceeded:                                  # swallow whatever the app tried to answer with
                if not started:
                    await self._reject(send, started=False)
                    started = True
                return
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _TooLarge:
            if not started:
                await self._reject(send, started=False)

    @staticmethod
    async def _reject(send, started: bool) -> None:
        body = b'{"error":"request body too large"}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


class Sampling(BaseModel):
    max_tokens: int = Field(128, ge=1)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(50, ge=1, le=100_000)
    top_p: Optional[float] = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.5, le=2.0)
    no_repeat_ngram_size: int = Field(0, ge=0, le=8)
    stop: List[str] = Field(default_factory=list, max_length=4)
    seed: Optional[int] = Field(None, ge=0, le=2**62)
    stream: bool = False


class GenerateRequest(Sampling):
    prompt: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., max_length=MAX_TEXT_CHARS)


class ChatRequest(Sampling):
    messages: List[ChatMessage] = Field(..., min_length=1, max_length=64)


class NativeFormat:
    """The original /generate and /chat wire format. Other formats (api/openai_compat.py) implement the same five
    methods, so generation, queueing, streaming and cleanup live in exactly one place (`run` in create_app)."""

    def final(self, text: str, meta: dict) -> dict:
        return {"text": text, **meta}

    def chunk(self, delta: str) -> str:
        return f"data: {json.dumps({'delta': delta})}\n\n"

    def done(self, meta: dict) -> str:
        return f"data: {json.dumps({'done': True, **meta})}\n\n"

    def error(self, detail: str, status: int) -> str:
        return f"data: {json.dumps({'error': detail, 'status': status})}\n\n"


class Stats:
    def __init__(self) -> None:
        self.started = time.time()
        self.requests = 0
        self.rejected = 0
        self.errors = 0
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.gen_seconds = 0.0
        self.in_flight = 0
        self.waiting = 0
        self.usage_write_failures = 0
        self._lock = threading.Lock()

    def add(self, **kw: float) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_seconds": round(time.time() - self.started, 1),
                "requests": self.requests, "rejected": self.rejected, "errors": self.errors,
                "in_flight": self.in_flight, "waiting": self.waiting, "usage_write_failures": self.usage_write_failures,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "avg_tokens_per_sec": round(self.completion_tokens / self.gen_seconds, 1) if self.gen_seconds else 0.0,
            }


def create_app(
    engine: InferenceEngine,
    key_store: Optional[KeyStore] = None,
    require_auth: bool = True,
    max_tokens_cap: int = 512,
    rate_limit_per_min: float = 60.0,
    rate_limit_burst: int = 10,
    queue_timeout: float = 30.0,
    max_waiters: int = 8,
    request_timeout: float = 60.0,
    cors_origins: tuple = (),
    model_name: str = "aarav-lm",
    mode: str = "auto",
    serve_web: bool = True,
    signup_code: Optional[str] = None,
    open_signup: bool = False,
    web_rpm: float = 20.0,
    web_tpm: int = 30_000,
    default_key_rpm: float = 20.0,
    default_key_tpm: int = 30_000,
    max_keys_per_user: int = 5,
    cookie_secure: Optional[bool] = None,
    trust_proxy: bool = False,
    batch_size: int = 1,
    batch_wait_ms: float = 15.0,
    batch_queue: int = 64,
) -> FastAPI:
    can_sign_up = bool(signup_code) or open_signup
    has_users = key_store is not None and hasattr(key_store, "user_count") and key_store.user_count() > 0
    if require_auth and (key_store is None or (len(key_store) == 0 and not can_sign_up and not has_users)):
        raise RuntimeError("authentication is on but there are no API keys — run scripts/create_key.py "
                           "or start the server with --no-auth (localhost only)")
    shutdown_hooks: List = []                                      # things to close when the server stops (the batch scheduler)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for hook in shutdown_hooks:
            hook()

    app = FastAPI(title="aarav-lm", version="0.1.0", docs_url=None, redoc_url=None, lifespan=lifespan)
    stats = Stats()
    limiter = RateLimiter(rate_limit_per_min, rate_limit_burst)
    chat_mode = resolve_mode(engine, mode)

    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(cors_origins), allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type"])

    app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_BODY_BYTES)

    # The account database (api/store.py) adds per-key limits and usage records; the JSON KeyStore only authenticates.
    accounts = key_store if hasattr(key_store, "record_usage") else None
    if accounts is not None:
        accounts.register_model(model_name, "", engine.info()["params"], engine.info()["context_length"])

    def record(caller, model_id: str, endpoint: str, status: int, error_type: Optional[str] = None,
               prompt_tokens: int = 0, completion_tokens: int = 0, duration_ms: int = 0) -> None:
        """One usage row. Recording must never fail a request, but a failure is counted, not hidden (/stats)."""
        if accounts is None or (getattr(caller, "key_id", None) is None and getattr(caller, "user_id", None) is None):
            return
        try:
            accounts.record_usage(key_id=caller.key_id, user_id=caller.user_id, model_id=model_id, endpoint=endpoint,
                                  prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, duration_ms=duration_ms,
                                  status=status, error_type=error_type)
        except Exception:                                                        # noqa: BLE001
            stats.add(usage_write_failures=1)

    async def identify(request: Request) -> Caller:
        """Authenticate (if required) and rate-limit. Returns who is calling (a str identity with the key's limits)."""
        caller = Caller(client_ip(request, trust_proxy))
        if require_auth:
            header = request.headers.get("authorization", "")
            token = header[7:].strip() if header.lower().startswith("bearer ") else None
            session = accounts.get_session(request.cookies.get(COOKIE)) if (accounts is not None and not header) else None
            if session is not None:                                     # a signed-in browser (the web chat)
                if request.method not in SAFE_METHODS and not csrf_ok(request, session):
                    raise HTTPException(403, "missing or invalid CSRF token")
                caller = Caller(f"user:{session.user_id}", None, session.user_id, web_rpm, web_tpm)
            elif key_store is not None and hasattr(key_store, "lookup"):
                p = key_store.lookup(token)
                caller = Caller(f"key:{p.key_id}", p.key_id, p.user_id, p.rpm_limit, p.tpm_limit) if p else None
            else:
                key_id = key_store.verify(token) if key_store else None
                caller = Caller(f"key:{key_id}") if key_id is not None else None
            if caller is None:
                stats.add(rejected=1)
                raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
        ok, retry = limiter.allow(caller, caller.rpm_limit)
        if not ok:
            stats.add(rejected=1)
            record(caller, model_name, request.url.path, 429, "rate_limited")
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(int(retry) + 1)})
        used = 0
        if accounts is not None and caller.tpm_limit:
            used = (accounts.tokens_used_since(caller.key_id, 60) if caller.key_id is not None
                    else accounts.tokens_used_by_user_since(caller.user_id, 60) if caller.user_id is not None else 0)
        if caller.tpm_limit and used >= caller.tpm_limit:
            stats.add(rejected=1)
            record(caller, model_name, request.url.path, 429, "token_limit")
            raise HTTPException(429, f"token limit of {caller.tpm_limit} per minute reached", headers={"Retry-After": "30"})
        return caller

    def make_config(body: Sampling, stop_ids: tuple = ()) -> GenerationConfig:
        try:
            return GenerationConfig(
                max_new_tokens=min(body.max_tokens, max_tokens_cap), temperature=body.temperature,
                top_k=body.top_k, top_p=body.top_p, repetition_penalty=body.repetition_penalty,
                no_repeat_ngram_size=body.no_repeat_ngram_size, max_seconds=request_timeout or None,
                stop_sequences=tuple(s for s in body.stop if s), stop_ids=stop_ids, seed=body.seed,
            ).validate()
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    def acquire_slot() -> None:
        """Wait (bounded) for the single generation slot. Runs in a worker thread."""
        with stats._lock:
            if stats.waiting >= max_waiters:            # never let queued requests exhaust the thread pool
                stats.rejected += 1
                raise HTTPException(503, "too many requests waiting, try again shortly", headers={"Retry-After": "5"})
            stats.waiting += 1
        try:
            got = engine.lock.acquire(timeout=queue_timeout)
        finally:
            with stats._lock:
                stats.waiting -= 1
        if not got:
            stats.add(rejected=1)
            raise HTTPException(503, "model is busy, try again shortly", headers={"Retry-After": "5"})
        stats.add(in_flight=1)

    def release_slot() -> None:
        stats.add(in_flight=-1)
        engine.lock.release()

    # Batched serving (--batch-size > 1): concurrent requests share each decode step (inference/batching.py). A learned-position
    # model cannot be batched exactly, so it silently keeps the single-request path.
    scheduler = None
    if batch_size > 1 and engine.model.cfg.arch != "gpt2":
        scheduler = BatchScheduler(engine, max_batch=batch_size, max_wait=batch_wait_ms / 1000.0, max_queue=batch_queue)
        shutdown_hooks.append(scheduler.close)

    def begin(ticket, state: dict) -> None:
        """Wait (bounded) for the ticket's wave to start. Runs in a worker thread, like acquire_slot."""
        got = ticket.wait_started(queue_timeout)
        state["begun"] = True
        with stats._lock:
            stats.waiting -= 1
        if not got:
            ticket.cancel()
            stats.add(rejected=1)
            raise HTTPException(503, "model is busy, try again shortly", headers={"Retry-After": "5"})
        stats.add(in_flight=1)
        state["running"] = True

    def submit_ticket(prompt_ids: List[int], cfg: GenerationConfig, log_usage):
        try:
            ticket = scheduler.submit(prompt_ids, cfg)
        except QueueFull:
            stats.add(rejected=1)
            log_usage(503, "busy")
            raise HTTPException(503, "too many requests waiting, try again shortly", headers={"Retry-After": "5"}) from None
        stats.add(waiting=1)
        return ticket

    def run_batched(prompt_ids: List[int], cfg: GenerationConfig, stream: bool, fmt, log_usage, finish):
        ticket = submit_ticket(prompt_ids, cfg, log_usage)
        state = {"begun": False, "running": False}

        def cleanup() -> None:
            ticket.cancel()
            if not state["begun"]:
                stats.add(waiting=-1)
            if state["running"]:
                stats.add(in_flight=-1)

        def done_ok(result) -> str:
            return "timeout" if result.finish_reason == "timeout" else None

        if not stream:
            try:
                try:
                    begin(ticket, state)
                except HTTPException as e:
                    log_usage(e.status_code, "busy")
                    raise
                try:
                    for _ in ticket.deltas():
                        pass
                    result = ticket.result
                    if result is None:
                        raise RuntimeError("no result")
                except Exception:                                    # noqa: BLE001
                    stats.add(errors=1)
                    log_usage(500, "generation_failed")
                    raise HTTPException(500, "generation failed") from None
            finally:
                cleanup()
            log_usage(200, done_ok(result), result.generated_tokens)
            return fmt.final(result.text, finish(result))

        end = object()

        async def events():
            it = None
            try:
                await anyio.to_thread.run_sync(begin, ticket, state)
                it = ticket.deltas()
                if hasattr(fmt, "first"):
                    yield fmt.first()
                while True:
                    delta = await anyio.to_thread.run_sync(next, it, end)
                    if delta is end:
                        break
                    yield fmt.chunk(delta)
                result = ticket.result
                if result is None:
                    raise RuntimeError("no result")
                log_usage(200, done_ok(result), result.generated_tokens)
                yield fmt.done(finish(result))
            except HTTPException as e:
                log_usage(e.status_code, "busy" if e.status_code == 503 else "error")
                yield fmt.error(e.detail, e.status_code)
            except Exception:                                        # noqa: BLE001
                stats.add(errors=1)
                log_usage(500, "generation_failed")
                yield fmt.error("generation failed", 500)
            finally:
                log_usage(499, "client_disconnected")               # no-op unless the stream ended without finishing
                if it is not None:
                    it.close()
                cleanup()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def run(prompt_ids: List[int], cfg: GenerationConfig, stream: bool, extra: dict, fmt=None, caller=None, endpoint: str = ""):
        fmt = fmt or NativeFormat()
        stats.add(requests=1, prompt_tokens=len(prompt_ids))
        started = time.perf_counter()
        outcome = {"done": False}

        def log_usage(status: int, error_type: Optional[str] = None, completion_tokens: int = 0) -> None:
            if outcome["done"]:
                return
            outcome["done"] = True
            record(caller, model_name, endpoint, status, error_type, len(prompt_ids), completion_tokens,
                   int((time.perf_counter() - started) * 1000))

        def finish(result) -> dict:
            stats.add(completion_tokens=result.generated_tokens, gen_seconds=result.seconds)
            return {
                "finish_reason": result.finish_reason,
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": result.generated_tokens},
                "tokens_per_sec": round(result.tokens_per_sec, 1), "model": model_name, **extra,
            }

        if scheduler is not None:
            return run_batched(prompt_ids, cfg, stream, fmt, log_usage, finish)

        if not stream:
            try:
                acquire_slot()
            except HTTPException as e:
                log_usage(e.status_code, "busy")
                raise
            try:
                for _ in engine.stream(prompt_ids, cfg):
                    pass
                result = engine.last_result
            except Exception:                                        # noqa: BLE001
                stats.add(errors=1)
                log_usage(500, "generation_failed")
                raise HTTPException(500, "generation failed") from None
            finally:
                release_slot()
            log_usage(200, "timeout" if result.finish_reason == "timeout" else None, result.generated_tokens)
            return fmt.final(result.text, finish(result))

        # Fail with a real 503 *before* the 200 stream starts if the model is busy.
        try:
            acquire_slot()
        except HTTPException as e:
            log_usage(e.status_code, "busy")
            raise
        release_slot()

        end = object()

        async def events():
            """Async on purpose: on client disconnect Starlette cancels this generator, so `finally`
            always runs and the slot is released. (A sync generator is only closed by the garbage
            collector, which left the model locked after every "Stop" click.)"""
            held = {"slot": False}
            it = None

            def take_slot() -> None:
                acquire_slot()
                held["slot"] = True                # set inside the thread: cancellation can't lose it

            try:
                await anyio.to_thread.run_sync(take_slot)
                it = engine.stream(prompt_ids, cfg)
                if hasattr(fmt, "first"):
                    yield fmt.first()
                while True:
                    delta = await anyio.to_thread.run_sync(next, it, end)
                    if delta is end:
                        break
                    yield fmt.chunk(delta)
                log_usage(200, "timeout" if engine.last_result.finish_reason == "timeout" else None,
                          engine.last_result.generated_tokens)
                yield fmt.done(finish(engine.last_result))
            except HTTPException as e:
                log_usage(e.status_code, "busy" if e.status_code == 503 else "error")
                yield fmt.error(e.detail, e.status_code)
            except Exception:                                        # noqa: BLE001
                stats.add(errors=1)
                log_usage(500, "generation_failed")
                yield fmt.error("generation failed", 500)
            finally:
                log_usage(499, "client_disconnected")               # no-op unless the stream ended without finishing
                if it is not None:
                    it.close()
                if held["slot"]:
                    release_slot()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if serve_web:
        def web_route(filename: str, media_type: str):
            body = (WEB_DIR / filename).read_bytes()          # read once at start-up; assets are static

            def handler() -> Response:
                return Response(body, media_type=media_type, headers=SECURITY_HEADERS)
            return handler

        for route, (filename, media_type) in WEB_FILES.items():
            app.add_api_route(route, web_route(filename, media_type), methods=["GET"], include_in_schema=False)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": True}

    @app.get("/model")
    async def model_info(_: str = Depends(identify)):
        info = engine.info()
        return {
            "name": model_name, "params": info["params"], "context_length": info["context_length"],
            "vocab_size": info["vocab_size"], "device": info["device"], "mode": chat_mode,
            "instruction_tuned": chat_mode == "chat", "stage": info.get("stage", "pretrain"),
            "step": info.get("step"), "val_loss": info.get("val_loss"),
            "tokenizer_fingerprint": info["tokenizer_fingerprint"], "max_tokens_cap": max_tokens_cap,
        }

    @app.get("/stats")
    async def get_stats(_: str = Depends(identify)):
        snap = stats.snapshot()
        if scheduler is not None:
            snap["batching"] = {"max_batch": batch_size, **scheduler.stats()}
        return snap

    @app.get("/v1/usage")
    async def my_usage(days: float = 30.0, caller: Caller = Depends(identify)):
        """The calling key's own usage (token counts and request outcomes; prompts are never stored)."""
        if accounts is None or caller.key_id is None:
            raise HTTPException(404, "usage tracking needs the account database (--db) and an API key")
        if not 0 < days <= 366:
            raise HTTPException(400, "days must be between 0 and 366")
        return {"object": "usage", "key_id": caller.key_id, **accounts.usage_summary(key_id=caller.key_id, days=days)}

    @app.post("/generate")
    def generate(body: GenerateRequest, caller: Caller = Depends(identify)):
        cfg = make_config(body)
        ids = engine.tok.encode(body.prompt)
        if not ids:
            raise HTTPException(422, "prompt is empty after tokenisation")
        return run(ids, cfg, body.stream, {}, caller=caller, endpoint="/generate")

    @app.post("/chat")
    def chat(body: ChatRequest, caller: Caller = Depends(identify)):
        cfg = make_config(body)
        try:
            prompt = build_prompt(engine, [Message(m.role, m.content) for m in body.messages],
                                  cfg.max_new_tokens, chat_mode)
        except ChatFormatError as e:
            raise HTTPException(422, str(e)) from e
        cfg = replace(cfg, stop_ids=tuple({*cfg.stop_ids, *prompt.stop_ids}))
        extra = {"mode": chat_mode, "dropped_messages": prompt.dropped}
        return run(prompt.ids, cfg, body.stream, extra, caller=caller, endpoint="/chat")

    auth_mode = "none" if not require_auth else ("account" if accounts is not None else "key")
    if accounts is not None:
        add_account_routes(app, accounts, model_name=model_name, auth_mode=auth_mode, signup_code=signup_code, open_signup=open_signup,
                           default_key_rpm=default_key_rpm, default_key_tpm=default_key_tpm, web_rpm=web_rpm, web_tpm=web_tpm,
                           max_keys_per_user=max_keys_per_user, cookie_secure=cookie_secure, trust_proxy=trust_proxy)
    else:
        @app.get("/app/config")
        async def app_config():
            return {"auth": auth_mode, "signup": "closed", "model": model_name, "limits": {}}
    add_openai_routes(app, identify=identify, make_config=make_config, run=run, engine=engine, chat_mode=chat_mode,
                      model_name=model_name, sampling_cls=Sampling)
    return app
