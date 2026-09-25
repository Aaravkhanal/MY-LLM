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

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from aarav_lm.api.auth import KeyStore, RateLimiter
from aarav_lm.inference.chat_format import ChatFormatError, Message
from aarav_lm.inference.engine import GenerationConfig, InferenceEngine
from aarav_lm.inference.prompting import build_prompt, resolve_mode

MAX_BODY_BYTES = 1_000_000
MAX_TEXT_CHARS = 20_000


class Sampling(BaseModel):
    max_tokens: int = Field(128, ge=1)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(50, ge=1, le=100_000)
    top_p: Optional[float] = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.5, le=2.0)
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
                "in_flight": self.in_flight,
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
    cors_origins: tuple = (),
    model_name: str = "aarav-lm",
    mode: str = "auto",
) -> FastAPI:
    if require_auth and (key_store is None or len(key_store) == 0):
        raise RuntimeError("authentication is on but there are no API keys — run scripts/create_key.py "
                           "or start the server with --no-auth (localhost only)")
    app = FastAPI(title="aarav-lm", version="0.1.0", docs_url=None, redoc_url=None)
    stats = Stats()
    limiter = RateLimiter(rate_limit_per_min, rate_limit_burst)
    chat_mode = resolve_mode(engine, mode)

    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(cors_origins), allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type"])

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse({"error": "request body too large"}, status_code=413)
        return await call_next(request)

    def identify(request: Request) -> str:
        """Authenticate (if required) and rate-limit. Returns the caller identity."""
        identity = request.client.host if request.client else "unknown"
        if require_auth:
            header = request.headers.get("authorization", "")
            token = header[7:].strip() if header.lower().startswith("bearer ") else None
            key_id = key_store.verify(token) if key_store else None
            if key_id is None:
                stats.add(rejected=1)
                raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
            identity = f"key:{key_id}"
        ok, retry = limiter.allow(identity)
        if not ok:
            stats.add(rejected=1)
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(int(retry) + 1)})
        return identity

    def make_config(body: Sampling, stop_ids: tuple = ()) -> GenerationConfig:
        try:
            return GenerationConfig(
                max_new_tokens=min(body.max_tokens, max_tokens_cap), temperature=body.temperature,
                top_k=body.top_k, top_p=body.top_p, repetition_penalty=body.repetition_penalty,
                stop_sequences=tuple(s for s in body.stop if s), stop_ids=stop_ids, seed=body.seed,
            ).validate()
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @contextmanager
    def generation_slot():
        if not engine.lock.acquire(timeout=queue_timeout):
            stats.add(rejected=1)
            raise HTTPException(503, "model is busy, try again shortly", headers={"Retry-After": "5"})
        stats.add(in_flight=1)
        try:
            yield
        finally:
            stats.add(in_flight=-1)
            engine.lock.release()

    def run(prompt_ids: List[int], cfg: GenerationConfig, stream: bool, extra: dict):
        stats.add(requests=1, prompt_tokens=len(prompt_ids))

        def finish(result) -> dict:
            stats.add(completion_tokens=result.generated_tokens, gen_seconds=result.seconds)
            return {
                "finish_reason": result.finish_reason,
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": result.generated_tokens},
                "tokens_per_sec": round(result.tokens_per_sec, 1), "model": model_name, **extra,
            }

        if not stream:
            with generation_slot():
                try:
                    for _ in engine.stream(prompt_ids, cfg):
                        pass
                except Exception:                                    # noqa: BLE001
                    stats.add(errors=1)
                    raise HTTPException(500, "generation failed") from None
                result = engine.last_result
            return {"text": result.text, **finish(result)}

        with generation_slot():        # fail with a real 503 *before* the 200 stream starts if the model is busy
            pass

        def events() -> Iterator[str]:
            try:
                with generation_slot():
                    for delta in engine.stream(prompt_ids, cfg):
                        yield f"data: {json.dumps({'delta': delta})}\n\n"
                    yield f"data: {json.dumps({'done': True, **finish(engine.last_result)})}\n\n"
            except HTTPException as e:
                yield f"data: {json.dumps({'error': e.detail, 'status': e.status_code})}\n\n"
            except Exception:                                        # noqa: BLE001
                stats.add(errors=1)
                yield f"data: {json.dumps({'error': 'generation failed', 'status': 500})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True}

    @app.get("/model")
    def model_info(_: str = Depends(identify)):
        info = engine.info()
        return {
            "name": model_name, "params": info["params"], "context_length": info["context_length"],
            "vocab_size": info["vocab_size"], "device": info["device"], "mode": chat_mode,
            "instruction_tuned": chat_mode == "chat", "stage": info.get("stage", "pretrain"),
            "step": info.get("step"), "val_loss": info.get("val_loss"),
            "tokenizer_fingerprint": info["tokenizer_fingerprint"], "max_tokens_cap": max_tokens_cap,
        }

    @app.get("/stats")
    def get_stats(_: str = Depends(identify)):
        return stats.snapshot()

    @app.post("/generate")
    def generate(body: GenerateRequest, _: str = Depends(identify)):
        cfg = make_config(body)
        ids = engine.tok.encode(body.prompt)
        if not ids:
            raise HTTPException(422, "prompt is empty after tokenisation")
        return run(ids, cfg, body.stream, {})

    @app.post("/chat")
    def chat(body: ChatRequest, _: str = Depends(identify)):
        cfg = make_config(body)
        try:
            prompt = build_prompt(engine, [Message(m.role, m.content) for m in body.messages],
                                  cfg.max_new_tokens, chat_mode)
        except ChatFormatError as e:
            raise HTTPException(422, str(e)) from e
        cfg = replace(cfg, stop_ids=tuple({*cfg.stop_ids, *prompt.stop_ids}))
        extra = {"mode": chat_mode, "dropped_messages": prompt.dropped}
        return run(prompt.ids, cfg, body.stream, extra)

    return app
