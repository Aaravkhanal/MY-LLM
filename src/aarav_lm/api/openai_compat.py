"""
aarav-lm · OpenAI-compatible routes

    GET  /v1/models                 the one model being served
    GET  /v1/models/{id}
    POST /v1/completions            {"prompt": "...", ...}
    POST /v1/chat/completions       {"messages": [...], ...}   (also "stream": true)

Enough of the OpenAI wire format that the official SDKs work by changing only `base_url`:

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="aak_...")
    client.chat.completions.create(model="aarav-lm", messages=[{"role": "user", "content": "Tell me a story."}])

Everything else (authentication, rate limits, the single-flight generation slot, size caps, disconnect
handling) is the same code as the native /generate and /chat routes: these routes only translate the
request into a `Sampling` and format the response. Errors on /v1/... paths use OpenAI's error object;
the native routes keep their own shape.

What is NOT supported, and how it fails: `n` other than 1 (400), several prompts per request (400), image or
audio content parts (400), tools / function calling (ignored: this model cannot call tools),
`frequency_penalty` / `presence_penalty` / `logit_bias` / `logprobs` (accepted and ignored). `repetition_penalty`
and `top_k` are accepted as extensions. If `temperature` is omitted the server default (0.8) applies, not OpenAI's 1.0.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import replace
from typing import Any, Callable, List, Literal, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from aarav_lm.inference.chat_format import ChatFormatError, Message
from aarav_lm.inference.prompting import build_prompt

MODEL_ALIAS = "aarav-lm"
MAX_TEXT_CHARS = 20_000

# status -> (OpenAI error type, default code)
_ERROR_TYPES = {
    400: ("invalid_request_error", None), 401: ("authentication_error", "invalid_api_key"),
    404: ("invalid_request_error", "not_found"), 413: ("invalid_request_error", "request_too_large"),
    422: ("invalid_request_error", None), 429: ("rate_limit_error", "rate_limit_exceeded"),
    503: ("server_error", "server_busy"),
}


def _error_body(status: int, detail: Any, param: Optional[str] = None) -> dict:
    etype, code = _ERROR_TYPES.get(status, ("server_error", None))
    if isinstance(detail, dict):
        message, code = detail.get("message", ""), detail.get("code", code)
    else:
        message = detail if isinstance(detail, str) else json.dumps(detail)
    return {"error": {"message": message, "type": etype, "param": param, "code": code}}


# ── request schemas ──────────────────────────────────────────────────────────

class _Params(BaseModel):
    model_config = ConfigDict(extra="ignore")               # unknown OpenAI fields (tools, user, ...) are ignored

    model: Optional[str] = None
    max_tokens: Optional[int] = Field(None, ge=1)
    max_completion_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=100_000)                    # extension
    repetition_penalty: Optional[float] = Field(None, ge=0.5, le=2.0)       # extension
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = Field(None, ge=0, le=2**62)


class ChatMessageIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "developer", "user", "assistant"]
    content: Union[str, List[dict]]


class ChatCompletionRequest(_Params):
    messages: List[ChatMessageIn] = Field(..., min_length=1, max_length=64)


class CompletionRequest(_Params):
    prompt: Union[str, List[str]]


def _text_of(content: Union[str, List[dict]]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part.get("type") not in ("text", "input_text") or not isinstance(part.get("text"), str):
            raise HTTPException(400, "only text content parts are supported")
        parts.append(part["text"])
    return "".join(parts)


# ── response formats (same five methods as api.app.NativeFormat) ─────────────

def _usage(meta: dict) -> dict:
    u = meta["usage"]
    return {"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
            "total_tokens": u["prompt_tokens"] + u["completion_tokens"]}


class ChatFormat:
    def __init__(self, model: str):
        self.model, self.id, self.created = model, "chatcmpl-" + secrets.token_hex(12), int(time.time())

    def _frame(self, delta: dict, finish: Optional[str]) -> str:
        body = {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(body)}\n\n"

    def final(self, text: str, meta: dict) -> dict:
        return {"id": self.id, "object": "chat.completion", "created": self.created, "model": self.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": meta["finish_reason"]}],
                "usage": _usage(meta)}

    def first(self) -> str:
        return self._frame({"role": "assistant", "content": ""}, None)

    def chunk(self, delta: str) -> str:
        return self._frame({"content": delta}, None)

    def done(self, meta: dict) -> str:
        return self._frame({}, meta["finish_reason"]) + "data: [DONE]\n\n"

    def error(self, detail: Any, status: int) -> str:
        return f"data: {json.dumps(_error_body(status, detail))}\n\ndata: [DONE]\n\n"


class CompletionFormat(ChatFormat):
    def __init__(self, model: str):
        super().__init__(model)
        self.id = "cmpl-" + secrets.token_hex(12)

    def _frame(self, delta: dict, finish: Optional[str]) -> str:              # streaming chunks of a text completion
        body = {"id": self.id, "object": "text_completion", "created": self.created, "model": self.model,
                "choices": [{"index": 0, "text": delta.get("content", ""), "finish_reason": finish}]}
        return f"data: {json.dumps(body)}\n\n"

    def final(self, text: str, meta: dict) -> dict:
        return {"id": self.id, "object": "text_completion", "created": self.created, "model": self.model,
                "choices": [{"index": 0, "text": text, "finish_reason": meta["finish_reason"]}], "usage": _usage(meta)}

    def first(self) -> str:                                                     # completions have no role frame
        return ""


# ── wiring ───────────────────────────────────────────────────────────────────

def add_openai_routes(
    app: FastAPI, *, identify: Callable, make_config: Callable, run: Callable, engine, chat_mode: str,
    model_name: str, sampling_cls: type,
) -> None:
    created = int(time.time())
    known = {model_name, MODEL_ALIAS}

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        if not request.url.path.startswith("/v1"):
            return await http_exception_handler(request, exc)
        return JSONResponse(_error_body(exc.status_code, exc.detail), status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        if not request.url.path.startswith("/v1"):
            return await request_validation_exception_handler(request, exc)
        first = exc.errors()[0]
        param = ".".join(str(p) for p in first["loc"][1:]) or None
        return JSONResponse(_error_body(400, f"{param or 'request'}: {first['msg']}", param), status_code=400)

    def check_model(requested: Optional[str]) -> None:
        if requested is not None and requested not in known:
            raise HTTPException(404, {"message": f"The model '{requested}' does not exist; this server serves "
                                                 f"'{model_name}' (alias '{MODEL_ALIAS}').", "code": "model_not_found"})

    def to_sampling(body: _Params, max_default: int = 128):
        if body.n != 1:
            raise HTTPException(400, "only n=1 is supported")
        stop = [body.stop] if isinstance(body.stop, str) else (body.stop or [])
        if len(stop) > 4:
            raise HTTPException(400, "at most 4 stop sequences are supported")
        kw = {"max_tokens": body.max_completion_tokens or body.max_tokens or max_default, "stop": stop,
              "seed": body.seed, "stream": body.stream}
        for ours, theirs in (("temperature", body.temperature), ("top_p", body.top_p), ("top_k", body.top_k),
                             ("repetition_penalty", body.repetition_penalty)):
            if theirs is not None:
                kw[ours] = theirs
        return sampling_cls(**kw)

    def model_object() -> dict:
        info = engine.info()
        return {"id": model_name, "object": "model", "created": created, "owned_by": "aarav-lm",
                "context_length": info["context_length"], "parameters": info["params"],
                "instruction_tuned": chat_mode == "chat"}

    @app.get("/v1/models")
    async def list_models(_: str = Depends(identify)):
        return {"object": "list", "data": [model_object()]}

    @app.get("/v1/models/{model_id}")
    async def get_model(model_id: str, _: str = Depends(identify)):
        check_model(model_id)
        return model_object()

    @app.post("/v1/completions")
    def completions(body: CompletionRequest, caller: str = Depends(identify)):
        check_model(body.model)
        prompt = body.prompt
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise HTTPException(400, "exactly one prompt per request is supported")
            prompt = prompt[0]
        if not prompt or len(prompt) > MAX_TEXT_CHARS:
            raise HTTPException(400, f"prompt must be 1-{MAX_TEXT_CHARS} characters")
        cfg = make_config(to_sampling(body))
        ids = engine.tok.encode(prompt)
        if not ids:
            raise HTTPException(400, "prompt is empty after tokenisation")
        return run(ids, cfg, body.stream, {}, CompletionFormat(model_name), caller=caller, endpoint="/v1/completions")

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionRequest, caller: str = Depends(identify)):
        check_model(body.model)
        messages = []
        for m in body.messages:
            text = _text_of(m.content)
            if len(text) > MAX_TEXT_CHARS:
                raise HTTPException(400, f"message content is longer than {MAX_TEXT_CHARS} characters")
            messages.append(Message("system" if m.role == "developer" else m.role, text))
        cfg = make_config(to_sampling(body))
        try:
            prompt = build_prompt(engine, messages, cfg.max_new_tokens, chat_mode)
        except ChatFormatError as e:
            raise HTTPException(400, str(e)) from e
        cfg = replace(cfg, stop_ids=tuple({*cfg.stop_ids, *prompt.stop_ids}))
        return run(prompt.ids, cfg, body.stream, {}, ChatFormat(model_name), caller=caller, endpoint="/v1/chat/completions")
