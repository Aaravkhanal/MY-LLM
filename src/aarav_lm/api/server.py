"""
aarav-lm · API server

Local OpenAI-compatible inference server for a trained AaravLM checkpoint.

    python scripts/serve.py

Other people call it with a bearer key created by scripts/create_key.py.
The server loads one checkpoint into memory and runs one generation at a time,
which is what an 8 GB Mac can sustain.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Union

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from aarav_lm.data.chat_format import decode_answer, messages_to_ids
from aarav_lm.model.core import AaravLM
from aarav_lm.tokenizer.tokenizer import Tokenizer

from aarav_lm.paths import CHECKPOINT_DIR as _CKPT, REPO_ROOT as ROOT, SECRETS_DIR
CKPT_DIR = _CKPT.resolve()
KEYS_FILE = SECRETS_DIR / "api_keys.txt"
CHAT_CKPT = CKPT_DIR / "aarav_chat.pt"
DEFAULT_CKPT = CKPT_DIR / "aarav_lm_modern_5m_best.pt"

_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_infer_lock = threading.Lock()
_model: Optional[AaravLM] = None
_tok: Optional[Tokenizer] = None
_ckpt_name = ""
_keys: set[str] = set()


def load_api_keys() -> set[str]:
    """Keys come from AARAV_API_KEYS (comma-separated) and secrets/api_keys.txt."""
    found: set[str] = set()
    for part in os.environ.get("AARAV_API_KEYS", "").split(","):
        part = part.strip()
        if part:
            found.add(part)
    if KEYS_FILE.exists():
        for line in KEYS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                found.add(line)
    return found


def _resolve_ckpt(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if CKPT_DIR != candidate and CKPT_DIR not in candidate.parents:
        raise HTTPException(400, "Checkpoint must be a file inside checkpoints/")
    if not candidate.is_file():
        raise HTTPException(404, f"Checkpoint not found: {candidate.name}")
    return candidate


def _load_checkpoint(path: Path) -> None:
    global _model, _tok, _ckpt_name
    try:
        ckpt = torch.load(path, map_location=_device, weights_only=False)
        cfg = ckpt["config"]
        model = AaravLM(cfg).to(_device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        tok = Tokenizer()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Could not load checkpoint: {exc}") from exc
    with _infer_lock:
        _model = model
        _tok = tok
        _ckpt_name = path.name


def _messages_to_prompt(messages: list["ChatMessage"]) -> str:
    chunks: list[str] = []
    for message in messages:
        text = _as_text(message.content).strip()
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _as_text(content: Union[str, list, Any]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in (None, "text"):
                    parts.append(str(part.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content)


def _generate(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    token_ids: Optional[list] = None,
    stop_ids: Optional[list] = None,
) -> dict:
    if token_ids is None and not prompt.strip():
        raise HTTPException(400, "Prompt is empty.")

    with _infer_lock:
        if _model is None or _tok is None:
            raise HTTPException(503, "No checkpoint loaded.")

        max_tokens = max(1, min(max_tokens, _model.cfg.block_size - 1, 128))
        ids = list(token_ids) if token_ids is not None else _tok.encode(prompt)
        if not ids:
            raise HTTPException(400, "Prompt is empty.")
        room = _model.cfg.block_size - max_tokens
        if room < 1:
            raise HTTPException(400, "max_tokens leaves no room for the prompt.")
        if len(ids) > room:
            ids = ids[-room:]

        idx = torch.tensor([ids], dtype=torch.long, device=_device)
        started = time.time()
        with torch.no_grad():
            out = _model.generate(
                idx,
                max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=True,
                stop_ids=stop_ids,
            )
        elapsed = time.time() - started
        new_ids = out[0, len(ids):].tolist()
        completion = decode_answer(_tok, new_ids) if stop_ids else _tok.decode(new_ids)
        full = _tok.decode(out[0].tolist())
        return {
            "text": full,
            "completion": completion,
            "prompt_tokens": len(ids),
            "tokens_generated": len(new_ids),
            "ms": elapsed * 1000,
            "tok_per_sec": (len(new_ids) / elapsed) if elapsed > 0 else 0.0,
            "device": str(_device),
            "model": _ckpt_name or "aarav-lm",
        }


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _keys
    _keys = load_api_keys()
    if not _keys:
        raise RuntimeError(
            "No API keys configured. Run: python scripts/create_key.py"
        )
    default_ckpt = CHAT_CKPT if CHAT_CKPT.exists() else DEFAULT_CKPT
    ckpt = os.environ.get("AARAV_CKPT", str(default_ckpt))
    _load_checkpoint(_resolve_ckpt(ckpt))
    print(f"aarav-lm ready  device={_device}  checkpoint={_ckpt_name}  keys={len(_keys)}")
    yield


app = FastAPI(title="aarav-lm API", version="1.1.0", lifespan=_lifespan)


def require_api_key(authorization: Optional[str] = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Send Authorization: Bearer <api-key>")
    key = authorization.split(" ", 1)[1].strip()
    if key not in _keys:
        raise HTTPException(401, "Invalid API key")
    return key


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt: str
    max_tokens: int = Field(80, ge=1, le=256)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(40, ge=1)
    top_p: Optional[float] = Field(0.9, gt=0.0, le=1.0)


class GenerateResponse(BaseModel):
    text: str
    completion: str
    tokens_generated: int
    ms: float
    tok_per_sec: float
    device: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str = "user"
    content: Union[str, list, Any] = ""


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "aarav-lm"
    messages: list[ChatMessage]
    max_tokens: Optional[int] = Field(80, ge=1, le=256)
    temperature: Optional[float] = Field(0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, gt=0.0, le=1.0)
    stream: bool = False


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "aarav-lm"
    prompt: Union[str, list[str]]
    max_tokens: Optional[int] = Field(80, ge=1, le=256)
    temperature: Optional[float] = Field(0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, gt=0.0, le=1.0)
    stream: bool = False


def _chat_payload(result: dict, model_name: str) -> dict:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["completion"]},
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["tokens_generated"],
            "total_tokens": result["prompt_tokens"] + result["tokens_generated"],
        },
    }


def _stream_chat(payload: dict):
    created = payload["created"]
    model_name = payload["model"]
    text = payload["choices"][0]["message"]["content"]
    base = {
        "id": payload["id"],
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
    }
    first = {
        **base,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    }
    last = {
        **base,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(first)}\n\n"
    yield f"data: {json.dumps(last)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
def health():
    return {
        "loaded": _model is not None,
        "device": str(_device),
        "checkpoint": _ckpt_name,
        "keys": len(_keys),
    }


@app.get("/v1/models")
def list_models(_: str = Depends(require_api_key)):
    name = _ckpt_name or "aarav-lm"
    return {
        "object": "list",
        "data": [
            {
                "id": "aarav-lm",
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "checkpoint": name,
            }
        ],
    }


@app.post("/load")
def load(req: LoadRequest, _: str = Depends(require_api_key)):
    _load_checkpoint(_resolve_ckpt(req.path))
    return {"status": "loaded", "checkpoint": _ckpt_name, "device": str(_device)}


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest, _: str = Depends(require_api_key)):
    result = _generate(req.prompt, req.max_tokens, req.temperature, req.top_k, req.top_p)
    return GenerateResponse(
        text=result["text"],
        completion=result["completion"],
        tokens_generated=result["tokens_generated"],
        ms=result["ms"],
        tok_per_sec=result["tok_per_sec"],
        device=result["device"],
    )


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, _: str = Depends(require_api_key)):
    if _tok is None:
        raise HTTPException(503, "No checkpoint loaded.")
    messages = [{"role": m.role, "content": _as_text(m.content)} for m in req.messages]
    ids = messages_to_ids(_tok, messages)
    result = _generate(
        "",
        req.max_tokens or 80,
        0.7 if req.temperature is None else req.temperature,
        top_k=40,
        top_p=req.top_p,
        token_ids=ids,
        stop_ids=[_tok.eot],
    )
    payload = _chat_payload(result, req.model or "aarav-lm")
    if req.stream:
        return StreamingResponse(_stream_chat(payload), media_type="text/event-stream")
    return payload


@app.post("/v1/completions")
def completions(req: CompletionRequest, _: str = Depends(require_api_key)):
    prompt = req.prompt[0] if isinstance(req.prompt, list) else req.prompt
    result = _generate(
        prompt,
        req.max_tokens or 80,
        0.8 if req.temperature is None else req.temperature,
        top_k=40,
        top_p=req.top_p,
    )
    body = {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model or "aarav-lm",
        "choices": [
            {"index": 0, "text": result["completion"], "finish_reason": "length"}
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["tokens_generated"],
            "total_tokens": result["prompt_tokens"] + result["tokens_generated"],
        },
    }
    if req.stream:
        chunk = {
            "id": body["id"],
            "object": "text_completion",
            "created": body["created"],
            "model": body["model"],
            "choices": [{"index": 0, "text": result["completion"], "finish_reason": "stop"}],
        }

        def events():
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")
    return body
