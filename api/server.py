"""
aarav-lm · API Server
Author: Aarav Khanal

FastAPI inference server.  Start with:

    uvicorn aarav_lm.api.server:app --port 8000

Load a checkpoint before accepting requests by calling:

    POST /load  {"path": "aarav-lm/checkpoints/aarav_lm_modern_5m_best.pt"}
"""

import os
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aarav_lm.config.config import ModelConfig
from aarav_lm.model.core import AaravLM
from aarav_lm.tokenizer.tokenizer import Tokenizer

app     = FastAPI(title="aarav-lm API", version="1.0.0")
_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_model: Optional[AaravLM]  = None
_tok:   Optional[Tokenizer] = None


class LoadRequest(BaseModel):
    path: str


class GenerateRequest(BaseModel):
    prompt:      str   = Field(..., example="Once upon a time")
    max_tokens:  int   = Field(80,  ge=1, le=512)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k:       Optional[int]   = Field(40)
    top_p:       Optional[float] = Field(0.9)
    use_cache:   bool = True


class GenerateResponse(BaseModel):
    text:            str
    tokens_generated: int
    ms:              float
    tok_per_sec:     float
    device:          str


@app.get("/health")
def health():
    return {"loaded": _model is not None, "device": str(_device)}


@app.post("/load")
def load(req: LoadRequest):
    global _model, _tok
    if not os.path.exists(req.path):
        raise HTTPException(404, f"Checkpoint not found: {req.path}")
    ckpt    = torch.load(req.path, map_location=_device, weights_only=False)
    cfg     = ckpt["config"]
    _model  = AaravLM(cfg).to(_device)
    _model.load_state_dict(ckpt["model"])
    _model.eval()
    _tok    = Tokenizer()
    return {"status": "loaded", "device": str(_device)}


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    if _model is None:
        raise HTTPException(503, "No checkpoint loaded.  POST /load first.")
    ids = _tok.encode(req.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=_device)
    t0  = time.time()
    with torch.no_grad():
        out = _model.generate(idx, req.max_tokens, req.temperature, req.top_k, req.top_p, req.use_cache)
    dt  = time.time() - t0
    new = out.shape[1] - len(ids)
    return GenerateResponse(
        text=_tok.decode(out[0].tolist()),
        tokens_generated=new,
        ms=dt * 1000,
        tok_per_sec=new / dt if dt > 0 else 0,
        device=str(_device),
    )
