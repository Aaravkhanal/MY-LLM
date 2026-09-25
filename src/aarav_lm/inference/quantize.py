"""
aarav-lm · int8 weight quantization for checkpoints

Symmetric per-row int8: each row of a weight matrix gets scale = max|w| / 127 and is
stored as int8 (a tied embedding / LM head is stored once). Norm scales and other
1-D tensors stay fp32. The result is ~3.7x smaller on disk.

What this does and does not buy on an Apple-silicon Mac: PyTorch/MPS has no int8
matmul, so weights are dequantized to fp32 at load time. You get smaller files and
faster loading/transfer, NOT faster or lower-memory inference. (Real int8 inference
would need a kernel backend, e.g. MLX or llama.cpp-style ggml; see docs.)

Loadable anywhere a normal checkpoint is: train.checkpoint.load_model() detects
kind == "model_int8" and dequantizes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from aarav_lm.model.core import AaravLM

MIN_NUMEL = 1024          # tiny tensors aren't worth quantizing


def quantize_tensor(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(int8 tensor, fp32 per-row scale). All-zero rows get scale 1 so they stay exactly zero."""
    w = w.detach().float()
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(1)


def dequantize_tensor(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float().unsqueeze(1)


def quantize_model(model: AaravLM) -> Dict[str, Any]:
    """Split a model's state dict into int8 matrices and fp32 leftovers."""
    quantized: Dict[str, Dict[str, torch.Tensor]] = {}
    floats: Dict[str, torch.Tensor] = {}
    ties: Dict[str, str] = {}
    seen: Dict[int, str] = {}
    for name, t in model.state_dict().items():
        key = t.data_ptr()
        if key in seen and t.dim() == 2:                 # tied weight: remember, don't store twice
            ties[name] = seen[key]
            continue
        seen[key] = name
        if t.dim() == 2 and t.numel() >= MIN_NUMEL and t.is_floating_point():
            q, s = quantize_tensor(t)
            quantized[name] = {"q": q.cpu(), "scale": s.cpu()}
        else:
            floats[name] = t.detach().cpu().clone()
    return {"quantized": quantized, "float": floats, "ties": ties}


def dequantize_state_dict(packed: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    sd = {k: v.clone() for k, v in packed["float"].items()}
    for name, qs in packed["quantized"].items():
        sd[name] = dequantize_tensor(qs["q"], qs["scale"])
    for name, target in packed["ties"].items():
        sd[name] = sd[target]
    return sd


def save_int8_checkpoint(
    path: Path, model: AaravLM, tokenizer_fingerprint: str, meta: Optional[dict] = None
) -> Dict[str, Any]:
    from aarav_lm.train.checkpoint import FORMAT_VERSION, atomic_save
    packed = quantize_model(model)
    atomic_save(
        {
            "format_version": FORMAT_VERSION, "kind": "model_int8", **packed,
            "model_config": model.cfg.to_dict(), "tokenizer_fingerprint": tokenizer_fingerprint,
            "meta": {**(meta or {}), "quantization": "int8-per-row"},
        },
        Path(path),
    )
    return {"path": str(path), "bytes": Path(path).stat().st_size,
            "quantized_tensors": len(packed["quantized"]), "float_tensors": len(packed["float"])}


def dequantize_checkpoint(raw: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """State dict (fp32) from a loaded model_int8 checkpoint dict."""
    return dequantize_state_dict(raw)


def max_relative_error(model: AaravLM) -> float:
    """Worst per-tensor relative Frobenius error introduced by quantization."""
    worst = 0.0
    for name, t in model.state_dict().items():
        if t.dim() == 2 and t.numel() >= MIN_NUMEL:
            q, s = quantize_tensor(t)
            err = (dequantize_tensor(q, s) - t.float()).norm() / t.float().norm().clamp_min(1e-12)
            worst = max(worst, float(err))
    return worst
