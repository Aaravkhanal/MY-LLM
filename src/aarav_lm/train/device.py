"""
aarav-lm · Device selection, seeding, precision

Model weights are always fp32. "bf16" means torch.autocast around the forward pass
(matmuls in bf16, reductions and the optimizer in fp32), which halves activation
memory traffic without the loss-scaling machinery fp16 needs.
"""

from __future__ import annotations

import random
from contextlib import nullcontext

import numpy as np
import torch


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    dev = torch.device(name)
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    return dev


SMALL_MODEL_PARAMS = 50_000_000     # below this, accelerators lose to the CPU at batch-1 decoding


def pick_inference_device(name: str, n_params: int) -> torch.device:
    """Device for generation. For "auto", small models run on the CPU.

    Measured on an M3 (5M-parameter model, batch-1 decode with KV cache): CPU 732 tok/s vs MPS
    148 tok/s — each decoded token launches many tiny kernels and the GPU can't amortise that
    overhead. Training is the opposite (large batches: MPS 18k tok/s), so this applies only to
    inference. Pass an explicit device to override.
    """
    if name == "auto" and n_params < SMALL_MODEL_PARAMS:
        return torch.device("cpu")
    return pick_device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # seeds CPU, MPS and CUDA generators


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError(f"unknown precision {precision!r} (use 'fp32' or 'bf16')")


def synchronize(device: torch.device) -> None:
    """Block until queued device work is done (needed for honest timing)."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
