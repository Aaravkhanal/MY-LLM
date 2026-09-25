"""
aarav-lm · Resource and throughput monitoring

What can be measured on Apple Silicon without root:
  - process RSS and system memory pressure   (psutil)
  - Metal memory held by PyTorch             (torch.mps.*_allocated_memory)
What cannot: per-core GPU utilisation and power need `powermetrics` (sudo), so GPU
load is reported indirectly, through tokens/s and Metal memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import psutil
import torch

MB = 1024 * 1024


def memory_snapshot(device: Optional[torch.device] = None) -> Dict[str, float]:
    vm = psutil.virtual_memory()
    snap = {
        "rss_mb": round(psutil.Process().memory_info().rss / MB, 1),
        "sys_used_pct": vm.percent,
        "sys_available_mb": round(vm.available / MB, 1),
    }
    if device is not None and device.type == "mps":
        snap["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / MB, 1)
        snap["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / MB, 1)
    elif device is not None and device.type == "cuda":
        snap["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / MB, 1)
        snap["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated() / MB, 1)
    return snap


def tensors_nbytes(obj: torch.nn.Module) -> int:
    seen, total = set(), 0
    for p in obj.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel() * p.element_size()
    return total


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


@dataclass
class ThroughputTracker:
    """Steady-state tokens/s and ETA.

    Time spent in evaluation and checkpointing is excluded (call pause()/resume()),
    otherwise tokens/s is understated and the ETA is wrong.
    """

    smoothing: float = 0.9
    _t: float = field(default=0.0, init=False)
    _step_time_ema: Optional[float] = field(default=None, init=False)
    _tokens_ema: Optional[float] = field(default=None, init=False)
    _paused: bool = field(default=True, init=False)

    def start(self) -> None:
        self._t = time.perf_counter()
        self._paused = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._t = time.perf_counter()
        self._paused = False

    def tick(self, tokens: int) -> None:
        """Call once per optimizer step, after device work for it has finished."""
        now = time.perf_counter()
        if self._paused:
            return
        dt = now - self._t
        self._t = now
        a = self.smoothing
        self._step_time_ema = dt if self._step_time_ema is None else a * self._step_time_ema + (1 - a) * dt
        rate = tokens / dt if dt > 0 else 0.0
        self._tokens_ema = rate if self._tokens_ema is None else a * self._tokens_ema + (1 - a) * rate

    @property
    def tokens_per_sec(self) -> float:
        return self._tokens_ema or 0.0

    @property
    def step_seconds(self) -> float:
        return self._step_time_ema or 0.0

    def eta_seconds(self, steps_left: int) -> float:
        return steps_left * self.step_seconds
