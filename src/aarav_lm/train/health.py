"""
aarav-lm · Training health diagnostics

Observe, never intervene. Nothing here changes a gradient, a weight, the data order or an RNG, so a
run with these diagnostics is bit-identical to one without them, and results stay comparable with
runs made before they existed. Problems are logged (`[anomaly]` / `[health]` lines) and written to
metrics.jsonl with enough detail to diagnose them; the only thing that stops a run is what already
did (non-finite loss, or five non-finite gradient steps in a row).

  SpikeDetector      a value far above its own recent history (robust: rolling median + MAD)
  gradient_health    gradient/parameter norms per layer group, measured before clipping
  OverfitDetector    validation loss rising while training loss keeps falling

The thresholds are heuristics, not laws. They are deliberately loose (few false alarms); a flag is a
reason to look at the plots, not proof of a problem.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Dict, List, Optional

import torch


class SpikeDetector:
    """Flags a value that exceeds `median + z * scale` of its previous `window` values.

    `scale` is 1.4826 * MAD (a robust standard deviation) but never smaller than `rel_floor * |median|`,
    so a very smooth series does not turn every tiny wobble into a "spike". Only upward moves are
    flagged: for both loss and gradient norm, down is good. Silent until `min_history` values exist.
    """

    def __init__(self, name: str, window: int = 50, z: float = 6.0, min_history: int = 20, rel_floor: float = 0.02):
        self.name, self.z, self.min_history, self.rel_floor = name, z, min_history, rel_floor
        self._hist: deque = deque(maxlen=window)

    def update(self, value: float) -> Optional[Dict[str, float]]:
        hist = list(self._hist)
        if math.isfinite(value):
            self._hist.append(value)
        if len(hist) < self.min_history or not math.isfinite(value):
            return None
        med = statistics.median(hist)
        mad = statistics.median(abs(h - med) for h in hist)
        scale = max(1.4826 * mad, self.rel_floor * abs(med), 1e-12)
        threshold = med + self.z * scale
        if value > threshold:
            return {"kind": self.name, "value": value, "median": med, "threshold": threshold,
                    "z": (value - med) / scale}
        return None


def group_of(param_name: str) -> str:
    """'blocks.3.attn.q.weight' -> 'blocks.3.attn'; 'token_emb.weight' -> 'token_emb'."""
    parts = param_name.split(".")
    if parts[0] == "blocks" and len(parts) > 2:
        sub = parts[2]
        return f"blocks.{parts[1]}." + (sub if sub in ("attn", "mlp") else "norm")
    return parts[0]


@torch.no_grad()
def gradient_health(model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    """Per layer group: L2 norm of the gradient, of the weights, and their ratio.

    Call after `backward()` and **before** clipping, so the numbers are the raw gradients. A group whose
    gradient/weight ratio is orders of magnitude below the others is barely learning; one orders of
    magnitude above is exploding. One device sync (a single `.tolist()`).
    """
    names: List[str] = []
    g_norms: List[torch.Tensor] = []
    p_norms: List[torch.Tensor] = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        names.append(group_of(name))
        g_norms.append(p.grad.detach().float().norm())
        p_norms.append(p.detach().float().norm())
    if not names:
        return {}
    g, w = torch.stack(g_norms).tolist(), torch.stack(p_norms).tolist()
    acc: Dict[str, List[float]] = {}
    for grp, gi, wi in zip(names, g, w):
        a = acc.setdefault(grp, [0.0, 0.0])
        a[0] += gi * gi
        a[1] += wi * wi
    out = {}
    for grp, (gg, ww) in acc.items():
        grad, weight = math.sqrt(gg), math.sqrt(ww)
        out[grp] = {"grad": grad, "param": weight, "ratio": grad / weight if weight > 0 else float("inf")}
    return out


def health_flags(stats: Dict[str, Dict[str, float]], explode: float = 100.0, vanish: float = 0.01) -> List[str]:
    """Human-readable problems in a `gradient_health` result.

    non-finite gradient; a group whose gradient norm is more than `explode` x the median group's
    (exploding) or less than `vanish` x the median (vanishing / barely learning).
    """
    if not stats:
        return []
    flags = [f"{g}: non-finite gradient" for g, s in stats.items() if not math.isfinite(s["grad"])]
    finite = [s["grad"] for s in stats.values() if math.isfinite(s["grad"])]
    if len(finite) >= 3:
        med = statistics.median(finite)
        if med > 0:
            for g, s in stats.items():
                if math.isfinite(s["grad"]) and s["grad"] > explode * med:
                    flags.append(f"{g}: gradient norm {s['grad']:.3g} is {s['grad'] / med:.0f}x the median group")
                elif math.isfinite(s["grad"]) and s["grad"] < vanish * med:
                    flags.append(f"{g}: gradient norm {s['grad']:.3g} is {s['grad'] / med:.4f}x the median group")
    return flags


class OverfitDetector:
    """Validation loss up and training loss down at each of the last `patience` consecutive evaluations.

    `train_loss` should be the mean training loss since the previous evaluation. Under SFT the
    validation loss is measured on a small fixed subset (see docs/CORRECTNESS.md, entry 1), so a single
    rise is noise; requiring `patience` rises in a row with falling training loss is the usual signature.
    """

    def __init__(self, patience: int = 2):
        self.patience = patience
        self._hist: List[tuple] = []

    def update(self, train_loss: float, val_loss: float) -> Optional[Dict[str, float]]:
        self._hist.append((train_loss, val_loss))
        if len(self._hist) < self.patience + 1:
            return None
        recent = self._hist[-(self.patience + 1):]
        if all(b[1] > a[1] and b[0] < a[0] for a, b in zip(recent, recent[1:])):
            return {"train_loss": train_loss, "val_loss": val_loss, "val_rise": val_loss - recent[0][1],
                    "gap": val_loss - train_loss, "evals": self.patience}
        return None
