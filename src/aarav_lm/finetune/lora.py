"""
aarav-lm · LoRA (low-rank adaptation)

Freezes the base weights and learns a rank-r update for chosen Linear layers:

    y = W x + (alpha / r) * B (A x)          A: r x in (random), B: out x r (zeros)

B starts at zero, so a freshly adapted model is *exactly* the base model. After
training, merge_lora() folds B·A into W and returns a plain model with no adapter
overhead, which is what checkpoints for inference contain.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aarav_lm.model.core import AaravLM

DEFAULT_TARGETS = ("attn.q", "attn.v", "attn.k", "attn.out", "mlp.gate", "mlp.up", "mlp.down")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1")
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B)
        return self.base(x) + self.scale * update

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scale * (self.lora_B @ self.lora_A)


def _matches(name: str, targets: Iterable[str]) -> bool:
    return any(name.endswith(t) for t in targets)


def apply_lora(
    model: AaravLM, rank: int, alpha: float = 16.0, dropout: float = 0.0,
    targets: Tuple[str, ...] = DEFAULT_TARGETS,
) -> List[str]:
    """Freeze the model and wrap the target Linear layers. Returns the wrapped module names."""
    for p in model.parameters():
        p.requires_grad = False
    wrapped: List[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and _matches(full, targets) and "lm_head" not in full:
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                wrapped.append(full)
    if not wrapped:
        raise ValueError(f"no Linear layers matched targets {targets}")
    return wrapped


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": train}


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())


@torch.no_grad()
def merge_lora(model: AaravLM) -> AaravLM:
    """Fold adapters into the base weights in place and return the (now plain) model."""
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.base.weight.copy_(child.merged_weight())
                child.base.weight.requires_grad = True
                setattr(module, child_name, child.base)
    for p in model.parameters():
        p.requires_grad = True
    return model
