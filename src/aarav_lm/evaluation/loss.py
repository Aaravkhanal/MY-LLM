"""
aarav-lm · Held-out loss and perplexity

Token-weighted: the mean is over target tokens that are actually scored (targets of
-1 are ignored), so a batch with many masked positions doesn't get equal weight to a
full one. That matters for supervised fine-tuning, where prompts are masked.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from aarav_lm.model.core import AaravLM


@torch.no_grad()
def evaluate_loss(
    model: AaravLM,
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    autocast_ctx=None,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        if autocast_ctx is not None:
            with autocast_ctx():
                logits, _ = model(x)
        else:
            logits, _ = model(x)
        loss_sum = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction="sum"
        )
        total += float(loss_sum)
        count += int((y != -1).sum())
    model.train(was_training)
    if count == 0:
        raise ValueError("evaluate_loss received no scored tokens")
    mean = total / count
    return {"loss": mean, "perplexity": math.exp(min(mean, 50.0)), "tokens": count}
