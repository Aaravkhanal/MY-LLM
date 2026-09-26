"""
aarav-lm · Held-out loss and perplexity

Token-weighted: the mean is over target tokens that are actually scored (targets of
-1 are ignored), so a batch with many masked positions doesn't get equal weight to a
full one. That matters for supervised fine-tuning, where prompts are masked.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
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


@torch.no_grad()
def next_token_accuracy(
    model: AaravLM,
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    ks: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """Top-k accuracy of the next-token prediction over scored positions."""
    was_training = model.training
    model.eval()
    hits = {k: 0 for k in ks}
    count = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        mask = y != -1
        top = logits.topk(max(ks), dim=-1).indices
        for k in ks:
            hits[k] += int(((top[..., :k] == y[..., None]).any(-1) & mask).sum())
        count += int(mask.sum())
    model.train(was_training)
    if count == 0:
        raise ValueError("next_token_accuracy received no scored tokens")
    return {f"top{k}": hits[k] / count for k in ks}


@torch.no_grad()
def heldout_rows(
    model: AaravLM,
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    autocast_ctx=None,
) -> Dict[str, np.ndarray]:
    """One forward pass, per-row results: summed loss, number of scored targets, top-1 and top-5 hits.

    A row is a window (pretraining) or a conversation (SFT). Keeping rows separate is what lets
    `summarize_rows` put a standard error on the loss: the unit of sampling is the row, not the token.
    """
    was_training = model.training
    model.eval()
    sums, counts, hit1, hit5 = [], [], [], []
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        if autocast_ctx is not None:
            with autocast_ctx():
                logits, _ = model(x)
        else:
            logits, _ = model(x)
        logits = logits.float()
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction="none").view(y.shape)
        mask = y != -1
        top = logits.topk(5, dim=-1).indices
        sums.append(ce.sum(1).double().cpu())                       # ignored targets contribute exactly 0
        counts.append(mask.sum(1).cpu())
        hit1.append(((top[..., :1] == y[..., None]).any(-1) & mask).sum(1).cpu())
        hit5.append(((top == y[..., None]).any(-1) & mask).sum(1).cpu())
    model.train(was_training)
    if not sums:
        raise ValueError("heldout_rows received no batches")
    out = {"sum": torch.cat(sums).numpy(), "count": torch.cat(counts).numpy().astype(np.int64),
           "hit1": torch.cat(hit1).numpy().astype(np.int64), "hit5": torch.cat(hit5).numpy().astype(np.int64)}
    if out["count"].sum() == 0:
        raise ValueError("heldout_rows received no scored tokens")
    return out


def summarize_rows(rows: Dict[str, np.ndarray], n_boot: int = 400, seed: int = 0) -> Dict[str, float]:
    """Token-weighted loss over all rows, with a bootstrap standard error (resampling rows).

    stderr is what makes two checkpoints' losses comparable: a difference smaller than about twice
    the standard error is not evidence of anything. The bootstrap RNG is seeded, so results are reproducible.
    """
    total, count = rows["sum"], rows["count"].astype(np.float64)
    n = len(total)
    loss = float(total.sum() / count.sum())
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    boot = total[idx].sum(1) / count[idx].sum(1)
    se = float(boot.std(ddof=1)) if n > 1 else float("nan")
    ppl = math.exp(min(loss, 50.0))
    return {
        "loss": loss, "stderr": se, "perplexity": ppl, "perplexity_stderr": ppl * se,
        "tokens": int(count.sum()), "rows": n,
        "top1": float(rows["hit1"].sum() / count.sum()), "top5": float(rows["hit5"].sum() / count.sum()),
    }
