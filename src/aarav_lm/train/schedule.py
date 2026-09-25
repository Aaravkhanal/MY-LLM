"""aarav-lm · Learning-rate schedule: linear warmup, then cosine decay to a floor."""

import math


def lr_at(step: int, lr: float, min_lr_ratio: float, warmup_steps: int, max_steps: int) -> float:
    """LR for the optimizer step with 0-based index `step`.

    Warmup reaches the full LR exactly at step == warmup_steps; the cosine then
    decays to lr * min_lr_ratio at step == max_steps and stays there.
    """
    min_lr = lr * min_lr_ratio
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (lr - min_lr)
