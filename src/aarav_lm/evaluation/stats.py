"""aarav-lm · Small-sample statistics for evaluation reports."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion. Sound for small n and for rates near 0 or 1
    (unlike the normal approximation), which is exactly the regime of a 30-200 prompt eval set."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError("successes must be within [0, n]")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_with_ci(successes: int, n: int) -> dict:
    lo, hi = wilson_interval(successes, n)
    return {"pass": successes, "n": n, "rate": successes / n if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def intervals_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def cluster_rate_ci(clusters: Sequence[Tuple[int, int]], n_boot: int = 2000, seed: int = 0) -> List[float]:
    """95% bootstrap interval for a pooled pass rate when outcomes come in clusters.

    `clusters` is one (passes, draws) pair per DISTINCT prompt. Twenty-four requests that are really six
    different strings are six samples of "how well does the model follow requests", not twenty-four: the
    duplicates repeat the same test, and under greedy decoding give the identical answer. Resampling
    whole clusters keeps the interval honest about that. With one draw per cluster this is an ordinary
    bootstrap. Seeded, so reproducible.
    """
    if not clusters:
        return [0.0, 1.0]
    c = np.asarray(clusters, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(c), size=(n_boot, len(c)))
    rates = c[idx, 0].sum(1) / c[idx, 1].sum(1)
    return [round(float(np.percentile(rates, 2.5)), 4), round(float(np.percentile(rates, 97.5)), 4)]


def paired_difference(
    clusters: Sequence[Tuple[float, float, float]], n_boot: int = 4000, seed: int = 0,
) -> Dict[str, float]:
    """Difference in pooled pass rate (A minus B) on the SAME prompts, with a cluster-bootstrap interval.

    `clusters` holds one (a_passes, b_passes, draws) triple per distinct prompt. Because both models are
    scored on identical prompts, prompt difficulty cancels, which is far more sensitive than comparing two
    separately computed intervals. `p` is the two-sided bootstrap p-value (with add-one smoothing).
    """
    c = np.asarray(clusters, dtype=np.float64)
    if len(c) == 0:
        raise ValueError("no prompts to compare")
    rate = lambda a: (a[..., 0].sum(-1) - a[..., 1].sum(-1)) / a[..., 2].sum(-1)          # noqa: E731
    point = float(rate(c))
    idx = np.random.default_rng(seed).integers(0, len(c), size=(n_boot, len(c)))
    boot = rate(c[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((np.sum(boot <= 0) + 1) / (n_boot + 1), (np.sum(boot >= 0) + 1) / (n_boot + 1))
    return {"diff": point, "ci95": [round(float(lo), 4), round(float(hi), 4)], "p": float(min(1.0, p))}
