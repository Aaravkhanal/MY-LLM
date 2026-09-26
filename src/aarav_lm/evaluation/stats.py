"""aarav-lm · Small-sample statistics for evaluation reports."""

from __future__ import annotations

import math
from typing import Tuple


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
