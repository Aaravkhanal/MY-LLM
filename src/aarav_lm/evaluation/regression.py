"""
aarav-lm · Regression checks against a baseline

A baseline is just a saved evaluation-results file from a checkpoint you trust.
A new checkpoint "regresses" if a key metric gets worse by more than its tolerance.
Results are only comparable when both used the same evaluation sets, and that is
checked first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# (dotted metric path, rule, tolerance)
#   rel_increase   fail if new > old * (1 + tol)          lower is better
#   abs_drop       fail if new < old - tol                 higher is better
#   abs_increase   fail if new > old + tol                 lower is better
#   must_be_true   fail unless the new value is True
DEFAULT_RULES: List[Tuple[str, str, float]] = [
    ("loss.val.loss", "rel_increase", 0.02),
    ("loss.test.loss", "rel_increase", 0.02),
    ("loss.val.top1", "abs_drop", 0.01),
    ("probes.accuracy", "abs_drop", 0.05),
    ("generation.greedy.distinct_2", "abs_drop", 0.05),
    ("generation.sampled.distinct_2", "abs_drop", 0.05),
    ("generation.greedy.loop_fraction", "abs_increase", 0.10),
    ("generation.sampled.loop_fraction", "abs_increase", 0.10),
    ("generation.sampled.nonword_rate", "abs_increase", 0.03),
    ("generation.sampled.entity_consistency", "abs_drop", 0.10),
    ("edge_cases.all_passed", "must_be_true", 0.0),
]


# Two results only mean something side by side if these match exactly.
COMPARABILITY_KEYS: List[Tuple[str, str]] = [
    ("eval_sets.prompts.sha256", "evaluation prompt sets"),
    ("eval_sets.probes.sha256", "probe sets"),
    ("eval_sets.dataset_fingerprint", "held-out datasets"),
    ("model.tokenizer_fingerprint", "tokenizers (per-token loss is not comparable across vocabularies)"),
    ("settings.max_new_tokens", "generation lengths"),
    ("settings.max_windows", "numbers of evaluation windows"),
    ("settings.sampled", "sampling settings"),
]


@dataclass
class RuleResult:
    metric: str
    rule: str
    baseline: Any
    new: Any
    passed: bool
    note: str = ""


def get_path(d: Dict, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def _check(rule: str, tol: float, old: Any, new: Any) -> Tuple[bool, str]:
    if rule == "must_be_true":
        return new is True, "" if new is True else "expected True"
    if old is None or new is None:
        return False, "metric missing"
    if rule == "rel_increase":
        limit = old * (1 + tol)
        return new <= limit, f"limit {limit:.4f}"
    if rule == "abs_drop":
        limit = old - tol
        return new >= limit, f"limit {limit:.4f}"
    if rule == "abs_increase":
        limit = old + tol
        return new <= limit, f"limit {limit:.4f}"
    raise ValueError(f"unknown rule {rule!r}")


def compare_to_baseline(
    new: Dict, baseline: Dict, rules: Optional[List[Tuple[str, str, float]]] = None
) -> Tuple[bool, List[RuleResult]]:
    problems: List[RuleResult] = []
    for path, what in COMPARABILITY_KEYS:
        a, b = get_path(baseline, path), get_path(new, path)
        if a != b:
            problems.append(RuleResult(path, "same_setup", a, b, False,
                                       f"{what} differ; results are not comparable"))
    if problems:
        return False, problems
    results = []
    for path, rule, tol in (rules or DEFAULT_RULES):
        old, cur = get_path(baseline, path), get_path(new, path)
        # a metric absent from BOTH (e.g. no entity prompts) is not a regression
        if old is None and cur is None and rule != "must_be_true":
            continue
        ok, note = _check(rule, tol, old, cur)
        results.append(RuleResult(path, rule, old, cur, ok, note))
    return all(r.passed for r in results), results


def format_report(results: List[RuleResult]) -> str:
    lines = []
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)   # noqa: E731
        lines.append(f"  [{mark}] {r.metric:<44} baseline {fmt(r.baseline):>9}  new {fmt(r.new):>9}  {r.note}")
    return "\n".join(lines)
