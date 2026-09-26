"""
aarav-lm · Paired comparison of two checkpoints on the same instruction-following prompts

Needs the `per_prompt` records that `run_instruction_eval` writes. Both results must come from the same
prompt file. The unit of resampling is the distinct prompt string (see `stats.cluster_rate_ci`).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from aarav_lm.evaluation.stats import paired_difference


def _clusters(a: List[dict], b: List[dict]) -> Dict[str, Dict[str, List[float]]]:
    ids_a, ids_b = {r["id"] for r in a}, {r["id"] for r in b}
    if ids_a != ids_b:
        raise ValueError(f"the two results scored different prompts ({len(ids_a ^ ids_b)} ids differ): not comparable")
    b_by_id = {r["id"]: r for r in b}
    out: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for ra in a:
        rb = b_by_id[ra["id"]]
        if ra["prompt"] != rb["prompt"] or ra["kind"] != rb["kind"]:
            raise ValueError(f"prompt {ra['id']} differs between the two results: not comparable")
        cell = out[ra["kind"]].setdefault(ra["prompt"], [0.0, 0.0, 0.0])
        cell[0] += ra["passed"]
        cell[1] += rb["passed"]
        cell[2] += 1
    return out


def paired_compare(a: List[dict], b: List[dict], n_boot: int = 4000) -> Dict[str, dict]:
    """{kind: {...}, "ALL": {...}}: each has n, n_distinct, rate_a, rate_b, diff (a - b), ci95, p,
    and the prompt-level counts only_a / only_b (passed by one model but not the other)."""
    by_kind = _clusters(a, b)
    result: Dict[str, dict] = {}
    everything: List[List[float]] = []
    for kind in sorted(by_kind):
        cl = list(by_kind[kind].values())
        everything += cl
        result[kind] = _summarise(cl, n_boot)
    result["ALL"] = _summarise(everything, n_boot)
    return result


def _summarise(clusters: List[List[float]], n_boot: int) -> dict:
    draws = sum(c[2] for c in clusters)
    a_pass, b_pass = sum(c[0] for c in clusters), sum(c[1] for c in clusters)
    d = paired_difference([tuple(c) for c in clusters], n_boot=n_boot)
    return {
        "n": int(draws), "n_distinct": len(clusters), "rate_a": a_pass / draws, "rate_b": b_pass / draws,
        "diff": d["diff"], "ci95": d["ci95"], "p": d["p"],
        "only_a": sum(1 for c in clusters if c[0] > c[1]), "only_b": sum(1 for c in clusters if c[1] > c[0]),
    }


def format_paired(result: Dict[str, dict], name_a: str, name_b: str) -> str:
    lines = [f"{'kind':14s}{name_a[:12]:>13s}{name_b[:12]:>13s}{'A-B':>8s}   {'95% CI':16s}{'p':>7s}  {'distinct':>8s}"]
    for kind, r in result.items():
        if kind == "ALL":
            lines.append("-" * 82)
        lines.append(f"{kind:14s}{100 * r['rate_a']:12.1f}%{100 * r['rate_b']:12.1f}%{100 * r['diff']:+7.1f}   "
                     f"[{100 * r['ci95'][0]:+6.1f},{100 * r['ci95'][1]:+6.1f}]{r['p']:7.3f}  {r['n_distinct']:>8d}")
    return "\n".join(lines)
