"""
aarav-lm · Forced-choice likelihood probes

Each probe is a minimal pair: a prefix with one correct and one incorrect
continuation ("The dogs" -> " are" vs " is"). The model "answers" by giving the
correct continuation the higher per-token log-probability. This is how factual and
grammatical knowledge is tested without any judge model or generation. Chance is 50%.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from aarav_lm.inference.engine import InferenceEngine


def load_jsonl(path: str | Path) -> List[dict]:
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON ({e})") from e
    return rows


@torch.no_grad()
def continuation_logprob(engine: InferenceEngine, prefix: str, continuation: str) -> Dict[str, float]:
    """Total and per-token log-probability of `continuation` given `prefix`."""
    p_ids = engine.tok.encode(prefix)
    c_ids = engine.tok.encode(continuation)
    ids = (p_ids + c_ids)[-engine.block_size:]
    n_cont = min(len(c_ids), len(ids) - 1)
    if n_cont < 1:
        raise ValueError("continuation is empty after encoding")
    x = torch.tensor([ids], dtype=torch.long, device=engine.device)
    logits, _ = engine.model(x)
    logp = F.log_softmax(logits[0].float(), dim=-1)
    targets = x[0, 1:]
    per_token = logp[:-1].gather(1, targets[:, None]).squeeze(1)[-n_cont:]
    return {"sum": float(per_token.sum()), "mean": float(per_token.mean()), "tokens": n_cont}


def run_probes(engine: InferenceEngine, probes: List[dict]) -> Dict:
    if not probes:
        raise ValueError("no probes given")
    items, by_cat = [], {}
    for pr in probes:
        good = continuation_logprob(engine, pr["prefix"], pr["correct"])
        bad = continuation_logprob(engine, pr["prefix"], pr["incorrect"])
        ok = good["mean"] > bad["mean"]                       # ties count as wrong
        items.append({**pr, "correct_logp": good["mean"], "incorrect_logp": bad["mean"], "passed": ok})
        c = by_cat.setdefault(pr["category"], [0, 0])
        c[0] += ok
        c[1] += 1
    return {
        "accuracy": sum(i["passed"] for i in items) / len(items),
        "n": len(items),
        "by_category": {k: {"accuracy": v[0] / v[1], "n": v[1]} for k, v in sorted(by_cat.items())},
        "items": items,
    }
