"""
aarav-lm · Instruction-following evaluation

Each prompt carries rule-based checks (see data/instruct_tasks.py). A reply passes a prompt
only if it satisfies every check. Reported with Wilson 95% intervals, per task kind and per
check kind, for greedy and for sampled decoding. Prompts with no checks (format-only) are
excluded from the rates.

Three additions make the numbers interpretable (docs/CORRECTNESS.md, entries 3 and 5):
  * `chance` / `lift`: the pass rate a story that ignores the request would get (measured on real
    held-out stories, see scripts/make_eval_controls.py), and the model's rate minus that.
  * `n_distinct` / `ci95_clustered`: kinds with no story-dependent slot (short, long, dialogue) repeat a
    few prompt strings many times, so the effective sample size is the number of DISTINCT prompts.
  * `per_prompt`: every prompt's outcome, so two checkpoints can be compared on identical prompts
    (evaluation/paired.py).

Chat-tuned checkpoints are prompted through the chat template; base checkpoints get the request
as plain text to continue, so a base model can be scored on the same prompts as a reference.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

from aarav_lm.data.instruct_tasks import Check, run_checks, word_count
from aarav_lm.evaluation.stats import cluster_rate_ci, rate_with_ci
from aarav_lm.inference.chat_format import Message
from aarav_lm.inference.engine import GenerationConfig, InferenceEngine
from aarav_lm.inference.prompting import build_prompt, resolve_mode

MODES = {
    "greedy": dict(temperature=0.0, top_k=None, top_p=None),
    "sampled": dict(temperature=0.7, top_k=50, top_p=0.95),
}


def _reply(engine: InferenceEngine, prompt: str, cfg: GenerationConfig, chat: bool):
    if chat:
        p = build_prompt(engine, [Message("user", prompt)], cfg.max_new_tokens, "chat")
        return engine.generate(p.ids, replace(cfg, stop_ids=p.stop_ids))
    return engine.generate(prompt, cfg)


def load_chance(path: str | Path) -> Dict[str, float]:
    """{kind: chance-level pass rate} from the file scripts/make_eval_controls.py writes next to an eval set."""
    return {k: float(v) for k, v in json.loads(Path(path).read_text())["by_kind"].items()}


def _with_context(base: dict, clusters: Dict[str, List[int]], chance: Optional[float]) -> dict:
    """Add the effective sample size, the clustered interval and (if known) chance level and lift."""
    cl = list(clusters.values())
    base["n_distinct"] = len(cl)
    base["ci95_clustered"] = cluster_rate_ci([(c[0], c[1]) for c in cl])
    if chance is not None and base["rate"] is not None:
        base["chance"], base["lift"] = round(chance, 4), round(base["rate"] - chance, 4)
    return base


def run_instruction_eval(
    engine: InferenceEngine, prompts: List[dict], max_new_tokens: int = 400, n_examples: int = 6,
    decoding_overrides: Dict[str, dict] | None = None, chance: Optional[Dict[str, float]] = None,
) -> Dict:
    chat = resolve_mode(engine, "auto") == "chat"
    out: Dict[str, Dict] = {}
    for mode, kw in MODES.items():
        kw = {**kw, **((decoding_overrides or {}).get(mode, {}))}
        scored, by_kind, by_check, unseen, seen = 0, {}, {}, [0, 0], [0, 0]
        n_pass, lengths, natural, examples = 0, [], 0, []
        per_prompt: List[dict] = []
        clusters: Dict[str, Dict[str, List[int]]] = defaultdict(dict)        # kind -> prompt text -> [passes, draws]
        greedy_cache: Dict[str, object] = {}                                  # greedy is deterministic: identical prompt, identical reply
        for i, pr in enumerate(prompts):
            cfg = GenerationConfig(max_new_tokens=max_new_tokens, seed=2000 + i, **kw)
            if kw.get("temperature") == 0.0:
                res = greedy_cache.get(pr["prompt"]) or greedy_cache.setdefault(pr["prompt"], _reply(engine, pr["prompt"], cfg, chat))
            else:
                res = _reply(engine, pr["prompt"], cfg, chat)
            lengths.append(word_count(res.text))
            natural += res.finish_reason != "length"
            checks = [Check.from_dict(c) for c in pr.get("checks", [])]
            if not checks:
                continue
            results = run_checks(checks, res.text)
            ok = all(results.values())
            scored += 1
            n_pass += ok
            k = by_kind.setdefault(pr["kind"], [0, 0]); k[0] += ok; k[1] += 1
            cell = clusters[pr["kind"]].setdefault(pr["prompt"], [0, 0]); cell[0] += ok; cell[1] += 1
            per_prompt.append({"id": pr.get("id", str(i)), "kind": pr["kind"], "prompt": pr["prompt"], "passed": bool(ok),
                               "checks": {ck: bool(v) for ck, v in results.items()}, "words": word_count(res.text)})
            for ck, passed in results.items():
                c = by_check.setdefault(ck, [0, 0]); c[0] += passed; c[1] += 1
            if pr["kind"] in ("name", "name_topic", "name_emotion"):
                bucket = unseen if pr.get("unseen_name") else seen
                bucket[0] += ok; bucket[1] += 1
            if len(examples) < n_examples:
                examples.append({"prompt": pr["prompt"], "kind": pr["kind"], "passed": ok, "reply": res.text.strip()[:700]})
        every = {(kind, text): c for kind, texts in clusters.items() for text, c in texts.items()}
        overall_chance = None
        if chance and scored:
            overall_chance = sum(chance.get(k, 0.0) * v[1] for k, v in by_kind.items()) / scored
        out[mode] = {
            "pass_rate": _with_context(rate_with_ci(n_pass, scored), {f"{k}|{t}": c for (k, t), c in every.items()}, overall_chance),
            "by_task": {k: _with_context(rate_with_ci(*v), clusters[k], (chance or {}).get(k) if chance else None)
                        for k, v in sorted(by_kind.items())},
            "by_check": {k: rate_with_ci(*v) for k, v in sorted(by_check.items())},
            "name_seen": rate_with_ci(*seen) if seen[1] else None,
            "name_unseen": rate_with_ci(*unseen) if unseen[1] else None,
            "natural_end_rate": natural / len(prompts), "mean_words": sum(lengths) / len(lengths),
            "prompts": len(prompts), "scored": scored, "examples": examples, "per_prompt": per_prompt,
        }
    return out
