"""
aarav-lm · Instruction-following evaluation

Each prompt carries rule-based checks (see data/instruct_tasks.py). A reply passes a prompt
only if it satisfies every check. Reported with Wilson 95% intervals, per task kind and per
check kind, for greedy and for sampled decoding. Prompts with no checks (format-only) are
excluded from the rates.

Chat-tuned checkpoints are prompted through the chat template; base checkpoints get the request
as plain text to continue, so a base model can be scored on the same prompts as a reference.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from aarav_lm.data.instruct_tasks import Check, run_checks, word_count
from aarav_lm.evaluation.stats import rate_with_ci
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


def run_instruction_eval(
    engine: InferenceEngine, prompts: List[dict], max_new_tokens: int = 400, n_examples: int = 6,
    decoding_overrides: Dict[str, dict] | None = None,
) -> Dict:
    chat = resolve_mode(engine, "auto") == "chat"
    out: Dict[str, Dict] = {}
    for mode, kw in MODES.items():
        kw = {**kw, **((decoding_overrides or {}).get(mode, {}))}
        scored, by_kind, by_check, unseen, seen = 0, {}, {}, [0, 0], [0, 0]
        n_pass, lengths, natural, examples = 0, [], 0, []
        for i, pr in enumerate(prompts):
            res = _reply(engine, pr["prompt"], GenerationConfig(max_new_tokens=max_new_tokens, seed=2000 + i, **kw), chat)
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
            for ck, passed in results.items():
                c = by_check.setdefault(ck, [0, 0]); c[0] += passed; c[1] += 1
            if pr["kind"] in ("name", "name_topic", "name_emotion"):
                bucket = unseen if pr.get("unseen_name") else seen
                bucket[0] += ok; bucket[1] += 1
            if len(examples) < n_examples:
                examples.append({"prompt": pr["prompt"], "kind": pr["kind"], "passed": ok, "reply": res.text.strip()[:700]})
        out[mode] = {
            "pass_rate": rate_with_ci(n_pass, scored),
            "by_task": {k: rate_with_ci(*v) for k, v in sorted(by_kind.items())},
            "by_check": {k: rate_with_ci(*v) for k, v in sorted(by_check.items())},
            "name_seen": rate_with_ci(*seen) if seen[1] else None,
            "name_unseen": rate_with_ci(*unseen) if unseen[1] else None,
            "natural_end_rate": natural / len(prompts), "mean_words": sum(lengths) / len(lengths),
            "prompts": len(prompts), "scored": scored, "examples": examples,
        }
    return out
