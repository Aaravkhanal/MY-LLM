"""
aarav-lm · Experiment comparison

    Model version -> training configuration -> validation metrics -> evaluation
    metrics -> generation samples

Reads experiments/*/ (config.yaml, summary.json, eval/*_results.json) and renders one
Markdown report so runs can be compared side by side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def collect_experiments(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for d in sorted(p for p in Path(root).glob("exp_*") if p.is_dir()):
        summary = _load(d / "summary.json")
        cfg = yaml.safe_load((d / "config.yaml").read_text()) if (d / "config.yaml").exists() else {}
        evals = sorted((d / "eval").glob("*_results.json")) if (d / "eval").exists() else []
        ev = _load(evals[-1]) if evals else None
        rows.append({"dir": d.name, "summary": summary, "config": cfg or {}, "eval": ev,
                     "eval_file": evals[-1].name if evals else None})
    return rows


def _f(v: Any, spec: str = ".3f", dash: str = "–") -> str:
    return dash if v is None else format(v, spec)


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    out = ["# Experiment comparison", ""]
    if not rows:
        return "\n".join(out + ["_No experiments found._", ""])

    out += ["## Training configuration", "",
            "| experiment | params | precision | seq | eff. batch | lr | steps | tokens | train time | status |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s, c = r["summary"] or {}, r["config"]
        t = c.get("train", {})
        seq = (c.get("model_overrides") or {}).get("block_size", "preset")
        eff = t.get("batch_size", 0) * t.get("grad_accum", 0)
        out.append(
            f"| {r['dir']} | {_f((s.get('params') or 0) / 1e6, '.2f')}M | {t.get('precision', '–')} | {seq} | {eff} "
            f"| {t.get('lr', '–')} | {s.get('steps', '–')} | {_f((s.get('tokens_seen') or 0) / 1e6, '.1f')}M "
            f"| {_f((s.get('train_seconds') or 0) / 60, '.1f')} min | {s.get('status', 'no summary')} |"
        )

    out += ["", "## Validation and evaluation metrics", "",
            "| experiment | checkpoint | val loss | val ppl | test ppl | top-1 | probes | distinct-2 (sampled) | loops | non-words | edge cases |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s, e = r["summary"] or {}, r["eval"]
        if e:
            L, G, P, E = e["loss"], e["generation"]["sampled"], e["probes"], e["edge_cases"]
            out.append(
                f"| {r['dir']} | {r['eval_file'].replace('_results.json', '')} | {_f(L['val']['loss'])} "
                f"| {_f(L['val']['perplexity'], '.2f')} | {_f(L['test']['perplexity'], '.2f')} "
                f"| {_f(L['val']['top1'])} | {_f(P['accuracy'])} | {_f(G['distinct_2'])} "
                f"| {_f(G['loop_fraction'])} | {_f(G['nonword_rate'])} | {E['passed']}/{E['total']} |"
            )
        else:
            bv = s.get("best_val_loss")
            out.append(f"| {r['dir']} | (not evaluated) | {_f(bv)} | {_f(s.get('best_val_perplexity'), '.2f')} "
                       f"| – | – | – | – | – | – | – |")

    out += ["", "## Generation samples (greedy, first prompts)", ""]
    for r in rows:
        if r["eval"]:
            out += [f"### {r['dir']}", ""]
            for smp in r["eval"]["generation"]["greedy"]["samples"][:3]:
                text = smp["text"].strip().replace("\n", " ")
                out += [f"> **{smp['prompt']}** {text[:400]}{'…' if len(text) > 400 else ''}", ""]
    return "\n".join(out) + "\n"
