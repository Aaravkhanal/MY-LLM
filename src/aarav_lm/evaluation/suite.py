"""
aarav-lm · Evaluation suite

Every checkpoint goes through the same fixed steps, so any two checkpoints are
directly comparable:

  1. held-out loss / perplexity / next-token accuracy   (val and test splits)
  2. generation on a fixed prompt set, greedy and sampled (fixed seeds)
       -> repetition, form, coherence proxies, length (see metrics.py)
  3. forced-choice probes                                (grammar, common sense, story logic)
  4. edge cases                                          (inference-path robustness)

The prompt/probe files are versioned in evals/ and their SHA-256 is stored with the
results, so a result can always be traced to the exact evaluation set that produced it.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import torch

from aarav_lm.data.dataset import TokenDataset
from aarav_lm.data.sft import SFTDataset
from aarav_lm.data.shards import Shard
from aarav_lm.evaluation.edge_cases import run_edge_cases
from aarav_lm.evaluation.loss import evaluate_loss, next_token_accuracy
from aarav_lm.evaluation.metrics import build_vocabulary, summarize_generations
from aarav_lm.evaluation.probes import load_jsonl, run_probes
from aarav_lm.inference.chat_format import Message
from aarav_lm.inference.engine import GenerationConfig, InferenceEngine
from aarav_lm.inference.prompting import build_prompt, resolve_mode

RESULTS_VERSION = 1
SAMPLED = dict(temperature=0.8, top_k=50, top_p=0.95)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_fingerprint(data_dir: str | Path) -> Optional[str]:
    """Hash of what the dataset *is* (sources, config, tokenizer, per-split counts) — not of the
    manifest file, which also holds timings and would change on every rebuild."""
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    keep = {k: m.get(k) for k in ("kind", "tokenizer", "sources_sha256", "data_config", "splits", "counters", "max_len")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def is_sft_dir(data_dir: Path) -> bool:
    return (data_dir / "val.npz").exists()


def eval_batches(data_dir: Path, split: str, engine: InferenceEngine, max_windows: int, batch_size: int = 16):
    """Fixed held-out batches for either a pretraining shard dir or an SFT dir."""
    if is_sft_dir(data_dir):
        ds = SFTDataset.load(data_dir / f"{split}.npz", engine.tok.pad_id)
        n_items = len(ds)
    else:
        ds = TokenDataset(Shard.open(data_dir / f"{split}.bin"), engine.block_size)
        n_items = ds.n_windows
    n_batches = -(-min(max_windows, n_items) // batch_size)
    return ds.fixed_eval_batches(batch_size, n_batches)


def load_wordlist(data_dir: str | Path, engine: InferenceEngine, max_tokens: int = 3_000_000) -> Set[str]:
    """Words seen (>=3 times) in the start of the training split; cached next to the shards."""
    data_dir = Path(data_dir)
    cache = data_dir / "wordlist.json"
    key = {"tokenizer_fingerprint": engine.tok.fingerprint, "dataset_fingerprint": dataset_fingerprint(data_dir)}
    if cache.exists():
        d = json.loads(cache.read_text())
        if all(d.get(k) == v for k, v in key.items()):        # rebuilt data or a new tokenizer invalidates the cache
            return set(d["words"])
    if is_sft_dir(data_dir):
        with np.load(data_dir / "train.npz", allow_pickle=False) as z:
            toks = z["ids"][:max_tokens]
    else:
        shard = Shard.open(data_dir / "train.bin")
        toks = shard.tokens[: min(max_tokens, shard.n_tokens)]
    texts = [engine.tok.decode(toks[i : i + 4000].tolist()) for i in range(0, len(toks), 4000)]
    vocab = build_vocabulary(texts)
    cache.write_text(json.dumps({**key, "words": sorted(vocab)}))
    return vocab


def loss_section(engine: InferenceEngine, data_dir: Path, max_windows: int, batch_size: int = 16) -> Dict:
    """Held-out metrics per split. bits_per_byte = loss / ln 2 / (bytes per token): unlike
    perplexity it does not depend on the tokenizer, so it stays comparable across vocabularies."""
    out = {}
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for split in ("val", "test"):
        batches = eval_batches(data_dir, split, engine, max_windows, batch_size)
        out[split] = {
            **evaluate_loss(engine.model, batches, engine.device),
            **next_token_accuracy(engine.model, batches, engine.device),
            "windows": sum(x.shape[0] for x, _ in batches),          # windows (pretrain) or conversations (SFT)
        }
        bpt = (manifest.get("splits", {}).get(split) or {}).get("bytes_per_token")
        out[split]["bits_per_byte"] = out[split]["loss"] / math.log(2) / bpt if bpt else None
    return out


def generation_section(
    engine: InferenceEngine, prompts: List[dict], vocabulary: Set[str], max_new_tokens: int,
    n_samples_in_report: int = 12,
) -> Dict:
    """Instruction-tuned checkpoints are prompted through the chat template (and stop at <|end|>);
    base checkpoints continue the raw prompt text."""
    out: Dict[str, Dict] = {}
    chat = resolve_mode(engine, "auto") == "chat"
    for name, gen_kw in (("greedy", dict(temperature=0.0, top_k=None, top_p=None)), ("sampled", SAMPLED)):
        samples = []
        for i, pr in enumerate(prompts):
            cfg = GenerationConfig(max_new_tokens=max_new_tokens, seed=1000 + i, **gen_kw)
            if chat:
                prompt = build_prompt(engine, [Message("user", pr["prompt"])], max_new_tokens, "chat")
                cfg = replace(cfg, stop_ids=prompt.stop_ids)
                res = engine.generate(prompt.ids, cfg)
            else:
                res = engine.generate(pr["prompt"], cfg)
            samples.append({
                "id": pr["id"], "prompt": pr["prompt"], "entity": pr.get("entity"),
                "text": res.text, "tokens": res.token_ids, "finish_reason": res.finish_reason,
                "tokens_per_sec": res.tokens_per_sec,
            })
        metrics = summarize_generations(samples, vocabulary)
        metrics["mean_tokens_per_sec"] = sum(s["tokens_per_sec"] for s in samples) / len(samples)
        out[name] = {
            **metrics,
            "samples": [{k: s[k] for k in ("id", "prompt", "text", "finish_reason")} for s in samples[:n_samples_in_report]],
        }
    return out


def run_suite(
    engine: InferenceEngine,
    data_dir: str | Path,
    prompts_path: str | Path,
    probes_path: str | Path,
    max_new_tokens: int = 200,
    max_windows: int = 1000,
    log: Callable[[str], None] = print,
) -> Dict:
    data_dir = Path(data_dir)
    t0 = time.time()
    prompts, probes = load_jsonl(prompts_path), load_jsonl(probes_path)
    vocabulary = load_wordlist(data_dir, engine)

    log("[eval] held-out loss / perplexity / next-token accuracy")
    loss = loss_section(engine, data_dir, max_windows)
    log(f"[eval]   val loss {loss['val']['loss']:.4f} ppl {loss['val']['perplexity']:.2f}  "
        f"test loss {loss['test']['loss']:.4f}  top1 {loss['val']['top1']:.3f}")
    log(f"[eval] generation on {len(prompts)} fixed prompts (greedy + sampled)")
    generation = generation_section(engine, prompts, vocabulary, max_new_tokens)
    log("[eval] probes")
    probe_res = run_probes(engine, probes)
    log(f"[eval]   probe accuracy {probe_res['accuracy']:.3f} over {probe_res['n']} items")
    log("[eval] edge cases")
    edge = run_edge_cases(engine)
    log(f"[eval]   {edge['passed']}/{edge['total']} edge cases passed")

    manifest_path = data_dir / "manifest.json"
    return {
        "version": RESULTS_VERSION,
        "model": engine.info(),
        "loss": loss,
        "generation": generation,
        "probes": {k: v for k, v in probe_res.items() if k != "items"} | {"items": probe_res["items"]},
        "edge_cases": edge,
        "eval_sets": {
            "prompts": {"path": str(prompts_path), "sha256": file_sha256(prompts_path), "n": len(prompts)},
            "probes": {"path": str(probes_path), "sha256": file_sha256(probes_path), "n": len(probes)},
            "dataset_fingerprint": dataset_fingerprint(data_dir),
        },
        "settings": {"max_new_tokens": max_new_tokens, "max_windows": max_windows, "sampled": SAMPLED},
        "torch": torch.__version__,
        "eval_seconds": round(time.time() - t0, 1),
    }


def save_results(results: Dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
