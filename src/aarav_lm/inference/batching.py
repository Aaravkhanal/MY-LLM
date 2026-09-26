"""
aarav-lm · Batched generation: several users' requests share every decode step

One model forward over a batch costs little more than a forward over one sequence when the model is small (each step is dominated
by fixed per-call overhead, not arithmetic), so decoding B requests together serves B users for roughly the price of one.

How it stays exact. The requests' prompts have different lengths, so they are LEFT-padded to one common length and every row shares
the cache position. The padding is hidden with a key mask (model/layers.py). Because the model uses RoPE, attention scores depend
only on the RELATIVE distance between two tokens, so shifting a sequence right by its padding changes nothing for its real tokens:
each row gets the same logits it would get alone (tests/test_batched_model.py). Everything after the logits is per row and
identical to `InferenceEngine.stream`: the same sampler, the same seeded random generator, the same stop and text handling
(`TextEmitter`), so a request's output does not depend on who else is in the batch (tests/test_batching.py).

Limits, on purpose: RoPE models only (a learned-position model is not shift-invariant); a wave is static (rows that finish early
are masked out rather than removed, and new requests wait for the next wave); and every row must fit in the cache
(prompt + max_new_tokens <= context), because the sliding-window rebuild of the single-request path is not batched. Requests that
do not fit are run one at a time by the scheduler instead.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch

from aarav_lm.inference.engine import GenerationConfig, GenerationResult, InferenceEngine, TextEmitter, sample_next
from aarav_lm.model.layers import KVCache

PAD_ID = 0                                     # value is irrelevant: padded positions are masked out


class BatchNotPossible(ValueError):
    """The requests cannot share a batch (learned positions, or a row would overflow the cache)."""


@dataclass
class Job:
    prompt_ids: Sequence[int]
    cfg: GenerationConfig
    on_delta: Optional[Callable[[str], None]] = None        # called with each safe text delta, in order
    cancelled: Callable[[], bool] = lambda: False           # polled once per step; True stops the row (finish "cancelled")
    result: Optional[GenerationResult] = field(default=None, init=False)


def can_batch(engine: InferenceEngine, prompt_len: int, cfg: GenerationConfig) -> bool:
    """Whether a request may join a batch: RoPE model, and prompt + output fit in the cache without sliding."""
    return engine.model.cfg.arch != "gpt2" and 0 < prompt_len and prompt_len + cfg.max_new_tokens <= engine.block_size


@torch.no_grad()
def generate_batch(engine: InferenceEngine, jobs: List[Job]) -> List[GenerationResult]:
    """Run `jobs` together to completion. Each job's `result` is set; deltas go to its `on_delta`. Returns the results in order."""
    if not jobs:
        return []
    cfgs = [j.cfg.validate() for j in jobs]
    for j, c in zip(jobs, cfgs):
        if not len(j.prompt_ids):
            raise ValueError("prompt is empty")
        if not can_batch(engine, len(j.prompt_ids), c):
            raise BatchNotPossible("a request does not fit a batch (learned positions, or prompt + max_new_tokens > context)")
    B, tok, mc = len(jobs), engine.tok, engine.model.cfg
    t0 = time.perf_counter()

    prompts = [list(j.prompt_ids) for j in jobs]
    L = max(len(p) for p in prompts)
    x = torch.full((B, L), PAD_ID, dtype=torch.long)
    real = torch.zeros(B, L, dtype=torch.bool)
    for i, p in enumerate(prompts):
        x[i, L - len(p):] = torch.tensor(p)
        real[i, L - len(p):] = True
    cache = KVCache(mc.n_layer, B, engine.block_size, mc.n_kv_head, mc.n_embd // mc.n_head, engine.device,
                    engine.model.token_emb.weight.dtype)
    logits, _ = engine.model(x.to(engine.device), cache=cache, last_only=True, key_mask=real.to(engine.device))
    cache.step(L)
    logits = logits[:, -1]                                                         # (B, V): every row's last position is a real token

    gens = [torch.Generator().manual_seed(c.seed if c.seed is not None else int.from_bytes(os.urandom(8), "little") >> 1)
            for c in cfgs]
    emitters = [TextEmitter(tok, c.stop_sequences) for c in cfgs]
    gen_ids: List[List[int]] = [[] for _ in jobs]
    finish: List[Optional[str]] = [None] * B
    ended_at = [0.0] * B
    mask = real                                                                     # grows by one column per step

    def close(i: int, reason: str) -> None:
        finish[i] = reason
        ended_at[i] = time.perf_counter() - t0

    for step in range(max(c.max_new_tokens for c in cfgs)):
        feed = torch.full((B, 1), PAD_ID, dtype=torch.long)
        new_col = torch.zeros(B, 1, dtype=torch.bool)
        for i, (job, cfg) in enumerate(zip(jobs, cfgs)):
            if finish[i] is not None:
                continue
            if job.cancelled():
                close(i, "cancelled")
                continue
            if cfg.max_seconds is not None and time.perf_counter() - t0 > cfg.max_seconds:
                close(i, "timeout")
                continue
            if step >= cfg.max_new_tokens:
                close(i, "length")
                continue
            nxt = sample_next(logits[i].cpu(), cfg, gens[i], gen_ids[i], exempt_ids=(tok.eot_id, *cfg.stop_ids))
            if cfg.stop_at_eot and nxt == tok.eot_id:
                close(i, "eot")
                continue
            if nxt in cfg.stop_ids:
                close(i, "stop")
                continue
            gen_ids[i].append(nxt)
            delta = emitters[i].push(nxt)
            if delta and job.on_delta:
                job.on_delta(delta)
            if emitters[i].stopped:
                close(i, "stop")
                continue
            feed[i, 0], new_col[i, 0] = nxt, True
        active = [i for i in range(B) if finish[i] is None]
        if not active:
            break
        if step + 1 >= max(cfgs[i].max_new_tokens for i in active):                 # the last sampled tokens need no further forward
            for i in active:
                close(i, "length")
            break
        mask = torch.cat([mask, new_col], dim=1)
        logits, _ = engine.model(feed.to(engine.device), cache=cache, last_only=True, key_mask=mask.to(engine.device))
        cache.step(1)
        logits = logits[:, -1]
    for i in range(B):
        if finish[i] is None:
            close(i, "length")

    for i, job in enumerate(jobs):
        final, tail = emitters[i].finish(finish[i])
        if tail and job.on_delta:
            job.on_delta(tail)
        job.result = GenerationResult(text=final, token_ids=gen_ids[i], finish_reason=finish[i], prompt_tokens=len(job.prompt_ids),
                                      generated_tokens=len(gen_ids[i]), seconds=ended_at[i])
    return [j.result for j in jobs]
