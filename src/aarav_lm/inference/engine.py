"""
aarav-lm · Inference engine

Loads OUR checkpoint + OUR tokenizer (verified to match) and generates text with a
KV cache, token by token. Nothing here talks to any external model or service.

Behaviour worth knowing
  - Streaming: stream() yields text deltas as tokens arrive.
  - Stop sequences are matched on decoded text and never partially emitted: the tail
    of the output is held back until it can no longer become a stop sequence.
  - Byte-level BPE can split one character across tokens; incomplete UTF-8 is held
    back rather than printed as U+FFFD.
  - Context: a prompt longer than the window keeps its most recent tokens; when
    generation fills the window, the cache is rebuilt from the latest half (RoPE is
    relative, so this degrades gracefully instead of crashing).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import torch

from aarav_lm.model.core import AaravLM
from aarav_lm.model.layers import KVCache
from aarav_lm.tokenizer import AaravTokenizer
from aarav_lm.train import checkpoint as ckpt
from aarav_lm.train.device import pick_device

REPETITION_WINDOW = 256


@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8            # 0 = greedy
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    repetition_penalty: float = 1.0     # >1 discourages repeating recent tokens
    stop_sequences: Tuple[str, ...] = ()
    stop_ids: Tuple[int, ...] = ()      # extra token ids that end generation (e.g. <|end|>)
    stop_at_eot: bool = True
    seed: Optional[int] = None

    def validate(self) -> "GenerationConfig":
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if any(not s for s in self.stop_sequences):
            raise ValueError("stop sequences must be non-empty")
        return self


@dataclass
class GenerationResult:
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    finish_reason: str = ""             # "length" | "eot" | "stop"
    prompt_tokens: int = 0
    generated_tokens: int = 0
    seconds: float = 0.0

    @property
    def tokens_per_sec(self) -> float:
        return self.generated_tokens / self.seconds if self.seconds > 0 else 0.0


def sample_next(
    logits: torch.Tensor,
    cfg: GenerationConfig,
    generator: Optional[torch.Generator],
    recent_ids: Sequence[int] = (),
) -> int:
    """Pick the next token id from a (V,) logits vector."""
    logits = logits.float().clone()
    if cfg.repetition_penalty != 1.0 and len(recent_ids):
        idx = torch.tensor(sorted(set(recent_ids[-REPETITION_WINDOW:])), dtype=torch.long)
        picked = logits[idx]
        logits[idx] = torch.where(picked > 0, picked / cfg.repetition_penalty, picked * cfg.repetition_penalty)
    if cfg.temperature == 0.0:
        return int(torch.argmax(logits))
    logits = logits / cfg.temperature
    if cfg.top_k is not None and cfg.top_k < logits.numel():
        kth = torch.topk(logits, cfg.top_k).values[-1]
        logits[logits < kth] = float("-inf")
    if cfg.top_p is not None and cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = (cum - probs) >= cfg.top_p        # keep the token that crosses the threshold
        logits[sorted_idx[remove]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


class InferenceEngine:
    def __init__(self, model: AaravLM, tokenizer: AaravTokenizer, device: torch.device, meta: Optional[dict] = None):
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.device = device
        self.meta = meta or {}
        self.block_size = model.cfg.block_size
        self.lock = threading.Lock()           # one generation at a time (shared by the API)
        self.last_result: Optional[GenerationResult] = None

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, tokenizer_dir: str | Path, device: str = "auto"
    ) -> "InferenceEngine":
        tok = AaravTokenizer.load(tokenizer_dir)
        dev = pick_device(device)
        model, raw = ckpt.load_model(Path(checkpoint), dev, expected_tokenizer_fingerprint=tok.fingerprint)
        meta = {**raw.get("meta", {}), "checkpoint": str(checkpoint), "tokenizer_fingerprint": tok.fingerprint}
        return cls(model, tok, dev, meta)

    # ── info ───────────────────────────────────────────────────────────────
    def info(self) -> dict:
        return {
            "params": self.model.num_params(), "context_length": self.block_size,
            "vocab_size": self.tok.vocab_size, "device": str(self.device),
            "tokenizer_fingerprint": self.tok.fingerprint, **self.meta,
        }

    # ── core ───────────────────────────────────────────────────────────────
    def _prefill(self, ids: List[int], cache: KVCache) -> torch.Tensor:
        cache.reset()
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, _ = self.model(x, cache=cache, last_only=True)
        cache.step(len(ids))
        return logits[0, -1]

    @torch.no_grad()
    def stream(self, prompt_ids: Sequence[int], cfg: Optional[GenerationConfig] = None) -> Iterator[str]:
        """Yield decoded text deltas. The final GenerationResult is left in self.last_result."""
        cfg = (cfg or GenerationConfig()).validate()
        if not len(prompt_ids):
            raise ValueError("prompt is empty")
        mc = self.model.cfg
        t0 = time.perf_counter()

        window = list(prompt_ids)[-(self.block_size - 1):]
        ids = list(window)                                            # tokens currently in context
        cache = KVCache(mc.n_layer, 1, self.block_size, mc.n_kv_head, mc.n_embd // mc.n_head,
                        self.device, self.model.token_emb.weight.dtype)
        gen = torch.Generator().manual_seed(cfg.seed if cfg.seed is not None else int.from_bytes(os.urandom(8), "little") >> 1)

        result = GenerationResult(prompt_tokens=len(prompt_ids))
        gen_ids: List[int] = []
        emitted = 0                                                    # chars of decoded text already yielded
        hold = max((len(s) for s in cfg.stop_sequences), default=1) - 1
        logits = self._prefill(ids, cache)
        finish = "length"

        for _ in range(cfg.max_new_tokens):
            nxt = sample_next(logits.cpu(), cfg, gen, ids)
            if cfg.stop_at_eot and nxt == self.tok.eot_id:
                finish = "eot"
                break
            if nxt in cfg.stop_ids:
                finish = "stop"
                break
            gen_ids.append(nxt)
            ids.append(nxt)

            text = self.tok.decode(gen_ids)
            if cfg.stop_sequences:
                cut = min((i for i in (text.find(s, max(0, emitted - hold)) for s in cfg.stop_sequences) if i != -1),
                          default=-1)
                if cut != -1:
                    if cut > emitted:
                        yield text[emitted:cut]
                    emitted, finish = cut, "stop"
                    break
            safe = len(text) - hold
            if text.endswith("�"):
                safe = min(safe, len(text) - 1)                        # incomplete UTF-8 tail
            if safe > emitted:
                yield text[emitted:safe]
                emitted = safe

            if cache.seen + 1 > self.block_size:                       # window full: rebuild from the latest half
                ids = ids[-(self.block_size // 2):]
                logits = self._prefill(ids, cache)
            else:
                x = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                out, _ = self.model(x, cache=cache, last_only=True)
                cache.step(1)
                logits = out[0, -1]

        final_text = self.tok.decode(gen_ids)
        if finish == "stop" and cfg.stop_sequences:
            cut = min((i for i in (final_text.find(s) for s in cfg.stop_sequences) if i != -1), default=-1)
            if cut != -1:
                final_text = final_text[:cut]
        if len(final_text) > emitted:
            yield final_text[emitted:]
        result.text = final_text
        result.token_ids = gen_ids
        result.finish_reason = finish
        result.generated_tokens = len(gen_ids)
        result.seconds = time.perf_counter() - t0
        self.last_result = result

    def generate(self, prompt: str | Sequence[int], cfg: Optional[GenerationConfig] = None) -> GenerationResult:
        ids = self.tok.encode(prompt) if isinstance(prompt, str) else list(prompt)
        for _ in self.stream(ids, cfg):
            pass
        assert self.last_result is not None
        return self.last_result
