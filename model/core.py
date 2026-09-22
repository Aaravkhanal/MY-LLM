"""
aarav-lm · Core Transformer
Author: Aarav Khanal

AaravLM is a decoder-only transformer that can run in two modes:
  arch='gpt2'   — learned absolute position embeddings, GELU, LayerNorm
  arch='modern' — RoPE, SwiGLU, RMSNorm, optional GQA

Both modes share the same forward / generate API so experiments are
a single flag change, not a code change.

Key methods
-----------
forward(idx, targets)       training / evaluation
generate(idx, max_new_tokens) autoregressive sampling with optional KV-cache
configure_optimizers(...)   AdamW with weight-decay split between 2-D and 1-D params
estimate_mfu(...)           compute approximate Model FLOPs Utilisation per iteration
"""

import inspect
import math
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aarav_lm.config.config import ModelConfig
from aarav_lm.model.layers import (
    Attention,
    Block,
    GELUMLP,
    KVCache,
    LayerNorm,
    RMSNorm,
    SwiGLUMLP,
    apply_rope,
    precompute_rope,
    repeat_kv,
)


class AaravLM(nn.Module):
    """Decoder-only transformer — the heart of the aarav-lm project."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ── Embedding ──────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        if cfg.arch == "gpt2":
            # Learned absolute position embeddings (GPT-2 style)
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.drop    = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
            self.register_buffer("rope_cos", None)
            self.register_buffer("rope_sin", None)
        else:
            # RoPE — precomputed once, shared across all layers
            self.pos_emb = None
            self.drop    = nn.Identity()
            cos, sin = precompute_rope(cfg.n_embd // cfg.n_head, cfg.block_size, theta=cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        # ── Transformer blocks ─────────────────────────────────────────────
        self.blocks    = nn.ModuleList([Block(cfg, layer_idx=i) for i in range(cfg.n_layer)])

        # ── Final normalisation ────────────────────────────────────────────
        if cfg.norm_type == "rmsnorm":
            self.final_norm = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        else:
            self.final_norm = LayerNorm(cfg.n_embd, bias=cfg.bias, eps=cfg.norm_eps)

        # ── Language model head ────────────────────────────────────────────
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight  # weight tying

        # ── Weight initialisation ──────────────────────────────────────────
        self.apply(self._init_weights)
        # Scale down residual projections — GPT-2 paper recommendation
        for name, param in self.named_parameters():
            if name.endswith(("out.weight", "down.weight", "c_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    # ── Init ───────────────────────────────────────────────────────────────
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ── Parameter count ────────────────────────────────────────────────────
    def num_params(self, exclude_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings and self.pos_emb is not None:
            n -= self.pos_emb.weight.numel()
        return n

    # ── Forward ────────────────────────────────────────────────────────────
    def forward(
        self,
        idx:     torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache:   Optional[KVCache]      = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        idx     : (B, T)  token indices
        targets : (B, T)  next-token targets for loss; pass None during generation
        cache   : KVCache instance for incremental decoding; None during training

        Returns
        -------
        logits : (B, T, V)  or  (B, 1, V) during cached generation
        loss   : scalar cross-entropy loss when targets is not None, else None
        """
        device = idx.device
        B, T   = idx.shape
        assert T <= self.cfg.block_size, \
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}"

        x = self.token_emb(idx)   # (B, T, C)

        if self.cfg.arch == "gpt2":
            offset  = cache.seen if cache is not None else 0
            pos     = torch.arange(offset, offset + T, dtype=torch.long, device=device)
            x       = self.drop(x + self.pos_emb(pos))
            cos, sin = None, None
        else:
            x        = self.drop(x)
            offset   = cache.seen if cache is not None else 0
            cos      = self.rope_cos[offset : offset + T].to(device)
            sin      = self.rope_sin[offset : offset + T].to(device)

        for block in self.blocks:
            x = block(x, cos=cos, sin=sin, cache=cache)

        x = self.final_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # During generation only compute logits for the final position
            logits = self.lm_head(x[:, [-1], :]) if (cache is not None and cache.seen > 0) else self.lm_head(x)
            loss   = None

        return logits, loss

    # ── Optimizer ──────────────────────────────────────────────────────────
    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        """
        AdamW with separate weight-decay groups:
          - 2-D weight matrices  → decayed
          - biases, norms, embed → NOT decayed
        """
        params     = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay      = [p for p in params.values() if p.dim() >= 2]
        no_decay   = [p for p in params.values() if p.dim() < 2]
        groups     = [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_ok   = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra      = dict(fused=True) if (fused_ok and device_type == "cuda") else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    # ── MFU ────────────────────────────────────────────────────────────────
    def estimate_mfu(self, iters_per_sec: float) -> float:
        """Rough Model FLOPs Utilisation estimate (FLOP/s achieved)."""
        N = self.num_params()
        L, H, Q, T = (
            self.cfg.n_layer,
            self.cfg.n_head,
            self.cfg.n_embd // self.cfg.n_head,
            self.cfg.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        return flops_per_token * T * iters_per_sec

    # ── Generation ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float          = 1.0,
        top_k:          Optional[int]  = None,
        top_p:          Optional[float] = None,
        use_cache:      bool            = True,
    ) -> torch.Tensor:
        """
        Auto-regressive text generation.

        Parameters
        ----------
        idx            : (1, T_prompt)  starting tokens
        max_new_tokens : number of tokens to generate
        temperature    : 0.0 = greedy, higher = more random
        top_k          : keep only top-k logits before sampling
        top_p          : nucleus sampling threshold
        use_cache      : whether to use pre-allocated KV-cache for speed
        """
        B, prompt_len = idx.shape
        device = idx.device
        dtype  = self.token_emb.weight.dtype

        if use_cache:
            cache = KVCache(
                n_layers    = self.cfg.n_layer,
                batch_size  = B,
                max_seq_len = self.cfg.block_size,
                n_kv_head   = self.cfg.n_kv_head or self.cfg.n_head,
                head_dim    = self.cfg.n_embd // self.cfg.n_head,
                device      = device,
                dtype       = dtype,
            )
            logits, _ = self.forward(idx, cache=cache)
            cache.step(prompt_len)
        else:
            cache  = None
            logits, _ = self.forward(idx)

        for _ in range(max_new_tokens):
            if cache is not None and cache.seen >= self.cfg.block_size:
                break

            next_logits = logits[:, -1, :]   # (B, V)

            if temperature == 0.0:
                idx_next = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float("-inf")

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(next_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

            if cache is not None:
                logits, _ = self.forward(idx_next, cache=cache)
                cache.step(1)
            else:
                cond   = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
                logits, _ = self.forward(cond)

        return idx
