"""
aarav-lm · Neural Network Layers
Author: Aarav Khanal

Contains every reusable building block used by the transformer:

  LayerNorm       — standard layer normalisation with optional bias
  RMSNorm         — root-mean-square normalisation (lighter, no mean centering)
  precompute_rope — build the RoPE cosine/sine tables once, reuse every forward pass
  apply_rope      — rotate query / key tensors in-place
  GELUMLP         — classic two-linear + GELU feed-forward block
  SwiGLUMLP       — gated three-linear SwiGLU feed-forward block
  KVCache         — pre-allocated key-value cache for fast autoregressive decoding
  repeat_kv       — expand KV heads to match query heads (GQA helper)
  Attention       — causal multi-head / grouped-query attention with RoPE + KV-cache support
  Block           — pre-norm transformer block (Attention → MLP, both with residual)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aarav_lm.config.config import ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """LayerNorm with optional bias.  PyTorch's built-in forces bias=True."""

    def __init__(self, dim: int, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """Root Mean Square Normalisation — no mean shift, just scale by RMS."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Position Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def precompute_rope(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine RoPE frequency tables for all positions."""
    freqs     = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    positions = torch.arange(max_seq_len)
    freqs     = torch.outer(positions, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate tensor x (B, H, T, D) by RoPE frequencies cos/sin (T, D//2)."""
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(1)   # (1, 1, T, D//2)
    sin = sin[:T].unsqueeze(0).unsqueeze(1)

    # Treat pairs of features as complex numbers and rotate
    x_r, x_i = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    out = torch.stack([x_r * cos - x_i * sin, x_r * sin + x_i * cos], dim=-1)
    return out.flatten(-2).type_as(x)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Networks
# ─────────────────────────────────────────────────────────────────────────────

class GELUMLP(nn.Module):
    """Standard GPT-2-style feed-forward with GELU activation."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hid = 4 * cfg.n_embd
        self.fc   = nn.Linear(cfg.n_embd, hid,      bias=cfg.bias)
        self.proj = nn.Linear(hid,        cfg.n_embd, bias=cfg.bias)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(self.act(self.fc(x))))


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward: output = down( silu(gate(x)) * up(x) )."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # 2/3 × 4d, rounded up to the nearest multiple of ffn_multiple_of
        hid = int(2 * (4 * cfg.n_embd) / 3)
        if cfg.ffn_dim_multiplier is not None:
            hid = int(cfg.ffn_dim_multiplier * hid)
        m   = cfg.ffn_multiple_of
        hid = m * ((hid + m - 1) // m)

        self.gate = nn.Linear(cfg.n_embd, hid,       bias=cfg.bias)
        self.up   = nn.Linear(cfg.n_embd, hid,       bias=cfg.bias)
        self.down = nn.Linear(hid,        cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ─────────────────────────────────────────────────────────────────────────────
# KV Cache
# ─────────────────────────────────────────────────────────────────────────────

class KVCache:
    """
    Pre-allocated key-value cache for single-device autoregressive decoding.

    Allocates one large contiguous buffer at construction time.  Each
    generation step writes into a slice of that buffer — no allocation
    occurs at generation time, which matters a lot on unified memory.
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        max_seq_len: int,
        n_kv_head: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        shape = (n_layers, batch_size, n_kv_head, max_seq_len, head_dim)
        self.k          = torch.zeros(shape, device=device, dtype=dtype)
        self.v          = torch.zeros(shape, device=device, dtype=dtype)
        self.seen       = 0           # tokens already in the cache

    def update(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T   = k_new.shape[2]
        end = self.seen + T
        self.k[layer, :, :, self.seen:end] = k_new
        self.v[layer, :, :, self.seen:end] = v_new
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def step(self, n: int = 1):
        self.seen += n

    def reset(self):
        self.seen = 0
        self.k.zero_()
        self.v.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Tile KV heads along the head dimension to match query head count."""
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


class Attention(nn.Module):
    """Causal self-attention supporting MHA, GQA, RoPE, and KV-cache."""

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg       = cfg
        self.layer_idx = layer_idx
        self.n_head    = cfg.n_head
        self.n_kv      = cfg.n_kv_head or cfg.n_head
        self.n_rep     = self.n_head // self.n_kv
        self.head_dim  = cfg.n_embd // cfg.n_head

        self.q    = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=cfg.bias)
        self.k    = nn.Linear(cfg.n_embd, self.n_kv   * self.head_dim, bias=cfg.bias)
        self.v    = nn.Linear(cfg.n_embd, self.n_kv   * self.head_dim, bias=cfg.bias)
        self.out  = nn.Linear(cfg.n_embd, cfg.n_embd,                  bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(
        self,
        x:       torch.Tensor,
        cos:     Optional[torch.Tensor] = None,
        sin:     Optional[torch.Tensor] = None,
        cache:   Optional[KVCache]      = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv,   self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv,   self.head_dim).transpose(1, 2)

        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        offset = 0
        if cache is not None:
            offset = cache.seen
            k, v = cache.update(self.layer_idx, k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Three cases: fresh sequence -> plain causal; single new token -> attends
        # to everything cached; chunk appended to a non-empty cache -> explicit
        # mask so query i (absolute position offset+i) only sees keys <= offset+i.
        attn_mask = None
        is_causal = False
        if offset == 0:
            is_causal = T > 1
        elif T > 1:
            rows = torch.arange(offset, offset + T, device=x.device).unsqueeze(1)
            cols = torch.arange(offset + T, device=x.device).unsqueeze(0)
            attn_mask = cols <= rows

        if key_mask is not None:
            # Batched decoding of sequences of different lengths, left-padded to one common length: every row shares the
            # cache position, and `key_mask` (B, offset+T; True = a real token) hides the padding. With RoPE, scores depend
            # only on relative positions, so shifting a sequence right by its padding changes nothing for its real tokens.
            Tk = k.shape[2]
            rows = torch.arange(offset, offset + T, device=x.device).unsqueeze(1)          # absolute query positions
            cols = torch.arange(Tk, device=x.device).unsqueeze(0)
            allowed = (cols <= rows)[None, None] & key_mask[:, None, None, :Tk]            # (B, 1, T, Tk)
            allowed = allowed | ((~allowed.any(-1, keepdim=True)) & (cols == rows)[None, None])   # padded query: see itself, no NaNs
            attn_mask, is_causal = allowed, False

        dropout_p = self.cfg.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )
        out         = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.drop(self.out(out))


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """Pre-norm transformer block: Attn → Residual → MLP → Residual."""

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        Norm = RMSNorm if cfg.norm_type == "rmsnorm" else lambda d, **kw: LayerNorm(d, bias=cfg.bias, **kw)
        kw   = {"eps": cfg.norm_eps}
        self.norm1  = Norm(cfg.n_embd, **kw)
        self.norm2  = Norm(cfg.n_embd, **kw)
        self.attn   = Attention(cfg, layer_idx=layer_idx)
        self.mlp    = SwiGLUMLP(cfg) if cfg.activation == "swiglu" else GELUMLP(cfg)

    def forward(
        self,
        x:     torch.Tensor,
        cos:   Optional[torch.Tensor] = None,
        sin:   Optional[torch.Tensor] = None,
        cache: Optional[KVCache]      = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos=cos, sin=sin, cache=cache, key_mask=key_mask)
        x = x + self.mlp(self.norm2(x))
        return x
