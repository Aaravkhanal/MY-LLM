"""
aarav-lm · Model Configuration
Author: Aarav Khanal

Defines hyperparameters for two architecture flavours:
  - 'gpt2'   — classic decoder-only transformer (learned position, GELU, LayerNorm)
  - 'modern' — current-gen decoder-only transformer (RoPE, SwiGLU, RMSNorm, GQA)

Four size presets are provided, all calibrated to fit comfortably inside
8 GB of Apple Unified Memory while leaving room for the OS and activations.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """Full configuration for a single AaravLM model."""

    # --- Architecture family ---
    arch: Literal["gpt2", "modern"] = "modern"

    # --- Core dimensions ---
    vocab_size: int   = 50257   # GPT-2 BPE vocabulary size
    block_size: int   = 256     # Maximum context / sequence length
    n_layer:    int   = 6       # Number of transformer blocks
    n_head:     int   = 8       # Number of query attention heads
    n_kv_head:  Optional[int] = None  # KV heads (None → same as n_head; < n_head → GQA)
    n_embd:     int   = 256     # Hidden / embedding dimension
    dropout:    float = 0.0     # Dropout (0 for pre-training, small value for fine-tuning)
    bias:       bool  = False   # Whether to add bias to Linear layers

    # --- Modern (LLaMA-style) knobs ---
    norm_type:           Literal["layernorm", "rmsnorm"] = "rmsnorm"
    norm_eps:            float = 1e-5
    activation:          Literal["gelu", "swiglu"]      = "swiglu"
    ffn_multiple_of:     int   = 64           # SwiGLU hidden dim is rounded to this multiple
    ffn_dim_multiplier:  Optional[float] = None
    rope_theta:          float = 10000.0      # Base for RoPE frequency computation
    tie_embeddings:      bool  = True         # Share embedding & LM-head weights

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        assert self.n_head % self.n_kv_head == 0, \
            f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"


# ---------------------------------------------------------------------------
# Preset factory functions — calibrated for M3 8 GB
# ---------------------------------------------------------------------------

def cfg_5m(
    arch: Literal["gpt2", "modern"] = "modern",
    vocab_size: int = 50257,
    block_size: int = 256,
) -> ModelConfig:
    """~5 M non-embedding parameters — instant feedback, great for debugging."""
    return ModelConfig(
        arch=arch,
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=6, n_head=8, n_kv_head=8, n_embd=256,
        norm_type="rmsnorm" if arch == "modern" else "layernorm",
        activation="swiglu" if arch == "modern" else "gelu",
        tie_embeddings=True,
    )


def cfg_10m(
    arch: Literal["gpt2", "modern"] = "modern",
    vocab_size: int = 50257,
    block_size: int = 512,
) -> ModelConfig:
    """~10 M non-embedding parameters — solid TinyStories baseline on M3."""
    return ModelConfig(
        arch=arch,
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=8, n_head=6, n_kv_head=6, n_embd=384,
        norm_type="rmsnorm" if arch == "modern" else "layernorm",
        activation="swiglu" if arch == "modern" else "gelu",
        tie_embeddings=True,
    )


def cfg_20m(
    arch: Literal["gpt2", "modern"] = "modern",
    vocab_size: int = 50257,
    block_size: int = 512,
) -> ModelConfig:
    """~20 M non-embedding parameters — noticeably better story generation."""
    return ModelConfig(
        arch=arch,
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=10, n_head=8,
        n_kv_head=8 if arch == "gpt2" else 4,  # GQA for modern arch
        n_embd=512,
        norm_type="rmsnorm" if arch == "modern" else "layernorm",
        activation="swiglu" if arch == "modern" else "gelu",
        tie_embeddings=True,
    )


def cfg_30m(
    arch: Literal["gpt2", "modern"] = "modern",
    vocab_size: int = 50257,
    block_size: int = 512,
) -> ModelConfig:
    """~30 M non-embedding parameters — upper bound for comfortable 8 GB training."""
    return ModelConfig(
        arch=arch,
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=14, n_head=8, n_kv_head=4, n_embd=512,
        norm_type="rmsnorm" if arch == "modern" else "layernorm",
        activation="swiglu" if arch == "modern" else "gelu",
        tie_embeddings=True,
    )
