"""
AaravLM Bits-Per-Byte (BPB) Evaluator
Tokenizer-invariant compression metric.
BPB measures how many bits of information the model requires to predict one UTF-8 byte of text.

Attribution:
- Adapted from nanogpt-mps-lab (MIT)
"""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from aarav_lm.config.config import ModelConfig
from aarav_lm.model.core import AaravLM
from aarav_lm.tokenizer.tokenizer import Tokenizer


@torch.no_grad()
def evaluate_bpb(
    model: AaravLM,
    raw_text: str,
    tokenizer: Tokenizer,
    device: torch.device,
    block_size: Optional[int] = None,
) -> float:
    """Calculates bits-per-byte (BPB) on a raw text string."""
    model.eval()
    if block_size is None:
        block_size = model.config.block_size

    # Count actual UTF-8 bytes
    num_bytes = len(raw_text.encode("utf-8"))
    assert num_bytes > 0, "Empty text provided"

    tokens = tokenizer.encode(raw_text)
    num_tokens = len(tokens)

    # Slice tokens into context windows
    total_nats = 0.0
    for i in range(0, num_tokens - 1, block_size):
        chunk_tokens = tokens[i : i + block_size + 1]
        if len(chunk_tokens) < 2:
            break
        x = torch.tensor([chunk_tokens[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk_tokens[1:]], dtype=torch.long, device=device)

        logits, loss = model(x, targets=y)
        # Multiply mean cross entropy (nats/token) by number of tokens in slice
        slice_len = x.shape[1]
        total_nats += loss.item() * slice_len

    # Convert nats to bits and divide by total UTF-8 bytes
    total_bits = total_nats / math.log(2.0)
    bpb = total_bits / num_bytes
    return bpb
