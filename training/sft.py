"""
aarav-lm · Supervised Fine-Tuning (SFT / Instruction Tuning)
Author: Aarav Khanal

Fine-tunes a pre-trained AaravLM model on instruction-response pairs.

Format:
  <|user|>
  {instruction}
  <|assistant|>
  {response}
"""

import argparse
import math
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

from aarav_lm.config.config import ModelConfig
from aarav_lm.model.core import AaravLM
from aarav_lm.tokenizer.tokenizer import Tokenizer

# Sample synthetic instruction dataset for SFT demonstration
SAMPLE_INSTRUCTIONS: List[Tuple[str, str]] = [
    ("Write a story about a brave cat named Leo.", "Once upon a time, in a small quiet village, there lived a brave cat named Leo. Leo was not afraid of anything."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("Tell me a funny joke.", "Why don't scientists trust atoms? Because they make up everything!"),
    ("Explain gravity in simple terms.", "Gravity is an invisible force that pulls objects toward each other, like holding us down on Earth."),
    ("Write a poem about space.", "Stars above in dark of night,\nTwinkling with a friendly light,\nGalaxies so far away,\nShining till the break of day."),
]


def format_prompt(instruction: str, response: str = "") -> str:
    res = f"<|user|>\n{instruction}\n<|assistant|>\n"
    if response:
        res += f"{response}<|endoftext|>"
    return res


def train_sft(
    ckpt_path: str,
    out_dir: str = "checkpoints",
    epochs: int = 5,
    lr: float = 2e-4,
    device_name: str = "",
):
    device = torch.device(device_name or ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\n{'='*55}")
    print(f"  aarav-lm  ·  Supervised Fine-Tuning (SFT)")
    print(f"  Loading base model: {ckpt_path}")
    print(f"{'='*55}\n")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = AaravLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()

    tok = Tokenizer()
    optimizer = model.configure_optimizers(0.01, lr, (0.9, 0.95), device.type)

    # Encode sample dataset
    dataset = []
    for inst, resp in SAMPLE_INSTRUCTIONS:
        text = format_prompt(inst, resp)
        tokens = tok.encode(text)
        if len(tokens) <= cfg.block_size:
            dataset.append(tokens)

    print(f"Loaded {len(dataset)} instruction pairs for fine-tuning.\n")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for tokens in dataset:
            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"  Epoch {epoch:2d}/{epochs}  SFT Loss: {avg_loss:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    sft_ckpt_path = Path(out_dir) / "aarav_lm_sft_final.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, sft_ckpt_path)
    print(f"\n✓ Saved SFT checkpoint to {sft_ckpt_path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune aarav-lm on instructions")
    ap.add_argument("--ckpt", required=True, help="Base pre-trained model checkpoint")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    args = ap.parse_args()

    train_sft(args.ckpt, out_dir=args.out_dir, epochs=args.epochs, lr=args.lr)
