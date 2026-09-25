"""
aarav-lm · Training Loop
Author: Aarav Khanal

Single-device training pipeline for Apple Silicon (MPS) or CPU.

Highlights
----------
  bfloat16 autocast on MPS  — halves activation memory, fast on M-series chips
  cosine LR schedule        — linear warmup → cosine decay → floor
  gradient accumulation     — decouple batch size from VRAM
  gradient clipping         — prevents gradient explosion on early steps
  checkpointing             — saves best-val-loss checkpoint every eval interval
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch

from aarav_lm.config.config import ModelConfig, cfg_5m, cfg_10m, cfg_20m, cfg_30m
from aarav_lm.data.dataloader import get_batch
from aarav_lm.model.core import AaravLM
from aarav_lm.tokenizer.tokenizer import Tokenizer

_PRESETS = {"5m": cfg_5m, "10m": cfg_10m, "20m": cfg_20m, "30m": cfg_30m}
from aarav_lm.paths import CHECKPOINT_DIR, LEGACY_DATA_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(
    step: int,
    peak_lr: float,
    min_lr:  float,
    warmup:  int,
    total:   int,
) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / (warmup + 1)
    if step > total:
        return min_lr
    t = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * t)) * (peak_lr - min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(
    model:       AaravLM,
    eval_iters:  int,
    block_size:  int,
    batch_size:  int,
    device:      torch.device,
    data_dir:    Path,
    ctx,
) -> dict:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size, device, data_dir=data_dir)
            with ctx:
                _, loss = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    arch:             str   = "modern",
    size:             str   = "5m",
    max_iters:        int   = 300,
    batch_size:       int   = 16,
    grad_accum_steps: int   = 2,
    peak_lr:          float = 8e-4,
    min_lr:           float = 8e-5,
    warmup_iters:     int   = 20,
    eval_interval:    int   = 50,
    eval_iters:       int   = 5,
    out_dir:          str   = "",
    data_dir:         str   = "",
    device_name:      Optional[str] = None,
    block_size:       int   = 0,
):
    out_dir = out_dir or str(CHECKPOINT_DIR)
    data_dir = data_dir or str(LEGACY_DATA_DIR)
    device = torch.device(device_name or ("mps" if torch.backends.mps.is_available() else "cpu"))

    print(f"\n{'='*55}")
    print(f"  aarav-lm  ·  training run")
    print(f"  arch={arch.upper()}  size={size.upper()}  device={device}")
    print(f"  batch={batch_size}  accum={grad_accum_steps}  eff_batch={batch_size*grad_accum_steps}")
    print(f"{'='*55}\n")

    os.makedirs(out_dir, exist_ok=True)

    tok = Tokenizer()
    cfg = _PRESETS[size](arch=arch, vocab_size=tok.vocab_size)
    if block_size:
        cfg.block_size = block_size
    model = AaravLM(cfg).to(device)

    total_p   = sum(p.numel() for p in model.parameters())
    non_emb_p = model.num_params(exclude_embeddings=True)
    print(f"  Parameters : {total_p:,}  (non-embedding: {non_emb_p:,})\n")

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=peak_lr, betas=(0.9, 0.95), device_type=device.type
    )

    ctx = torch.amp.autocast(device_type="mps", dtype=torch.bfloat16) \
          if device.type == "mps" else nullcontext()

    best_val  = float("inf")
    t0        = time.time()
    tokens_so_far = 0

    x, y = get_batch("train", cfg.block_size, batch_size, device, data_dir=Path(data_dir))

    for step in range(1, max_iters + 1):
        lr = cosine_lr(step, peak_lr, min_lr, warmup_iters, max_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ── Evaluation & checkpoint ───────────────────────────────────────
        if step % eval_interval == 0 or step == 1:
            if device.type == "mps":
                torch.mps.empty_cache()
            losses = eval_loss(model, eval_iters, cfg.block_size, batch_size, device, Path(data_dir), ctx)
            print(f"  step {step:4d}/{max_iters}  "
                  f"train={losses['train']:.4f}  val={losses['val']:.4f}  lr={lr:.2e}")

            if losses["val"] < best_val:
                best_val  = losses["val"]
                ckpt_path = Path(out_dir) / f"aarav_lm_{arch}_{size}_best.pt"
                torch.save({
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "config":     cfg,
                    "step":       step,
                    "best_val":   best_val,
                }, ckpt_path)
                print(f"           → checkpoint saved  (val={best_val:.4f})")

        # ── Forward / backward with gradient accumulation ─────────────────
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(grad_accum_steps):
            with ctx:
                _, loss = model(x, targets=y)
                scaled  = loss / grad_accum_steps
            total_loss += loss.item()
            x, y = get_batch("train", cfg.block_size, batch_size, device, data_dir=Path(data_dir))
            scaled.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tokens_so_far += batch_size * grad_accum_steps * cfg.block_size

        if step % 25 == 0:
            elapsed = time.time() - t0
            tok_s   = tokens_so_far / elapsed
            print(f"  step {step:4d}  loss={total_loss/grad_accum_steps:.4f}  "
                  f"tok/s={tok_s:,.0f}  t={elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.0f}s  "
          f"({tokens_so_far/elapsed:,.0f} tok/s avg).")
    print(f"  Best val loss : {best_val:.4f}")

    final_path = Path(out_dir) / f"aarav_lm_{arch}_{size}_final.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "best_val": best_val}, final_path)
    print(f"  Final checkpoint → {final_path}\n")
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train aarav-lm")
    ap.add_argument("--arch",             choices=["gpt2", "modern"], default="modern")
    ap.add_argument("--size",             choices=["5m", "10m", "20m", "30m"], default="5m")
    ap.add_argument("--max_iters",        type=int,   default=300)
    ap.add_argument("--batch_size",       type=int,   default=16)
    ap.add_argument("--grad_accum_steps", type=int,   default=2)
    ap.add_argument("--lr",               type=float, default=8e-4)
    ap.add_argument("--out_dir",          type=str,   default="")
    ap.add_argument("--data_dir",         type=str,   default="")
    ap.add_argument("--block_size",       type=int,   default=0)
    args = ap.parse_args()

    train(
        arch=args.arch,
        size=args.size,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        peak_lr=args.lr,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        block_size=args.block_size,
    )
