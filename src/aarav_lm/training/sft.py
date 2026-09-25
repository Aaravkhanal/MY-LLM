"""
Fine-tune AaravLM on instruction pairs.

Loss is applied only to the assistant tokens. The user prompt is context,
not something the model is trained to copy.

  PYTHONPATH=. python training/sft.py --ckpt checkpoints/pretrain/aarav_lm_modern_10m_best.pt
"""

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from aarav_lm.data.chat_format import decode_answer, encode_pair, user_prompt
from aarav_lm.model.core import AaravLM
from aarav_lm.tokenizer.tokenizer import Tokenizer

from aarav_lm.paths import CHECKPOINT_DIR, LEGACY_DATA_DIR


def _load_rows(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _encode_rows(tok, rows: list, block_size: int) -> list:
    encoded = []
    for row in rows:
        ids, prompt_len = encode_pair(tok, row["user"], row["assistant"])
        if prompt_len < len(ids) <= block_size and prompt_len > 1:
            encoded.append((ids, prompt_len))
    return encoded


def _batch(examples: list, device: torch.device):
    width = max(len(ids) for ids, _ in examples)
    batch = len(examples)
    x = torch.zeros(batch, width - 1, dtype=torch.long, device=device)
    y = torch.full((batch, width - 1), -1, dtype=torch.long, device=device)
    for i, (ids, prompt_len) in enumerate(examples):
        inp = ids[:-1]
        tgt = ids[1:]
        x[i, : len(inp)] = torch.tensor(inp, dtype=torch.long, device=device)
        for j, token in enumerate(tgt):
            if j + 1 >= prompt_len:
                y[i, j] = token
    return x, y


@torch.no_grad()
def _loss(model, examples, device, ctx, max_batches: int = 16) -> float:
    model.eval()
    total = 0.0
    seen = 0
    step = max(1, len(examples) // max_batches)
    picked = examples[::step][:max_batches]
    for ids, prompt_len in picked:
        x, y = _batch([(ids, prompt_len)], device)
        with ctx:
            _, loss = model(x, targets=y)
        total += loss.item()
        seen += 1
    model.train()
    return total / max(seen, 1)


@torch.no_grad()
def _sample(model, tok, device, question: str) -> str:
    model.eval()
    ids = tok.encode(user_prompt(question))
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=40, temperature=0.0, use_cache=True, stop_ids=[tok.eot])
    model.train()
    return decode_answer(tok, out[0, len(ids):].tolist())


def train_sft(
    ckpt_path: str,
    train_path: str,
    val_path: str,
    out_path: str,
    epochs: int = 2,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 3e-4,
    device_name: str = "",
):
    device = torch.device(device_name or ("mps" if torch.backends.mps.is_available() else "cpu"))
    ctx = (
        torch.amp.autocast(device_type="mps", dtype=torch.bfloat16)
        if device.type == "mps"
        else nullcontext()
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = AaravLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()

    tok = Tokenizer()
    train_ex = _encode_rows(tok, _load_rows(Path(train_path)), cfg.block_size)
    val_ex = _encode_rows(tok, _load_rows(Path(val_path)), cfg.block_size)
    if not train_ex:
        raise SystemExit(f"No training rows fit block_size {cfg.block_size}")

    steps_per_epoch = math.ceil(len(train_ex) / (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup = min(40, max(5, total_steps // 20))
    optimizer = model.configure_optimizers(0.1, lr, (0.9, 0.95), device.type)

    print(f"\n{'='*55}")
    print("  aarav-lm  ·  instruction fine-tune")
    print(f"  examples={len(train_ex):,}  val={len(val_ex):,}  block={cfg.block_size}")
    print(f"  steps={total_steps}  batch={batch_size}  accum={grad_accum}  device={device}")
    print(f"{'='*55}\n")

    best = float("inf")
    t0 = time.time()
    order = train_ex[:]
    cursor = 0

    def take_chunk():
        nonlocal order, cursor
        if cursor + batch_size > len(order):
            random.shuffle(order)
            cursor = 0
        chunk = order[cursor: cursor + batch_size]
        cursor += batch_size
        return chunk

    for step in range(1, total_steps + 1):
        if step < warmup:
            cur_lr = lr * step / warmup
        else:
            t = (step - warmup) / max(total_steps - warmup, 1)
            cur_lr = 3e-5 + 0.5 * (1 + math.cos(math.pi * min(t, 1.0))) * (lr - 3e-5)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(grad_accum):
            x, y = _batch(take_chunk(), device)
            with ctx:
                _, loss = model(x, targets=y)
                scaled = loss / grad_accum
            running += loss.item()
            scaled.backward()
            del x, y, loss, scaled
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "mps":
            torch.mps.empty_cache()

        if step % 25 == 0 or step == 1:
            print(f"  step {step:4d}/{total_steps}  loss={running / grad_accum:.3f}  lr={cur_lr:.2e}")

        if step % 100 == 0 or step == total_steps:
            val_loss = _loss(model, val_ex or train_ex, device, ctx)
            sample = _sample(model, tok, device, "What is 2 + 2?")
            print(f"           val={val_loss:.3f}  sample: {sample}")
            if val_loss < best:
                best = val_loss
                _save(model, cfg, out_path, step, best)
                print(f"           saved {out_path}")
            if device.type == "mps":
                torch.mps.empty_cache()

    _save(model, cfg, out_path, step, best)
    elapsed = time.time() - t0
    print(f"\n  done in {elapsed / 60:.1f} min  best val {best:.3f}")
    print(f"  checkpoint {out_path}\n")
    for question in ("Who are you?", "What is the capital of France?", "Say hello in Spanish."):
        print(f"  Q: {question}")
        print(f"  A: {_sample(model, tok, device, question)}\n")


def _save(model, cfg, path: str, step: int, val: float) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": cfg, "step": step, "best_val": val, "kind": "sft"},
        dest,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune aarav-lm on instructions")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--train", default=str(LEGACY_DATA_DIR / "sft" / "train.jsonl"))
    parser.add_argument("--val", default=str(LEGACY_DATA_DIR / "sft" / "val.jsonl"))
    parser.add_argument("--out", default=str(CHECKPOINT_DIR / "aarav_chat.pt"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train_sft(
        args.ckpt,
        args.train,
        args.val,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
    )
