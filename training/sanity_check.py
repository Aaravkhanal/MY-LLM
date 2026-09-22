"""
aarav-lm · Sanity Check  (Stage 7 — Overfit One Batch)
Author: Aarav Khanal

This is a mandatory health-check I run before every full training run.
The idea is simple:  take one tiny batch, train on it for ~100 steps, and
verify the loss collapses near zero.  If it doesn't, something is broken
in the forward pass or optimizer — better to catch it here than after
hours of training.

Run with:
    python -m aarav_lm.training.sanity_check
"""

import time
import torch

from aarav_lm.config.config import cfg_5m
from aarav_lm.model.core import AaravLM
from aarav_lm.tokenizer.tokenizer import Tokenizer


def sanity_check(arch: str = "modern", steps: int = 150, lr: float = 3e-3) -> bool:
    device  = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok     = Tokenizer()

    sample  = (
        "Once upon a time, in a bright sunny valley, a small robotic friend named Aarav "
        "learned to speak. Every day, Aarav read wonderful stories and brought joy to everyone!"
    )
    tokens = tok.encode(sample)
    seq    = len(tokens) - 1
    assert seq > 10

    x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([tokens[1:]],  dtype=torch.long, device=device)

    cfg    = cfg_5m(arch=arch, vocab_size=tok.vocab_size, block_size=max(256, seq + 16))
    model  = AaravLM(cfg).to(device)
    model.train()

    opt    = model.configure_optimizers(0.0, lr, (0.9, 0.95), device.type)

    print(f"\n{'─'*52}")
    print(f"  Sanity check  ·  arch={arch}  device={device}")
    print(f"{'─'*52}")

    t0         = time.time()
    init_loss  = None
    final_loss = None

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        v = loss.item()
        if init_loss is None:
            init_loss = v
        final_loss = v

        if step in (1, 25, 50, 75, 100, 125, 150) or v < 0.01:
            print(f"  step {step:4d}  loss={v:.6f}")
        if v < 0.01:
            print(f"  Early stop — loss collapsed at step {step}!")
            break

    elapsed = time.time() - t0
    print(f"\n  {init_loss:.4f}  →  {final_loss:.6f}  in {elapsed:.1f}s")

    assert final_loss < 0.1, f"FAILED: loss={final_loss:.4f} did not collapse"
    print("  ✓  Loss collapse confirmed!")

    # Quick generation sanity check
    model.eval()
    prompt_toks = tokens[:5]
    idx_out = model.generate(
        torch.tensor([prompt_toks], dtype=torch.long, device=device),
        max_new_tokens=seq - 4,
        temperature=0.0,
        use_cache=True,
    )
    print(f"\n  Prompt  : «{tok.decode(prompt_toks)}»")
    print(f"  Output  : «{tok.decode(idx_out[0].tolist())}»")
    print(f"{'─'*52}\n")
    return True


if __name__ == "__main__":
    sanity_check(arch="gpt2",   steps=120)
    sanity_check(arch="modern", steps=120)
    print("ALL SANITY CHECKS PASSED ✓")
