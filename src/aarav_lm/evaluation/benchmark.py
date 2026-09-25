"""
aarav-lm · Benchmark
Author: Aarav Khanal

Measures training throughput and inference latency across devices.
Compare CPU vs MPS and cached vs naive generation on a single command.
"""

import time
import torch

from aarav_lm.config.config import cfg_5m
from aarav_lm.model.core import AaravLM


def _time_training(model: AaravLM, device: torch.device, batch: int = 8, steps: int = 15):
    cfg = model.cfg
    x   = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device=device)
    y   = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device=device)
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), device.type)

    for _ in range(3):   # warmup
        opt.zero_grad(set_to_none=True); _, l = model(x, targets=y); l.backward(); opt.step()
    if device.type == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True); _, l = model(x, targets=y); l.backward(); opt.step()
    if device.type == "mps":
        torch.mps.synchronize()
    dt = (time.time() - t0) / steps

    return dt * 1000, batch * cfg.block_size / dt   # (ms/step, tok/s)


def _time_inference(model: AaravLM, device: torch.device, prompt: int = 16, gen: int = 50):
    idx = torch.randint(0, model.cfg.vocab_size, (1, prompt), device=device)

    t0  = time.time()
    model.generate(idx.clone(), max_new_tokens=gen, temperature=1.0, use_cache=False)
    if device.type == "mps": torch.mps.synchronize()
    t_naive = time.time() - t0

    t0  = time.time()
    model.generate(idx.clone(), max_new_tokens=gen, temperature=1.0, use_cache=True)
    if device.type == "mps": torch.mps.synchronize()
    t_cache = time.time() - t0

    return gen / t_naive, gen / t_cache, t_naive / t_cache


def run():
    print(f"\n{'═'*55}")
    print(f"  aarav-lm  ·  benchmark")
    print(f"{'═'*55}")

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

    for arch in ("gpt2", "modern"):
        print(f"\n  [{arch.upper()} — 5M]")
        cfg = cfg_5m(arch=arch, vocab_size=50257, block_size=256)

        for dev_str in devices:
            dev   = torch.device(dev_str)
            model = AaravLM(cfg).to(dev)
            ms, tok_s = _time_training(model, dev)
            print(f"  {dev_str.upper():4s}  training  {ms:6.1f} ms/step  {tok_s:>9,.0f} tok/s")

        # Inference only on MPS
        if "mps" in devices:
            dev   = torch.device("mps")
            model = AaravLM(cfg).to(dev).eval()
            s_naive, s_cache, speedup = _time_inference(model, dev)
            print(f"  MPS   inference  naive={s_naive:,.1f} tok/s  "
                  f"cached={s_cache:,.1f} tok/s  ({speedup:.2f}x speedup)")

    print(f"\n{'═'*55}\n")


if __name__ == "__main__":
    run()
