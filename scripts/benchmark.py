"""Measure training and inference throughput and memory on this machine.

    python scripts/benchmark.py                       # default sweep on the tiny preset
    python scripts/benchmark.py --preset small --out bench.json

Every configuration runs in its own process so Metal/RSS memory numbers don't leak between
runs. Data is synthetic (random tokens): this measures compute and memory, not learning.
Run it on an otherwise idle machine; the M3 Air is fanless and throttles under sustained load.
"""

import argparse
import json
import subprocess
import sys
import time

import torch

from aarav_lm.config.config import get_preset
from aarav_lm.model.core import AaravLM
from aarav_lm.model.layers import KVCache
from aarav_lm.train.device import autocast_context, pick_device, seed_everything, synchronize
from aarav_lm.train.monitor import memory_snapshot, tensors_nbytes

TRAIN_SWEEP = [(8, "fp32"), (16, "fp32"), (32, "fp32"), (16, "bf16"), (32, "bf16")]


def child_train(args) -> dict:
    seed_everything(0)
    dev = pick_device(args["device"])
    cfg = get_preset(args["preset"])
    model = AaravLM(cfg).to(dev).train()
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), dev.type)
    B, T = args["batch"], cfg.block_size
    x = torch.randint(0, cfg.vocab_size, (B, T), device=dev)

    def step():
        with autocast_context(dev, args["precision"]):
            _, loss = model(x, targets=x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    synchronize(dev)
    t0 = time.perf_counter()
    n = args["steps"]
    for _ in range(n):
        step()
    synchronize(dev)
    dt = (time.perf_counter() - t0) / n
    mem = memory_snapshot(dev)
    return {"kind": "train", "batch": B, "seq": T, "precision": args["precision"], "step_ms": round(dt * 1000, 1),
            "tokens_per_sec": round(B * T / dt), "params_m": round(model.num_params() / 1e6, 2),
            "flops_per_token_m": round(model.flops_per_token() / 1e6, 1),
            "achieved_tflops": round(model.flops_per_token() * B * T / dt / 1e12, 3), **mem}


@torch.no_grad()
def child_infer(args) -> dict:
    seed_everything(0)
    dev = pick_device(args["device"])
    cfg = get_preset(args["preset"])
    model = AaravLM(cfg).to(dev).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 64), device=dev)
    n_new = min(128, cfg.block_size - 64)
    out = {"kind": "infer", "params_m": round(model.num_params() / 1e6, 2), "weights_mb": round(tensors_nbytes(model) / 1e6, 1)}
    for use_cache in (True, False):
        model.generate(prompt, 8, temperature=0.0, use_cache=use_cache)              # warmup
        synchronize(dev)
        t0 = time.perf_counter()
        model.generate(prompt, n_new, temperature=0.0, use_cache=use_cache)
        synchronize(dev)
        out[f"decode_tok_s_{'cache' if use_cache else 'nocache'}"] = round(n_new / (time.perf_counter() - t0))
    ids = torch.randint(0, cfg.vocab_size, (1, cfg.block_size), device=dev)
    kv = KVCache(cfg.n_layer, 1, cfg.block_size, cfg.n_kv_head, cfg.n_embd // cfg.n_head, dev, torch.float32)
    synchronize(dev)
    t0 = time.perf_counter()
    model(ids, cache=kv, last_only=True)
    synchronize(dev)
    out["prefill_tok_s"] = round(cfg.block_size / (time.perf_counter() - t0))
    out["kv_cache_mb"] = round(2 * kv.k.numel() * kv.k.element_size() / 1e6, 2)
    return {**out, **memory_snapshot(dev)}


def run_child(spec: dict) -> dict:
    r = subprocess.run([sys.executable, __file__, "--child", json.dumps(spec)], capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        return {"kind": spec["kind"], "error": r.stderr.strip().splitlines()[-1] if r.stderr else "failed", **spec}
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        spec = json.loads(args.child)
        print(json.dumps(child_train(spec) if spec["kind"] == "train" else child_infer(spec)))
        return

    results = []
    print(f"benchmark: preset={args.preset} device={pick_device(args.device)} torch={torch.__version__}\n")
    print(f"{'batch':>5} {'prec':>5} {'ms/step':>8} {'tok/s':>9} {'TFLOPS':>7} {'metal MB':>9} {'rss MB':>7}")
    for batch, prec in TRAIN_SWEEP:
        r = run_child({"kind": "train", "preset": args.preset, "device": args.device, "batch": batch,
                       "precision": prec, "steps": args.steps})
        results.append(r)
        if "error" in r:
            print(f"{batch:>5} {prec:>5}  FAILED: {r['error']}")
        else:
            print(f"{batch:>5} {prec:>5} {r['step_ms']:>8} {r['tokens_per_sec']:>9,} {r['achieved_tflops']:>7} "
                  f"{r.get('mps_driver_mb', r['rss_mb']):>9} {r['rss_mb']:>7}")
    r = run_child({"kind": "infer", "preset": args.preset, "device": args.device})
    results.append(r)
    print("\ninference:", json.dumps({k: v for k, v in r.items() if k not in ("kind",)}))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
