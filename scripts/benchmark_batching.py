"""How much does batching raise throughput? The same requests, one at a time and in batches of increasing size.

    python scripts/benchmark_batching.py --checkpoint experiments/exp_009_.../checkpoints/best.pt --device cpu

For each batch size B it generates `--requests` replies of `--new-tokens` tokens (different prompts, fixed seeds) in waves of B, and
reports completion tokens per second across all of them, the speedup over doing the requests one at a time, and the time one wave
takes (a user in a wave of B waits about that long for a reply of this length). Outputs are checked equal to the solo outputs, so a
speedup that came from a wrong answer would be reported as a failure.

Run it on an idle machine: other work slows both arms, so absolute numbers drop (the ratio is more robust, but still: idle).
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import yaml

from aarav_lm.chat.cli import tokenizer_dir_for
from aarav_lm.inference.batching import Job, generate_batch
from aarav_lm.inference.engine import GenerationConfig, InferenceEngine
from aarav_lm.paths import resolve

PROMPTS = ["Once upon a time there was a little girl named Mia.", "Tom had a big red ball.", "One day a dog and a cat went to the park.",
           "The sun was bright and the bird sang.", "A boy named Sam found a box.", "Lily loved her teddy bear very much.",
           "The old king had a magic key.", "In the forest lived a small brown rabbit."]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--repeat", type=int, default=2, help="repeat each measurement and keep the fastest (least disturbed) one")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ck = Path(a.checkpoint)
    cfgfile = ck.parent.parent / "config.yaml"
    tok_dir = resolve(yaml.safe_load(cfgfile.read_text())["tokenizer"]["artifact_dir"]) if cfgfile.exists() else None
    engine = InferenceEngine.from_checkpoint(ck, tok_dir or tokenizer_dir_for(ck, None), a.device)
    prompts = [engine.tok.encode(PROMPTS[i % len(PROMPTS)])[:24] for i in range(a.requests)]
    cfgs = [GenerationConfig(temperature=0.8, top_k=50, top_p=0.95, seed=100 + i, max_new_tokens=a.new_tokens, stop_at_eot=False) for i in range(a.requests)]

    def timed(fn):
        best = None
        for _ in range(a.repeat):
            t = time.perf_counter(); out = fn(); dt = time.perf_counter() - t
            best = (dt, out) if best is None or dt < best[0] else best
        return best

    solo_dt, solo = timed(lambda: [engine.generate(p, c) for p, c in zip(prompts, cfgs)])
    tokens = sum(r.generated_tokens for r in solo)
    rows = [{"batch": "solo (one at a time)", "seconds": round(solo_dt, 2), "tokens_per_second": round(tokens / solo_dt, 1), "speedup": 1.0, "wave_seconds": None, "exact": True}]
    for B in a.batch_sizes:
        def run():
            out = []
            for i in range(0, a.requests, B):
                out += generate_batch(engine, [Job(p, c) for p, c in zip(prompts[i:i + B], cfgs[i:i + B])])
            return out
        dt, out = timed(run)
        waves = -(-a.requests // B)
        rows.append({"batch": B, "seconds": round(dt, 2), "tokens_per_second": round(sum(r.generated_tokens for r in out) / dt, 1),
                     "speedup": round(solo_dt / dt, 2), "wave_seconds": round(dt / waves, 2),
                     "exact": all(o.token_ids == s.token_ids for o, s in zip(out, solo))})
    info = engine.info()
    result = {"model_parameters": info["params"], "device": info["device"], "requests": a.requests, "new_tokens": a.new_tokens, "rows": rows}
    if a.json:
        print(json.dumps(result, indent=2)); return
    print(f"{info['params'] / 1e6:.1f}M parameters on {info['device']}, {a.requests} requests x {a.new_tokens} tokens")
    print(f"{'batch':>22} {'seconds':>8} {'tokens/s':>9} {'speedup':>8} {'wave s':>7}  exact")
    for r in rows:
        print(f"{str(r['batch']):>22} {r['seconds']:>8} {r['tokens_per_second']:>9} {r['speedup']:>7}x {str(r['wave_seconds'] or ''):>7}  {'yes' if r['exact'] else 'NO'}")


if __name__ == "__main__":
    main()
