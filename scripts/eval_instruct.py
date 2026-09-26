"""Run only the instruction-following evaluation (minutes, not the full 20-minute suite).

    python scripts/eval_instruct.py --checkpoint experiments/exp_006_x/checkpoints/best.pt \
        --instruct evals/instruct_v3.jsonl evals/instruct_v3.intemplate.jsonl

Writes <run>/eval/<checkpoint>__<evalfile>.json for each eval file, including per-prompt outcomes (for
scripts/compare_instruct.py) and chance levels when `<evalfile>.chance.json` exists.
"""

import argparse
import json
from pathlib import Path

import yaml

from aarav_lm.evaluation.instruct import load_chance, run_instruction_eval
from aarav_lm.evaluation.probes import load_jsonl
from aarav_lm.evaluation.suite import file_sha256, save_results
from aarav_lm.inference.engine import InferenceEngine
from aarav_lm.paths import resolve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--instruct", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    run_dir = ckpt.parent.parent
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    engine = InferenceEngine.from_checkpoint(ckpt, resolve(cfg["tokenizer"]["artifact_dir"]), args.device)
    engine.meta["experiment"] = run_dir.name
    for path in map(resolve, args.instruct):
        prompts = load_jsonl(path)
        chance_file = path.with_suffix(".chance.json")
        res = run_instruction_eval(engine, prompts, max_new_tokens=args.max_new_tokens,
                                   chance=load_chance(chance_file) if chance_file.exists() else None)
        for mode, r in res.items():
            pr = r["pass_rate"]
            print(f"[{path.name}] {mode}: pass {pr['rate']:.3f}  chance {pr.get('chance', float('nan')):.3f}  "
                  f"95% CI (clustered, {pr['n_distinct']} distinct prompts) {pr['ci95_clustered']}")
        out = run_dir / "eval" / f"{ckpt.stem}__{path.stem}.json"
        save_results({"model": engine.info(), "instruction": res,
                      "eval_sets": {"instruct": {"path": str(path), "sha256": file_sha256(path), "n": len(prompts)}}}, out)
        print(f"[{path.name}] wrote {out}")


if __name__ == "__main__":
    main()
