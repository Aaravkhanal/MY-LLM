"""Run the fixed evaluation suite on a checkpoint, optionally checking for regressions.

    python scripts/evaluate.py --checkpoint experiments/exp_001_x/checkpoints/best.pt
    python scripts/evaluate.py --checkpoint ... --save-baseline evals/baselines/tiny_v1.json
    python scripts/evaluate.py --checkpoint ... --baseline evals/baselines/tiny_v1.json   # exit 1 on regression
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from aarav_lm.evaluation.regression import compare_to_baseline, format_report
from aarav_lm.evaluation.suite import run_suite, save_results
from aarav_lm.inference.engine import InferenceEngine
from aarav_lm.paths import REPO_ROOT, resolve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--prompts", default=None,
                    help="default: evals/chat_prompts_v1.jsonl for instruction-tuned checkpoints, else evals/prompts_v1.jsonl")
    ap.add_argument("--probes", default="evals/probes_v1.jsonl")
    ap.add_argument("--instruct", default="evals/instruct_v2.jsonl",
                    help="rule-checked instruction-following prompts (skipped if the file does not exist)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", help="results file (default: <experiment>/eval/<checkpoint>_results.json)")
    ap.add_argument("--baseline", help="results file to compare against; exit code 1 if a metric regresses")
    ap.add_argument("--save-baseline", help="also write the results here to use as a future baseline")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    run_dir = ckpt.parent.parent
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    engine = InferenceEngine.from_checkpoint(ckpt, resolve(cfg["tokenizer"]["artifact_dir"]), args.device)
    engine.meta["experiment"] = run_dir.name

    prompts = args.prompts or ("evals/chat_prompts_v1.jsonl" if engine.meta.get("stage") == "sft" else "evals/prompts_v1.jsonl")
    print(f"[eval] stage={engine.meta.get('stage', 'pretrain')}  prompts={prompts}")
    results = run_suite(engine, resolve(cfg["data"]["out_dir"]), resolve(prompts), resolve(args.probes),
                        max_new_tokens=args.max_new_tokens,
                        instruct_path=resolve(args.instruct) if resolve(args.instruct).exists() else None)
    out = Path(args.out) if args.out else run_dir / "eval" / f"{ckpt.stem}_results.json"
    print(f"[eval] wrote {save_results(results, out)}")
    if args.save_baseline:
        print(f"[eval] baseline saved to {save_results(results, resolve(args.save_baseline))}")

    if args.baseline:
        ok, report = compare_to_baseline(results, json.loads(resolve(args.baseline).read_text()))
        print("\nRegression check vs", args.baseline)
        print(format_report(report))
        print("\nRESULT:", "no regressions" if ok else "REGRESSION DETECTED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
