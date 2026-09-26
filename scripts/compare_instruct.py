"""Compare two checkpoints on the SAME instruction-following prompts, with paired confidence intervals.

    python scripts/compare_instruct.py experiments/exp_006_x/eval/best__instruct_v3.json \
                                       experiments/exp_004_y/eval/best__instruct_v3.json --mode greedy

Positive A-B means the first checkpoint passes more. The interval resamples distinct prompts, with both
models' outcomes on a prompt kept together; a difference whose interval contains 0 is not evidence.
"""

import argparse
import json
from pathlib import Path

from aarav_lm.evaluation.paired import format_paired, paired_compare


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--mode", choices=["greedy", "sampled"], default="greedy")
    args = ap.parse_args()
    ra, rb = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    sa, sb = ra["eval_sets"]["instruct"]["sha256"], rb["eval_sets"]["instruct"]["sha256"]
    if sa != sb:
        raise SystemExit("the two results used different eval files: not comparable")
    name = lambda r: r["model"].get("experiment", "?")                                              # noqa: E731
    print(f"{args.mode} decoding, {name(ra)} (A) vs {name(rb)} (B)\n")
    print(format_paired(paired_compare(ra["instruction"][args.mode]["per_prompt"], rb["instruction"][args.mode]["per_prompt"]),
                        name(ra)[:12], name(rb)[:12]))


if __name__ == "__main__":
    main()
