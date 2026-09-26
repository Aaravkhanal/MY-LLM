"""Build the two controls that make instruction-following pass rates interpretable.

    python scripts/make_eval_controls.py --eval evals/instruct_v3.jsonl --stories data/raw/tinystories_v2_400mb.txt

Writes, next to the eval file:
  <eval>.chance.json            per task kind, the pass rate of a real story that ignores the request
  <name>.intemplate.jsonl       the same kinds and held-out stories, but request wording taken from the TRAINING
  <name>.intemplate.chance.json templates: separates "did not learn the task" from "learned only the wording it saw"

The held-out stories are the test split, so no model has been trained on them.
"""

import argparse
import json
from pathlib import Path

from aarav_lm.data.instruct_data import chance_levels, make_eval_prompts, write_jsonl
from aarav_lm.data.instruct_tasks import TRAIN_TEMPLATES
from aarav_lm.data.story_requests import load_stories
from aarav_lm.paths import resolve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, help="the held-out eval file (instruct_vN.jsonl)")
    ap.add_argument("--stories", nargs="+", required=True)
    ap.add_argument("--per-kind", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=1337, help="must equal data.seed of the pretraining config")
    ap.add_argument("--val-fraction", type=float, default=0.005)
    ap.add_argument("--test-fraction", type=float, default=0.005)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=1500)
    args = ap.parse_args()

    stories = load_stories([resolve(s) for s in args.stories], args.split_seed, args.val_fraction, args.test_fraction,
                           min_chars=args.min_chars, max_chars=args.max_chars)["test"]
    eval_path = resolve(args.eval)
    held_out = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
    chance_path = eval_path.with_suffix(".chance.json")
    chance_path.write_text(json.dumps(chance_levels(held_out, stories), indent=2) + "\n")
    print(f"wrote {chance_path}")

    in_template = make_eval_prompts(stories, args.per_kind, args.seed, templates=TRAIN_TEMPLATES)
    ctrl = eval_path.with_name(eval_path.stem + ".intemplate.jsonl")
    write_jsonl(in_template, ctrl)
    ctrl.with_suffix(".chance.json").write_text(json.dumps(chance_levels(in_template, stories), indent=2) + "\n")
    print(f"wrote {ctrl} ({len(in_template)} prompts) and its chance file")
    for kind, rate in json.loads(chance_path.read_text())["by_kind"].items():
        print(f"  chance {kind:13s} {100 * rate:5.1f}%")


if __name__ == "__main__":
    main()
