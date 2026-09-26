"""Build verifiable story-request conversations (training) and held-out evaluation prompts.

    python scripts/make_instruct_data.py --stories data/raw/tinystories_v2_400mb.txt \
        --out data/raw/instruct_v2.jsonl --eval-out evals/instruct_v2.jsonl --n-train 60000

Every training answer is a real corpus story that satisfies its request's checks; evaluation prompts
use different phrasings, unseen names, and features of stories from the held-out test split.
"""

import argparse
import json

from aarav_lm.data.instruct_data import make_conversations, make_eval_prompts, write_jsonl
from aarav_lm.data.story_requests import load_stories
from aarav_lm.paths import resolve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stories", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--per-kind", type=int, default=16, help="evaluation prompts per task kind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=1337, help="must equal data.seed of the pretraining config")
    ap.add_argument("--val-fraction", type=float, default=0.005)
    ap.add_argument("--test-fraction", type=float, default=0.005)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=1500)
    args = ap.parse_args()

    stories = load_stories([resolve(s) for s in args.stories], args.split_seed, args.val_fraction, args.test_fraction,
                           min_chars=args.min_chars, max_chars=args.max_chars)
    print({k: len(v) for k, v in stories.items()}, f"usable stories ({args.min_chars}-{args.max_chars} chars)")
    recs = list(make_conversations(stories, args.n_train, args.seed))
    n = write_jsonl(recs, resolve(args.out))
    kinds = {}
    for r in recs:
        if r["split"] == "train":
            kinds[r["task"]] = kinds.get(r["task"], 0) + 1
    print(f"wrote {n} conversations -> {resolve(args.out)}\ntask mix (train):", json.dumps(dict(sorted(kinds.items()))))
    prompts = make_eval_prompts(stories["test"], args.per_kind, args.seed)
    write_jsonl(prompts, resolve(args.eval_out))
    print(f"wrote {len(prompts)} held-out evaluation prompts -> {resolve(args.eval_out)}")


if __name__ == "__main__":
    main()
