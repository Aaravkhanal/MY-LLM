"""Build the story-request conversation file used for the SFT demo (no external model involved).

    python scripts/make_story_requests.py --stories data/raw/tinystories_v2_head.txt \
        --out data/raw/story_requests_v1.jsonl --n-train 20000
"""

import argparse

from aarav_lm.data.story_requests import load_stories, make_conversations, write_jsonl
from aarav_lm.paths import resolve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stories", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=1337, help="must equal data.seed of the pretraining config")
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--test-fraction", type=float, default=0.01)
    args = ap.parse_args()
    stories = load_stories([resolve(s) for s in args.stories], args.split_seed, args.val_fraction, args.test_fraction)
    print({k: len(v) for k, v in stories.items()}, "usable stories (250-820 chars) per pretraining split")
    counts = write_jsonl(make_conversations(stories, args.n_train, args.seed), resolve(args.out))
    print("conversations:", counts, "->", resolve(args.out))


if __name__ == "__main__":
    main()
