"""Build the tokenizer and token shards for an experiment config.

    python scripts/prepare_data.py --config configs/tiny_tinystories.yaml
    python scripts/prepare_data.py --config configs/tiny_tinystories.yaml data.min_chars=200
"""

import argparse
import json

from aarav_lm.config import ExperimentConfig
from aarav_lm.data.pipeline import build_dataset, validate_dataset
from aarav_lm.tokenizer import AaravTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--retrain-tokenizer", action="store_true", help="ignore an existing tokenizer artifact")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides, e.g. data.dedup=false")
    args = ap.parse_args()

    exp = ExperimentConfig.from_yaml(args.config).with_overrides(args.overrides)
    manifest = build_dataset(exp, force_tokenizer=args.retrain_tokenizer)
    tok = AaravTokenizer.load(exp.tokenizer.artifact_dir)
    problems = validate_dataset(exp.data.out_dir, tok)
    if problems:
        raise SystemExit("dataset validation FAILED:\n  " + "\n  ".join(problems))
    print("[data] validation OK")
    print(json.dumps({s: v["tokens"] for s, v in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
