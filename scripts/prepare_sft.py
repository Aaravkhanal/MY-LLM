"""Tokenise conversation JSONL into SFT shards (uses the existing tokenizer; never retrains it).

    python scripts/prepare_sft.py --config configs/sft_tinystories.yaml
"""

import argparse
import json

from aarav_lm.config import ExperimentConfig
from aarav_lm.data.sft import build_sft_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    exp = ExperimentConfig.from_yaml(args.config).with_overrides(args.overrides)
    manifest = build_sft_dataset(exp)
    print(json.dumps({s: v["conversations"] for s, v in manifest["splits"].items()}))


if __name__ == "__main__":
    main()
