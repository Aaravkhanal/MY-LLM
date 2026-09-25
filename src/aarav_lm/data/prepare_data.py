"""
aarav-lm · Dataset Preparation
Author: Aarav Khanal

Prepares binary token files for training:

  shakespeare  — ~1 MB text, ~338 k tokens, downloads in seconds.
                 Great for a first smoke-test of the training loop.

  tinystories  — short synthetic English stories.
                 Better training signal.  Requires the 'datasets' library.

Output
------
  data/train.bin   (uint16 token IDs)
  data/val.bin     (uint16 token IDs)
"""

import argparse
from pathlib import Path

import numpy as np
import requests
import tiktoken
from tqdm import tqdm

from aarav_lm.paths import LEGACY_DATA_DIR as _DATA_DIR


def _write_bin(ids: list, path: Path):
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)
    size_kb = arr.nbytes / 1024
    print(f"  ✓  {path.name}  —  {len(ids):,} tokens  ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# TinyShakespeare
# ─────────────────────────────────────────────────────────────────────────────

def prepare_shakespeare(out_dir: Path = _DATA_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "shakespeare_raw.txt"
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    if not raw_path.exists():
        print(f"Downloading TinyShakespeare …")
        raw_path.write_text(requests.get(url).text, encoding="utf-8")
    else:
        print(f"Using cached  {raw_path.name}")

    text    = raw_path.read_text(encoding="utf-8")
    split   = int(len(text) * 0.9)
    enc     = tiktoken.get_encoding("gpt2")

    print("Tokenising …")
    train_ids = enc.encode_ordinary(text[:split])
    val_ids   = enc.encode_ordinary(text[split:])

    _write_bin(train_ids, out_dir / "train.bin")
    _write_bin(val_ids,   out_dir / "val.bin")
    print("Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# TinyStories
# ─────────────────────────────────────────────────────────────────────────────

def prepare_tinystories(out_dir: Path = _DATA_DIR, max_samples: int = 20_000):
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("'datasets' not installed — falling back to TinyShakespeare.")
        prepare_shakespeare(out_dir)
        return

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    print(f"Streaming {max_samples:,} TinyStories documents from HuggingFace …")
    ds      = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    stories = []
    for i, item in enumerate(tqdm(ds, total=max_samples, unit="doc")):
        if i >= max_samples:
            break
        stories.append(item["text"])

    n_train  = int(len(stories) * 0.9)
    splits   = {"train": stories[:n_train], "val": stories[n_train:]}

    for name, docs in splits.items():
        ids = []
        for doc in tqdm(docs, desc=f"Tokenising {name}", unit="doc"):
            ids.extend(enc.encode_ordinary(doc))
            ids.append(eot)
        _write_bin(ids, out_dir / f"{name}.bin")

    print("Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     choices=["shakespeare", "tinystories"], default="shakespeare")
    parser.add_argument("--max_samples", type=int, default=20_000,
                        help="Max stories to stream (tinystories only)")
    args = parser.parse_args()

    if args.dataset == "shakespeare":
        prepare_shakespeare()
    else:
        prepare_tinystories(max_samples=args.max_samples)
