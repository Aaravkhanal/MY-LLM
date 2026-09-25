"""
aarav-lm · Custom Text File Data Preparation
Author: Aarav Khanal

Converts arbitrary .txt files into uint16 tokenized train.bin and val.bin.

Usage:
  python -m aarav_lm.data.prepare_text --input path/to/book.txt --out_dir data/my_book
  python -m aarav_lm.data.prepare_text --input doc1.txt doc2.txt --out_dir data/combined
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import tiktoken


def prepare_text_files(inputs: List[Path], out_dir: Path, val_ratio: float = 0.1):
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    print(f"Reading {len(inputs)} text file(s)...")
    full_text = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        full_text.append(text)

    combined = "\n\n".join(full_text)
    split_idx = int(len(combined) * (1.0 - val_ratio))

    train_text = combined[:split_idx]
    val_text   = combined[split_idx:]

    print("Tokenizing train split...")
    train_ids = enc.encode_ordinary(train_text)
    train_ids.append(eot)

    print("Tokenizing val split...")
    val_ids = enc.encode_ordinary(val_text)
    val_ids.append(eot)

    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr   = np.array(val_ids, dtype=np.uint16)

    train_path = out_dir / "train.bin"
    val_path   = out_dir / "val.bin"

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    print(f"\n✓ Saved {len(train_ids):,} tokens to {train_path} ({train_arr.nbytes / 1024:.1f} KB)")
    print(f"✓ Saved {len(val_ids):,} tokens to {val_path} ({val_arr.nbytes / 1024:.1f} KB)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare custom .txt files for aarav-lm")
    parser.add_argument("--input", nargs="+", required=True, type=Path, help="One or more .txt input files")
    parser.add_argument("--out_dir", type=Path, default=Path("data/custom"), help="Output directory")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio (default 0.1)")
    args = parser.parse_args()

    prepare_text_files(args.input, args.out_dir, args.val_ratio)
