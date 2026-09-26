"""Dataset audit: near-duplicates, train/val/test overlap, length outliers, junk characters.

    python scripts/audit_dataset.py --source data/raw/tinystories_v2_400mb.txt --out experiments/dataset_audit_v2.json

The pipeline only removes EXACT duplicates, so this measures what it does not: how many held-out stories
share long word-sequences with training stories (leakage), and how many stories are near-copies of each
other. Overlap is the fraction of a story's 8-word shingles that also occur in the sampled training stories
(containment), so a value near 1.0 means the story is essentially in the training set.

To stay light on an 8 GB machine, held-out stories are compared against a random sample of `--train-sample`
training stories; the reported overlap is therefore a lower bound on true overlap with the full split.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from aarav_lm.data.split import split_of
from aarav_lm.paths import resolve

SEP = "<|endoftext|>"
WORD = re.compile(r"[a-z0-9']+")
K = 8


def stories(path: Path, chunk: int = 1 << 24):
    buf = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            buf += data
            parts = buf.split(SEP)
            buf = parts.pop()
            for p in parts:
                p = p.strip()
                if p:
                    yield p
    if buf.strip():
        yield buf.strip()


def shingles(text: str) -> np.ndarray:
    w = WORD.findall(text.lower())
    if len(w) < K:
        return np.empty(0, dtype=np.uint64)
    return np.fromiter((int.from_bytes(hashlib.blake2b(" ".join(w[i:i + K]).encode(), digest_size=8).digest(), "little")
                        for i in range(len(w) - K + 1)), dtype=np.uint64)


def dist(v):
    a = np.asarray(v, dtype=float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 4), "p50": round(float(np.percentile(a, 50)), 4),
            "p95": round(float(np.percentile(a, 95)), 4), "max": round(float(a.max()), 4),
            "ge_0.5": int((a >= 0.5).sum()), "ge_0.8": int((a >= 0.8).sum())} if a.size else {"n": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=1337, help="split seed of the dataset config")
    ap.add_argument("--val-fraction", type=float, default=0.005)
    ap.add_argument("--test-fraction", type=float, default=0.005)
    ap.add_argument("--train-sample", type=int, default=50000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    keep_p = None
    counts = {"train": 0, "val": 0, "test": 0}
    train_hashes, held = [], {"val": [], "test": []}
    lens, junk, non_ascii = [], 0, 0
    prefix_seen, prefix_dupes = {}, 0
    total = sum(1 for _ in stories(resolve(args.source)))
    keep_p = min(1.0, args.train_sample / max(1, total * (1 - args.val_fraction - args.test_fraction)))
    for s in stories(resolve(args.source)):
        sp = split_of(s, args.seed, args.val_fraction, args.test_fraction)
        counts[sp] += 1
        lens.append(len(s))
        if re.search(r"<[a-zA-Z/][^>]*>|https?://|�|\x00", s):
            junk += 1
        if not s.isascii():
            non_ascii += 1
        pre = hashlib.blake2b(" ".join(s.split()[:24]).encode(), digest_size=8).digest()   # same first 24 words
        prefix_dupes += pre in prefix_seen
        prefix_seen[pre] = 1
        if sp == "train":
            if rng.random() < keep_p:
                train_hashes.append(shingles(s))
        else:
            held[sp].append(shingles(s))
    ref = np.unique(np.concatenate(train_hashes)) if train_hashes else np.empty(0, dtype=np.uint64)

    def containment(hs):
        out = []
        for h in hs:
            if h.size:                                    # `ref` is sorted (np.unique): binary search, no re-sorting
                idx = np.searchsorted(ref, h)
                idx[idx == ref.size] = 0
                out.append(float((ref[idx] == h).mean()))
        return out

    # near-duplicates among the sampled training stories: a story whose shingles are mostly in *other* stories
    # is approximated by the sampled-train vs held-out check above; within-train we report the same-opening rate.
    lens = np.asarray(lens)
    report = {
        "source": str(args.source), "stories": total, "split_counts": counts,
        "train_stories_in_reference_sample": len(train_hashes), "reference_shingles": int(ref.size), "shingle_words": K,
        "val_overlap_with_train": dist(containment(held["val"])),
        "test_overlap_with_train": dist(containment(held["test"])),
        "chars": {"min": int(lens.min()), "p50": int(np.percentile(lens, 50)), "p99": int(np.percentile(lens, 99)),
                  "max": int(lens.max())},
        "stories_shorter_than_200_chars": int((lens < 200).sum()), "stories_longer_than_2000_chars": int((lens > 2000).sum()),
        "stories_with_html_url_or_replacement_chars": junk, "stories_with_non_ascii": non_ascii,
        "stories_sharing_their_first_24_words_with_an_earlier_story": prefix_dupes,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = resolve(args.out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")


if __name__ == "__main__":
    main()
