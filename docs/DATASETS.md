# Datasets

| Name | Used for | Source | Licence |
|---|---|---|---|
| TinyStories V2 (GPT-4), first 60 MB | First milestone pretraining | https://huggingface.co/datasets/roneneldan/TinyStories (`TinyStoriesV2-GPT4-train.txt`) | CDLA-Sharing-1.0 |

## Reproducing the milestone slice

The full file is 2.2 GB; only the first 60,000,000 bytes are used, cut at the last complete
`<|endoftext|>` boundary:

```bash
mkdir -p data/raw
curl -sL -r 0-59999999 -o data/raw/tinystories_v2_head.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
python scripts/prepare_data.py --config configs/tiny_tinystories.yaml
```

`data/` is git-ignored. `manifest.json` in each processed dataset records the SHA-256 of every
source file, the config fingerprint and the tokenizer fingerprint.

## Result of the milestone build

73,139 stories in the slice; 2 dropped as too short; no exact duplicates. Split by content hash
(seed 1337): 71,764 train / 709 val / 666 test stories = 15.77M / 0.16M / 0.14M tokens with the
2048-token byte-level BPE (3.66 bytes per token, median story 194 tokens).
