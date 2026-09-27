# Packaged models

Small int8 checkpoints so the chat and API work straight after cloning, without
retraining: two 5 MB `tiny` models (tokenizer `artifacts/tokenizer_tinystories_v1`) and the 14 MB `small` model (tokenizer
`artifacts/tokenizer_tinystories_v2`). The scripts find the tokenizer by the fingerprint stored inside each checkpoint.

| File | What it is | Val loss (ppl) | Notes |
|---|---|---|---|
| `tiny-tinystories-v1-int8.pt` | Base model: 4.95M parameters, pretrained from scratch for 3,000 steps (49M tokens) on a 60 MB TinyStories slice | 1.644 (5.18) | Continues text. `python chat.py --checkpoint models/tiny-tinystories-v1-int8.pt` |
| `small-14m-sft-int8.pt` | **The model the demo site serves:** the 14.16M-parameter `small` model (8 layers, width 384, context 512), pretrained 237.6M tokens (exp_004 then exp_010), then fine-tuned about two epochs on rule-checkable story requests (exp_013, training seed 1338). fp32 checkpoint 56.7 MB, int8 14.35 MB | held-out fine-tuning loss 1.2547 -> 1.2548 after quantisation (+0.01%) | `python chat.py --checkpoint models/small-14m-sft-int8.pt` or `scripts/serve.py --checkpoint models/small-14m-sft-int8.pt`. 4 of 5 greedy replies were identical to the fp32 model on a spot check. On unseen request wording it passes 52% (greedy) and 50% (sampled) against 40% and 35% for the untuned base, but this is one draw of three training seeds (45 to 53% greedy), see docs/RESULTS.md and docs/MODEL_CARD.md |
| `tiny-tinystories-v1-sft-int8.pt` | The base model fine-tuned for 600 steps on 19.8k templated story-request conversations | 1.421 (4.14), chat-format assistant tokens | Speaks the chat template; see limits below |

```
python chat.py                                    # picks the SFT model here
python chat.py --checkpoint models/tiny-tinystories-v1-int8.pt
python scripts/serve.py --no-auth --checkpoint models/tiny-tinystories-v1-sft-int8.pt   # http://127.0.0.1:8000
```

## What these models are and are not

They are small research artefacts: fluent, simple children's-story text from a model trained in ~80
minutes on a laptop. They are **not** general assistants. They know one domain (short simple
stories), have a 256-token context, and make the mistakes you would expect at this size: drifting
logic within a story, occasional repeated phrases, weak common-sense facts (they miss 4 of 19
common-sense probes such as "snow is white"). The SFT model follows the chat format and reuses a
character name from the request; it is **not** reliably following topic requests (see docs/RESULTS.md).

## Provenance

* Data: TinyStories V2 (GPT-4 generated stories), first 60 MB, CDLA-Sharing-1.0 — see docs/DATASETS.md.
  The stories in that dataset were written by another model, so these weights are a distillation of
  that text; keep the dataset licence in mind if you redistribute them.
* Code and training: this repository, from scratch (no pretrained weights, no external model calls).
* Exact configs and metrics: `configs/`, and `experiments/exp_00*/` (config, environment, every
  logged metric, evaluation results).
