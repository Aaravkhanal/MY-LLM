# Architecture

`aarav-lm` is a small decoder-only language model and everything around it: tokenizer, data
pipeline, trainer, evaluation, inference engine, chat, API and web UI. Everything runs locally on
one machine and calls no external model or service.

## Data flow

```
 raw text / JSONL
      │  data/ingest.py      stream files (chunked, UTF-8 safe), count malformed records
      ▼
 data/clean.py ──► data/dedup.py ──► data/split.py        normalise · filter · exact-dedup · content-hash split
      │
      ├────────────► tokenizer/bpe.py                      byte-level BPE trained on the TRAIN split only
      ▼
 data/shards.py   train.bin / val.bin / test.bin            memmap token shards, validated headers
      │           + manifest.json (config, source SHA-256, tokenizer fingerprint) + stats
      ▼
 data/dataset.py  BatchStream                               packed windows, per-epoch shuffle, state = (epoch, pos)
      ▼
 train/trainer.py ──► checkpoints/ (step_*.pt, best.pt, final.pt)   atomic writes, exact resume
      │                     │
      │                     ├──► evaluation/suite.py        same fixed suite for every checkpoint
      │                     ├──► finetune (SFT/LoRA)        same Trainer, different data + init
      │                     └──► inference/engine.py        KV cache, streaming, stop sequences
      ▼                                   │
 experiments/exp_NNN_name/                ├──► chat/ (terminal)       python chat.py
   config.yaml env.json metrics.jsonl     └──► api/ (FastAPI) ──► web/ (browser UI)
   summary.json eval/*.json
```

## Module map

| Package | Responsibility | Depends on |
|---|---|---|
| `config` | `ModelConfig` (+ presets), `ExperimentConfig` (typed YAML, dotted overrides) | – |
| `tokenizer` | Byte-level BPE, special tokens, sha256 fingerprint, injection-safe `encode` | tokenizers |
| `data` | ingest, clean, dedup, split, shards, packed batching, SFT conversations | config, tokenizer |
| `model` | RMSNorm, RoPE, SwiGLU, GQA attention, KV cache, `AaravLM` | config |
| `train` | trainer, schedule, checkpoints, monitor, device/precision, SFT trainer | data, model, evaluation.loss |
| `finetune` | LoRA (apply / merge) | model |
| `evaluation` | loss/ppl/accuracy, text metrics, probes, edge cases, suite, regression, compare | inference |
| `inference` | engine (sampling, streaming), chat template, prompt builder, int8 quantisation | model, tokenizer |
| `chat` | session state, terminal UI | inference |
| `api` | FastAPI app, OpenAI-compatible routes, keys and accounts (SQLite), rate limiter | inference |
| `web` | static chat page (HTML/JS/CSS) | api |

Rule: data → tokenizer → model → train → evaluation → inference → chat/api. Nothing lower imports
anything higher, and nothing reaches into another package's internals.

## The model (v0 "tiny", 4.95M parameters)

| Choice | Value | Why |
|---|---|---|
| Family | Decoder-only transformer, Llama-style | RoPE + RMSNorm + SwiGLU is the current standard and needs no learned position table |
| Layers / width / heads | 6 / 256 / 8 query heads, 4 KV heads (GQA) | ~4.4M in the blocks; GQA shrinks the KV cache and K/V projections |
| Context | 256 tokens | Median TinyStories story is 194 tokens; attention and logits cost grow with context |
| Vocabulary | 2,048 (byte-level BPE) | A 50k vocab would add ~12.9M embedding parameters (3x the whole 4.4M transformer body) and make the logits tensor 25x bigger |
| Embeddings | Tied input/output | Saves 0.5M parameters, standard for small models |
| Init | N(0, 0.02); residual projections scaled by 1/√(2L) | GPT-2 recipe, keeps early activations stable |
| Attention | `scaled_dot_product_attention` (fused, causal) | Correct and fast on MPS; explicit mask only for chunked cache prefill |
| Dropout / bias | 0 / none | One or a few epochs over plenty of data; fewer parameters |
| Precision | fp32 weights; optional bf16 autocast | Measured: bf16 was ~8% faster, saved no memory, changed loss in the 4th decimal |
| Optimizer | AdamW (0.9, 0.95), weight decay 0.1 on matrices only | Embeddings and norms are not decayed |
| Schedule | linear warmup → cosine to 10% | Standard; LR peak 1e-3 is fine at this size |
| Batch | micro-batch 16 × 4 accumulation = 64 sequences = 16,384 tokens/step | Micro-batch 32 held 2.3 GB of Metal memory; 16 held 1.3 GB, same loss, slightly faster |
| Checkpoints | every 250 steps, keep 2 + best + final | Cheap insurance against a throttled or interrupted fanless laptop |

Scaling path: the `small` preset (8 layers, d=384, 2 KV heads, vocab 4096, context 512) is 14.2M
parameters; the design has no component that stops at that size. Nothing here trains a model that
can't fit in memory — at 1B+ parameters a different machine or cloud GPU is needed.

## Reproducibility

* One seed (`train.seed`) drives model init, data order and sampling; the data order depends only on
  `(seed, epoch)`, so the position in the data stream is two integers.
* A run's directory contains its resolved config, environment (torch/python/device/git commit),
  every metric, and a summary. `fingerprint()` hashes the config.
* Datasets carry a manifest with SHA-256 of every source file; tokenizers, shards and checkpoints
  carry the tokenizer's SHA-256, and mismatches are hard errors.
* Tests prove that resuming from a checkpoint (including after Ctrl-C) reproduces the uninterrupted
  run **bit for bit**, and that this check fails if either the optimizer or the data-stream state
  is not restored.

## Checkpoint format

Plain tensors and JSON-like values only, loaded with `torch.load(weights_only=True)`, so opening a
checkpoint cannot execute code. Writes are atomic (temp file, fsync, rename).

| File | Contains | Used by |
|---|---|---|
| `step_XXXXXXXX.pt` | model, optimizer, data-stream position, RNG state, counters, config | resume |
| `best.pt`, `final.pt` | model, model config, tokenizer fingerprint, meta (step, val loss, stage) | eval, chat, API, SFT |
| `*_int8.pt` | int8 matrices + fp32 leftovers | same loaders (dequantised on load) |

`meta.stage` is `pretrain` or `sft`; it decides whether chat/API use the instruction template or
plain text continuation.

## Fine-tuning

SFT is a separate stage (`stage: sft`) that reuses the trainer. It starts from a pretrained
checkpoint (architecture and tokenizer are verified), trains on conversations with an
assistant-only loss mask, and writes checkpoints with `stage: sft`. Optional LoRA freezes the base,
trains rank-r adapters, and merges them into plain weights on export. Continued pretraining is the
same `pretrain` stage with `init_from` set: only the weights carry over (fresh optimiser, schedule and data order),
and architecture and tokenizer must match exactly.
Not built yet: DPO/preference tuning, adapter hot-swapping, checkpoint merging across runs.
