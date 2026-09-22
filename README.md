# aarav-lm

A small but modern decoder-only language model I built from scratch on my MacBook Air M3 (8 GB).

The goal was to understand every part of the stack — tokenisation, attention, position encoding, the training loop, checkpointing, and inference — by actually writing it, not just reading about it.

---

## What it is

`aarav-lm` is a decoder-only transformer that can run in two architecture modes:

| Mode | Position | Norm | Activation | Attention |
|------|----------|------|------------|-----------|
| `gpt2` | Learned embeddings | LayerNorm | GELU | MHA |
| `modern` | RoPE | RMSNorm | SwiGLU | GQA |

Both modes are trained with the same loop and can be swapped with a single flag.

---

## Directory layout

```
aarav-lm/
├── config/         model hyperparameters and size presets
├── model/
│   ├── layers.py   LayerNorm, RMSNorm, RoPE, SwiGLU, Attention, Block, KVCache
│   └── core.py     AaravLM transformer (forward + generate + configure_optimizers)
├── tokenizer/      tiktoken BPE wrapper
├── data/
│   ├── prepare_data.py   download & tokenise Shakespeare / TinyStories
│   └── dataloader.py     cached memmap batch sampler
├── training/
│   ├── train.py          main training loop
│   └── sanity_check.py   single-batch overfit test
├── inference/
│   ├── generate.py       CLI sampler
│   └── chat.py           interactive REPL
├── evaluation/
│   ├── benchmark.py      throughput numbers (CPU vs MPS, cached vs naive)
│   └── bpb.py            bits-per-byte metric
├── api/
│   └── server.py         FastAPI inference server
└── tests/
    └── test_model.py     shape, causal mask, gradient, KV-cache tests
```

---

## Hardware

MacBook Air · Apple M3 · 8 GB Unified Memory

Benchmark on the 5M-parameter model with context length 256:

| Backend | Training (tok/s) | Inference naive | Inference + KV-cache |
|---------|-----------------|-----------------|----------------------|
| CPU | ~2,000 | — | — |
| MPS | ~8,000 | ~33 tok/s | ~80 tok/s **(2.4×)** |

---

## Quickstart

### 1 — Run the tests

```bash
PYTHONPATH=.. python tests/test_model.py
```

### 2 — Single-batch sanity check (must pass before training)

```bash
PYTHONPATH=.. python training/sanity_check.py
```

### 3 — Prepare data

```bash
# ~1 MB, 338 k tokens — fastest to start
PYTHONPATH=.. python data/prepare_data.py --dataset shakespeare

# ~50 k TinyStories documents — better quality
PYTHONPATH=.. python data/prepare_data.py --dataset tinystories --max_samples 20000
```

### 4 — Train

```bash
PYTHONPATH=.. python training/train.py \
  --arch modern --size 5m \
  --max_iters 500 --batch_size 16 --grad_accum_steps 4
```

### 5 — Generate text

```bash
PYTHONPATH=.. python inference/generate.py \
  --ckpt checkpoints/aarav_lm_modern_5m_best.pt \
  --prompt "To be or not to be,"
```

### 6 — Start the API server

```bash
uvicorn aarav_lm.api.server:app --port 8000
# POST /load  {"path": "checkpoints/aarav_lm_modern_5m_best.pt"}
# POST /generate  {"prompt": "...", "max_tokens": 80}
```

---

## Architecture walk-through

```
Token Embeddings   (or RoPE-augmented projections)
         ↓
 ┌──────────────────────────────┐
 │  RMSNorm                     │  ← pre-norm
 │  Causal Attention (GQA)      │  ← with KV-cache during inference
 │  + Residual                  │
 │  RMSNorm                     │
 │  SwiGLU MLP                  │
 │  + Residual                  │
 └──────────────────────────────┘  × N layers
         ↓
     RMSNorm
         ↓
     LM Head (weight-tied to token embedding)
```

---

## Model sizes

| Preset | Layers | Hidden | Heads | Context | ~Params |
|--------|--------|--------|-------|---------|---------|
| `5m`   |  6     | 256    | 8     | 256     |  5 M    |
| `10m`  |  8     | 384    | 6     | 512     | 10 M    |
| `20m`  | 10     | 512    | 8     | 512     | 20 M    |
| `30m`  | 14     | 512    | 8/4   | 512     | 30 M    |

All presets are calibrated to stay under 5–6 GB peak during training on an 8 GB M3.

---

## Roadmap

- [x] GPT-2 baseline model
- [x] RoPE + RMSNorm + SwiGLU + GQA (modern mode)
- [x] KV-cache inference
- [x] bfloat16 MPS training
- [x] FastAPI server
- [ ] TinyStories training to coherent output
- [ ] Scale to 10M / 20M
- [ ] SFT / instruction tuning
- [ ] MLX backend experiment
