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

Run these from the `aarav-lm` directory. On this Mac, PyTorch is installed for the Python at `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3` (Homebrew's `python3` does not have it).

### 1 — Run the tests

```bash
PYTHONPATH=. python tests/test_model.py
```

### 2 — Single-batch sanity check (must pass before training)

```bash
PYTHONPATH=. python training/sanity_check.py
```

### 3 — Prepare data

```bash
# ~1 MB, 338 k tokens — fastest to start. Already prepared in data/.
PYTHONPATH=. python data/prepare_data.py --dataset shakespeare

# Short English stories. Better training signal than Shakespeare.
# Needs: pip install datasets
PYTHONPATH=. python data/prepare_data.py --dataset tinystories --max_samples 20000
```

### 4 — Train

```bash
# Safe batch for 8 GB. Checkpoints land in checkpoints/.
PYTHONPATH=. python training/train.py \
  --arch modern --size 10m \
  --max_iters 2000 --batch_size 8 --grad_accum_steps 4
```

Close browsers first. Plug the laptop in. This run is hours, not minutes.

### 5 — Generate text

```bash
PYTHONPATH=. python inference/generate.py \
  --ckpt checkpoints/aarav_lm_modern_5m_best.pt \
  --prompt "To be or not to be,"
```

### 6 — Train your own chat model, then serve it

WikiText did not need to be present. `data/prepare_chat.py` builds a pretraining corpus from Databricks Dolly answers and an instruction set (Dolly pairs that fit 256 tokens, plus a repeated seed of everyday questions).

```bash
PYTHONPATH=. python data/prepare_chat.py

# Language modeling. About 15 minutes on this Mac.
PYTHONPATH=. python training/train.py \
  --arch modern --size 10m --block_size 256 \
  --max_iters 400 --batch_size 8 --grad_accum_steps 4 \
  --data_dir data/pretrain --out_dir checkpoints/pretrain

# Teach it to answer. Loss is only on the assistant tokens.
PYTHONPATH=. python training/sft.py \
  --ckpt checkpoints/pretrain/aarav_lm_modern_10m_best.pt
```

The chat weights land in `checkpoints/aarav_chat.pt`.

### 7 — Serve it with API keys

```bash
python scripts/serve.py
```

The server prefers `checkpoints/aarav_chat.pt`. The first launch writes a key to `secrets/api_keys.txt` and prints it. Extra keys:

```bash
python scripts/create_key.py
```

Restart the server after creating a key. Every request except `GET /health` needs:

```
Authorization: Bearer ak-...
```

OpenAI-style clients point at this machine:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="ak-...")
print(client.chat.completions.create(
    model="aarav-lm",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    max_tokens=40,
).choices[0].message.content)
```

To let someone off this laptop call it, keep the Mac awake and open a tunnel:

```bash
caffeinate -dimsu python scripts/serve.py
# in another terminal, after: brew install cloudflared
cloudflared tunnel --url http://127.0.0.1:8000
```

Give them the `https://….trycloudflare.com/v1` base URL and a key you created for them. The tunnel URL changes every time you restart `cloudflared`. One generation runs at a time. Set `AARAV_CKPT` to serve a different checkpoint, such as the Shakespeare run.

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

## What this Mac can actually host

8 GB unified memory and a 256 GB SSD. Check free space with `df -h /` before downloading data; this disk had about 56 GB free when the server was added. That is enough to train and serve the presets in this repo. It is not enough to train a general chatbot, and a 7B model does not fit comfortably next to macOS.

The chat model this Mac can train is the 10M preset: about 14 million unique parameters, context 256, vocabulary 50,257. It is pretrained on the Dolly answers, then fine-tuned so the loss falls only on the reply. It can hold a short conversation and answer simple questions. It will miss facts a larger model would know, and replies stay short because the context window is 256 tokens.

`checkpoints/aarav_chat.pt` is the file the API loads. The older Shakespeare run is `checkpoints/aarav_lm_modern_5m_best.pt` (300 steps, validation loss 4.53, play-style text). `aaravlm_modern_5m_best.pt` was saved under an old module name and does not load.

## Roadmap

- [x] GPT-2 baseline model
- [x] RoPE + RMSNorm + SwiGLU + GQA (modern mode)
- [x] KV-cache inference
- [x] bfloat16 MPS training
- [x] FastAPI server with API keys and `/v1/chat/completions`
- [ ] TinyStories training to coherent output
- [ ] Scale to 10M / 20M
- [ ] SFT / instruction tuning
- [ ] Optional 4-bit instruct model behind the same API
- [ ] MLX backend experiment
