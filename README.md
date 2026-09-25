# aarav-lm

A small decoder-only language model **and the whole stack around it**, built from scratch and run on a
MacBook Air M3 with 8 GB of memory: tokenizer, data pipeline, trainer, evaluation, fine-tuning,
terminal chat, HTTP API and a web chat page.

Nothing here calls an external model or service. The weights are trained in this repository from
random initialisation.

```
$ python chat.py
System: model loaded — 5.0M parameters, context 256, device cpu, tiny-tinystories-v1-sft-int8.pt

You: Tell me a story about Zoe.
Model: Once upon a time, there was a little girl named Zoe. Zoe was very enthusiastic. She loved to
play with her toys in the kitchen. One day, she found a big box in the kitchen. …
```

## What was built, and what came out

| | |
|---|---|
| **Model** | 4.95M-parameter Llama-style transformer (RMSNorm, RoPE, SwiGLU, grouped-query attention, tied embeddings), 256-token context |
| **Tokenizer** | Byte-level BPE, 2,048 tokens, trained here on the training split only; versioned by SHA-256 |
| **Data** | 60 MB slice of TinyStories → 15.8M training tokens (docs/DATASETS.md) |
| **Pretraining** | 3,000 steps, 49.2M tokens (3.1 epochs), **79 minutes** on the M3; peak process memory 225 MB |
| **Result** | validation loss **7.66 → 1.642 (perplexity 5.17)**; test perplexity 5.09; 0.65 bits per byte |
| **Fine-tuning** | SFT on 19.8k story-request conversations (6.5 min) + LoRA support |
| **Size** | 19.8 MB fp32, **5.06 MB int8** (+0.01% loss) — both packaged in `models/` |
| **Tests** | 293 pytest tests + a 12-check end-to-end smoke test (~10 s) |

### How good is it, honestly?

The base model writes fluent, grammatical, well-formed children's stories: 98% of sentences are
capitalised correctly, ~0% non-words, 80% of stories end by themselves, and the character named in the
prompt was reused in all 20 name prompts of the evaluation set. On 55 held-out minimal-pair probes it scores **92.7%** (grammar 100%, story
logic 100%, common sense 79%; chance is 50%).

It is also a 5M-parameter model. Story logic drifts ("the pillow was not a pillow"), names and
pronouns sometimes disagree ("a little girl named Tom… he"), greedy decoding loops in 12.5% of
samples (sampling: 0%), and common-sense facts are shaky (it misses "snow is white", "an ant is
small"). It knows one domain. **It is not a general assistant.** The SFT model learned the chat
format and, for named characters, gender-consistent names; measured request-following beyond that is
weak — see [docs/RESULTS.md](docs/RESULTS.md), which reports the numbers that don't flatter it too.

## Quick start

```bash
# 1. environment (a repo-local venv that reuses your torch install; nothing global is touched)
python -m venv --system-site-packages .venv && .venv/bin/pip install -e ".[api,data,dev]" --no-deps
#    (needs torch, numpy, tokenizers, pyyaml, psutil; fastapi/uvicorn/pydantic for the API)

# 2. is everything fundamentally working?  (~10 s, no downloads, no real data)
python tests/smoke_test.py            # add --mps to also run a few steps on the Apple GPU

# 3. chat with the packaged model straight away
python chat.py                                                    # newest experiment, else the packaged SFT model
python chat.py --checkpoint models/tiny-tinystories-v1-int8.pt    # base model: continues your text
```

### Train it yourself

```bash
# data: first 60 MB of TinyStories (curl command in docs/DATASETS.md)
python scripts/prepare_data.py --config configs/tiny_tinystories.yaml      # tokenizer + shards + manifest
python scripts/train.py        --config configs/tiny_tinystories.yaml      # ~80 min; Ctrl-C is safe
python scripts/train.py        --resume experiments/exp_001_tiny_tinystories
python scripts/evaluate.py     --checkpoint experiments/exp_001_tiny_tinystories/checkpoints/best.pt

# fine-tune it into a chat model
python scripts/make_story_requests.py --stories data/raw/tinystories_v2_head.txt --out data/raw/story_requests_v1.jsonl
python scripts/prepare_sft.py  --config configs/sft_tinystories.yaml
python scripts/train.py        --config configs/sft_tinystories.yaml

python scripts/compare.py                        # side-by-side report of every experiment
python scripts/export_int8.py --checkpoint …     # 3.9x smaller checkpoint
python scripts/benchmark.py                      # throughput and memory on your machine
```

Any config value can be overridden on the command line: `train.lr=5e-4 train.max_steps=500`.

### Serve it

```bash
python scripts/create_key.py --label laptop            # key shown once; only its hash is stored
python scripts/serve.py --checkpoint models/tiny-tinystories-v1-sft-int8.pt
#   web chat  http://127.0.0.1:8000/     API: /health /model /stats /generate /chat  (+ "stream": true)
python scripts/serve.py --no-auth                      # localhost only, no key
```

## Terminal chat commands

`/help` `/clear` (forget the conversation) `/reset` (also clear system prompt and settings)
`/save [file]` `/load [file]` `/stats` `/system <text>` `/set temperature|top_k|top_p|max_tokens|repetition_penalty|seed <v>` `/exit`

## Repository layout

```
src/aarav_lm/
  config/      typed model + experiment configs (YAML, dotted overrides, strict type coercion)
  tokenizer/   byte-level BPE, special tokens, fingerprinting, injection-safe encoding
  data/        ingest · clean · dedup · split · token shards · packed batching · SFT conversations
  model/       RMSNorm · RoPE · SwiGLU · GQA attention · KV cache · AaravLM
  train/       trainer · checkpoints · schedule · monitoring · SFT trainer
  finetune/    LoRA (apply, train, merge)
  evaluation/  loss · text metrics · probes · edge cases · suite · regression · comparison
  inference/   engine (streaming, sampling, stop sequences) · chat template · int8 quantisation
  chat/        session + terminal UI            api/  FastAPI app, keys, rate limiter
  web/         the browser chat page
configs/  evals/  scripts/  tests/  docs/  models/  artifacts/  experiments/
```

## What is in the box

* **Reproducible training.** One seed governs weight init and data order. Resuming from a
  checkpoint — including after Ctrl-C — reproduces the uninterrupted run **bit for bit**; the tests
  prove it, and prove they fail if the optimizer or data-position state is not restored. Checkpoints
  are written atomically and refuse to store NaN weights.
* **Safe by construction.** Checkpoints load with `weights_only=True` (no code execution). User text
  can never become a control token. The API stores only hashed keys, rate-limits, caps sizes, bounds
  its queue, and serves the web page under a strict CSP.
* **Honest evaluation.** No judge model, so quality is measured with checkable proxies (documented as
  such in `evaluation/metrics.py`): held-out loss/perplexity/bits-per-byte, forced-choice probes,
  repetition and loop detection, entity consistency, non-word rate, 14 edge cases. The evaluation
  sets are versioned and hashed into every result, and the regression check refuses to compare
  results from different data, tokenizers or settings.
* **Independent review.** A separate reviewer with no stake in the code found nine real bugs (the
  worst: the web UI's Stop button could wedge the model lock); each is fixed with a regression test.
  See the commit history.

## Hardware notes (8 GB M3)

Details and measurements in [docs/PERFORMANCE.md](docs/PERFORMANCE.md). The short version: training
runs at ~18k tokens/s in ~1.3 GB of GPU memory; **bf16 was not faster than fp32**; **CPU decodes this
model 5× faster than the GPU** (so inference defaults to CPU below 50M parameters); the fanless Air
throttles, so measurements vary ±30% and checkpointing often is the right default.

## Documentation

| | |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | every experiment, what worked, what didn't |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | design decisions and why, checkpoint format, data flow |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | measured throughput, memory, int8, thermal behaviour |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | plan for serving other people (design only; nothing deployed) |
| [docs/DATASETS.md](docs/DATASETS.md) | data sources and licences |
| [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) | works that informed the design |
| [models/README.md](models/README.md) | the packaged checkpoints |

## Scaling path

The `small` preset (8 layers, d=384, vocab 4,096, context 512; 14.2M parameters) is defined and
tested but not yet trained. Nothing in the design stops at that size; the next steps are more data
(the pipeline streams and shards), a bigger tokenizer, then a cloud GPU for anything past ~100M
parameters. On 8 GB, roughly 30–50M parameters is the practical ceiling for pretraining.

Not built: preference tuning (DPO), continuous batching, real int8 kernels, multi-GPU training,
cloud deployment. See docs/DEPLOYMENT.md for when each becomes worth doing.

## Licence

MIT (see LICENSE). The training data has its own licence (CDLA-Sharing-1.0, docs/DATASETS.md).
