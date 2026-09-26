# AARAV-LLM

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

Two models have been trained from random initialisation on the M3, each with the same pipeline:

| | `tiny` (v1) | `small` (v2) |
|---|---|---|
| **Model** | 4.95M-parameter Llama-style transformer (RMSNorm, RoPE, SwiGLU, grouped-query attention, tied embeddings) | **14,162,304** parameters, same design: 8 layers, width 384, 6 query / 2 KV heads |
| **Context / vocabulary** | 256 / 2,048 | 512 / 4,096 |
| **Tokenizer** | Byte-level BPE trained here on the training split only; versioned by SHA-256 | same method; 3.96 characters per token on the corpus |
| **Data** | 60 MB of TinyStories → 15.8M training tokens | ~400 MB → **103.0M training tokens** (507,597 stories) |
| **Pretraining** | 3,000 steps, 49.2M tokens (3.1 epochs), **79 min** | 5,500 steps, 90.1M tokens (0.87 epoch), **4.1 h** |
| **Validation loss** | 7.66 → **1.642** (perplexity 5.17) | 8.43 → **1.398** (perplexity 4.05); test loss 1.406 |
| **Probes** (55 minimal pairs, chance 50%) | 92.7% | 94.5% |
| **Size** | 19.8 MB fp32, **5.06 MB int8** (+0.01% loss) — packaged in `models/` | 54 MB fp32 (not packaged yet) |

Around them: SFT and LoRA fine-tuning, an evaluation suite with a regression gate, terminal chat, an
HTTP API with hashed keys and rate limiting, and a web chat page. **318 pytest tests** plus a 12-check
end-to-end smoke test (~10 s).

Token counts are not parameter counts: the corpus is 103M *tokens*, the model 14.2M *parameters*
(about 6.4 tokens seen per parameter; loss was still falling when training stopped).

### How good is it, honestly?

The base models write fluent, grammatical, well-formed children's stories (`tiny`: 98% of sentences
capitalised correctly, ~0% non-words, 80% of stories end by themselves; the character named in the
prompt was reused in all 20 name prompts). On 55 held-out probes `tiny` scores **92.7%** and `small`
**94.5%** (grammar, story logic, common sense; chance is 50%).

They are also very small models. Story logic drifts ("the pillow was not a pillow"), names and
pronouns sometimes disagree, greedy decoding loops in 12.5% of `tiny` samples (sampling: 0%), and
common-sense facts are shaky. Each knows one domain. **It is not a general assistant**: it has no world
knowledge and cannot reason, summarise, answer questions, write code or do maths.

**Instruction following is the open problem.** The first SFT on `small` (exp_005, 60k conversations)
lowered loss but did **not** make it follow requests better (27% vs 23% pass rate for the untuned
model on 208 checkable prompts, within noise; requests for short or long stories passed 0 of 32). The
cause was found and is a data problem, not a model limit: each length request had only three
phrasings, which the model memorised. The generator now produces 70–84 phrasings each for the length and dialogue
requests (14–57 for the other kinds; `data/instruct_tasks.py`), evaluated on 312 held-out prompts with different wording. exp_006 tests the
fix; its results are being evaluated and will be added to [docs/RESULTS.md](docs/RESULTS.md), which
reports the numbers that don't flatter the model too. See [docs/AUDIT.md](docs/AUDIT.md) for the full
audit, gap analysis and plan.

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

# the 14M `small` model (needs ~400 MB of TinyStories in data/raw/, see docs/DATASETS.md; ~4 h)
python scripts/prepare_data.py --config configs/small_tinystories.yaml
python scripts/train.py        --config configs/small_tinystories.yaml

# verifiable instruction data + a held-out request-following eval, then SFT
python scripts/make_instruct_data.py --stories data/raw/tinystories_v2_400mb.txt \
    --out data/raw/instruct_v3.jsonl --eval-out evals/instruct_v3.jsonl --n-train 60000 --per-kind 24
python scripts/prepare_sft.py  --config configs/sft_small_instruct_v3.yaml
python scripts/train.py        --config configs/sft_small_instruct_v3.yaml
python scripts/evaluate.py     --checkpoint experiments/exp_006_sft_small_instruct_v3/checkpoints/best.pt \
    --instruct evals/instruct_v3.jsonl

python scripts/audit_dataset.py --source data/raw/tinystories_v2_400mb.txt   # train/held-out overlap, junk, lengths
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
| [docs/AUDIT.md](docs/AUDIT.md) | audit of the whole project: architecture, model and dataset reports, gap matrix, risk register, phased plan |
| [docs/CORRECTNESS.md](docs/CORRECTNESS.md) | running log of scientific problems found in training/evaluation: evidence, fix, test, what is still unmeasured |
| [docs/RESULTS.md](docs/RESULTS.md) | every experiment, what worked, what didn't |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | design decisions and why, checkpoint format, data flow |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | measured throughput, memory, int8, thermal behaviour |
| [docs/API.md](docs/API.md) | the HTTP API as it exists today: endpoints, fields, errors, streaming |
| [docs/SECURITY.md](docs/SECURITY.md) | controls that exist, known gaps, what must close before going public |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | plan for serving other people (design only; nothing deployed) |
| [docs/DATASETS.md](docs/DATASETS.md) | data sources and licences |
| [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) | works that informed the design |
| [models/README.md](models/README.md) | the packaged checkpoints |

## Scaling path and what is not built

The `small` preset is trained (above). Its pretraining loss was still falling, and only ~1/5 of the
2.2 GB TinyStories file has been used, so more tokens are the cheapest next gain, followed by better
instruction data. On 8 GB, roughly 30–50M parameters is the practical ceiling for pretraining; anything
larger needs rented GPU time.

Not built: OpenAI-style `/v1/...` endpoints, user accounts and per-user usage records, key
list/revoke/rotate through the API, tokens-per-minute limits, preference tuning (DPO), continuous
batching, real int8 kernels, multi-GPU training, cloud deployment. The order they will be tackled in,
and what has to be true first, is in [docs/AUDIT.md](docs/AUDIT.md).

## Licence

MIT (see LICENSE). The training data has its own licence (CDLA-Sharing-1.0, docs/DATASETS.md).
