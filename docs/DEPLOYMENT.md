# Deployment design (future — nothing here is deployed)

The goal is that other people can use the model. The rule for this project is **make everything work
locally first, and don't buy infrastructure before it is needed**. This document is the plan and the
triggers for each step, not a description of running systems.

## Where things stand

```
Browser ──HTTP/SSE──► FastAPI (scripts/serve.py) ──► InferenceEngine ──► model + tokenizer
                       keys · rate limit · caps         one generation at a time      checkpoint on local disk
```

Already implemented and tested: bearer keys stored as SHA-256, per-key/IP token-bucket rate limiting,
request validation and caps, body-size limit, single-flight generation with `503 + Retry-After`,
real server-sent-event streaming, `/health` `/model` `/stats`, strict CSP for the bundled UI,
localhost-only by default (`--no-auth` refused elsewhere).

Measured on this machine (see PERFORMANCE.md): a single generation stream on CPU produced ~380
tokens/s for the 5M model through the HTTP API.

## Target architecture

```
                 ┌────────────┐   TLS    ┌───────────────┐        ┌──────────────────┐
 Frontend  ────► │ reverse    │ ───────► │ API gateway   │ ─────► │ inference server │ ──► model
 (web/mobile)    │ proxy/CDN  │          │ auth · quotas │        │ (queue + batcher)│      │
                 └────────────┘          │ rate limits   │        └──────────────────┘      ▼
                                         └───────┬───────┘                          object storage
                                                 ▼                                  (versioned checkpoints,
                                          metrics · logs                             tokenizers, manifests)
```

## Decisions, and what would trigger them

| Concern | Now | Next step | Trigger |
|---|---|---|---|
| **Hosting** | The Mac, localhost | One small CPU VM (a 5M–15M model decodes at hundreds of tok/s on a few cores) | Someone besides you needs to use it |
| **GPU** | none | Only for models ≳100M parameters or many concurrent users | Latency/throughput measured as the bottleneck |
| **Quantisation** | int8 export (size only; loads dequantised) | int8/4-bit *kernels* via a runtime that has them (llama.cpp/ggml, MLX on Apple silicon, ONNX Runtime) | Memory or CPU cost on the server matters |
| **Batching** | One request at a time (a lock) | Continuous batching of decode steps across requests; requires a batched KV cache | More than a handful of concurrent users |
| **Concurrency** | Queue up to 30 s, then 503 | Worker processes behind the gateway, one model replica each | Queue timeouts show up in `/stats` |
| **Streaming** | SSE, works | Same; keep proxies from buffering (`X-Accel-Buffering: no` is already set) | – |
| **Authentication** | Hashed bearer keys in a file | Per-user accounts / OAuth at the gateway, keys in a database with revocation and per-key quotas | Real users |
| **Rate limiting** | In-process token bucket | Shared store (Redis) so limits hold across replicas | More than one API process |
| **Monitoring** | `/stats` JSON | Prometheus metrics (latency, tokens/s, queue depth, errors), structured logs, alerting on 5xx and p95 latency | Anything public |
| **Model versioning** | Experiment dirs; tokenizer + config fingerprints; `/model` reports them | A registry layout `models/<name>/<version>/{checkpoint, tokenizer, manifest.json, eval_results.json}`, immutable, promoted only if `scripts/evaluate.py --baseline` passes | A second model version exists |
| **Rollout** | – | Blue/green: start the new version beside the old, shift traffic, keep rollback | Users depend on it |
| **Docker** | – | One image: Python slim + torch CPU + the package + a mounted model directory (weights are *not* baked into the image) | First remote deploy |
| **Cloud** | – | Cheapest first: a single VM. Managed containers (Cloud Run/Fargate) next. GPUs last | See above |
| **Secrets/TLS** | Keys file, 0600, git-ignored | Secrets manager; TLS terminates at the proxy | Any non-localhost exposure |

## Things to decide before going public (not technical)

* **Content and misuse:** a base story model is low risk, but any chat model needs a stated policy,
  logging rules (what is stored, for how long) and a way to report problems.
* **Licences:** the training data's licence (TinyStories: CDLA-Sharing-1.0) and any model-weight
  terms you want to apply; both are noted in DATASETS.md.
* **Cost ceiling:** a hard budget alert before the first cloud resource is created.

## Capacity back-of-envelope

Decode cost per token for the 5M model is ~34 MFLOP (`flops_per_token` counts training FLOPs, so
inference is a third of that), i.e. trivial for any CPU; memory-bandwidth and Python overhead
dominate. That is why single-stream CPU speed is already in the hundreds of tokens/s and why the
first scaling lever is batching, not bigger hardware. Re-measure with `scripts/benchmark.py` on the
target machine before choosing instance sizes.
