# Project audit, gap analysis and phased plan

Date: 2026-09-26. Everything below was measured or read from this repository unless marked *(design)*
or *(not measured)*. Raw evidence: `experiments/exp_00N_*/`, `experiments/dataset_audit_v2.json`
(regenerate with `scripts/audit_dataset.py`), and the other docs in this folder.

## 0. Where the brief and the repository disagree

| Brief said | Repository |
|---|---|
| Model is ~1M tokens in scale | The latest run (exp_004) is a **14.16M-parameter** model trained on **90.1M tokens** (train split: 103.0M tokens, 507,604 stories). The first model (exp_001) was 4.95M parameters on 49.2M tokens seen (15.8M unique). |
| "Audit first" | Done in an earlier session; the code has since been reviewed by an independent pass (nine defects fixed with tests). This document is the consolidated report. |
| Build inference, server, keys, rate limits | Built and tested (see the gap matrix, §11). What is missing is listed there, not assumed. |
| Train is running | exp_004 finished; exp_005 (SFT) finished; exp_006 (SFT v3) was running when this was written. |

## A. Current architecture

Full description: [ARCHITECTURE.md](ARCHITECTURE.md). The brief's pipeline, mapped to files:

| Stage | Module / script | Notes |
|---|---|---|
| Dataset ingest | `data/ingest.py`, `scripts/prepare_data.py` | streaming, UTF-8 safe, counts malformed records |
| Cleaning + dedup | `data/clean.py`, `data/dedup.py` | length/encoding filters; **exact** duplicates only |
| Split | `data/split.py` | content-hash split, seed 1337: stable when data is appended |
| Tokenizer | `tokenizer/bpe.py` | byte-level BPE trained on the **train split only**; sha256 fingerprint |
| Tokenization → shards | `data/shards.py` | memmap `.bin` + `manifest.json` (source SHA-256, config, tokenizer fingerprint) |
| DataLoader | `data/dataset.py` | packed windows, per-epoch shuffle, position = (epoch, index) |
| Training | `train/trainer.py`, `train/sft_trainer.py`, `finetune/lora.py` | AdamW, cosine, clipping, exact resume |
| Checkpointing | `train/checkpoint.py` | atomic writes; weights-only loadable; rolling + best + final |
| Validation / evaluation | `evaluation/*`, `scripts/evaluate.py` | one fixed suite for every checkpoint |
| Model selection | `scripts/compare.py`, `evaluation/regression.py` | `--baseline` gate exits non-zero on regression |
| Inference | `inference/engine.py` | KV cache, sampling, streaming, stop sequences, int8 load |
| Serving / API | `api/app.py`, `scripts/serve.py`, `web/` | FastAPI, SSE, bundled chat UI |
| Authentication | `api/auth.py` | hashed bearer keys in a file, token-bucket limiter |
| Users / applications | – | **not built** (Phase 4) |

Dependency rule: data → tokenizer → model → train → evaluation → inference → chat/api. Inference does
not import the trainer.

## B. Repository contribution report

`aarav-lm` is the user's own repository. The four cloned repositories were audited in full and
**no code was copied from them**; they informed design choices (attribution in
[THIRD_PARTY.md](THIRD_PARTY.md)).

| Repository | Useful | Unused | Conflicts | Recommended use |
|---|---|---|---|---|
| **nanoGPT** (Karpathy, MIT) | AdamW decay/no-decay split; scaled residual init; memmap token files | GPT-2 50k tokenizer, OpenWebText prep, DDP/`torch.compile` CUDA paths | 50k vocab would be 3× the whole 4.4M transformer body | Reference for the training loop only |
| **llama2.c** (Karpathy, MIT) | RMSNorm + RoPE + SwiGLU + GQA block design | C inference (`run.c`), its export format, Llama tokenizer | Fixed 32k SentencePiece vocab; float32 export not needed here | Reference for the architecture and for a future C/edge export |
| **llm.c** (Karpathy, MIT) | Shard header + magic-number idea; PyTorch-vs-C parity testing habit | CUDA/C training, GPT-2 weights format | CUDA-first, no MPS | Reference only; useful if an exact, dependency-free CPU reference is ever needed |
| **nanogpt-mps-lab** (P. Lin) | `-1` masked SFT targets; KV cache on Apple silicon | Downloader for an external chat dataset (`prepare_chat.py`), wandb hooks | Downloads external data and depends on GPT-2 BPE | Reference for MPS pitfalls; nothing imported |

## C. Model report

Parameter counts below were **recomputed by hand** and equal the trainer's numbers exactly.

| | `tiny` (exp_001–003) | `small` (exp_004–006) |
|---|---|---|
| Architecture | decoder-only, Llama-style ("modern") | same |
| Layers / hidden | 6 / 256 | 8 / 384 |
| Query heads / KV heads / head dim | 8 / 4 / 32 | 6 / 2 / 64 |
| FFN (checked in the state dict) | SwiGLU, 704 hidden | SwiGLU, 1,024 hidden |
| Context | 256 | 512 |
| Vocabulary | 2,048 | 4,096 |
| Norm / positions / bias | RMSNorm (eps 1e-5) / RoPE (θ=10,000) / none | same |
| Embeddings | tied input/output | tied |
| Parameters | 4.95M | **14,162,304** (blocks 12,589,440 + embedding 1,572,864) |

Hand count for `small`, per layer: attention 384·384 + 2·(384·128) + 384·384 = 393,216; FFN 3·384·1,024
= 1,179,648; two RMSNorm vectors 768; total 1,573,632 × 8 layers = 12,589,056, plus the final norm
384 = 12,589,440 — matching `summary.json`.

Covered by `tests/test_model.py`: output shapes and loss, untrained loss near uniform, ignore-index
masking, causal mask, gradient flow, KV-cache equivalence with full recompute, chunked prefill equals full
forward, scaled residual init, weight-decay groups (no decay on embeddings/norms), tied embeddings counted once.

## D. Training report

**exp_004 (`small`, pretraining):** micro-batch 1 × accumulation 32 × 512 tokens = 16,384 tokens/step,
5,500 steps, AdamW (0.9, 0.95), weight decay 0.1 on matrices, LR 8e-4 → 8e-5 cosine, warmup 200,
clip 1.0, fp32, MPS. **Result:** 90.1M tokens in 4.09 h; val loss 8.43 → **1.398** (ppl 4.05); test loss
1.406; top-1 0.637; probes 94.5% (chance 50%).

**Health check (`metrics.jsonl`, 220 logged points):** no NaN/Inf; no step-to-step loss jump above 25%;
gradient norm peaked at 10.5 in the first 25 steps (all clipped during warmup) and sat at ≈0.5 with 0
clipped steps by the end; validation loss fell at all 11 evaluations (2.20, 1.88, 1.74, 1.65, 1.60,
1.54, 1.49, 1.46, 1.43, 1.41, 1.40). Train loss (1.41) ≈ val loss (1.40) and less than one epoch was
seen (0.87), so **no overfitting**. Improvement is flattening: the last 500 steps gained 0.9%.
Throughput median 6.4k tokens/s (min 1.6k, max 8.6k, varying with thermals and other work), 0.8–2 GB Metal memory; micro-batch 1 was measured fastest on this machine.

**Observed problems**
1. exp_005 (SFT, 60k conversations): validation loss improved (1.49 → 1.29) but **request-following
   did not**: 27% greedy vs 23% for the un-tuned model (208 prompts, within noise). Per kind: dialogue
   0% → 100%; named-character and place requests up; **length requests 0/32** (the base model "passed"
   only because it happens to write short stories). Cause found: the generator had 2–3 phrasings per
   length/dialogue request and the eval used a phrasing not seen in training, so the model memorised
   strings instead of learning the concept. Also, the eval repeated one prompt 16× (one sample under
   greedy decoding), so its confidence intervals were too narrow. Fixed in `instruct_tasks.py` (70–84
   phrasings per kind) and a 312-prompt eval with 4–6 held-out phrasings; exp_006 tests the fix.
2. Unseen names are copied only 12% of the time after SFT (base 4%). A 14M model has little capacity for it.
3. `exp_004/env.json` records `git dirty: true`: the code that trained it was not exactly a commit.
   Now committed (`d1d98cc`); future runs should be started from a clean tree.
4. The fanless Air throttles (±30% step time) and swaps (1.4 GB of 2 GB swap in use during
   concurrent jobs). Running an audit alongside training slowed it by roughly 40% (1.26 → 1.77 s/step).

**Potential improvements, ranked by expected value** (each is one experiment, compared on the same eval)
1. Data diversity for SFT (done in v3) — cheap, targets the observed failure.
2. More pretraining tokens: exp_004 saw 6.4 tokens/parameter; loss is still falling. The corpus has
   ~2.2 GB; ~5× more unused text is available.
3. Bigger model only after 2, on rented GPU time; at 8 GB unified memory, `small` is close to the useful ceiling.
4. Decoding settings (repetition penalty / no-repeat n-gram) against loop rate — measured, not guessed.
5. Not worthwhile now: bf16 (measured no faster on MPS), Flash Attention / `torch.compile` (CUDA-oriented or
   unmeasured benefit on MPS), gradient checkpointing (memory is not the constraint at this size).

## E. Dataset report

| | |
|---|---|
| Source | TinyStories V2 (GPT-4), first ~400 MB; CDLA-Sharing-1.0 |
| Stories | 512,647 read → 507,597 train / 2,532 val / 2,486 test kept (content-hash split; `manifest.json`) |
| Tokens (4,096-vocab BPE) | train 103.0M; val and test about 0.5M each |
| Story length | median 181 tokens, p95 397, max 1,876 |
| Rejected by cleaning | 31 too short, 1 malformed UTF-8; **0 exact duplicates** (the audit script's own parse counts 512,622 stories; the ±25 difference is parsing, not filtering) |
| Suspicious content | 2 stories with HTML/URLs/replacement chars; 13,753 (2.7%) non-ASCII; 13 shorter than 200 characters |
| Repetitive openings | 12,938 stories (2.5%) share their first 24 words with an earlier story (expected in TinyStories) |
| **Train/held-out overlap** | share of a held-out story's 8-word sequences found in a random 10% sample of train stories: val mean 0.114, p95 0.257, **max 0.495; 0 stories ≥ 0.5** (test: mean 0.114, max 0.491) |

Limits of the leakage check: it compares against **10%** of the training stories, so it can miss a
near-copy in the other 90%; the reported overlap is a lower bound. It shows there is no *systematic*
contamination and that overlap is the ordinary phrase reuse of a templated corpus. Near-duplicates
*within* train were not measured (they cost capacity, not validity). The same check on `instruct_v3`
evaluation stories is inherited: they come from the test split.

**Is the data enough?** The corpus is a synthetic children's-story set designed so small models can
learn fluent English, so it is a good fit for 5–30M parameters. It cannot teach anything else.

| Level | Reached? |
|---|---|
| Prototype feasibility (whole pipeline works end to end) | **Yes** |
| Research experiment (controlled comparisons, honest metrics) | **Yes**, one seed per configuration |
| Demonstration model (fluent short stories, follows simple requests) | **Partly**: fluent and grammatical; follows named-character requests; unreliable on topic/length (exp_006 pending) |
| Production-quality general LLM | **No, and not on this path.** It has no world knowledge, cannot reason, summarise, answer questions, write code or do maths. Those need orders of magnitude more parameters, data and compute; no result here says otherwise. |

## F. Evaluation plan

Implemented (`scripts/evaluate.py`; identical for every checkpoint):

| Layer | What | Notes |
|---|---|---|
| Intrinsic | val/test loss, perplexity, top-1/top-5, bits-per-byte | tokenizer-independent number: bits/byte |
| Generation | 40 (pretrain) / 30 (chat) fixed prompts, greedy and sampled: capitalisation, non-word rate, loop rate, distinct-n, natural-ending rate, name reuse, length | proxy metrics; no judge model |
| Capability probes | 55 minimal pairs (grammar, story logic, common sense), choices verified single-token | chance 50% |
| Instruction following | 312 held-out prompts (13 request kinds × 24), rule-based checks, unseen names, Wilson 95% CIs | replaces the earlier 208-prompt set |
| Edge cases | 14 (single-character / whitespace / newline / Unicode / emoji prompts, control-token text inert, prompt and generation longer than context, determinism, seeded reproducibility, no state leaking between requests, stop sequences, tokenizer round-trip) | must all pass |
| Regression gate | `--baseline results.json`; refuses to compare different datasets/tokenizers | exit code 1 on regression |
| Comparison | `scripts/compare.py` → `experiments/COMPARISON.md` | records params, tokens, losses, speed, memory |

**Selection rule:** choose the checkpoint by evidence — validation loss for pretraining; for chat models,
instruction-following pass rate with CIs, then probes (no forgetting), then loop rate — never simply the latest.

**Deliberately not built:** the brief's general-knowledge, reasoning, summarisation, QA, code and
maths suites. This model cannot do them, so they would only record zeros. Instead, edge cases and an
out-of-domain check document the *boundary* honestly. A hallucination test is meaningful only for a
model that makes factual claims; this one writes fiction. Revisit if the model or corpus changes.

**Statistical caution:** one run per configuration. With 24 prompts per kind the 95% interval is about
±20 points, so only large per-kind gaps (e.g. 0% vs 100%) are conclusive; the 312-prompt aggregate is
about ±5 points. Under greedy decoding a repeated prompt adds no information. Seed-to-seed variance is *not measured*.

## G. Inference plan

Built: `InferenceEngine` (checkpoint → text; temperature, top-k, top-p, repetition penalty, no-repeat
n-gram, stop sequences, streaming, KV cache, int8-file loading), separate from the trainer.
Measured (`tiny`): CPU 732 tok/s vs MPS 148 tok/s at batch 1, so small models default to CPU; KV cache
~6× over recompute. Not yet measured for `small`: tokens/s, time-to-first-token, context-length impact
(`scripts/benchmark.py` supports it). Next: benchmark `small`; batched decoding only when concurrency
demands it; real int8 kernels only when serving cost matters (MPS/PyTorch have none).

## H. Deployment plan

Detailed in [DEPLOYMENT.md](DEPLOYMENT.md). The path, each step gated by a trigger:
MacBook (now) → local server with keys (done) → one small CPU VM behind a TLS reverse proxy
(the 5M model decodes at hundreds of tok/s on CPU, measured; the 14M model is *not yet measured*; **no GPU should be needed**) → shared rate-limit
store and metrics → containers/managed hosting only if load requires. The Mac should not host the
public API (sleep, thermals, RAM). Rough budget: a small VM is a few USD to ~20 USD per month *(estimate,
not quoted)*; set a hard budget alert before creating anything.

## I. API platform plan *(design — Phase 4, not built)*

Endpoints to add, keeping `/generate` and `/chat`: `GET /v1/models`, `POST /v1/completions`,
`POST /v1/chat/completions` (OpenAI-compatible request/response and SSE `data: [DONE]` framing, so
common SDKs work by changing `base_url`), plus key management (`/v1/keys` list/create/revoke/rotate) for
authenticated users.

Storage: SQLite first (single process, zero ops), same schema portable to Postgres.

```sql
users      (id INTEGER PK, email TEXT UNIQUE NOT NULL, created_at TEXT, disabled INTEGER DEFAULT 0);
api_keys   (id INTEGER PK, user_id INTEGER REFERENCES users, prefix TEXT NOT NULL,      -- first 8 chars, shown in lists
            key_sha256 TEXT UNIQUE NOT NULL, label TEXT, created_at TEXT, last_used_at TEXT,
            revoked_at TEXT, expires_at TEXT, rpm_limit INTEGER, tpm_limit INTEGER);
models     (id TEXT PK, checkpoint TEXT, tokenizer_fingerprint TEXT, context_length INTEGER, active INTEGER);
usage      (id INTEGER PK, ts TEXT, user_id INTEGER, key_id INTEGER, model_id TEXT, endpoint TEXT,
            prompt_tokens INTEGER, completion_tokens INTEGER, duration_ms INTEGER, status INTEGER, error_type TEXT);
```

Rate limits live in memory per process for now (token bucket already exists), with `rpm_limit` and a new
tokens-per-minute bucket read from `api_keys`; a shared store (Redis) only when more than one process serves.
Storage rules: keys as SHA-256 only (256-bit random, so no slow hash needed), plaintext shown once;
never log keys or request bodies; store no prompts, only token counts; the only personal datum is an email.
Users and key creation via CLI/admin first; self-serve signup only after abuse controls exist.
SDK strategy: OpenAI compatibility first; a thin Python client only if it adds something.

## J. Risk register

| # | Risk | Severity | Probability | Impact | Detection | Mitigation |
|---|---|---|---|---|---|---|
| 1 | Model is mistaken for a general assistant | High | High | Users get fluent nonsense outside stories; reputational | Out-of-domain prompt tests; stated in README/API docs | Say plainly what it is; system prompt; scope note in `/v1/models` |
| 2 | Instruction following stays weak | High | Medium | The "chat" product doesn't do what it says | 312-prompt eval with CIs per kind | v3 data; more pretraining tokens; report per-kind, not aggregate |
| 3 | Public exposure without TLS/accounts/quotas | High | Medium if rushed | Key theft, cost, abuse | Checklist in SECURITY.md | Don't expose before Phase 4–5 |
| 4 | Metric misleads (aggregate hid a length collapse) | Medium | Happened once | Wrong model chosen | Per-kind breakdown; regression gate | Always read per-kind; multiple held-out phrasings |
| 5 | Eval too small / one seed | Medium | High | False conclusions | CIs; kinds with n=24 | Larger eval; repeat seeds for decisions that matter |
| 6 | Data contamination | Medium | Low | Inflated scores | `audit_dataset.py`; content-hash split | Re-run on any new corpus; check against the full train split before publishing numbers |
| 7 | Training lost to sleep/thermals/OOM | Medium | Medium | Hours wasted | Trainer logs memory and warns | Checkpoints every 200–250 steps; exact resume tested |
| 8 | Irreproducible run (dirty git) | Low | Happened once | Can't recreate a result | `env.json` records dirty flag | Start runs from a clean tree |
| 9 | Model-lock wedge / DoS by slow clients | Medium | Fixed | Server unusable | Real-socket regression tests | Add per-request wall-clock timeout (open) |
| 10 | Licence/policy | Medium | Low | Legal | DATASETS.md | CDLA-Sharing-1.0 obligations; choose weight licence before release |
| 11 | Cost runaway on cloud | Medium | Low | Bill | Budget alert | Alert before first resource; CPU VM first |

## 11. Gap matrix against the brief's 29 sections

| § | Topic | Status |
|---|---|---|
| 1–2 | Audit, architecture map | **Done** (this file, ARCHITECTURE.md) |
| 3 | Monitor training | **Done**: per-25-step log of loss, LR, grad norm, tokens/s, RSS/Metal memory, system memory; health summary in D |
| 4 | Dataset audit | **Done**, with the stated 10%-sample limit |
| 5 | Tokenizer audit | **Done**: in-domain 3.96 chars/token, 0.79 words/token, 205 tokens/story; byte-level so no unknown token; round-trip lossless on accents, CJK, numbers, URLs, code, whitespace; special-token text in prompts stays inert. Weak spots: non-English costs ~3 tokens/char; numbers/code are 1.4–1.8 chars/token (irrelevant for this corpus) |
| 6 | Model audit | **Done** (C) |
| 7 | Training improvements | Evaluated in D; applied only what was measured to help |
| 8 | Mac constraints | **Done**; each optimisation classified in PERFORMANCE.md: bf16 (MPS, no gain), CUDA-only features not used |
| 9 | Checkpointing/recovery | **Done**: weights, optimizer, data position, RNG, config, tokenizer fingerprint; bit-exact resume tested |
| 10–11 | Evaluation, comparison | **Done** except the general-capability suites explained in F |
| 12 | Tests | 318 pytest tests + a smoke test (tokenizer, data, model, masking, loss, resume, generation, API, auth, rate limit, SFT/LoRA, quantisation). Gaps: CORS, `--no-auth` refusal, TPM limits (not built) |
| 13–14 | Inference + optimisation | **Done for `tiny`**; `small` benchmark pending |
| 15 | Model server | **Partial**: `/health`, `/model`, `/generate`, `/chat`; no `/v1/...` routes |
| 16–18 | API keys, usage, rate limits | **Partial**: hashed keys, request limiter, output cap, body limit, bounded queue. **Missing:** users, list/revoke/rotate API, usage table, TPM limit, per-request timeout, max-context check on input beyond truncation |
| 19 | API docs | **Done** (API.md) for what exists |
| 20–22 | Deployment, readiness | **Design only** (DEPLOYMENT.md, SECURITY.md) |
| 25–26 | Experiment tracking, docs | Experiment records are complete. Docs: README, ARCHITECTURE, DEPLOYMENT, API, SECURITY, AUDIT, RESULTS, PERFORMANCE, DATASETS exist; **TRAINING, EVALUATION, INFERENCE, EXPERIMENTS.md are folded into README/RESULTS/PERFORMANCE/COMPARISON and can be split out on request** |

## 12. Phased execution plan

| Phase | Work | Exit criterion |
|---|---|---|
| **Now** | exp_006 finishes (SFT v3, data fix only) | run completes, no divergence |
| **After training** | evaluate exp_006 vs exp_004 and exp_005 on the 312-prompt eval; per-kind table; keep whichever wins on evidence | decision recorded in COMPARISON.md; `docs/RESULTS.md` updated |
| **Improve model** | (a) if instruction following is still weak: more pretraining tokens (extend exp_004 on unused corpus) then re-SFT; (b) decoding-setting sweep vs loop rate; (c) repeat the winner with a second seed | topic/length/name pass rates with CIs; loop rate ≤ 5% |
| **Inference** | benchmark `small` (tok/s, TTFT, context); package int8 model in `models/`; add per-request timeout | numbers in PERFORMANCE.md; packaged model chats from a fresh clone |
| **API (Phase 3→4)** | `/v1/models`, `/v1/completions`, `/v1/chat/completions` with SSE; tests | OpenAI Python SDK works against it with only `base_url` changed |
| **Auth / platform (Phase 4)** | SQLite users/keys/usage/models; key create/list/revoke/rotate; TPM limit; usage recording (counts only); CLI admin | tests for hashing, revocation, limits, usage rows; no key or prompt in logs |
| **Deployment (Phase 5)** | container; one small CPU VM behind TLS proxy; health checks, metrics, budget alert; load test | 20 concurrent requests without 5xx; p95 latency documented |
| **Public launch** | only after the checklist in SECURITY.md is closed and the model card states its scope | written go/no-go |

Do not start billing, dashboards, Kubernetes or microservices before the model does what its card claims.
