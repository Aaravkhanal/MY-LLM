# Performance and memory (MacBook Air M3, 8 GB, fanless)

All numbers below were measured on this machine with `scripts/benchmark.py` (synthetic tokens, each
configuration in its own process) or taken from the real training runs. Model: the 4.95M-parameter
`tiny` preset, sequence length 256.

## Training throughput (idle machine)

| micro-batch | precision | ms / step | tokens / s | Metal memory | note |
|---|---|---|---|---|---|
| 8  | fp32 | 111 | **18,449** | 1.20 GB | |
| 16 | fp32 | 222 | **18,474** | 1.25 GB | **default** (16 × 4 accumulation) |
| 32 | fp32 | 458 | 17,880 | 2.35 GB | 2× the memory, no faster |
| 16 | bf16 | 232 | 17,647 | 1.25 GB | slower than fp32 |
| 32 | bf16 | 466 | 17,590 | 2.35 GB | |

* Achieved ≈ 0.63 TFLOPS (fp32). The GPU is far from saturated by a model this small; larger
  micro-batches don't help, they only cost memory.
* **bf16 autocast is *not* faster here** and saves no memory, so the default stays fp32. (An earlier
  short run suggested +8% for bf16; that was contention noise from other work on the machine. Always
  benchmark on an idle machine.)
* Gradient accumulation is exact: micro-batches of 32×2, 16×4 and 8×8 produced identical losses to
  four decimals, so the micro-batch size is purely a memory knob.
* The real 3,000-step run sustained 9–10k tokens/s while other work shared the machine and
  **14.6k tokens/s** once it was alone. The benchmark's 18k is the ceiling.

Whole-run figures (`experiments/exp_001_tiny_tinystories`): 49.2M tokens in 79 minutes; peak process
RSS 225 MB; peak Metal memory 1.26 GB.

## Inference (batch 1, KV cache)

| device | decode tok/s | decode without cache | prefill (256 tok) |
|---|---|---|---|
| **CPU** | **732** | 123 | 17,061 tok/s |
| MPS (GPU) | 148 | 30 | 5,274 tok/s |

* For a model this small the **CPU decodes ~5× faster than the GPU**: each generated token launches
  many tiny kernels and the GPU cannot amortise that overhead. `InferenceEngine` therefore picks the
  CPU for `device="auto"` when the model has < 50M parameters (`--device mps` overrides). Training
  is the opposite: large batches favour the GPU.
* The KV cache is ~6× faster than recomputing the context, on both devices. The cache for a full
  256-token context is 1.6 MB.

## Memory notes

* macOS plus ordinary apps already used ~70–77% of RAM before training started (1.9–2.3 GB
  available). The trainer logs system-available memory and warns below 700 MB. The `tiny` model
  leaves that headroom untouched (1.3 GB of Metal memory); a 32-sequence micro-batch would not.
* **Variable-length batches inflate Metal memory.** Padding SFT batches to ~250 different widths made
  the allocator hold **3.45 GB**; bucketing widths to multiples of 32 brought it to **1.26 GB** with
  identical loss and the same speed (A/B, 60 steps each).
* Checkpoints: model-only 19.8 MB fp32 / **5.06 MB int8**; a resumable training checkpoint (weights +
  Adam state) is 59.5 MB. Rolling checkpoints keep the last 2.

## int8 export

Per-row symmetric int8 of every weight matrix (`scripts/export_int8.py`): **3.92× smaller
(19.83 → 5.06 MB) for +0.01% validation loss** (perplexity 5.146 → 5.147; worst per-tensor relative
error 0.0077). MPS/PyTorch have no int8 matmul, so weights are dequantised at load: the gain is file
size and load time, not speed or runtime memory. Real int8 inference needs a kernel backend (MLX,
llama.cpp-style runtimes) and is worth doing only when serving cost matters.

## Thermal behaviour

The Air is fanless. The same SFT configuration measured 545, 574, 724 and 745 ms/step across
back-to-back runs as the chassis warmed up (≈ 35% spread), and pretraining slowed while other work
ran. Treat single measurements as ±30%, checkpoint often (the trainer does), and prefer several
short runs over one long unattended one when the machine is hot.

## MLX

Not adopted. Every component already runs on PyTorch/MPS, throughput is limited by model size rather
than the framework, and the one place MLX could matter — small-batch decode — is already fast on the
CPU. Reconsider MLX for (a) quantised inference kernels, or (b) training a model large enough that
MPS overheads stop dominating.

## Batched serving (`--batch-size N`)

By default the server generates one reply at a time. With `--batch-size N` concurrent requests are decoded together, one model
call per step for up to N users (`inference/batching.py`, `inference/scheduler.py`). It is exact: prompts are left-padded to one
length and the padding is masked, which changes nothing for a RoPE model because attention depends only on relative positions
(`tests/test_batched_model.py` compares logits row by row; `tests/test_batching.py` compares whole generations, greedy and seeded
sampling, stop sequences and per-request limits, against running each request alone, whoever else is in the batch). A learned-
position model, or a request whose prompt plus output does not fit the context, is served alone instead.

Measured on the 14M model (exp_009), CPU, **on an otherwise idle machine** (M3 Air; the web server for the demo was the only other process),
best of 3 (`scripts/benchmark_batching.py --requests 16 --new-tokens 120 --batch-sizes 1 2 4 8 16 --repeat 3`). Every batched output was
checked identical to the solo output.

| Batch | Tokens/s, 16 requests × 120 tokens | Speedup | One wave takes | Identical to solo |
|---|---|---|---|---|
| one at a time | 397 | 1.0× | 0.32 s per reply | – |
| 1 (through the batching code) | 375 | 0.94× | 0.32 s | yes |
| 2 | 345 | 0.87× | 0.70 s | yes |
| 4 | 590 | 1.49× | 0.81 s | yes |
| 8 | 897 | 2.26× | 1.07 s | yes |
| 16 | 1,310 | 3.3× | 1.47 s | yes |

Waves of one or two are **slower** than not batching (padding, masks and the scheduler cost a little and two rows do not yet amortise
them); the gain starts at four and grows with the wave. An earlier measurement taken while a training run was using the CPU showed
2.5× at 16; the idle machine shows 3.3×.

**Through the HTTP API** (`scripts/loadtest.py --stream --max-tokens 120`, a throwaway server on localhost, one key, 32 requests per
level), the gain is smaller than the raw decode benchmark, because each token also goes through per-row text decoding and server-sent
events in Python (not profiled further):

| Concurrent users | Unbatched: tokens/s, median reply, time to first token | `--batch-size 8`: tokens/s, median reply, time to first token |
|---|---|---|
| 1 | 374, 0.32 s, 0.008 s | 307, 0.39 s, 0.033 s |
| 4 | 374, 1.28 s, 0.97 s | 371, 1.27 s, 0.04 s |
| 8 | 374, 2.49 s, 2.18 s | **530 (+42%), 1.77 s (−29%), 0.04 s** |

At 16 simultaneous users (64 requests, `--batch-size 16 --batch-wait-ms 30`) batching served **all 64** at 706 tokens/s (median reply
2.9 s, first token in 0.07 s). The unbatched server, whose queue holds 8 waiters, answered 9 of the 64 and refused the rest with
503 (46) or 429 (6, the per-key burst limit of 10 requests at once), so its throughput figure is not comparable. What batching buys is
therefore mostly **capacity and first-token latency at 8 or more simultaneous users**, not raw speed for one user (about 20% slower
for a lone user). At 4 users it adds no throughput but users see their first token immediately instead of waiting their turn.

This benchmark found and fixed a real bug: an unbatched *stream* under load could receive a 200 header followed by an error event
("too many requests waiting"), because the admission check ran twice and the second, after the 200, could refuse it. Overload now
always shows as a clean 503 before the stream starts (`tests/test_loadtest.py`: failed 5 of 5 runs before the fix, passes after).

Per decode step, one thread (`scripts/benchmark_batching.py` is the repeatable version): the forward pass costs 5.6 ms for 1 row and
38 ms for 16, so each row is about **2.1× cheaper batched** (5.97 → 2.82 ms per row including sampling; this step measurement was taken
under contention). Sampling is done per row in Python (0.4 ms per row per step) and is the next thing to vectorise.

**Recommendation:** leave it off (the default) for a demo with one or two users; use `--batch-size 8` when you expect eight or more
people at once. Each wave costs a few milliseconds of waiting (`--batch-wait-ms`, default 15) when users do not arrive together.

### Tried and rejected: half-precision weights for decoding

Where the time goes (profiler, batch 1, 14M model, CPU): 55% is the matrix multiplies of the linear layers (about 3.1 ms of a
5.6 ms step), the rest is per-operation overhead. Hypothesis: single-user decoding is memory-bandwidth-bound (54 MB of fp32 weights
per token), so 16-bit weights should be up to ~1.7x faster. **Not supported:** batch 1 did not speed up (fp32 5.4 ms, fp16 5.8 ms,
bf16 6.8 ms at one thread; 7.9 / 7.7 / 7.5 ms at four), and the logits drifted (max 0.017 fp16, 0.16 bf16). Only larger batches
gained (batch 8, four threads: 38.8 ms fp32 vs 15.2 ms fp16), which batching already provides. Measured on a machine that was training,
so the ratios are indicative; not adopted.
