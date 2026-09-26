# Training

Everything runs on one MacBook Air M3 (8 GB, fanless) through PyTorch/MPS. Design rationale: [ARCHITECTURE.md](ARCHITECTURE.md);
measured speeds and memory: [PERFORMANCE.md](PERFORMANCE.md); results: [RESULTS.md](RESULTS.md).

## The pipeline

```bash
python scripts/prepare_data.py --config configs/small_tinystories.yaml   # clean -> dedup -> split -> tokenizer (train split only) -> shards + manifest
python scripts/train.py        --config configs/small_tinystories.yaml   # Ctrl-C is safe: it finishes the step and saves
python scripts/train.py        --resume experiments/exp_004_small_tinystories
python scripts/plot_run.py     experiments/exp_004_small_tinystories     # loss / perplexity / LR / gradient norm / per-layer health
python scripts/evaluate.py     --checkpoint experiments/exp_004_*/checkpoints/best.pt
```

Any config value can be overridden on the command line (`train.lr=5e-4 train.max_steps=500`). Each run writes
`experiments/exp_NNN_name/` with its resolved `config.yaml`, `env.json` (torch, python, device, git commit and whether
the tree was dirty), `metrics.jsonl`, `train.log`, `summary.json` and `checkpoints/`.

## Recipes that were run

| | exp_001 `tiny` | exp_004 `small` | exp_005–007 SFT (on exp_004) |
|---|---|---|---|
| Parameters / context / vocab | 4.95M / 256 / 2,048 | 14.16M / 512 / 4,096 | same as exp_004 |
| Data | 15.8M train tokens | 103.0M train tokens | 60k conversations, 12.5M tokens, assistant-only loss |
| Optimiser | AdamW (0.9, 0.95), weight decay 0.1 on matrices only | same | AdamW, weight decay 0.01 |
| LR schedule | warmup 100 → 1e-3, cosine to 10% | warmup 200 → 8e-4, cosine to 10% | warmup 60 → 2e-4, cosine to 10% |
| Batch | 16 × 4 accumulation × 256 tokens | 1 × 32 × 512 tokens (16,384 tokens/step) | 2 × 16 conversations (padded widths bucketed to multiples of 32) |
| Steps / tokens | 3,000 / 49.2M | 5,500 / 90.1M | 1,800 (≈ 0.96 epoch) |
| Clipping | global norm 1.0 | 1.0 | 1.0 |
| Precision | fp32 (bf16 measured no faster on MPS) | fp32 | fp32 |
| Wall time | 79 min | 4.09 h | 30–50 min (about 1.0–1.7 s/step, depending on other load) |

**Why these choices.** Micro-batch 1 with accumulation is fastest on this machine (6.6k tok/s at 1 vs 4.4k at 3; larger
batches push macOS into compressing memory). The clip threshold 1.0 sits above the steady-state gradient norm (95th
percentile 0.56–0.84 across runs) and only bites during warmup (46/3000, 117/5500 and 8/1800 steps): a safety net, not a
brake ([CORRECTNESS.md](CORRECTNESS.md), entry 4). Gradient accumulation is **token-weighted** under the SFT loss mask
(entry 2) and exactly `loss / grad_accum` when nothing is masked.

## Continued pretraining and fine-tuning

`init_from: <checkpoint>` starts from another run's weights (architecture and tokenizer fingerprint must match). For
`stage: sft` it is required; for `stage: pretrain` it is continued pretraining. Only the weights carry over: a fresh
optimiser, LR schedule (it warms up again) and data order. Optional LoRA (`train.lora_rank > 0`) trains adapters and merges
them on export.

## Stability and recovery

* Non-finite loss stops the run and leaves the last good checkpoint untouched; a non-finite gradient skips the update, and
  five in a row stop the run. Checkpoints refuse to store NaN weights and are written atomically (temp file, fsync, rename).
* A rolling `step_*.pt` (weights, optimiser, data position, RNG, counters) every `ckpt_interval` steps, plus `best.pt`
  (lowest validation loss) and `final.pt`. Resuming reproduces the uninterrupted run **bit for bit** (tested, including
  after Ctrl-C and for SFT/LoRA).
* Observe-only diagnostics (`train/health.py`): per-layer gradient and weight norms before clipping, robust loss and
  gradient-norm spike detection, an overfitting check across consecutive evaluations. They never change training
  (bit-identical weights, tested) and a fault inside them is logged, not raised (CORRECTNESS.md entry 7).

## Validation during training

The trainer evaluates on a fixed subset (`eval_iters` × `eval_batch_size` windows or conversations), which is fast but noisy
under SFT (sd ≈ 0.03 at 160 conversations). Use it to watch a run and to pick `best.pt`; quote losses from
`scripts/evaluate.py`, which scores the whole split with a standard error.

## Hardware notes (8 GB fanless M3)

Keep other memory-hungry apps closed (the trainer warns below 700 MB free). Run evaluation on CPU (`--device cpu`) while a job
trains on the GPU, but expect both to slow by ~40%. Step time varies ±30% with chassis temperature. Long runs should be
wrapped in `caffeinate -i` so the Mac does not idle-sleep.
