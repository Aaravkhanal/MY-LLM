# Results

Everything here was produced by the code in this repository on a MacBook Air M3 (8 GB). Raw records
live in `experiments/exp_00N_*/` (config, environment, every logged metric, evaluation results);
`python scripts/compare.py` regenerates `experiments/COMPARISON.md`.

## Experiments

| | exp_001 | exp_002 | exp_003 |
|---|---|---|---|
| **What** | pretrain from scratch | SFT (600 steps) from exp_001 | SFT (2,000 steps) from exp_001 |
| Data | TinyStories 60 MB slice, 15.8M train tokens | 19.8k story-request conversations | same |
| Steps / tokens | 3,000 / 49.2M (3.1 epochs) | 600 (≈1 epoch) | 2,000 (≈3.2 epochs) |
| Wall time | 79 min | 6.5 min | 27 min |
| Val loss / perplexity | **1.642 / 5.17** (7.66 at init) | 1.421 / 4.14 (chat format, assistant tokens) | 1.441 / 4.22 |
| Best step | 3,000 (still improving) | 600 | **1,750** (then overfits) |
| Next-token top-1 / top-5 | 59.4% / 84.1% | 63.6% | – |
| Probes (55 pairs, chance 50%) | **92.7%** | 87.3% | 87.3% |
| Edge cases | 14/14 | 14/14 | 14/14 |

The SFT loss values are not comparable to pretraining's: they are measured only on assistant tokens
of templated story requests, which are easier to predict than raw stories.

## Pretraining (exp_001)

* Validation loss fell at every one of 12 evaluations (2.66 at step 250 → 1.64 at step 3,000), with
  no divergence between training loss (1.54) and validation loss (1.64) after 3 epochs. It was still
  improving, so more steps or more data would help. Test loss (1.627) agrees with validation.
* Bits per byte: 0.651 (val), 0.640 (test) — the tokenizer-independent number to compare against
  other models.
* Generation quality (40 fixed prompts; full data in `experiments/exp_001…/eval/best_results.json`):

| | greedy | sampled (T=0.8, top-k 50, top-p 0.95) |
|---|---|---|
| Sentences correctly capitalised | 97.8% | 97.3% |
| Non-word rate | 0.0% | 0.1% |
| Samples containing a degenerate loop | **12.5%** | 0% |
| distinct-2 (repetition; higher is more varied) | 0.37 | 0.54 |
| Stories ending on their own | 80% | 80% |
| Prompt character reused (20 name prompts) | 100% | 100% |
| Mean length | 139 tokens | 140 tokens |

* Probes: grammar 26/26, story logic 10/10, common sense 15/19. The four misses: "Ice is very
  **cold**" (chose *hot*), "We smell with our **nose**", "Snow is **white**", "An ant is very
  **small**".

Typical output (greedy, prompt in bold):

> **Once upon a time, there was a little girl named Mia.** Mia loved to play with her toys. One day, she found a big box in her room. She was very happy and wanted to see what was inside. Mia opened the box and found a big, soft pillow. […] But when she woke up, she saw that the pillow was not a pillow.

Fluent and structurally right; the plot logic breaks down at the end, which is what this size does.

## Fine-tuning (exp_002, exp_003)

What worked:

* The model learned the chat template, and replies stop by themselves at `<|end|>` (87–93% of
  replies vs 77–93% for the base model's `<|endoftext|>`, depending on decoding).
* Named characters get **consistent genders** (the base model wrote "a little girl named Tom").
* Names never seen in training ("Zoe", "Ruby") work.

What did not:

| Requested-word reuse, 25 prompts (20 name/topic-word checks + …) | greedy | sampled |
|---|---|---|
| Base model, request as plain text | 0.68 | 0.84 |
| SFT, 600 steps | 0.72 | 0.76 |
| SFT, 2,000 steps | 0.88 | 0.84 |

* Asking for a story **with a dragon** often returns a story about a boy. Reuse of the requested
  topic word is not distinguishable from the base model given the same text (65% vs 68% over 60
  samples in an ad-hoc check; 0/3 for "robot", "park", "cake").
* With 25 prompts, a 5-prompt difference (0.68 → 0.88) is **not statistically meaningful**. Only the
  greedy improvement for exp_003 is suggestive, and it does not appear when sampling.
* Longer SFT overfits after ~1,750 steps (val loss 1.470 → 1.477) and costs common-sense probes
  (92.7% → 87.3%): a little forgetting.
* Why it is weak: the requests are templated, the "topic" label is the first matching keyword in the
  story's opening, and a 5M-parameter model has little capacity to condition 150 tokens of story on a
  short request. The packaged SFT model is the 600-step one (lower loss, less forgetting).

## Efficiency

See PERFORMANCE.md. Headlines: 18k tokens/s training in 1.3 GB of GPU memory; bf16 no faster than
fp32; CPU decodes 5× faster than GPU for this model (732 tok/s); int8 export is 3.9× smaller for
+0.01% loss.

## What broke along the way (and was fixed)

Kept here because the process is part of the result.

* A probe set built without checking tokenisation compared `" umbrella"` (6 tokens) with `" oven"`
  (2 tokens). A test now requires every probe choice to be a single token, and the set was rebuilt
  (55 fair pairs).
* PyYAML reads `5e-4` as a *string*; a learning rate would have been silently wrong. Configs are now
  strictly type-coerced.
* Paths in configs resolved against the current directory; they now resolve against the repo root.
* An independent review found nine real defects, including: the web UI's Stop button left the model
  locked (sync generators are not closed on disconnect); repetition penalty suppressed the stop
  token; chunked uploads bypassed the size limit; the regression check compared results from
  different datasets and tokenizers. All fixed, each with a regression test — several against a real
  server on a real socket, since the in-process test client cannot disconnect mid-stream.
* Chat models finished with `<|end|>`, which the "natural ending" metric did not count: it reported
  0% for a model that ended every reply. Metric fixed (`natural_end_rate`).

## Limits of this evaluation

* No judge model is used (nothing external is called), so coherence is measured by proxies:
  probe accuracy, loop and repetition rates, entity reuse, non-word rate. They catch regressions and
  gross failures; they do not measure whether a story is *good*.
* The probe set is small (55 pairs). The evaluation prompt sets are 40 and 30 prompts. Differences
  of a few points are within noise.
* One training run per configuration: no seed variance is measured.
* In-domain only. Nothing here says how the model behaves outside children's stories (it isn't
  expected to do well).

## Next steps, in the order I would take them

1. Train the `small` preset (14M parameters, vocab 4,096, context 512) on more of TinyStories — the
   pretraining run was still improving.
2. Build a larger and more varied SFT set (real topic diversity, multi-turn), and grow the
   instruction-following eval set well past 25 checkable prompts.
3. Add a bootstrap confidence interval to the evaluation report so small differences are labelled as
   noise automatically.
4. Only then consider preference tuning or a larger model on rented GPU time.
