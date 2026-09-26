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
| Wall time | 79 min | 6.5 min | 24.8 min |
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

* The model learned the chat template and stops replies itself with `<|end|>`. That is a change of
  mechanism, not of rate: replies end on their own 80–93% of the time, about the same as the base
  model's `<|endoftext|>` endings (77–93%).
* Named characters get **consistent genders** (the base model wrote "a little girl named Tom").
* Names never seen in training ("Zoe", "Ruby") work.

What did not:

| Requested name/topic word reused (25 prompts: 15 names, 10 topics) | greedy | sampled |
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

## The 14M model and request following (exp_004 – exp_006)

Everything in this section is reproducible from the repository: configs in `configs/`, per-run records in
`experiments/exp_00{4,5,6}_*/`, evaluation files in `evals/`, scripts named below.

### Pretraining the `small` model (exp_004)

14,162,304 parameters (8 layers, width 384, 6 query / 2 KV heads, context 512, vocabulary 4,096), 5,500 steps,
90.1M tokens (0.87 epoch of the 103M-token training split), 4.09 h on the M3.

| Held-out split (scored in full) | Loss ± std. error | Perplexity | Bits per byte | Top-1 |
|---|---|---|---|---|
| validation (999 windows) | **1.3948 ± 0.0065** | 4.03 | 0.509 | 0.637 |
| test (995 windows) | 1.4056 ± 0.0072 | 4.08 | 0.513 | 0.634 |

Bits per byte is the tokenizer-independent number: exp_001 (the 5M model, a different tokenizer) had
0.651 on its validation split, so the 14M model is about 22% better. Probes 94.5% (55 pairs, chance 50%), edge
cases 14/14. Training was healthy (no NaN, no loss spikes; one isolated gradient spike absorbed by clipping,
see [CORRECTNESS.md](CORRECTNESS.md), entry 4) and train and validation loss overlap throughout: no overfitting,
and both were still falling when it stopped.

### Instruction tuning: what the data taught, measured properly

Two SFT runs of exp_004 on 60,000 verifiable conversations each: **exp_005** (each length or dialogue request
had 2–3 phrasings) and **exp_006** (70–84 phrasings each; recipe otherwise identical). Request following is
scored by rules, not by a judge (`evaluation/instruct.py`): a reply passes if it satisfies every check, e.g. "contains
the requested name", "at most 90 words", "has dialogue".

Two prompt sets, both built from held-out (test-split) stories, 24 prompts for each of 13 request kinds:

* **held-out phrasings** (`evals/instruct_v3.jsonl`): request wording never used in training; for name requests,
  half use names that never occur in the corpus.
* **in-template control** (`evals/instruct_v3.intemplate.jsonl`): the same kinds, but the wording the model *was*
  trained on. The gap between the two sets is what separates "did not learn the task" from "learned only the
  wording it saw".

Every rate is reported next to **chance**: the pass rate of a real story that ignores the request (aggregate
10.6%; per kind below). Confidence intervals resample *distinct* prompts (190 distinct prompts in the held-out
set, 277 in the control), because kinds like "short story" repeat a few strings. Greedy decoding, pass rate %:

| Request kind | chance | base 004 | SFT 005 | SFT 006 | | base 004 | SFT 005 | SFT 006 |
|---|---|---|---|---|---|---|---|---|
| | | **held-out phrasings** | | | | **in-template control** | | |
| dialogue | 66.8 | 96 | 100 | 100 | | 62 | 75 | 100 |
| short | 1.4 | 71 | 0 | 0 | | 46 | 29 | **100** |
| long | 11.6 | 21 | 0 | **58** | | 12 | 21 | **75** |
| name | 3.9 | 46 | 50 | **67** | | 58 | 75 | 79 |
| name + topic | 0.7 | 33 | 38 | 42 | | 29 | 62 | 71 |
| topic | 8.8 | 62 | 42 | 46 | | 75 | 88 | 83 |
| place | 10.2 | 62 | 29 | 21 | | 54 | 33 | 21 |
| emotion | 19.3 | 46 | 50 | 46 | | 58 | 50 | 25 |
| **all scored kinds** | 10.6 | 40.6 | 28.8 | 37.5 | | 41.7 | 44.4 | **56.9** |
| 95% CI of the aggregate | | 30–51 | 20–38 | 27–48 | | 36–48 | 39–50 | 51–63 |

(The base model receives the request as plain text to continue; the SFT models get the chat template.)

Paired comparisons on identical prompts (`scripts/compare_instruct.py`, greedy; difference in points, 95% interval):

| A vs B | held-out phrasings | in-template control |
|---|---|---|
| SFT 006 vs SFT 005 | **+8.7 [+2.8, +15.4], p = 0.002** | **+12.5 [+6.3, +18.5]** |
| SFT 006 vs base 004 | −3.1 [−15.6, +8.7], p = 0.67 (no difference) | **+15.3 [+7.4, +22.8]** |

What this establishes:

* **The phrasing fix worked.** exp_006 beats exp_005 on unseen wording, mostly through length control (long 0 → 58%,
  p = 0.005) and names (+17 points, p = 0.025).
* **The model really learned to follow requests, for wording it saw.** On the in-template control it is far above
  chance and the base model: short 100%, dialogue 100%, long 75%, name 79%, name + topic 71%.
* **It does not generalise to unseen wording.** On held-out phrasings it is indistinguishable from the untuned
  base model overall (and worse on "short": 0% vs 71%; the replies are 91–142 words, i.e. its default length).
  A 14M model trained on ~80 phrasings per kind is learning a template family, not the concept of a request.
* **Content requests (topic, place, emotion) are its weak spot**, worse than the base model even in-template
  (place 21% vs 54%, emotion 25% vs 58%). The base model just continues the prompt text, which repeats the
  requested word; the SFT model opens most stories with the same generic "a little girl named Lily…". The training data
  explains part of it: a place request's story mentions the place late (median 19 words in) and `park` is 40% of
  the 5,114 place examples (`library` 14).
* **Sampling hurts control** (in-template aggregate 56.9% greedy vs 43.1% sampled at T = 0.7).

Losses: validation loss on the *full* split is 1.3190 ± 0.0079 (exp_005) and 1.3185 ± 0.0076 (exp_006), but they are
scored on different conversation sets, so this is **not** a comparison of the two models
([CORRECTNESS.md](CORRECTNESS.md), entry 1). Per-conversation and per-prompt outcomes are in
`experiments/exp_00{4,5,6}_*/eval/`.

Next: exp_007 tests data fixes aimed at the weak spots (more request wordings, requested words that the story
actually dwells on and introduces early, balanced sampling across rare places and topics, casual typing); it is
judged on a *fresh* held-out set so that iterating against the current one cannot flatter it.

Reproduce: `python scripts/make_eval_controls.py --eval evals/instruct_v3.jsonl --stories <TinyStories file>`,
then `python scripts/eval_instruct.py --checkpoint <run>/checkpoints/best.pt --instruct evals/instruct_v3.jsonl
evals/instruct_v3.intemplate.jsonl` and `python scripts/compare_instruct.py A.json B.json`.

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
