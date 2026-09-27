# Pipeline correctness log

Scientific problems found in the training and evaluation pipeline, how each was found, what was
changed, and what is still unmeasured. Newest first. The rule for this file: an entry says **fixed**
only if a test proves it, and says **measured** only if the number came from a run.

| # | Problem | Status |
|---|---|---|
| 9 | The OpenAI routes returned `finish_reason: "eot"`, which no OpenAI client knows | **fixed** (test) |
| 8 | Under load an unbatched stream got HTTP 200 and then an error event, instead of a clean 503 | **fixed** (test failed 5/5 before, passes after), found by a load test |
| 7 | The new training diagnostics crashed a run (exp_007, step ~625) | **fixed** (tests), found by a real run |
| 6 | Content requests (topic, place, emotion) are barely learnable from the data as generated | **fixed** by data changes (exp_007), at the cost of length control |
| 5 | One pass rate hid two different failures: not learning the task vs not generalising to new wording | **measured**; in-template control added |
| 4 | No layer-wise gradient health, spike/overfit detection or plots during training | **added** (tests), thresholds not yet validated on a real run |
| 3 | Instruction-following pass rates had no chance-level control, inflated sample sizes and no paired test | **fixed** (tests) |
| 2 | SFT gradient accumulation weighted micro-batches equally instead of by scored tokens | **fixed** (test), effect on real runs not yet measured |
| 1 | Validation loss depends on how many conversations are scored (1.2895 / 1.3276 / 1.3190 for one checkpoint; noise sd 0.031 at 160) | **fixed** (tests): the full split is scored by default, with a standard error |

## 9. `finish_reason: "eot"` on the OpenAI routes

The engine's own reasons include `eot` (the model wrote its end-of-text token). The OpenAI-compatible routes passed it through, so a client that
validates the field (`stop` or `length`) would reject a normal reply. Found while adding real token counts to the closing stream frame: the new test
asserted `stop` or `length` and saw `eot`. **Fix:** `eot`, stop sequences and cancellation read `stop`; a time-out reads `length`
(`tests/test_demo.py::test_finish_reasons_are_the_two_values_openai_clients_know`). The native routes keep the engine's own values.

## 8. Streams that failed after the 200

`scripts/loadtest.py` against 16 concurrent streaming users of an unbatched server reported three "faults": HTTP 200 streams that ended in an
error event, "too many requests waiting". The stream path checked admission (acquire the model slot, release it) before sending the 200 header,
then took the slot again inside the stream; under load the second check could hit the waiter limit after the header was sent, so a request
that had been admitted was refused with no status code to carry it. **Fix:** the second acquisition skips the waiter limit, because the request was
already admitted. `tests/test_loadtest.py::test_streams_are_refused_with_a_real_503_never_as_an_error_after_the_200` (24 concurrent streams, waiter
limit 2) failed on 5 of 5 runs before the fix and passes after. Overload now always shows as a clean 503 before the stream starts.

---

## 7. The diagnostics themselves crashed a training run

**Evidence.** exp_007 (SFT on the v4 data) stopped at step ~625 with
`TypeError: Trainer._anomaly() got multiple values for argument 'kind'` (`train.log`, 16:32:55). A real gradient
spike (max norm 8.74, one step clipped: ordinary, and handled by clipping) made the spike detector fire, and the
code that reports a finding passed the detector's dict, which contains a key named `kind`, as keyword arguments
alongside a positional parameter also named `kind`. The 13 unit tests of the diagnostics all passed, because they
exercised the detectors and the layer statistics but never a detector *firing inside the real training loop*.

**Why it is serious.** Entry 4 promises the diagnostics "observe, never intervene". A monitoring feature that can kill a
multi-hour run is worse than none. The run was recoverable because checkpoints are written every 200 steps and resume
is bit-exact: it restarted from step 600 and lost about eight minutes.

**Fix.** (1) The report call no longer forwards the colliding key, and `_anomaly` takes its name as a positional-only
argument so this class of collision cannot recur. (2) A safety net: recording a finding and measuring layer health are
wrapped so that a fault in the diagnostics is logged (`[warn] could not record diagnostic ...`) and training goes on.
(3) Three integration tests run real training loops: a forced loss/gradient spike is recorded (log line, metrics
record with value, median, threshold and z, count in `summary.json`) and the run completes; a fault injected into the
metrics writer and into the health measurement is logged and does not stop the run; an overfitting finding is recorded
without stopping it. `test_a_loss_spike_during_training_is_recorded_and_the_run_completes` was run against the previous
code and fails with the identical `TypeError`.

**Lesson kept.** A component whose contract is "never interferes" needs a test in which it fires inside the real loop,
not only a test of its parts. Also: detectors start empty after a resume, so a spike in the first 20 steps after a
resume is not flagged (by design, and now documented).

---

## 6. Content requests are barely learnable from the data as generated

**Evidence.** Even on wording it was trained on, exp_006 passes only 21% of place requests and 25% of emotion
requests, below the untuned model (54%, 58%). Reading its replies: it opens with the same generic "little girl
named Lily" story regardless of the requested topic or place. In the training data the requested word appears
in the story 99.9% of the time, but the first mention is late (place: median 19 words, 90th percentile 68), and
the distribution is heavily skewed (`park` is 40% of 5,114 place examples; `library` 14, `mountain` 21).
The "topic" of a story is its most frequent noun even if it occurs once.

**Added (opt-in; the default output is byte-identical, pinned by a hash test).** `make_conversations` /
`scripts/make_instruct_data.py`: `--templates v4` (23 request openers instead of 8, disjoint from the eval
wording), `--min-mentions N` (the requested word must occur at least N times), `--first-mention-max W` (and first
within W words), `--balance-alpha a` (sample values with probability ∝ count^a, flattening rare places/topics/
names), `--augment p` (casual typing: lowercase, no final punctuation; names keep their capital).

**Measured (exp_007, all four options together; see [RESULTS.md](RESULTS.md)).** Content requests improved sharply and the
gain holds on the pre-registered final set (topic 62 → 88%, emotion 71 → 83%, place 17 → 62%; in-template place 21 → 79%,
emotion 25 → 96%), but length control was lost (long 75 → 0%), and the overall gain is not statistically established
(+5.2 [−5.4, +15.3] on the final set). The options were applied together, so the individual contributions are not separated;
exp_008 tested the length hypothesis and rejected it (length requests sampled 3x as often did not restore greedy length control); exp_009 added a two-topic fix. The recipes v4, v5 and v6 are statistically indistinguishable on the fresh final set, i.e. the data recipe has reached the noise floor of the evaluation (RESULTS.md).

---

## 5. One pass rate hid two different failures

**Evidence.** exp_006 scored 37.5% on held-out phrasings, no better than the untuned model (40.6%): "the SFT did
not work". The in-template control (same kinds and held-out stories, but the training wording) shows 56.9% vs
41.7% (paired +15.3 points, 95% interval +7.4 to +22.8): the model *did* learn to follow requests, for the
wording it had seen. The two questions have different answers and different fixes (more phrasing diversity vs a
bigger model or more data), and only the pair of numbers distinguishes them. Full tables are in
[RESULTS.md](RESULTS.md).

**Fix.** `scripts/make_eval_controls.py` builds the control from the training templates and the held-out
stories; `make_eval_prompts(..., templates=...)` makes any phrasing family. Both sets are reported together.

**Caveat and pre-registration.** The held-out set has been looked at while designing fixes, so a gain measured on
it could be partly tuning to it. `evals/instruct_v4_final.jsonl` (312 prompts, 13 kinds x 24) is the confirmation
set: new request wordings (`EVAL_TEMPLATES_FINAL`: openers "Craft", "Spin", "Got", "Dream up", "Narrate", "Whip up";
none appears in any training or earlier eval wording, enforced by a test) and different held-out stories
(`--final-seed 1`). It was committed **before any model was scored on it**; the commit that added it is the
proof. Use it once per candidate model, report every model that was tried on it, and do not tune against it.

---

## 4. Training diagnostics: gradient health, anomalies, overfitting, plots

**Gap.** A run logged loss, learning rate and the largest gradient norm per 25 steps. Nothing told you
*which* layers were exploding or barely learning, a loss spike was noticed only by reading the log, and
"train loss falling while validation rises" needed a person to spot it.

**Added (`train/health.py`, `train/plots.py`, `scripts/plot_run.py`). Observe only:** nothing changes a
gradient, weight, RNG or the data order. `test_diagnostics_do_not_change_training` trains twice with
different measurement schedules and requires bit-identical final weights.

| Diagnostic | How | Recorded as |
|---|---|---|
| Layer gradient health | per group (`blocks.N.attn`, `.mlp`, `.norm`, embeddings): L2 gradient norm and weight norm, taken **before** clipping, every `log_interval` steps; flags a group > 100× or < 0.01× the median group, or non-finite | `health` records; `[anomaly] gradient_health` |
| Loss spike | a step loss above rolling median + 6 × robust sigma (1.4826·MAD over 50 steps, floor 2% of the median) | `anomaly` record + log line |
| Gradient-norm spike | same, z = 8, floor 10% (gradient norms are heavier-tailed) | same |
| Overfitting | validation up **and** training down at each of 2 consecutive evaluations | same |
| Plots | `python scripts/plot_run.py <run>`: train/val loss, val perplexity, LR, max grad norm vs the clip line, anomaly markers, per-layer gradient norms | `training_curves.png` |

Anomaly counts are also written to `summary.json`. The run still stops only for a non-finite loss or five
non-finite gradient steps in a row, exactly as before.

**Why the clip threshold stays at 1.0 (evidence, not a guess).** Per-window maximum gradient norms:
exp_001 median 0.62, exp_004 median 0.48, exp_006 median 0.80; after the first 20% of steps the 95th
percentile is 0.69 / 0.56 / 0.84. So 1.0 sits about 1.2–1.8× above the steady state: it acts as a safety
net, not as a constant brake. Clipped steps: 46/3000 (exp_001), 117/5500 (exp_004), 8/1800 (exp_006), almost
all during warmup. A lower threshold would clip routinely and slow learning; a much higher one would stop
protecting against bad batches.

**A real event the plot exposed.** exp_004 has one isolated spike in the window ending at step 5100: max
gradient norm 3.92 (about 8× the steady 0.48), one step clipped. The window's training loss did not move
(1.424 vs 1.427 before) and the next windows were normal, so clipping absorbed a bad batch as intended. The
new detectors would flag it.

**Also visible in exp_004's curves:** training and validation loss overlap for the whole run (no
overfitting; the run ended after 0.87 epoch) and both are still falling slowly.

**Not yet verified.** The detector thresholds were checked on synthetic noise (no false alarms in 3,000
steps, the injected 1.6× jump is caught). Whether they are quiet on a real run is unknown until one completes with
them on; `summary.json → anomalies` will show it. The first real run that used them (exp_007) exposed a crash in
the reporting path, see entry 7.

---

## 3. Instruction-following pass rates had no baseline, inflated sample sizes and no paired test

**Evidence.** For each of the 312 prompts in `evals/instruct_v3.jsonl`, every held-out story was checked against
the prompt's rules, i.e. what a model that ignores the request would score. Chance-level pass rate by kind:
name 3.9%, topic 8.8%, name+topic 0.7%, two topics 1.8%, place 10.2%, emotion 19.3%, name+emotion 1.6%, short
1.4%, long 11.6%, short+topic 0.2%, long+topic 0.9%, **dialogue 66.8%**, generic 100% (no rule, excluded). Over
the scored kinds the average is **10.6%**. Also, the kinds with no story-dependent slot (short, long, dialogue,
generic) are only 6, 6, 4 and 4 distinct strings repeated 24 times each: under greedy decoding a repeat is the
same test, so "n = 24" was really 4–6, and the overall n = 312 was 190 distinct prompts. Finally, per-prompt
outcomes were not saved, so no paired test was possible.

**Why it mattered.** The un-tuned 14M model scored 23% on the earlier set; that is meaningful only against ~10%
chance, and a kind like dialogue is 67% "correct" for free.

**Fix (`evaluation/instruct.py`, `paired.py`, `stats.py`; `scripts/make_eval_controls.py`, `eval_instruct.py`,
`compare_instruct.py`).** Results now carry, per kind and overall: `chance` and `lift`, `n_distinct`, a
`ci95_clustered` that resamples distinct prompts, and `per_prompt` outcomes. `paired_compare` runs two
checkpoints through identical prompts with a cluster bootstrap on the difference. Tests:
`test_cluster_interval_is_wider_when_draws_are_duplicates_of_few_prompts`, `test_paired_difference_detects_a_real_gap_and_not_a_null_one`,
`test_chance_levels_measure_what_a_request_ignoring_story_scores`, and the instruction-eval structure test.

---

## 2. Gradient accumulation was not token-weighted under SFT masking

**Problem.** `Trainer.run` computed `loss / grad_accum` for every micro-batch, where `loss` is a mean over
that micro-batch's *scored* targets. In SFT the prompt is masked, so micro-batches have different numbers
of scored tokens. Averaging the means gives a micro-batch with 4 scored tokens the same influence as one
with 30. The evaluator (`evaluation/loss.py`) is token-weighted, so training optimised a slightly different
objective from the one reported. Pretraining is unaffected: every token is scored, counts are equal.

**Evidence.** `tests/test_train.py::test_accumulation_equals_one_big_batch_when_masks_differ`: with scored
counts 30 / 4 / 17, the old gradient differs from the gradient of one big batch; the new one matches it
(`allclose`, atol 1e-6) and its loss equals the big-batch loss.

**Fix.** `Trainer._accumulate` fetches the step's micro-batches, counts scored targets, and weights each
micro-batch mean by `n_i / total`. When all counts are equal it uses exactly `loss / grad_accum`, so
pretraining numerics are bit-identical to before (`test_accumulation_is_unchanged_without_masking`). A step
whose every target is masked now raises instead of producing NaN.

**Not yet measured.** How much this changes a real SFT run. exp_006 was trained with the old averaging.
The fixed code will be rerun (same data and seed) and compared; the difference is expected to be small
because most micro-batches have similar counts, and it will be reported either way.

---

## 1. Validation loss depends on which conversations are scored

**Evidence (measured on exp_005 `best.pt`, `data/processed/instruct_v2_sft`, script in the session
scratchpad; per-conversation loss sums, token-weighted).** The same checkpoint has three different
"validation loss" numbers depending on how many conversations are scored:

| Scored | Where it comes from | Loss |
|---|---|---|
| 160 evenly spaced conversations | the trainer's own eval (`eval_iters` 40 × `eval_batch_size` 4) | 1.2895 |
| 1,008 evenly spaced conversations | `scripts/evaluate.py` (`windows` in its result file) | 1.3276 |
| **all 2,423** | the truth for this split | **1.3190** |

Sampling noise of a random subset (500 draws): 160 conversations → sd **0.031** (5–95% range 1.265–1.367);
400 → sd 0.018; bootstrap standard error of the full-split loss → **0.008**. The trainer's evenly spaced
160 happens to sit 0.030 *below* the full value. Test split, all 2,356 conversations: 1.3294.

Weighting also matters: an equal-weight mean of per-conversation losses is 1.2908, against 1.3190
token-weighted (assistant tokens per conversation: mean 188, sd 50). Every loss quoted in this repository is
token-weighted.

**Consequences.**
* A validation loss is only comparable *within* one run, where the subset is fixed and the comparison is
  paired. Differences of ~0.01 between checkpoints of one run are meaningful; the absolute number is not.
* Across runs with different SFT data (exp_005 vs exp_006), validation sets are different conversations, so
  their losses **must not be compared**. The trainer's reported best loss is also selected on the same subset.
* `best.pt` is chosen on 160 conversations. With subset noise of 0.03 that choice between neighbouring
  checkpoints is close to arbitrary late in training, when the true differences are ~0.005.

**Fix.** `scripts/evaluate.py` now scores the whole validation and test splits by default
(`--loss-windows N` restores a subset) and reports a bootstrap standard error (`evaluation/loss.py`:
`heldout_rows` in one forward pass, `summarize_rows`); results say whether the split was scored `complete`, and
the regression gate refuses to compare a subset with a full-split result. Tests require the new function to
agree exactly with the old loss and accuracy functions, and its standard error to match the analytic value for
equal-weight rows. Measured with it: exp_004 validation 1.3948 ± 0.0065 (999 windows), test 1.4056 ± 0.0072;
exp_005 1.3190 ± 0.0079; exp_006 1.3185 ± 0.0076. The trainer's 160-conversation subset is unchanged (speed).
