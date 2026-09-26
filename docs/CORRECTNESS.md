# Pipeline correctness log

Scientific problems found in the training and evaluation pipeline, how each was found, what was
changed, and what is still unmeasured. Newest first. The rule for this file: an entry says **fixed**
only if a test proves it, and says **measured** only if the number came from a run.

| # | Problem | Status |
|---|---|---|
| 3 | Instruction-following pass rates are reported without a chance-level control | **found**, fix planned (step 1) |
| 2 | SFT gradient accumulation weighted micro-batches equally instead of by scored tokens | **fixed** (test), effect on real runs not yet measured |
| 1 | Two "validation loss" numbers for one checkpoint differ by 0.04 (subset choice) | **found**, measurement in progress, fix planned |

---

## 3. Chance-level control is missing from the instruction-following eval

**Evidence.** For each of the 312 prompts in `evals/instruct_v3.jsonl`, 200 random real held-out
stories were checked against the prompt's rules, i.e. what a model that ignores the request would score.
Chance-level pass rate by kind: name 3.7%, topic 8.6%, name+topic 0.5%, two topics 1.6%, place 10.3%,
emotion 18.6%, name+emotion 2.0%, short 1.4%, long 11.8%, short+topic 0.2%, long+topic 0.8%,
**dialogue 66.4%**, generic 100% (no rule, so it is excluded from the scored rate). Over the scored kinds
the average is about **10.5%**.

**Why it matters.** The un-tuned 14M model scored 23% (greedy) and the first SFT run 27% on the earlier
208-prompt set. Those numbers only mean something relative to about 10% chance, and a kind such as
dialogue is 66% "correct" for free. A raw pass rate can therefore hide a model that is doing nothing.

**Planned fix (step 1).** Report each kind's chance level next to its pass rate, and lift over chance;
save per-prompt outcomes so two checkpoints can be compared on the same prompts with a paired test;
tighten the dialogue rule.

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

**Evidence.** exp_005's own log reports SFT validation loss **1.2895** at the end of training;
`scripts/evaluate.py` printed **1.3276** for the same checkpoint. The trainer scores 160 evenly spaced
conversations (`eval_iters` 40 × `eval_batch_size` 4); `evaluate.py` scores 400; the split holds 2,423.

**Consequences.** (a) A validation loss is only comparable within a run, because the subset is fixed.
(b) Across runs with different SFT data (exp_005 vs exp_006) validation sets are different conversations,
so their losses **must not be compared**. (c) Picking `best.pt` on a small subset is noisy.

**Status.** The size of the subset noise is being measured on the full split (bootstrap over conversations)
and will be filled in here. Planned fix: final numbers use the full validation and test splits and carry a
standard error; the training-time subset stays for speed and is labelled as such.
