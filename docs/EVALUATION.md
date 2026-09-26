# Evaluation

How quality is measured here, why each number can be trusted, and how to read the results. The rule: **a number
is only reported next to what it should be compared with and how uncertain it is.** There is no judge model and nothing
external is called, so every metric is a checkable proxy, and each says so in `src/aarav_lm/evaluation/`.

## What is measured

| Layer | What | Comparison / uncertainty | Code |
|---|---|---|---|
| Held-out loss | token-weighted loss, perplexity, top-1/top-5, bits per byte on the **whole** validation and test splits | bootstrap standard error over windows or conversations; bits per byte is tokenizer-independent | `evaluation/loss.py` |
| Generation | 40 (base) / 30 (chat) fixed prompts, greedy and sampled: capitalisation, non-word rate, loop rate, distinct-n, natural endings, name reuse, length | fixed seeds; proxies, not quality | `evaluation/metrics.py`, `suite.py` |
| Probes | 55 forced-choice minimal pairs (grammar, story logic, common sense); every choice verified to be one token | chance is 50% | `evaluation/probes.py` |
| Edge cases | 14 (empty/whitespace/Unicode/emoji prompts, control-token text inert, prompt and output longer than context, determinism, seeded reproducibility, no state leaking between requests, stop sequences, round-trip) | all must pass | `evaluation/edge_cases.py` |
| **Request following** | rule-checked requests: "contains the name", "at most 90 words", "has dialogue", ... | **chance level, lift, distinct-prompt count, clustered interval, paired comparison** | `evaluation/instruct.py`, `paired.py` |
| Regression gate | `--baseline results.json`: exit 1 if a metric drops; refuses to compare results from different eval sets, datasets, tokenizers or settings | tolerances per metric | `evaluation/regression.py` |

## Request following, in detail

A reply passes a prompt only if every check passes. The prompt sets are built from **held-out stories** (the test split; no
model trains on them). Each has 13 request kinds × 24 prompts:

| File | Wording | Role |
|---|---|---|
| `evals/instruct_v3.jsonl` | never used in training | development set: unseen wording |
| `evals/instruct_v3.intemplate.jsonl` | the wording the model **was** trained on | control: did it learn the task at all? |
| `evals/instruct_v4_final.jsonl` | new openers and stories, committed **before** anything was scored on it | confirmation set: use once per candidate, do not tune on it |

Each set has a `*.chance.json`: the pass rate of a real story that **ignores** the request (aggregate about 10.6%;
"contains dialogue" is 67% by luck alone). Results carry, per kind and overall:

* `rate` and `chance`, `lift` = rate − chance;
* `n_distinct` and `ci95_clustered`: kinds like "a short story" repeat a few strings 24 times, so the effective sample
  size is the number of *distinct* prompts (190 of 312 in the development set); the interval resamples distinct prompts;
* `per_prompt`: each prompt's outcome, for **paired** comparison of two checkpoints on identical prompts
  (difference in points, cluster-bootstrap interval, p-value).

```bash
python scripts/make_eval_controls.py --eval evals/instruct_v3.jsonl --stories <TinyStories .txt> [--final-out evals/instruct_v4_final.jsonl]
python scripts/eval_instruct.py --checkpoint experiments/exp_006_*/checkpoints/best.pt \
       --instruct evals/instruct_v3.jsonl evals/instruct_v3.intemplate.jsonl        # minutes; writes <run>/eval/*.json
python scripts/compare_instruct.py A.json B.json --mode greedy                     # paired, per kind and overall
python scripts/evaluate.py --checkpoint <ckpt> [--instruct evals/instruct_v3.jsonl] # the full suite (~20 min on CPU)
python scripts/compare.py                                                         # side-by-side table of every experiment
```

## Reading a result

1. Look at **lift over chance**, not the raw rate.
2. Look at the **in-template control next to the unseen-wording set**. High on the first and low on the second means
   the model learned a template family, not the request (that is exactly exp_006, see [RESULTS.md](RESULTS.md)).
3. Prefer **paired** differences to two separate intervals. A difference whose interval contains 0 is not evidence.
4. A validation loss is comparable **only** between models scored on the same split with the same tokenizer; each
   SFT dataset has its own validation conversations. Quote the standard error.
5. Greedy and sampled decoding can disagree by 10+ points; report both.

## Guarantees the tests enforce

Held-out loss agrees exactly with the older loss functions; its standard error matches the analytic value for
equal-weight rows; the chance computation is checked on toy stories; duplicated prompts widen the interval; the paired
test detects a real gap and stays quiet on a null one; training and every eval wording are disjoint strings; the default
data generator is byte-identical to the one exp_006 was trained on (hash-pinned).

## Known limits

* Proxies, not a judge: they catch regressions and gross failures, not whether a story is *good*.
* One training run per configuration, so seed-to-seed variance is not measured. With 24 prompts per kind the per-kind
  interval is about ±20 points; only large per-kind gaps (0% vs 100%) are conclusive.
* In-domain only. The models write simple children's stories; the suite does not test knowledge, reasoning, code or
  maths because they cannot do those, and a suite of zeros would say nothing. The out-of-domain behaviour that matters
  (empty, huge, Unicode, injected control tokens) is in the edge cases.
* Every scientific problem found in this pipeline, with evidence, is in [CORRECTNESS.md](CORRECTNESS.md).
