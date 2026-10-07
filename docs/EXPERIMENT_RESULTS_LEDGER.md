# Experiment and test results ledger

This file is the compact, auditable index of completed runs and verification. Detailed interpretation remains in
`docs/RESULTS.md`; scientific corrections remain in `docs/CORRECTNESS.md`.

## Current verification

- Full repository test suite: **564 passed** on 2026-10-07.
- Local `main` and `origin/main` were synchronized before these documentation updates.
- The repository had no uncommitted model or code changes before this ledger and README refresh.

## Completed experiments

`Best val loss` is each run summary's periodic validation result; it is not directly comparable between
pretraining and SFT datasets. Full-split values are called out separately.

| Experiment | Stage | Parameters | Steps | Tokens seen | Best val loss | Main purpose/result |
|---|---:|---:|---:|---:|---:|---|
| exp_001 tiny | pretrain | 4.95M | 3,000 | 49.2M | 1.6422 | First complete from-scratch model. |
| exp_002–003 tiny SFT | SFT | 4.95M | 600 / 2,000 | 4.6M / 16.2M | 1.4479 / 1.4696 | Established the first chat/instruction pipeline. |
| exp_004 small | pretrain | 14.16M | 5,500 | 90.1M | 1.3978 | First 14M base model. |
| exp_005–009 small SFT | SFT | 14.16M | 1,800 each | 14.6–15.6M | 1.2895–1.3334 | Iterated instruction-data recipes; exp_009 final-set score 46.5% greedy / 41.7% sampled. |
| exp_010 small continued | pretrain | 14.16M | 9,000 | 147.5M | 1.2855 | Full-split loss 1.2876 ± 0.0037 after 237.6M cumulative tokens. |
| exp_011–016 small SFT/seed study | SFT | 14.16M | 1,800–3,600 | 15.6–31.1M | 1.2282–1.3305 | Showed that two-epoch length control is seed-dependent; one-epoch runs fail more consistently on `long`. |
| exp_017 context 1024 | pretrain | 14.16M | 3,000 | 49.2M | 1.2367 | Verified principled 512→1024 context extension and fixed continued-training config handling. |
| exp_018 medium | pretrain | 25.70M | 11,000 | 180.2M | 1.2588 | Full-split loss 1.2630 ± 0.0036, better than small with fewer tokens seen. |
| exp_019–021 medium SFT seeds | SFT | 25.70M | 1,800 each | 15.6M each | 1.2017–1.2036 | Final-set greedy 49.0%, 49.7%, 52.4%; the short-request regression persists across the scale-up. |
| exp_022 large | pretrain | 48.10M | 15,000 | 245.8M | 1.1767 | Completed in 33.5 h; full-split loss **1.1713 ± 0.0027**, perplexity 3.2261, 94.5% on 55 probes. |
| exp_023 large SFT | SFT | 48.10M | 2,400 | 20.7M | 1.1230 | Final set **57.6% greedy / 48.6% sampled**; 0% greedy short-request success and only one seed, so not yet a stable scale claim. |

## Current model conclusion

The 48.1M model is complete, not merely planned. It materially improves language-model validation loss and posts
the best single-run greedy instruction score so far. The instruction result still needs seed replication, and the
known length-control failure remains: larger parameter count alone did not solve it.
