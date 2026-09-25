"""Shared builders for tests that need a real (tiny) dataset and experiment."""

from pathlib import Path

from aarav_lm.config import ExperimentConfig
from aarav_lm.data.pipeline import build_dataset

EOT = "<|endoftext|>"
WORDS = ["dog", "cat", "sun", "park", "ball", "tree", "girl", "boy", "bird", "fish"]


def make_corpus(n=400):
    return [
        f"Once upon a time there was a little {WORDS[i % 10]} who liked the {WORDS[(i * 3 + 1) % 10]}. "
        f"Every day number {i} they went to the {WORDS[(i * 7 + 2) % 10]} and played happily together."
        for i in range(n)
    ]


def tiny_experiment(root: Path, name="unit", max_steps=12, **train_overrides) -> ExperimentConfig:
    """A ~30k-parameter experiment on a synthetic corpus; trains in well under a second per step on CPU."""
    src = root / "corpus.txt"
    if not src.exists():
        src.write_text((EOT + "\n").join(make_corpus()) + "\n", encoding="utf-8")
    exp = ExperimentConfig(name=name, model_preset="tiny")
    exp.model_overrides = {"n_layer": 2, "n_embd": 32, "n_head": 4, "n_kv_head": 2, "block_size": 32}
    exp.tokenizer.vocab_size = 300
    exp.tokenizer.artifact_dir = str(root / "tok")
    exp.data.sources = [str(src)]
    exp.data.out_dir = str(root / "processed")
    exp.data.val_fraction = 0.1
    exp.data.test_fraction = 0.1
    exp.data.min_chars = 20
    t = exp.train
    t.device, t.precision, t.seed = "cpu", "fp32", 7
    t.max_steps, t.batch_size, t.grad_accum = max_steps, 4, 2
    t.lr, t.warmup_steps, t.min_lr_ratio = 3e-3, 2, 0.1
    t.eval_interval, t.eval_iters, t.log_interval, t.ckpt_interval, t.keep_last = 4, 3, 4, 4, 5
    for k, v in train_overrides.items():
        setattr(t, k, v)
    if not (Path(exp.data.out_dir) / "manifest.json").exists():
        build_dataset(exp, log=lambda _: None)
    return exp
