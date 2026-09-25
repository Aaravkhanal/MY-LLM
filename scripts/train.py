"""Train from a config, or resume an interrupted experiment.

    python scripts/train.py --config configs/tiny_tinystories.yaml
    python scripts/train.py --config configs/tiny_tinystories.yaml train.max_steps=500 train.lr=5e-4
    python scripts/train.py --resume experiments/exp_001_tiny_tinystories

Ctrl-C finishes the current step, saves a checkpoint and exits cleanly; --resume continues
from it with identical results.
"""

import argparse
from pathlib import Path

from aarav_lm.config import ExperimentConfig
from aarav_lm.paths import EXPERIMENT_DIR, REPO_ROOT
from aarav_lm.train.sft_trainer import make_trainer
from aarav_lm.train.trainer import next_experiment_dir

# Changing anything else mid-experiment would make the resumed run differ from the recorded config.
RESUME_SAFE = {"train.max_steps", "train.eval_interval", "train.log_interval", "train.ckpt_interval", "train.device"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="YAML experiment config (starts a new experiment)")
    src.add_argument("--resume", help="experiment directory to resume")
    ap.add_argument("--exp-root", default=str(EXPERIMENT_DIR), help="where new experiment dirs are created")
    ap.add_argument("overrides", nargs="*", help="key=value overrides, e.g. train.lr=5e-4")
    args = ap.parse_args()

    if args.resume:
        run_dir = Path(args.resume)
        bad = [o for o in args.overrides if o.split("=", 1)[0] not in RESUME_SAFE]
        if bad:
            raise SystemExit(f"cannot override {bad} when resuming (allowed: {sorted(RESUME_SAFE)})")
        exp = ExperimentConfig.from_yaml(run_dir / "config.yaml").with_overrides(args.overrides)
    else:
        exp = ExperimentConfig.from_yaml(args.config).with_overrides(args.overrides)
        run_dir = next_experiment_dir(Path(args.exp_root), exp.name)

    summary = make_trainer(exp, run_dir, repo_root=REPO_ROOT).run(resume=bool(args.resume))
    print(f"\nexperiment: {run_dir}\nstatus: {summary['status']}  best val loss: {summary['best_val_loss']}")


if __name__ == "__main__":
    main()
