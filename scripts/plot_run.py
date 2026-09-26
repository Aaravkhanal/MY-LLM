"""Plot a training run's curves (needs matplotlib: pip install -e ".[plots]").

    python scripts/plot_run.py experiments/exp_004_small_tinystories            # writes <run>/training_curves.png
    python scripts/plot_run.py experiments/exp_004_small_tinystories --out curves.png
"""

import argparse

from aarav_lm.paths import resolve
from aarav_lm.train.plots import plot_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    print(plot_run(resolve(args.run_dir), args.out))


if __name__ == "__main__":
    main()
