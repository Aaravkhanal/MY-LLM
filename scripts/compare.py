"""Render a Markdown comparison of every experiment under experiments/.

    python scripts/compare.py                      # writes experiments/COMPARISON.md and prints it
"""

import argparse
from pathlib import Path

from aarav_lm.evaluation.compare import collect_experiments, render_markdown
from aarav_lm.paths import EXPERIMENT_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(EXPERIMENT_DIR))
    ap.add_argument("--out", default=None, help="default: <root>/COMPARISON.md")
    args = ap.parse_args()
    md = render_markdown(collect_experiments(Path(args.root)))
    out = Path(args.out) if args.out else Path(args.root) / "COMPARISON.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()
