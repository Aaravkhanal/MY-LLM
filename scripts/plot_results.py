"""Chart of the headline result: request-following pass rate on the FINAL held-out set, per model, with 95% intervals and chance.

    python scripts/plot_results.py --out docs/img/final_set.png            (needs matplotlib: pip install -e ".[plots]")

Reads the result files under experiments/*/eval/ (nothing is typed in by hand). Colours are the project palette (warm neutrals, green
and ochre: no blue, no purple), checked with the dataviz palette validator; the two series also differ in marker shape and are labelled
directly, so colour is never the only cue. Intervals are the clustered bootstrap intervals over distinct prompts.
"""

import argparse
import json
from pathlib import Path

MODELS = [("untuned 14M base", "exp_004_small_tinystories"), ("fine-tuned, exp_006", "exp_006_sft_small_instruct_v3"),
          ("fine-tuned, exp_007", "exp_007_sft_small_instruct_v4"), ("fine-tuned, exp_008", "exp_008_sft_small_instruct_v5"),
          ("fine-tuned, exp_009", "exp_009_sft_small_instruct_v6"),
          ("fine-tuned, exp_011", "exp_011_sft_small_instruct_v6_long")]
BG, INK, MUTED, RULE = "#fcfcfb", "#1b1b19", "#66645c", "#e6e4dd"
GREEN, OCHRE = "#2f8558", "#a8611f"


def load(root: Path, exp: str, mode: str) -> dict:
    d = json.loads((root / "experiments" / exp / "eval" / "best__instruct_v4_final.json").read_text())["instruction"][mode]["pass_rate"]
    return {"rate": 100 * d["rate"], "lo": 100 * d["ci95_clustered"][0], "hi": 100 * d["ci95_clustered"][1], "chance": 100 * d.get("chance", 0)}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/img/final_set.png")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = {m: {name: load(root, exp, m) for name, exp in MODELS} for m in ("greedy", "sampled")}
    chance = data["greedy"][MODELS[0][0]]["chance"]

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "text.color": INK})
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    n = len(MODELS)
    for i, (name, _) in enumerate(MODELS):
        y = n - 1 - i
        for mode, dy, color, marker in (("greedy", 0.16, GREEN, "o"), ("sampled", -0.16, OCHRE, "D")):
            d = data[mode][name]
            ax.plot([d["lo"], d["hi"]], [y + dy] * 2, color=color, lw=1.6, solid_capstyle="round", alpha=0.9, zorder=2)
            ax.plot(d["rate"], y + dy, marker=marker, ms=7 if marker == "o" else 6.5, color=color, mec=BG, mew=1.5, zorder=3)
            ax.text(d["hi"] + 1.2, y + dy, f"{d['rate']:.0f}%", va="center", fontsize=8.5, color=MUTED)
    ax.axvline(chance, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(chance + 0.8, n - 0.42, f"chance {chance:.1f}%\n(a story that ignores the request)", fontsize=8, color=MUTED, va="top")
    ax.set_yticks(range(n)); ax.set_yticklabels([m[0] for m in reversed(MODELS)], fontsize=10)
    ax.set_xlim(0, 72); ax.set_ylim(-0.6, n - 0.35)
    ax.set_xlabel("requests passed, % (95% interval over distinct prompts)", fontsize=9, color=MUTED)
    ax.tick_params(axis="both", length=0, colors=MUTED); ax.tick_params(axis="y", colors=INK)
    ax.grid(axis="x", color=RULE, lw=1, zorder=0); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Line2D([], [], marker="o", color=GREEN, lw=1.6, ms=7, mec=BG, label="greedy decoding"),
               plt.Line2D([], [], marker="D", color=OCHRE, lw=1.6, ms=6.5, mec=BG, label="sampled (temperature 0.7)")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK, bbox_to_anchor=(0.985, 0.995),
               handletextpad=0.4, columnspacing=1.6)                      # in the header, clear of every data point
    fig.text(0.01, 0.985, "Following requests it has never seen worded that way", fontsize=12, fontweight="bold", va="top", color=INK)
    fig.text(0.01, 0.925, "312 held-out prompts, wording and stories written before any model was scored", fontsize=9, color=MUTED, va="top")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.14)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=BG)
    print(out)


if __name__ == "__main__":
    main()
