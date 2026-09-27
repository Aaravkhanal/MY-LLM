"""Build src/aarav_lm/web/results.json: the numbers the website's interactive results explorer draws.

    python scripts/build_site_data.py

Everything comes from the committed result files under experiments/*/eval/ (nothing is typed in by hand), so the site can only show
what the evaluation actually produced. Re-run it after a new model is evaluated and commit the result.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("base", "Untuned 14M base", "exp_004_small_tinystories"),
    ("exp006", "Fine-tuned, exp_006", "exp_006_sft_small_instruct_v3"),
    ("exp007", "Fine-tuned, exp_007", "exp_007_sft_small_instruct_v4"),
    ("exp008", "Fine-tuned, exp_008", "exp_008_sft_small_instruct_v5"),
    ("exp009", "Fine-tuned, exp_009", "exp_009_sft_small_instruct_v6"),
    ("exp011", "Fine-tuned, exp_011 (longer-pretrained base)", "exp_011_sft_small_instruct_v6_long"),
    ("exp012", "Fine-tuned, exp_012 (longer base, 2 epochs)", "exp_012_sft_small_instruct_v6_long_2ep"),
]
SETS = [
    ("final", "Wording it has never seen (final set, written before any model was scored)", "instruct_v4_final"),
    ("template", "Wording it was trained on (same kinds, held-out stories)", "instruct_v3.intemplate"),
]
KIND_LABELS = {
    "dialogue": "Story with dialogue", "emotion": "Story about a feeling", "long": "Long story", "long_topic": "Long story about a thing",
    "name": "Story about a named character", "name_emotion": "Named character and a feeling", "name_topic": "Named character and a thing",
    "place": "Story set at a place", "short": "Short story", "short_topic": "Short story about a thing", "topic": "Story about a thing",
    "two_topics": "Story with two things",
}


def cell(d: dict) -> dict:
    lo, hi = d.get("ci95_clustered") or d["ci95"]
    return {"rate": round(100 * d["rate"], 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1),
            "chance": round(100 * d.get("chance", 0.0), 1), "prompts": d.get("n_distinct", d["n"])}


def build() -> dict:
    out = {"models": [{"id": i, "label": l} for i, l, _ in MODELS], "sets": [], "kinds": [{"id": k, "label": v} for k, v in KIND_LABELS.items()]}
    for sid, slabel, fname in SETS:
        block = {"id": sid, "label": slabel, "results": {}}
        for mid, _, exp in MODELS:
            res = json.loads((ROOT / "experiments" / exp / "eval" / f"best__{fname}.json").read_text())["instruction"]
            block["results"][mid] = {mode: {"all": cell(res[mode]["pass_rate"]),
                                            **{k: cell(v) for k, v in res[mode]["by_task"].items()}} for mode in ("greedy", "sampled")}
        out["sets"].append(block)
    return out


def main() -> None:
    data = build()
    path = ROOT / "src" / "aarav_lm" / "web" / "results.json"
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
