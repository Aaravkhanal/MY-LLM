"""Build src/aarav_lm/web/results.json: the numbers the website's interactive results explorer draws.

    python scripts/build_site_data.py

Everything comes from the committed result files under experiments/*/eval/ (nothing is typed in by hand), so the site can only show
what the evaluation actually produced. Re-run it after a new model is evaluated and commit the result.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                              # runnable from a checkout without installing

from aarav_lm.evaluation.instruct import MODES                     # noqa: E402  (the decoding settings the evaluation really used)
from aarav_lm.evaluation.paired import paired_compare               # noqa: E402
MODELS = [
    ("base", "Untuned 14M base", "exp_004_small_tinystories"),
    ("exp006", "Fine-tuned, exp_006", "exp_006_sft_small_instruct_v3"),
    ("exp007", "Fine-tuned, exp_007", "exp_007_sft_small_instruct_v4"),
    ("exp008", "Fine-tuned, exp_008", "exp_008_sft_small_instruct_v5"),
    ("exp009", "Fine-tuned, exp_009", "exp_009_sft_small_instruct_v6"),
    ("exp011", "Fine-tuned, exp_011 (longer-pretrained base)", "exp_011_sft_small_instruct_v6_long"),
    ("exp012", "Fine-tuned, exp_012 recipe, seed 1337 (longer base, 2 epochs)", "exp_012_sft_small_instruct_v6_long_2ep"),
    ("exp013", "Fine-tuned, exp_012 recipe, seed 1338", "exp_013_sft_small_instruct_v6_long_2ep_seed1338"),
    ("exp014", "Fine-tuned, exp_012 recipe, seed 1339", "exp_014_sft_small_instruct_v6_long_2ep_seed1339"),
    ("exp015", "Fine-tuned, exp_009 recipe, seed 1338", "exp_015_sft_small_instruct_v6_seed1338"),
    ("exp016", "Fine-tuned, exp_009 recipe, seed 1339", "exp_016_sft_small_instruct_v6_seed1339"),
]
DEFAULT_MODEL = "exp013"          # served by the demo site: chosen by the mean of the development and trained-wording sets (not the final set), see RESULTS.md
SEED_MODELS = ["exp012", "exp013", "exp014"]                        # exp_012's recipe (2 epochs) with three seeds
SEED_MODELS_9 = ["exp009", "exp015", "exp016"]                      # exp_009's recipe (1 epoch) with three seeds
ROLES = {                                                           # one plain sentence each: what the version is (no measurements)
    "base": "The pretrained base with no fine-tuning: continues text, does not follow requests on purpose.",
    "exp006": "First real recipe (v3 data). Learned request wording it had seen; lost on new wording.",
    "exp007": "More varied wording and requests that must dwell on the asked-for word (v4 data).",
    "exp008": "exp_007 with length requests sampled three times as often. Did not help.",
    "exp009": "Adds prominence rules for two-topic requests (v6 data). Best on the training wording for short requests.",
    "exp011": "exp_009's recipe on a base pretrained 147M tokens longer, one epoch.",
    "exp012": "exp_011 for about two epochs, training seed 1337.",
    "exp013": "The exp_012 recipe with training seed 1338 (nothing else changed). The default model of the demo site.",
    "exp014": "The exp_012 recipe with training seed 1339 (nothing else changed).",
    "exp015": "exp_009's recipe (one epoch) with training seed 1338 (nothing else changed).",
    "exp016": "exp_009's recipe (one epoch) with training seed 1339 (nothing else changed).",
}
SETS = [
    ("final", "Wording it has never seen (final set, written before any model was scored)", "instruct_v4_final"),
    ("template", "Wording it was trained on (same kinds, held-out stories)", "instruct_v3.intemplate"),
    ("dev", "Development wording (used while tuning the recipes, so not held back)", "instruct_v3"),
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


def _read(exp: str, name: str) -> dict:
    return json.loads((ROOT / "experiments" / exp / name).read_text())


def _paired(res_a: dict, res_b: dict, mode: str) -> dict:
    r = paired_compare(res_a["instruction"][mode]["per_prompt"], res_b["instruction"][mode]["per_prompt"])
    # A is the model, B the reference: diff = model minus reference, so a positive number means the model passes more
    return {k: {"diff": round(v["diff"], 4), "lo": round(v["ci95"][0], 4), "hi": round(v["ci95"][1], 4), "p": round(v["p"], 4),
                "distinct": v["n_distinct"]} for k, v in r.items()}


def training() -> dict:
    """Facts about each stage, read from the run folders (never typed)."""
    def stage(exp: str, kind: str) -> dict:
        s = _read(exp, "summary.json")
        return {"experiment": exp, "kind": kind, "steps": s["steps"], "tokens_seen": s["tokens_seen"], "seconds": round(s["train_seconds"]),
                "best_val_loss": round(s["best_val_loss"], 4), "params": s["params"]}
    full = {}
    for exp in ("exp_004_small_tinystories", "exp_010_small_long_tinystories"):
        f = _read(exp, "eval/best__fullsplit_loss_1200mb.json")
        full[exp] = {sp: {k: f[sp][k] for k in ("loss", "stderr", "perplexity", "top1", "bits_per_byte", "windows")} for sp in ("val", "test")}
    return {"stages": [stage("exp_004_small_tinystories", "pretrain"), stage("exp_010_small_long_tinystories", "pretrain, continued from exp_004"),
                       stage("exp_009_sft_small_instruct_v6", "fine-tune"), stage("exp_011_sft_small_instruct_v6_long", "fine-tune"),
                       stage("exp_012_sft_small_instruct_v6_long_2ep", "fine-tune"),
                       stage("exp_013_sft_small_instruct_v6_long_2ep_seed1338", "fine-tune (seed 1338)"),
                       stage("exp_014_sft_small_instruct_v6_long_2ep_seed1339", "fine-tune (seed 1339)"),
                       stage("exp_009_sft_small_instruct_v6", "fine-tune (seed 1337, 1 epoch)"),
                       stage("exp_015_sft_small_instruct_v6_seed1338", "fine-tune (seed 1338, 1 epoch)"),
                       stage("exp_016_sft_small_instruct_v6_seed1339", "fine-tune (seed 1339, 1 epoch)")],
            "full_split_loss": full}


def _one_seed_spread(sets_raw: dict, models: list) -> dict:
    import statistics as st
    out = {"models": models, "sets": {}, "kinds": {}}
    for sid, raw in sets_raw.items():
        out["sets"][sid] = {}
        for mode in ("greedy", "sampled"):
            v = [100 * raw[m]["instruction"][mode]["pass_rate"]["rate"] for m in models]
            out["sets"][sid][mode] = {"values": [round(x, 1) for x in v], "mean": round(st.mean(v), 1), "sd": round(st.stdev(v), 1), "range": round(max(v) - min(v), 1)}
    tmpl = sets_raw["template"]
    for k in KIND_LABELS:
        out["kinds"][k] = {mode: [round(100 * tmpl[m]["instruction"][mode]["by_task"][k]["rate"], 1) for m in models] for mode in ("greedy", "sampled")}
    return out


def seed_spread(sets_raw: dict) -> dict:
    """Mean, standard deviation and range of the pass rate over the training seeds of each recipe, per set and decoding, and the
    per-kind rates (which show the exp_012 recipe's length skills come and go with the seed, while exp_009's recipe's do not)."""
    return {"epoch2": _one_seed_spread(sets_raw, SEED_MODELS), "epoch1": _one_seed_spread(sets_raw, SEED_MODELS_9),
            "models": SEED_MODELS, "sets": _one_seed_spread(sets_raw, SEED_MODELS)["sets"], "kinds": _one_seed_spread(sets_raw, SEED_MODELS)["kinds"]}


def build() -> dict:
    out = {"models": [{"id": i, "label": l, "experiment": e, "stage": "base" if i == "base" else "fine-tuned", "about": ROLES[i]}
                      for i, l, e in MODELS],
           "default_model": DEFAULT_MODEL, "sets": [], "kinds": [{"id": k, "label": v} for k, v in KIND_LABELS.items()],
           "decoding": {m: {k: v for k, v in kw.items() if v is not None} for m, kw in MODES.items()}, "max_new_tokens": 400,
           "training": training()}
    exps = {mid: exp for mid, _, exp in MODELS}
    sets_raw = {}
    for sid, slabel, fname in SETS:
        raw = {mid: _read(exp, f"eval/best__{fname}.json") for mid, exp in exps.items()}
        first = raw["base"]
        assert len({r["eval_sets"]["instruct"]["sha256"] for r in raw.values()}) == 1, "models were scored on different files"
        block = {"id": sid, "label": slabel, "prompts": first["instruction"]["greedy"]["prompts"], "scored": first["instruction"]["greedy"]["scored"],
                 "sha256": first["eval_sets"]["instruct"]["sha256"], "results": {}, "vs_base": {}, "vs_default": {}}
        for mid in exps:
            res = raw[mid]["instruction"]
            block["results"][mid] = {mode: {"all": cell(res[mode]["pass_rate"]),
                                            **{k: cell(v) for k, v in res[mode]["by_task"].items()}} for mode in ("greedy", "sampled")}
            if mid != "base":
                block["vs_base"][mid] = {mode: _paired(raw[mid], raw["base"], mode) for mode in ("greedy", "sampled")}
                if mid != DEFAULT_MODEL:
                    block["vs_default"][mid] = {mode: _paired(raw[mid], raw[DEFAULT_MODEL], mode)["ALL"] for mode in ("greedy", "sampled")}
        out["sets"].append(block)
        sets_raw[sid] = raw
    out["seeds"] = seed_spread(sets_raw)
    return out


def main() -> None:
    data = build()
    path = ROOT / "src" / "aarav_lm" / "web" / "results.json"
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
