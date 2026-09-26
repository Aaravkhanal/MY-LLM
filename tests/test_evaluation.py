import copy
import json
from pathlib import Path

import pytest
import torch

from aarav_lm.evaluation import metrics as M
from aarav_lm.evaluation import probes as P
from aarav_lm.evaluation.compare import collect_experiments, render_markdown
from aarav_lm.evaluation.edge_cases import CASES, run_edge_cases
from aarav_lm.evaluation.loss import evaluate_loss, next_token_accuracy
from aarav_lm.evaluation.regression import DEFAULT_RULES, compare_to_baseline
from aarav_lm.evaluation.suite import file_sha256, run_suite, save_results

REPO = Path(__file__).resolve().parents[1]
PROMPTS, PROBES = REPO / "evals/prompts_v1.jsonl", REPO / "evals/probes_v1.jsonl"


# ── metrics ──────────────────────────────────────────────────────────────────

def test_distinct_n():
    assert M.distinct_n(list("abcd"), 1) == 1.0
    assert M.distinct_n(list("aaaa"), 1) == 0.25
    assert M.distinct_n(list("abab"), 2) == pytest.approx(2 / 3)
    assert M.distinct_n(["x"], 3) == 1.0                    # too short to have any 3-gram


def test_loop_detection():
    assert M.has_loop(list("abcabcabc"))
    assert M.has_loop(["the", "dog"] * 6 + ["end"])
    assert M.has_loop([1] * 4)
    assert not M.has_loop(list("abcdefghijkl"))
    assert not M.has_loop(list("abcabd"))                    # repeats only twice
    assert not M.has_loop([1, 2, 3, 1, 2, 4, 5, 6])


def test_max_token_run():
    assert M.max_token_run([1, 1, 2, 2, 2, 2, 3]) == 4 and M.max_token_run([]) == 0


def test_sentence_and_ending_metrics():
    s = M.sentence_stats("The dog ran. the cat sat! Then it slept.")
    assert s["sentences"] == 3 and s["capitalised_frac"] == pytest.approx(2 / 3)
    assert M.ends_cleanly("It was fun.") and M.ends_cleanly('He said "hi"') and not M.ends_cleanly("It was")
    assert not M.ends_cleanly("   ")


def test_nonword_rate_and_vocabulary():
    vocab = M.build_vocabulary(["the dog runs", "the dog runs", "the dog runs", "a rare word"], min_count=3)
    assert vocab == {"the", "dog", "runs"}
    assert M.nonword_rate("The dog blorps", vocab) == pytest.approx(1 / 3)
    assert M.nonword_rate("", vocab) == 0.0


def test_entity_mentions_are_whole_word_and_case_insensitive():
    assert M.mentions("Then lily smiled.", "Lily") and not M.mentions("Lilypad", "Lily")


def test_summarize_generations_values():
    samples = [
        {"text": "Lily ran home.", "tokens": [1, 2, 3, 4], "finish_reason": "eot", "entity": "Lily", "prompt": "p"},
        {"text": "the the the the", "tokens": [5, 5, 5, 5, 5, 5, 5, 5], "finish_reason": "length", "entity": "Tom", "prompt": "p"},
    ]
    r = M.summarize_generations(samples, {"lily", "ran", "home", "the"})
    assert r["samples"] == 2 and r["eot_rate"] == 0.5 and r["entity_consistency"] == 0.5
    assert r["loop_fraction"] == 0.5 and r["max_token_run"] == 8 and r["mean_tokens"] == 6
    assert r["nonword_rate"] == 0.0 and r["clean_ending_frac"] == 0.5
    with pytest.raises(ValueError):
        M.summarize_generations([], set())


# ── evaluation sets ──────────────────────────────────────────────────────────

def test_committed_prompt_and_probe_sets_are_well_formed():
    prompts, probes = P.load_jsonl(PROMPTS), P.load_jsonl(PROBES)
    assert len(prompts) >= 30 and len({p["id"] for p in prompts}) == len(prompts)
    assert all(p["prompt"].strip() for p in prompts)
    assert any(p["entity"] for p in prompts) and any(not p["entity"] for p in prompts)
    for p in prompts:
        if p["entity"]:
            assert p["entity"] in p["prompt"]
    assert len(probes) >= 50
    assert {p["category"] for p in probes} == {"grammar", "commonsense", "story_logic"}
    for p in probes:
        assert p["correct"] != p["incorrect"] and p["correct"].startswith(" ") and p["incorrect"].startswith(" ")
    keys = [(p["prefix"], p["correct"], p["incorrect"]) for p in probes]
    assert len(set(keys)) == len(keys)


def test_probe_choices_are_single_tokens_so_scoring_is_fair():
    """Minimal pairs are only fair if both choices are scored over comparable token counts."""
    tok_dir = REPO / "artifacts/tokenizer_tinystories_v1"
    if not tok_dir.exists():
        pytest.skip("milestone tokenizer not built")
    from aarav_lm.tokenizer import AaravTokenizer
    tok = AaravTokenizer.load(tok_dir)
    for p in P.load_jsonl(PROBES):
        a, b = len(tok.encode(p["correct"])), len(tok.encode(p["incorrect"]))
        assert a == b == 1, p          # unequal token counts would bias the log-prob comparison


def test_load_jsonl_reports_the_bad_line(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\nnot json\n')
    with pytest.raises(ValueError, match="x.jsonl:2"):
        P.load_jsonl(f)


# ── probes ───────────────────────────────────────────────────────────────────

def test_continuation_logprob_matches_a_manual_computation(engine):
    got = P.continuation_logprob(engine, "The dogs", " are")
    p, c = engine.tok.encode("The dogs"), engine.tok.encode(" are")
    x = torch.tensor([p + c])
    with torch.no_grad():
        lp = torch.log_softmax(engine.model(x)[0][0], dim=-1)
    want = sum(float(lp[len(p) - 1 + i, t]) for i, t in enumerate(c))
    assert got["sum"] == pytest.approx(want, abs=1e-4) and got["tokens"] == len(c)


def test_run_probes_scores_by_category(engine, monkeypatch):
    fake = {("a", " y"): -1.0, ("a", " n"): -2.0, ("b", " y"): -2.0, ("b", " n"): -1.0}
    monkeypatch.setattr(P, "continuation_logprob",
                        lambda e, pre, cont: {"sum": fake[(pre, cont)], "mean": fake[(pre, cont)], "tokens": 1})
    res = P.run_probes(engine, [
        {"category": "g", "prefix": "a", "correct": " y", "incorrect": " n"},
        {"category": "g", "prefix": "b", "correct": " y", "incorrect": " n"},
        {"category": "c", "prefix": "a", "correct": " y", "incorrect": " n"},
    ])
    assert res["accuracy"] == pytest.approx(2 / 3)
    assert res["by_category"]["g"] == {"accuracy": 0.5, "n": 2} and res["by_category"]["c"]["accuracy"] == 1.0


def test_probe_ties_count_as_wrong(engine, monkeypatch):
    monkeypatch.setattr(P, "continuation_logprob", lambda *a: {"sum": -1.0, "mean": -1.0, "tokens": 1})
    assert P.run_probes(engine, [{"category": "g", "prefix": "a", "correct": " y", "incorrect": " n"}])["accuracy"] == 0.0


# ── loss / accuracy ──────────────────────────────────────────────────────────

def test_loss_is_token_weighted_and_ignores_masked_targets(engine):
    x = torch.randint(0, 300, (2, 16))
    y = x.roll(-1, 1).clone()
    y[1, :12] = -1                                            # second row mostly masked
    res = evaluate_loss(engine.model, [(x, y)], engine.device)
    assert res["tokens"] == 16 + 4
    import torch.nn.functional as F
    with torch.no_grad():
        logits, _ = engine.model(x)
    want = F.cross_entropy(logits.view(-1, 300), y.view(-1), ignore_index=-1).item()
    assert res["loss"] == pytest.approx(want, rel=1e-5)


def test_next_token_accuracy_bounds_and_ordering(engine, tiny_run):
    from aarav_lm.data.dataset import TokenDataset
    from aarav_lm.data.shards import Shard
    ds = TokenDataset(Shard.open(Path(tiny_run["exp"].data.out_dir) / "val.bin"), engine.block_size)
    acc = next_token_accuracy(engine.model, ds.fixed_eval_batches(4, 3), engine.device)
    assert 0 <= acc["top1"] <= acc["top5"] <= 1 and acc["top1"] > 1 / 300      # better than chance after training


# ── edge cases ───────────────────────────────────────────────────────────────

def test_edge_cases_all_pass_on_a_trained_model(engine):
    res = run_edge_cases(engine)
    failed = [c for c in res["cases"] if not c["passed"]]
    assert res["all_passed"] and res["total"] == len(CASES), failed


def test_edge_cases_report_a_crash_instead_of_raising(engine, monkeypatch):
    monkeypatch.setitem(CASES, "boom", lambda e: (_ for _ in ()).throw(RuntimeError("kaput")))
    res = run_edge_cases(engine)
    boom = next(c for c in res["cases"] if c["name"] == "boom")
    assert not boom["passed"] and "kaput" in boom["detail"] and not res["all_passed"]


# ── suite / reproducibility / regression / compare ───────────────────────────

@pytest.fixture(scope="module")
def small_sets(tmp_path_factory):
    d = tmp_path_factory.mktemp("sets")
    (d / "prompts.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "a", "prompt": "Once upon a time there was a little dog", "entity": "dog"},
        {"id": "b", "prompt": "One day the sun", "entity": None},
        {"id": "c", "prompt": "Every day number", "entity": None},
    ]) + "\n")
    (d / "probes.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"category": "grammar", "prefix": "Once upon a time there was a little", "correct": " dog", "incorrect": " park"},
        {"category": "grammar", "prefix": "Every day number", "correct": " 3", "incorrect": " the"},
    ]) + "\n")
    return d / "prompts.jsonl", d / "probes.jsonl"


def _suite(engine, tiny_run, sets):
    return run_suite(engine, tiny_run["exp"].data.out_dir, sets[0], sets[1], max_new_tokens=15,
                     max_windows=8, log=lambda _: None)


def test_suite_produces_every_section_and_is_reproducible(engine, tiny_run, small_sets, tmp_path):
    a = _suite(engine, tiny_run, small_sets)
    b = _suite(engine, tiny_run, small_sets)
    for section in ("loss", "generation", "probes", "edge_cases", "eval_sets", "model", "settings"):
        assert section in a
    assert set(a["generation"]) == {"greedy", "sampled"}
    assert a["loss"]["val"]["loss"] == b["loss"]["val"]["loss"]
    assert a["generation"]["sampled"]["samples"] == b["generation"]["sampled"]["samples"]   # fixed seeds -> same text
    assert a["probes"]["items"] == b["probes"]["items"]
    assert a["eval_sets"]["prompts"]["sha256"] == file_sha256(small_sets[0])
    path = save_results(a, tmp_path / "r" / "x_results.json")
    assert json.loads(path.read_text())["loss"]["val"]["loss"] == a["loss"]["val"]["loss"]


def test_regression_check(engine, tiny_run, small_sets):
    base = _suite(engine, tiny_run, small_sets)
    ok, results = compare_to_baseline(base, base)
    assert ok and all(r.passed for r in results)

    worse = copy.deepcopy(base)
    worse["loss"]["val"]["loss"] *= 1.10
    ok, results = compare_to_baseline(worse, base)
    assert not ok and [r.metric for r in results if not r.passed] == ["loss.val.loss"]

    clean_base = copy.deepcopy(base)                                  # explicit values: the tiny model may already loop
    clean_base["generation"]["greedy"]["loop_fraction"] = 0.0
    looping = copy.deepcopy(clean_base)
    looping["generation"]["greedy"]["loop_fraction"] = 0.5
    ok, results = compare_to_baseline(looping, clean_base)
    assert not ok and [r.metric for r in results if not r.passed] == ["generation.greedy.loop_fraction"]
    slightly = copy.deepcopy(clean_base)
    slightly["generation"]["greedy"]["loop_fraction"] = 0.05          # within the 0.10 tolerance
    assert compare_to_baseline(slightly, clean_base)[0]

    broken = copy.deepcopy(base)
    broken["edge_cases"]["all_passed"] = False
    assert not compare_to_baseline(broken, base)[0]

    tiny_drop = copy.deepcopy(base)
    tiny_drop["loss"]["val"]["loss"] *= 1.01                          # within the 2% tolerance
    assert compare_to_baseline(tiny_drop, base)[0]


def test_regression_refuses_to_compare_different_eval_sets(engine, tiny_run, small_sets):
    base = _suite(engine, tiny_run, small_sets)
    other = copy.deepcopy(base)
    other["eval_sets"]["probes"]["sha256"] = "0" * 64
    ok, results = compare_to_baseline(other, base)
    assert not ok and "not comparable" in results[0].note


def test_missing_metric_in_new_results_is_a_failure(engine, tiny_run, small_sets):
    base = _suite(engine, tiny_run, small_sets)
    new = copy.deepcopy(base)
    del new["probes"]["accuracy"]
    ok, results = compare_to_baseline(new, base)
    assert not ok and any(r.metric == "probes.accuracy" and r.note == "metric missing" for r in results)


@pytest.fixture()
def instruct_file(tmp_path):
    rows = [
        {"id": "a", "kind": "topic", "prompt": "Tell me about a dog", "checks": [{"kind": "contains_all", "words": ["dog"], "n": 0}], "unseen_name": False},
        {"id": "b", "kind": "short", "prompt": "Write a tiny story", "checks": [{"kind": "max_words", "words": [], "n": 400}], "unseen_name": False},
        {"id": "c", "kind": "name", "prompt": "A story about Zara", "checks": [{"kind": "contains_all", "words": ["Zara"], "n": 0}], "unseen_name": True},
        {"id": "d", "kind": "generic", "prompt": "Tell me a story", "checks": [], "unseen_name": False},
    ]
    f = tmp_path / "instruct.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return f


def test_default_rules_reference_real_result_paths(engine, tiny_run, small_sets, instruct_file):
    from aarav_lm.evaluation.regression import get_path
    res = run_suite(engine, tiny_run["exp"].data.out_dir, small_sets[0], small_sets[1], max_new_tokens=15,
                    max_windows=8, log=lambda _: None, instruct_path=instruct_file)
    for path, _, _ in DEFAULT_RULES:
        if path.endswith("entity_consistency"):
            continue                                                  # legitimately None without entity prompts
        assert get_path(res, path) is not None, path


def test_instruction_section_structure_and_ci(engine, tiny_run, small_sets, instruct_file):
    res = run_suite(engine, tiny_run["exp"].data.out_dir, small_sets[0], small_sets[1], max_new_tokens=15,
                    max_windows=8, log=lambda _: None, instruct_path=instruct_file)
    ins = res["instruction"]
    assert set(ins) == {"greedy", "sampled"} and res["eval_sets"]["instruct"]["n"] == 4
    for mode in ins.values():
        assert mode["prompts"] == 4 and mode["scored"] == 3                       # the check-less prompt is excluded from rates
        r = mode["pass_rate"]
        assert r["n"] == 3 and 0 <= r["ci95"][0] <= r["rate"] <= r["ci95"][1] <= 1
        assert set(mode["by_task"]) == {"topic", "short", "name"}
        assert mode["name_unseen"]["n"] == 1 and mode["name_seen"] is None
    plain = run_suite(engine, tiny_run["exp"].data.out_dir, small_sets[0], small_sets[1], max_new_tokens=15,
                      max_windows=8, log=lambda _: None)
    assert plain["instruction"] is None and plain["eval_sets"]["instruct"] is None


def test_regression_refuses_different_instruction_sets_and_flags_a_drop(engine, tiny_run, small_sets, instruct_file):
    base = run_suite(engine, tiny_run["exp"].data.out_dir, small_sets[0], small_sets[1], max_new_tokens=15,
                     max_windows=8, log=lambda _: None, instruct_path=instruct_file)
    other = copy.deepcopy(base)
    other["eval_sets"]["instruct"]["sha256"] = "0" * 64
    ok, results = compare_to_baseline(other, base)
    assert not ok and "instruction-following prompt sets" in results[0].note
    worse = copy.deepcopy(base)
    worse["instruction"]["sampled"]["pass_rate"]["rate"] = max(0.0, base["instruction"]["sampled"]["pass_rate"]["rate"] - 0.2)
    if base["instruction"]["sampled"]["pass_rate"]["rate"] >= 0.2:
        ok, results = compare_to_baseline(worse, base)
        assert not ok and any(r.metric == "instruction.sampled.pass_rate.rate" and not r.passed for r in results)


def test_comparison_report(engine, tiny_run, small_sets, tmp_path):
    import shutil
    root = tmp_path / "experiments"
    run = root / "exp_001_demo"
    shutil.copytree(tiny_run["run"], run, ignore=shutil.ignore_patterns("checkpoints"))
    save_results(_suite(engine, tiny_run, small_sets), run / "eval" / "final_results.json")
    (root / "exp_002_unevaluated").mkdir()
    md = render_markdown(collect_experiments(root))
    for needle in ("Training configuration", "Validation and evaluation metrics", "Generation samples",
                   "exp_001_demo", "exp_002_unevaluated", "not evaluated"):
        assert needle in md, needle
    assert "No experiments found" in render_markdown([])


def test_bits_per_byte_is_derived_from_loss_and_dataset_bytes_per_token(engine, tiny_run, small_sets):
    import math
    res = _suite(engine, tiny_run, small_sets)
    manifest = json.loads((Path(tiny_run["exp"].data.out_dir) / "manifest.json").read_text())
    for split in ("val", "test"):
        bpt = manifest["splits"][split]["bytes_per_token"]
        assert res["loss"][split]["bits_per_byte"] == pytest.approx(res["loss"][split]["loss"] / math.log(2) / bpt)


# ── regressions found by the independent review ──────────────────────────────

@pytest.mark.parametrize("path,value,fragment", [
    ("eval_sets.dataset_fingerprint", "0" * 64, "held-out datasets"),
    ("model.tokenizer_fingerprint", "0" * 64, "tokenizers"),
    ("settings.max_new_tokens", 7, "generation lengths"),
    ("settings.max_windows", 3, "evaluation windows"),
])
def test_regression_refuses_results_from_a_different_dataset_tokenizer_or_settings(engine, tiny_run, small_sets, path, value, fragment):
    from aarav_lm.evaluation.regression import get_path
    base = _suite(engine, tiny_run, small_sets)
    other = copy.deepcopy(base)
    node = other
    *parents, leaf = path.split(".")
    for part in parents:
        node = node[part]
    assert node[leaf] != value
    node[leaf] = value
    ok, results = compare_to_baseline(other, base)
    assert not ok and fragment in results[0].note and get_path(base, path) != value


def test_dataset_fingerprint_ignores_build_timing_but_tracks_content(tiny_run, tmp_path):
    from aarav_lm.evaluation.suite import dataset_fingerprint
    data_dir = Path(tiny_run["exp"].data.out_dir)
    a = dataset_fingerprint(data_dir)
    copy_dir = tmp_path / "d"
    shutil.copytree(data_dir, copy_dir)
    m = json.loads((copy_dir / "manifest.json").read_text())
    m["build_seconds"] = 9999.0
    (copy_dir / "manifest.json").write_text(json.dumps(m))
    assert dataset_fingerprint(copy_dir) == a                          # a rebuild's different timing must not matter
    m["splits"]["val"]["tokens"] += 1
    (copy_dir / "manifest.json").write_text(json.dumps(m))
    assert dataset_fingerprint(copy_dir) != a
    assert dataset_fingerprint(tmp_path / "missing") is None


def test_stale_wordlist_cache_is_rebuilt(engine, tiny_run, tmp_path):
    from aarav_lm.evaluation.suite import load_wordlist
    data_dir = tmp_path / "d"
    shutil.copytree(tiny_run["exp"].data.out_dir, data_dir)
    real = load_wordlist(data_dir, engine)
    cache = data_dir / "wordlist.json"
    d = json.loads(cache.read_text())
    d["dataset_fingerprint"], d["words"] = "stale", ["bogus"]
    cache.write_text(json.dumps(d))
    assert load_wordlist(data_dir, engine) == real != {"bogus"}


import shutil                                                            # noqa: E402


def test_natural_end_rate_counts_chat_stops_as_well_as_eot():
    samples = [
        {"text": "a b.", "tokens": [1, 2], "finish_reason": "eot", "entity": None, "prompt": "p"},
        {"text": "c d.", "tokens": [3, 4], "finish_reason": "stop", "entity": None, "prompt": "p"},
        {"text": "e f g", "tokens": [5, 6, 7], "finish_reason": "length", "entity": None, "prompt": "p"},
    ]
    r = M.summarize_generations(samples, {"a", "b", "c", "d", "e", "f", "g"})
    assert r["natural_end_rate"] == pytest.approx(2 / 3) and r["eot_rate"] == pytest.approx(1 / 3)


# ── held-out loss with a standard error ──────────────────────────────────────

def _masked_eval_batches(n_batches=3, rows=4, width=16, vocab=500, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        x = torch.randint(0, vocab, (rows, width), generator=g)
        y = torch.randint(0, vocab, (rows, width), generator=g)
        for r in range(rows):
            y[r, : int(torch.randint(0, width - 1, (1,), generator=g))] = -1        # a different mask per row
        out.append((x, y))
    return out


def test_heldout_rows_agree_exactly_with_the_existing_loss_and_accuracy_functions():
    import torch
    from aarav_lm.config.config import cfg_5m
    from aarav_lm.evaluation.loss import evaluate_loss, heldout_rows, next_token_accuracy, summarize_rows
    from aarav_lm.model.core import AaravLM
    torch.manual_seed(0)
    model = AaravLM(cfg_5m(arch="modern", vocab_size=500, block_size=16))
    batches = _masked_eval_batches()
    dev = torch.device("cpu")
    old, acc = evaluate_loss(model, batches, dev), next_token_accuracy(model, batches, dev)
    new = summarize_rows(heldout_rows(model, batches, dev))
    assert new["loss"] == pytest.approx(old["loss"], rel=1e-6) and new["tokens"] == old["tokens"]
    assert new["perplexity"] == pytest.approx(old["perplexity"], rel=1e-6)
    assert new["top1"] == pytest.approx(acc["top1"]) and new["top5"] == pytest.approx(acc["top5"])
    assert new["rows"] == 12 and new["stderr"] > 0


def test_stderr_is_reproducible_and_shrinks_with_more_rows():
    import numpy as np
    from aarav_lm.evaluation.loss import summarize_rows
    rng = np.random.default_rng(1)

    def rows(n):
        c = rng.integers(20, 60, n)
        return {"sum": c * rng.normal(1.3, 0.4, n).clip(0.2), "count": c, "hit1": c // 2, "hit5": c // 2}
    small, big = rows(50), rows(2000)
    assert summarize_rows(small)["stderr"] == summarize_rows(small)["stderr"]            # seeded bootstrap
    assert summarize_rows(big)["stderr"] < summarize_rows(small)["stderr"] / 3           # ~ 1/sqrt(n)


def test_stderr_matches_the_analytic_value_for_equal_weight_rows():
    import numpy as np
    from aarav_lm.evaluation.loss import summarize_rows
    rng = np.random.default_rng(2)
    per_row = rng.normal(1.5, 0.3, 1000)
    rows = {"sum": per_row * 40, "count": np.full(1000, 40), "hit1": np.zeros(1000, int), "hit5": np.zeros(1000, int)}
    got = summarize_rows(rows, n_boot=1000)["stderr"]
    assert got == pytest.approx(per_row.std(ddof=1) / np.sqrt(1000), rel=0.15)


def test_the_suite_scores_the_whole_split_by_default_and_says_so(engine, tiny_run, small_sets):
    full = run_suite(engine, tiny_run["exp"].data.out_dir, small_sets[0], small_sets[1], max_new_tokens=8,
                     log=lambda _: None)
    part = run_suite(engine, tiny_run["exp"].data.out_dir, small_sets[0], small_sets[1], max_new_tokens=8,
                     max_windows=3, log=lambda _: None)
    for split in ("val", "test"):
        assert full["loss"][split]["complete"] is True and part["loss"][split]["complete"] is False
        assert part["loss"][split]["windows"] <= 16 < full["loss"][split]["windows"]      # subsets round up to whole batches of 16
        assert full["loss"][split]["stderr"] >= 0
    assert full["settings"]["max_windows"] is None


def test_instruction_eval_reports_chance_lift_effective_n_and_per_prompt(engine, instruct_file):
    from aarav_lm.evaluation.instruct import run_instruction_eval
    rows = [json.loads(l) for l in instruct_file.read_text().splitlines()]
    dup = dict(rows[1], id="b2")                                   # the same "short" prompt again: 2 rows, 1 distinct string
    dup2 = dict(rows[1], id="b3")
    res = run_instruction_eval(engine, rows + [dup, dup2], max_new_tokens=10,
                               chance={"topic": 0.1, "short": 0.5, "name": 0.0})
    for mode in res.values():
        assert mode["prompts"] == 6 and mode["scored"] == 5
        short = mode["by_task"]["short"]
        assert short["n"] == 3 and short["n_distinct"] == 1         # three rows, one distinct request
        assert short["chance"] == 0.5 and short["lift"] == pytest.approx(short["rate"] - 0.5, abs=1e-4)
        assert 0 <= short["ci95_clustered"][0] <= short["ci95_clustered"][1] <= 1
        assert mode["pass_rate"]["n_distinct"] == 3
        assert mode["pass_rate"]["chance"] == pytest.approx((0.1 * 1 + 0.5 * 3 + 0.0 * 1) / 5, abs=1e-4)
        pp = mode["per_prompt"]
        assert [r["id"] for r in pp] == ["a", "b", "c", "b2", "b3"]  # the check-less prompt is not scored
        assert all(set(r) >= {"id", "kind", "prompt", "passed", "checks", "words"} for r in pp)
        assert sum(r["passed"] for r in pp) == mode["pass_rate"]["pass"]
    # greedy decoding is deterministic, so identical prompts must have identical outcomes
    g = [r["passed"] for r in res["greedy"]["per_prompt"] if r["prompt"] == rows[1]["prompt"]]
    assert len(set(g)) == 1
    plain = run_instruction_eval(engine, rows, max_new_tokens=10)
    assert "chance" not in plain["greedy"]["pass_rate"] and "n_distinct" in plain["greedy"]["pass_rate"]
