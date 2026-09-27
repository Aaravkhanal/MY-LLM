"""The public website: its data must come from the evaluation files, and its interactive parts must stay."""

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aarav_lm.api.app import create_app
from aarav_lm.api.auth import KeyStore

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "aarav_lm" / "web"

spec = importlib.util.spec_from_file_location("build_site_data", ROOT / "scripts" / "build_site_data.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_the_site_data_is_exactly_what_the_evaluation_files_say():
    """results.json is generated, never edited: rebuilding it from experiments/*/eval must reproduce the committed file."""
    if not (ROOT / "experiments" / "exp_009_sft_small_instruct_v6" / "eval").exists():
        pytest.skip("result files not present")
    assert json.loads((WEB / "results.json").read_text()) == builder.build()


def test_the_site_data_is_complete_and_sane():
    d = json.loads((WEB / "results.json").read_text())
    assert [m["id"] for m in d["models"]][0] == "base" and len(d["models"]) == 11 and len(d["kinds"]) == 12
    assert [s["id"] for s in d["sets"]] == ["final", "template", "dev"]
    for s in d["sets"]:
        for m in d["models"]:
            for mode in ("greedy", "sampled"):
                cells = s["results"][m["id"]][mode]
                assert set(cells) == {"all"} | {k["id"] for k in d["kinds"]}
                for c in cells.values():
                    assert 0 <= c["lo"] <= c["rate"] <= c["hi"] <= 100 and c["prompts"] >= 1 and 0 <= c["chance"] <= 100


def test_results_are_served_publicly_with_strict_headers_and_the_config_says_what_is_running(engine, tmp_path):
    store = KeyStore(tmp_path / "k.json"); store.create("x")
    c = TestClient(create_app(engine, store, model_name="exp_test"))
    r = c.get("/results.json")                                              # public: the front page needs no sign-in
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert "default-src 'none'" in r.headers["content-security-policy"] and r.headers["x-content-type-options"] == "nosniff"
    assert r.json()["models"][0]["id"] == "base"
    cfg = c.get("/app/config").json()
    mi = cfg["model_info"]
    assert set(mi) >= {"parameters", "context_length", "instruction_tuned", "batching"} and mi["parameters"] > 0
    assert "path" not in json.dumps(cfg).lower() and "checkpoint" not in json.dumps(cfg).lower()      # nothing about the filesystem


def test_the_landing_page_keeps_its_parts():
    js = (WEB / "js" / "pages" / "landing.js").read_text()
    for part in ("function hero", "function demoSection", "/demo/chat", "IntervalChart", "fillBuilt", "CodeTabs", "Limitations", "readDifference",
                 "/models/aarav-lm/evaluations", "/playground", "/docs"):
        assert part in js, part
    assert "aria-live" in js and "aria-busy" in js                               # the demo announces its progress to screen readers


def test_the_evaluation_page_separates_measured_uncertain_and_interpreted():
    js = (WEB / "js" / "pages" / "evaluations.js").read_text()
    for part in ("SegmentedControl", "Select", "IntervalChart", "DataTable", "vs_base", "Overlapping intervals neither prove nor rule out",
                 "Measured", "Uncertain", "Interpreted", "not established", "set=", "history.replaceState"):
        assert part in js or part.replace("=", "") in js, part


def test_the_site_data_carries_paired_comparisons_that_match_what_the_results_document_says():
    """The claims in docs/RESULTS.md about exp_012 are numbers in this file: if the evaluation or the document changes, one of them must too."""
    d = json.loads((WEB / "results.json").read_text())
    by = {s["id"]: s for s in d["sets"]}
    assert d["default_model"] == "exp013" and {m["id"] for m in d["models"]} >= {"base", "exp009", "exp012", "exp013", "exp014"}
    final = by["final"]["vs_base"]["exp012"]["sampled"]["ALL"]
    assert round(100 * final["diff"], 1) == 9.4 and round(100 * final["lo"], 1) == 0.3 and round(100 * final["hi"], 1) == 18.0
    dev = by["dev"]["vs_base"]["exp012"]["sampled"]["ALL"]
    assert round(100 * dev["diff"], 1) == 14.2 and round(100 * dev["lo"], 1) == 6.2 and round(100 * dev["hi"], 1) == 22.2
    assert by["final"]["vs_base"]["exp012"]["greedy"]["ALL"]["lo"] < 0 < by["final"]["vs_base"]["exp012"]["greedy"]["ALL"]["hi"]   # greedy: not established
    for s in d["sets"]:                                                     # every comparison is over the same prompts as the results
        assert s["prompts"] > 0 and 0 < s["scored"] <= s["prompts"] and len(s["sha256"]) == 64
        for m, modes in s["vs_base"].items():
            for mode, kinds in modes.items():
                assert set(kinds) == {"ALL"} | {k["id"] for k in d["kinds"]}
                assert all(c["lo"] <= c["diff"] <= c["hi"] and 0 < c["p"] <= 1 for c in kinds.values()), (s["id"], m, mode)
    assert d["decoding"]["greedy"]["temperature"] == 0 and d["decoding"]["sampled"]["temperature"] == 0.7


def test_the_site_data_carries_training_facts_read_from_the_run_folders():
    d = json.loads((WEB / "results.json").read_text())
    stages = {s["experiment"]: s for s in d["training"]["stages"]}
    assert stages["exp_010_small_long_tinystories"]["steps"] == 9000 and stages["exp_010_small_long_tinystories"]["tokens_seen"] == 147_456_000
    assert stages["exp_004_small_tinystories"]["params"] == 14_162_304
    full = d["training"]["full_split_loss"]
    assert full["exp_010_small_long_tinystories"]["val"]["loss"] < full["exp_004_small_tinystories"]["val"]["loss"]
    assert full["exp_004_small_tinystories"]["val"]["windows"] == full["exp_010_small_long_tinystories"]["val"]["windows"] == 2927


def test_the_seed_spread_is_computed_from_the_three_runs_and_matches_the_write_up():
    d = json.loads((WEB / "results.json").read_text())
    sd = d["seeds"]
    assert sd["epoch2"]["models"] == ["exp012", "exp013", "exp014"] and sd["models"] == sd["epoch2"]["models"]
    fin = sd["epoch2"]["sets"]["final"]["greedy"]
    assert fin["values"] == [45.1, 52.1, 52.8] and fin["mean"] == 50.0 and fin["sd"] == 4.2 and fin["range"] == 7.6      # docs/RESULTS.md quotes these
    assert sd["epoch2"]["sets"]["final"]["sampled"]["values"] == [44.1, 50.3, 44.1]
    assert sd["epoch2"]["kinds"]["long"]["greedy"] == [79.2, 16.7, 20.8] and sd["epoch2"]["kinds"]["short"]["greedy"] == [25.0, 87.5, 0.0]    # the length skills flip with the seed
    for group in ("epoch1", "epoch2"):
        for sid in ("final", "template", "dev"):
            for mode in ("greedy", "sampled"):
                x = sd[group]["sets"][sid][mode]
                assert len(x["values"]) == 3 and min(x["values"]) <= x["mean"] <= max(x["values"]) and x["sd"] >= 0 and abs(x["range"] - (max(x["values"]) - min(x["values"]))) < 0.11


def test_the_exp009_recipe_seed_spread_shows_a_stable_but_different_length_failure():
    """exp_009's recipe (one epoch), three seeds: never learns `long`, always keeps `short` — the opposite instability from exp_012's recipe."""
    d = json.loads((WEB / "results.json").read_text())
    e1 = d["seeds"]["epoch1"]
    assert e1["models"] == ["exp009", "exp015", "exp016"]
    fin = e1["sets"]["final"]["greedy"]
    assert fin["values"] == [46.5, 50.3, 46.9] and fin["mean"] == 47.9 and fin["sd"] == 2.1                              # docs/RESULTS.md quotes these
    assert e1["sets"]["template"]["greedy"]["sd"] == 0.7                                                                 # far tighter than exp_012's recipe (6.7)
    assert e1["kinds"]["long"]["greedy"] == [20.8, 8.3, 20.8] and e1["kinds"]["short"]["greedy"] == [75.0, 100.0, 100.0]
    e2 = d["seeds"]["epoch2"]
    # the two recipes' three-seed means are close relative to either recipe's own spread: no winner is established
    assert abs(e1["sets"]["final"]["greedy"]["mean"] - e2["sets"]["final"]["greedy"]["mean"]) < e2["sets"]["final"]["greedy"]["sd"] * 2
