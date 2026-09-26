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
    assert [m["id"] for m in d["models"]][0] == "base" and len(d["models"]) == 6 and len(d["kinds"]) == 12
    assert [s["id"] for s in d["sets"]] == ["final", "template"]
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


def test_the_home_page_keeps_its_interactive_parts():
    js = (WEB / "app.js").read_text()
    for part in ("function viewHome", "function explorer", "function pipeline", "function tabs", "/results.json", 'role: "tablist"',
                 "aria-expanded", "aria-pressed", "ArrowRight", "aria-live", "Not stored", "Honest limits"):
        assert part in js, part
    assert 'default "home"' not in js and '|| "home"' in js                       # the front page is the default route
    assert '"home"' in js and "needsAccount" in js
