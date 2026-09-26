"""The results chart is generated from the committed result files and must keep working."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_results_chart_regenerates_from_the_committed_result_files(tmp_path):
    pytest.importorskip("matplotlib")
    if not (ROOT / "experiments" / "exp_009_sft_small_instruct_v6" / "eval" / "best__instruct_v4_final.json").exists():
        pytest.skip("result files not present")
    spec = importlib.util.spec_from_file_location("plot_results", ROOT / "scripts" / "plot_results.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for exp in (e for _, e in mod.MODELS):
        for mode in ("greedy", "sampled"):
            d = mod.load(ROOT, exp, mode)
            assert 0 <= d["lo"] <= d["rate"] <= d["hi"] <= 100 and 5 < d["chance"] < 20
    import sys
    argv = sys.argv
    sys.argv = ["plot_results.py", "--out", str(tmp_path / "chart.png")]
    try:
        mod.main()
    finally:
        sys.argv = argv
    out = tmp_path / "chart.png"
    assert out.exists() and out.stat().st_size > 20_000 and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
