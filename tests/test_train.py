import json
import os
import shutil
import signal

import pytest
import torch

from aarav_lm.model.core import AaravLM
from aarav_lm.train import checkpoint as ck
from aarav_lm.train.schedule import lr_at
from aarav_lm.train.trainer import Trainer, next_experiment_dir
from helpers import tiny_experiment

quiet = lambda _line: None                                      # noqa: E731


def _params(path):
    ckpt = torch.load(path, weights_only=True)
    return ckpt["model"]


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ── schedule ─────────────────────────────────────────────────────────────────

def test_schedule_warmup_peak_and_floor():
    kw = dict(lr=1e-3, min_lr_ratio=0.1, warmup_steps=10, max_steps=100)
    assert lr_at(0, **kw) == pytest.approx(1e-4)
    assert lr_at(9, **kw) == pytest.approx(1e-3)
    assert lr_at(10, **kw) == pytest.approx(1e-3)
    assert lr_at(100, **kw) == pytest.approx(1e-4)
    assert lr_at(500, **kw) == pytest.approx(1e-4)
    seq = [lr_at(s, **kw) for s in range(10, 100)]
    assert all(a >= b for a, b in zip(seq, seq[1:]))            # monotone decay after warmup


def test_experiment_dirs_are_numbered(tmp_path):
    a = next_experiment_dir(tmp_path, "x")
    b = next_experiment_dir(tmp_path, "y")
    assert (a.name, b.name) == ("exp_001_x", "exp_002_y")


# ── training behaviour ───────────────────────────────────────────────────────

def test_training_learns_and_writes_a_complete_experiment_record(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=40, eval_interval=20)
    run = tmp_path / "run"
    summary = Trainer(exp, run, log=quiet).run()

    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    evals = [m for m in metrics if m["type"] == "eval"]
    assert evals[-1]["val_loss"] < evals[0]["val_loss"] - 0.5          # real learning, not noise
    assert summary["status"] == "completed" and summary["steps"] == 40
    assert summary["tokens_seen"] == 40 * 4 * 2 * 32
    for name in ("config.yaml", "env.json", "metrics.jsonl", "train.log", "summary.json"):
        assert (run / name).exists(), name
    for name in ("best.pt", "final.pt", "step_00000040.pt"):
        assert (run / "checkpoints" / name).exists(), name
    assert summary["best_val_loss"] == pytest.approx(min(e["val_loss"] for e in evals))


def test_same_seed_gives_bit_identical_training(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=8)
    Trainer(exp, tmp_path / "a", log=quiet).run()
    Trainer(exp, tmp_path / "b", log=quiet).run()
    assert _same(_params(tmp_path / "a/checkpoints/final.pt"), _params(tmp_path / "b/checkpoints/final.pt"))


@pytest.mark.parametrize("crash_after", [4, 8])
def test_resume_after_crash_matches_uninterrupted_run(tmp_path, crash_after):
    """Simulate a crash by resuming from an intermediate checkpoint: results must be identical."""
    exp = tiny_experiment(tmp_path, max_steps=12)
    full = tmp_path / "full"
    Trainer(exp, full, log=quiet).run()

    crashed = tmp_path / "crashed"
    (crashed / "checkpoints").mkdir(parents=True)
    shutil.copy(full / "config.yaml", crashed / "config.yaml")
    shutil.copy(full / "checkpoints" / f"step_{crash_after:08d}.pt", crashed / "checkpoints")
    summary = Trainer(exp, crashed, log=quiet).run(resume=True)

    assert summary["resumed_from"] == f"step_{crash_after:08d}.pt" and summary["steps"] == 12
    assert _same(_params(full / "checkpoints/final.pt"), _params(crashed / "checkpoints/final.pt"))
    a = torch.load(full / "checkpoints/step_00000012.pt", weights_only=True)
    b = torch.load(crashed / "checkpoints/step_00000012.pt", weights_only=True)
    assert a["stream_state"] == b["stream_state"]                     # data position identical too
    assert a["train_state"]["tokens_seen"] == b["train_state"]["tokens_seen"]


def test_ctrl_c_saves_a_checkpoint_and_resume_completes_identically(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=12)
    Trainer(exp, tmp_path / "full", log=quiet).run()

    fired = []

    def log_and_interrupt(line):
        if "step      6/12" in line and not fired:                     # a train log line at step 6
            fired.append(1)
            os.kill(os.getpid(), signal.SIGINT)

    exp2 = tiny_experiment(tmp_path, max_steps=12, log_interval=2)
    run = tmp_path / "interrupted"
    s1 = Trainer(exp2, run, log=log_and_interrupt).run()
    assert fired and s1["status"] == "interrupted" and s1["steps"] == 6
    assert (run / "checkpoints" / "step_00000006.pt").exists()

    s2 = Trainer(exp2, run, log=quiet).run(resume=True)
    assert s2["status"] == "completed" and s2["steps"] == 12
    assert _same(_params(tmp_path / "full/checkpoints/final.pt"), _params(run / "checkpoints/final.pt"))


def test_early_stopping_triggers_when_validation_stops_improving(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=40, lr=0.0, eval_interval=4, early_stop_patience=2)
    s = Trainer(exp, tmp_path / "es", log=quiet).run()
    assert s["status"] == "early_stopped" and s["steps"] < 40


def test_rolling_checkpoints_are_pruned(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=16, keep_last=2, ckpt_interval=4)
    Trainer(exp, tmp_path / "p", log=quiet).run()
    steps = [s for s, _ in ck.list_step_checkpoints(tmp_path / "p" / "checkpoints")]
    assert steps == [12, 16]


def test_resume_with_nothing_to_resume_is_a_clear_error(tmp_path):
    exp = tiny_experiment(tmp_path)
    with pytest.raises(ck.CheckpointError, match="nothing to resume"):
        Trainer(exp, tmp_path / "empty", log=quiet).run(resume=True)


def test_resume_rejects_a_different_model_config(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=4)
    Trainer(exp, tmp_path / "r", log=quiet).run()
    exp.model_overrides = {**exp.model_overrides, "n_layer": 3}
    with pytest.raises(ck.CheckpointError, match="differs"):
        Trainer(exp, tmp_path / "r", log=quiet).run(resume=True)


# ── checkpoints ──────────────────────────────────────────────────────────────

def test_best_checkpoint_loads_into_an_identical_model(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=8)
    t = Trainer(exp, tmp_path / "m", log=quiet)
    t.run()
    model, raw = ck.load_model(tmp_path / "m/checkpoints/final.pt", "cpu", t.tok.fingerprint)
    assert raw["kind"] == "model" and not model.training
    x = torch.randint(0, 300, (2, 16))
    with torch.no_grad():
        assert torch.allclose(model(x)[0], t.model.cpu()(x)[0], atol=1e-6)


def test_loading_with_wrong_tokenizer_fingerprint_fails(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=4)
    Trainer(exp, tmp_path / "w", log=quiet).run()
    with pytest.raises(ck.CheckpointError, match="tokenizer"):
        ck.load_model(tmp_path / "w/checkpoints/final.pt", "cpu", "0" * 64)


def test_atomic_save_leaves_previous_file_intact_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "c.pt"
    ck.atomic_save({"v": torch.ones(3)}, path)
    real = torch.save

    def boom(obj, f):
        real({"partial": 1}, f)
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError):
        ck.atomic_save({"v": torch.zeros(3)}, path)
    monkeypatch.undo()
    assert torch.equal(torch.load(path, weights_only=True)["v"], torch.ones(3))
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_loads_without_executing_pickled_code(tmp_path):
    import pickle

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned > " + str(tmp_path / "pwned"),))

    bad = tmp_path / "bad.pt"
    torch.save({"format_version": 1, "kind": "model", "x": Evil()}, bad)
    with pytest.raises(ck.CheckpointError):
        ck.load_model(bad)
    assert not (tmp_path / "pwned").exists()


def test_rng_state_round_trip():
    torch.manual_seed(5)
    state = ck.capture_rng_state()
    a = torch.rand(4)
    ck.restore_rng_state(state)
    assert torch.equal(a, torch.rand(4))
