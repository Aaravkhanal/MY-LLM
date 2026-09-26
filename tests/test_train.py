import json
import os
import shutil
import signal

import pytest
from aarav_lm.config import ExperimentConfig
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


def test_eval_batch_size_is_independent_of_the_training_micro_batch(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=4, batch_size=1, grad_accum=8, eval_batch_size=3, eval_iters=2)
    t = Trainer(exp, tmp_path / "run", log=quiet)
    t.run()
    assert [x.shape[0] for x, _ in t.val_batches] == [3, 3]           # evaluation batches of 3 while training used 1
    d = ExperimentConfig.from_dict(exp.to_dict()).to_dict()
    d["train"]["eval_batch_size"] = -1
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(d)


# ── token-weighted gradient accumulation ─────────────────────────────────────

def _accum_trainer(model, batches):
    import types
    from contextlib import nullcontext
    t = Trainer.__new__(Trainer)
    it = iter(batches)
    t.stream = types.SimpleNamespace(next_batch=lambda: next(it))
    t.cfg = types.SimpleNamespace(grad_accum=len(batches))
    t.model, t.device, t.ctx, t.step = model, torch.device("cpu"), nullcontext, 0
    return t


def _masked_batches(counts, seed=0, width=16, vocab=500):
    """Micro-batches whose loss masks leave a different number of scored targets in each."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for n in counts:
        x = torch.randint(0, vocab, (2, width), generator=g)
        y = torch.randint(0, vocab, (2, width), generator=g)
        y.view(-1)[n:] = -1                                   # exactly n scored targets
        out.append((x, y))
    return out


def _grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def test_accumulation_equals_one_big_batch_when_masks_differ():
    from aarav_lm.config.config import cfg_5m
    from aarav_lm.model.core import AaravLM
    torch.manual_seed(0)
    model = AaravLM(cfg_5m(arch="modern", vocab_size=500, block_size=16)).eval()   # eval: no dropout noise
    batches = _masked_batches([30, 4, 17])                    # very unequal scored-token counts

    loss, _ = _accum_trainer(model, batches)._accumulate()
    got = _grads(model).clone()
    model.zero_grad()
    _, big = model(torch.cat([x for x, _ in batches]), targets=torch.cat([y for _, y in batches]))
    big.backward()                                            # mean over ALL scored tokens
    want = _grads(model).clone()

    assert torch.allclose(got, want, atol=1e-6, rtol=1e-4)
    assert abs(loss.item() - big.item()) < 1e-5

    # the naive average of micro-batch means (the previous behaviour) is a different gradient here
    model.zero_grad()
    for x, y in batches:
        (model(x, targets=y)[1] / len(batches)).backward()
    assert not torch.allclose(_grads(model), want, atol=1e-6, rtol=1e-4)


def test_accumulation_is_unchanged_without_masking():
    """Equal counts (all pretraining) must reproduce loss/grad_accum exactly, so old runs stay comparable."""
    from aarav_lm.config.config import cfg_5m
    from aarav_lm.model.core import AaravLM
    torch.manual_seed(1)
    model = AaravLM(cfg_5m(arch="modern", vocab_size=500, block_size=16)).eval()
    batches = _masked_batches([32, 32, 32])
    _accum_trainer(model, batches)._accumulate()
    got = _grads(model).clone()
    model.zero_grad()
    for x, y in batches:
        (model(x, targets=y)[1] / 3).backward()
    assert torch.equal(got, _grads(model))


def test_a_step_with_every_target_masked_is_an_error_not_nan():
    from aarav_lm.config.config import cfg_5m
    from aarav_lm.model.core import AaravLM
    model = AaravLM(cfg_5m(arch="modern", vocab_size=500, block_size=16))
    with pytest.raises(ValueError, match="masked"):
        _accum_trainer(model, _masked_batches([0, 0]))._accumulate()


# ── continued pretraining ────────────────────────────────────────────────────

def test_continued_pretraining_starts_from_the_given_weights_and_only_the_weights(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=20, eval_interval=20)
    first = Trainer(exp, tmp_path / "a", log=quiet).run()
    fresh = tiny_experiment(tmp_path, max_steps=4, eval_interval=4)
    Trainer(fresh, tmp_path / "fresh", log=quiet).run()

    cont = tiny_experiment(tmp_path, max_steps=4, eval_interval=4)
    cont.init_from = str(tmp_path / "a" / "checkpoints" / "final.pt")
    summary = Trainer(cont, tmp_path / "b", log=quiet).run()

    def step0(run):
        evals = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if '"eval"' in line]
        return next(e for e in evals if e["step"] == 0)["val_loss"]
    assert step0(tmp_path / "b") == pytest.approx(first["last_eval"]["val_loss"], rel=1e-5)   # identical weights, identical val batches
    assert step0(tmp_path / "fresh") > step0(tmp_path / "b") + 1.0                           # a random init is far worse
    assert summary["steps"] == 4 and summary["tokens_seen"] == 4 * 4 * 2 * 32                # counters start from zero


def test_continued_pretraining_rejects_a_different_architecture_or_tokenizer(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=4, eval_interval=4)
    Trainer(exp, tmp_path / "a", log=quiet).run()
    other = tiny_experiment(tmp_path, max_steps=4, eval_interval=4)
    other.model_overrides = {**other.model_overrides, "n_layer": 3}
    other.init_from = str(tmp_path / "a" / "checkpoints" / "final.pt")
    with pytest.raises(RuntimeError, match="different architecture"):
        Trainer(other, tmp_path / "b", log=quiet).run()
