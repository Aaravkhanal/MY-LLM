"""Training diagnostics: they must catch real problems, stay quiet on normal noise, and never alter training."""

import json
import math

import numpy as np
import pytest
import torch

from aarav_lm.config.config import cfg_5m
from aarav_lm.model.core import AaravLM
from aarav_lm.train.health import (
    OverfitDetector, SpikeDetector, gradient_health, group_of, health_flags,
)
from aarav_lm.train.trainer import Trainer
from helpers import tiny_experiment

quiet = lambda _l: None                                             # noqa: E731


# ── spike detector ───────────────────────────────────────────────────────────

def _loss_curve(n, seed=0):
    """A realistic-looking training loss: slow decay plus per-step noise (sd ~2% of the value)."""
    rng = np.random.default_rng(seed)
    return [1.4 + 2.0 * math.exp(-i / 60) + rng.normal(0, 0.03) for i in range(n)]


def test_normal_noisy_decaying_loss_raises_no_false_alarms():
    det = SpikeDetector("loss")
    assert [i for i, v in enumerate(_loss_curve(3000)) if det.update(v)] == []


def test_a_sudden_loss_jump_is_flagged_with_its_context():
    det = SpikeDetector("loss")
    curve = _loss_curve(200)
    hits = [det.update(v) for v in curve]
    assert not any(hits)
    spike = det.update(curve[-1] * 1.6)
    assert spike and spike["kind"] == "loss" and spike["value"] > spike["threshold"] > spike["median"]


def test_a_drop_in_loss_is_never_a_spike():
    det = SpikeDetector("loss")
    for v in _loss_curve(100):
        det.update(v)
    assert det.update(0.1) is None


def test_detector_is_silent_until_it_has_history_and_ignores_non_finite_values():
    det = SpikeDetector("loss", min_history=20)
    assert all(det.update(1.0) is None for _ in range(20))
    assert det.update(float("nan")) is None and det.update(float("inf")) is None
    assert det.update(5.0) is not None                          # history is intact after the NaNs


# ── gradient health ──────────────────────────────────────────────────────────

def _model_with_grads(seed=0):
    torch.manual_seed(seed)
    m = AaravLM(cfg_5m(arch="modern", vocab_size=300, block_size=16))
    x = torch.randint(0, 300, (4, 16))
    m(x, targets=torch.randint(0, 300, (4, 16)))[1].backward()
    return m


def test_parameters_are_grouped_per_layer_component():
    assert group_of("blocks.3.attn.q.weight") == "blocks.3.attn"
    assert group_of("blocks.0.mlp.gate.weight") == "blocks.0.mlp"
    assert group_of("blocks.2.norm1.weight") == "blocks.2.norm"
    assert group_of("token_emb.weight") == "token_emb" and group_of("final_norm.weight") == "final_norm"


def test_gradient_health_reports_finite_positive_norms_for_every_group():
    m = _model_with_grads()
    stats = gradient_health(m)
    assert "token_emb" in stats and any(k.endswith(".attn") for k in stats) and any(k.endswith(".mlp") for k in stats)
    for s in stats.values():
        assert s["grad"] > 0 and s["param"] > 0 and math.isfinite(s["ratio"])
    assert health_flags(stats) == []                            # a healthy fresh model raises nothing


def test_gradient_health_matches_a_direct_computation():
    m = _model_with_grads()
    stats = gradient_health(m)
    direct = math.sqrt(sum(float(p.grad.norm()) ** 2 for n, p in m.named_parameters() if n.startswith("blocks.0.attn")))
    assert stats["blocks.0.attn"]["grad"] == pytest.approx(direct, rel=1e-5)


def test_exploding_vanishing_and_non_finite_groups_are_flagged_and_gradients_are_untouched():
    m = _model_with_grads()
    for n, p in m.named_parameters():
        if n.startswith("blocks.0.attn"):
            p.grad.mul_(1e4)
        if n.startswith("blocks.1.mlp"):
            p.grad.mul_(1e-6)
        if n.startswith("final_norm"):
            p.grad.fill_(float("nan"))
    before = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]).clone()
    flags = " | ".join(health_flags(gradient_health(m)))
    assert "blocks.0.attn" in flags and "the median group" in flags
    assert "blocks.1.mlp" in flags
    assert "final_norm: non-finite" in flags
    after = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    assert torch.equal(torch.nan_to_num(before), torch.nan_to_num(after))   # measuring must not modify gradients


def test_parameters_without_gradients_are_skipped():
    m = AaravLM(cfg_5m(arch="modern", vocab_size=300, block_size=16))
    assert gradient_health(m) == {}


# ── overfitting detector ─────────────────────────────────────────────────────

def test_overfitting_needs_val_up_and_train_down_repeatedly():
    det = OverfitDetector(patience=2)
    assert det.update(1.5, 1.60) is None
    assert det.update(1.4, 1.55) is None                        # both improving
    assert det.update(1.3, 1.57) is None                        # one rise is not enough
    hit = det.update(1.2, 1.60)                                 # second consecutive rise, train still falling
    assert hit and hit["gap"] == pytest.approx(0.40) and hit["val_rise"] == pytest.approx(0.05)


def test_rising_val_with_rising_train_is_not_overfitting():
    det = OverfitDetector(patience=2)
    for t, v in [(1.2, 1.3), (1.3, 1.4), (1.4, 1.5)]:
        assert det.update(t, v) is None


# ── inside a real training run ───────────────────────────────────────────────

def test_a_run_records_layer_health_and_reports_anomaly_counts(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=12, log_interval=4, eval_interval=6)
    run = tmp_path / "run"
    summary = Trainer(exp, run, log=quiet).run()
    recs = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    health = [r for r in recs if r["type"] == "health"]
    assert [h["step"] for h in health] == [4, 8, 12]
    g = health[0]["groups"]
    assert "token_emb" in g and all(len(v) == 2 and v[0] > 0 for v in g.values())
    assert isinstance(summary["anomalies"], dict)


def test_diagnostics_do_not_change_training(tmp_path):
    """Same seed, different log_interval => different health-measurement schedule, identical weights."""
    def final_weights(root, log_interval):
        root.mkdir()
        exp = tiny_experiment(root, max_steps=8, log_interval=log_interval, eval_interval=4)
        Trainer(exp, root / "run", log=quiet).run()
        sd = torch.load(root / "run" / "checkpoints" / "final.pt", weights_only=True)["model"]
        return sd
    a = final_weights(tmp_path / "a", 1)
    b = final_weights(tmp_path / "b", 4)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_plot_of_a_run_is_written_and_marks_anomalies(tmp_path):
    pytest.importorskip("matplotlib")
    from aarav_lm.train.plots import plot_run
    exp = tiny_experiment(tmp_path, max_steps=12, log_interval=4, eval_interval=4)
    run = tmp_path / "run"
    Trainer(exp, run, log=quiet).run()
    with open(run / "metrics.jsonl", "a", encoding="utf-8") as f:                 # an injected finding must be drawn
        f.write(json.dumps({"type": "anomaly", "step": 6, "anomaly": "loss_spike"}) + "\n")
    out = plot_run(run)
    assert out.name == "training_curves.png" and out.stat().st_size > 5_000
