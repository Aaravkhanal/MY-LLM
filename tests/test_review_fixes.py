"""Regression tests for problems found by the independent code review, plus tests strengthened
because the reviewer found them vacuous."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from aarav_lm.chat.session import ChatSession, SessionFileError
from aarav_lm.config import ExperimentConfig
from aarav_lm.data.dataset import BatchStream, TokenDataset
from aarav_lm.data.shards import Shard, ShardWriter
from aarav_lm.data.sft import validate_conversation
from aarav_lm.inference.engine import GenerationConfig
from aarav_lm.train import checkpoint as ck
from aarav_lm.train.trainer import Trainer
from helpers import tiny_experiment

quiet = lambda _l: None                                             # noqa: E731


# ── trainer ──────────────────────────────────────────────────────────────────

def test_resume_that_changes_the_config_is_recorded_not_silently_lost(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=8)
    run = tmp_path / "run"
    Trainer(exp, run, log=quiet).run()
    env_before = (run / "env.json").read_text()
    longer = ExperimentConfig.from_dict(exp.to_dict())
    longer.train.max_steps = 12                                     # extends training (and moves the cosine schedule)
    Trainer(longer, run, log=quiet).run(resume=True)
    assert yaml.safe_load((run / "config.yaml").read_text())["train"]["max_steps"] == 12     # config == what actually ran
    archived = list(run.glob("config.before_resume_at_step*.yaml"))
    assert len(archived) == 1 and yaml.safe_load(archived[0].read_text())["train"]["max_steps"] == 8
    assert (run / "env.json").read_text() == env_before             # original environment/commit preserved
    assert list(run.glob("env.resume_at_step*.json"))


def test_unchanged_resume_does_not_churn_the_config(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=8)
    run = tmp_path / "run"
    Trainer(exp, run, log=quiet).run()
    Trainer(exp, run, log=quiet).run(resume=True)
    assert not list(run.glob("config.before_resume*"))


def test_non_finite_weights_are_never_checkpointed(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=8)
    t = Trainer(exp, tmp_path / "run", log=quiet)
    t.run()
    before = sorted(p.name for p in (tmp_path / "run/checkpoints").iterdir())
    with torch.no_grad():
        t.model.blocks[0].attn.q.weight[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite weights"):
        t.save_checkpoint()
    assert sorted(p.name for p in (tmp_path / "run/checkpoints").iterdir()) == before      # last good files untouched


def test_resume_refuses_a_changed_dataset(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=8)
    full = tmp_path / "full"
    Trainer(exp, full, log=quiet).run()
    crashed = tmp_path / "crashed"
    (crashed / "checkpoints").mkdir(parents=True)
    shutil.copy(full / "config.yaml", crashed / "config.yaml")
    state = torch.load(full / "checkpoints/step_00000004.pt", weights_only=True)
    state["train_state"]["data_id"][0] += 1000                       # as if the shards had been rebuilt
    torch.save(state, crashed / "checkpoints/step_00000004.pt")
    with pytest.raises(ck.CheckpointError, match="dataset changed"):
        Trainer(exp, crashed, log=quiet).run(resume=True)


def test_stream_rejects_a_position_outside_the_dataset(tmp_path):
    with ShardWriter(tmp_path / "s.bin", 300) as w:
        w.append(np.random.default_rng(0).integers(0, 300, 2000))
    ds = TokenDataset(Shard.open(tmp_path / "s.bin"), 32)
    s = BatchStream(ds, 4, seed=1)
    with pytest.raises(ValueError, match="outside the dataset"):
        s.load_state_dict({"seed": 1, "epoch": 0, "pos": 10 ** 6})


@pytest.mark.parametrize("field,value", [("eval_interval", 0), ("ckpt_interval", 0), ("log_interval", 0), ("max_steps", 0),
                                          ("batch_size", 0), ("grad_accum", 0), ("eval_iters", 0), ("precision", "fp16"),
                                          ("min_lr_ratio", 2.0), ("grad_clip", 0.0)])
def test_invalid_training_settings_fail_up_front_not_mid_run(tmp_path, field, value):
    exp = tiny_experiment(tmp_path)
    setattr(exp.train, field, value)
    with pytest.raises(ValueError):
        Trainer(exp, tmp_path / "r", log=quiet)
    d = ExperimentConfig().to_dict()
    d["train"][field] = value
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(d)


def test_early_stopping_fires_at_exactly_the_patience_boundary(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=40, lr=0.0, eval_interval=4, early_stop_patience=2)
    s = Trainer(exp, tmp_path / "es", log=quiet).run()
    assert s["status"] == "early_stopped" and s["steps"] == 12       # evals at 4 (improves), 8 and 12 (don't)
    evals = [json.loads(l) for l in (tmp_path / "es/metrics.jsonl").read_text().splitlines() if '"eval"' in l]
    assert [e.get("improved") for e in evals[1:]] == [True, False, False]


def test_resume_restores_optimizer_and_bookkeeping_not_just_weights(tmp_path):
    exp = tiny_experiment(tmp_path, max_steps=12)
    full, crashed = tmp_path / "full", tmp_path / "crashed"
    Trainer(exp, full, log=quiet).run()
    (crashed / "checkpoints").mkdir(parents=True)
    shutil.copy(full / "config.yaml", crashed / "config.yaml")
    shutil.copy(full / "checkpoints/step_00000004.pt", crashed / "checkpoints")
    Trainer(exp, crashed, log=quiet).run(resume=True)
    a = torch.load(full / "checkpoints/step_00000012.pt", weights_only=True)
    b = torch.load(crashed / "checkpoints/step_00000012.pt", weights_only=True)
    for key in ("best_val", "evals_without_improvement", "tokens_seen", "step"):
        assert a["train_state"][key] == b["train_state"][key], key
    for pa, pb in zip(a["optimizer"]["state"].values(), b["optimizer"]["state"].values()):
        assert all(torch.equal(pa[k], pb[k]) for k in pa)            # Adam moments identical
    assert a["rng"]["python"] == b["rng"]["python"]


# ── data ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("role", [["user"], {"a": 1}, 5, None])
def test_unhashable_or_non_string_roles_are_rejected_not_raised(role):
    raw = {"messages": [{"role": role, "content": "x"}, {"role": "assistant", "content": "y"}]}
    assert validate_conversation(raw) == (None, "bad_message")


def test_every_window_is_visited_exactly_once_per_epoch(tmp_path):
    with ShardWriter(tmp_path / "s.bin", 60000) as w:
        w.append(np.random.default_rng(0).integers(0, 60000, 4000))     # big vocab: windows are unique
    ds = TokenDataset(Shard.open(tmp_path / "s.bin"), 32)
    s = BatchStream(ds, 1, seed=3)
    seen = [s.next_batch()[0][0].numpy().tobytes() for _ in range(ds.n_windows)]
    assert s.epoch == 1 and s.pos == 0
    assert len(set(seen)) == ds.n_windows == len(seen)                  # each window exactly once, none repeated or skipped


def test_fixed_eval_batches_span_the_whole_split(tmp_path):
    with ShardWriter(tmp_path / "s.bin", 60000) as w:
        w.append(np.random.default_rng(1).integers(0, 60000, 8000))
    ds = TokenDataset(Shard.open(tmp_path / "s.bin"), 32)
    batches = ds.fixed_eval_batches(4, 5)
    xs = torch.cat([x for x, _ in batches])
    assert torch.equal(xs[0], torch.from_numpy(ds.window(0)[0]))                     # starts at the beginning
    assert torch.equal(xs[-1], torch.from_numpy(ds.window(ds.n_windows - 1)[0]))     # reaches the end
    assert len({tuple(r.tolist()) for r in xs}) == len(xs)                           # no duplicates


# ── chat session files ───────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", ["[]", "null", "3", '"x"', "true"])
def test_non_object_session_files_fail_cleanly(engine, tmp_path, payload):
    (tmp_path / "s.json").write_text(payload)
    s = ChatSession(engine, system_prompt="keep", mode="chat")
    with pytest.raises(SessionFileError):
        s.load(tmp_path / "s.json")
    assert s.system_prompt == "keep"


@pytest.mark.parametrize("field,value", [("stop_sequences", "abc"), ("top_k", 2.5), ("temperature", "hot"), ("max_new_tokens", True),
                                          ("stop_ids", ["x"]), ("stop_at_eot", 1), ("bogus", 1), ("top_p", [0.5])])
def test_session_generation_settings_are_type_checked(engine, tmp_path, field, value):
    s = ChatSession(engine, gen_cfg=GenerationConfig(), mode="chat")
    data = json.loads(s.save(tmp_path / "ok.json").read_text())
    data["generation"][field] = value
    (tmp_path / "bad.json").write_text(json.dumps(data))
    with pytest.raises(SessionFileError):
        ChatSession(engine, mode="chat").load(tmp_path / "bad.json")


def test_a_good_session_file_still_loads_after_the_stricter_checks(engine, tmp_path):
    s = ChatSession(engine, system_prompt="x", gen_cfg=GenerationConfig(top_k=None, seed=4, stop_sequences=("a",)), mode="chat")
    "".join(s.send("hello"))
    fresh = ChatSession(engine, mode="chat")
    fresh.load(s.save(tmp_path / "s.json"))
    assert fresh.gen_cfg == s.gen_cfg
