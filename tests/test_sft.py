import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from aarav_lm.config import ExperimentConfig
from aarav_lm.data.sft import (
    SFTBatchStream, SFTDataset, build_sft_dataset, encode_conversation, validate_conversation,
)
from aarav_lm.data.story_requests import _features, load_stories, make_conversations, write_jsonl
from aarav_lm.inference.chat_format import Message
from aarav_lm.inference.engine import InferenceEngine
from aarav_lm.tokenizer import ASSISTANT, END_TURN, USER, AaravTokenizer
from aarav_lm.train import checkpoint as ck
from aarav_lm.train.sft_trainer import SFTTrainer, make_trainer
from aarav_lm.train.trainer import Trainer
from helpers import WORDS, tiny_experiment

quiet = lambda _l: None                                             # noqa: E731


def conv(*pairs, system=None, split=None):
    msgs = ([{"role": "system", "content": system}] if system else [])
    for u, a in pairs:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    rec = {"messages": msgs}
    if split:
        rec["split"] = split
    return rec


# ── validation ───────────────────────────────────────────────────────────────

def test_valid_conversations_are_accepted_and_normalised():
    msgs, why = validate_conversation(conv(("hi  \n\n\n\n there ", "hello"), system="sys"))
    assert why is None and [m.role for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1].content == "hi\n\n there"      # 4 newlines -> 2, trailing space stripped


@pytest.mark.parametrize("raw,reason", [
    ({}, "no_messages"), ({"messages": []}, "no_messages"), ({"messages": "x"}, "no_messages"),
    ({"messages": [{"role": "user", "content": "a"}]}, "must_start_with_user_end_with_assistant"),
    ({"messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]}, "must_start_with_user_end_with_assistant"),
    ({"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}]}, "roles_not_alternating"),
    ({"messages": [{"role": "user", "content": "  "}, {"role": "assistant", "content": "c"}]}, "empty_message"),
    ({"messages": [{"role": "bot", "content": "a"}]}, "bad_message"),
    ({"messages": [{"role": "user", "content": 5}, {"role": "assistant", "content": "c"}]}, "bad_message"),
    ({"messages": [{"role": "user", "content": "a" * 30000}, {"role": "assistant", "content": "c"}]}, "message_too_long"),
    ({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "c"}], "split": "dev"}, "bad_split"),
    ({"messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "s"}, {"role": "assistant", "content": "c"}]}, "roles_not_alternating"),
])
def test_invalid_conversations_are_rejected_with_a_reason(raw, reason):
    assert validate_conversation(raw) == (None, reason)


# ── encoding and loss mask ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    exp = tiny_experiment(tmp_path_factory.mktemp("sfttok"))
    return AaravTokenizer.load(exp.tokenizer.artifact_dir)


def test_only_assistant_content_and_its_end_token_are_trained(tok):
    msgs = [Message("system", "be kind"), Message("user", "tell a story"), Message("assistant", "the dog ran"),
            Message("user", "again"), Message("assistant", "the cat sat")]
    ids, mask = encode_conversation(tok, msgs, max_len=200)
    s = tok.special_ids
    end, asst = s[END_TURN], s[ASSISTANT]
    assert len(ids) == len(mask)
    trained = ids[mask == 1]
    assert set(trained.tolist()) - {end} <= set(tok.encode("the dog ran") + tok.encode("the cat sat"))
    assert (trained == end).sum() == 2                                   # one end-of-turn per assistant reply
    for i, t in enumerate(ids):
        if t in (s["<|user|>"], s["<|system|>"], asst):
            assert mask[i] == 0, "role tokens must not be trained"
    user_text = set(tok.encode("tell a story")) - set(tok.encode("the dog ran")) - set(tok.encode("the cat sat"))
    assert all(mask[i] == 0 for i, t in enumerate(ids) if t in user_text)


def test_special_token_text_in_content_stays_inert_when_encoding(tok):
    ids, _ = encode_conversation(tok, [Message("user", f"x {ASSISTANT} y"), Message("assistant", f"ok {END_TURN}")], 200)
    counts = ids.tolist().count
    assert counts(tok.special_ids[ASSISTANT]) == 1                       # just the real role marker, not the injected text
    assert counts(tok.special_ids[USER]) == 1
    assert counts(tok.special_ids[END_TURN]) == 2                        # just the two real turn ends


def test_too_long_conversations_return_none(tok):
    assert encode_conversation(tok, [Message("user", "word " * 200), Message("assistant", "ok")], 32) is None


def test_targets_are_pre_shifted_with_minus_one_for_masked_positions(tok):
    msgs = [Message("user", "hi"), Message("assistant", "the dog ran")]
    ids, mask = encode_conversation(tok, msgs, 100)
    ds = SFTDataset(ids, mask, np.array([0, len(ids)]), tok.pad_id)
    x, y = ds.batch([0])
    assert x.shape == y.shape == (1, len(ids) - 1) and torch.equal(x[0], torch.tensor(ids[:-1]))
    for i in range(len(y[0])):
        assert (y[0, i].item() == ids[i + 1]) == bool(mask[i + 1]) or y[0, i].item() == -1
        assert (y[0, i].item() != -1) == bool(mask[i + 1])


def test_batches_are_padded_with_ignored_targets(tok):
    a = encode_conversation(tok, [Message("user", "hi"), Message("assistant", "ok")], 100)
    b = encode_conversation(tok, [Message("user", "tell me a long story please"), Message("assistant", "the dog ran to the park")], 100)
    ids = np.concatenate([a[0], b[0]])
    ds = SFTDataset(ids, np.concatenate([a[1], b[1]]), np.array([0, len(a[0]), len(ids)]), tok.pad_id)
    x, y = ds.batch([0, 1])
    assert x.shape[0] == 2 and x.shape[1] == len(b[0]) - 1
    pad_from = len(a[0]) - 1
    assert (x[0, pad_from:] == tok.pad_id).all() and (y[0, pad_from:] == -1).all()


def test_inconsistent_arrays_are_rejected():
    ones = np.ones(10, np.uint8)
    SFTDataset(np.arange(10), ones, np.array([0, 4, 10]), 0)                             # consistent: fine
    with pytest.raises(ValueError):
        SFTDataset(np.arange(10), np.ones(9, np.uint8), np.array([0, 10]), 0)            # ids/mask length differ
    with pytest.raises(ValueError):
        SFTDataset(np.arange(10), ones, np.array([0, 4, 9]), 0)                          # offsets don't cover ids
    with pytest.raises(ValueError):
        SFTDataset(np.arange(10), ones, np.array([0, 1, 10]), 0)                         # a 1-token conversation


# ── build pipeline ───────────────────────────────────────────────────────────

def _sft_exp(tmp_path, rows, max_steps=12, **train):
    pre = tiny_experiment(tmp_path, name="pre")
    src = tmp_path / "convs.jsonl"
    src.write_text("\n".join(json.dumps(r) if not isinstance(r, str) else r for r in rows) + "\n")
    exp = ExperimentConfig.from_dict({**pre.to_dict(), "name": "sft", "stage": "sft"})
    exp.data.sources, exp.data.out_dir = [str(src)], str(tmp_path / "sft_processed")
    exp.data.val_fraction = exp.data.test_fraction = 0.1
    exp.train.max_steps, exp.train.lr, exp.train.eval_interval, exp.train.ckpt_interval = max_steps, 1e-3, 4, 4
    for k, v in train.items():
        setattr(exp.train, k, v)
    return exp


def _rows(n=120):
    # short on purpose: the tiny test model has a 32-token window, and over-long conversations are dropped
    return [conv((f"Tell {WORDS[i % 10]} {i}", f"The {WORDS[(i * 3) % 10]} is nice.")) for i in range(n)]


def test_build_sft_dataset_counts_rejects_dedups_and_honours_explicit_splits(tmp_path):
    rows = _rows(60) + [_rows(1)[0]] * 2                              # two exact duplicates
    rows += ["not json", {"messages": [{"role": "user", "content": "only user"}]}]
    rows += [conv(("word " * 100, "ok"))]                             # too long for the 32-token window
    rows += [conv((f"q{i}", f"a{i} something here"), split="test") for i in range(6)]
    exp = _sft_exp(tmp_path, rows)
    m = build_sft_dataset(exp, log=quiet)
    r = m["counters"]["rejects"]
    assert r["bad_json"] == 1 and r["must_start_with_user_end_with_assistant"] == 1 and r["too_long_for_context"] == 1
    assert m["counters"]["duplicates_removed"] == 2
    assert m["splits"]["test"]["conversations"] >= 6                  # explicit test records all landed in test
    out = Path(exp.data.out_dir)
    tok = AaravTokenizer.load(exp.tokenizer.artifact_dir)
    ds = SFTDataset.load(out / "train.npz", tok.pad_id)
    assert len(ds) == m["splits"]["train"]["conversations"] and ds.n_tokens == m["splits"]["train"]["tokens"]
    assert m["tokenizer"]["fingerprint"] == tok.fingerprint and 0 < m["splits"]["train"]["trained_tokens"] < ds.n_tokens


def test_build_needs_at_least_one_training_conversation(tmp_path):
    exp = _sft_exp(tmp_path, [{"messages": [{"role": "user", "content": "x"}]}])
    with pytest.raises(RuntimeError, match="no training"):
        build_sft_dataset(exp, log=quiet)


def test_sft_stream_resume_is_exact_across_epochs(tmp_path, tok):
    exp = _sft_exp(tmp_path, _rows(40))
    build_sft_dataset(exp, log=quiet)
    ds = SFTDataset.load(Path(exp.data.out_dir) / "train.npz", tok.pad_id)
    ref = SFTBatchStream(ds, 7, seed=3)
    batches = [ref.next_batch() for _ in range(3 * len(ds) // 7 + 3)]
    assert ref.epoch >= 2
    for cut in (1, len(ds) // 7, len(ds) // 7 + 2):
        s = SFTBatchStream(ds, 7, seed=3)
        [s.next_batch() for _ in range(cut)]
        r = SFTBatchStream(ds, 7, seed=3)
        r.load_state_dict(s.state_dict())
        for i in range(cut, len(batches)):
            x, y = r.next_batch()
            assert torch.equal(x, batches[i][0]) and torch.equal(y, batches[i][1]), (cut, i)


# ── generator ────────────────────────────────────────────────────────────────

def test_story_features_and_split_hygiene(tmp_path):
    f = _features("Once upon a time there was a little girl named Lily. She had a dog.")
    assert f == {"name": "Lily", "topic": "girl"}
    stories = "\n<|endoftext|>\n".join(
        f"Once upon a time there was a boy named Tom{chr(65 + i % 26)}{chr(97 + i // 26 % 26)}. The dog and the "
        f"ball were fun on day number {i}. " + "They played happily in the park all day long and laughed. " * 4
        for i in range(400)
    )
    src = tmp_path / "s.txt"
    src.write_text(stories + "\n")
    pool = load_stories([src], seed=5, val_fraction=0.1, test_fraction=0.1)
    assert sum(len(v) for v in pool.values()) == 400 and all(pool[k] for k in pool)
    recs = list(make_conversations(pool, n_train=50, seed=1))
    assert recs == list(make_conversations(pool, n_train=50, seed=1))              # deterministic
    for split in ("val", "test"):                                                  # held-out answers are unseen stories
        answers = {m["content"] for r in recs if r["split"] == split for m in r["messages"] if m["role"] == "assistant"}
        assert answers and answers <= set(pool[split])
        assert not answers & set(pool["train"])
    assert sum(r["split"] == "train" for r in recs) == 50
    counts = write_jsonl(iter(recs), tmp_path / "o.jsonl")
    assert counts["train"] == 50 and len((tmp_path / "o.jsonl").read_text().splitlines()) == len(recs)
    for r in recs:
        assert validate_conversation(r)[1] is None


# ── SFT training ─────────────────────────────────────────────────────────────

def _pretrain_and_sft(tmp_path, tiny_run, **kw):
    exp = _sft_exp(tmp_path / "s", _rows(160), **kw)
    exp.init_from = str(tiny_run["final"])
    exp.tokenizer.artifact_dir = tiny_run["tok_dir"]
    exp.model_overrides = tiny_run["exp"].model_overrides
    exp.data.out_dir = str(tmp_path / "sft_data")
    exp.data.sources = [str(tmp_path / "s" / "convs.jsonl")]
    build_sft_dataset(exp, log=quiet)
    return exp


@pytest.fixture()
def sft_exp(tmp_path, tiny_run):
    (tmp_path / "s").mkdir()
    return _pretrain_and_sft(tmp_path, tiny_run)


def test_make_trainer_picks_the_stage(sft_exp, tiny_run, tmp_path):
    assert type(make_trainer(sft_exp, tmp_path / "a")) is SFTTrainer
    assert type(make_trainer(tiny_run["exp"], tmp_path / "b")) is Trainer
    sft_exp.stage = "rlhf"
    with pytest.raises(ValueError, match="unknown stage"):
        make_trainer(sft_exp, tmp_path / "c")


def test_sft_reduces_loss_and_marks_the_checkpoint_as_instruction_tuned(sft_exp, tiny_run, tmp_path):
    summary = SFTTrainer(sft_exp, tmp_path / "run", log=quiet).run()
    evals = [json.loads(l) for l in (tmp_path / "run/metrics.jsonl").read_text().splitlines() if '"eval"' in l]
    assert summary["status"] == "completed" and summary["stage"] == "sft"
    assert evals[-1]["val_loss"] < evals[0]["val_loss"]
    e = InferenceEngine.from_checkpoint(tmp_path / "run/checkpoints/best.pt", tiny_run["tok_dir"], "cpu")
    assert e.meta["stage"] == "sft"
    from aarav_lm.chat.session import ChatSession
    assert ChatSession(e).mode == "chat"                                # SFT checkpoints switch chat to the template
    # the pretrained checkpoint it started from is untouched
    assert torch.load(tiny_run["final"], weights_only=True)["meta"]["stage"] == "pretrain"


def test_sft_crash_resume_matches_the_uninterrupted_run(sft_exp, tmp_path):
    full = tmp_path / "full"
    SFTTrainer(sft_exp, full, log=quiet).run()
    crashed = tmp_path / "crashed"
    (crashed / "checkpoints").mkdir(parents=True)
    shutil.copy(full / "config.yaml", crashed / "config.yaml")
    shutil.copy(full / "checkpoints/step_00000004.pt", crashed / "checkpoints")
    SFTTrainer(sft_exp, crashed, log=quiet).run(resume=True)
    a = torch.load(full / "checkpoints/final.pt", weights_only=True)["model"]
    b = torch.load(crashed / "checkpoints/final.pt", weights_only=True)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_sft_requires_init_from_and_matching_architecture(sft_exp, tmp_path):
    sft_exp.init_from = ""
    with pytest.raises(RuntimeError, match="init_from"):
        SFTTrainer(sft_exp, tmp_path / "r1", log=quiet).run()
    sft_exp.init_from = str(Path(sft_exp.tokenizer.artifact_dir).parent / "nope.pt")
    with pytest.raises(ck.CheckpointError):
        SFTTrainer(sft_exp, tmp_path / "r2", log=quiet).run()


def test_sft_rejects_a_pretraining_dataset_and_an_architecture_mismatch(sft_exp, tiny_run, tmp_path):
    bad = ExperimentConfig.from_dict(sft_exp.to_dict())
    bad.data.out_dir = tiny_run["exp"].data.out_dir                       # a pretraining shard directory
    with pytest.raises(RuntimeError, match="not an SFT dataset|missing"):
        SFTTrainer(bad, tmp_path / "r1", log=quiet).run()
    wrong = ExperimentConfig.from_dict(sft_exp.to_dict())
    wrong.model_overrides = {**wrong.model_overrides, "n_layer": 3}
    with pytest.raises(RuntimeError, match="architecture"):
        SFTTrainer(wrong, tmp_path / "r2", log=quiet).run()


def test_lora_sft_trains_only_adapters_and_exports_a_plain_merged_model(tmp_path, tiny_run):
    (tmp_path / "s").mkdir()
    exp = _pretrain_and_sft(tmp_path, tiny_run, lora_rank=4, lora_alpha=8, lr=5e-3)
    logs = []
    SFTTrainer(exp, tmp_path / "lora", log=logs.append).run()
    assert any("[lora]" in l and "trainable" in l for l in logs)
    base = torch.load(tiny_run["final"], weights_only=True)["model"]
    tuned, raw = ck.load_model(tmp_path / "lora/checkpoints/final.pt", "cpu", None)
    assert set(tuned.state_dict()) == set(base)                           # plain model: no adapter keys
    changed = [k for k in base if not torch.equal(base[k], tuned.state_dict()[k])]
    assert changed and all(any(t in k for t in ("attn.", "mlp.")) for k in changed), changed
    assert torch.equal(base["token_emb.weight"], tuned.state_dict()["token_emb.weight"])    # embeddings frozen
    # LoRA crash-resume: the training checkpoint holds the adapters
    st = torch.load(tmp_path / "lora/checkpoints/step_00000012.pt", weights_only=True)
    assert any("lora_A" in k for k in st["model"])


def test_lora_crash_resume_matches_the_uninterrupted_run(tmp_path, tiny_run):
    """The earlier LoRA test only checked adapter keys existed; this one actually resumes."""
    (tmp_path / "s").mkdir()
    exp = _pretrain_and_sft(tmp_path, tiny_run, lora_rank=4, lora_alpha=8, lr=5e-3)
    full, crashed = tmp_path / "full", tmp_path / "crashed"
    SFTTrainer(exp, full, log=quiet).run()
    (crashed / "checkpoints").mkdir(parents=True)
    shutil.copy(full / "config.yaml", crashed / "config.yaml")
    shutil.copy(full / "checkpoints/step_00000004.pt", crashed / "checkpoints")
    SFTTrainer(exp, crashed, log=quiet).run(resume=True)
    a = torch.load(full / "checkpoints/step_00000012.pt", weights_only=True)["model"]
    b = torch.load(crashed / "checkpoints/step_00000012.pt", weights_only=True)["model"]
    assert any("lora_A" in k for k in a) and all(torch.equal(a[k], b[k]) for k in a)
    fa = torch.load(full / "checkpoints/final.pt", weights_only=True)["model"]
    fb = torch.load(crashed / "checkpoints/final.pt", weights_only=True)["model"]
    assert all(torch.equal(fa[k], fb[k]) for k in fa)                # the merged export matches too


def test_evaluation_suite_handles_an_sft_checkpoint_and_sft_data(sft_exp, tiny_run, tmp_path):
    """Used to crash: the suite only knew pretraining .bin shards and raw-text prompts."""
    from aarav_lm.evaluation.suite import run_suite
    SFTTrainer(sft_exp, tmp_path / "run", log=quiet).run()
    engine = InferenceEngine.from_checkpoint(tmp_path / "run/checkpoints/best.pt", tiny_run["tok_dir"], "cpu")
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("\n".join(json.dumps(r) for r in [{"id": "a", "prompt": "Tell dog 3", "entity": "dog"},
                                                          {"id": "b", "prompt": "Tell cat 4", "entity": None}]) + "\n")
    probes = Path(__file__).resolve().parents[1] / "evals/probes_v1.jsonl"
    res = run_suite(engine, sft_exp.data.out_dir, prompts, probes, max_new_tokens=10, max_windows=8, log=quiet)
    assert res["model"]["stage"] == "sft"
    assert res["loss"]["val"]["windows"] > 0 and res["loss"]["val"]["loss"] > 0 and res["loss"]["val"]["bits_per_byte"] is None
    assert {s["finish_reason"] for s in res["generation"]["greedy"]["samples"]} <= {"length", "stop", "eot"}
    assert res["edge_cases"]["all_passed"], [c for c in res["edge_cases"]["cases"] if not c["passed"]]
    assert res["eval_sets"]["dataset_fingerprint"]


def test_padded_widths_are_bucketed_to_few_shapes_and_never_exceed_the_longest_conversation(tok):
    from aarav_lm.data.sft import PAD_BUCKET
    rows = [encode_conversation(tok, [Message("user", "hi " * n), Message("assistant", "ok")], 400) for n in range(1, 40)]
    ids = np.concatenate([r[0] for r in rows])
    mask = np.concatenate([r[1] for r in rows])
    offsets = np.concatenate([[0], np.cumsum([len(r[0]) for r in rows])])
    ds = SFTDataset(ids, mask, offsets, tok.pad_id)
    widths = {ds.batch([i])[0].shape[1] for i in range(len(rows))}
    assert len(widths) <= len(rows) // 3                                  # far fewer distinct shapes than lengths
    assert all(w % PAD_BUCKET == 0 or w == ds.max_width for w in widths)
    assert max(widths) <= ds.max_width
    x, y = ds.batch([0])                                                  # padding is still ignored by the loss
    real = len(rows[0][0]) - 1
    assert (y[0, real:] == -1).all() and (x[0, real:] == tok.pad_id).all()
