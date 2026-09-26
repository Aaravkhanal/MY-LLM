import json

import numpy as np
import pytest
import torch

from aarav_lm.config import ExperimentConfig
from aarav_lm.data import ingest
from aarav_lm.data.clean import normalize_text, quality_reject_reason
from aarav_lm.data.dataset import BatchStream, TokenDataset
from aarav_lm.data.dedup import ExactDeduper
from aarav_lm.data.ingest import IngestStats, iter_documents
from aarav_lm.data.pipeline import build_dataset, validate_dataset
from aarav_lm.data.shards import HEADER_BYTES, Shard, ShardError, ShardWriter
from aarav_lm.data.split import split_of
from aarav_lm.tokenizer import AaravTokenizer

EOT = "<|endoftext|>"


# ── clean / dedup / split ────────────────────────────────────────────────────

def test_normalize_is_idempotent_and_tidies_whitespace():
    raw = "Hello\r\nworld  \n\n\n\nbye\x00\x07 \n"
    once = normalize_text(raw)
    assert once == "Hello\nworld\n\nbye"
    assert normalize_text(once) == once


def test_normalize_unifies_unicode_forms():
    assert normalize_text("café") == normalize_text("café")


@pytest.mark.parametrize("text,reason", [
    ("short", "too_short"),
    ("x" * 200_000, "too_long"),
    ("a" * 50 + "�" * 10, "malformed_utf8"),
    ("1234 5678 9012 3456 7890 1234 5678", "low_alpha"),
    ("\n".join(["same line here again"] * 20), "repetitive"),
])
def test_quality_rejections_are_attributed(text, reason):
    assert quality_reject_reason(text) == reason


def test_good_text_passes_quality():
    assert quality_reject_reason("Once upon a time there was a little dog who loved to run.") is None


def test_dedup_ignores_case_and_whitespace():
    d = ExactDeduper()
    assert not d.is_duplicate("The  cat sat.")
    assert d.is_duplicate("the cat   sat.")
    assert not d.is_duplicate("The cat sat down.")
    assert len(d) == 2


def test_split_is_deterministic_seeded_and_roughly_proportional():
    docs = [f"story number {i} about a dog" for i in range(5000)]
    a = [split_of(t, 1, 0.05, 0.05) for t in docs]
    assert a == [split_of(t, 1, 0.05, 0.05) for t in docs]
    assert a != [split_of(t, 2, 0.05, 0.05) for t in docs]
    val_frac = a.count("val") / len(a)
    assert 0.03 < val_frac < 0.07


def test_split_is_stable_when_data_is_appended():
    docs = [f"doc {i} with some words" for i in range(1000)]
    before = {t: split_of(t, 7, 0.1, 0.1) for t in docs}
    after = {t: split_of(t, 7, 0.1, 0.1) for t in docs + [f"new doc {i}" for i in range(500)]}
    assert all(after[t] == before[t] for t in docs)


# ── ingest ───────────────────────────────────────────────────────────────────

def test_txt_split_across_read_chunks(tmp_path, monkeypatch):
    """Separators and multibyte characters straddling chunk boundaries must survive."""
    docs = [f"Story {i}: café ☃ " + "word " * 30 for i in range(40)]
    p = tmp_path / "s.txt"
    p.write_text((EOT + "\n").join(docs) + EOT + "\n", encoding="utf-8")
    monkeypatch.setattr(ingest, "CHUNK_BYTES", 37)      # tiny chunks, odd size on purpose
    got = [d.text.strip() for d in iter_documents([p], IngestStats())]
    assert got == [d.strip() for d in docs]


def test_jsonl_counts_malformed_records(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "good one"}\nnot json\n{"other": 1}\n\n{"text": 5}\n{"text": "fine"}\n')
    st = IngestStats()
    got = [d.text for d in iter_documents([p], st, text_field="text")]
    assert got == ["good one", "fine"]
    assert st.rejects["bad_json"] == 1 and st.rejects["missing_text_field"] == 2


def test_missing_source_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_documents([tmp_path / "nope.txt"], IngestStats()))


# ── shards ───────────────────────────────────────────────────────────────────

def _write_shard(path, n=1000, vocab=300, seed=0):
    toks = np.random.default_rng(seed).integers(0, vocab, size=n)
    with ShardWriter(path, vocab) as w:
        w.append(toks[: n // 2], n_docs=2)
        w.append(toks[n // 2:], n_docs=3)
    return toks


def test_shard_round_trip(tmp_path):
    toks = _write_shard(tmp_path / "a.bin")
    s = Shard.open(tmp_path / "a.bin")
    assert (s.n_tokens, s.vocab_size, s.n_docs) == (1000, 300, 5)
    assert np.array_equal(np.asarray(s.tokens), toks)
    assert (tmp_path / "a.bin").stat().st_size == HEADER_BYTES + 1000 * 2


def test_shard_detects_truncation_and_bad_magic(tmp_path):
    _write_shard(tmp_path / "a.bin")
    raw = (tmp_path / "a.bin").read_bytes()
    (tmp_path / "trunc.bin").write_bytes(raw[:-10])
    with pytest.raises(ShardError, match="truncated"):
        Shard.open(tmp_path / "trunc.bin")
    (tmp_path / "junk.bin").write_bytes(b"\x01" * 2048)
    with pytest.raises(ShardError, match="magic"):
        Shard.open(tmp_path / "junk.bin")


def test_shard_writer_rejects_out_of_range_ids(tmp_path):
    with pytest.raises(ShardError):
        with ShardWriter(tmp_path / "a.bin", 100) as w:
            w.append(np.array([5, 100]))


# ── dataset / batch stream ───────────────────────────────────────────────────

@pytest.fixture()
def ds(tmp_path):
    _write_shard(tmp_path / "t.bin", n=5000, vocab=300)
    return TokenDataset(Shard.open(tmp_path / "t.bin"), seq_len=32)


def test_targets_are_inputs_shifted_by_one(ds):
    x, y = ds.window(3)
    assert np.array_equal(x[1:], y[:-1])
    xb, yb = ds.batch(np.array([0, 1, 2]))
    assert xb.shape == yb.shape == (3, 32) and xb.dtype == torch.int64


def test_too_small_shard_is_rejected(tmp_path):
    _write_shard(tmp_path / "s.bin", n=40)
    with pytest.raises(ValueError):
        TokenDataset(Shard.open(tmp_path / "s.bin"), seq_len=32)


def test_stream_is_deterministic_for_a_seed(ds):
    a, b = BatchStream(ds, 4, seed=5), BatchStream(ds, 4, seed=5)
    for _ in range(10):
        (xa, ya), (xb, yb) = a.next_batch(), b.next_batch()
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    other = BatchStream(ds, 4, seed=6)
    assert not torch.equal(BatchStream(ds, 4, seed=5).next_batch()[0], other.next_batch()[0])


def test_stream_resume_is_exact_across_epoch_boundaries(ds):
    """Crash at any point, restore state, and the future batches are identical."""
    ref = BatchStream(ds, 5, seed=11)
    n_steps = ds.n_windows * 3 // 5 + 4          # crosses several epochs
    batches = [ref.next_batch() for _ in range(n_steps)]
    assert ref.epoch >= 2
    for cut in (1, ds.n_windows // 5, ds.n_windows // 5 + 1, n_steps - 3):
        s = BatchStream(ds, 5, seed=11)
        for _ in range(cut):
            s.next_batch()
        state = s.state_dict()
        resumed = BatchStream(ds, 5, seed=11)
        resumed.load_state_dict(state)
        for i in range(cut, n_steps):
            x, y = resumed.next_batch()
            assert torch.equal(x, batches[i][0]) and torch.equal(y, batches[i][1]), (cut, i)


def test_each_window_is_visited_once_per_epoch(ds):
    s = BatchStream(ds, 1, seed=3)
    seen = [tuple(s.next_batch()[0][0, :4].tolist()) for _ in range(ds.n_windows)]
    assert s.epoch == 1 and s.pos == 0
    assert len(seen) == ds.n_windows


def test_epoch_offset_changes_window_alignment(ds):
    s = BatchStream(ds, 1, seed=1)
    offsets = {s._offset(e) for e in range(20)}
    assert len(offsets) > 1 and all(0 <= o < ds.seq_len for o in offsets)


def test_stream_rejects_mismatched_seed(ds):
    s = BatchStream(ds, 2, seed=1)
    with pytest.raises(ValueError):
        s.load_state_dict({"seed": 2, "epoch": 0, "pos": 0})


def test_fixed_eval_batches_are_deterministic_and_cover_the_split(ds):
    a = ds.fixed_eval_batches(4, 5)
    b = ds.fixed_eval_batches(4, 5)
    assert all(torch.equal(x1, x2) for (x1, _), (x2, _) in zip(a, b))
    assert sum(x.shape[0] for x, _ in a) == 20


# ── end-to-end build ─────────────────────────────────────────────────────────

def _tiny_exp(tmp_path, vocab=320):
    stories = [
        f"Once upon a time there was a dog named Rex number {i}. Rex loved to run and play "
        f"in the park with his friend {['Tom', 'Mia', 'Sam', 'Ann'][i % 4]}. It was a sunny day."
        for i in range(300)
    ]
    stories += [stories[0], stories[1].upper()]           # exact + case-only duplicates
    stories += ["tiny", "1234 5678 9012 3456 7890 1234 5678"]   # rejected by quality filters
    src = tmp_path / "corpus.txt"
    src.write_text((EOT + "\n").join(stories) + "\n", encoding="utf-8")
    exp = ExperimentConfig(name="unit")
    exp.data.sources = [str(src)]
    exp.data.out_dir = str(tmp_path / "processed")
    exp.data.val_fraction = 0.1
    exp.data.test_fraction = 0.1
    exp.tokenizer.vocab_size = vocab
    exp.tokenizer.artifact_dir = str(tmp_path / "tok")
    return exp


def test_build_dataset_end_to_end(tmp_path):
    exp = _tiny_exp(tmp_path)
    m = build_dataset(exp, log=lambda _: None)
    c = m["counters"]
    assert c["duplicates_removed"] == 2
    assert c["quality_rejects"] == {"too_short": 1, "low_alpha": 1}
    assert sum(c["docs_kept"].values()) == 300
    assert all(c["docs_kept"][s] > 0 for s in ("train", "val", "test"))

    tok = AaravTokenizer.load(exp.tokenizer.artifact_dir)
    assert validate_dataset(exp.data.out_dir, tok) == []
    val = Shard.open(f"{exp.data.out_dir}/val.bin")
    assert int((val.tokens == tok.eot_id).sum()) == val.n_docs == c["docs_kept"]["val"]


def test_build_is_reproducible(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = build_dataset(_tiny_exp(tmp_path / "a"), log=lambda _: None)
    b = build_dataset(_tiny_exp(tmp_path / "b"), log=lambda _: None)
    assert a["tokenizer"]["fingerprint"] == b["tokenizer"]["fingerprint"]
    for s in ("train", "val", "test"):
        assert (tmp_path / "a/processed" / f"{s}.bin").read_bytes() == (tmp_path / "b/processed" / f"{s}.bin").read_bytes()


def test_validate_flags_wrong_tokenizer_and_corruption(tmp_path):
    exp = _tiny_exp(tmp_path)
    build_dataset(exp, log=lambda _: None)
    good = AaravTokenizer.load(exp.tokenizer.artifact_dir)
    other = AaravTokenizer.load(exp.tokenizer.artifact_dir)
    other.meta = {**other.meta, "sha256": "f" * 64}
    assert any("different tokenizer" in p for p in validate_dataset(exp.data.out_dir, other))

    p = tmp_path / "processed" / "train.bin"
    p.write_bytes(p.read_bytes()[:-6])
    assert any("train" in msg for msg in validate_dataset(exp.data.out_dir, good))


def test_existing_tokenizer_with_other_vocab_is_not_silently_reused(tmp_path):
    exp = _tiny_exp(tmp_path)
    build_dataset(exp, log=lambda _: None)
    exp.tokenizer.vocab_size = 400
    with pytest.raises(RuntimeError, match="vocab"):
        build_dataset(exp, log=lambda _: None)


# ── tokenizer sampling ───────────────────────────────────────────────────────

def test_tokenizer_sample_is_deterministic_evenly_spaced_and_about_the_requested_size():
    from aarav_lm.data.pipeline import sample_for_tokenizer
    docs = [f"document number {i} " + "x" * 100 for i in range(1000)]
    total_mb = sum(len(d) for d in docs) / 1e6
    assert sample_for_tokenizer(docs, 0) is docs and sample_for_tokenizer(docs, total_mb * 2) is docs
    a = sample_for_tokenizer(docs, total_mb / 4)
    assert a == sample_for_tokenizer(docs, total_mb / 4)
    assert 0.2 * total_mb <= sum(len(d) for d in a) / 1e6 <= 0.3 * total_mb
    ids = [int(d.split()[2]) for d in a]
    assert ids[0] == 0 and len({b - a_ for a_, b in zip(ids, ids[1:])}) == 1          # constant stride: no ordering bias


def test_build_with_tokenizer_sampling_records_it_and_still_validates(tmp_path):
    exp = _tiny_exp(tmp_path)
    exp.tokenizer.train_sample_mb = 0.02
    m = build_dataset(exp, log=lambda _: None)
    tok = AaravTokenizer.load(exp.tokenizer.artifact_dir)
    assert 0 < tok.meta["corpus"]["n_docs_used"] < tok.meta["corpus"]["n_train_docs"]
    assert tok.meta["corpus"]["sample_mb"] == 0.02 and validate_dataset(exp.data.out_dir, tok) == []
