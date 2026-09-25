import pytest
import torch

from aarav_lm.inference import engine as eng_mod
from aarav_lm.inference.engine import GenerationConfig, InferenceEngine, sample_next
from aarav_lm.train.checkpoint import CheckpointError


def greedy(**kw):
    return GenerationConfig(temperature=0.0, top_k=None, top_p=None, stop_at_eot=False, **kw)


# ── sampling ─────────────────────────────────────────────────────────────────

def test_greedy_and_top_k_one_pick_the_argmax():
    logits = torch.tensor([0.1, 2.0, 0.5, 1.9])
    assert sample_next(logits, greedy(), None) == 1
    cfg = GenerationConfig(temperature=1.0, top_k=1, top_p=None)
    assert all(sample_next(logits, cfg, torch.Generator().manual_seed(s)) == 1 for s in range(20))


def test_top_p_excludes_the_tail_but_keeps_the_crossing_token():
    probs = torch.tensor([0.5, 0.3, 0.2])
    logits = probs.log()
    cfg = GenerationConfig(temperature=1.0, top_k=None, top_p=0.7)
    g = torch.Generator().manual_seed(0)
    seen = {sample_next(logits, cfg, g) for _ in range(300)}
    assert seen == {0, 1}                                  # cumulative 0.5 < 0.7 -> token 1 is kept, token 2 is not
    cfg = GenerationConfig(temperature=1.0, top_k=None, top_p=0.5)
    assert {sample_next(logits, cfg, g) for _ in range(100)} == {0}


def test_sampling_is_deterministic_for_a_seed_and_varies_across_seeds():
    logits = torch.randn(200)
    cfg = GenerationConfig(temperature=1.0, top_k=None, top_p=None)
    a = [sample_next(logits, cfg, torch.Generator().manual_seed(1)) for _ in range(3)]
    assert a == [sample_next(logits, cfg, torch.Generator().manual_seed(1)) for _ in range(3)]
    assert len({sample_next(logits, cfg, torch.Generator().manual_seed(s)) for s in range(50)}) > 5


def test_repetition_penalty_moves_probability_away_from_recent_tokens():
    logits = torch.tensor([3.0, 2.9, -1.0])
    plain = greedy()
    penalised = greedy(repetition_penalty=2.0)
    assert sample_next(logits, plain, None, recent_ids=[0]) == 0
    assert sample_next(logits, penalised, None, recent_ids=[0]) == 1
    negative = torch.tensor([-1.0, -1.1])          # negative logits are multiplied, so they get *less* likely
    assert sample_next(negative, greedy(repetition_penalty=2.0), None, recent_ids=[0]) == 1


@pytest.mark.parametrize("bad", [
    dict(max_new_tokens=0), dict(temperature=-1), dict(top_k=0), dict(top_p=0.0),
    dict(top_p=1.5), dict(repetition_penalty=0), dict(stop_sequences=("",)),
])
def test_invalid_generation_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        GenerationConfig(**bad).validate()


# ── engine ───────────────────────────────────────────────────────────────────

def test_engine_greedy_matches_model_generate(engine):
    prompt = engine.tok.encode("Once upon a time there was")
    ref = engine.model.generate(torch.tensor([prompt]), max_new_tokens=15, temperature=0.0)[0, len(prompt):].tolist()
    res = engine.generate(prompt, greedy(max_new_tokens=15))
    assert res.token_ids == ref and res.finish_reason == "length" and res.generated_tokens == 15


def test_seeded_generation_is_reproducible(engine):
    cfg = GenerationConfig(max_new_tokens=20, temperature=1.0, seed=42, stop_at_eot=False)
    assert engine.generate("Once upon", cfg).token_ids == engine.generate("Once upon", cfg).token_ids
    other = GenerationConfig(max_new_tokens=20, temperature=1.0, seed=43, stop_at_eot=False)
    assert engine.generate("Once upon", cfg).token_ids != engine.generate("Once upon", other).token_ids


def test_streamed_deltas_concatenate_to_the_final_text(engine):
    cfg = GenerationConfig(max_new_tokens=25, temperature=1.0, seed=3, stop_at_eot=False)
    deltas = list(engine.stream(engine.tok.encode("The dog"), cfg))
    assert len(deltas) > 1
    assert "".join(deltas) == engine.last_result.text == engine.tok.decode(engine.last_result.token_ids)


def test_stop_sequence_truncates_and_never_leaks_partial_matches(engine):
    prompt = engine.tok.encode("Once upon a time")
    base = engine.generate(prompt, greedy(max_new_tokens=40)).text
    assert len(base) > 20
    stop = base[12:16]
    cfg = greedy(max_new_tokens=40, stop_sequences=(stop,))
    deltas = list(engine.stream(prompt, cfg))
    out = "".join(deltas)
    assert engine.last_result.finish_reason == "stop"
    assert out == base[: base.find(stop)] and stop not in out
    # no single delta may end in a partial stop sequence that the final text then contradicts
    assert "".join(deltas) == engine.last_result.text


def test_length_limit_and_finish_reasons(engine):
    res = engine.generate("Once", greedy(max_new_tokens=7))
    assert res.generated_tokens == 7 and res.finish_reason == "length" and res.tokens_per_sec > 0
    forced = greedy(max_new_tokens=50, stop_ids=(engine.generate("Once", greedy(max_new_tokens=1)).token_ids[0],))
    assert engine.generate("Once", forced).finish_reason == "stop"


def test_eot_ends_generation(engine, monkeypatch):
    monkeypatch.setattr(eng_mod, "sample_next", lambda *a, **k: engine.tok.eot_id)
    res = engine.generate("Once", GenerationConfig(max_new_tokens=10))
    assert res.finish_reason == "eot" and res.generated_tokens == 0


def test_generation_slides_the_window_instead_of_failing(engine):
    """Ask for far more tokens than the 32-token context; the engine must keep going."""
    assert engine.block_size == 32
    res = engine.generate("Once upon a time", greedy(max_new_tokens=100))
    assert res.generated_tokens == 100 and res.finish_reason == "length"


def test_prompt_longer_than_the_window_is_accepted(engine):
    long_prompt = engine.tok.encode("The dog ran to the park and played. " * 30)
    assert len(long_prompt) > engine.block_size * 3
    assert engine.generate(long_prompt, greedy(max_new_tokens=5)).generated_tokens == 5


def test_incomplete_utf8_is_held_back_not_printed_as_replacement_chars(engine, monkeypatch):
    text = "café ☃☃ ok"
    script = iter(engine.tok.encode(text))
    monkeypatch.setattr(eng_mod, "sample_next", lambda *a, **k: next(script))
    ids = engine.tok.encode("The")
    deltas = list(engine.stream(ids, GenerationConfig(max_new_tokens=len(engine.tok.encode(text)))))
    assert all("�" not in d for d in deltas)
    assert "".join(deltas) == text


def test_empty_prompt_is_an_error(engine):
    with pytest.raises(ValueError):
        list(engine.stream([], GenerationConfig()))


def test_loading_with_the_wrong_tokenizer_fails_loudly(tiny_run, tmp_path):
    from aarav_lm.tokenizer import train_tokenizer

    other = train_tokenizer(["some entirely different corpus of words"] * 20, vocab_size=300)
    other.save(tmp_path / "other")
    with pytest.raises(CheckpointError, match="tokenizer"):
        InferenceEngine.from_checkpoint(tiny_run["final"], tmp_path / "other", device="cpu")


def test_engine_info_reports_model_facts(engine):
    info = engine.info()
    assert info["context_length"] == 32 and info["vocab_size"] == 300 and info["params"] > 0
