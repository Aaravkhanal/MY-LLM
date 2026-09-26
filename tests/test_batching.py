"""Batched generation must give each request exactly what it would get alone, whoever else shares the batch."""

from dataclasses import replace

import pytest

from aarav_lm.inference.batching import BatchNotPossible, Job, can_batch, generate_batch
from aarav_lm.inference.engine import GenerationConfig

TEXTS = ["Once upon a time there was a little dog", "The cat", "A girl who liked the park and the tree went out",
         "Every day the boy", "One", "The bird and the fish and the sun"]


def prompts(engine, n=None):
    ids = [engine.tok.encode(t) for t in TEXTS]
    ids = [i[:14] for i in ids]
    return ids if n is None else ids[:n]


def greedy(**kw):
    return GenerationConfig(temperature=0.0, top_k=None, top_p=None, stop_at_eot=False, max_new_tokens=10, **kw)


def alone(engine, p, cfg):
    return engine.generate(p, cfg)


def same(a, b):
    return (a.text, a.token_ids, a.finish_reason, a.generated_tokens) == (b.text, b.token_ids, b.finish_reason, b.generated_tokens)


def test_greedy_requests_of_different_lengths_match_their_solo_runs(engine):
    ps = prompts(engine)
    assert len({len(p) for p in ps}) > 3                                    # genuinely different lengths
    got = generate_batch(engine, [Job(p, greedy()) for p in ps])
    for p, g in zip(ps, got):
        assert same(g, alone(engine, p, greedy()))


@pytest.mark.parametrize("kw", [dict(temperature=0.8, top_k=20, top_p=0.9), dict(temperature=1.2, top_k=None, top_p=0.7),
                                dict(temperature=0.7, top_k=5, top_p=None, repetition_penalty=1.3),
                                dict(temperature=0.9, top_k=None, top_p=None, no_repeat_ngram_size=2)])
def test_seeded_sampling_matches_solo_runs(engine, kw):
    ps = prompts(engine, 4)
    cfgs = [GenerationConfig(max_new_tokens=10, seed=100 + i, stop_at_eot=False, **kw) for i in range(4)]
    got = generate_batch(engine, [Job(p, c) for p, c in zip(ps, cfgs)])
    for p, c, g in zip(ps, cfgs, got):
        assert same(g, alone(engine, p, c))


def test_a_requests_output_does_not_depend_on_its_batch_mates_or_their_order(engine):
    ps = prompts(engine)
    cfg = lambda i: GenerationConfig(temperature=0.9, top_k=30, top_p=0.95, seed=7 + i, max_new_tokens=9, stop_at_eot=False)   # noqa: E731
    target = generate_batch(engine, [Job(ps[2], cfg(2))])[0]
    for order in ([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [2, 4], [3, 2, 0]):
        got = generate_batch(engine, [Job(ps[i], cfg(i)) for i in order])
        assert same(got[order.index(2)], target), order


def test_per_request_limits_stop_tokens_and_stop_sequences_are_honoured_per_row(engine):
    ps = prompts(engine, 3)
    probe = alone(engine, ps[2], greedy())                                   # the row that gets the stop sequence
    assert len(probe.text) > 6
    cut = probe.text[3:6]
    cfgs = [greedy(), replace(greedy(), max_new_tokens=3), replace(greedy(), stop_sequences=(cut,))]
    got = generate_batch(engine, [Job(p, c) for p, c in zip(ps, cfgs)])
    assert [g.generated_tokens for g in got][1] == 3 and got[1].finish_reason == "length"
    assert got[2].finish_reason == "stop" and cut not in got[2].text
    for p, c, g in zip(ps, cfgs, got):
        assert same(g, alone(engine, p, c))
    stop_id = probe.token_ids[2]
    first = probe.token_ids.index(stop_id)                                   # the token may already occur earlier
    r = generate_batch(engine, [Job(ps[2], replace(greedy(), stop_ids=(stop_id,)))])[0]
    assert r.finish_reason == "stop" and r.generated_tokens == first
    assert same(r, alone(engine, ps[2], replace(greedy(), stop_ids=(stop_id,))))


def test_streamed_deltas_concatenate_to_the_final_text(engine):
    ps = prompts(engine, 4)
    seen = [[] for _ in ps]
    jobs = [Job(p, greedy(), on_delta=seen[i].append) for i, p in enumerate(ps)]
    for i, g in enumerate(generate_batch(engine, jobs)):
        assert "".join(seen[i]) == g.text and g.text


def test_cancelling_one_row_leaves_the_others_untouched(engine):
    ps = prompts(engine, 3)
    state = {"n": 0}
    jobs = [Job(p, greedy()) for p in ps]
    jobs[1].cancelled = lambda: state.__setitem__("n", state["n"] + 1) or state["n"] > 3          # cancelled a few steps in
    got = generate_batch(engine, jobs)
    assert got[1].finish_reason == "cancelled" and got[1].generated_tokens < 10
    assert same(got[0], alone(engine, ps[0], greedy())) and same(got[2], alone(engine, ps[2], greedy()))


def test_a_row_over_its_time_budget_times_out_alone(engine):
    ps = prompts(engine, 2)
    jobs = [Job(ps[0], greedy()), Job(ps[1], replace(greedy(), max_seconds=1e-9))]
    got = generate_batch(engine, jobs)
    assert got[1].finish_reason == "timeout" and same(got[0], alone(engine, ps[0], greedy()))


def test_requests_that_cannot_share_a_batch_are_refused_clearly(engine):
    too_long = engine.tok.encode("the " * 40)
    assert not can_batch(engine, len(too_long), greedy())
    with pytest.raises(BatchNotPossible):
        generate_batch(engine, [Job(too_long, greedy())])
    with pytest.raises(BatchNotPossible):
        generate_batch(engine, [Job(prompts(engine)[0], replace(greedy(), max_new_tokens=engine.block_size))])
    with pytest.raises(ValueError, match="empty"):
        generate_batch(engine, [Job([], greedy())])
    assert generate_batch(engine, []) == []


def test_learned_position_models_are_never_batched(engine, monkeypatch):
    monkeypatch.setattr(engine.model.cfg, "arch", "gpt2")
    assert not can_batch(engine, 5, greedy())


def test_the_batch_is_faster_than_running_the_same_requests_one_after_another(engine):
    """A sanity check on the point of it all (not a benchmark): 6 requests together need fewer model calls than 6 alone."""
    import time
    ps = prompts(engine)
    cfg = greedy()
    t = time.perf_counter()
    for p in ps:
        alone(engine, p, cfg)
    solo = time.perf_counter() - t
    t = time.perf_counter()
    generate_batch(engine, [Job(p, cfg) for p in ps])
    batch = time.perf_counter() - t
    assert batch < solo
