"""The batch scheduler: concurrent requests become waves, and nothing one user does can hurt another."""

import threading
import time
from dataclasses import replace

import pytest

from aarav_lm.inference import scheduler as sched_mod
from aarav_lm.inference.engine import GenerationConfig
from aarav_lm.inference.scheduler import BatchScheduler, QueueFull

TEXTS = ["Once upon a time there was a little dog", "The cat", "A girl who liked the park", "Every day the boy", "One", "The bird and the fish"]


def ids(engine, n=None):
    p = [engine.tok.encode(t)[:14] for t in TEXTS]
    return p if n is None else p[:n]


def cfg(seed=1, **kw):
    return GenerationConfig(temperature=0.9, top_k=30, top_p=0.95, seed=seed, max_new_tokens=10, stop_at_eot=False, **kw)


@pytest.fixture()
def scheduler(engine):
    s = BatchScheduler(engine, max_batch=8, max_wait=0.15)
    yield s
    s.close()


def drain(ticket):
    return "".join(ticket.deltas())


def test_concurrent_requests_are_batched_and_each_matches_its_solo_run(engine, scheduler):
    ps = ids(engine)
    solo = [engine.generate(p, cfg(i)) for i, p in enumerate(ps)]
    tickets, out = [], {}

    def user(i):
        t = scheduler.submit(ps[i], cfg(i)); tickets.append(t); out[i] = (drain(t), t)
    threads = [threading.Thread(target=user, args=(i,)) for i in range(len(ps))]
    [t.start() for t in threads]; [t.join(20) for t in threads]
    for i in range(len(ps)):
        text, t = out[i]
        assert text == solo[i].text == t.result.text and t.result.token_ids == solo[i].token_ids
    st = scheduler.stats()
    assert st["batched_requests"] == len(ps) and st["largest_wave"] > 1 and st["waves"] < len(ps)        # they really shared waves


def test_a_single_request_still_works_and_reports_its_usage(engine, scheduler):
    p = ids(engine)[0]
    t = scheduler.submit(p, cfg(5))
    text = drain(t)
    assert text == engine.generate(p, cfg(5)).text and t.result.prompt_tokens == len(p) and t.result.generated_tokens > 0


def test_the_queue_is_bounded_and_a_closed_scheduler_refuses_work(engine):
    s = BatchScheduler(engine, max_batch=1, max_wait=0, max_queue=2)
    engine.lock.acquire()                                     # the model is busy: nothing can start
    try:
        first = s.submit(ids(engine)[0], cfg())
        time.sleep(0.05)                                      # the thread takes `first` and waits for the lock
        s.submit(ids(engine)[1], cfg()); s.submit(ids(engine)[2], cfg())
        with pytest.raises(QueueFull, match="waiting"):
            s.submit(ids(engine)[3], cfg())
        assert not first.wait_started(0.1)                    # a caller can tell it has not reached the model
    finally:
        engine.lock.release()
    assert first.wait_started(5)
    s.close()
    with pytest.raises(QueueFull):
        s.submit(ids(engine)[0], cfg())


def test_a_failed_wave_fails_only_its_own_requests_and_the_scheduler_survives(engine, scheduler, monkeypatch):
    real = sched_mod.generate_batch
    calls = {"n": 0}

    def flaky(eng, jobs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("out of memory")
        return real(eng, jobs)
    monkeypatch.setattr(sched_mod, "generate_batch", flaky)
    bad = scheduler.submit(ids(engine)[0], cfg())
    with pytest.raises(RuntimeError, match="out of memory"):
        drain(bad)
    good = scheduler.submit(ids(engine)[1], cfg())
    assert drain(good) == engine.generate(ids(engine)[1], cfg()).text


def test_a_client_that_stops_reading_frees_its_row_without_touching_the_others(engine, scheduler):
    a, b = ids(engine)[0], ids(engine)[1]
    long_cfg = replace(cfg(3), max_new_tokens=20)
    quitter = scheduler.submit(a, long_cfg)
    stayer = scheduler.submit(b, cfg(4))
    it = quitter.deltas()
    next(it)                                                  # reads one delta, then the client "disconnects"
    it.close()
    assert quitter.cancelled()
    assert drain(stayer) == engine.generate(b, cfg(4)).text
    time.sleep(0.2)
    assert quitter.result is None or quitter.result.finish_reason in ("cancelled", "length", "stop")


def test_a_request_cancelled_while_still_queued_never_reaches_the_model(engine):
    s = BatchScheduler(engine, max_batch=1, max_wait=0)
    engine.lock.acquire()
    try:
        s.submit(ids(engine)[0], cfg())
        time.sleep(0.05)
        waiting = s.submit(ids(engine)[1], cfg())
        waiting.cancel()                                      # e.g. its queue timeout passed
    finally:
        engine.lock.release()
    assert list(waiting.deltas()) == [] and waiting.result is None and not waiting.started.is_set()
    s.close()


def test_requests_that_do_not_fit_a_batch_run_alone_and_still_answer(engine, scheduler):
    long_prompt = engine.tok.encode("the " * 40)              # longer than the context: the single path slides its window
    solo = engine.generate(long_prompt, cfg(9))
    t = scheduler.submit(long_prompt, cfg(9))
    assert drain(t) == solo.text
    assert scheduler.stats()["solo_requests"] == 1


def test_validation_and_shutdown(engine):
    with pytest.raises(ValueError):
        BatchScheduler(engine, max_batch=0)
    s = BatchScheduler(engine)
    with pytest.raises(ValueError, match="empty"):
        s.submit([], cfg())
    s.close()
    assert not s._thread.is_alive()
