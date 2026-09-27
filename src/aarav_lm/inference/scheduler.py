"""
aarav-lm · Batch scheduler: turns concurrent requests into batched waves

    ticket = scheduler.submit(prompt_ids, cfg)        # raises QueueFull when too many are waiting
    ticket.wait_started(timeout)                      # False if it has not reached the model in time
    for delta in ticket.deltas(): ...                 # text as the model produces it; raises if the wave failed
    ticket.result                                     # a GenerationResult once deltas() ends

A background thread waits for a request, then collects more for up to `max_wait` seconds (or until `max_batch` are in), and runs
them as one wave (inference/batching.py). A wave holds `engine.lock`, so it never overlaps any other use of the engine. Requests
that cannot join a batch (a very long prompt, or a learned-position model) run alone, one after another, in the same thread.

Failure isolation: if a wave raises, every ticket in it gets that error and the scheduler carries on with the next wave. A consumer
that stops reading (a client that disconnected) cancels its row, which frees it at the next step without touching the others.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Deque, Iterator, List, Optional, Sequence

from aarav_lm.inference.batching import Job, can_batch, generate_batch
from aarav_lm.inference.engine import GenerationConfig, GenerationResult, InferenceEngine

_DONE = object()


class QueueFull(RuntimeError):
    """Too many requests are already waiting."""


class Ticket:
    def __init__(self, prompt_ids: Sequence[int], cfg: GenerationConfig):
        self.prompt_ids, self.cfg = list(prompt_ids), cfg
        self.result: Optional[GenerationResult] = None
        self.submitted = time.monotonic()
        self.started = threading.Event()
        self._cancel = threading.Event()
        self._q: "queue.Queue" = queue.Queue()

    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def wait_started(self, timeout: Optional[float]) -> bool:
        return self.started.wait(timeout)

    def deltas(self) -> Iterator[str]:
        """Blocks until the model produces text. Raises the wave's exception if it failed. Cancels the row if the consumer stops early."""
        try:
            while True:
                item = self._q.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self.cancel()                                   # a no-op after normal completion; frees the row after an abandoned read

    # called by the scheduler thread
    def _push(self, delta: str) -> None:
        self._q.put(delta)

    def _finish(self, error: Optional[BaseException] = None) -> None:
        if error is not None:
            self._q.put(error)
        self._q.put(_DONE)


class BatchScheduler:
    def __init__(self, engine: InferenceEngine, max_batch: int = 8, max_wait: float = 0.015, max_queue: int = 64):
        if max_batch < 1 or max_queue < 1 or max_wait < 0:
            raise ValueError("max_batch and max_queue must be >= 1 and max_wait >= 0")
        self.engine, self.max_batch, self.max_wait, self.max_queue = engine, max_batch, max_wait, max_queue
        self._pending: Deque[Ticket] = deque()
        self._cond = threading.Condition()
        self._closed = False
        self.waves = 0
        self.batched_requests = 0
        self.solo_requests = 0
        self.largest_wave = 0
        self._thread = threading.Thread(target=self._loop, name="aarav-batch-scheduler", daemon=True)
        self._thread.start()

    # ── public ──────────────────────────────────────────────────────────────
    def submit(self, prompt_ids: Sequence[int], cfg: GenerationConfig) -> Ticket:
        cfg = cfg.validate()
        if not len(prompt_ids):
            raise ValueError("prompt is empty")
        ticket = Ticket(prompt_ids, cfg)
        with self._cond:
            if self._closed:
                raise QueueFull("the scheduler is shut down")
            if len(self._pending) >= self.max_queue:
                raise QueueFull(f"{self.max_queue} requests are already waiting")
            self._pending.append(ticket)
            self._cond.notify()
        return ticket

    def alive(self) -> bool:
        """Whether the scheduler thread is running (a stopped one would leave every request waiting forever)."""
        return self._thread.is_alive() and not self._closed

    def waiting(self) -> int:
        with self._cond:
            return len(self._pending)

    def stats(self) -> dict:
        return {"waves": self.waves, "batched_requests": self.batched_requests, "solo_requests": self.solo_requests,
                "largest_wave": self.largest_wave, "mean_wave_size": round(self.batched_requests / self.waves, 2) if self.waves else 0.0,
                "waiting": self.waiting()}

    def close(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._closed = True
            for t in self._pending:                         # nobody is waiting on these any more: end them cleanly
                t._finish(QueueFull("the scheduler is shut down"))
            self._pending.clear()
            self._cond.notify_all()
        self._thread.join(timeout)

    # ── the thread ──────────────────────────────────────────────────────────
    def _collect(self) -> List[Ticket]:
        with self._cond:
            while not self._pending and not self._closed:
                self._cond.wait()
            if self._closed:
                return []
            batch = [self._pending.popleft()]
            deadline = time.monotonic() + self.max_wait
            while len(batch) < self.max_batch:
                if self._pending:
                    batch.append(self._pending.popleft())
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._closed:
                    break
                self._cond.wait(remaining)
            return batch

    def _loop(self) -> None:
        while True:
            batch = self._collect()
            if not batch:
                if self._closed:
                    return
                continue
            live = []
            for t in batch:
                if t.cancelled():                           # gave up while queued (e.g. its queue timeout passed)
                    t._finish()
                else:
                    live.append(t)
            if not live:
                continue
            fit = [t for t in live if can_batch(self.engine, len(t.prompt_ids), t.cfg)]
            solo = [t for t in live if t not in fit]
            with self.engine.lock:
                if fit:
                    self._run_wave(fit)
                for t in solo:
                    self._run_solo(t)

    def _run_wave(self, tickets: List[Ticket]) -> None:
        for t in tickets:
            t.started.set()
        jobs = [Job(t.prompt_ids, t.cfg, on_delta=t._push, cancelled=t.cancelled) for t in tickets]
        try:
            results = generate_batch(self.engine, jobs)
        except BaseException as e:                          # noqa: BLE001 - every ticket must be told, the thread must survive
            for t in tickets:
                t._finish(e)
            return
        self.waves += 1
        self.batched_requests += len(tickets)
        self.largest_wave = max(self.largest_wave, len(tickets))
        for t, r in zip(tickets, results):
            t.result = r
            t._finish()

    def _run_solo(self, t: Ticket) -> None:
        t.started.set()
        try:
            gen = self.engine.stream(t.prompt_ids, t.cfg)
            cancelled = False
            for delta in gen:
                t._push(delta)
                if t.cancelled():
                    cancelled = True
                    break
            if cancelled:
                gen.close()                                 # last_result would be stale (a previous generation's): no result
            else:
                t.result = self.engine.last_result
            self.solo_requests += 1
            t._finish()
        except BaseException as e:                          # noqa: BLE001
            t._finish(e)
