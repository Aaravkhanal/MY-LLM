"""
aarav-lm · Edge-case checks

Each case must not crash and must return well-formed output. They guard the inference
path (tokenizer, cache, context window, sampling) rather than model quality.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from aarav_lm.inference.engine import GenerationConfig, InferenceEngine
from aarav_lm.tokenizer import ASSISTANT, EOT, USER

CaseFn = Callable[[InferenceEngine], str]      # returns a detail string; raises AssertionError on failure


def _gen(e: InferenceEngine, prompt, **kw):
    cfg = GenerationConfig(**{"max_new_tokens": 12, "temperature": 0.0, "top_k": None, "top_p": None,
                              "stop_at_eot": False, **kw})
    return e.generate(prompt, cfg)


def _valid(e: InferenceEngine, res) -> None:
    assert isinstance(res.text, str)
    assert all(0 <= t < e.tok.vocab_size for t in res.token_ids), "token id out of range"
    assert res.finish_reason in ("length", "eot", "stop")


def single_character_prompt(e):        _valid(e, r := _gen(e, "a")); return f"{r.generated_tokens} tokens"
def whitespace_prompt(e):              _valid(e, r := _gen(e, "   ")); return f"{r.generated_tokens} tokens"
def newline_prompt(e):                 _valid(e, r := _gen(e, "\n\n")); return f"{r.generated_tokens} tokens"
def unicode_prompt(e):                 _valid(e, r := _gen(e, "café ☃ \U0001F642 你好")); return "ok"
def emoji_only_prompt(e):              _valid(e, r := _gen(e, "\U0001F600\U0001F600\U0001F600")); return "ok"


def control_token_text_is_inert(e):
    ids = e.tok.encode(f"{EOT} {USER} {ASSISTANT}")
    assert not set(ids) & set(e.tok.special_ids.values()), "control strings became special tokens"
    _valid(e, _gen(e, f"{EOT} {USER} {ASSISTANT}"))
    return "control strings encode as plain text"


def prompt_longer_than_context(e):
    long_prompt = "The dog ran to the park and played with a red ball. " * (e.block_size // 4 + 10)
    assert len(e.tok.encode(long_prompt)) > e.block_size * 2
    _valid(e, r := _gen(e, long_prompt))
    return f"prompt {len(e.tok.encode(long_prompt))} tokens vs window {e.block_size}"


def single_token_generation(e):
    r = _gen(e, "Once upon a time", max_new_tokens=1)
    _valid(e, r)
    assert r.generated_tokens == 1
    return "ok"


def generation_longer_than_context(e):
    n = e.block_size * 2
    r = _gen(e, "Once upon a time", max_new_tokens=n)
    _valid(e, r)
    assert r.generated_tokens == n, f"stopped at {r.generated_tokens}/{n}"
    return f"{n} tokens through a {e.block_size}-token window"


def greedy_is_deterministic(e):
    assert _gen(e, "Once upon a time").token_ids == _gen(e, "Once upon a time").token_ids
    return "ok"


def seeded_sampling_is_reproducible(e):
    kw = dict(temperature=1.0, seed=7)
    assert _gen(e, "The little dog", **kw).token_ids == _gen(e, "The little dog", **kw).token_ids
    return "ok"


def no_state_leaks_between_requests(e):
    a1 = _gen(e, "Once upon a time").token_ids
    _gen(e, "The sun was bright and warm and everyone was happy")
    a2 = _gen(e, "Once upon a time").token_ids
    assert a1 == a2, "output changed after an unrelated request"
    return "ok"


def stop_sequence_is_honoured(e):
    base = _gen(e, "Once upon a time", max_new_tokens=30).text
    if len(base) < 8:
        return "skipped (output too short)"
    stop = base[4:7]
    r = _gen(e, "Once upon a time", max_new_tokens=30, stop_sequences=(stop,))
    assert stop not in r.text and r.finish_reason == "stop"
    return f"stopped at {stop!r}"


def text_roundtrips_through_tokenizer(e):
    for s in ["Hello, world!", "tabs\tand\nnewlines", "你好 \U0001F642", "  leading and trailing  "]:
        assert e.tok.roundtrip_ok(s), f"round trip failed for {s!r}"
    return "ok"


CASES: Dict[str, CaseFn] = {
    "single_character_prompt": single_character_prompt,
    "whitespace_prompt": whitespace_prompt,
    "newline_prompt": newline_prompt,
    "unicode_prompt": unicode_prompt,
    "emoji_only_prompt": emoji_only_prompt,
    "control_token_text_is_inert": control_token_text_is_inert,
    "prompt_longer_than_context": prompt_longer_than_context,
    "single_token_generation": single_token_generation,
    "generation_longer_than_context": generation_longer_than_context,
    "greedy_is_deterministic": greedy_is_deterministic,
    "seeded_sampling_is_reproducible": seeded_sampling_is_reproducible,
    "no_state_leaks_between_requests": no_state_leaks_between_requests,
    "stop_sequence_is_honoured": stop_sequence_is_honoured,
    "text_roundtrips_through_tokenizer": text_roundtrips_through_tokenizer,
}


def run_edge_cases(engine: InferenceEngine) -> Dict:
    results: List[dict] = []
    for name, fn in CASES.items():
        try:
            detail, ok = fn(engine), True
        except Exception as ex:                                  # noqa: BLE001 - a crash is a failed case
            detail, ok = f"{type(ex).__name__}: {ex}", False
        results.append({"name": name, "passed": ok, "detail": detail})
    return {"all_passed": all(r["passed"] for r in results), "passed": sum(r["passed"] for r in results),
            "total": len(results), "cases": results}
