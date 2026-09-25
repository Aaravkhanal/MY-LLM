"""
aarav-lm · Text-quality metrics (pure functions, no model needed)

These are *proxies*. There is no judge model here on purpose (nothing external is
called), so coherence is approximated from measurable properties of the text:

  repetition   distinct-n ratios, degenerate loops, longest single-token run
  form         sentence capitalisation / termination, non-word rate
  coherence    whether a character introduced in the prompt is still mentioned
  length       tokens / words / characters, and how often generation ends naturally
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_END = ".!?\"”’'"
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def words(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def distinct_n(tokens: Sequence, n: int) -> float:
    """Unique n-grams / total n-grams (1.0 = no repetition). Undefined (1.0) for short input."""
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def max_token_run(tokens: Sequence) -> int:
    """Longest run of one identical token repeated back-to-back."""
    best = run = 0
    prev = object()
    for t in tokens:
        run = run + 1 if t == prev else 1
        prev = t
        best = max(best, run)
    return best


def has_loop(tokens: Sequence, min_repeats: int = 3, max_period: int = 20) -> bool:
    """True if some chunk of 1..max_period tokens repeats `min_repeats` times in a row."""
    n = len(tokens)
    for p in range(1, max_period + 1):
        if p * min_repeats > n:
            break
        streak = 0
        for i in range(n - p):
            if tokens[i] == tokens[i + p]:
                streak += 1
                if streak >= p * (min_repeats - 1):
                    return True
            else:
                streak = 0
    return False


def sentence_stats(text: str) -> Dict[str, float]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    if not sents:
        return {"sentences": 0, "capitalised_frac": 0.0}
    cap = sum(1 for s in sents if s.lstrip("\"“'(")[:1].isupper())
    return {"sentences": len(sents), "capitalised_frac": cap / len(sents)}


def ends_cleanly(text: str) -> bool:
    t = text.rstrip()
    return bool(t) and t[-1] in SENTENCE_END


def nonword_rate(text: str, vocabulary: Set[str]) -> float:
    """Fraction of alphabetic words not found in the training vocabulary."""
    ws = words(text)
    if not ws:
        return 0.0
    return sum(1 for w in ws if w.strip("'") not in vocabulary) / len(ws)


def mentions(text: str, entity: str) -> bool:
    return re.search(rf"\b{re.escape(entity)}\b", text, flags=re.IGNORECASE) is not None


def build_vocabulary(texts: Iterable[str], min_count: int = 3) -> Set[str]:
    c: Counter = Counter()
    for t in texts:
        c.update(w.strip("'") for w in words(t))
    return {w for w, k in c.items() if k >= min_count and w}


def summarize_generations(
    samples: Sequence[Dict], vocabulary: Set[str]
) -> Dict[str, float]:
    """Aggregate quality metrics over generations.

    Each sample: {"text": str, "tokens": [ids], "finish_reason": str, "entity": str|None, "prompt": str}
    """
    if not samples:
        raise ValueError("no samples to summarise")
    n = len(samples)
    texts = [s["text"] for s in samples]
    word_lists = [words(t) for t in texts]
    all_words = [w for ws in word_lists for w in ws]
    ent = [s for s in samples if s.get("entity")]
    sent = [sentence_stats(t) for t in texts]
    tot_sent = sum(x["sentences"] for x in sent)
    return {
        "samples": n,
        "distinct_1": distinct_n(all_words, 1),
        "distinct_2": distinct_n(all_words, 2),
        "distinct_3": distinct_n(all_words, 3),
        "mean_sample_distinct_2": sum(distinct_n(ws, 2) for ws in word_lists) / n,
        "loop_fraction": sum(has_loop(s["tokens"]) for s in samples) / n,
        "max_token_run": max(max_token_run(s["tokens"]) for s in samples),
        "nonword_rate": nonword_rate(" ".join(texts), vocabulary),
        "capitalised_sentence_frac": (
            sum(x["capitalised_frac"] * x["sentences"] for x in sent) / tot_sent if tot_sent else 0.0
        ),
        "clean_ending_frac": sum(ends_cleanly(t) for t in texts) / n,
        "entity_consistency": (sum(mentions(s["text"], s["entity"]) for s in ent) / len(ent)) if ent else None,
        "eot_rate": sum(s["finish_reason"] == "eot" for s in samples) / n,
        # chat models end a reply with <|end|> ("stop"), base models with <|endoftext|> ("eot"):
        # either way the model chose to finish rather than being cut off at the token limit
        "natural_end_rate": sum(s["finish_reason"] in ("eot", "stop") for s in samples) / n,
        "mean_tokens": sum(len(s["tokens"]) for s in samples) / n,
        "mean_words": len(all_words) / n,
        "mean_chars": sum(len(t) for t in texts) / n,
        "empty_fraction": sum(not t.strip() for t in texts) / n,
    }
