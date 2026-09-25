"""
aarav-lm · Story-request conversations built from our own corpus

No external model writes any of this. A conversation is a templated request
("Tell me a story about a girl named Lily.") answered by a real story from the corpus
that actually matches (it contains the name / topic word). That is enough to teach a
story model the *format* of a chat and simple request-following, and it keeps the
answers in the training distribution.

Held-out hygiene: each record carries an explicit "split". Answers for val/test come
only from stories in the *pretraining* val/test splits, which the model never trained on.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from aarav_lm.data.clean import normalize_text, quality_reject_reason
from aarav_lm.data.dedup import ExactDeduper
from aarav_lm.data.ingest import IngestStats, iter_documents
from aarav_lm.data.split import split_of

NAME_RE = re.compile(r"\bnamed ([A-Z][a-z]{2,})\b")
TOPICS = ["dog", "cat", "bird", "rabbit", "fish", "dragon", "bear", "girl", "boy", "tree", "ball", "car",
          "garden", "park", "duck", "frog", "monkey", "princess", "robot", "boat", "cake", "star"]

NAME_TEMPLATES = [
    "Tell me a story about {name}.", "Can you tell me a story about a character named {name}?",
    "Write a short story about {name}.", "I want to hear about {name}. Please tell a story.",
    "Story about {name}, please!", "Please write a story with {name} in it.",
]
TOPIC_TEMPLATES = [
    "Tell me a story about a {topic}.", "Write a short story with a {topic} in it.",
    "Can you tell me a story about a {topic}?", "I would like a story about a {topic}.",
    "Please tell me a story with a {topic}.", "Story about a {topic}, please!",
]
GENERIC_TEMPLATES = [
    "Tell me a story.", "Tell me a short story.", "Can you tell me a bedtime story?",
    "I want a story!", "Write a story for kids.", "Please tell me something fun.",
]
FOLLOW_UPS = ["Tell me another one.", "Another story, please.", "Can you tell me a different story?", "One more!"]
SYSTEMS = ["You are a friendly storyteller for children.", "You tell short, happy stories.",
           "You are a helpful assistant that writes simple stories."]


def _features(story: str) -> Dict[str, Optional[str]]:
    m = NAME_RE.search(story[:400])
    head = story[:300].lower()
    hits = [(hit.start(), t) for t in TOPICS if (hit := re.search(rf"\b{t}\b", head))]
    return {"name": m.group(1) if m else None, "topic": min(hits)[1] if hits else None}   # earliest mention


def _request(feat: Dict[str, Optional[str]], rng: random.Random) -> str:
    if feat["name"] and rng.random() < 0.6:
        return rng.choice(NAME_TEMPLATES).format(name=feat["name"])
    if feat["topic"] and rng.random() < 0.8:
        return rng.choice(TOPIC_TEMPLATES).format(topic=feat["topic"])
    return rng.choice(GENERIC_TEMPLATES)


def load_stories(
    sources: Sequence[str | Path], seed: int, val_fraction: float, test_fraction: float,
    min_chars: int = 250, max_chars: int = 820,
) -> Dict[str, List[str]]:
    """Cleaned, de-duplicated stories grouped by their *pretraining* split."""
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    dedup = ExactDeduper()
    for raw in iter_documents(sources, IngestStats()):
        text = normalize_text(raw.text)
        if quality_reject_reason(text, min_chars=min_chars, max_chars=max_chars) or dedup.is_duplicate(text):
            continue
        out[split_of(text, seed, val_fraction, test_fraction)].append(text)
    return out


def make_conversations(
    stories: Dict[str, List[str]], n_train: int, seed: int = 0, system_frac: float = 0.3,
    two_turn_prob_if_short: float = 0.65, short_chars: int = 380,
) -> Iterator[dict]:
    rng = random.Random(seed)
    for split, pool in stories.items():
        pool = list(pool)
        rng.shuffle(pool)
        limit = n_train if split == "train" else len(pool)
        short = [s for s in pool if len(s) <= short_chars]
        for i, story in enumerate(pool[:limit]):
            msgs = []
            if rng.random() < system_frac:
                msgs.append({"role": "system", "content": rng.choice(SYSTEMS)})
            msgs += [{"role": "user", "content": _request(_features(story), rng)},
                     {"role": "assistant", "content": story}]
            if len(story) <= short_chars and short and rng.random() < two_turn_prob_if_short:
                second = rng.choice(short)
                if second != story:
                    msgs += [{"role": "user", "content": rng.choice(FOLLOW_UPS)},
                             {"role": "assistant", "content": second}]
            yield {"split": split, "messages": msgs}


def write_jsonl(records: Iterator[dict], path: str | Path) -> Dict[str, int]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            counts[r["split"]] = counts.get(r["split"], 0) + 1
    return counts
