"""
aarav-lm · Verifiable story-request tasks

A task is a request ("Write a very short story about a dragon.") plus machine-checkable
constraints on the reply (mentions "dragon", at most 90 words). The same definitions serve two
purposes, which keeps them honest:

  training data   requests are answered by REAL corpus stories that satisfy the constraints, so the
                  labels are correct by construction (no external model writes anything)
  evaluation      a model's own reply is scored against the same checks

Training and held-out evaluation use disjoint request templates, so evaluation measures whether the
model follows the *constraint*, not whether it memorised a phrasing.

Checks (all rule-based, no judge model):
  contains_all   every word appears (whole word, optional plural "s")
  contains_any   at least one of the words appears
  max_words / min_words   length of the reply
  has_dialogue   the reply contains quoted speech
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

WORD_RE = re.compile(r"[A-Za-z']+")
NAME_RE = re.compile(r"\bnamed ([A-Z][a-z]{2,})\b")

NOUNS = [
    "dog", "cat", "bird", "rabbit", "fish", "dragon", "bear", "duck", "frog", "robot", "monkey", "horse",
    "mouse", "elephant", "lion", "turtle", "butterfly", "princess", "king", "wizard", "pirate", "fairy",
    "teacher", "farmer", "boat", "car", "train", "kite", "balloon", "cake", "cookie", "apple", "flower",
    "tree", "star", "moon", "rainbow", "castle", "treasure", "toy", "doll", "ball", "book", "hat", "shoe",
    "box", "key", "map", "bell", "drum", "puppy", "kitten", "snowman", "garden", "cave", "bridge",
]
PLACES = ["park", "forest", "garden", "school", "beach", "farm", "zoo", "store", "kitchen", "library",
          "castle", "village", "river", "mountain", "playground", "town", "cave", "pond"]
EMOTIONS = ["sad", "scared", "angry", "proud", "lonely", "curious", "brave", "surprised", "excited", "nervous"]

SHORT_MAX_WORDS = 90
LONG_MIN_WORDS = 190

# ── request templates (train and eval are deliberately disjoint) ─────────────

TRAIN_TEMPLATES: Dict[str, List[str]] = {
    "name": ["Tell me a story about {name}.", "Can you tell me a story about a character named {name}?",
             "Write a short story about {name}.", "I want to hear about {name}. Please tell a story.",
             "Story about {name}, please!", "Please write a story with {name} in it."],
    "topic": ["Tell me a story about a {topic}.", "Write a short story with a {topic} in it.",
              "Can you tell me a story about a {topic}?", "I would like a story about a {topic}.",
              "Please tell me a story with a {topic}.", "Story about a {topic}, please!"],
    "name_topic": ["Tell me a story about {name} and a {topic}.", "Write a story about {name} who finds a {topic}.",
                   "Can you write a story with {name} and a {topic}?"],
    "two_topics": ["Write a story about a {topic} and a {topic2}.", "Tell me a story with a {topic} and a {topic2} in it.",
                   "I want a story about a {topic} and a {topic2}."],
    "place": ["Write a story set at the {place}.", "Tell me a story that takes place at the {place}.",
              "I want a story about the {place}."],
    "emotion": ["Write a story where someone feels {emotion}.", "Tell me a story about feeling {emotion}.",
                "Can you write a {emotion} story?"],
    "name_emotion": ["Write a story about {name}, who feels {emotion}.", "Tell me a story about {name} feeling {emotion}."],
    "short": ["Write a very short story.", "Tell me a quick, short story.", "A tiny story, please!"],
    "long": ["Write a long story.", "Tell me a long story.", "I want a longer story, please."],
    "short_topic": ["Write a very short story about a {topic}.", "Tell me a quick story about a {topic}."],
    "long_topic": ["Write a long story about a {topic}.", "Tell me a longer story with a {topic} in it."],
    "dialogue": ["Write a story with lots of talking.", "Tell me a story where the characters talk a lot."],
    "generic": ["Tell me a story.", "Tell me a short story.", "Can you tell me a bedtime story?", "I want a story!",
                "Write a story for kids.", "Please tell me something fun."],
}

EVAL_TEMPLATES: Dict[str, List[str]] = {           # phrasings never seen in training
    "name": ["Could you make up a tale starring {name}?", "Write me something about a girl or boy called {name}."],
    "topic": ["Give me a story featuring a {topic}.", "I'm in the mood for a story that has a {topic}."],
    "name_topic": ["Make up a story in which {name} meets a {topic}."],
    "two_topics": ["Create a tale involving both a {topic} and a {topic2}."],
    "place": ["Give me a story happening at the {place}."],
    "emotion": ["Compose a story in which somebody is {emotion}."],
    "name_emotion": ["Give me a story where {name} is {emotion}."],
    "short": ["Give me a brief little story."],
    "long": ["Give me a nice long story."],
    "short_topic": ["A brief story about a {topic}, please."],
    "long_topic": ["A long, detailed story about a {topic}, please."],
    "dialogue": ["Write a story that is full of conversation."],
    "generic": ["Surprise me with a story.", "Any story will do."],
}

TASK_WEIGHTS = {"name": 15, "topic": 15, "name_topic": 10, "two_topics": 8, "place": 9, "emotion": 8,
                "name_emotion": 6, "short": 5, "long": 5, "short_topic": 4, "long_topic": 4, "dialogue": 5, "generic": 10}

# Names that are rare or absent in the corpus, to test copying a name that was never seen.
UNSEEN_NAMES = ["Zara", "Otis", "Ingrid", "Kofi", "Mateo", "Priya", "Bram", "Yuki", "Nadia", "Felix",
                "Odette", "Rafi", "Tilda", "Anselm", "Marisol"]


# ── checks ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Check:
    kind: str
    words: Tuple[str, ...] = ()
    n: int = 0

    def to_dict(self) -> dict:
        return {"kind": self.kind, "words": list(self.words), "n": self.n}

    @staticmethod
    def from_dict(d: dict) -> "Check":
        return Check(d["kind"], tuple(d.get("words", ())), int(d.get("n", 0)))


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}(?:s|es)?\b", text, flags=re.IGNORECASE) is not None


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def has_dialogue(text: str) -> bool:
    return len(re.findall(r'["“”]', text)) >= 2


def run_check(check: Check, text: str) -> bool:
    if check.kind == "contains_all":
        return all(_has_word(text, w) for w in check.words)
    if check.kind == "contains_any":
        return any(_has_word(text, w) for w in check.words)
    if check.kind == "max_words":
        return 0 < word_count(text) <= check.n
    if check.kind == "min_words":
        return word_count(text) >= check.n
    if check.kind == "has_dialogue":
        return has_dialogue(text)
    raise ValueError(f"unknown check kind {check.kind!r}")


def run_checks(checks: Sequence[Check], text: str) -> Dict[str, bool]:
    """Result per check (keyed by kind; a repeated kind must all pass)."""
    out: Dict[str, bool] = {}
    for c in checks:
        out[c.kind] = out.get(c.kind, True) and run_check(c, text)
    return out


# ── story features ───────────────────────────────────────────────────────────

@dataclass
class Features:
    name: Optional[str] = None
    topics: List[str] = field(default_factory=list)      # most frequent first
    place: Optional[str] = None
    emotion: Optional[str] = None
    words: int = 0
    dialogue: bool = False


def extract_features(story: str) -> Features:
    m = NAME_RE.search(story[:500])
    counts = Counter(w.lower() for w in WORD_RE.findall(story))
    topics = sorted((n for n in NOUNS if counts[n] + counts[n + "s"] >= 1),
                    key=lambda n: (-(counts[n] + counts[n + "s"]), story.lower().find(n)))
    place = next((p for p in sorted(PLACES, key=lambda p: -counts[p]) if counts[p] >= 1), None)
    emotion = next((e for e in sorted(EMOTIONS, key=lambda e: -counts[e]) if counts[e] >= 1), None)
    return Features(m.group(1) if m else None, topics, place, emotion, word_count(story), has_dialogue(story))


def applicable_tasks(f: Features) -> List[str]:
    t = ["generic"]
    if f.name:
        t.append("name")
    if f.topics:
        t.append("topic")
    if f.name and f.topics:
        t.append("name_topic")
    if len(f.topics) >= 2:
        t.append("two_topics")
    if f.place:
        t.append("place")
    if f.emotion:
        t.append("emotion")
    if f.name and f.emotion:
        t.append("name_emotion")
    if f.words <= SHORT_MAX_WORDS:
        t.append("short")
        if f.topics:
            t.append("short_topic")
    if f.words >= LONG_MIN_WORDS:
        t.append("long")
        if f.topics:
            t.append("long_topic")
    if f.dialogue:
        t.append("dialogue")
    return t


@dataclass
class Task:
    kind: str
    request: str
    checks: List[Check]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "request": self.request, "checks": [c.to_dict() for c in self.checks]}


def build_task(kind: str, f: Features, rng: random.Random, templates: Dict[str, List[str]]) -> Task:
    """Instantiate `kind` for features `f`. Slots are filled from the story so its own text satisfies the checks."""
    slots: Dict[str, str] = {}
    checks: List[Check] = []
    if kind in ("name", "name_topic", "name_emotion"):
        slots["name"] = f.name
        checks.append(Check("contains_all", (f.name,)))
    if kind in ("topic", "name_topic", "short_topic", "long_topic"):
        slots["topic"] = f.topics[0]
        checks.append(Check("contains_all", (f.topics[0],)))
    if kind == "two_topics":
        slots["topic"], slots["topic2"] = f.topics[0], f.topics[1]
        checks.append(Check("contains_all", (f.topics[0], f.topics[1])))
    if kind == "place":
        slots["place"] = f.place
        checks.append(Check("contains_all", (f.place,)))
    if kind in ("emotion", "name_emotion"):
        slots["emotion"] = f.emotion
        checks.append(Check("contains_all", (f.emotion,)))
    if kind in ("short", "short_topic"):
        checks.append(Check("max_words", n=SHORT_MAX_WORDS))
    if kind in ("long", "long_topic"):
        checks.append(Check("min_words", n=LONG_MIN_WORDS))
    if kind == "dialogue":
        checks.append(Check("has_dialogue"))
    return Task(kind, rng.choice(templates[kind]).format(**slots), checks)


def pick_task(f: Features, rng: random.Random, templates: Dict[str, List[str]] = TRAIN_TEMPLATES) -> Task:
    kinds = applicable_tasks(f)
    kind = rng.choices(kinds, weights=[TASK_WEIGHTS[k] for k in kinds])[0]
    return build_task(kind, f, rng, templates)


def unseen_name_task(kind: str, name: str, f: Features, rng: random.Random, templates) -> Task:
    """Like build_task, but with a name that never occurs in the corpus (tests copying from the prompt)."""
    g = Features(name=name, topics=f.topics, place=f.place, emotion=f.emotion, words=f.words, dialogue=f.dialogue)
    return build_task(kind, g, rng, templates)
