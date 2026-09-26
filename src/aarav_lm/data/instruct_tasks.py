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

_QUESTION_OPENERS = ("Can you", "Could you", "Would you", "Can I have", "Do you know", "How about")


def _frames(openers: Sequence[str], bodies: Sequence[str]) -> List[str]:
    """Cross request openers with story descriptions, e.g. 'Please write' x 'a short story' -> many distinct requests.

    A request that is a question ("Can you ...") ends in '?', the rest in '.' or '!' variants, so the phrasing pool
    is large enough that the model must learn what 'short' or 'about a dog' MEANS rather than memorise a few strings.
    """
    out = []
    for o in openers:
        for body in bodies:
            if o.startswith(_QUESTION_OPENERS):
                out.append(f"{o} {body}?")
            else:
                out += [f"{o} {body}.", f"{o} {body}, please."]
    return out


_TRAIN_OPEN = ["Write", "Tell me", "I want", "Please write", "Please tell me", "I'd like", "Can you write", "Can you tell me"]

def _train_templates(openers: List[str]) -> Dict[str, List[str]]:
    return {
    "name": _frames(openers, ["a story about {name}", "a story with {name} in it", "a story about a character named {name}",
                                  "a tale about {name}"]) + ["Story about {name}, please!"],
    "topic": _frames(openers, ["a story about a {topic}", "a story with a {topic} in it", "a tale about a {topic}",
                                   "something about a {topic}"]) + ["Story about a {topic}, please!"],
    "name_topic": _frames(openers, ["a story about {name} and a {topic}", "a story where {name} finds a {topic}",
                                        "a story with {name} and a {topic}"]),
    "two_topics": _frames(openers, ["a story about a {topic} and a {topic2}", "a story with a {topic} and a {topic2} in it",
                                        "a tale about a {topic} and a {topic2}"]),
    "place": _frames(openers, ["a story set at the {place}", "a story that takes place at the {place}",
                                   "a story about the {place}", "a story about a day at the {place}"]),
    "emotion": _frames(openers, ["a story where someone feels {emotion}", "a story about feeling {emotion}",
                                     "a {emotion} story"]),
    "name_emotion": _frames(openers, ["a story about {name}, who feels {emotion}", "a story about {name} feeling {emotion}"]),
    "short": _frames(openers, ["a very short story", "a short story", "a tiny story", "a quick story", "a really short story",
                                   "a super short story"]),
    "long": _frames(openers, ["a long story", "a longer story", "a really long story", "a very long story",
                                  "a story that goes on for a while", "a big long story"]),
    "short_topic": _frames(openers, ["a very short story about a {topic}", "a quick story about a {topic}",
                                         "a short story with a {topic} in it", "a tiny tale about a {topic}"]),
    "long_topic": _frames(openers, ["a long story about a {topic}", "a longer story with a {topic} in it",
                                        "a really long story about a {topic}", "a big long tale about a {topic}"]),
    "dialogue": _frames(openers, ["a story with lots of talking", "a story where the characters talk a lot",
                                      "a story with lots of dialogue", "a story with a lot of speaking",
                                      "a story where everyone chats"]),
    "generic": ["Tell me a story.", "Tell me a short story.", "Can you tell me a bedtime story?", "I want a story!",
                "Write a story for kids.", "Please tell me something fun.", "Write a story.", "Tell me a fun story.",
                "I'd like a bedtime story.", "Please write a story for children.", "Can you tell me a story?",
                "Tell me a happy story.", "Story time, please!", "Write something for a child."],
}


TRAIN_TEMPLATES: Dict[str, List[str]] = _train_templates(_TRAIN_OPEN)

# v4: many more request frames (openers). Still none of them equals an eval string (tests/test_instruct.py), and the
# eval openers ("Give me", "Compose", "Create", "Make up", "Could you", "I'm in the mood for") are deliberately absent.
_TRAIN_OPEN_V4 = _TRAIN_OPEN + [
    "Would you please write", "I need", "Let's have", "Time for", "Show me", "Imagine", "Can I have", "I'd love",
    "How about", "Do you know", "Please make me", "I wish for", "Let me hear", "Please share", "Go ahead and write",
]
TRAIN_TEMPLATES_V4: Dict[str, List[str]] = _train_templates(_TRAIN_OPEN_V4)

# Phrasings never seen in training: new openers ("Give me", "Compose", "Create", "Make up", "Could you"), and for the
# length/dialogue requests both new frames around known keywords and synonyms the model never saw ("brief", "lengthy").
EVAL_TEMPLATES: Dict[str, List[str]] = {
    "name": ["Could you make up a tale starring {name}?", "Write me something about a girl or boy called {name}.",
             "Give me a story featuring {name}.", "Compose a tale in which {name} is the hero."],
    "topic": ["Give me a story featuring a {topic}.", "I'm in the mood for a story that has a {topic}.",
              "Compose a tale involving a {topic}.", "Make up something with a {topic} in the story."],
    "name_topic": ["Make up a story in which {name} meets a {topic}.", "Give me a tale starring {name} and a {topic}."],
    "two_topics": ["Create a tale involving both a {topic} and a {topic2}.", "Give me a story that features a {topic} and a {topic2}."],
    "place": ["Give me a story happening at the {place}.", "Compose a tale that unfolds at the {place}."],
    "emotion": ["Compose a story in which somebody is {emotion}.", "Give me a tale about a {emotion} character."],
    "name_emotion": ["Give me a story where {name} is {emotion}.", "Compose a tale in which {name} feels {emotion}."],
    "short": ["Give me a brief little story.", "Compose a short tale.", "Create a quick story for me.",
              "Make up a concise story.", "Could you create a bite-sized story?", "Give me a teeny story."],
    "long": ["Give me a nice long story.", "Compose a lengthy tale.", "Create a long story for me.",
             "Make up a long, detailed story.", "Could you create an extended story?", "Give me an epic long story."],
    "short_topic": ["A brief story about a {topic}, please.", "Compose a short tale about a {topic}.",
                    "Create a quick story with a {topic}."],
    "long_topic": ["A long, detailed story about a {topic}, please.", "Compose a lengthy tale about a {topic}.",
                   "Create a long story with a {topic}."],
    "dialogue": ["Write a story that is full of conversation.", "Compose a tale with plenty of dialogue.",
                 "Make up a story where people chat a lot.", "Give me a story with lots of speaking parts."],
    "generic": ["Surprise me with a story.", "Any story will do.", "Give me a story.", "Compose a tale."],
}

# A THIRD family of wordings, for the final held-out set (evals/instruct_v4_final.jsonl). Written and committed
# before any model was scored on it: the v3 eval set has been looked at while designing fixes, so improvements are
# confirmed here. New openers throughout ("Craft", "Spin", "Got", "Dream up", "Narrate", "Whip up") and new length words.
EVAL_TEMPLATES_FINAL: Dict[str, List[str]] = {
    "name": ["Craft a tale featuring {name}.", "Spin a yarn starring {name}.", "Got a story about someone called {name}?",
             "Dream up an adventure for {name}.", "Narrate the adventures of {name}."],
    "topic": ["Craft a tale featuring a {topic}.", "Spin a yarn about a {topic}.", "Got a story with a {topic} in it?",
              "Dream up an adventure involving a {topic}.", "Narrate a tale centred on a {topic}."],
    "name_topic": ["Craft a tale where {name} meets a {topic}.", "Spin a yarn about {name} and a {topic}.",
                   "Got a story about {name} finding a {topic}?"],
    "two_topics": ["Craft a tale featuring both a {topic} and a {topic2}.", "Spin a yarn that has a {topic} and a {topic2} in it.",
                   "Got a story involving a {topic} and a {topic2}?"],
    "place": ["Craft a tale that happens at the {place}.", "Spin a yarn set at the {place}.", "Got a story that takes us to the {place}?"],
    "emotion": ["Craft a tale about someone who is {emotion}.", "Spin a yarn where the hero feels {emotion}.",
                "Got a story about a {emotion} child?"],
    "name_emotion": ["Craft a tale in which {name} is {emotion}.", "Spin a yarn about {name} feeling {emotion}."],
    "short": ["Craft a tiny tale.", "Spin a brief yarn.", "Got a really short story?", "Dream up a quick little story.",
              "Narrate a compact tale.", "Whip up a mini story."],
    "long": ["Craft an epic long tale.", "Spin a lengthy yarn.", "Got a nice long story?", "Dream up a big, long story.",
             "Narrate a drawn-out tale.", "Whip up a long adventure story."],
    "short_topic": ["Craft a tiny tale about a {topic}.", "Spin a brief yarn with a {topic}.", "Got a really short story about a {topic}?"],
    "long_topic": ["Craft an epic long tale about a {topic}.", "Spin a lengthy yarn with a {topic}.",
                   "Got a nice long story about a {topic}?"],
    "dialogue": ["Craft a tale packed with dialogue.", "Spin a yarn where the characters chat constantly.",
                 "Got a story with lots of back-and-forth talk?", "Narrate a story full of speech."],
    "generic": ["Craft a tale.", "Got a story?", "Spin me a yarn.", "Any tale at all."],
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
    topic_mentions: int = 0       # how often the main topic / place / emotion word occurs: 1 = a passing mention
    place_mentions: int = 0
    emotion_mentions: int = 0
    topic_first: int = 10**9      # word index of the first mention (small = the story opens with it)
    place_first: int = 10**9
    emotion_first: int = 10**9
    topic2_mentions: int = 0      # the SECOND topic word, for two-topic requests (both must be things the story dwells on)
    topic2_first: int = 10**9


def extract_features(story: str) -> Features:
    m = NAME_RE.search(story[:500])
    counts = Counter(w.lower() for w in WORD_RE.findall(story))
    topics = sorted((n for n in NOUNS if counts[n] + counts[n + "s"] >= 1),
                    key=lambda n: (-(counts[n] + counts[n + "s"]), story.lower().find(n)))
    place = next((p for p in sorted(PLACES, key=lambda p: -counts[p]) if counts[p] >= 1), None)
    emotion = next((e for e in sorted(EMOTIONS, key=lambda e: -counts[e]) if counts[e] >= 1), None)
    first: Dict[str, int] = {}
    for i, w in enumerate(w.lower() for w in WORD_RE.findall(story)):
        first.setdefault(w, i)
    never = 10**9
    return Features(m.group(1) if m else None, topics, place, emotion, word_count(story), has_dialogue(story),
                    (counts[topics[0]] + counts[topics[0] + "s"]) if topics else 0,
                    counts[place] if place else 0, counts[emotion] if emotion else 0,
                    min(first.get(topics[0], never), first.get(topics[0] + "s", never)) if topics else never,
                    first.get(place, never) if place else never, first.get(emotion, never) if emotion else never,
                    (counts[topics[1]] + counts[topics[1] + "s"]) if len(topics) > 1 else 0,
                    min(first.get(topics[1], never), first.get(topics[1] + "s", never)) if len(topics) > 1 else never)


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


def _casual(request: str, keep: Sequence[str], rng: random.Random) -> str:
    """How people really type: drop the final punctuation and/or lowercase everything (names stay capitalised,
    so the check that the name appears still means the same thing)."""
    t = request
    if rng.random() < 0.5:
        t = t.rstrip(".!?")
    if rng.random() < 0.5:
        t = t.lower()
        for w in keep:
            t = t.replace(w.lower(), w)
    return t


def build_task(kind: str, f: Features, rng: random.Random, templates: Dict[str, List[str]], augment: float = 0.0) -> Task:
    """Instantiate `kind` for features `f`. Slots are filled from the story so its own text satisfies the checks.
    With `augment` > 0, that share of requests is rewritten in casual typing (training only)."""
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
    request = rng.choice(templates[kind]).format(**slots)
    if augment > 0 and rng.random() < augment:
        request = _casual(request, [slots["name"]] if "name" in slots else [], rng)
    return Task(kind, request, checks)


def pick_task(f: Features, rng: random.Random, templates: Dict[str, List[str]] = TRAIN_TEMPLATES) -> Task:
    kinds = applicable_tasks(f)
    kind = rng.choices(kinds, weights=[TASK_WEIGHTS[k] for k in kinds])[0]
    return build_task(kind, f, rng, templates)


def unseen_name_task(kind: str, name: str, f: Features, rng: random.Random, templates) -> Task:
    """Like build_task, but with a name that never occurs in the corpus (tests copying from the prompt)."""
    g = Features(name=name, topics=f.topics, place=f.place, emotion=f.emotion, words=f.words, dialogue=f.dialogue)
    return build_task(kind, g, rng, templates)
