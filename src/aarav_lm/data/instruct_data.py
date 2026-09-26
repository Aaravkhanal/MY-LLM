"""
aarav-lm · Instruction data and evaluation prompts built from the corpus (no external model)

Training conversations: a real corpus story answers a request whose constraints that story
satisfies. Held-out prompts: requests for stories from the validation split, using request
templates that were never used for training and, for name tasks, names that never occur in the
corpus. Both come from instruct_tasks.py, so training labels and evaluation checks agree.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from aarav_lm.data.instruct_tasks import (
    EVAL_TEMPLATES, NAME_RE, TASK_WEIGHTS, TRAIN_TEMPLATES, UNSEEN_NAMES, Check, applicable_tasks, build_task,
    extract_features, pick_task, run_checks, unseen_name_task,
)

SYSTEMS = ["You are a friendly storyteller for children.", "You tell short, happy stories.",
           "You are a helpful assistant that writes simple stories."]
FOLLOW_UPS = ["Tell me another one.", "Another story, please.", "Can you tell me a different story?", "One more!"]


def _finish(rng: random.Random, task, story: str, short: List[str], system_frac: float,
            two_turn_prob_if_short: float, short_chars: int) -> List[dict]:
    msgs: List[dict] = []
    if rng.random() < system_frac:
        msgs.append({"role": "system", "content": rng.choice(SYSTEMS)})
    msgs += [{"role": "user", "content": task.request}, {"role": "assistant", "content": story}]
    if len(story) <= short_chars and short and rng.random() < two_turn_prob_if_short:
        second = rng.choice(short)
        if second != story:
            msgs += [{"role": "user", "content": rng.choice(FOLLOW_UPS)}, {"role": "assistant", "content": second}]
    return msgs


def _feature_value(kind: str, f) -> object:
    """The value a request of this kind asks for, used to balance sampling across rare and common values."""
    if kind in ("topic", "short_topic", "long_topic", "name_topic", "two_topics"):
        return f.topics[0] if f.topics else None
    if kind == "place":
        return f.place
    if kind in ("emotion", "name_emotion"):
        return f.emotion
    if kind == "name":
        return f.name
    return None


def _prominent(kind: str, f, min_mentions: int, first_mention_max: Optional[int] = None) -> bool:
    """A request for a topic/place/emotion should be about something the story dwells on (`min_mentions`) and, if
    `first_mention_max` is set, introduces early: that makes "a story about a fox" and "the story opens with the
    fox" the same thing, a link a small model can actually learn."""
    if kind in ("topic", "short_topic", "long_topic", "name_topic", "two_topics"):
        count, first = f.topic_mentions, f.topic_first
    elif kind == "place":
        count, first = f.place_mentions, f.place_first
    elif kind in ("emotion", "name_emotion"):
        count, first = f.emotion_mentions, f.emotion_first
    else:
        return True
    return count >= min_mentions and (first_mention_max is None or first <= first_mention_max)


def make_conversations(
    stories: Dict[str, List[str]], n_train: int, seed: int = 0, system_frac: float = 0.3,
    two_turn_prob_if_short: float = 0.5, short_chars: int = 420,
    templates: Dict[str, List[str]] = TRAIN_TEMPLATES, min_mentions: int = 1, balance_alpha: float = 1.0,
    augment: float = 0.0, first_mention_max: Optional[int] = None,
) -> Iterator[dict]:
    """One conversation per story (each story used at most once).

    Training: the task kind is drawn FIRST from TASK_WEIGHTS, then a story that satisfies it is
    drawn. (Picking the story first starves rare kinds: only ~1% of stories are short enough for a
    "very short story" request, so length control would get a few dozen examples.)
    Validation/test: every story of that held-out split, with a task drawn from those it fits.

    Opt-in training-data improvements (the defaults reproduce the original data exactly):
      templates       request wording (TRAIN_TEMPLATES_V4 has ~3x more frames)
      min_mentions    a topic/place/emotion request needs a story that mentions that word at least this often
      first_mention_max  ... and (if set) first mentions it within this many words of the start
      balance_alpha   1.0 = follow the corpus (park is 40% of place requests); 0.5 flattens toward rare values
                      (probability proportional to count**alpha); 0 = uniform over values
      augment         share of requests rewritten in casual typing (lowercase, no final punctuation)
    """
    rng = random.Random(seed)
    for split, pool in stories.items():
        pool = list(pool)
        short = [s for s in pool if len(s) <= short_chars]
        feats = [extract_features(st) for st in pool]
        if split != "train":
            order = list(range(len(pool)))
            rng.shuffle(order)
            for i in order:
                task = pick_task(feats[i], rng)
                yield {"split": split, "task": task.kind,
                       "messages": _finish(rng, task, pool[i], short, system_frac, two_turn_prob_if_short, short_chars)}
            continue
        by_kind: Dict[str, List[int]] = {k: [] for k in TASK_WEIGHTS}
        for i, f in enumerate(feats):
            for k in applicable_tasks(f):
                if (min_mentions <= 1 and first_mention_max is None) or _prominent(k, f, min_mentions, first_mention_max):
                    by_kind[k].append(i)
        for lst in by_kind.values():
            rng.shuffle(lst)
        used: set = set()
        if balance_alpha != 1.0:
            yield from _balanced_train(pool, feats, by_kind, n_train, rng, templates, balance_alpha, augment, short, system_frac,
                                       two_turn_prob_if_short, short_chars)
            continue
        kinds = rng.choices(list(TASK_WEIGHTS), weights=list(TASK_WEIGHTS.values()), k=min(n_train, len(pool)))
        for kind in kinds:
            lst = by_kind[kind]
            while lst and lst[-1] in used:
                lst.pop()
            if not lst:                                   # this kind is exhausted: fall back to a generic request
                kind, lst = "generic", by_kind["generic"]
                while lst and lst[-1] in used:
                    lst.pop()
                if not lst:
                    break
            i = lst.pop()
            used.add(i)
            task = build_task(kind, feats[i], rng, templates, augment)
            yield {"split": split, "task": task.kind,
                   "messages": _finish(rng, task, pool[i], short, system_frac, two_turn_prob_if_short, short_chars)}


def _balanced_train(pool, feats, by_kind, n_train, rng, templates, alpha, augment, short, system_frac,
                    two_turn_prob_if_short, short_chars) -> Iterator[dict]:
    """Same task-first sampling as the default path, but the story is drawn value-first: pick a topic/place/name
    with probability proportional to (its remaining stories) ** alpha, then a story that has it."""
    buckets: Dict[str, Dict[object, List[int]]] = {}
    for kind, lst in by_kind.items():
        b: Dict[object, List[int]] = {}
        for i in lst:                                     # `lst` is already shuffled
            b.setdefault(_feature_value(kind, feats[i]), []).append(i)
        buckets[kind] = b
    used: set = set()

    def draw(kind: str):
        b = buckets[kind]
        while True:
            live = [v for v, l in b.items() if l]
            if not live:
                return None
            v = rng.choices(live, weights=[len(b[x]) ** alpha for x in live])[0]
            while b[v] and b[v][-1] in used:
                b[v].pop()
            if b[v]:
                return b[v].pop()

    kinds = rng.choices(list(TASK_WEIGHTS), weights=list(TASK_WEIGHTS.values()), k=min(n_train, len(pool)))
    for kind in kinds:
        i = draw(kind)
        if i is None:                                     # this kind is exhausted: fall back to a generic request
            kind, i = "generic", draw("generic")
            if i is None:
                break
        used.add(i)
        task = build_task(kind, feats[i], rng, templates, augment)
        yield {"split": "train", "task": task.kind,
               "messages": _finish(rng, task, pool[i], short, system_frac, two_turn_prob_if_short, short_chars)}


def make_eval_prompts(val_stories: List[str], per_kind: int = 16, seed: int = 0,
                      templates: Dict[str, List[str]] = EVAL_TEMPLATES) -> List[dict]:
    """Balanced held-out prompts: `per_kind` for every task kind, built from held-out stories.

    `templates` defaults to phrasings never used in training. Passing TRAIN_TEMPLATES builds the
    in-template control: held-out stories, but request wording the model WAS trained on. Comparing the two
    separates "did not learn the task" from "learned it only for the wording it saw"."""
    rng = random.Random(seed)
    stories = list(val_stories)
    rng.shuffle(stories)
    feats = [(s, extract_features(s)) for s in stories]
    rows: List[dict] = []
    for kind in TASK_WEIGHTS:
        pool = [f for _, f in feats if kind in applicable_tasks(f)]
        if not pool:
            continue
        for i in range(per_kind):
            f = pool[i % len(pool)]
            unseen = kind in ("name", "name_topic", "name_emotion") and i % 2 == 1
            task = (unseen_name_task(kind, UNSEEN_NAMES[(i // 2) % len(UNSEEN_NAMES)], f, rng, templates) if unseen
                    else build_task(kind, f, rng, templates))
            rows.append({"id": f"i{len(rows):03d}", "kind": kind, "prompt": task.request,
                         "checks": [c.to_dict() for c in task.checks], "unseen_name": unseen})
    return rows


def chance_levels(prompts: List[dict], stories: List[str]) -> dict:
    """Pass rate a story that IGNORES the request would get, per task kind.

    Every held-out story is checked against every prompt's rules; the pass rate of a prompt is the share of
    stories that satisfy all of them, and a kind's chance level is the mean over its prompts. Deterministic
    (the whole pool, no sampling). A model's pass rate means little until it is compared with this: for
    example 'the story contains dialogue' is met by about two thirds of ordinary stories.
    """
    cache: Dict[str, float] = {}
    per_kind: Dict[str, List[float]] = {}
    for pr in prompts:
        checks = [Check.from_dict(c) for c in pr.get("checks", [])]
        if not checks:
            continue                                          # format-only prompts are not scored
        key = json.dumps(pr["checks"], sort_keys=True)
        if key not in cache:
            cache[key] = sum(all(run_checks(checks, s).values()) for s in stories) / max(1, len(stories))
        per_kind.setdefault(pr["kind"], []).append(cache[key])
    return {"by_kind": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_kind.items())},
            "n_stories": len(stories),
            "note": "share of held-out stories passing a prompt's checks, averaged over the kind's prompts"}


def write_jsonl(records, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
