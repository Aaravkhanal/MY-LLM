import random

import pytest

from aarav_lm.data.instruct_data import make_conversations, make_eval_prompts
from aarav_lm.data.instruct_tasks import (
    EVAL_TEMPLATES, LONG_MIN_WORDS, SHORT_MAX_WORDS, TASK_WEIGHTS, TRAIN_TEMPLATES, Check, applicable_tasks,
    build_task, extract_features, has_dialogue, pick_task, run_check, run_checks, word_count,
)
from aarav_lm.data.sft import validate_conversation
from aarav_lm.evaluation.stats import intervals_overlap, rate_with_ci, wilson_interval

STORY = ('Once upon a time, there was a little girl named Lily. She had a red ball and a dog. One day at the park, '
         'Lily felt sad because the ball was lost. "Where is my ball?" she asked. The dog barked and found it. Lily was proud.')


# ── checks ───────────────────────────────────────────────────────────────────

def test_contains_is_whole_word_case_insensitive_and_allows_plurals():
    assert run_check(Check("contains_all", ("dog",)), "The Dogs ran.") and run_check(Check("contains_all", ("dog",)), "a DOG")
    assert not run_check(Check("contains_all", ("dog",)), "The dogma is old.")        # substring is not a match
    assert not run_check(Check("contains_all", ("cat", "dog")), "only a dog")          # all words required
    assert run_check(Check("contains_any", ("cat", "dog")), "only a dog")


def test_length_and_dialogue_checks():
    assert word_count("one two three's four.") == 4
    assert run_check(Check("max_words", n=5), "a b c d e") and not run_check(Check("max_words", n=4), "a b c d e")
    assert not run_check(Check("max_words", n=5), "")                                   # an empty reply never counts as "short"
    assert run_check(Check("min_words", n=3), "a b c") and not run_check(Check("min_words", n=4), "a b c")
    assert has_dialogue('He said "hi" today.') and has_dialogue("“yes”") and not has_dialogue('a single " quote')


def test_run_checks_reports_per_kind_and_requires_every_repeat():
    res = run_checks([Check("contains_all", ("dog",)), Check("contains_all", ("cat",)), Check("max_words", n=3)], "a dog")
    assert res == {"contains_all": False, "max_words": True}


def test_unknown_check_kind_is_an_error():
    with pytest.raises(ValueError):
        run_check(Check("nope"), "x")


# ── features and tasks ───────────────────────────────────────────────────────

def test_extract_features():
    f = extract_features(STORY)
    assert f.name == "Lily" and f.place == "park" and f.emotion in ("sad", "proud") and f.dialogue
    assert set(f.topics) >= {"dog", "ball"} and f.words == word_count(STORY)


def test_every_generated_task_is_satisfied_by_the_story_it_was_built_from():
    """Training labels are correct by construction. Verified over many stories, all task kinds."""
    from aarav_lm.data.instruct_tasks import NOUNS, PLACES
    rng = random.Random(0)
    stories = []
    for i in range(400):
        words = " ".join(rng.choice(NOUNS + PLACES + ["felt sad", "was brave", "very", "and", "the"]) for _ in range(rng.randint(20, 260)))
        stories.append(f"Once there was a boy named {rng.choice(['Tom', 'Zed', 'Kim'])}. {words}. " + ('"hi" he said. ' if i % 2 else ""))
    seen_kinds = set()
    for story in stories:
        f = extract_features(story)
        for kind in applicable_tasks(f):
            task = build_task(kind, f, rng, TRAIN_TEMPLATES)
            assert all(run_checks(task.checks, story).values()), (kind, task.request, task.checks)
            seen_kinds.add(kind)
    assert seen_kinds == set(TASK_WEIGHTS)                                              # the property was exercised for every kind


def test_pick_task_is_deterministic_and_prompts_have_no_unfilled_slots():
    f = extract_features(STORY)
    assert pick_task(f, random.Random(1)).to_dict() == pick_task(f, random.Random(1)).to_dict()
    for kind in applicable_tasks(f):
        for templates in (TRAIN_TEMPLATES, EVAL_TEMPLATES):
            assert "{" not in build_task(kind, f, random.Random(2), templates).request


def test_train_and_eval_templates_are_disjoint_and_cover_the_same_kinds():
    assert set(EVAL_TEMPLATES) == set(TRAIN_TEMPLATES) == set(TASK_WEIGHTS)
    train = {t for ts in TRAIN_TEMPLATES.values() for t in ts}
    assert not train & {t for ts in EVAL_TEMPLATES.values() for t in ts}                # eval measures generalisation, not recall


def test_length_thresholds_are_consistent():
    assert SHORT_MAX_WORDS < LONG_MIN_WORDS
    short, long_ = " ".join(["word"] * 50), " ".join(["word"] * 250)
    assert "short" in applicable_tasks(extract_features(short)) and "long" not in applicable_tasks(extract_features(short))
    assert "long" in applicable_tasks(extract_features(long_)) and "short" not in applicable_tasks(extract_features(long_))


# ── generated data ───────────────────────────────────────────────────────────

def _pool(n=300, seed=0):
    rng = random.Random(seed)
    return [f"Story {seed}-{i}: once upon a time there was a girl named {rng.choice(['Ann', 'Bo', 'Cy'])}. She saw a "
            f"{rng.choice(['dog', 'cat', 'bird'])} at the {rng.choice(['park', 'farm'])}. " + "It was fun and she felt brave. " * rng.randint(2, 30)
            for i in range(n)]                                                          # unique per (seed, i): pools never collide


def test_conversations_are_valid_deterministic_and_keep_held_out_stories_apart():
    stories = {"train": _pool(200, 1), "val": _pool(20, 2), "test": _pool(20, 3)}
    recs = list(make_conversations(stories, n_train=100, seed=5))
    assert recs == list(make_conversations(stories, n_train=100, seed=5))
    assert sum(r["split"] == "train" for r in recs) == 100
    for r in recs:
        msgs, why = validate_conversation(r)
        assert why is None and r["task"] in TASK_WEIGHTS
        answers = {m["content"] for m in r["messages"] if m["role"] == "assistant"}
        assert answers <= set(stories[r["split"]])                                      # answers come from that split's own stories
    train_answers = {m["content"] for r in recs if r["split"] == "train" for m in r["messages"] if m["role"] == "assistant"}
    assert not train_answers & {m["content"] for r in recs if r["split"] != "train" for m in r["messages"] if m["role"] == "assistant"}


def test_eval_prompts_are_balanced_use_held_out_templates_and_include_unseen_names():
    rows = make_eval_prompts(_pool(400, 4), per_kind=8, seed=1)
    kinds = [r["kind"] for r in rows]
    assert all(kinds.count(k) == 8 for k in set(kinds)) and len({r["id"] for r in rows}) == len(rows)
    train = {t for ts in TRAIN_TEMPLATES.values() for t in ts}
    assert all(r["prompt"] not in train for r in rows)
    unseen = [r for r in rows if r["unseen_name"]]
    assert unseen and all(r["kind"] in ("name", "name_topic", "name_emotion") for r in unseen)
    assert all(any(c["kind"] == "contains_all" and c["words"][0][0].isupper() for c in r["checks"]) for r in unseen)
    assert make_eval_prompts(_pool(400, 4), per_kind=8, seed=1) == rows                 # reproducible


# ── statistics ───────────────────────────────────────────────────────────────

def test_wilson_interval_known_values_and_properties():
    lo, hi = wilson_interval(15, 20)
    assert (round(lo, 3), round(hi, 3)) == (0.531, 0.888)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
    narrow, wide = wilson_interval(500, 1000), wilson_interval(5, 10)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]                                   # more data, tighter interval
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_rate_with_ci_and_overlap():
    r = rate_with_ci(3, 4)
    assert r["rate"] == 0.75 and r["n"] == 4 and r["ci95"][0] < 0.75 < r["ci95"][1]
    assert intervals_overlap((0.1, 0.4), (0.3, 0.6)) and not intervals_overlap((0.1, 0.2), (0.3, 0.6))


def test_training_task_mix_follows_the_weights_even_for_rare_kinds():
    """Regression: choosing the story first gave `short` ~0.2% of examples instead of its 5% share."""
    rng = random.Random(7)
    pool = []
    for i in range(6000):
        n = rng.choice([15, 25, 40, 55, 70, 90, 110])                            # ~3 words each: some short, some long
        pool.append(f"Story {i}: a girl named Ann saw a dog at the park and felt sad. " + ("\"hi\" she said. " if i % 3 == 0 else "") + "It was fun. " * n)
    recs = [r for r in make_conversations({"train": pool}, n_train=3000, seed=1) if r["split"] == "train"]
    assert len(recs) == 3000
    counts = {k: sum(r["task"] == k for r in recs) for k in TASK_WEIGHTS}
    total_w = sum(TASK_WEIGHTS.values())
    for kind in ("short", "long", "short_topic", "long_topic"):
        expected = 3000 * TASK_WEIGHTS[kind] / total_w
        assert 0.6 * expected <= counts[kind] <= 1.4 * expected, (kind, counts[kind], expected)
    answers = [m["content"] for r in recs for m in r["messages"] if m["role"] == "assistant"]
    first = [r["messages"][-1]["content"] if len(r["messages"]) <= 3 else r["messages"][-3]["content"] for r in recs]
    assert len(set(first)) == len(first)                                        # every story used at most once as a first answer
    for r in recs:                                                              # and each answer really satisfies its task kind
        story = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        if r["task"] in ("short", "short_topic"):
            assert word_count(story) <= SHORT_MAX_WORDS
        if r["task"] in ("long", "long_topic"):
            assert word_count(story) >= LONG_MIN_WORDS
