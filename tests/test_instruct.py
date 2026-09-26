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


# ── clustered intervals and paired comparison ────────────────────────────────

def test_cluster_interval_is_wider_when_draws_are_duplicates_of_few_prompts():
    from aarav_lm.evaluation.stats import cluster_rate_ci
    independent = [(1, 1)] * 6 + [(0, 1)] * 6 + [(1, 1)] * 6 + [(0, 1)] * 6        # 24 different prompts, 50% pass
    duplicated = [(4, 4), (0, 4), (4, 4), (0, 4), (4, 4), (0, 4)]                   # 6 prompts x 4 identical draws, 50% pass
    wi, wd = cluster_rate_ci(independent), cluster_rate_ci(duplicated)
    assert wd[1] - wd[0] > 1.5 * (wi[1] - wi[0])
    assert cluster_rate_ci([]) == [0.0, 1.0]


def test_paired_difference_detects_a_real_gap_and_not_a_null_one():
    from aarav_lm.evaluation.stats import paired_difference
    real = [(1, 0, 1)] * 30 + [(1, 1, 1)] * 20 + [(0, 0, 1)] * 20                  # A wins 30 prompts, loses none
    r = paired_difference(real)
    assert r["diff"] == pytest.approx(30 / 70) and r["ci95"][0] > 0.2 and r["p"] < 0.01
    null = [(1, 0, 1)] * 15 + [(0, 1, 1)] * 15 + [(1, 1, 1)] * 40
    n = paired_difference(null)
    assert n["diff"] == 0 and n["ci95"][0] < 0 < n["ci95"][1] and n["p"] > 0.5
    with pytest.raises(ValueError):
        paired_difference([])


def test_paired_compare_matches_by_prompt_and_refuses_mismatched_runs():
    from aarav_lm.evaluation.paired import format_paired, paired_compare
    mk = lambda pid, kind, text, ok: {"id": pid, "kind": kind, "prompt": text, "passed": ok}           # noqa: E731
    a = [mk("1", "topic", "story with a dog", True), mk("2", "topic", "story with a cat", True), mk("3", "short", "brief", False)]
    b = [mk("1", "topic", "story with a dog", False), mk("2", "topic", "story with a cat", True), mk("3", "short", "brief", False)]
    r = paired_compare(a, b, n_boot=200)
    assert r["topic"]["only_a"] == 1 and r["topic"]["only_b"] == 0 and r["ALL"]["n"] == 3
    assert r["ALL"]["rate_a"] == pytest.approx(2 / 3) and r["ALL"]["rate_b"] == pytest.approx(1 / 3)
    assert "topic" in format_paired(r, "A", "B")
    with pytest.raises(ValueError, match="different prompts"):
        paired_compare(a, b[:2])
    with pytest.raises(ValueError, match="differs between"):
        paired_compare(a, [mk("1", "topic", "OTHER TEXT", False), b[1], b[2]])


def test_chance_levels_measure_what_a_request_ignoring_story_scores():
    from aarav_lm.data.instruct_data import chance_levels
    stories = ["The dog ran to the park.", "The cat sat.", "A dog and a cat played.", "The sun was hot."]
    prompts = [
        {"kind": "topic", "prompt": "a", "checks": [{"kind": "contains_all", "words": ["dog"], "n": 0}]},
        {"kind": "topic", "prompt": "b", "checks": [{"kind": "contains_all", "words": ["cat"], "n": 0}]},
        {"kind": "name", "prompt": "c", "checks": [{"kind": "contains_all", "words": ["Zara"], "n": 0}]},
        {"kind": "generic", "prompt": "d", "checks": []},                                # format-only: not scored
    ]
    c = chance_levels(prompts, stories)
    assert c["by_kind"] == {"name": 0.0, "topic": 0.5} and c["n_stories"] == 4          # dog in 2 of 4, cat in 2 of 4


def test_in_template_control_uses_training_phrasings_and_held_out_eval_uses_none_of_them():
    from aarav_lm.data.instruct_data import make_eval_prompts
    stories = [f"Once upon a time there was a little girl named Mia. She had a dog and played in the park. "
               f"Then she said, \"Hello {i}!\" " * 3 + "The end." for i in range(40)]
    train = {t for ts in TRAIN_TEMPLATES.values() for t in ts}
    held = make_eval_prompts(stories, per_kind=6, seed=0)
    ctrl = make_eval_prompts(stories, per_kind=6, seed=0, templates=TRAIN_TEMPLATES)
    assert {p["kind"] for p in held} == {p["kind"] for p in ctrl}
    assert not any(p["prompt"] in train for p in held if "{" not in p["prompt"])         # fixed-string prompts never come from training
    generic_train = set(TRAIN_TEMPLATES["generic"])
    assert all(p["prompt"] in generic_train for p in ctrl if p["kind"] == "generic")     # the control really does use training wording
    assert not any(p["prompt"] in generic_train for p in held if p["kind"] == "generic")


# ── v4 data options: templates, prominence, balanced sampling, casual typing ─

def _synthetic_pool(n, seed, skew=False):
    """Stories with known features. With skew=True most stories are set at the park (like the real corpus)."""
    import random
    from aarav_lm.data.instruct_tasks import EMOTIONS, NOUNS, PLACES
    r = random.Random(seed)
    out = []
    for _ in range(n):
        name = r.choice(["Tim", "Mia", "Zoe", "Sam", "Lily", "Ben"])
        a, b = r.sample(NOUNS, 2)
        place = "park" if (skew and r.random() < 0.7) else r.choice(PLACES)
        em = r.choice(EMOTIONS)
        body = [f"Once upon a time, there was a little kid named {name}. {name} loved a {a} at the {place}."]
        for _ in range(r.choice([2, 4, 8, 12, 16])):
            body.append(f"The {r.choice([a, b])} was {r.choice([em, 'happy', 'big'])}. \"Look!\" said {name}." if r.random() < .5
                        else f"{name} saw the {r.choice([a, b])} and smiled at the {place}.")
        out.append(" ".join(body))
    return out


def test_default_generator_output_is_byte_identical_to_the_v3_data():
    """Pinned hash: any change to the defaults would silently alter the data exp_006 was trained on."""
    import hashlib
    import json
    from aarav_lm.data.instruct_data import make_conversations
    stories = {"train": _synthetic_pool(1500, 1), "val": _synthetic_pool(60, 2), "test": _synthetic_pool(60, 3)}
    recs = list(make_conversations(stories, 800, seed=0))
    assert len(recs) == 920
    assert hashlib.sha256(json.dumps(recs, sort_keys=True).encode()).hexdigest() == \
        "41b4a529737d154ff21084306dd733f974344cd835b17319d342509975d19dbf"


def test_v4_templates_share_no_string_with_the_eval_and_add_many_frames():
    from aarav_lm.data.instruct_tasks import TRAIN_TEMPLATES_V4
    v4 = {t for ts in TRAIN_TEMPLATES_V4.values() for t in ts}
    assert not v4 & {t for ts in EVAL_TEMPLATES.values() for t in ts}
    assert set(TRAIN_TEMPLATES_V4) == set(TRAIN_TEMPLATES)
    assert all(len(TRAIN_TEMPLATES_V4[k]) >= 2 * len(TRAIN_TEMPLATES[k]) for k in TRAIN_TEMPLATES if k != "generic")
    assert not any(t.startswith(("Give me", "Compose", "Create", "Make up", "Could you", "I'm in the mood")) for t in v4)


def test_balanced_sampling_flattens_a_skewed_place_distribution():
    from collections import Counter
    from aarav_lm.data.instruct_data import make_conversations
    from aarav_lm.data.instruct_tasks import PLACES

    def park_share(alpha):
        stories = {"train": _synthetic_pool(4000, 5, skew=True)}
        c = Counter()
        for rec in make_conversations(stories, 3000, seed=1, balance_alpha=alpha):
            if rec["task"] == "place":
                u = [m for m in rec["messages"] if m["role"] == "user"][0]["content"]
                c.update(p for p in PLACES if p in u)
        return c["park"] / sum(c.values()), sum(c.values())
    natural, n1 = park_share(1.0)
    balanced, n2 = park_share(0.3)
    assert natural > 0.5 and balanced < 0.5 * natural and n1 > 100 and n2 > 100


def test_balanced_and_default_paths_are_both_deterministic_and_use_each_story_once():
    from aarav_lm.data.instruct_data import make_conversations
    stories = {"train": _synthetic_pool(600, 7, skew=True)}
    a = list(make_conversations(stories, 400, seed=3, balance_alpha=0.5, augment=0.3, min_mentions=2))
    b = list(make_conversations(stories, 400, seed=3, balance_alpha=0.5, augment=0.3, min_mentions=2))
    assert a == b and len(a) > 300
    answers = [[m for m in r["messages"] if m["role"] == "assistant"][0]["content"] for r in a]
    assert len(set(answers)) == len(answers)                       # never the same story twice


def test_min_mentions_only_keeps_requests_about_words_the_story_dwells_on():
    from aarav_lm.data.instruct_tasks import extract_features
    f = extract_features("Once upon a time a dog lived in the park. The dog ran. The dog barked at the park. The end.")
    assert f.topic_mentions == 3 and f.place_mentions == 2
    once = extract_features("Once upon a time, a boy saw a cat in a forest and went home to eat dinner with mom and dad today.")
    assert once.topic_mentions == 1 and once.place_mentions == 1


def test_casual_typing_keeps_the_names_the_checks_depend_on():
    import random
    from aarav_lm.data.instruct_tasks import Features, build_task, run_check
    f = Features(name="Zara", topics=["dog"], place="park", emotion="sad", words=50, dialogue=True)
    rng = random.Random(0)
    seen_lower, seen_nopunct = False, False
    for _ in range(200):
        t = build_task("name", f, rng, TRAIN_TEMPLATES, augment=1.0)
        assert "Zara" in t.request                                # the requested name keeps its capital
        assert run_check(t.checks[0], "Once upon a time Zara went home.")
        seen_lower |= t.request.replace("Zara", "").lower() == t.request.replace("Zara", "")
        seen_nopunct |= not t.request.endswith((".", "!", "?"))
    assert seen_lower and seen_nopunct
    plain = build_task("name", f, random.Random(0), TRAIN_TEMPLATES, augment=0.0)
    assert plain.request == build_task("name", f, random.Random(0), TRAIN_TEMPLATES).request


def test_first_mention_positions_and_the_early_mention_filter():
    from aarav_lm.data.instruct_data import make_conversations
    from aarav_lm.data.instruct_tasks import extract_features
    early = extract_features("A dog lived in the park. The dog was happy in the park. The end.")
    late = extract_features("Mia woke up and ate her breakfast slowly and then brushed her teeth. Later a dog came to the park. The dog ran in the park.")
    assert early.topic_first == 1 and early.place_first == 5 and late.topic_first >= 15
    stories = {"train": _synthetic_pool(800, 9)}
    all_topic = [r for r in make_conversations(stories, 500, seed=1) if r["task"] == "topic"]
    early_topic = [r for r in make_conversations(stories, 500, seed=1, first_mention_max=20, min_mentions=1) if r["task"] == "topic"]
    assert early_topic and all_topic
    strict = [r for r in make_conversations(stories, 500, seed=1, first_mention_max=5) if r["task"] == "topic"]
    assert strict == []                                              # nothing in the fixture opens with its topic in 5 words
    for r in early_topic:                                            # every answer really opens with what was asked for
        user = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
        story = [m for m in r["messages"] if m["role"] == "assistant"][0]["content"]
        f = extract_features(story)
        assert f.topic_first <= 20 and f.topic_first < 10**9


def test_final_eval_wordings_share_nothing_with_any_training_or_earlier_eval_wording():
    from aarav_lm.data.instruct_tasks import EVAL_TEMPLATES_FINAL, TRAIN_TEMPLATES_V4
    final = {t for ts in EVAL_TEMPLATES_FINAL.values() for t in ts}
    seen = {t for d in (TRAIN_TEMPLATES, TRAIN_TEMPLATES_V4, EVAL_TEMPLATES) for ts in d.values() for t in ts}
    assert not final & seen
    assert set(EVAL_TEMPLATES_FINAL) == set(TRAIN_TEMPLATES)
    openers = lambda d: {t.split()[0] for ts in d.values() for t in ts}                  # noqa: E731
    assert len(openers(EVAL_TEMPLATES_FINAL) - openers(TRAIN_TEMPLATES_V4) - openers(EVAL_TEMPLATES)) >= 5   # genuinely new openers
