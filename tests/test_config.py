import pytest

from aarav_lm.config import ExperimentConfig, ModelConfig, get_preset
from aarav_lm.model.core import AaravLM


def test_model_config_dict_round_trip():
    cfg = get_preset("tiny", vocab_size=1024)
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg


def test_model_config_rejects_unknown_fields():
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"n_layer": 2, "bogus": 1})


def test_unknown_preset_is_an_error():
    with pytest.raises(KeyError):
        get_preset("nope")


def test_medium_preset_is_the_documented_scale_up_from_small():
    """Locks the exact parameter count so a future config.py edit that accidentally changes
    `medium`'s shape (rather than deliberately re-deriving a new target) fails loudly."""
    small = get_preset("small")
    medium = get_preset("medium")
    assert medium.n_layer == small.n_layer, "medium keeps small's depth as a fixed point of comparison"
    assert medium.n_embd > small.n_embd
    n_small = AaravLM(small).num_params()
    n_medium = AaravLM(medium).num_params()
    assert n_small == 14_162_304
    assert n_medium == 25_698_816
    # +81.5%, in the neighbourhood of a "roughly double the model" scale-up.
    assert 1.7 < n_medium / n_small < 1.95


def test_experiment_yaml_round_trip(tmp_path):
    exp = ExperimentConfig(name="t", model_preset="tiny")
    path = tmp_path / "exp.yaml"
    exp.to_yaml(path)
    assert ExperimentConfig.from_yaml(path) == exp


def test_overrides_and_fingerprint():
    exp = ExperimentConfig()
    new = exp.with_overrides(["train.lr=5e-4", "train.max_steps=10", "model_preset=small"])
    assert new.train.lr == 5e-4 and new.train.max_steps == 10 and new.model_preset == "small"
    assert exp.train.max_steps != 10                      # original untouched
    assert exp.fingerprint() != new.fingerprint()
    assert exp.fingerprint() == ExperimentConfig().fingerprint()   # deterministic


def test_bad_override_paths_fail_loudly():
    exp = ExperimentConfig()
    with pytest.raises(KeyError):
        exp.with_overrides(["train.nonexistent=1"])
    with pytest.raises(ValueError):
        exp.with_overrides(["no_equals_sign"])


def test_model_config_takes_vocab_from_tokenizer_section():
    exp = ExperimentConfig().with_overrides(["tokenizer.vocab_size=1500"])
    assert exp.build_model_config().vocab_size == 1500


def test_yaml_scientific_notation_without_dot_is_a_float(tmp_path):
    """PyYAML loads `5e-4` as a string; config loading must still produce a float."""
    path = tmp_path / "exp.yaml"
    path.write_text("train:\n  lr: 5e-4\n  max_steps: 1e3\n")
    exp = ExperimentConfig.from_yaml(path)
    assert exp.train.lr == 5e-4 and isinstance(exp.train.lr, float)
    assert exp.train.max_steps == 1000 and isinstance(exp.train.max_steps, int)


def test_wrong_types_are_rejected(tmp_path):
    with pytest.raises(TypeError):
        ExperimentConfig.from_dict({"train": {"lr": "fast"}})
    with pytest.raises(TypeError):
        ExperimentConfig.from_dict({"train": {"max_steps": 2.5}})
    with pytest.raises(TypeError):
        ExperimentConfig.from_dict({"data": {"dedup": "yes"}})
