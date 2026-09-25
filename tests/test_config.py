import pytest

from aarav_lm.config import ExperimentConfig, ModelConfig, get_preset


def test_model_config_dict_round_trip():
    cfg = get_preset("tiny", vocab_size=1024)
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg


def test_model_config_rejects_unknown_fields():
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"n_layer": 2, "bogus": 1})


def test_unknown_preset_is_an_error():
    with pytest.raises(KeyError):
        get_preset("nope")


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
