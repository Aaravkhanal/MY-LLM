import pytest
import torch


def pytest_configure(config):
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """A real (tiny) trained checkpoint on CPU, shared by the inference/chat/eval tests."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import tiny_experiment
    from aarav_lm.train.trainer import Trainer

    root = tmp_path_factory.mktemp("tiny_run")
    exp = tiny_experiment(root, max_steps=30, eval_interval=10)
    run = root / "run"
    Trainer(exp, run, log=lambda _l: None).run()
    return {"exp": exp, "run": run, "best": run / "checkpoints" / "best.pt",
            "final": run / "checkpoints" / "final.pt", "tok_dir": exp.tokenizer.artifact_dir}


@pytest.fixture()
def engine(tiny_run):
    from aarav_lm.inference.engine import InferenceEngine

    return InferenceEngine.from_checkpoint(tiny_run["final"], tiny_run["tok_dir"], device="cpu")
