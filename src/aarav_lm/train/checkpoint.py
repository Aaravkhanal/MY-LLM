"""
aarav-lm · Checkpoints

Two kinds of file, both plain tensors + JSON-like values (loadable with
torch.load(weights_only=True), so opening a checkpoint can never execute code):

  step_XXXXXXXX.pt   full training state: weights, optimizer, data-stream position,
                     RNG states, counters. Enough to resume with identical results.
  best.pt / final.pt model-only: weights + model config + tokenizer fingerprint.
                     What evaluation, chat and the API load.

Writes are atomic (temp file in the same directory, fsync, os.replace), so a crash
or power loss mid-save leaves the previous checkpoint intact rather than a
half-written file.
"""

from __future__ import annotations

import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from aarav_lm.config.config import ModelConfig
from aarav_lm.model.core import AaravLM

FORMAT_VERSION = 1
_STEP_RE = re.compile(r"^step_(\d{8})\.pt$")


class CheckpointError(RuntimeError):
    pass


def atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ── RNG state ────────────────────────────────────────────────────────────────

def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"torch_cpu": torch.get_rng_state(), "python": _py_random_to_plain(random.getstate())}
    if torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(_py_random_from_plain(state["python"]))
    if "torch_mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["torch_mps"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _py_random_to_plain(s):
    version, internal, gauss = s
    return {"version": version, "internal": list(internal), "gauss": gauss}


def _py_random_from_plain(d):
    return (d["version"], tuple(d["internal"]), d["gauss"])


# ── save ─────────────────────────────────────────────────────────────────────

def save_training_checkpoint(
    path: Path,
    model: AaravLM,
    optimizer: torch.optim.Optimizer,
    train_state: Dict[str, Any],
    stream_state: Dict[str, int],
    tokenizer_fingerprint: str,
    config_dict: Dict[str, Any],
) -> None:
    atomic_save(
        {
            "format_version": FORMAT_VERSION,
            "kind": "training",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model.cfg.to_dict(),
            "train_state": train_state,
            "stream_state": stream_state,
            "rng": capture_rng_state(),
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "config": config_dict,
        },
        path,
    )


def save_model_checkpoint(
    path: Path,
    model: AaravLM,
    tokenizer_fingerprint: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    atomic_save(
        {
            "format_version": FORMAT_VERSION,
            "kind": "model",
            "model": model.state_dict(),
            "model_config": model.cfg.to_dict(),
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "meta": meta or {},
        },
        path,
    )


# ── load ─────────────────────────────────────────────────────────────────────

def _load_raw(path: Path, map_location="cpu") -> Dict[str, Any]:
    if not Path(path).exists():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:                                        # noqa: BLE001
        raise CheckpointError(f"could not load {path}: {e}") from e
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(f"{path}: unsupported checkpoint format {ckpt.get('format_version')!r}")
    return ckpt


def load_training_checkpoint(path: Path, map_location="cpu") -> Dict[str, Any]:
    ckpt = _load_raw(path, map_location)
    if ckpt["kind"] != "training":
        raise CheckpointError(f"{path} is a {ckpt['kind']!r} checkpoint, not a resumable training one")
    return ckpt


def load_model(
    path: Path,
    device: torch.device | str = "cpu",
    expected_tokenizer_fingerprint: Optional[str] = None,
) -> Tuple[AaravLM, Dict[str, Any]]:
    """Load weights from a model *or* training checkpoint. Returns (model in eval mode, raw dict)."""
    ckpt = _load_raw(path)
    if expected_tokenizer_fingerprint and ckpt["tokenizer_fingerprint"] != expected_tokenizer_fingerprint:
        raise CheckpointError(
            f"{path} was trained with tokenizer {ckpt['tokenizer_fingerprint'][:12]}…, "
            f"expected {expected_tokenizer_fingerprint[:12]}…"
        )
    model = AaravLM(ModelConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


# ── discovery / retention ────────────────────────────────────────────────────

def list_step_checkpoints(directory: Path) -> List[Tuple[int, Path]]:
    if not directory.exists():
        return []
    found = []
    for p in directory.iterdir():
        m = _STEP_RE.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def latest_checkpoint(directory: Path) -> Optional[Path]:
    found = list_step_checkpoints(directory)
    return found[-1][1] if found else None


def prune_step_checkpoints(directory: Path, keep_last: int) -> None:
    for _, p in list_step_checkpoints(directory)[:-keep_last] if keep_last > 0 else []:
        p.unlink()


def step_path(directory: Path, step: int) -> Path:
    return directory / f"step_{step:08d}.pt"
