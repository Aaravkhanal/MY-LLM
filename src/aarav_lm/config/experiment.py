"""
aarav-lm · Experiment configuration

One YAML file fully describes a training run: model, data, tokenizer, optimizer,
schedule, batch sizes, seed and precision. The resolved config is copied into the
experiment folder so any run can be reproduced from its own record.

    exp = ExperimentConfig.from_yaml("configs/tiny_tinystories.yaml")
    exp = exp.with_overrides(["train.max_steps=200", "train.lr=5e-4"])
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, List, get_type_hints

import yaml

from aarav_lm.config.config import ModelConfig, get_preset


@dataclass
class TokenizerConfig:
    kind: str = "bpe"                 # "bpe" (byte-level BPE) | "byte" (raw bytes)
    vocab_size: int = 2048
    artifact_dir: str = "artifacts/tokenizer"   # versioned tokenizer output
    train_files: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    sources: List[str] = field(default_factory=list)   # .txt / .jsonl files or dirs
    format: str = "auto"              # auto | txt | jsonl
    text_field: str = "text"          # field to read from JSONL records
    doc_separator: str = "<|endoftext|>"   # splits .txt files into documents ("" = whole file)
    out_dir: str = "data/processed"   # where token shards + stats are written
    val_fraction: float = 0.01
    test_fraction: float = 0.01
    min_chars: int = 20
    max_chars: int = 100_000
    dedup: bool = True
    seed: int = 1337


@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "auto"              # auto | mps | cpu | cuda
    precision: str = "fp32"           # fp32 | bf16 (autocast; weights stay fp32)
    max_steps: int = 2000             # optimizer steps
    batch_size: int = 16              # sequences per micro-batch
    grad_accum: int = 4               # micro-batches per optimizer step
    lr: float = 1e-3
    min_lr_ratio: float = 0.1         # cosine floor = lr * min_lr_ratio
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 20
    log_interval: int = 20
    ckpt_interval: int = 200
    early_stop_patience: int = 0      # evals without improvement; 0 disables
    keep_last: int = 2                # rolling step checkpoints to keep
    lora_rank: int = 0                # >0: fine-tune with LoRA adapters of this rank (SFT only)
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0


@dataclass
class ExperimentConfig:
    name: str = "exp"
    description: str = ""
    stage: str = "pretrain"           # pretrain | sft
    init_from: str = ""               # sft: checkpoint (best.pt/final.pt/step_*.pt) to start from
    model_preset: str = "tiny"
    model_overrides: dict = field(default_factory=dict)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ── model ──────────────────────────────────────────────────────────────
    def build_model_config(self) -> ModelConfig:
        overrides = {"vocab_size": self.tokenizer.vocab_size, **self.model_overrides}
        return get_preset(self.model_preset, **overrides)

    @property
    def tokens_per_step(self) -> int:
        return self.train.batch_size * self.train.grad_accum * self.build_model_config().block_size

    # ── (de)serialisation ──────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = copy.deepcopy(d)
        sub = {"tokenizer": TokenizerConfig, "data": DataConfig, "train": TrainConfig}
        for key, klass in sub.items():
            if key in d:
                _check_keys(klass, d[key], key)
                d[key] = klass(**_coerce_section(klass, d[key], key))
        _check_keys(cls, d, "experiment")
        scalars = {k: v for k, v in d.items() if k not in sub}
        d.update(_coerce_section(cls, scalars, "experiment"))
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def fingerprint(self) -> str:
        """Stable short hash of the full resolved config."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    # ── CLI overrides: "train.lr=5e-4", "model_preset=small" ───────────────
    def with_overrides(self, overrides: List[str]) -> "ExperimentConfig":
        d = self.to_dict()
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must look like key=value, got {item!r}")
            key, raw = item.split("=", 1)
            value = yaml.safe_load(raw)
            node: Any = d
            parts = key.split(".")
            for p in parts[:-1]:
                if not isinstance(node, dict) or p not in node:
                    raise KeyError(f"unknown config path {key!r}")
                node = node[p]
            if not isinstance(node, dict) or (parts[-1] not in node and parts[0] != "model_overrides"):
                raise KeyError(f"unknown config key {key!r}")
            node[parts[-1]] = value
        return ExperimentConfig.from_dict(d)


def _check_keys(klass, d: dict, where: str) -> None:
    known = {f.name for f in fields(klass)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {sorted(unknown)}")


def _coerce_value(value: Any, typ: Any, where: str) -> Any:
    """Coerce/validate one scalar field.

    PyYAML follows YAML 1.1, where `5e-4` (no dot) loads as a *string*. A learning
    rate silently becoming '5e-4' is exactly the kind of hidden bug this prevents.
    """
    if typ is float:
        if isinstance(value, bool):
            raise TypeError(f"{where}: expected float, got bool")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise TypeError(f"{where}: expected float, got {value!r}")
    if typ is int:
        if isinstance(value, bool):
            raise TypeError(f"{where}: expected int, got bool")
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                raise TypeError(f"{where}: expected int, got {value!r}") from None
        if isinstance(value, (int, float)) and float(value) == int(value):
            return int(value)
        raise TypeError(f"{where}: expected int, got {value!r}")
    if typ is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"{where}: expected bool, got {value!r}")
    if typ is str:
        if isinstance(value, str):
            return value
        raise TypeError(f"{where}: expected str, got {value!r}")
    return value   # lists, dicts: shape is checked where they are consumed


def _coerce_section(klass, d: dict, where: str) -> dict:
    hints = get_type_hints(klass)
    return {k: _coerce_value(v, hints[k], f"{where}.{k}") for k, v in d.items()}
