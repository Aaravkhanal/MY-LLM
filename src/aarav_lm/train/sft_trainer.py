"""
aarav-lm · Supervised fine-tuning trainer

Same loop, checkpoints, resume and monitoring as pretraining (it is a Trainer); only
the data, the starting weights and the export differ:

  data      conversations with an assistant-only loss mask (data/sft.py)
  weights   start from a pretrained checkpoint (`init_from`); tokenizer must match
  LoRA      optional (`train.lora_rank > 0`): only adapters train; exported
            checkpoints have the adapters merged, so inference loads a plain model

Checkpoints written here carry meta["stage"] == "sft", which is what switches chat
and the API from raw text continuation to the instruction template.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from aarav_lm.config.experiment import ExperimentConfig
from aarav_lm.data.sft import SFTBatchStream, SFTDataset
from aarav_lm.finetune.lora import apply_lora, count_parameters, has_lora, merge_lora
from aarav_lm.model.core import AaravLM
from aarav_lm.paths import resolve
from aarav_lm.train import checkpoint as ckpt
from aarav_lm.train.trainer import Trainer


class SFTTrainer(Trainer):
    def _validate_data(self) -> None:
        out = resolve(self.exp.data.out_dir)
        manifest_path = out / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"{manifest_path} missing — run scripts/prepare_sft.py first")
        m = json.loads(manifest_path.read_text())
        if m.get("kind") != "sft":
            raise RuntimeError(f"{out} is not an SFT dataset (kind={m.get('kind')!r})")
        self.tok.check_fingerprint(m["tokenizer"]["fingerprint"], what="SFT dataset")
        if m["max_len"] != self.model_cfg.block_size + 1:
            raise RuntimeError("SFT dataset was built for a different context length")

    def _build_data(self) -> None:
        out = resolve(self.exp.data.out_dir)
        train = SFTDataset.load(out / "train.npz", self.tok.pad_id)
        val = SFTDataset.load(out / "val.npz", self.tok.pad_id)
        if len(val) == 0:
            raise RuntimeError("SFT validation split is empty; raise data.val_fraction")
        self.n_train_examples = len(train)
        self.stream = SFTBatchStream(train, self.cfg.batch_size, seed=self.cfg.seed)
        self.val_batches = val.fixed_eval_batches(self.cfg.batch_size, self.cfg.eval_iters)
        self.train_tokens, self.val_tokens = train.n_tokens, val.n_tokens
        self.data_summary = f"train {len(train):,} conversations ({train.n_tokens:,} tok) / val {len(val):,}"

    def _init_model(self) -> AaravLM:
        if not self.exp.init_from:
            raise RuntimeError("stage 'sft' needs init_from: the pretrained checkpoint to fine-tune")
        path = resolve(self.exp.init_from)
        model, raw = ckpt.load_model(path, "cpu", self.tok.fingerprint)
        if raw["model_config"] != self.model_cfg.to_dict():
            raise RuntimeError(f"{path.name} has a different architecture than this experiment's model config")
        self.init_from_name = f"{path.parent.parent.name}/{path.name}"
        if self.cfg.lora_rank > 0:
            names = apply_lora(model, self.cfg.lora_rank, self.cfg.lora_alpha, self.cfg.lora_dropout)
            c = count_parameters(model)
            self.log(f"[lora] rank {self.cfg.lora_rank}: {len(names)} layers wrapped, "
                     f"{c['trainable']:,} trainable of {c['total']:,} parameters")
        self.log(f"[sft] starting from {self.init_from_name}")
        return model

    def _export_model(self) -> AaravLM:
        if not has_lora(self.model):
            return self.model
        merged = copy.deepcopy(self.model)
        return merge_lora(merged)

    def epoch_estimate(self) -> float:
        return self.cfg.batch_size * self.cfg.grad_accum * self.cfg.max_steps / max(1, self.n_train_examples)


def make_trainer(exp: ExperimentConfig, run_dir: Path, **kwargs) -> Trainer:
    if exp.stage == "pretrain":
        return Trainer(exp, run_dir, **kwargs)
    if exp.stage == "sft":
        return SFTTrainer(exp, run_dir, **kwargs)
    raise ValueError(f"unknown stage {exp.stage!r} (use 'pretrain' or 'sft')")
