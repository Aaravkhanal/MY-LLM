"""
aarav-lm · Trainer

One experiment = one directory under experiments/:

    exp_001_<name>/
        config.yaml        resolved config (authoritative on resume)
        env.json           torch/python/device/git commit — what produced this run
        metrics.jsonl      one JSON object per log/eval event (append-only)
        train.log          human-readable log
        checkpoints/       step_XXXXXXXX.pt (rolling), best.pt, final.pt
        summary.json       written at the end (and on interruption)

Resuming restores weights, optimizer, LR-schedule position, data-stream position and
RNG state, so an interrupted run continues exactly where it stopped.
"""

from __future__ import annotations

import json
import math
import platform
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import yaml

from aarav_lm.config.experiment import ExperimentConfig
from aarav_lm.data.dataset import BatchStream, TokenDataset
from aarav_lm.data.pipeline import validate_dataset
from aarav_lm.data.shards import Shard
from aarav_lm.evaluation.loss import evaluate_loss
from aarav_lm.model.core import AaravLM
from aarav_lm.tokenizer import AaravTokenizer
from aarav_lm.train import checkpoint as ckpt
from aarav_lm.train.device import (
    autocast_context, empty_cache, pick_device, seed_everything, synchronize,
)
from aarav_lm.train.monitor import MB, ThroughputTracker, format_duration, memory_snapshot, tensors_nbytes
from aarav_lm.train.schedule import lr_at

LOW_MEMORY_MB = 700          # warn when the system has less than this available
MAX_CONSECUTIVE_BAD_STEPS = 5


def next_experiment_dir(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in root.iterdir():
        parts = p.name.split("_")
        if p.is_dir() and len(parts) >= 2 and parts[0] == "exp" and parts[1].isdigit():
            nums.append(int(parts[1]))
    d = root / f"exp_{(max(nums) + 1 if nums else 1):03d}_{name}"
    d.mkdir()
    return d


def _git_info(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        except Exception:                                          # noqa: BLE001
            return None
    dirty = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(dirty) if dirty is not None else None}


class Trainer:
    def __init__(
        self,
        exp: ExperimentConfig,
        run_dir: Path,
        log: Optional[Callable[[str], None]] = None,
        repo_root: Optional[Path] = None,
    ):
        self.exp = exp
        self.cfg = exp.train
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.repo_root = repo_root or Path.cwd()
        self._log_file = None
        self._user_log = log
        self._interrupted = False
        self.device = pick_device(self.cfg.device)
        self.ctx = lambda: autocast_context(self.device, self.cfg.precision)

        self.step = 0
        self.tokens_seen = 0
        self.best_val = math.inf
        self.evals_without_improvement = 0
        self.train_seconds = 0.0
        self.peak_rss_mb = 0.0
        self.peak_mps_mb = 0.0
        self.bad_steps = 0
        self.resumed_from: Optional[str] = None

    # ── logging ────────────────────────────────────────────────────────────
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        if self._user_log:
            self._user_log(line)
        else:
            print(line, flush=True)
        if self._log_file:
            self._log_file.write(line + "\n")
            self._log_file.flush()

    def _metric(self, kind: str, **fields: Any) -> None:
        rec = {"type": kind, "step": self.step, "time": round(time.time(), 3), **fields}
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ── setup ──────────────────────────────────────────────────────────────
    def _setup(self, resume: bool) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)
        self._log_file = open(self.run_dir / "train.log", "a", encoding="utf-8")

        if not (self.run_dir / "config.yaml").exists():
            self.exp.to_yaml(self.run_dir / "config.yaml")
        seed_everything(self.cfg.seed)

        self.tok = AaravTokenizer.load(self.exp.tokenizer.artifact_dir)
        problems = validate_dataset(self.exp.data.out_dir, self.tok)
        if problems:
            raise RuntimeError("dataset validation failed:\n  " + "\n  ".join(problems))

        self.model_cfg = self.exp.build_model_config()
        if self.model_cfg.vocab_size != self.tok.vocab_size:
            raise RuntimeError(
                f"model vocab {self.model_cfg.vocab_size} != tokenizer vocab {self.tok.vocab_size}"
            )
        seq_len = self.model_cfg.block_size
        out = Path(self.exp.data.out_dir)
        self.train_ds = TokenDataset(Shard.open(out / "train.bin"), seq_len)
        self.val_ds = TokenDataset(Shard.open(out / "val.bin"), seq_len)
        self.stream = BatchStream(self.train_ds, self.cfg.batch_size, seed=self.cfg.seed)
        self.val_batches = self.val_ds.fixed_eval_batches(self.cfg.batch_size, self.cfg.eval_iters)

        # Initialise on CPU so weights are identical across devices for a given seed.
        self.model = AaravLM(self.model_cfg)
        self.model.to(self.device).train()
        self.optimizer = self.model.configure_optimizers(
            self.cfg.weight_decay, self.cfg.lr, (self.cfg.beta1, self.cfg.beta2), self.device.type
        )
        self.tokens_per_step = self.cfg.batch_size * self.cfg.grad_accum * seq_len

        if resume:
            latest = ckpt.latest_checkpoint(self.ckpt_dir)
            if latest is None:
                raise ckpt.CheckpointError(f"nothing to resume in {self.ckpt_dir}")
            self._restore(latest)

        info = {
            "python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
            "device": str(self.device), "precision": self.cfg.precision,
            "git": _git_info(self.repo_root), "config_fingerprint": self.exp.fingerprint(),
            "tokenizer_fingerprint": self.tok.fingerprint,
        }
        (self.run_dir / "env.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

        n = self.model.num_params()
        self.log(
            f"[setup] {self.exp.name}  device={self.device}  precision={self.cfg.precision}  "
            f"params={n/1e6:.2f}M (body {self.model.num_params(True)/1e6:.2f}M)  "
            f"weights={tensors_nbytes(self.model)/MB:.1f}MB"
        )
        self.log(
            f"[setup] train {self.train_ds.shard.n_tokens:,} tok / val {self.val_ds.shard.n_tokens:,} tok  "
            f"seq={seq_len}  batch={self.cfg.batch_size}x{self.cfg.grad_accum}  "
            f"tokens/step={self.tokens_per_step:,}  steps={self.cfg.max_steps}  "
            f"total={self.tokens_per_step * self.cfg.max_steps/1e6:.1f}M tok "
            f"({self.tokens_per_step * self.cfg.max_steps / self.train_ds.shard.n_tokens:.2f} epochs)"
        )

    def _restore(self, path: Path) -> None:
        state = ckpt.load_training_checkpoint(path)
        self.tok.check_fingerprint(state["tokenizer_fingerprint"], what=f"checkpoint {path.name}")
        if state["model_config"] != self.model_cfg.to_dict():
            raise ckpt.CheckpointError("checkpoint model config differs from the experiment config")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.stream.load_state_dict(state["stream_state"])
        ts = state["train_state"]
        self.step, self.tokens_seen = ts["step"], ts["tokens_seen"]
        self.best_val = ts["best_val"] if ts["best_val"] is not None else math.inf
        self.evals_without_improvement = ts["evals_without_improvement"]
        self.train_seconds = ts["train_seconds"]
        self.peak_rss_mb, self.peak_mps_mb = ts.get("peak_rss_mb", 0.0), ts.get("peak_mps_mb", 0.0)
        ckpt.restore_rng_state(state["rng"])
        self.resumed_from = path.name
        self.log(f"[resume] {path.name}: step {self.step}, {self.tokens_seen:,} tokens, best val {self.best_val:.4f}")

    # ── checkpoints ────────────────────────────────────────────────────────
    def _train_state(self) -> Dict[str, Any]:
        return {
            "step": self.step, "tokens_seen": self.tokens_seen,
            "best_val": None if math.isinf(self.best_val) else self.best_val,
            "evals_without_improvement": self.evals_without_improvement,
            "train_seconds": self.train_seconds,
            "peak_rss_mb": self.peak_rss_mb, "peak_mps_mb": self.peak_mps_mb,
        }

    def save_checkpoint(self) -> Path:
        path = ckpt.step_path(self.ckpt_dir, self.step)
        ckpt.save_training_checkpoint(
            path, self.model, self.optimizer, self._train_state(), self.stream.state_dict(),
            self.tok.fingerprint, self.exp.to_dict(),
        )
        ckpt.prune_step_checkpoints(self.ckpt_dir, self.cfg.keep_last)
        return path

    def _save_best(self, val: Dict[str, float]) -> None:
        ckpt.save_model_checkpoint(
            self.ckpt_dir / "best.pt", self.model, self.tok.fingerprint,
            {"step": self.step, "val_loss": val["loss"], "val_perplexity": val["perplexity"]},
        )

    # ── evaluation ─────────────────────────────────────────────────────────
    def evaluate(self) -> Dict[str, float]:
        res = evaluate_loss(self.model, self.val_batches, self.device, self.ctx)
        empty_cache(self.device)
        return res

    # ── main loop ──────────────────────────────────────────────────────────
    def _handle_signal(self, signum, frame) -> None:                # noqa: ARG002
        if self._interrupted:
            raise KeyboardInterrupt      # second Ctrl-C: stop immediately
        self._interrupted = True
        self.log("[signal] stop requested — finishing this step, then saving a checkpoint (Ctrl-C again to abort)")

    def run(self, resume: bool = False) -> Dict[str, Any]:
        self._setup(resume)
        old = {s: signal.signal(s, self._handle_signal) for s in (signal.SIGINT, signal.SIGTERM)}
        status = "completed"
        tracker = ThroughputTracker()
        window_loss, window_n, grad_norm_max, clipped = 0.0, 0, 0.0, 0
        t_run = time.perf_counter()
        train_seconds_at_start = self.train_seconds
        try:
            if self.step == 0:
                v = self.evaluate()
                self.log(f"[eval] step 0  val loss {v['loss']:.4f}  ppl {v['perplexity']:.1f}  (untrained baseline)")
                self._metric("eval", val_loss=v["loss"], val_ppl=v["perplexity"])
            tracker.start()
            while self.step < self.cfg.max_steps:
                lr = lr_at(self.step, self.cfg.lr, self.cfg.min_lr_ratio, self.cfg.warmup_steps, self.cfg.max_steps)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr

                step_loss = torch.zeros((), device=self.device)
                for _ in range(self.cfg.grad_accum):
                    x, y = self.stream.next_batch()
                    x, y = x.to(self.device), y.to(self.device)
                    with self.ctx():
                        _, loss = self.model(x, targets=y)
                    (loss / self.cfg.grad_accum).backward()
                    step_loss += loss.detach() / self.cfg.grad_accum
                loss_val = float(step_loss)                          # single device sync per step
                if not math.isfinite(loss_val):
                    raise FloatingPointError(
                        f"non-finite loss ({loss_val}) at step {self.step}; last good checkpoint is untouched"
                    )

                gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip))
                if not math.isfinite(gnorm):
                    self.bad_steps += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    self.log(f"[warn] non-finite gradient norm at step {self.step}; skipped update ({self.bad_steps} in a row)")
                    if self.bad_steps >= MAX_CONSECUTIVE_BAD_STEPS:
                        raise FloatingPointError("gradients stayed non-finite for 5 consecutive steps")
                    continue
                self.bad_steps = 0
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.step += 1
                self.tokens_seen += self.tokens_per_step
                synchronize(self.device)
                tracker.tick(self.tokens_per_step)
                window_loss += loss_val
                window_n += 1
                grad_norm_max = max(grad_norm_max, gnorm)
                clipped += gnorm > self.cfg.grad_clip
                self.train_seconds = train_seconds_at_start + (time.perf_counter() - t_run)

                if self.step % self.cfg.log_interval == 0 or self.step == self.cfg.max_steps:
                    mem = memory_snapshot(self.device)
                    self.peak_rss_mb = max(self.peak_rss_mb, mem["rss_mb"])
                    self.peak_mps_mb = max(self.peak_mps_mb, mem.get("mps_driver_mb", 0.0))
                    eta = tracker.eta_seconds(self.cfg.max_steps - self.step)
                    avg = window_loss / window_n
                    self.log(
                        f"step {self.step:>6}/{self.cfg.max_steps}  loss {avg:.4f}  lr {lr:.2e}  "
                        f"gnorm max {grad_norm_max:.2f} (clipped {clipped}/{window_n})  "
                        f"{tracker.tokens_per_sec:,.0f} tok/s  {tracker.step_seconds*1000:.0f} ms/step  "
                        f"rss {mem['rss_mb']:.0f}MB  mps {mem.get('mps_driver_mb', 0):.0f}MB  "
                        f"free {mem['sys_available_mb']:.0f}MB  eta {format_duration(eta)}"
                    )
                    self._metric(
                        "train", loss=avg, lr=lr, grad_norm_max=grad_norm_max, clipped_steps=clipped,
                        tokens_per_sec=tracker.tokens_per_sec, step_seconds=tracker.step_seconds,
                        tokens_seen=self.tokens_seen, epochs=self.stream.epochs_completed, **mem,
                    )
                    if mem["sys_available_mb"] < LOW_MEMORY_MB:
                        self.log(f"[warn] only {mem['sys_available_mb']:.0f}MB of system memory free — close other apps or lower batch_size")
                    window_loss, window_n, grad_norm_max, clipped = 0.0, 0, 0.0, 0

                stop_early = False
                if self.step % self.cfg.eval_interval == 0 or self.step == self.cfg.max_steps:
                    tracker.pause()
                    v = self.evaluate()
                    improved = v["loss"] < self.best_val
                    if improved:
                        self.best_val = v["loss"]
                        self.evals_without_improvement = 0
                        self._save_best(v)
                    else:
                        self.evals_without_improvement += 1
                    self.log(
                        f"[eval] step {self.step}  val loss {v['loss']:.4f}  ppl {v['perplexity']:.2f}  "
                        f"best {self.best_val:.4f}{'  ★ new best' if improved else ''}"
                    )
                    self._metric("eval", val_loss=v["loss"], val_ppl=v["perplexity"], best_val=self.best_val, improved=improved)
                    patience = self.cfg.early_stop_patience
                    if patience and self.evals_without_improvement >= patience:
                        self.log(f"[early-stop] no improvement in {patience} evals")
                        stop_early, status = True, "early_stopped"
                    tracker.resume()

                if self.step % self.cfg.ckpt_interval == 0 and self.step < self.cfg.max_steps:
                    tracker.pause()
                    self.save_checkpoint()
                    tracker.resume()

                if self._interrupted:
                    status = "interrupted"
                    break
                if stop_early:
                    break
        except BaseException as e:
            status = f"failed: {type(e).__name__}: {e}"
            self.log(f"[error] {status}")
            self._write_summary(status)
            raise
        finally:
            for s, h in old.items():
                signal.signal(s, h)

        path = self.save_checkpoint()
        ckpt.save_model_checkpoint(
            self.ckpt_dir / "final.pt", self.model, self.tok.fingerprint,
            {"step": self.step, "status": status},
        )
        self.log(f"[done] {status} at step {self.step}; saved {path.name}, final.pt")
        summary = self._write_summary(status)
        self._log_file.close()
        return summary

    # ── summary ────────────────────────────────────────────────────────────
    def _write_summary(self, status: str) -> Dict[str, Any]:
        sizes = {p.name: round(p.stat().st_size / MB, 2) for p in sorted(self.ckpt_dir.glob("*.pt"))}
        last_val = None
        mfile = self.run_dir / "metrics.jsonl"
        if mfile.exists():
            evals = [json.loads(line) for line in mfile.read_text().splitlines() if '"type": "eval"' in line]
            last_val = evals[-1] if evals else None
        summary = {
            "name": self.exp.name, "status": status, "steps": self.step, "tokens_seen": self.tokens_seen,
            "train_seconds": round(self.train_seconds, 1), "resumed_from": self.resumed_from,
            "params": self.model.num_params(), "params_body": self.model.num_params(True),
            "model_mb_fp32": round(tensors_nbytes(self.model) / MB, 2),
            "best_val_loss": None if math.isinf(self.best_val) else self.best_val,
            "best_val_perplexity": None if math.isinf(self.best_val) else math.exp(self.best_val),
            "last_eval": last_val, "checkpoint_mb": sizes,
            "dataset_train_tokens": self.train_ds.shard.n_tokens,
            "peak_rss_mb": self.peak_rss_mb, "peak_mps_driver_mb": self.peak_mps_mb,
            "config_fingerprint": self.exp.fingerprint(), "tokenizer_fingerprint": self.tok.fingerprint,
            "device": str(self.device), "precision": self.cfg.precision,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
