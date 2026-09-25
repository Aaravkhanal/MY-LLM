"""
aarav-lm · Supervised fine-tuning data (conversations)

Input: JSONL, one conversation per line: {"messages": [{"role": ..., "content": ...}, ...]}
Valid conversations: optional leading system message, then strictly alternating
user / assistant turns, starting with user and ending with assistant.

Encoding uses the chat template (role token, content, <|end|>). The loss mask is 1 only
for assistant content and the <|end|> that closes it; the model is never trained to
predict the user's text, the system prompt, or the role tokens (it is *given* the
<|assistant|> token at inference, so it never has to produce it).

Each split is stored as an .npz with three flat arrays (no pickles):
    ids      token ids, all conversations concatenated
    mask     1 where the token is a training target
    offsets  int64, conversation i occupies ids[offsets[i]:offsets[i+1]]
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from aarav_lm.config.experiment import ExperimentConfig
from aarav_lm.data.clean import normalize_text
from aarav_lm.data.dedup import ExactDeduper
from aarav_lm.data.ingest import expand_sources
from aarav_lm.data.shards import token_dtype
from aarav_lm.data.split import split_of
from aarav_lm.inference.chat_format import MAX_MESSAGE_CHARS, ROLE_TOKEN, Message
from aarav_lm.paths import resolve
from aarav_lm.tokenizer import ASSISTANT, END_TURN, AaravTokenizer

SPLITS = ("train", "val", "test")


def validate_conversation(raw) -> Tuple[Optional[List[Message]], Optional[str]]:
    """Return (messages, None) if valid, else (None, reason)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list) or not raw["messages"]:
        return None, "no_messages"
    msgs: List[Message] = []
    for m in raw["messages"]:
        if not isinstance(m, dict) or m.get("role") not in ROLE_TOKEN or not isinstance(m.get("content"), str):
            return None, "bad_message"
        content = normalize_text(m["content"])
        if not content:
            return None, "empty_message"
        if len(content) > MAX_MESSAGE_CHARS:
            return None, "message_too_long"
        msgs.append(Message(m["role"], content))
    body = msgs[1:] if msgs[0].role == "system" else msgs
    if not body or body[0].role != "user" or body[-1].role != "assistant":
        return None, "must_start_with_user_end_with_assistant"
    if any(m.role != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(body)):
        return None, "roles_not_alternating"
    if raw.get("split") is not None and raw["split"] not in SPLITS:
        return None, "bad_split"
    return msgs, None


def encode_conversation(
    tok: AaravTokenizer, messages: Sequence[Message], max_len: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(ids, mask), or None if the conversation is longer than max_len tokens."""
    ids: List[int] = []
    mask: List[int] = []
    end = tok.special_ids[END_TURN]
    for m in messages:
        train = 1 if m.role == "assistant" else 0
        body = tok.encode(m.content)               # content is never allowed to become a control token
        ids += [tok.special_ids[ROLE_TOKEN[m.role]], *body, end]
        mask += [0, *([train] * len(body)), train]
    if len(ids) > max_len:
        return None
    return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=np.uint8)


def _iter_conversations(paths: Sequence[Path], rejects: Counter):
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    rejects["bad_json"] += 1


def build_sft_dataset(exp: ExperimentConfig, log: Callable[[str], None] = print) -> dict:
    """Tokenise conversations with the (existing) tokenizer and write train/val/test .npz + manifest."""
    t0 = time.time()
    tok = AaravTokenizer.load(resolve(exp.tokenizer.artifact_dir))
    out_dir = resolve(exp.data.out_dir)
    max_len = exp.build_model_config().block_size + 1
    rejects: Counter = Counter()
    deduper = ExactDeduper()
    per_split: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {s: [] for s in SPLITS}
    n_read = n_dupes = 0

    for raw in _iter_conversations(expand_sources(exp.data.sources), rejects):
        n_read += 1
        msgs, reason = validate_conversation(raw)
        if reason:
            rejects[reason] += 1
            continue
        key_text = "\n".join(f"{m.role}: {m.content}" for m in msgs)
        if exp.data.dedup and deduper.is_duplicate(key_text):
            n_dupes += 1
            continue
        enc = encode_conversation(tok, msgs, max_len)
        if enc is None:
            rejects["too_long_for_context"] += 1
            continue
        # an explicit "split" in the record wins (used to keep held-out answers from unseen source text)
        split = raw.get("split") or split_of(key_text, exp.data.seed, exp.data.val_fraction, exp.data.test_fraction)
        per_split[split].append(enc)

    if not per_split["train"]:
        raise RuntimeError("no training conversations survived validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = token_dtype(tok.vocab_size)
    stats: Dict[str, dict] = {}
    for split, items in per_split.items():
        lens = np.asarray([len(i) for i, _ in items], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        ids = np.concatenate([i for i, _ in items]).astype(dtype) if items else np.zeros(0, dtype)
        mask = np.concatenate([m for _, m in items]).astype(np.uint8) if items else np.zeros(0, np.uint8)
        np.savez(out_dir / f"{split}.npz", ids=ids, mask=mask, offsets=offsets)
        stats[split] = {
            "conversations": len(items), "tokens": int(lens.sum()), "trained_tokens": int(mask.sum()),
            "mean_len": round(float(lens.mean()), 1) if len(lens) else None,
            "max_len": int(lens.max()) if len(lens) else None,
        }
        log(f"[sft-data] {split:5s} {len(items):>7,} conversations {int(lens.sum()):>11,} tokens "
            f"({int(mask.sum()):,} trained)")
    manifest = {
        "format_version": 1, "kind": "sft", "experiment": exp.name, "max_len": max_len,
        "tokenizer": {"fingerprint": tok.fingerprint, "vocab_size": tok.vocab_size},
        "counters": {"read": n_read, "duplicates_removed": n_dupes, "rejects": dict(rejects)},
        "splits": stats, "build_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log(f"[sft-data] rejected {dict(rejects) or 0}, duplicates {n_dupes}; wrote {out_dir}")
    return manifest


class SFTDataset:
    def __init__(self, ids: np.ndarray, mask: np.ndarray, offsets: np.ndarray, pad_id: int):
        if len(ids) != len(mask) or offsets[0] != 0 or offsets[-1] != len(ids) or np.any(np.diff(offsets) < 2):
            raise ValueError("inconsistent SFT arrays")
        self.ids, self.mask, self.offsets, self.pad_id = ids, mask, offsets, pad_id

    @classmethod
    def load(cls, path: str | Path, pad_id: int) -> "SFTDataset":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["ids"], z["mask"], z["offsets"], pad_id)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_tokens(self) -> int:
        return int(len(self.ids))

    def batch(self, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Padded (x, y). y is -1 wherever the next token is not a training target."""
        rows = []
        for i in indices:
            a, b = int(self.offsets[i]), int(self.offsets[i + 1])
            ids = self.ids[a:b].astype(np.int64)
            m = self.mask[a:b].astype(bool)
            rows.append((ids[:-1], np.where(m[1:], ids[1:], -1)))
        width = max(len(x) for x, _ in rows)
        X = np.full((len(rows), width), self.pad_id, dtype=np.int64)
        Y = np.full((len(rows), width), -1, dtype=np.int64)
        for r, (x, y) in enumerate(rows):
            X[r, : len(x)], Y[r, : len(y)] = x, y
        return torch.from_numpy(X), torch.from_numpy(Y)

    def fixed_eval_batches(self, batch_size: int, n_batches: int):
        n = min(batch_size * n_batches, len(self))
        idx = np.linspace(0, len(self) - 1, num=n).astype(np.int64)
        return [self.batch(idx[i : i + batch_size]) for i in range(0, n, batch_size)]


class SFTBatchStream:
    """Shuffled conversation batches; state is (seed, epoch, pos), so resume is exact."""

    def __init__(self, dataset: SFTDataset, batch_size: int, seed: int):
        self.ds, self.batch_size, self.seed = dataset, batch_size, seed
        self.epoch = self.pos = 0
        self._order = self._make_order(0)

    def _make_order(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.seed, epoch]).permutation(len(self.ds))

    def next_batch(self):
        picked: List[int] = []
        while len(picked) < self.batch_size:
            take = min(self.batch_size - len(picked), len(self._order) - self.pos)
            picked.extend(self._order[self.pos : self.pos + take].tolist())
            self.pos += take
            if self.pos >= len(self._order):
                self.epoch, self.pos = self.epoch + 1, 0
                self._order = self._make_order(self.epoch)
        return self.ds.batch(picked)

    @property
    def epochs_completed(self) -> float:
        return self.epoch + self.pos / len(self._order)

    def state_dict(self) -> Dict[str, int]:
        return {"seed": self.seed, "epoch": self.epoch, "pos": self.pos}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        if state["seed"] != self.seed:
            raise ValueError(f"stream seed mismatch: checkpoint {state['seed']} vs {self.seed}")
        self.epoch, self.pos = int(state["epoch"]), int(state["pos"])
        self._order = self._make_order(self.epoch)
