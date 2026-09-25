"""
aarav-lm · Packed, resumable batching

Documents are already concatenated with EOT separators inside the shard, so a
training example is just a window of seq_len+1 consecutive tokens (inputs are the
first seq_len, targets the same window shifted by one). There is no padding.

BatchStream visits every window once per epoch in a shuffled order. The window grid
is shifted by a different offset each epoch, so small datasets don't see identical
example boundaries every pass. The order depends only on (seed, epoch), which makes
the stream's state two integers — exact resume after a crash costs nothing.

Known trade-off: windows can span a document boundary and attention is not masked
across it (the EOT token marks the boundary). Standard for pretraining at this
scale; revisit if cross-document contamination shows up in evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from aarav_lm.data.shards import Shard


class TokenDataset:
    def __init__(self, shard: Shard, seq_len: int):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.shard = shard
        self.seq_len = seq_len
        # keep seq_len tokens of slack so any epoch offset still yields the same count
        self.n_windows = (shard.n_tokens - 1 - seq_len) // seq_len
        if self.n_windows < 1:
            raise ValueError(
                f"{shard.path.name}: {shard.n_tokens} tokens is too few for seq_len={seq_len}"
            )

    def window(self, index: int, offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        start = offset + index * self.seq_len
        chunk = np.asarray(self.shard.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        return chunk[:-1], chunk[1:]

    def batch(self, indices: np.ndarray, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = zip(*(self.window(int(i), offset) for i in indices))
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

    def fixed_eval_batches(self, batch_size: int, n_batches: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Deterministic evaluation set: windows spread evenly over the whole split."""
        n = min(batch_size * n_batches, self.n_windows)
        idx = np.linspace(0, self.n_windows - 1, num=n).astype(np.int64)
        return [self.batch(idx[i : i + batch_size]) for i in range(0, n, batch_size)]


class BatchStream:
    """Infinite shuffled batches with exact save/restore of its position."""

    def __init__(self, dataset: TokenDataset, batch_size: int, seed: int):
        self.ds = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.pos = 0
        self._order = self._make_order(0)

    def _rng(self, epoch: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, epoch])

    def _make_order(self, epoch: int) -> np.ndarray:
        return self._rng(epoch).permutation(self.ds.n_windows)

    def _offset(self, epoch: int) -> int:
        # independent stream from the permutation: seed the grid shift separately
        return int(np.random.default_rng([self.seed, epoch, 1]).integers(0, self.ds.seq_len))

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        offset = self._offset(self.epoch)
        picked: List[int] = []
        while len(picked) < self.batch_size:
            take = min(self.batch_size - len(picked), len(self._order) - self.pos)
            picked.extend(self._order[self.pos : self.pos + take].tolist())
            self.pos += take
            if self.pos >= len(self._order):     # epoch boundary: reshuffle, shift grid
                self.epoch += 1
                self.pos = 0
                self._order = self._make_order(self.epoch)
                # a batch straddling the boundary keeps the offset it started with
        return self.ds.batch(np.asarray(picked), offset)

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
