"""
aarav-lm · Dataloader
Author: Aarav Khanal

Reads memory-mapped binary token files (train.bin / val.bin) and returns
random (input, target) batches.  File handles are cached on first open so
there is zero per-batch I/O overhead on repeat calls.

Synthetic mode returns random tensors and never touches disk — useful for
smoke-testing the training loop before data is prepared.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

_CACHE: Dict[str, np.memmap] = {}   # global memmap handle cache

_HERE = Path(__file__).resolve().parent


def get_batch(
    split:       str,
    block_size:  int,
    batch_size:  int,
    device:      torch.device,
    data_dir:    Optional[Union[str, Path]] = None,
    synthetic:   bool = False,
    vocab_size:  int  = 50257,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return a (x, y) batch of shape (batch_size, block_size).

    Parameters
    ----------
    split      : 'train' or 'val'
    data_dir   : folder containing train.bin / val.bin
    synthetic  : if True, return random tensors (no disk access)
    """
    if synthetic:
        x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
        y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
        return x, y

    root = Path(data_dir) if data_dir is not None else _HERE
    path = str(root / f"{split}.bin")

    if path not in _CACHE:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"'{path}' not found. "
                "Run  python -m aarav_lm.data.prepare_data  first, "
                "or pass  synthetic=True  to use random data."
            )
        _CACHE[path] = np.memmap(path, dtype=np.uint16, mode="r")

    data    = _CACHE[path]
    max_idx = len(data) - block_size - 1

    if max_idx <= 0:
        raise ValueError(
            f"Dataset at '{path}' only has {len(data)} tokens "
            f"but block_size={block_size} requires at least {block_size + 2}."
        )

    ix  = np.random.randint(0, max_idx, size=batch_size)
    x   = torch.from_numpy(np.stack([data[i     : i + block_size].astype(np.int64) for i in ix])).to(device)
    y   = torch.from_numpy(np.stack([data[i + 1 : i + block_size + 1].astype(np.int64) for i in ix])).to(device)

    return x, y
