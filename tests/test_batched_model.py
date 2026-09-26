"""Left-padded batches must give exactly the logits each sequence gives alone (RoPE makes padding invisible)."""

import pytest
import torch

from aarav_lm.config.config import cfg_5m
from aarav_lm.model.core import AaravLM
from aarav_lm.model.layers import KVCache


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return AaravLM(cfg_5m(arch="modern", vocab_size=300, block_size=64)).eval()


def cache_for(model, batch):
    c = model.cfg
    return KVCache(c.n_layer, batch, c.block_size, c.n_kv_head, c.n_embd // c.n_head, torch.device("cpu"), torch.float32)


def alone(model, ids, extra=()):
    """Reference: the sequence by itself: prefill then feed `extra` tokens one at a time; return each step's logits."""
    cache = cache_for(model, 1)
    with torch.no_grad():
        out = [model(torch.tensor([ids]), cache=cache, last_only=True)[0][0, -1]]
        cache.step(len(ids))
        for t in extra:
            out.append(model(torch.tensor([[t]]), cache=cache, last_only=True)[0][0, -1])
            cache.step(1)
    return out


def batched(model, seqs, extras):
    """The same sequences as one left-padded batch (pad id 0), each fed its own extra tokens in lock step."""
    L = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), L, dtype=torch.long)
    mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, L - len(s):] = torch.tensor(s)
        mask[i, L - len(s):] = True
    cache = cache_for(model, len(seqs))
    outs = []
    with torch.no_grad():
        logits, _ = model(x, cache=cache, last_only=True, key_mask=mask)
        cache.step(L)
        outs.append(logits[:, -1])
        for step in range(len(extras[0])):
            tok = torch.tensor([[e[step]] for e in extras])
            mask = torch.cat([mask, torch.ones(len(seqs), 1, dtype=torch.bool)], dim=1)
            logits, _ = model(tok, cache=cache, last_only=True, key_mask=mask)
            cache.step(1)
            outs.append(logits[:, -1])
    return outs


def test_batched_prefill_and_decode_match_each_sequence_alone(model):
    seqs = [[5, 9, 200, 3, 44], [7, 7, 1, 250, 18, 90, 12, 5, 33], [101], [60, 61, 62, 63, 64, 65, 66]]
    extras = [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]]
    got = batched(model, seqs, extras)
    for i, (s, e) in enumerate(zip(seqs, extras)):
        want = alone(model, s, e)
        for step, w in enumerate(want):
            assert torch.allclose(got[step][i], w, atol=2e-5, rtol=1e-4), (i, step, (got[step][i] - w).abs().max().item())


def test_padding_content_is_irrelevant_only_the_mask_matters(model):
    a, b = [5, 9, 200, 3, 44], [7, 7, 1, 250, 18, 90, 12, 5, 33]
    ref = batched(model, [a, b], [[11], [21]])[1]                          # step-1 logits of both rows: (2, V)
    L = len(b)
    x = torch.randint(1, 299, (2, L))                                     # garbage in the padding instead of zeros
    x[0, L - len(a):], x[1] = torch.tensor(a), torch.tensor(b)
    mask = torch.zeros(2, L, dtype=torch.bool); mask[0, L - len(a):] = True; mask[1] = True
    cache = cache_for(model, 2)
    with torch.no_grad():
        model(x, cache=cache, last_only=True, key_mask=mask); cache.step(L)
        logits, _ = model(torch.tensor([[11], [21]]), cache=cache, last_only=True, key_mask=torch.cat([mask, torch.ones(2, 1, dtype=torch.bool)], 1))
    assert torch.allclose(logits[:, -1], ref, atol=2e-5, rtol=1e-4)


def test_no_mask_is_the_old_behaviour_bit_for_bit(model):
    x = torch.randint(0, 300, (3, 12))
    a, _ = model(x)
    b, _ = model(x, key_mask=None)
    assert torch.equal(a, b)
    all_true, _ = model(x, key_mask=torch.ones(3, 12, dtype=torch.bool))
    assert torch.allclose(a, all_true, atol=1e-5)                                          # an all-real mask equals plain causal


def test_a_fully_padded_row_produces_finite_logits(model):
    x = torch.randint(1, 300, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.bool); mask[1] = False                              # row 1 is nothing but padding
    logits, _ = model(x, cache=cache_for(model, 2), last_only=True, key_mask=mask)
    assert torch.isfinite(logits).all()
