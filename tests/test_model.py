"""Model tests: shapes, causality, gradients, KV-cache equivalence, init and optimizer groups."""

import torch
import torch.nn as nn

from aarav_lm.config.config import cfg_5m, cfg_tiny
from aarav_lm.model.core import AaravLM

ARCHS = ("gpt2", "modern")


def _small(arch, block_size=64, vocab=500):
    return AaravLM(cfg_5m(arch=arch, vocab_size=vocab, block_size=block_size))


def test_output_shapes_and_loss():
    for arch in ARCHS:
        model = _small(arch).eval()
        x = torch.randint(0, 500, (2, 32))
        y = torch.randint(0, 500, (2, 32))
        logits, loss = model(x, targets=y)
        assert logits.shape == (2, 32, 500)
        assert loss is not None and torch.isfinite(loss)


def test_untrained_loss_is_near_uniform():
    """A freshly initialised model should score about ln(vocab) per token."""
    model = _small("modern").eval()
    x = torch.randint(0, 500, (4, 32))
    y = torch.randint(0, 500, (4, 32))      # independent targets: with tied embeddings,
    _, loss = model(x, targets=y)           # targets == inputs would favour copying
    assert abs(loss.item() - torch.log(torch.tensor(500.0)).item()) < 0.5


def test_ignore_index_masks_targets():
    model = _small("modern").eval()
    x = torch.randint(0, 500, (1, 16))
    y = x.clone()
    y[:, :8] = -1
    _, masked = model(x, targets=y)
    _, tail_only = model(x, targets=torch.cat([torch.full((1, 8), -1), x[:, 8:]], dim=1))
    assert torch.allclose(masked, tail_only)


def test_causal_mask():
    for arch in ARCHS:
        model = _small(arch).eval()
        x1 = torch.tensor([[10, 20, 30, 40, 50]])
        x2 = torch.tensor([[10, 20, 30, 99, 88]])
        with torch.no_grad():
            l1, _ = model(x1)
            l2, _ = model(x2)
        assert (l1[:, :3] - l2[:, :3]).abs().max().item() < 1e-4, arch


def test_gradient_flow():
    for arch in ARCHS:
        model = _small(arch, block_size=32).train()
        x = torch.randint(0, 500, (2, 16))
        _, loss = model(x, targets=x)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{arch}: no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"{arch}: bad grad for {name}"


def test_kvcache_equivalence_greedy():
    for arch in ARCHS:
        model = _small(arch).eval()
        prompt = torch.randint(0, 500, (1, 8))
        a = model.generate(prompt.clone(), max_new_tokens=10, temperature=0.0, use_cache=False)
        b = model.generate(prompt.clone(), max_new_tokens=10, temperature=0.0, use_cache=True)
        assert torch.equal(a, b), arch


def test_chunked_prefill_matches_full_forward():
    """Feeding a prompt in two chunks through the cache must equal one full pass."""
    from aarav_lm.model.layers import KVCache

    for arch in ARCHS:
        model = _small(arch).eval()
        cfg = model.cfg
        x = torch.randint(0, 500, (1, 24))
        with torch.no_grad():
            full, _ = model(x)
            cache = KVCache(cfg.n_layer, 1, cfg.block_size, cfg.n_kv_head,
                            cfg.n_embd // cfg.n_head, torch.device("cpu"), torch.float32)
            first, _ = model(x[:, :10], cache=cache)
            cache.step(10)
            second, _ = model(x[:, 10:], cache=cache)
            cache.step(14)
        assert torch.allclose(full[:, :10], first, atol=1e-5), arch
        assert torch.allclose(full[:, 10:], second, atol=1e-5), arch


def test_last_only_returns_final_position():
    model = _small("modern").eval()
    x = torch.randint(0, 500, (2, 12))
    with torch.no_grad():
        full, _ = model(x)
        last, _ = model(x, last_only=True)
    assert last.shape == (2, 1, 500)
    assert torch.allclose(full[:, -1:], last, atol=1e-6)


def test_residual_projections_get_scaled_init_in_both_archs():
    for arch in ARCHS:
        model = _small(arch)
        target = 0.02 / (2 * model.cfg.n_layer) ** 0.5
        names = [n for n, _ in model.named_parameters()
                 if n.endswith(("attn.out.weight", "mlp.down.weight", "mlp.proj.weight"))]
        assert names, arch
        for n, p in model.named_parameters():
            if n in names:
                assert abs(p.std().item() - target) < target * 0.25, (arch, n)


def test_weight_decay_groups_exclude_embeddings_norms_and_biases():
    model = _small("modern")
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    decay_ids = {id(p) for p in decay["params"]}
    assert id(model.token_emb.weight) not in decay_ids
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(id(m.weight) not in decay_ids
               for m in model.modules() if isinstance(m, (nn.Embedding,)))
    total = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert total == model.num_params()   # every parameter is in exactly one group


def test_tied_embeddings_counted_once():
    model = AaravLM(cfg_tiny(vocab_size=2048))
    assert model.lm_head.weight is model.token_emb.weight
    body = model.num_params(exclude_embeddings=True)
    assert model.num_params() == body + 2048 * model.cfg.n_embd


def test_tiny_preset_size():
    n = AaravLM(cfg_tiny(vocab_size=2048)).num_params()
    assert 4_500_000 < n < 5_500_000, n
