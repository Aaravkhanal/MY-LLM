import copy

import pytest
import torch

from aarav_lm.config.config import cfg_tiny
from aarav_lm.finetune.lora import (
    LoRALinear, apply_lora, count_parameters, has_lora, merge_lora, trainable_parameters,
)
from aarav_lm.model.core import AaravLM


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return AaravLM(cfg_tiny(vocab_size=200, block_size=32)).eval()


def test_fresh_adapters_leave_the_model_exactly_unchanged(model):
    x = torch.randint(0, 200, (2, 16))
    before = model(x)[0].detach().clone()
    apply_lora(model, rank=4)
    assert torch.equal(model(x)[0], before)                      # B = 0, so the update is exactly zero


def test_only_adapter_parameters_train_and_they_are_a_small_fraction(model):
    names = apply_lora(model, rank=4)
    assert names and has_lora(model)
    train = trainable_parameters(model)
    assert train, 'no trainable parameters'
    assert {n.split(".")[-1] for n, p in model.named_parameters() if p.requires_grad} == {"lora_A", "lora_B"}
    counts = count_parameters(model)
    assert counts["trainable"] == sum(p.numel() for p in train) and counts["trainable"] < 0.15 * counts["total"]


def test_gradients_reach_only_the_adapters(model):
    apply_lora(model, rank=4)
    x = torch.randint(0, 200, (2, 16))
    model.train()
    model(x, targets=x)[1].backward()
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, n
        else:
            assert p.grad is None, n


def test_merge_reproduces_the_adapted_model_and_removes_the_wrappers(model):
    apply_lora(model, rank=4, alpha=8)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B, std=0.05)            # pretend training happened
    x = torch.randint(0, 200, (2, 16))
    adapted = model(x)[0].detach().clone()
    merge_lora(model)
    assert not has_lora(model) and all(p.requires_grad for p in model.parameters())
    assert torch.allclose(model(x)[0], adapted, atol=1e-5)
    assert set(model.state_dict()) == set(AaravLM(model.cfg).state_dict())     # loads into a plain model


def test_merged_model_differs_from_base_once_adapters_are_trained(model):
    base = copy.deepcopy(model)
    apply_lora(model, rank=2)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.lora_B, std=0.05)
    merge_lora(model)
    x = torch.randint(0, 200, (1, 16))
    assert not torch.allclose(model(x)[0], base(x)[0], atol=1e-4)


def test_target_selection_and_errors(model):
    names = apply_lora(model, rank=2, targets=("attn.q", "attn.v"))
    assert all(n.endswith(("attn.q", "attn.v")) for n in names) and len(names) == 2 * model.cfg.n_layer
    with pytest.raises(ValueError, match="no Linear"):
        apply_lora(AaravLM(cfg_tiny(vocab_size=200, block_size=32)), rank=2, targets=("nonexistent",))
    with pytest.raises(ValueError):
        LoRALinear(torch.nn.Linear(4, 4), rank=0, alpha=1)


def test_lm_head_is_never_wrapped(model):
    names = apply_lora(model, rank=2, targets=("lm_head", "attn.q"))
    assert not any("lm_head" in n for n in names)
