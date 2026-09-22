"""
aarav-lm · Tests  (Stage 6)
Author: Aarav Khanal

Five checks that must all pass before I touch the training loop:

  1. tokenizer_roundtrip   — encode → decode → same string
  2. model_output_shapes   — correct (B, T, V) logit tensor + scalar loss
  3. causal_mask           — token[i] cannot see token[j > i]
  4. gradient_flow         — every parameter gets a non-NaN gradient
  5. kvcache_equivalence   — greedy output with cache == greedy without cache
"""

import torch
import torch.nn.functional as F

from aarav_lm.config.config import cfg_5m
from aarav_lm.model.core import AaravLM
from aarav_lm.tokenizer.tokenizer import Tokenizer


# ─────────────────────────────────────────────────────────────────────────────

def test_tokenizer_roundtrip():
    tok  = Tokenizer()
    text = "Once upon a time, Aarav built his own language model from scratch."
    assert tok.roundtrip_ok(text), "Encode→decode did not recover the original string"
    assert tok.eot == tok._enc.eot_token


def test_model_output_shapes():
    for arch in ("gpt2", "modern"):
        cfg    = cfg_5m(arch=arch, vocab_size=500, block_size=64)
        model  = AaravLM(cfg).eval()
        x      = torch.randint(0, 500, (2, 32))
        y      = torch.randint(0, 500, (2, 32))
        logits, loss = model(x, targets=y)
        assert logits.shape == (2, 32, 500), f"Bad shape for arch={arch}: {logits.shape}"
        assert loss is not None and not torch.isnan(loss)


def test_causal_mask():
    """Perturbing position ≥ 3 must not change logits at positions 0–2."""
    for arch in ("gpt2", "modern"):
        cfg    = cfg_5m(arch=arch, vocab_size=500, block_size=64)
        model  = AaravLM(cfg).eval()
        x1     = torch.tensor([[10, 20, 30, 40, 50]])
        x2     = torch.tensor([[10, 20, 30, 99, 88]])   # differs at idx ≥ 3
        with torch.no_grad():
            l1, _ = model(x1)
            l2, _ = model(x2)
        diff = (l1[:, :3] - l2[:, :3]).abs().max().item()
        assert diff < 1e-4, f"Causal mask broken in arch={arch}  (diff={diff:.2e})"


def test_gradient_flow():
    for arch in ("gpt2", "modern"):
        cfg    = cfg_5m(arch=arch, vocab_size=500, block_size=32)
        model  = AaravLM(cfg).train()
        x      = torch.randint(0, 500, (2, 16))
        y      = torch.randint(0, 500, (2, 16))
        _, loss = model(x, targets=y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None,           f"No grad: {name} in arch={arch}"
                assert not torch.isnan(p.grad).any(), f"NaN grad: {name} in arch={arch}"


def test_kvcache_equivalence():
    torch.manual_seed(42)
    for arch in ("gpt2", "modern"):
        cfg    = cfg_5m(arch=arch, vocab_size=500, block_size=64)
        model  = AaravLM(cfg).eval()
        prompt = torch.randint(0, 500, (1, 8))
        out_a  = model.generate(prompt.clone(), max_new_tokens=10, temperature=0.0, use_cache=False)
        out_b  = model.generate(prompt.clone(), max_new_tokens=10, temperature=0.0, use_cache=True)
        assert torch.equal(out_a, out_b), f"KV-cache mismatch in arch={arch}"


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    checks = [
        test_tokenizer_roundtrip,
        test_model_output_shapes,
        test_causal_mask,
        test_gradient_flow,
        test_kvcache_equivalence,
    ]
    for fn in checks:
        fn()
        print(f"  ✓  {fn.__name__}")
    print("\nAll tests passed ✓")
