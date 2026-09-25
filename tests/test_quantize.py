import pytest
import torch

from aarav_lm.config.config import cfg_tiny
from aarav_lm.inference.engine import GenerationConfig, InferenceEngine
from aarav_lm.inference.quantize import (
    dequantize_tensor, max_relative_error, quantize_model, quantize_tensor, save_int8_checkpoint,
)
from aarav_lm.model.core import AaravLM
from aarav_lm.train import checkpoint as ck


def test_per_row_error_is_at_most_half_a_step():
    w = torch.randn(64, 128) * torch.rand(64, 1) * 3
    q, s = quantize_tensor(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127 and s.shape == (64,)
    err = (dequantize_tensor(q, s) - w).abs()
    assert (err <= s.unsqueeze(1) / 2 + 1e-6).all()


def test_zero_rows_stay_exactly_zero_and_extremes_are_exact():
    w = torch.zeros(3, 8)
    w[1] = torch.tensor([1.0, -2.0, 0.5, 0, 0, 0, 0, 2.0])
    q, s = quantize_tensor(w)
    back = dequantize_tensor(q, s)
    assert torch.equal(back[0], torch.zeros(8)) and torch.equal(back[2], torch.zeros(8))
    assert back[1, 1] == pytest.approx(-2.0) and back[1, 7] == pytest.approx(2.0)      # the row max maps to ±127


def test_tied_embedding_is_stored_once_and_restored_tied():
    m = AaravLM(cfg_tiny(vocab_size=300, block_size=32))
    packed = quantize_model(m)
    assert "lm_head.weight" in packed["ties"] and "lm_head.weight" not in packed["quantized"]
    assert all(t["q"].dtype == torch.int8 for t in packed["quantized"].values())
    assert all(v.dim() < 2 or v.numel() < 1024 for v in packed["float"].values())      # norms etc. stay fp32


def test_int8_checkpoint_is_much_smaller_and_loads_like_any_other(tiny_run, tmp_path):
    from aarav_lm.tokenizer import AaravTokenizer
    tok = AaravTokenizer.load(tiny_run["tok_dir"])
    model, raw = ck.load_model(tiny_run["final"], "cpu", tok.fingerprint)
    out = tmp_path / "m_int8.pt"
    info = save_int8_checkpoint(out, model, tok.fingerprint, raw["meta"])
    assert out.stat().st_size < 0.4 * tiny_run["final"].stat().st_size
    q_model, qraw = ck.load_model(out, "cpu", tok.fingerprint)              # same loader as fp32 checkpoints
    assert qraw["kind"] == "model_int8" and qraw["meta"]["quantization"] == "int8-per-row"
    assert q_model.lm_head.weight is q_model.token_emb.weight or torch.equal(q_model.lm_head.weight, q_model.token_emb.weight)
    x = torch.randint(0, 300, (2, 16))
    with torch.no_grad():
        a, b = model(x)[0], q_model(x)[0]
    assert (a - b).abs().max() < 0.25 and torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1] > 0.999
    assert max_relative_error(model) < 0.02
    assert info["quantized_tensors"] > 0


def test_int8_checkpoint_works_end_to_end_in_the_engine(tiny_run, tmp_path):
    from aarav_lm.tokenizer import AaravTokenizer
    tok = AaravTokenizer.load(tiny_run["tok_dir"])
    model, raw = ck.load_model(tiny_run["final"], "cpu", tok.fingerprint)
    save_int8_checkpoint(tmp_path / "q.pt", model, tok.fingerprint, raw["meta"])
    eng = InferenceEngine.from_checkpoint(tmp_path / "q.pt", tiny_run["tok_dir"], "cpu")
    res = eng.generate("Once upon a time", GenerationConfig(max_new_tokens=8, temperature=0, top_k=None, top_p=None, stop_at_eot=False))
    assert res.generated_tokens == 8 and eng.meta["quantization"] == "int8-per-row"


def test_int8_checkpoint_rejects_the_wrong_tokenizer(tiny_run, tmp_path):
    from aarav_lm.tokenizer import AaravTokenizer
    tok = AaravTokenizer.load(tiny_run["tok_dir"])
    model, _ = ck.load_model(tiny_run["final"], "cpu")
    save_int8_checkpoint(tmp_path / "q.pt", model, tok.fingerprint)
    with pytest.raises(ck.CheckpointError, match="tokenizer"):
        ck.load_model(tmp_path / "q.pt", "cpu", "0" * 64)
