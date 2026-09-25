"""Write an int8-quantized copy of a checkpoint and report the size and quality cost.

    python scripts/export_int8.py --checkpoint experiments/exp_001_x/checkpoints/best.pt
"""

import argparse
from pathlib import Path

import yaml

from aarav_lm.data.dataset import TokenDataset
from aarav_lm.data.shards import Shard
from aarav_lm.evaluation.loss import evaluate_loss
from aarav_lm.inference.quantize import max_relative_error, save_int8_checkpoint
from aarav_lm.paths import resolve
from aarav_lm.tokenizer import AaravTokenizer
from aarav_lm.train import checkpoint as ck


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", help="default: <checkpoint>_int8.pt next to the source")
    ap.add_argument("--eval-windows", type=int, default=512, help="validation windows used to measure the loss change")
    args = ap.parse_args()

    src = Path(args.checkpoint)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_int8.pt")
    cfg = yaml.safe_load((src.parent.parent / "config.yaml").read_text())
    tok = AaravTokenizer.load(resolve(cfg["tokenizer"]["artifact_dir"]))
    model, raw = ck.load_model(src, "cpu", tok.fingerprint)
    info = save_int8_checkpoint(out, model, tok.fingerprint, raw.get("meta"))
    q_model, _ = ck.load_model(out, "cpu", tok.fingerprint)

    ds = TokenDataset(Shard.open(resolve(cfg["data"]["out_dir"]) / "val.bin"), model.cfg.block_size)
    batches = ds.fixed_eval_batches(16, max(1, args.eval_windows // 16))
    base = evaluate_loss(model, batches, model.token_emb.weight.device)
    quant = evaluate_loss(q_model, batches, model.token_emb.weight.device)
    mb = lambda p: Path(p).stat().st_size / 1e6                          # noqa: E731
    print(f"fp32 checkpoint : {mb(src):8.2f} MB")
    print(f"int8 checkpoint : {mb(out):8.2f} MB   ({mb(src) / mb(out):.2f}x smaller)")
    print(f"quantized tensors: {info['quantized_tensors']}, kept fp32: {info['float_tensors']}; "
          f"worst per-tensor relative error {max_relative_error(model):.4f}")
    print(f"validation loss : {base['loss']:.4f} -> {quant['loss']:.4f}  "
          f"(perplexity {base['perplexity']:.3f} -> {quant['perplexity']:.3f}, "
          f"{(quant['loss'] / base['loss'] - 1) * 100:+.2f}% loss)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
