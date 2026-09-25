"""
aarav-lm · Text Generation CLI
Author: Aarav Khanal
"""

import argparse
import time
import torch

from aarav_lm.config.config import ModelConfig
from aarav_lm.model.core import AaravLM
from aarav_lm.tokenizer.tokenizer import Tokenizer


def generate(
    ckpt_path:    str,
    prompt:       str   = "To be or not to be,",
    max_tokens:   int   = 100,
    temperature:  float = 0.8,
    top_k:        int   = 40,
    top_p:        float = 0.9,
    use_cache:    bool  = True,
    device_name:  str   = "",
):
    device = torch.device(device_name or ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Loading  {ckpt_path}  on  {device}")

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = AaravLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok   = Tokenizer()
    ids   = tok.encode(prompt)
    idx   = torch.tensor([ids], dtype=torch.long, device=device)

    print(f"\n{'─'*50}\n{prompt}", end="", flush=True)
    t0    = time.time()
    out   = model.generate(idx, max_tokens, temperature=temperature, top_k=top_k, top_p=top_p, use_cache=use_cache)
    dt    = time.time() - t0

    new_tok = out.shape[1] - len(ids)
    text    = tok.decode(out[0].tolist())
    print(text[len(prompt):])
    print(f"\n{'─'*50}")
    print(f"{new_tok} tokens in {dt:.2f}s  ({new_tok/dt:.1f} tok/s,  cache={use_cache})\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",        required=True)
    ap.add_argument("--prompt",      default="To be or not to be,")
    ap.add_argument("--max_tokens",  type=int,   default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k",       type=int,   default=40)
    ap.add_argument("--top_p",       type=float, default=0.9)
    ap.add_argument("--no_cache",    action="store_true")
    args = ap.parse_args()
    generate(args.ckpt, args.prompt, args.max_tokens, args.temperature, args.top_k, args.top_p, not args.no_cache)
