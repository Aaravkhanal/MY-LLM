"""
aarav-lm · Interactive Chat CLI
Author: Aarav Khanal

Multi-turn terminal chat with streaming response.
"""

import argparse
import sys
import time
import torch

from aarav_lm.data.chat_format import decode_answer, messages_to_ids
from aarav_lm.model.core import AaravLM
from aarav_lm.tokenizer.tokenizer import Tokenizer


def chat(ckpt_path: str, temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  aarav-lm  ·  Interactive Chat REPL")
    print(f"  Model checkpoint : {ckpt_path}")
    print(f"  Running on       : {device}")
    print(f"{'='*55}")
    print("  Type 'exit' or 'quit' to end the session.\n")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = AaravLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = Tokenizer()
    history = []

    while True:
        try:
            user_input = input("\nUser > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})
        ids = messages_to_ids(tok, history)
        room = cfg.block_size - 64
        if len(ids) > room:
            ids = ids[-room:]

        idx = torch.tensor([ids], dtype=torch.long, device=device)
        print("Assistant > ", end="", flush=True)
        out = model.generate(
            idx, max_new_tokens=80, temperature=temperature, top_k=top_k, top_p=top_p,
            use_cache=True, stop_ids=[tok.eot],
        )
        response = decode_answer(tok, out[0, len(ids):].tolist())
        print(response)
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint path")
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    chat(args.ckpt, temperature=args.temp)
