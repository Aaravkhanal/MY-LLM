"""
AaravLM Interactive Chat / Terminal REPL
"""

import argparse
import sys
import torch

from aarav_lm.config.config import ModelConfig
from aarav_lm.model.core import AaravLM
from aarav_lm.tokenizer.tokenizer import Tokenizer


def start_chat(checkpoint_path: str, temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model from {checkpoint_path} onto {device}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg: ModelConfig = checkpoint["config"]
    model = AaravLM(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = Tokenizer()

    print("\n" + "=" * 50)
    print("Welcome to AaravLM Interactive Terminal!")
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to end.")
    print("=" * 50 + "\n")

    while True:
        try:
            prompt = input("User > ")
            if not prompt.strip():
                continue
            if prompt.strip().lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            tokens = tokenizer.encode(prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)

            print("\nAaravLM > ", end="", flush=True)
            with torch.no_grad():
                out_idx = model.generate(
                    idx=idx,
                    max_new_tokens=100,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    use_cache=True,
                )
            # Decode only new tokens
            new_tokens = out_idx[0, len(tokens):].tolist()
            reply = tokenizer.decode(new_tokens)
            print(reply + "\n")

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with AaravLM")
    parser.add_argument("--checkpoint", type=str, default="aarav-lm/checkpoints/aaravlm_modern_5m_best.pt")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    start_chat(args.checkpoint, temperature=args.temperature)
