"""
aarav-lm · Tokenizer
Author: Aarav Khanal

Thin wrapper around tiktoken's byte-pair encoding so the rest of
the codebase never has to know which underlying tokenizer is in use.

Supported encodings
-------------------
  'gpt2'       — 50 257 tokens, standard BPE used throughout training
  'cl100k_base' — 100 277 tokens, OpenAI's newer encoding
"""

from typing import List, Set, Union
import tiktoken


class Tokenizer:
    """A clean, minimal BPE tokenizer interface for aarav-lm."""

    def __init__(self, encoding: str = "gpt2"):
        self._enc        = tiktoken.get_encoding(encoding)
        self.vocab_size  = self._enc.n_vocab
        self.eot         = self._enc.eot_token   # end-of-text / EOS
        self.bos         = self.eot               # reuse eot as BOS (like GPT-2)
        self.pad         = self.eot

    # ── Core API ───────────────────────────────────────────────────────────
    def encode(self, text: str, allowed_special: Union[str, Set[str]] = "all") -> List[int]:
        return self._enc.encode(text, allowed_special=allowed_special)

    def encode_ordinary(self, text: str) -> List[int]:
        """Encode without emitting any special tokens."""
        return self._enc.encode_ordinary(text)

    def decode(self, tokens: List[int]) -> str:
        return self._enc.decode(tokens)

    # ── Utility ────────────────────────────────────────────────────────────
    def roundtrip_ok(self, text: str) -> bool:
        """Return True if encode → decode recovers the original string exactly."""
        return self.decode(self.encode(text)) == text

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size})"


def build_tokenizer(encoding: str = "gpt2") -> Tokenizer:
    return Tokenizer(encoding=encoding)
