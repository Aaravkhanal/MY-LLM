"""
aarav-lm · Byte-level BPE tokenizer (trained from scratch, versioned with the model)

Why byte-level BPE: it can encode any text (no <unk>), and a small learned vocab
(2-4k) keeps the embedding table and the logits tensor cheap on an 8 GB machine.
Character-level would need 3-4x more positions per story; a 50k GPT-2 vocab would
make the embedding matrix ~65% of a 5M-parameter model.

A tokenizer is a directory:

    tokenizer.json   the HF `tokenizers` model (merges + vocab + specials)
    meta.json        version, vocab size, special tokens, sha256 of tokenizer.json,
                     training-corpus fingerprint

Checkpoints and token shards record `meta.json["sha256"]`, so loading a model with
the wrong tokenizer is a hard error rather than silent garbage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tokenizers import Tokenizer as _HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

TOKENIZER_FORMAT_VERSION = 1

EOT = "<|endoftext|>"     # document separator; also the generation stop token
PAD = "<|pad|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END_TURN = "<|end|>"      # closes a chat turn

# Order fixes the ids (0..6). Chat tokens are reserved now so that supervised
# fine-tuning later never has to resize the embedding table.
SPECIAL_TOKENS: List[str] = [EOT, PAD, SYSTEM, USER, ASSISTANT, END_TURN]


class TokenizerMismatchError(RuntimeError):
    """A checkpoint or dataset was built with a different tokenizer than the one loaded."""


class AaravTokenizer:
    """Thin wrapper over a trained byte-level BPE with explicit special-token handling."""

    def __init__(self, tok: _HFTokenizer, meta: dict):
        self._tok = tok
        self.meta = meta
        self.vocab_size: int = tok.get_vocab_size()
        self.special_ids = {s: tok.token_to_id(s) for s in SPECIAL_TOKENS}
        if any(i is None for i in self.special_ids.values()):
            raise ValueError("tokenizer is missing required special tokens")
        self.eot_id: int = self.special_ids[EOT]
        self.pad_id: int = self.special_ids[PAD]

    # ── identity ───────────────────────────────────────────────────────────
    @property
    def fingerprint(self) -> str:
        """sha256 of tokenizer.json — what checkpoints and shards are tied to."""
        return self.meta["sha256"]

    def check_fingerprint(self, expected: str, what: str = "artifact") -> None:
        if expected != self.fingerprint:
            raise TokenizerMismatchError(
                f"{what} was built with tokenizer {expected[:12]}…, "
                f"but the loaded tokenizer is {self.fingerprint[:12]}…"
            )

    # ── encode / decode ────────────────────────────────────────────────────
    def encode(self, text: str, allow_special: bool = False) -> List[int]:
        """Text -> ids.

        allow_special=False (default) treats strings like "<|endoftext|>" as ordinary
        text, so user input can never inject control tokens. Pass True only for
        trusted template text.
        """
        self._tok.encode_special_tokens = not allow_special
        return self._tok.encode(text).ids

    def encode_batch(self, texts: Sequence[str], allow_special: bool = False) -> List[List[int]]:
        self._tok.encode_special_tokens = not allow_special
        return [e.ids for e in self._tok.encode_batch(list(texts))]

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special)

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    def id_to_token(self, i: int) -> Optional[str]:
        return self._tok.id_to_token(i)

    def roundtrip_ok(self, text: str) -> bool:
        return self.decode(self.encode(text)) == text

    # ── persistence ────────────────────────────────────────────────────────
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "tokenizer.json").write_text(self._tok.to_str(), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2, sort_keys=True), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "AaravTokenizer":
        d = Path(directory)
        raw = (d / "tokenizer.json").read_text(encoding="utf-8")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        actual = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if actual != meta["sha256"]:
            raise TokenizerMismatchError(
                f"{d}: tokenizer.json does not match meta.json (file was modified after training)"
            )
        return cls(_HFTokenizer.from_str(raw), meta)


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 2048,
    corpus_info: Optional[dict] = None,
) -> AaravTokenizer:
    """Train a byte-level BPE on an iterable of documents.

    Train on the *training split only* — fitting merges on validation text would
    leak information into the held-out metrics.
    """
    n_base = 256 + len(SPECIAL_TOKENS)
    if vocab_size <= n_base:
        raise ValueError(f"vocab_size must exceed {n_base} (256 bytes + {len(SPECIAL_TOKENS)} specials)")

    tok = _HFTokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)

    raw = tok.to_str()
    meta = {
        "format_version": TOKENIZER_FORMAT_VERSION,
        "kind": "byte_level_bpe",
        "vocab_size": tok.get_vocab_size(),
        "requested_vocab_size": vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "corpus": corpus_info or {},
    }
    return AaravTokenizer(tok, meta)
