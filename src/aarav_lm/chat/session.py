"""
aarav-lm · Chat session

Holds the conversation (system prompt, user/assistant turns), builds prompts that fit
the model's context window, streams replies from the local InferenceEngine, and can
save/load itself as JSON.

Two modes, chosen from the checkpoint:
  chat        model was instruction-tuned on the chat template (stage "sft")
  completion  base model (stage "pretrain"): it has only seen raw text, so each user
              message is used as the start of a text to continue, without history
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from aarav_lm.inference.chat_format import ChatFormatError, Message, validate_messages
from aarav_lm.inference.engine import GenerationConfig, InferenceEngine
from aarav_lm.inference.prompting import build_prompt, resolve_mode

SESSION_VERSION = 1
MAX_SESSION_FILE_BYTES = 5_000_000


class SessionFileError(ValueError):
    pass


_GEN_TYPES = {                      # field -> accepted types (bool is never a valid number)
    "max_new_tokens": (int,), "temperature": (int, float), "top_k": (int, type(None)),
    "top_p": (int, float, type(None)), "repetition_penalty": (int, float), "stop_sequences": (list,),
    "stop_ids": (list,), "stop_at_eot": (bool,), "seed": (int, type(None)), "no_repeat_ngram_size": (int,),
}


def _generation_from_dict(gen) -> GenerationConfig:
    """Strictly typed: a hand-edited session file must fail here, not later inside torch."""
    if not isinstance(gen, dict):
        raise SessionFileError("generation must be an object")
    unknown = set(gen) - set(_GEN_TYPES)
    if unknown:
        raise SessionFileError(f"unknown generation field(s): {sorted(unknown)}")
    for k, v in gen.items():
        if isinstance(v, bool) and bool not in _GEN_TYPES[k]:
            raise SessionFileError(f"generation.{k} has the wrong type")
        if not isinstance(v, _GEN_TYPES[k]):
            raise SessionFileError(f"generation.{k} has the wrong type ({type(v).__name__})")
    if not all(isinstance(x, str) for x in gen.get("stop_sequences", [])) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in gen.get("stop_ids", [])):
        raise SessionFileError("stop_sequences must be strings and stop_ids integers")
    return GenerationConfig(**{**gen, "stop_sequences": tuple(gen.get("stop_sequences", ())),
                               "stop_ids": tuple(gen.get("stop_ids", ()))}).validate()


class ChatSession:
    def __init__(
        self,
        engine: InferenceEngine,
        system_prompt: str = "",
        gen_cfg: Optional[GenerationConfig] = None,
        mode: str = "auto",
    ):
        self.engine = engine
        self.gen_cfg = gen_cfg or GenerationConfig()
        self.mode = resolve_mode(engine, mode)
        self.system_prompt = system_prompt
        self.messages: List[Message] = []
        self.turns = 0
        self.dropped_messages = 0
        self.total_generated = 0
        self.last_tokens_per_sec = 0.0
        self.last_context_tokens = 0

    # ── conversation ───────────────────────────────────────────────────────
    def _prompt_messages(self) -> List[Message]:
        msgs = list(self.messages)
        if self.system_prompt:
            msgs = [Message("system", self.system_prompt), *msgs]
        return msgs

    def send(self, user_text: str) -> Iterator[str]:
        """Add a user message, stream the reply, and record it. Yields text deltas."""
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("empty message")
        Message("user", user_text).validate()
        eng = self.engine
        candidate = self._prompt_messages() + [Message("user", user_text)]
        prompt = build_prompt(eng, candidate, self.gen_cfg.max_new_tokens, self.mode)
        ids = prompt.ids
        self.dropped_messages += prompt.dropped
        cfg = replace(self.gen_cfg, stop_ids=tuple({*self.gen_cfg.stop_ids, *prompt.stop_ids}))

        self.last_context_tokens = len(ids)
        with eng.lock:
            yield from eng.stream(ids, cfg)
            result = eng.last_result
        self.messages += [Message("user", user_text), Message("assistant", result.text.strip())]
        self.turns += 1
        self.total_generated += result.generated_tokens
        self.last_tokens_per_sec = result.tokens_per_sec

    def clear(self) -> None:
        """Forget the conversation; keep the system prompt and generation settings."""
        self.messages, self.turns, self.dropped_messages = [], 0, 0

    def reset(self) -> None:
        """Back to a fresh session: also drops the system prompt and generation settings."""
        self.clear()
        self.system_prompt = ""
        self.gen_cfg = GenerationConfig()

    # ── persistence ────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        data = {
            "version": SESSION_VERSION,
            "tokenizer_fingerprint": self.engine.tok.fingerprint,
            "mode": self.mode,
            "system_prompt": self.system_prompt,
            "generation": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self.gen_cfg).items()},
            "messages": [asdict(m) for m in self.messages],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load(self, path: str | Path) -> int:
        path = Path(path)
        if not path.is_file():
            raise SessionFileError(f"no such file: {path}")
        if path.stat().st_size > MAX_SESSION_FILE_BYTES:
            raise SessionFileError("session file is too large")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise SessionFileError("session file must contain a JSON object")
            if data.get("version") != SESSION_VERSION:
                raise SessionFileError(f"unsupported session version {data.get('version')!r}")
            if data["tokenizer_fingerprint"] != self.engine.tok.fingerprint:
                raise SessionFileError("session was saved with a different tokenizer/model")
            if not isinstance(data["messages"], list) or not all(isinstance(m, dict) for m in data["messages"]):
                raise SessionFileError("messages must be a list of objects")
            messages = [Message(m["role"], m["content"]) for m in data["messages"]]
            validate_messages(messages)
            gen_cfg = _generation_from_dict(data["generation"])
            system_prompt = data["system_prompt"]
            if not isinstance(system_prompt, str):
                raise SessionFileError("system_prompt must be a string")
        except SessionFileError:
            raise
        except (KeyError, TypeError, AttributeError, ValueError, OSError, UnicodeDecodeError, ChatFormatError) as e:
            raise SessionFileError(f"invalid session file: {type(e).__name__}: {e}") from e
        self.messages, self.system_prompt, self.gen_cfg = messages, system_prompt, gen_cfg
        self.turns = sum(m.role == "assistant" for m in messages)
        return len(messages)

    # ── stats ──────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, object]:
        info = self.engine.info()
        return {
            "mode": self.mode, "turns": self.turns, "messages": len(self.messages),
            "dropped_for_context": self.dropped_messages, "generated_tokens": self.total_generated,
            "last_tokens_per_sec": round(self.last_tokens_per_sec, 1),
            "last_prompt_tokens": self.last_context_tokens, "context_length": info["context_length"],
            "params": info["params"], "device": info["device"],
        }
