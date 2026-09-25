"""
aarav-lm · Prompt construction shared by the chat CLI and the API

One place decides how a list of messages becomes token ids, so the terminal and the
HTTP API always apply the same context rules.

  chat        instruction-tuned model: full template, oldest turns dropped to fit
  completion  base model: the newest user message is the text to continue
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from aarav_lm.inference.chat_format import ChatFormatError, Message, fit_to_context, validate_messages
from aarav_lm.inference.engine import InferenceEngine
from aarav_lm.tokenizer import END_TURN


@dataclass
class Prompt:
    ids: List[int]
    stop_ids: Tuple[int, ...]
    dropped: int          # messages that no longer fit in the context window


def resolve_mode(engine: InferenceEngine, mode: str = "auto") -> str:
    if mode == "auto":
        return "chat" if engine.meta.get("stage") == "sft" else "completion"
    if mode not in ("chat", "completion"):
        raise ValueError("mode must be auto, chat or completion")
    return mode


def max_prompt_tokens(engine: InferenceEngine, max_new_tokens: int) -> int:
    """Leave room for the reply, but never more than half the window."""
    return max(8, engine.block_size - min(max_new_tokens, engine.block_size // 2))


def build_prompt(
    engine: InferenceEngine, messages: Sequence[Message], max_new_tokens: int, mode: str = "auto"
) -> Prompt:
    validate_messages(messages)
    if not messages or messages[-1].role != "user":
        raise ChatFormatError("the last message must be from the user")
    budget = max_prompt_tokens(engine, max_new_tokens)
    if resolve_mode(engine, mode) == "chat":
        ids, dropped = fit_to_context(engine.tok, messages, budget)
        return Prompt(ids, (engine.tok.special_ids[END_TURN],), dropped)
    ids = engine.tok.encode(messages[-1].content)[-budget:]
    if not ids:
        raise ChatFormatError("the message is empty after tokenisation")
    return Prompt(ids, (), 0)
