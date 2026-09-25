"""
aarav-lm · Chat template

    <|system|> ... <|end|> <|user|> ... <|end|> <|assistant|> ... <|end|>

Role and end markers are single reserved tokens. Message *content* is always encoded
with allow_special=False, so nothing a user types (including the literal text
"<|assistant|>") can turn into a control token.

fit_to_context() decides what survives when the conversation outgrows the window:
the system prompt is kept, the oldest turns are dropped first, and if even the newest
message alone is too long its beginning is cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from aarav_lm.tokenizer import ASSISTANT, END_TURN, SYSTEM, USER, AaravTokenizer

ROLE_TOKEN = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
MAX_MESSAGE_CHARS = 20_000


class ChatFormatError(ValueError):
    pass


@dataclass
class Message:
    role: str
    content: str

    def validate(self) -> "Message":
        if self.role not in ROLE_TOKEN:
            raise ChatFormatError(f"unknown role {self.role!r}")
        if not isinstance(self.content, str):
            raise ChatFormatError("message content must be a string")
        if len(self.content) > MAX_MESSAGE_CHARS:
            raise ChatFormatError(f"message longer than {MAX_MESSAGE_CHARS} characters")
        return self


def validate_messages(messages: Sequence[Message]) -> None:
    for i, m in enumerate(messages):
        m.validate()
        if m.role == "system" and i != 0:
            raise ChatFormatError("a system message is only allowed first")


def render_message(tok: AaravTokenizer, m: Message) -> List[int]:
    return [tok.special_ids[ROLE_TOKEN[m.role]], *tok.encode(m.content), tok.special_ids[END_TURN]]


def render_chat(tok: AaravTokenizer, messages: Sequence[Message], add_generation_prompt: bool = True) -> List[int]:
    validate_messages(messages)
    ids: List[int] = []
    for m in messages:
        ids.extend(render_message(tok, m))
    if add_generation_prompt:
        ids.append(tok.special_ids[ASSISTANT])
    return ids


def fit_to_context(
    tok: AaravTokenizer, messages: Sequence[Message], max_prompt_tokens: int
) -> Tuple[List[int], int]:
    """Token ids for the prompt that fit in `max_prompt_tokens`, plus how many messages were dropped."""
    validate_messages(messages)
    if max_prompt_tokens < 8:
        raise ChatFormatError("context window too small for a chat prompt")
    system = [m for m in messages[:1] if m.role == "system"]
    rest = list(messages[len(system):])

    head: List[int] = render_message(tok, system[0]) if system else []
    tail_reserve = 1                                              # the <|assistant|> generation prompt
    budget = max_prompt_tokens - len(head) - tail_reserve
    system_dropped = 0
    if system and budget < 4:                                     # system prompt alone eats the window
        head, budget, system_dropped = [], max_prompt_tokens - tail_reserve, 1

    kept: List[List[int]] = []
    used = 0
    for m in reversed(rest):
        chunk = render_message(tok, m)
        if used + len(chunk) > budget:
            break
        kept.append(chunk)
        used += len(chunk)
    kept.reverse()
    dropped = len(rest) - len(kept)

    if not kept and rest:                                         # newest message alone is too long: cut its start
        last = rest[-1]
        body = tok.encode(last.content)[-(budget - 2):]
        kept = [[tok.special_ids[ROLE_TOKEN[last.role]], *body, tok.special_ids[END_TURN]]]
        dropped = len(rest) - 1

    ids = list(head)
    for chunk in kept:
        ids.extend(chunk)
    ids.append(tok.special_ids[ASSISTANT])
    return ids, dropped + system_dropped
