"""Shared chat template for training, the CLI, and the API.

The prompt and the answer are encoded separately and concatenated.
Inference encodes the prompt the same way, so the boundary the model
was trained on is the boundary it sees at chat time.
"""

from typing import Iterable, List, Sequence, Tuple


def user_prompt(user: str) -> str:
    return f"<|user|>\n{user.strip()}\n<|assistant|>\n"


def encode_pair(tok, user: str, assistant: str) -> Tuple[List[int], int]:
    prompt_ids = tok.encode(user_prompt(user))
    answer_ids = tok.encode(assistant.strip()) + [tok.eot]
    return prompt_ids + answer_ids, len(prompt_ids)


def messages_to_ids(tok, messages: Sequence[dict]) -> List[int]:
    """Turn OpenAI-style messages into the token ids the model continues from."""
    system_parts: List[str] = []
    turns: List[Tuple[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        text = str(message.get("content", "")).strip()
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        elif role == "assistant":
            turns.append(("assistant", text))
        else:
            turns.append(("user", text))

    ids: List[int] = []
    system = "\n".join(system_parts).strip()
    first_user = True
    for role, text in turns:
        if role == "user":
            body = f"{system}\n\n{text}" if (first_user and system) else text
            first_user = False
            ids.extend(tok.encode(user_prompt(body)))
        else:
            ids.extend(tok.encode(text))
            ids.append(tok.eot)
    return ids


def decode_answer(tok, token_ids: Iterable[int]) -> str:
    text = tok.decode(list(token_ids))
    for marker in ("<|endoftext|>", "<|user|>", "<|assistant|>"):
        text = text.replace(marker, "")
    return text.strip()
