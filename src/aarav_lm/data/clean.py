"""
aarav-lm · Text normalisation and quality filtering

normalize_text() is a pure, idempotent transformation. quality_reject_reason()
returns why a document should be dropped (or None to keep it), so every rejection
is attributable and countable in the dataset statistics.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """NFC-normalise, unify newlines, drop control characters, tidy whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def quality_reject_reason(
    text: str,
    min_chars: int = 20,
    max_chars: int = 100_000,
    max_replacement_frac: float = 0.001,
    min_alpha_frac: float = 0.5,
    min_unique_line_frac: float = 0.3,
) -> Optional[str]:
    """Return a short reason string if the document should be dropped, else None."""
    n = len(text)
    if n < min_chars:
        return "too_short"
    if n > max_chars:
        return "too_long"
    if text.count("�") / n > max_replacement_frac:
        return "malformed_utf8"
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return "empty"
    if sum(c.isalpha() for c in non_space) / len(non_space) < min_alpha_frac:
        return "low_alpha"
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 8 and len(set(lines)) / len(lines) < min_unique_line_frac:
        return "repetitive"
    return None
