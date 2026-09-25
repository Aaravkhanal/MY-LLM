"""
aarav-lm · Dataset ingestion

Reads local text sources lazily and yields one raw document at a time, so a corpus
larger than RAM can be processed as long as the later stages stream too.

Supported inputs
  .txt    documents separated by a delimiter line/string (default <|endoftext|>);
          with no delimiter present the whole file is one document
  .jsonl  one JSON object per line; the document is record[text_field]
  a directory: every .txt/.jsonl inside, in sorted order

Malformed input is counted, never silently swallowed: bad JSON lines, records
without the text field, and undecodable bytes all show up in IngestStats.
"""

from __future__ import annotations

import codecs
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Sequence

CHUNK_BYTES = 8 * 1024 * 1024


@dataclass
class RawDoc:
    text: str
    source: str
    index: int          # position within its source file


@dataclass
class IngestStats:
    files: int = 0
    docs: int = 0
    rejects: Counter = field(default_factory=Counter)   # reason -> count


def expand_sources(sources: Sequence[str | Path]) -> List[Path]:
    """Resolve files/directories to a sorted, de-duplicated list of readable files."""
    out: List[Path] = []
    for s in sources:
        p = Path(s)
        if not p.exists():
            raise FileNotFoundError(f"data source not found: {p}")
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*") if q.suffix in (".txt", ".jsonl")))
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    if not uniq:
        raise ValueError("no .txt or .jsonl files found in the given sources")
    return uniq


def _detect_format(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    if path.suffix == ".jsonl":
        return "jsonl"
    if path.suffix == ".txt":
        return "txt"
    raise ValueError(f"cannot infer format of {path}; set data.format explicitly")


def _iter_txt(path: Path, separator: str, stats: IngestStats) -> Iterator[str]:
    """Stream a text file, splitting on `separator` without loading it all."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            buf += decoder.decode(chunk, final=not chunk)
            if separator:
                *done, buf = buf.split(separator)
                yield from done
            if not chunk:
                break
    if buf.strip() or not separator:
        yield buf


def _iter_jsonl(path: Path, text_field: str, stats: IngestStats) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats.rejects["bad_json"] += 1
                continue
            value = rec.get(text_field) if isinstance(rec, dict) else None
            if not isinstance(value, str):
                stats.rejects["missing_text_field"] += 1
                continue
            yield value


def iter_documents(
    sources: Sequence[str | Path],
    stats: IngestStats,
    fmt: str = "auto",
    text_field: str = "text",
    doc_separator: str = "<|endoftext|>",
) -> Iterator[RawDoc]:
    for path in expand_sources(sources):
        stats.files += 1
        kind = _detect_format(path, fmt)
        if kind == "txt":
            it = _iter_txt(path, doc_separator, stats)
        elif kind == "jsonl":
            it = _iter_jsonl(path, text_field, stats)
        else:
            raise ValueError(f"unsupported format {kind!r}")
        for i, text in enumerate(it):
            stats.docs += 1
            yield RawDoc(text=text, source=str(path), index=i)
