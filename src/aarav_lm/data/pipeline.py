"""
aarav-lm · Dataset build pipeline

    ingest -> normalise -> quality filter -> exact dedup -> split (by content hash)
           -> train tokenizer on the TRAIN split only (or reuse a versioned one)
           -> tokenise -> token shards + manifest.json + stats.json

Everything that could change the result (config, source-file hashes, tokenizer
fingerprint) is recorded in manifest.json, so a dataset directory is self-describing
and validate_dataset() can prove it still matches the tokenizer it will be used with.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from aarav_lm.config.experiment import ExperimentConfig
from aarav_lm.data.clean import normalize_text, quality_reject_reason
from aarav_lm.data.dedup import ExactDeduper
from aarav_lm.data.ingest import IngestStats, expand_sources, iter_documents
from aarav_lm.data.shards import Shard, ShardWriter
from aarav_lm.data.split import split_of
from aarav_lm.tokenizer import AaravTokenizer, train_tokenizer

SPLITS = ("train", "val", "test")
ENCODE_BATCH = 2000


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collect_documents(exp: ExperimentConfig) -> tuple[Dict[str, List[str]], dict]:
    """Run ingest -> clean -> filter -> dedup -> split. Returns docs per split + counters."""
    d = exp.data
    ingest = IngestStats()
    rejects: Counter = Counter()
    deduper = ExactDeduper()
    docs: Dict[str, List[str]] = {s: [] for s in SPLITS}
    n_dupes = 0

    for raw in iter_documents(d.sources, ingest, d.format, d.text_field, d.doc_separator):
        text = normalize_text(raw.text)
        reason = quality_reject_reason(text, min_chars=d.min_chars, max_chars=d.max_chars)
        if reason:
            rejects[reason] += 1
            continue
        if d.dedup and deduper.is_duplicate(text):
            n_dupes += 1
            continue
        docs[split_of(text, d.seed, d.val_fraction, d.test_fraction)].append(text)

    counters = {
        "files": ingest.files,
        "docs_read": ingest.docs,
        "ingest_rejects": dict(ingest.rejects),
        "quality_rejects": dict(rejects),
        "duplicates_removed": n_dupes,
        "docs_kept": {s: len(v) for s, v in docs.items()},
    }
    return docs, counters


def _tokenize_split(
    texts: List[str], tok: AaravTokenizer, writer: ShardWriter
) -> tuple[np.ndarray, int]:
    """Encode documents, append each followed by EOT, return per-doc token counts."""
    lens: List[int] = []
    n_bytes = 0
    for i in range(0, len(texts), ENCODE_BATCH):
        batch = texts[i : i + ENCODE_BATCH]
        encoded = tok.encode_batch(batch)
        flat: List[int] = []
        for text, ids in zip(batch, encoded):
            flat.extend(ids)
            flat.append(tok.eot_id)
            lens.append(len(ids) + 1)
            n_bytes += len(text.encode("utf-8"))
        writer.append(np.asarray(flat, dtype=np.int64), n_docs=len(batch))
    return np.asarray(lens, dtype=np.int64), n_bytes


def _length_summary(lens: np.ndarray) -> dict:
    if lens.size == 0:
        return {}
    p = np.percentile(lens, [5, 25, 50, 75, 95, 99])
    return {
        "min": int(lens.min()), "max": int(lens.max()), "mean": round(float(lens.mean()), 2),
        "p05": float(p[0]), "p25": float(p[1]), "p50": float(p[2]),
        "p75": float(p[3]), "p95": float(p[4]), "p99": float(p[5]),
    }


def build_dataset(exp: ExperimentConfig, log: Callable[[str], None] = print, force_tokenizer: bool = False) -> dict:
    """Build tokenizer + token shards for `exp`. Returns the manifest."""
    t0 = time.time()
    out_dir = Path(exp.data.out_dir)
    tok_dir = Path(exp.tokenizer.artifact_dir)

    log(f"[data] reading {len(expand_sources(exp.data.sources))} source file(s)")
    docs, counters = collect_documents(exp)
    log(f"[data] kept {counters['docs_kept']}  dupes removed {counters['duplicates_removed']}  "
        f"rejected {counters['quality_rejects'] or 0}")
    if not docs["train"]:
        raise RuntimeError("no training documents survived cleaning")

    # ── tokenizer: versioned artifact, trained on train split only ─────────
    if (tok_dir / "tokenizer.json").exists() and not force_tokenizer:
        tok = AaravTokenizer.load(tok_dir)
        if tok.meta["requested_vocab_size"] != exp.tokenizer.vocab_size:
            raise RuntimeError(
                f"{tok_dir} holds a vocab-{tok.meta['requested_vocab_size']} tokenizer but the config "
                f"asks for {exp.tokenizer.vocab_size}; pass force_tokenizer=True or use a new artifact_dir"
            )
        log(f"[tok] reusing tokenizer {tok.fingerprint[:12]} from {tok_dir}")
    else:
        log(f"[tok] training byte-level BPE (vocab {exp.tokenizer.vocab_size}) on {len(docs['train']):,} train docs")
        tok = train_tokenizer(
            docs["train"],
            vocab_size=exp.tokenizer.vocab_size,
            corpus_info={"n_train_docs": len(docs["train"]), "data_seed": exp.data.seed},
        )
        tok.save(tok_dir)
        log(f"[tok] saved {tok_dir}  vocab {tok.vocab_size}  fingerprint {tok.fingerprint[:12]}")

    # ── tokenise + write shards ────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    split_stats: Dict[str, dict] = {}
    for split in SPLITS:
        with ShardWriter(out_dir / f"{split}.bin", tok.vocab_size) as w:
            lens, n_bytes = _tokenize_split(docs[split], tok, w)
        n_tokens = int(lens.sum()) if lens.size else 0
        split_stats[split] = {
            "docs": len(docs[split]),
            "tokens": n_tokens,
            "bytes_per_token": round(n_bytes / n_tokens, 3) if n_tokens else None,
            "doc_token_length": _length_summary(lens),
        }
        log(f"[data] {split:5s} {len(docs[split]):>8,} docs {n_tokens:>12,} tokens")

    sources = {str(p): _file_sha256(p) for p in expand_sources(exp.data.sources)}
    manifest = {
        "format_version": 1,
        "experiment": exp.name,
        "config_fingerprint": exp.fingerprint(),
        "data_config": exp.data.__dict__,
        "tokenizer": {"dir": str(tok_dir), "fingerprint": tok.fingerprint, "vocab_size": tok.vocab_size},
        "sources_sha256": sources,
        "counters": counters,
        "splits": split_stats,
        "build_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log(f"[data] wrote {out_dir}/manifest.json  ({manifest['build_seconds']}s)")
    return manifest


def validate_dataset(out_dir: str | Path, tok: AaravTokenizer) -> List[str]:
    """Check shards against the manifest and tokenizer. Returns a list of problems (empty = OK)."""
    out_dir = Path(out_dir)
    problems: List[str] = []
    mpath = out_dir / "manifest.json"
    if not mpath.exists():
        return [f"{mpath} missing"]
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest["tokenizer"]["fingerprint"] != tok.fingerprint:
        problems.append("dataset was tokenised with a different tokenizer than the one loaded")
    for split in SPLITS:
        try:
            shard = Shard.open(out_dir / f"{split}.bin")
        except Exception as e:                          # noqa: BLE001 - report, don't crash
            problems.append(f"{split}: {e}")
            continue
        expect = manifest["splits"][split]["tokens"]
        if shard.n_tokens != expect:
            problems.append(f"{split}: header says {shard.n_tokens} tokens, manifest says {expect}")
        if shard.vocab_size != tok.vocab_size:
            problems.append(f"{split}: shard vocab {shard.vocab_size} != tokenizer vocab {tok.vocab_size}")
        if shard.n_tokens and int(shard.tokens.max()) >= tok.vocab_size:
            problems.append(f"{split}: token id out of range")
        if shard.n_tokens and int((shard.tokens == tok.eot_id).sum()) != shard.n_docs:
            problems.append(f"{split}: EOT count != document count")
    return problems
