"""Sparse vector encoding for Qdrant BM25 replacement.

Uses term-frequency weighting with CRC32 term hashing.
No external model dependency — purely lexical, deterministic.
"""

from __future__ import annotations

import re
import zlib

_CODE_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*|[^\s\w]",
    re.UNICODE,
)


def code_aware_tokenizer(text: str) -> list[str]:
    """Keep codes like SF-424, 2 CFR 200, FOA-XXXX as useful tokens."""
    return [t.lower() for t in _CODE_TOKEN_RE.findall(text or "") if t.strip()]


def _term_hash(term: str) -> int:
    return zlib.crc32(term.encode()) & 0x7FFFFFFF


def _encode_single(text: str) -> tuple[list[int], list[float]]:
    terms = code_aware_tokenizer(text)
    tf: dict[int, float] = {}
    for t in terms:
        tid = _term_hash(t)
        tf[tid] = tf.get(tid, 0.0) + 1.0
    indices = list(tf.keys())
    # Dampened TF: 1 + log(count)
    values = [1.0 + (v - 1.0) * 0.5 for v in tf.values()]
    return indices, values


def encode_docs(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    """Encode document texts as sparse vectors (for Qdrant ingest)."""
    indices: list[list[int]] = []
    values: list[list[float]] = []
    for text in texts:
        idx, val = _encode_single(text)
        indices.append(idx)
        values.append(val)
    return indices, values


def encode_query(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    """Encode query texts as sparse vectors (for Qdrant search)."""
    return encode_docs(texts)
