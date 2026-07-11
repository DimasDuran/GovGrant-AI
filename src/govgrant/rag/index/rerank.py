"""Lightweight lexical re-ranker (R6) — no external API required."""

from __future__ import annotations

import re
from collections import Counter

from llama_index.core.schema import NodeWithScore


_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-./]*", re.I)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if len(t) > 1]


def lexical_rerank(
    query: str,
    hits: list[NodeWithScore],
    *,
    top_k: int | None = None,
) -> list[NodeWithScore]:
    """
    Re-score hits by query-term overlap (BM25-ish boost on exact codes).

    Combines original score with normalized term overlap. Useful when a remote
    re-ranker (Cohere/BGE) is not configured yet.
    """
    if not hits:
        return []
    q_tokens = _tokens(query)
    if not q_tokens:
        return hits[: top_k or len(hits)]

    q_set = set(q_tokens)
    scored: list[NodeWithScore] = []
    for h in hits:
        text = h.node.get_content() or ""
        t_tokens = _tokens(text)
        if not t_tokens:
            overlap = 0.0
        else:
            counts = Counter(t_tokens)
            # exact code / rare token boost
            overlap = sum(1.0 + (0.5 if "-" in tok or tok.isupper() else 0.0)
                          for tok in q_set if tok in counts)
            overlap = overlap / max(len(q_set), 1)
        base = float(h.score or 0.0)
        new_score = 0.55 * base + 0.45 * overlap
        scored.append(NodeWithScore(node=h.node, score=new_score))

    scored.sort(key=lambda x: x.score or 0.0, reverse=True)
    return scored[: top_k or len(scored)]
