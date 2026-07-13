"""
R5 multi-source query router.

Routes a user question to one or more backends:
  - user docs hybrid RAG (prose)
  - tables (RAG modality=table and/or structured SQLite)
  - figures/charts
  - SBIR topics (public hybrid index)
  - cross_check: docs + SBIR in parallel
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from govgrant.rag.config import Settings, get_settings
from govgrant.rag.index.hybrid import HybridRAGService
from govgrant.rag.sbir.disclaimer import with_disclaimer
from govgrant.rag.sbir.service import SBIRTopicService


class RouteIntent(StrEnum):
    DOC_QA = "doc_qa"
    TABLE = "table"
    FIGURE = "figure"
    TOPIC_SEARCH = "topic_search"
    CROSS_CHECK = "cross_check"
    MIXED = "mixed"


_TOPIC_CUES = re.compile(
    r"\b(sbir\.gov|open topics?|funding opportunit|solicitation|topic id|"
    r"which agency|topics? (?:open|available|match)|FOA|BAA)\b",
    re.I,
)
_TABLE_CUES = re.compile(
    r"\b(table|row|column|budget (?:table|line|form)|indirect cost|"
    r"spreadsheet|cell values?|tabular)\b",
    re.I,
)
_FIGURE_CUES = re.compile(
    r"\b(figure|fig\.|chart|graph|plot|diagram|image|screenshot|seal|logo)\b",
    re.I,
)
_CROSS_CUES = re.compile(
    r"\b(does my|align(?:s|ed|ment)? with|cross[- ]?check|eligible for topic|"
    r"fit(?:s)? (?:with|my)|compare (?:my|our)|"
    r"my (?:proposal|abstract|draft).*(?:topic|solicitation|sbir)|"
    r"(?:topic|solicitation).*(?:my|proposal|abstract|draft))\b",
    re.I,
)
_AGENCY = re.compile(r"\b(DOD|DoD|NIH|NSF|NASA|DOE|HHS|MDA|DARPA|USDA|EPA|DOT|DHS)\b")
# Map common branch / component names to parent agency codes used in SBIR index
_AGENCY_ALIASES = {
    "MDA": "DOD",
    "DARPA": "DOD",
    "SOCOM": "DOD",
    "ARMY": "DOD",
    "NAVY": "DOD",
    "AF": "DOD",
    "USAF": "DOD",
    "CDC": "HHS",
    "FDA": "HHS",
}


@dataclass
class RouteResult:
    intent: RouteIntent
    text: str
    sources_used: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "sources_used": self.sources_used,
            "meta": self.meta,
            "text": self.text,
        }


class QueryRouter:
    """Heuristic multi-source router (LLM classifier can replace classify later)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        docs: HybridRAGService | None = None,
        sbir: SBIRTopicService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.docs = docs or HybridRAGService(self.settings)
        self.sbir = sbir or SBIRTopicService(self.settings)

    def classify(self, query: str) -> RouteIntent:
        q = query.strip()
        if _CROSS_CUES.search(q):
            return RouteIntent.CROSS_CHECK
        topic = bool(_TOPIC_CUES.search(q))
        table = bool(_TABLE_CUES.search(q))
        figure = bool(_FIGURE_CUES.search(q))
        # SBIR-only phrasing
        if (
            topic
            and not table
            and not figure
            and not re.search(
                r"\b(my |our |proposal|application guide|policy directive)\b", q, re.I
            )
        ):
            return RouteIntent.TOPIC_SEARCH
        if table and not topic:
            return RouteIntent.TABLE
        if figure and not topic:
            return RouteIntent.FIGURE
        if topic and (table or figure or re.search(r"\b(my |proposal)\b", q, re.I)):
            return RouteIntent.CROSS_CHECK
        if sum([topic, table, figure]) >= 2:
            return RouteIntent.MIXED
        return RouteIntent.DOC_QA

    def ask(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        doc_id: str | None = None,
        agency: str | None = None,
        top_k: int = 5,
        intent: RouteIntent | None = None,
    ) -> RouteResult:
        tenant_id = tenant_id or self.settings.default_tenant_id
        intent = intent or self.classify(query)
        if not agency:
            m = _AGENCY.search(query)
            if m:
                agency = m.group(1).upper()
        if agency:
            agency = _AGENCY_ALIASES.get(agency.upper(), agency.upper())

        if intent == RouteIntent.TOPIC_SEARCH:
            return self._route_sbir(query, agency=agency, top_k=top_k)
        if intent == RouteIntent.TABLE:
            return self._route_table(query, tenant_id=tenant_id, doc_id=doc_id, top_k=top_k)
        if intent == RouteIntent.FIGURE:
            return self._route_figure(query, tenant_id=tenant_id, doc_id=doc_id, top_k=top_k)
        if intent == RouteIntent.CROSS_CHECK:
            return self._route_cross(
                query,
                tenant_id=tenant_id,
                doc_id=doc_id,
                agency=agency,
                top_k=top_k,
            )
        if intent == RouteIntent.MIXED:
            return self._route_mixed(
                query,
                tenant_id=tenant_id,
                doc_id=doc_id,
                agency=agency,
                top_k=top_k,
            )
        return self._route_docs(query, tenant_id=tenant_id, doc_id=doc_id, top_k=top_k)

    # ---------------------------------------------------------------- routes
    def _route_docs(
        self,
        query: str,
        *,
        tenant_id: str,
        doc_id: str | None,
        top_k: int,
        modality: str | None = None,
    ) -> RouteResult:
        # Broader pack for long multi-part compliance questions
        k = max(top_k, 10) if len(query) > 280 else top_k
        hits = self.docs.retrieve(
            query,
            tenant_id=tenant_id,
            doc_id=doc_id,
            modality=modality,
            top_k=k,
        )
        text = self.docs.format_hits(hits, for_llm=True)
        if not hits:
            text = self._insufficient("user documents", query)
        if modality in {"figure", "chart"}:
            intent_out = RouteIntent.FIGURE
        elif modality == "table":
            intent_out = RouteIntent.TABLE
        else:
            intent_out = RouteIntent.DOC_QA
        return RouteResult(
            intent=intent_out,
            text=text,
            sources_used=["user_docs"],
            meta={"n_hits": len(hits), "modality": modality, "doc_id": doc_id},
        )

    def _route_table(
        self,
        query: str,
        *,
        tenant_id: str,
        doc_id: str | None,
        top_k: int,
    ) -> RouteResult:
        rag = self.docs.retrieve(
            query,
            tenant_id=tenant_id,
            doc_id=doc_id,
            modality="table",
            top_k=top_k,
        )
        structured = self.docs.search_tables(query, tenant_id=tenant_id, doc_id=doc_id, limit=top_k)
        parts = ["## Table RAG hits", self.docs.format_hits(rag)]
        parts += ["", "## Structured cell hits", self.docs.format_table_hits(structured)]
        if not rag and not structured:
            body = self._insufficient("tables", query)
        else:
            body = "\n".join(parts)
        return RouteResult(
            intent=RouteIntent.TABLE,
            text=body,
            sources_used=["user_docs:table", "tabular_sqlite"],
            meta={
                "n_rag": len(rag),
                "n_structured": len(structured),
                "doc_id": doc_id,
            },
        )

    def _route_figure(
        self,
        query: str,
        *,
        tenant_id: str,
        doc_id: str | None,
        top_k: int,
    ) -> RouteResult:
        # Try figure then chart
        figs = self.docs.retrieve(
            query,
            tenant_id=tenant_id,
            doc_id=doc_id,
            modality="figure",
            top_k=top_k,
        )
        charts = self.docs.retrieve(
            query,
            tenant_id=tenant_id,
            doc_id=doc_id,
            modality="chart",
            top_k=max(2, top_k // 2),
        )
        # Merge unique by node id
        seen: set[str] = set()
        merged = []
        for h in list(figs) + list(charts):
            if h.node.node_id in seen:
                continue
            seen.add(h.node.node_id)
            merged.append(h)
        merged = merged[:top_k]
        text = self.docs.format_hits(merged)
        if not merged:
            text = self._insufficient("figures/charts", query)
        return RouteResult(
            intent=RouteIntent.FIGURE,
            text=text,
            sources_used=["user_docs:figure", "user_docs:chart"],
            meta={"n_hits": len(merged), "doc_id": doc_id},
        )

    def _route_sbir(
        self,
        query: str,
        *,
        agency: str | None,
        top_k: int,
    ) -> RouteResult:
        result = self.sbir.search(query, agency=agency, top_k=top_k, include_disclaimer=True)
        text = result["text"]
        if not result["topic_ids"]:
            text = with_disclaimer(
                self._insufficient("SBIR topics", query),
                topic_ids=[],
            )
        return RouteResult(
            intent=RouteIntent.TOPIC_SEARCH,
            text=text,
            sources_used=["sbir_topics"],
            meta={
                "topic_ids": result["topic_ids"],
                "stale": result["stale"],
                "source": result["source"],
                "agency": agency,
            },
        )

    def _route_cross(
        self,
        query: str,
        *,
        tenant_id: str,
        doc_id: str | None,
        agency: str | None,
        top_k: int,
    ) -> RouteResult:
        docs = self.docs.retrieve(query, tenant_id=tenant_id, doc_id=doc_id, top_k=top_k)
        # Strip proposal-framing so hybrid topic search focuses on tech keywords
        sbir_query = re.sub(
            r"(?i)\b(does|do|can|will)?\s*(my|our)\s+(proposal|abstract|draft|application)\b",
            " ",
            query,
        )
        sbir_query = re.sub(
            r"(?i)\b(align(?:s|ed)? with|cross[- ]?check|eligible for|compare with)\b",
            " ",
            sbir_query,
        )
        sbir_query = re.sub(r"\s+", " ", sbir_query).strip() or query
        sbir = self.sbir.search(sbir_query, agency=agency, top_k=top_k, include_disclaimer=False)
        parts = [
            "## A) User document evidence",
            self.docs.format_hits(docs) if docs else self._insufficient("user documents", query),
            "",
            "## B) SBIR topic evidence",
            sbir["text"] if sbir["topic_ids"] else self._insufficient("SBIR topics", query),
            "",
            "## Cross-check note",
            "Compare section A (your corpus) with section B (official topics). "
            "Do not claim eligibility unless both sides provide explicit supporting text.",
        ]
        body = "\n".join(parts)
        body = with_disclaimer(body, topic_ids=sbir.get("topic_ids") or [])
        return RouteResult(
            intent=RouteIntent.CROSS_CHECK,
            text=body,
            sources_used=["user_docs", "sbir_topics"],
            meta={
                "n_doc_hits": len(docs),
                "topic_ids": sbir.get("topic_ids"),
                "stale": sbir.get("stale"),
                "agency": agency,
            },
        )

    def _route_mixed(
        self,
        query: str,
        *,
        tenant_id: str,
        doc_id: str | None,
        agency: str | None,
        top_k: int,
    ) -> RouteResult:
        # Lightweight fan-out: docs + tables + sbir snippets
        docs = self._route_docs(query, tenant_id=tenant_id, doc_id=doc_id, top_k=max(3, top_k // 2))
        tables = self._route_table(query, tenant_id=tenant_id, doc_id=doc_id, top_k=3)
        sbir = self._route_sbir(query, agency=agency, top_k=3)
        body = "\n\n".join(
            [
                "# Intent: mixed\n",
                "## Documents\n" + docs.text,
                "## Tables\n" + tables.text,
                "## SBIR topics\n" + sbir.text,
            ]
        )
        return RouteResult(
            intent=RouteIntent.MIXED,
            text=body,
            sources_used=list(
                dict.fromkeys(docs.sources_used + tables.sources_used + sbir.sources_used)
            ),
            meta={"parts": ["docs", "tables", "sbir"]},
        )

    @staticmethod
    def _insufficient(source: str, query: str) -> str:
        return (
            f"[insufficient evidence] No strong hits in {source} for: {query!r}. "
            "Refuse to invent facts; refine the query or ingest more sources."
        )
