"""Tools wrapping the LlamaIndex RAG stack for LangGraph (R7)."""

from __future__ import annotations

from typing import Any

from govgrant.rag.config import get_settings
from govgrant.rag.index.hybrid import HybridRAGService
from govgrant.rag.router.query_router import QueryRouter, RouteIntent
from govgrant.rag.sbir.service import SBIRTopicService


class RagToolBelt:
    """
    Thin façade over R1-R5 services.

    Designed to be registered as LangGraph node helpers / @tool callables later
    when a chat LLM is available. Today the graph calls these methods directly.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.docs = HybridRAGService(self.settings)
        self.sbir = SBIRTopicService(self.settings)
        self.router = QueryRouter(self.settings, docs=self.docs, sbir=self.sbir)

    def classify(self, query: str) -> str:
        return self.router.classify(query).value

    def ask(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        doc_id: str | None = None,
        agency: str | None = None,
        intent: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        forced = RouteIntent(intent) if intent else None
        result = self.router.ask(
            query,
            tenant_id=tenant_id or self.settings.default_tenant_id,
            doc_id=doc_id,
            agency=agency,
            top_k=top_k,
            intent=forced,
        )
        return result.to_dict()

    def search_docs(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        doc_id: str | None = None,
        modality: str | None = None,
        top_k: int = 5,
    ) -> str:
        hits = self.docs.retrieve(
            query,
            tenant_id=tenant_id or self.settings.default_tenant_id,
            doc_id=doc_id,
            modality=modality,
            top_k=top_k,
        )
        return self.docs.format_hits(hits)

    def search_sbir(
        self,
        query: str,
        *,
        agency: str | None = None,
        top_k: int = 5,
    ) -> str:
        return self.sbir.search(query, agency=agency, top_k=top_k, include_disclaimer=True)["text"]
