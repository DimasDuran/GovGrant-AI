"""SBIR topic sync + hybrid search service (R3)."""

from __future__ import annotations

import json
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core import Settings as LISettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.retrievers.bm25 import BM25Retriever

from govgrant.rag.config import Settings, get_settings
from govgrant.rag.index.embeddings import build_embed_model
from govgrant.rag.index.hybrid import code_aware_tokenizer
from govgrant.rag.index.qdrant_store import build_vector_store
from govgrant.rag.sbir.client import SBIRAPIError, SBIRTopicClient
from govgrant.rag.sbir.disclaimer import with_disclaimer
from govgrant.rag.sbir.models import TopicDocument
from govgrant.rag.sbir.normalizer import load_fixture_json, normalize_solicitations
from govgrant.rag.sbir.store import SBIRStructuredStore


class SBIRTopicService:
    """
    Sync open SBIR topics from API (or fixtures) into:
      - SQLite structured store
      - Qdrant collection sbir_topics + BM25 hybrid retrieve
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embed_model = build_embed_model(self.settings)
        LISettings.embed_model = self.embed_model

        self.store = SBIRStructuredStore(self.settings.sbir_db_path)
        self.vector_store = build_vector_store(
            self.settings,
            ensure=True,
            collection_name=self.settings.sbir_qdrant_collection,
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self._nodes: list[BaseNode] = []
        self._load_bm25()

    # ------------------------------------------------------------------- sync
    def sync(
        self,
        *,
        keyword: str | None = None,
        agency: str | None = None,
        force_fixtures: bool = False,
    ) -> dict[str, Any]:
        """
        Pull open solicitations, normalize to topics, upsert store + reindex RAG.
        Falls back to fixtures if API fails and SBIR_USE_FIXTURES_ON_FAIL=true.
        """
        source = "api"
        stale = False
        raw: list[dict[str, Any]] = []
        error: str | None = None

        if force_fixtures:
            raw = load_fixture_json(self.settings.sbir_fixture_path)
            source = "fixture"
            stale = True
        else:
            try:
                client = SBIRTopicClient(self.settings)
                raw = client.fetch_all_open(keyword=keyword, agency=agency)
                source = "api"
                stale = False
            except SBIRAPIError as exc:
                error = str(exc)
                if self.settings.sbir_use_fixtures_on_fail:
                    raw = load_fixture_json(self.settings.sbir_fixture_path)
                    source = "fixture"
                    stale = True
                else:
                    raise

        topics = normalize_solicitations(raw, source=source, stale=stale)
        if agency:
            topics = [
                t
                for t in topics
                if (t.agency or "").upper() == agency.upper()
            ]

        n_store = self.store.upsert_topics(topics)
        n_index = self._index_topics(topics)

        now = datetime.now(timezone.utc).isoformat()
        self.store.set_meta("last_sync_at", now)
        self.store.set_meta("last_source", source)
        self.store.set_meta("last_error", error or "")
        self.store.set_meta(
            "last_sync_summary",
            json.dumps(
                {
                    "topics": n_store,
                    "indexed": n_index,
                    "source": source,
                    "stale": stale,
                    "agency": agency,
                    "keyword": keyword,
                }
            ),
        )

        return {
            "topics": n_store,
            "indexed": n_index,
            "source": source,
            "stale": stale,
            "error": error,
            "collection": self.settings.sbir_qdrant_collection,
            "last_sync_at": now,
            "sample_topic_ids": [t.topic_id for t in topics[:5]],
        }

    def _index_topics(self, topics: list[TopicDocument]) -> int:
        if not topics:
            return 0
        nodes: list[BaseNode] = []
        for t in topics:
            # Qdrant point IDs must be UUID or unsigned int — not free-form strings
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"sbir-topic:{t.topic_id}"))
            node = TextNode(
                text=t.to_rag_text(),
                metadata=t.to_metadata(),
                id_=point_id,
            )
            # content-only embeddings
            node.excluded_embed_metadata_keys = list(node.metadata.keys())
            node.excluded_llm_metadata_keys = [
                k
                for k in node.metadata
                if k
                not in {
                    "topic_id",
                    "topic_title",
                    "agency",
                    "status",
                    "citation_uri",
                    "program",
                    "phase",
                }
            ]
            nodes.append(node)

        # Replace entire BM25 corpus for SBIR (small public set for open topics)
        # Merge by topic_id
        by_id = {
            n.metadata.get("topic_id"): n
            for n in self._nodes
            if n.metadata.get("topic_id")
        }
        for n in nodes:
            by_id[n.metadata["topic_id"]] = n
        self._nodes = list(by_id.values())
        self._persist_bm25()

        VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )
        return len(nodes)

    # ------------------------------------------------------------------ query
    def search(
        self,
        query: str,
        *,
        agency: str | None = None,
        status: str | None = "open",
        program: str | None = None,
        top_k: int = 6,
        include_disclaimer: bool = True,
    ) -> dict[str, Any]:
        filters = self._filters(agency=agency, status=status, program=program)
        vector_index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=self.embed_model,
        )
        vector_hits = list(
            vector_index.as_retriever(
                similarity_top_k=max(top_k, 8),
                filters=filters if filters.filters else None,
            ).retrieve(query)
        )

        bm25_nodes = self._filter_nodes(agency=agency, status=status, program=program)
        if bm25_nodes:
            bm25_hits = list(
                BM25Retriever.from_defaults(
                    nodes=bm25_nodes,
                    similarity_top_k=max(top_k, 8),
                    tokenizer=code_aware_tokenizer,
                ).retrieve(query)
            )
        else:
            bm25_hits = []

        fused = self._rrf([vector_hits, bm25_hits], top_k=top_k)
        formatted = self.format_hits(fused)
        topic_ids = []
        for h in fused:
            tid = (h.node.metadata or {}).get("topic_id")
            if tid:
                topic_ids.append(str(tid))

        stale_any = any((h.node.metadata or {}).get("stale") for h in fused)
        body = formatted
        if include_disclaimer:
            body = with_disclaimer(formatted, topic_ids=topic_ids)

        return {
            "hits": fused,
            "text": body,
            "topic_ids": topic_ids,
            "stale": stale_any,
            "last_sync_at": self.store.get_meta("last_sync_at"),
            "source": self.store.get_meta("last_source"),
        }

    def get_topic(self, topic_id: str, *, include_disclaimer: bool = True) -> dict[str, Any]:
        doc = self.store.get(topic_id)
        if not doc:
            return {"found": False, "topic_id": topic_id, "text": f"Topic {topic_id} not found in local store."}
        text = doc.to_rag_text() + f"\n\nAgency URL: {doc.solicitation_agency_url or 'n/a'}"
        if include_disclaimer:
            text = with_disclaimer(text, topic_ids=[doc.topic_id])
        return {
            "found": True,
            "topic": doc.model_dump(),
            "text": text,
            "stale": doc.stale,
        }

    def format_hits(self, hits: list[NodeWithScore]) -> str:
        if not hits:
            return "(no SBIR topic hits)"
        blocks = []
        for i, h in enumerate(hits, start=1):
            md = h.node.metadata or {}
            score = h.score if h.score is not None else 0.0
            text = h.node.get_content().strip().replace("\n", " ")
            if len(text) > 450:
                text = text[:450] + "…"
            blocks.append(
                f"[{i}] score={score:.4f} | topic_id={md.get('topic_id')} | "
                f"agency={md.get('agency')} | status={md.get('status')} | "
                f"program={md.get('program')}\n"
                f"    title: {md.get('topic_title')}\n"
                f"    cite: {md.get('citation_uri')}\n"
                f"    {text}"
            )
        return "\n\n".join(blocks)

    # ---------------------------------------------------------------- helpers
    def _filters(
        self,
        *,
        agency: str | None,
        status: str | None,
        program: str | None,
    ) -> MetadataFilters:
        filters: list[MetadataFilter] = [
            MetadataFilter(
                key="source_type",
                value="sbir_topic",
                operator=FilterOperator.EQ,
            )
        ]
        if agency:
            filters.append(
                MetadataFilter(
                    key="agency", value=agency.upper(), operator=FilterOperator.EQ
                )
            )
        if status:
            filters.append(
                MetadataFilter(
                    key="status", value=status.lower(), operator=FilterOperator.EQ
                )
            )
        if program:
            filters.append(
                MetadataFilter(
                    key="program", value=program.upper(), operator=FilterOperator.EQ
                )
            )
        return MetadataFilters(filters=filters)

    def _filter_nodes(
        self,
        *,
        agency: str | None,
        status: str | None,
        program: str | None,
    ) -> list[BaseNode]:
        out = []
        for n in self._nodes:
            md = n.metadata or {}
            if agency and (md.get("agency") or "").upper() != agency.upper():
                continue
            if status and (md.get("status") or "").lower() != status.lower():
                continue
            if program and (md.get("program") or "").upper() != program.upper():
                continue
            out.append(n)
        return out

    @staticmethod
    def _rrf(
        result_lists: list[list[NodeWithScore]],
        *,
        top_k: int,
        k: int = 60,
    ) -> list[NodeWithScore]:
        scores: dict[str, float] = {}
        nodes: dict[str, BaseNode] = {}
        for results in result_lists:
            for rank, item in enumerate(results):
                nid = item.node.node_id
                scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
                nodes[nid] = item.node
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [NodeWithScore(node=nodes[i], score=s) for i, s in ranked]

    def _bm25_path(self) -> Path:
        return self.settings.sbir_bm25_dir / "sbir_nodes.pkl"

    def _persist_bm25(self) -> None:
        path = self._bm25_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "id_": n.node_id,
                "text": n.get_content(),
                "metadata": dict(n.metadata or {}),
            }
            for n in self._nodes
        ]
        with path.open("wb") as f:
            pickle.dump(payload, f)
        path.with_suffix(".json").write_text(
            json.dumps({"count": len(payload)}, indent=2), encoding="utf-8"
        )

    def _load_bm25(self) -> None:
        path = self._bm25_path()
        if not path.exists():
            self._nodes = []
            return
        with path.open("rb") as f:
            raw = pickle.load(f)
        self._nodes = [
            TextNode(id_=i["id_"], text=i["text"], metadata=i.get("metadata") or {})
            for i in raw
        ]
