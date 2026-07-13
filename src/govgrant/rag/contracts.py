"""Canonical contracts for RAG nodes and metadata (R0)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    USER_DOC = "user_doc"
    SBIR_TOPIC = "sbir_topic"
    POLICY = "policy"


class Modality(StrEnum):
    PROSE = "prose"
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    FORM = "form"


class DocumentMeta(BaseModel):
    """Document-level metadata applied to every node from a file."""

    source_type: SourceType = SourceType.USER_DOC
    modality: Modality = Modality.PROSE
    tenant_id: str
    doc_id: str
    version: str = "v1"
    file_name: str
    citation_uri: str
    parser_name: str = "llamaparse"
    lane: str = "prose"
    agency: str | None = None
    topic_id: str | None = None
    status: str | None = None
    parse_confidence: float = 1.0
    stale: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Qdrant / LlamaIndex metadata payload (flat, filterable)."""
        # gg_doc_id is the stable app-level document key. LlamaIndex may overwrite
        # payload field "doc_id" with its internal ref_doc_id UUID — always filter
        # on gg_doc_id for tenant/doc scoping.
        payload: dict[str, Any] = {
            "source_type": self.source_type.value,
            "modality": self.modality.value,
            "tenant_id": self.tenant_id,
            "doc_id": self.doc_id,
            "gg_doc_id": self.doc_id,
            "version": self.version,
            "file_name": self.file_name,
            "citation_uri": self.citation_uri,
            "parser_name": self.parser_name,
            "lane": self.lane,
            "parse_confidence": self.parse_confidence,
            "stale": self.stale,
        }
        if self.agency:
            payload["agency"] = self.agency
        if self.topic_id:
            payload["topic_id"] = self.topic_id
        if self.status:
            payload["status"] = self.status
        for key, value in self.extra.items():
            if value is not None and key not in payload:
                payload[key] = value
        return payload


def build_node_metadata(
    doc: DocumentMeta,
    *,
    page: int | str | None = None,
    section_path: str | None = None,
    parent_id: str | None = None,
    modality: Modality | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Merge document meta with node-level fields for a single chunk."""
    meta = doc.to_payload()
    if page is not None:
        meta["page"] = page
    if section_path is not None:
        meta["section_path"] = section_path
    if parent_id is not None:
        meta["parent_id"] = parent_id
    if modality is not None:
        meta["modality"] = modality.value
    meta.update({k: v for k, v in overrides.items() if v is not None})
    return meta
