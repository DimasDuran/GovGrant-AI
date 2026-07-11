"""LlamaIndex RAG engine for GovGrant AI (R0/R1)."""

from govgrant.rag.contracts import (
    Modality,
    SourceType,
    DocumentMeta,
    build_node_metadata,
)

__all__ = [
    "Modality",
    "SourceType",
    "DocumentMeta",
    "build_node_metadata",
]
