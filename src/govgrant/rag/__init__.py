"""LlamaIndex RAG engine for GovGrant AI (R0/R1)."""

from govgrant.rag.contracts import (
    DocumentMeta,
    Modality,
    SourceType,
    build_node_metadata,
)

__all__ = [
    "DocumentMeta",
    "Modality",
    "SourceType",
    "build_node_metadata",
]
