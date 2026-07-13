"""Qdrant vector store helpers with hybrid (dense + sparse) support."""

from __future__ import annotations

from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from govgrant.rag.config import Settings, get_settings
from govgrant.rag.index.sparse import encode_docs, encode_query

SPARSE_VECTOR_NAME = "sparse-bm25"


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    settings = settings or get_settings()
    if settings.qdrant_api_key:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(
    settings: Settings | None = None,
    *,
    collection_name: str | None = None,
) -> None:
    """Create collection if missing (dense + sparse vector config)."""
    settings = settings or get_settings()
    client = get_qdrant_client(settings)
    name = collection_name or settings.qdrant_collection
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=settings.embedding_dim,
            distance=models.Distance.COSINE,
        ),
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
            )
        },
    )


def build_vector_store(
    settings: Settings | None = None,
    *,
    ensure: bool = True,
    collection_name: str | None = None,
) -> QdrantVectorStore:
    settings = settings or get_settings()
    name = collection_name or settings.qdrant_collection
    if ensure:
        ensure_collection(settings, collection_name=name)
    client = get_qdrant_client(settings)
    return QdrantVectorStore(
        client=client,
        collection_name=name,
        enable_hybrid=True,
        sparse_doc_fn=encode_docs,
        sparse_query_fn=encode_query,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


def retriever_kwargs(
    *,
    top_k: int = 10,
    sparse_top_k: int | None = None,
    hybrid_top_k: int | None = None,
    filters: MetadataFilters | None = None,
) -> dict:
    """Return kwargs to pass to VectorStoreIndex.as_retriever for hybrid mode."""
    return {
        "vector_store_query_mode": VectorStoreQueryMode.HYBRID,
        "similarity_top_k": top_k,
        "sparse_top_k": sparse_top_k or top_k,
        "hybrid_top_k": hybrid_top_k or top_k,
        "filters": filters,
    }
