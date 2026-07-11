"""Qdrant vector store helpers."""

from __future__ import annotations

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from govgrant.rag.config import Settings, get_settings


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
    """Create collection if missing (cosine, nomic 768-dim)."""
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
    )
