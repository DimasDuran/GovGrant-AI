"""Local embeddings via Ollama (nomic-embed-text)."""

from __future__ import annotations

import re
import time
from typing import Any

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from govgrant.rag.config import Settings, get_settings

# TOC leader dots / long punctuation runs crash some nomic+ollama builds
_DOT_LEADERS = re.compile(r"[.\u2024\u2026\u00b7]{5,}")
_WS = re.compile(r"[ \t]{2,}")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize_for_embed(text: str, max_chars: int = 1500) -> str:
    """Normalize text so local nomic-embed-text stays stable."""
    text = (text or "").replace("\x00", " ")
    text = _CTRL.sub(" ", text)
    text = _DOT_LEADERS.sub(" ", text)
    text = _WS.sub(" ", text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text or " "


class RobustOllamaEmbedding(BaseEmbedding):
    """
    Ollama embeddings with:
    - single-text requests (avoids batch EOF crashes on some Ollama builds)
    - retries with backoff
    - sanitization for TOC leader-dots that trigger nomic 500s
    """

    model_name: str = Field(default="nomic-embed-text")
    base_url: str = Field(default="http://localhost:11434")
    request_timeout: float = Field(default=180.0)
    max_chars: int = Field(default=1500)
    max_retries: int = Field(default=4)
    embed_batch_size: int = Field(default=1)

    _client: httpx.Client = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            timeout=self.request_timeout,
        )

    def _prepare(self, text: str) -> str:
        return sanitize_for_embed(text, max_chars=self.max_chars)

    def _embed_one(self, text: str) -> list[float]:
        prompt = self._prepare(text)
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(
                    "/api/embeddings",
                    json={"model": self.model_name, "prompt": prompt},
                )
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding")
                if not emb:
                    raise RuntimeError(f"Empty embedding response: {data}")
                return list(emb)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                # Progressive shorten on server errors (pathological tokens)
                if attempt >= 2:
                    prompt = prompt[: max(200, len(prompt) // 2)]
                time.sleep(min(2 ** attempt * 0.2, 3.0))
        raise RuntimeError(f"Ollama embed failed after retries: {last_err}")

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed_one(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


def build_embed_model(settings: Settings | None = None) -> RobustOllamaEmbedding:
    settings = settings or get_settings()
    return RobustOllamaEmbedding(
        model_name=settings.embedding_model,
        base_url=settings.ollama_base_url,
        request_timeout=180.0,
        max_chars=2000,
        embed_batch_size=1,
    )
