"""Base parser interface (isolated lanes)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from llama_index.core.schema import Document

from govgrant.rag.contracts import DocumentMeta


class BaseModalityParser(ABC):
    """Every modality lane implements this contract."""

    lane: str = "base"

    @abstractmethod
    def parse(self, path: Path, meta: DocumentMeta) -> list[Document]:
        """Parse a file into LlamaIndex Documents with canonical metadata."""
