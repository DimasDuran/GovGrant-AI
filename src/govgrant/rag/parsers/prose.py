"""Prose / layout PDF lane via LlamaParse (tables+images embedded as markdown)."""

from __future__ import annotations

import os
from pathlib import Path

from llama_index.core.schema import Document

from govgrant.rag.config import Settings, get_settings
from govgrant.rag.contracts import DocumentMeta, Modality, build_node_metadata
from govgrant.rag.parsers.base import BaseModalityParser


class ProsePDFParser(BaseModalityParser):
    """
    R1 lane: complex PDFs (text, tables, figures as layout markdown).

    Uses LlamaParse so tables/images are not discarded. Dedicated table/figure
    dual stores land in R2/R4; this lane still captures multimodal layout text
    for hybrid retrieval.
    """

    lane = "prose"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def parse(self, path: Path, meta: DocumentMeta) -> list[Document]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if not self.settings.llamaparse_api_key:
            raise RuntimeError(
                "LLAMAPARSE_API_KEY is required for ProsePDFParser. "
                "Set it in .env (see .env.example)."
            )

        # Ensure llama-parse client sees the key
        os.environ.setdefault("LLAMA_CLOUD_API_KEY", self.settings.llamaparse_api_key)

        from llama_parse import LlamaParse

        parser = LlamaParse(
            api_key=self.settings.llamaparse_api_key,
            result_type="markdown",
            verbose=True,
            # Balanced mode: good for tables/layout without premium multimodal cost
            premium_mode=False,
        )

        raw_docs = parser.load_data(str(path))
        out: list[Document] = []
        for i, doc in enumerate(raw_docs):
            page = doc.metadata.get("page_label") or doc.metadata.get("page") or i + 1
            # Keep metadata compact so hierarchical chunking stays healthy
            node_meta = build_node_metadata(
                meta,
                page=page,
                section_path=f"page:{page}",
                modality=Modality.PROSE,
                lane=self.lane,
            )

            text = (doc.text or "").strip()
            if not text:
                continue
            out.append(Document(text=text, metadata=node_meta))
        return out


class LocalPDFFallbackParser(BaseModalityParser):
    """Offline fallback with pypdf when LlamaParse is unavailable."""

    lane = "prose_fallback"

    def parse(self, path: Path, meta: DocumentMeta) -> list[Document]:
        from pypdf import PdfReader

        path = Path(path)
        reader = PdfReader(str(path))
        out: list[Document] = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            node_meta = build_node_metadata(
                meta,
                page=i,
                section_path=f"page:{i}",
                modality=Modality.PROSE,
                lane=self.lane,
                parser_name="pypdf",
                parse_confidence=0.6,
            )
            out.append(Document(text=text, metadata=node_meta))
        return out
