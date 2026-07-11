"""Figure / chart lane (R4): extract embedded images + captions for hybrid RAG."""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.schema import Document

from govgrant.rag.config import Settings, get_settings
from govgrant.rag.contracts import DocumentMeta, Modality, build_node_metadata
from govgrant.rag.parsers.base import BaseModalityParser

# Markdown image / figure cues from LlamaParse and narrative text
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_FIGURE_CAPTION = re.compile(
    r"(?im)^\s*(?:figure|fig\.|chart|graph|plot|exhibit|image)\s*[\d.IVX]*\s*[:.\-–]?\s*(.+)$"
)
_CHART_HINTS = re.compile(
    r"\b(chart|graph|plot|histogram|bar chart|line chart|scatter|axis|series|trend)\b",
    re.I,
)


@dataclass
class ExtractedFigure:
    figure_id: str
    page: int | str | None
    caption: str
    ocr_text: str = ""
    image_path: str | None = None
    modality: Modality = Modality.FIGURE
    parse_confidence: float = 0.5
    source: str = "markdown"  # markdown | embedded | vision
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_rag_text(self) -> str:
        kind = "CHART" if self.modality == Modality.CHART else "FIGURE"
        parts = [
            f"{kind} id={self.figure_id}",
            f"page={self.page}",
            f"caption: {self.caption or '(no caption)'}",
        ]
        if self.ocr_text:
            parts.append(f"ocr_text: {self.ocr_text[:1500]}")
        if self.image_path:
            parts.append(f"image_path: {self.image_path}")
        parts.append(f"parse_confidence: {self.parse_confidence:.2f}")
        return "\n".join(parts)


def _guess_modality(caption: str, ocr: str = "") -> Modality:
    blob = f"{caption} {ocr}"
    if _CHART_HINTS.search(blob):
        return Modality.CHART
    return Modality.FIGURE


def extract_figures_from_markdown(
    prose_docs: list[Document],
    *,
    doc_id: str,
) -> list[ExtractedFigure]:
    """Pull figure/chart cues and markdown images from parsed page text."""
    figures: list[ExtractedFigure] = []
    idx = 0
    for doc in prose_docs:
        page = doc.metadata.get("page")
        text = doc.text or ""
        # Explicit markdown images
        for m in _MD_IMAGE.finditer(text):
            alt, src = m.group(1).strip(), m.group(2).strip()
            caption = alt or f"Embedded image reference ({src[:80]})"
            modality = _guess_modality(caption)
            figures.append(
                ExtractedFigure(
                    figure_id=f"{doc_id}::p{page}::f{idx}",
                    page=page,
                    caption=caption,
                    ocr_text="",
                    image_path=src if not src.startswith("http") else None,
                    modality=modality,
                    parse_confidence=0.55 if alt else 0.4,
                    source="markdown",
                )
            )
            idx += 1
        # Caption lines
        for m in _FIGURE_CAPTION.finditer(text):
            caption = m.group(1).strip()
            if len(caption) < 3:
                continue
            modality = _guess_modality(caption)
            figures.append(
                ExtractedFigure(
                    figure_id=f"{doc_id}::p{page}::f{idx}",
                    page=page,
                    caption=caption,
                    modality=modality,
                    parse_confidence=0.65,
                    source="markdown",
                )
            )
            idx += 1
    return figures


def extract_embedded_images(
    pdf_path: Path,
    *,
    doc_id: str,
    out_dir: Path,
    min_bytes: int = 4_000,
    max_images: int = 40,
) -> list[ExtractedFigure]:
    """
    Extract embedded raster images from PDF via PyMuPDF (if installed).
    Falls back to empty list if pymupdf is unavailable.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[figures] pymupdf not installed — skipping embedded image extract")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[ExtractedFigure] = []
    doc = fitz.open(str(pdf_path))
    count = 0
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            for img_i, img in enumerate(page.get_images(full=True)):
                if count >= max_images:
                    break
                xref = img[0]
                try:
                    base = doc.extract_image(xref)
                except Exception:  # noqa: BLE001
                    continue
                image_bytes = base.get("image") or b""
                if len(image_bytes) < min_bytes:
                    continue
                ext = base.get("ext") or "png"
                digest = hashlib.sha1(image_bytes).hexdigest()[:10]
                fname = f"{doc_id}_p{page_num}_{img_i}_{digest}.{ext}"
                path = out_dir / fname
                path.write_bytes(image_bytes)
                figures.append(
                    ExtractedFigure(
                        figure_id=f"{doc_id}::p{page_num}::img{img_i}",
                        page=page_num,
                        caption=f"Embedded image extracted from page {page_num}",
                        image_path=str(path.resolve()),
                        modality=Modality.FIGURE,
                        parse_confidence=0.45,
                        source="embedded",
                    )
                )
                count += 1
            if count >= max_images:
                break
    finally:
        doc.close()
    return figures


def caption_with_ollama_vision(
    image_path: Path,
    *,
    settings: Settings | None = None,
) -> tuple[str, float]:
    """
    Optional local vision caption via Ollama (llava/moondream/etc).
    Returns (caption, confidence). Empty caption if model unavailable.
    """
    settings = settings or get_settings()
    model = settings.ollama_vision_model
    if not model:
        return "", 0.0
    try:
        import httpx
    except ImportError:
        return "", 0.0

    path = Path(image_path)
    if not path.exists():
        return "", 0.0
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    prompt = (
        "Describe this figure from a federal grant / SBIR technical document. "
        "If it is a chart or plot, list axes, series, units, and readable values. "
        "If it is a diagram, summarize components and relationships. "
        "Be concise (max 120 words). Do not invent numbers you cannot read."
    )
    try:
        with httpx.Client(timeout=settings.ollama_vision_timeout) as client:
            resp = client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [b64],
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()
            if not text:
                return "", 0.0
            return text[:2000], 0.75
    except Exception as exc:  # noqa: BLE001
        print(f"[figures] vision caption failed ({exc})")
        return "", 0.0


def enrich_with_vision(
    figures: list[ExtractedFigure],
    *,
    settings: Settings | None = None,
    max_vision: int = 15,
) -> list[ExtractedFigure]:
    """Apply Ollama vision captions to figures that have local image paths."""
    settings = settings or get_settings()
    if not settings.ollama_vision_model:
        return figures
    n = 0
    for fig in figures:
        if n >= max_vision:
            break
        if not fig.image_path:
            continue
        path = Path(fig.image_path)
        if not path.exists():
            continue
        caption, conf = caption_with_ollama_vision(path, settings=settings)
        if caption:
            fig.ocr_text = caption
            if conf > fig.parse_confidence:
                fig.parse_confidence = conf
            # Prefer vision text as richer caption when generic
            if fig.caption.startswith("Embedded image") or len(fig.caption) < 20:
                fig.caption = caption[:300]
            fig.modality = _guess_modality(fig.caption, fig.ocr_text)
            fig.source = "vision"
            n += 1
    return figures


class FigureChartParser(BaseModalityParser):
    """
    R4 lane: markdown figure cues + embedded images (+ optional vision captions).
    """

    lane = "figure"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def parse(self, path: Path, meta: DocumentMeta) -> list[Document]:
        raise NotImplementedError("Use extract() with prose docs + pdf path")

    def extract(
        self,
        pdf_path: Path,
        prose_docs: list[Document],
        meta: DocumentMeta,
        *,
        extract_embedded: bool = True,
        use_vision: bool = True,
    ) -> tuple[list[Document], list[ExtractedFigure]]:
        figures = extract_figures_from_markdown(prose_docs, doc_id=meta.doc_id)

        if extract_embedded:
            img_dir = self.settings.figures_dir / meta.doc_id
            embedded = extract_embedded_images(
                Path(pdf_path),
                doc_id=meta.doc_id,
                out_dir=img_dir,
                max_images=self.settings.figures_max_per_doc,
            )
            # Prefer embedded images; keep markdown captions that add value
            figures.extend(embedded)

        # Deduplicate by figure_id
        by_id: dict[str, ExtractedFigure] = {}
        for f in figures:
            by_id[f.figure_id] = f
        figures = list(by_id.values())

        if use_vision:
            figures = enrich_with_vision(figures, settings=self.settings)

        rag_docs: list[Document] = []
        for fig in figures:
            node_meta = build_node_metadata(
                meta,
                page=fig.page,
                section_path=f"page:{fig.page}/figure:{fig.figure_id.split('::')[-1]}",
                modality=fig.modality,
                lane=self.lane,
                figure_id=fig.figure_id,
                parse_confidence=fig.parse_confidence,
                image_path=fig.image_path or "",
                figure_source=fig.source,
            )
            node_meta["gg_doc_id"] = meta.doc_id
            rag_docs.append(Document(text=fig.to_rag_text(), metadata=node_meta))
        return rag_docs, figures
