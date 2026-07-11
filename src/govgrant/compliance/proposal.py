"""Extract text from a user proposal PDF for draft scoring / light ingest."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ProposalExtract:
    path: str
    file_name: str
    pages: int
    chars: int
    text: str
    parser: str
    page_previews: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_markdown(self) -> str:
        return (
            f"**File:** `{self.file_name}` · **pages:** {self.pages} · "
            f"**chars:** {self.chars:,} · **parser:** `{self.parser}`"
        )


def extract_proposal_text(path: Path | str, *, max_chars: int = 400_000) -> ProposalExtract:
    """
    Extract plain text from a proposal PDF (local, no LlamaParse required).

    Prefer PyMuPDF (fitz) for layout fidelity; fall back to pypdf.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got {path.suffix!r}")

    pages_text: list[str] = []
    parser = "none"

    try:
        import fitz  # type: ignore

        doc = fitz.open(path)
        for i in range(doc.page_count):
            t = (doc[i].get_text() or "").strip()
            pages_text.append(t)
        doc.close()
        parser = "pymupdf"
    except Exception:  # noqa: BLE001
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for page in reader.pages:
            t = (page.extract_text() or "").strip()
            pages_text.append(t)
        parser = "pypdf"

    # Join with page markers so draft signals can still match
    chunks: list[str] = []
    previews: list[dict[str, Any]] = []
    for i, t in enumerate(pages_text, start=1):
        if not t:
            continue
        chunks.append(f"--- page {i} ---\n{t}")
        previews.append({"page": i, "chars": len(t), "preview": t[:240]})

    text = "\n\n".join(chunks)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[truncated]"

    return ProposalExtract(
        path=str(path),
        file_name=path.name,
        pages=len(pages_text),
        chars=len(text),
        text=text,
        parser=parser,
        page_previews=previews[:12],
    )


def proposal_doc_id(file_name: str) -> str:
    """Stable gg_doc_id for a user-uploaded proposal."""
    stem = Path(file_name).stem
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip("-").lower()
    return f"user-proposal-{safe[:80] or 'upload'}"
