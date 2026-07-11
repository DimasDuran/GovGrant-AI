"""Table lane: extract markdown tables from parsed page text (R2)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.schema import Document

from govgrant.rag.contracts import DocumentMeta, Modality, build_node_metadata
from govgrant.rag.parsers.base import BaseModalityParser

# GFM-style markdown tables
_TABLE_BLOCK = re.compile(
    r"(?:^|\n)"
    r"(\|[^\n]+\|\s*\n"  # header
    r"\|[-:\s|]+\|\s*\n"  # separator
    r"(?:\|[^\n]+\|\s*\n?)+)",  # body rows
    re.MULTILINE,
)


@dataclass
class ExtractedTable:
    """One table extracted from a page/document."""

    table_id: str
    headers: list[str]
    rows: list[list[str]]
    markdown: str
    page: int | str | None = None
    section_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_rag_text(self) -> str:
        """Text representation for hybrid RAG (headers + row lines)."""
        lines = [
            f"TABLE headers: {' | '.join(self.headers)}",
            f"rows: {self.row_count}",
        ]
        for i, row in enumerate(self.rows, start=1):
            pairs = []
            for h, v in zip(self.headers, row, strict=False):
                pairs.append(f"{h}={v}")
            # pad if row shorter
            if len(row) > len(self.headers):
                pairs.extend(row[len(self.headers) :])
            lines.append(f"row[{i}]: " + " | ".join(pairs))
        return "\n".join(lines)

    def to_row_dicts(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for row in self.rows:
            d: dict[str, str] = {}
            for i, h in enumerate(self.headers):
                d[h or f"col_{i}"] = row[i] if i < len(row) else ""
            out.append(d)
        return out


def _split_md_row(line: str) -> list[str]:
    line = line.strip().strip("|")
    return [c.strip() for c in line.split("|")]


def _is_separator(cells: list[str]) -> bool:
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells if c)


def parse_markdown_tables(
    text: str,
    *,
    page: int | str | None = None,
    doc_id: str = "doc",
    start_index: int = 0,
) -> list[ExtractedTable]:
    """Find GFM tables in text and parse headers/rows."""
    tables: list[ExtractedTable] = []
    if not text:
        return tables

    for i, match in enumerate(_TABLE_BLOCK.finditer(text)):
        block = match.group(1).strip()
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        headers = _split_md_row(lines[0])
        body_start = 1
        if body_start < len(lines) and _is_separator(_split_md_row(lines[body_start])):
            body_start += 1
        rows: list[list[str]] = []
        for ln in lines[body_start:]:
            cells = _split_md_row(ln)
            if _is_separator(cells):
                continue
            # normalize width
            if len(cells) < len(headers):
                cells = cells + [""] * (len(headers) - len(cells))
            rows.append(cells[: len(headers)] if headers else cells)
        if not headers:
            continue
        idx = start_index + i
        table_id = f"{doc_id}::p{page}::t{idx}"
        tables.append(
            ExtractedTable(
                table_id=table_id,
                headers=headers,
                rows=rows,
                markdown=block,
                page=page,
                section_path=f"page:{page}/table:{idx}" if page is not None else f"table:{idx}",
            )
        )
    return tables


class TableMarkdownParser(BaseModalityParser):
    """
    Build LlamaIndex Documents (modality=table) from prose Documents
    that already contain markdown tables (e.g. LlamaParse output).
    """

    lane = "table"

    def parse(self, path: Path, meta: DocumentMeta) -> list[Document]:
        raise NotImplementedError(
            "Use extract_from_documents() — tables come from prior prose parse."
        )

    def extract_from_documents(
        self,
        prose_docs: list[Document],
        meta: DocumentMeta,
    ) -> tuple[list[Document], list[ExtractedTable]]:
        rag_docs: list[Document] = []
        extracted: list[ExtractedTable] = []
        global_i = 0
        for doc in prose_docs:
            page = doc.metadata.get("page")
            found = parse_markdown_tables(
                doc.text or "",
                page=page,
                doc_id=meta.doc_id,
                start_index=global_i,
            )
            global_i += len(found)
            for t in found:
                extracted.append(t)
                node_meta = build_node_metadata(
                    meta,
                    page=t.page,
                    section_path=t.section_path,
                    modality=Modality.TABLE,
                    lane=self.lane,
                    table_id=t.table_id,
                    row_count=t.row_count,
                    headers="|".join(t.headers[:20]),
                )
                # Canonical filter field (not overwritten by LlamaIndex ref_doc_id)
                node_meta["gg_doc_id"] = meta.doc_id
                rag_docs.append(
                    Document(
                        text=t.to_rag_text(),
                        metadata=node_meta,
                    )
                )
        return rag_docs, extracted
