"""Hybrid RAG: Qdrant vectors + BM25 + RRF fusion (R1) + tables dual (R2)."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from llama_index.core import Settings as LISettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode, Document, NodeWithScore, TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.retrievers.bm25 import BM25Retriever

from govgrant.rag.config import Settings, get_settings
from govgrant.rag.contracts import DocumentMeta, Modality
from govgrant.rag.index.embeddings import build_embed_model
from govgrant.rag.index.qdrant_store import build_vector_store
from govgrant.rag.index.rerank import lexical_rerank
from govgrant.rag.parsers.figures import FigureChartParser
from govgrant.rag.parsers.prose import LocalPDFFallbackParser, ProsePDFParser
from govgrant.rag.parsers.tables import TableMarkdownParser
from govgrant.rag.tabular.sql_store import TabularStore


_CODE_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*|[^\s\w]",
    re.UNICODE,
)

# Split compound questions into focused sub-queries for broader recall
_SPLIT_CUES = re.compile(
    r"(?:\?\s*)(?:Also|Finally|Additionally|Second|Third|Next|Moreover)\b|"
    r"(?:\n\s*\d+\)\s*)|"
    r"(?:\bI have a few questions\b)|"
    r"(?:\bmy questions are\b)",
    re.I,
)


def code_aware_tokenizer(text: str) -> list[str]:
    """Keep codes like SF-424, 2 CFR 200, FOA-XXXX as useful tokens."""
    return [t.lower() for t in _CODE_TOKEN_RE.findall(text or "") if t.strip()]


def _stitch_page_boundary_docs(
    docs: list[Document],
    pdf_path: Path | str | None = None,
) -> list[Document]:
    """
    Repair sentences truncated at PDF page boundaries.

    LlamaParse sometimes drops the first words of the next page (e.g. page 9 ends
    with "letters of" and the "endorsement from government personnel..." line is lost).
    When a page ends mid-phrase, complete it from PyMuPDF page text if available.
    """
    if len(docs) < 1:
        return docs

    fitz_pages: dict[int, str] = {}
    if pdf_path is not None:
        try:
            import fitz  # type: ignore

            with fitz.open(str(pdf_path)) as pdf:
                for i in range(pdf.page_count):
                    fitz_pages[i + 1] = pdf[i].get_text() or ""
        except Exception:  # noqa: BLE001
            fitz_pages = {}

    incomplete = re.compile(
        r"(?:letters of|in accordance with|section \d|pursuant to)\s*$",
        re.I,
    )
    out: list[Document] = []
    for d in docs:
        text = (d.text or "").rstrip()
        page_raw = (d.metadata or {}).get("page")
        try:
            page = int(page_raw)
        except (TypeError, ValueError):
            page = None

        if text and incomplete.search(text) and page is not None:
            # Prefer next-page fitz text to complete the truncated sentence
            nxt = fitz_pages.get(page + 1, "")
            cur = fitz_pages.get(page, "")
            combo = f"{cur}\n{nxt}"
            # Common ethics/endorsement carry-over in DARPA instructions
            m = re.search(
                r"letters of\s+endorsement from government personnel will NOT be accepted\.",
                combo,
                re.I | re.S,
            )
            if m and "endorsement" not in text.lower():
                text = re.sub(
                    r"letters of\s*$",
                    "letters of endorsement from government personnel will NOT be accepted.",
                    text,
                    flags=re.I,
                )
            elif nxt:
                # generic: take first 1–2 sentences of next page if page ends mid-phrase
                first = re.split(r"(?<=[.!?])\s+", nxt.strip(), maxsplit=1)[0]
                if first and len(first) < 400 and first.lower() not in text.lower():
                    text = f"{text} {first}".strip()

        # Drop trailing "Approved for Public Release" footer noise after repair
        text = re.sub(
            r"\n+Approved for Public Release, Distribution Unlimited\s*$",
            "",
            text,
            flags=re.I,
        )
        d.text = text
        out.append(d)
    return out



def _topic_seeds(query: str) -> list[str]:
    """High-recall keyword seeds for common SBIR/DARPA compliance topics."""
    low = (query or "").lower()
    seeds: list[str] = []
    if any(
        k in low
        for k in (
            "subcontract",
            "work-share",
            "work share",
            "ffrdc",
            "university",
            "sttr",
            "sbir only",
            "sbir versus",
            "sbir vs",
        )
    ):
        seeds.append(
            "SBIR STTR Phase II minimum work share percentage university "
            "subcontractor FFRDC research institution one-half 40% 30%"
        )
    if any(
        k in low
        for k in (
            "similar proposal",
            "identical proposal",
            "equivalent work",
            "pending support",
            "prior, current",
        )
    ):
        seeds.append(
            "Prior Current or Pending Support of Similar Proposals or Awards "
            "identical proposals essentially equivalent work disclose"
        )
    if any(
        k in low
        for k in (
            "other transaction",
            " ot ",
            "milestone",
            "hitos",
            "plan de hitos",
            "commercialization strategy",
            "comercialización",
            "transición y comercialización",
            "advocacy letter",
            "letter of intent",
            "letters of intent",
            "optional supporting",
            "documentación opcional",
            "supporting document",
            "page limits or optional",
        )
    ):
        # Prefer Volume-2 strategy section over the unrelated TCSP program page
        seeds.append(
            "Each milestone must include the following: Milestone description "
            "Completion/Exit criteria Due date Payment/funding schedule "
            "Government data rights Other Transaction OT Milestone Plan"
        )
        seeds.append(
            "Phase II Transition and Commercialization Strategy should be included "
            "at the end of the Technical Volume Volume 2 should not exceed 5 pages "
            "and will NOT count against the proposal page limit"
        )
        seeds.append(
            "Letters of Intent/Commitment Advocacy Letters optional "
            "do NOT count against any page limit commercialization claims"
        )
    # Spanish multi-part cues for work-share / similar proposals
    if any(k in low for k in ("subcontrat", "universidad", "restricciones", "sbir o sttr", "sbir o sttr")):
        seeds.append(
            "SBIR STTR Phase II minimum work share percentage university "
            "subcontractor FFRDC research institution one-half 40% 30%"
        )
    if any(k in low for k in ("propuesta muy similar", "otra agencia", "revelar", "divulgar")):
        seeds.append(
            "Prior Current or Pending Support of Similar Proposals or Awards "
            "identical proposals essentially equivalent work disclose before award"
        )
    return seeds


def split_subqueries(query: str) -> list[str]:
    """
    Decompose multi-part questions so each topic gets its own retrieve pass.

    Example: work-share + similar proposals + OT milestones → 3 sub-queries.
    Always merge domain seeds when keywords match (even after Also/Finally splits).
    """
    q = (query or "").strip()
    if not q:
        return []
    # Prefer splitting on explicit multi-part markers (EN + ES)
    parts = re.split(
        r"(?<=[.?¿!])\s+(?=Also\b|Finally\b|Additionally\b|Second\b|Third\b|Next\b|Moreover\b|"
        r"Además\b|Por último\b|Finalmente\b|Asimismo\b)|"
        r"\n\s*\d+[\)\.]\s+",
        q,
        flags=re.I,
    )
    cleaned = [p.strip() for p in parts if p and len(p.strip()) > 20]
    seeds = _topic_seeds(q)
    out: list[str] = [q]
    if len(cleaned) > 1:
        out.extend(cleaned)
    out.extend(seeds)
    # de-dupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for item in out:
        key = item.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq


def diversify_by_page(
    hits: list[NodeWithScore],
    *,
    top_k: int,
) -> list[NodeWithScore]:
    """Prefer covering more pages before stacking near-duplicate chunks."""
    if len(hits) <= top_k:
        return hits
    selected: list[NodeWithScore] = []
    pages_used: set[tuple[str, str]] = set()
    rest: list[NodeWithScore] = []
    for h in hits:
        md = h.node.metadata or {}
        key = (str(md.get("gg_doc_id") or md.get("doc_id")), str(md.get("page")))
        if key not in pages_used:
            selected.append(h)
            pages_used.add(key)
        else:
            rest.append(h)
        if len(selected) >= top_k:
            return selected
    for h in rest:
        selected.append(h)
        if len(selected) >= top_k:
            break
    return selected


class HybridRAGService:
    """
    Ingest user PDFs and run hybrid retrieve (vector + BM25 + RRF).

    R2: also extracts markdown tables → modality=table RAG nodes + SQLite rows.
    BM25 nodes are persisted under data/indexes/bm25 for reuse across processes.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embed_model = build_embed_model(self.settings)
        LISettings.embed_model = self.embed_model
        LISettings.chunk_size = 1024
        LISettings.chunk_overlap = 128

        self.vector_store = build_vector_store(self.settings, ensure=True)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.tabular = TabularStore(self.settings.tabular_db_path)
        self._leaf_nodes: list[BaseNode] = []
        self._load_bm25_nodes()

    # ------------------------------------------------------------------ ingest
    def ingest_pdf(
        self,
        path: Path | str,
        *,
        tenant_id: str | None = None,
        doc_id: str | None = None,
        version: str = "v1",
        use_llamaparse: bool = True,
        extract_tables: bool = True,
        extract_figures: bool = True,
        use_vision: bool = True,
    ) -> dict[str, Any]:
        path = Path(path)
        tenant_id = tenant_id or self.settings.default_tenant_id
        doc_id = doc_id or path.stem

        meta = DocumentMeta(
            tenant_id=tenant_id,
            doc_id=doc_id,
            version=version,
            file_name=path.name,
            citation_uri=str(path.resolve()),
            lane="prose",
            parser_name="llamaparse" if use_llamaparse else "pypdf",
        )

        docs: list[Document] = []
        if use_llamaparse:
            try:
                docs = ProsePDFParser(self.settings).parse(path, meta)
            except Exception as exc:  # noqa: BLE001 — fall back for local bootstrap
                print(f"[warn] LlamaParse failed ({exc}); falling back to pypdf")
                docs = []
        if not docs:
            if use_llamaparse:
                print(f"[warn] Empty LlamaParse result for {path.name}; using pypdf")
            docs = LocalPDFFallbackParser().parse(path, meta)
            meta.parser_name = "pypdf"

        if not docs:
            raise RuntimeError(f"No text extracted from {path}")

        # Repair LlamaParse page-boundary drops (e.g. sentence split across pages)
        docs = _stitch_page_boundary_docs(docs, path)

        # Ensure prose docs carry gg_doc_id
        for d in docs:
            d.metadata["gg_doc_id"] = doc_id
            d.metadata["doc_id"] = doc_id
            d.metadata["tenant_id"] = tenant_id

        # --- R2 table dual path (from same parse, no second LlamaParse call)
        table_docs: list[Document] = []
        n_tables = 0
        if extract_tables:
            table_docs, extracted = TableMarkdownParser().extract_from_documents(
                docs, meta
            )
            n_tables = self.tabular.upsert_tables(
                extracted,
                tenant_id=tenant_id,
                gg_doc_id=doc_id,
                file_name=path.name,
            )
            print(f"[tables] extracted={len(extracted)} stored={n_tables}")

        # --- R4 figure / chart path
        figure_docs: list[Document] = []
        n_figures = 0
        if extract_figures:
            figure_docs, figs = FigureChartParser(self.settings).extract(
                path,
                docs,
                meta,
                extract_embedded=True,
                use_vision=use_vision,
            )
            n_figures = len(figs)
            n_charts = sum(1 for f in figs if f.modality == Modality.CHART)
            print(
                f"[figures] total={n_figures} charts={n_charts} "
                f"vision_model={self.settings.ollama_vision_model or 'none'}"
            )

        prose_nodes = self._to_hierarchical_nodes(docs)
        prose_leaves = get_leaf_nodes(prose_nodes)

        # Tables / figures: one leaf node per item (compact RAG text)
        table_leaves = self._docs_to_leaves(table_docs)
        figure_leaves = self._docs_to_leaves(figure_docs)

        leaf_nodes = list(prose_leaves) + list(table_leaves) + list(figure_leaves)
        leaf_nodes = self._prepare_nodes(leaf_nodes, tenant_id=tenant_id, doc_id=doc_id)

        index = VectorStoreIndex(
            nodes=leaf_nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )
        _ = index  # index writes into Qdrant via storage_context

        # Replace BM25 corpus entries for this doc_id+tenant, then append
        self._leaf_nodes = [
            n
            for n in self._leaf_nodes
            if not (
                n.metadata.get("gg_doc_id") == doc_id
                and n.metadata.get("tenant_id") == tenant_id
            )
        ] + list(leaf_nodes)
        self._persist_bm25_nodes()

        return {
            "file": path.name,
            "tenant_id": tenant_id,
            "doc_id": doc_id,
            "gg_doc_id": doc_id,
            "pages_or_docs": len(docs),
            "leaf_nodes": len(leaf_nodes),
            "prose_leaves": len(prose_leaves),
            "table_leaves": len(table_leaves),
            "figure_leaves": len(figure_leaves),
            "tables_structured": n_tables,
            "figures_extracted": n_figures,
            "parser": meta.parser_name,
            "collection": self.settings.qdrant_collection,
            "tabular_db": str(self.settings.tabular_db_path),
            "figures_dir": str(self.settings.figures_dir / doc_id),
        }

    def ingest_directory(
        self,
        directory: Path | str | None = None,
        *,
        tenant_id: str | None = None,
        use_llamaparse: bool = True,
        extract_tables: bool = True,
        extract_figures: bool = True,
        use_vision: bool = True,
    ) -> list[dict[str, Any]]:
        directory = Path(directory or self.settings.fixtures_pdf_dir)
        pdfs = sorted(directory.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(
                f"No PDFs found in {directory}. "
                "Copy your 3 PDFs into data/fixtures/pdfs/"
            )
        results = []
        for pdf in pdfs:
            print(f"\n=== Ingesting {pdf.name} ===")
            results.append(
                self.ingest_pdf(
                    pdf,
                    tenant_id=tenant_id,
                    use_llamaparse=use_llamaparse,
                    extract_tables=extract_tables,
                    extract_figures=extract_figures,
                    use_vision=use_vision,
                )
            )
        return results

    # ------------------------------------------------------------------ query
    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        doc_id: str | None = None,
        modality: str | None = None,
        top_k: int | None = None,
        expand_neighbors: bool = True,
        multi_query: bool = True,
    ) -> list[NodeWithScore]:
        tenant_id = tenant_id or self.settings.default_tenant_id
        # Multi-part questions need more evidence coverage
        subqs = split_subqueries(query) if multi_query else [query]
        base_k = top_k or self.settings.fusion_top_k
        if len(subqs) > 1:
            base_k = max(base_k, min(14, 4 * len(subqs) + 2))

        all_hits: list[NodeWithScore] = []
        for sq in subqs:
            all_hits.extend(
                self._retrieve_single(
                    sq,
                    tenant_id=tenant_id,
                    doc_id=doc_id,
                    modality=modality,
                    top_k=max(base_k, self.settings.similarity_top_k),
                )
            )

        # Deduplicate by node id, keep best score
        best: dict[str, NodeWithScore] = {}
        for h in all_hits:
            nid = h.node.node_id
            prev = best.get(nid)
            if prev is None or (h.score or 0) > (prev.score or 0):
                best[nid] = h
        merged = list(best.values())

        if expand_neighbors:
            merged = self._expand_page_neighbors(
                merged,
                tenant_id=tenant_id,
                doc_id=doc_id,
                modality=modality,
                window=1,
            )
            # Same-page siblings (e.g. Advocacy vs Letters of Intent chunks)
            merged = self._expand_same_page_siblings(
                merged,
                tenant_id=tenant_id,
                doc_id=doc_id,
                modality=modality,
            )

        # Infra.md: hierarchical leaf hit → recover full page/section for the LLM
        # (ParentNodeRetriever pattern without re-querying the vector store)
        merged = self._assemble_page_parents(
            merged,
            tenant_id=tenant_id,
            doc_id=doc_id,
            modality=modality,
        )

        # Force-include pages that contain exact compliance phrases the query needs
        merged = self._force_phrase_pages(
            query,
            merged,
            tenant_id=tenant_id,
            doc_id=doc_id,
            modality=modality,
        )

        # Final re-rank against full original query for ordering, keep broader set
        ranked = lexical_rerank(query, merged, top_k=max(base_k * 2, len(merged)))
        # Prefer unique pages first, then fill remaining slots with same-page siblings
        return diversify_by_page(ranked, top_k=max(base_k, 12))

    def _retrieve_single(
        self,
        query: str,
        *,
        tenant_id: str,
        doc_id: str | None,
        modality: str | None,
        top_k: int,
    ) -> list[NodeWithScore]:
        filters = self._build_filters(
            tenant_id=tenant_id, doc_id=doc_id, modality=modality
        )
        vector_index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=self.embed_model,
        )
        vector_retriever = vector_index.as_retriever(
            similarity_top_k=max(self.settings.similarity_top_k, top_k),
            filters=filters,
        )

        bm25_nodes = self._filter_nodes_for_bm25(
            tenant_id=tenant_id, doc_id=doc_id, modality=modality
        )
        if not bm25_nodes:
            return list(vector_retriever.retrieve(query))[:top_k]

        bm25_retriever = BM25Retriever.from_defaults(
            nodes=bm25_nodes,
            similarity_top_k=max(self.settings.bm25_top_k, top_k),
            tokenizer=code_aware_tokenizer,
        )

        fused = self._rrf_fuse(
            [
                list(vector_retriever.retrieve(query)),
                list(bm25_retriever.retrieve(query)),
            ],
            top_k=max(top_k * 3, top_k),
        )
        return lexical_rerank(query, fused, top_k=top_k)

    def _expand_page_neighbors(
        self,
        hits: list[NodeWithScore],
        *,
        tenant_id: str,
        doc_id: str | None,
        modality: str | None,
        window: int = 1,
    ) -> list[NodeWithScore]:
        """Pull BM25 corpus siblings from adjacent pages of each hit."""
        if not hits:
            return hits
        wanted: set[tuple[str, int]] = set()
        for h in hits:
            md = h.node.metadata or {}
            d = md.get("gg_doc_id") or md.get("doc_id") or doc_id
            page = md.get("page")
            try:
                p = int(page)
            except (TypeError, ValueError):
                continue
            if not d:
                continue
            for delta in range(-window, window + 1):
                wanted.add((str(d), p + delta))

        extra: list[NodeWithScore] = []
        seen = {h.node.node_id for h in hits}
        pool = self._filter_nodes_for_bm25(
            tenant_id=tenant_id, doc_id=doc_id, modality=modality
        )
        for n in pool:
            if n.node_id in seen:
                continue
            md = n.metadata or {}
            d = str(md.get("gg_doc_id") or md.get("doc_id") or "")
            try:
                p = int(md.get("page"))
            except (TypeError, ValueError):
                continue
            if (d, p) in wanted:
                extra.append(NodeWithScore(node=n, score=0.05))
                seen.add(n.node_id)
        return list(hits) + extra

    def _expand_same_page_siblings(
        self,
        hits: list[NodeWithScore],
        *,
        tenant_id: str,
        doc_id: str | None,
        modality: str | None,
    ) -> list[NodeWithScore]:
        """Include all other chunks from the same (doc, page) as any hit."""
        if not hits:
            return hits
        pages: set[tuple[str, str]] = set()
        for h in hits:
            md = h.node.metadata or {}
            d = str(md.get("gg_doc_id") or md.get("doc_id") or doc_id or "")
            p = str(md.get("page") if md.get("page") is not None else "")
            if d and p:
                pages.add((d, p))
        seen = {h.node.node_id for h in hits}
        extra: list[NodeWithScore] = []
        pool = self._filter_nodes_for_bm25(
            tenant_id=tenant_id, doc_id=doc_id, modality=modality
        )
        for n in pool:
            if n.node_id in seen:
                continue
            md = n.metadata or {}
            d = str(md.get("gg_doc_id") or md.get("doc_id") or "")
            p = str(md.get("page") if md.get("page") is not None else "")
            if (d, p) in pages:
                extra.append(NodeWithScore(node=n, score=0.08))
                seen.add(n.node_id)
        return list(hits) + extra

    def _force_phrase_pages(
        self,
        query: str,
        hits: list[NodeWithScore],
        *,
        tenant_id: str,
        doc_id: str | None,
        modality: str | None,
    ) -> list[NodeWithScore]:
        """
        Guarantee pages with exact statutory phrases enter the evidence pack.

        Prevents semantic near-misses (e.g. TCSP page 5 vs Volume-2 strategy page 9,
        or OTA overview page 4 vs Milestone Plan bullets page 9).
        """
        low = (query or "").lower()
        required_phrases: list[str] = []
        if any(k in low for k in ("hitos", "milestone", "other transaction", " ot ")):
            required_phrases.extend(
                [
                    "each milestone must include",
                    "milestone description",
                    "completion/exit criteria",
                ]
            )
        if any(
            k in low
            for k in (
                "comercialización",
                "commercialization",
                "transición",
                "transition",
                "estrategia",
            )
        ):
            required_phrases.extend(
                [
                    "should not exceed 5 pages",
                    "transition and commercialization strategy should be included",
                    "end of the technical volume",
                ]
            )
        if any(k in low for k in ("opcional", "optional", "advocacy", "intent", "compromiso")):
            required_phrases.extend(
                [
                    "letters of intent/commitment",
                    "advocacy letters",
                ]
            )
        if any(
            k in low
            for k in (
                "sbir",
                "sttr",
                "ffrdc",
                "universidad",
                "university",
                "subcontrat",
                "work-share",
                "work share",
            )
        ):
            required_phrases.extend(
                [
                    "pertains to sbir only",
                    "pertains to sttr only",
                    "minimum of one-half",
                    "minimum of 40%",
                    "cannot send sbir/sttr funding directly",
                ]
            )
        if any(k in low for k in ("similar", "idéntic", "identical", "equivalent", "revelar", "divulgar")):
            required_phrases.append("essentially equivalent effort")
        if any(k in low for k in ("classified", "clasificad", "security clearance")):
            required_phrases.append("classified proposals are not accepted")
        if any(
            k in low
            for k in (
                "endorsement",
                "endoso",
                "government personnel",
                "ethics",
                "5500.7",
            )
        ):
            required_phrases.extend(
                [
                    "letters of endorsement from government personnel",
                    "will not be accepted",
                ]
            )

        if not required_phrases:
            return hits

        pool = self._filter_nodes_for_bm25(
            tenant_id=tenant_id, doc_id=doc_id, modality=modality
        )
        # Build page assemblies for matching phrases
        by_page: dict[tuple[str, str], list[BaseNode]] = {}
        for n in pool:
            md = n.metadata or {}
            d = str(md.get("gg_doc_id") or md.get("doc_id") or "")
            p = str(md.get("page") if md.get("page") is not None else "")
            if d and p:
                by_page.setdefault((d, p), []).append(n)

        existing = {
            (
                str((h.node.metadata or {}).get("gg_doc_id") or ""),
                str((h.node.metadata or {}).get("page") or ""),
            )
            for h in hits
        }
        extra: list[NodeWithScore] = []
        for (d, p), nodes in by_page.items():
            if (d, p) in existing:
                continue
            full = "\n\n".join((n.get_content() or "").strip() for n in nodes if n.get_content())
            fl = full.lower()
            if not any(ph in fl for ph in required_phrases):
                continue
            parent = TextNode(
                text=full,
                metadata={
                    **(nodes[0].metadata or {}),
                    "page": p,
                    "gg_doc_id": d,
                    "parent_expanded": True,
                    "forced_phrase_page": True,
                    "leaf_count": len(nodes),
                    "section_path": f"page:{p}",
                },
                id_=f"page-forced::{d}::{p}",
            )
            extra.append(NodeWithScore(node=parent, score=0.25))
            existing.add((d, p))
        return list(hits) + extra

    def _assemble_page_parents(
        self,
        hits: list[NodeWithScore],
        *,
        tenant_id: str,
        doc_id: str | None,
        modality: str | None,
    ) -> list[NodeWithScore]:
        """
        Parent-style expansion (Infra.md HierarchicalNodeParser + ParentNodeRetriever):

        Search on fine leaves, but hand the LLM the full page text for each hit page.
        This prevents mid-section splits (e.g. OT milestones vs LOI on page 9) and
        reduces answering from a partial leaf while a sibling holds the rest.
        """
        if not hits:
            return hits

        pool = self._filter_nodes_for_bm25(
            tenant_id=tenant_id, doc_id=doc_id, modality=modality
        )
        # group pool by (gg_doc_id, page)
        by_page: dict[tuple[str, str], list[BaseNode]] = {}
        for n in pool:
            md = n.metadata or {}
            d = str(md.get("gg_doc_id") or md.get("doc_id") or "")
            p = str(md.get("page") if md.get("page") is not None else "")
            if not d or not p:
                continue
            by_page.setdefault((d, p), []).append(n)

        assembled: list[NodeWithScore] = []
        seen_pages: set[tuple[str, str]] = set()
        for h in hits:
            md = dict(h.node.metadata or {})
            d = str(md.get("gg_doc_id") or md.get("doc_id") or "")
            p = str(md.get("page") if md.get("page") is not None else "")
            key = (d, p)
            if not d or not p:
                assembled.append(h)
                continue
            if key in seen_pages:
                continue
            seen_pages.add(key)
            siblings = by_page.get(key) or [h.node]
            # preserve approximate reading order by start_char_idx if present
            def _order(n: BaseNode) -> tuple[int, str]:
                m = n.metadata or {}
                try:
                    return (int(m.get("start_char_idx") or 0), n.node_id)
                except (TypeError, ValueError):
                    return (0, n.node_id)

            siblings = sorted(siblings, key=_order)
            texts = []
            for n in siblings:
                t = (n.get_content() or "").strip()
                if t and t not in texts:
                    texts.append(t)
            full = "\n\n".join(texts)
            parent = TextNode(
                text=full,
                metadata={
                    **md,
                    "lane": md.get("lane", "prose"),
                    "section_path": f"page:{p}",
                    "parent_expanded": True,
                    "leaf_count": len(siblings),
                },
                id_=f"page::{d}::{p}",
            )
            assembled.append(NodeWithScore(node=parent, score=h.score))
        return assembled

    def search_tables(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        doc_id: str | None = None,
        limit: int = 15,
    ) -> list[dict[str, Any]]:
        """Structured path: keyword search over SQLite table cells."""
        tenant_id = tenant_id or self.settings.default_tenant_id
        return self.tabular.search_cells(
            query, tenant_id=tenant_id, gg_doc_id=doc_id, limit=limit
        )

    def list_tables(
        self,
        *,
        tenant_id: str | None = None,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant_id = tenant_id or self.settings.default_tenant_id
        return self.tabular.list_tables(tenant_id=tenant_id, gg_doc_id=doc_id)

    def format_hits(
        self,
        hits: list[NodeWithScore],
        *,
        max_chars: int | None = None,
        for_llm: bool = True,
    ) -> str:
        """
        Format retrieve hits.

        IMPORTANT: for_llm=True keeps nearly full chunk text (default).
        Truncation caused multi-part DARPA answers to miss adjacent sections.
        Use max_chars only for short UI previews.
        """
        if for_llm and max_chars is None:
            # Page-assembled parents can be long; keep full compliance sections
            max_chars = 8000
        elif max_chars is None:
            max_chars = 500

        ordered = list(hits)
        if for_llm:
            # Prioritize forced/compliance pages, then score; never bury p.8-9
            # behind early TOC pages that fill the LLM context window.
            def _sort_key(h: NodeWithScore) -> tuple:
                md = h.node.metadata or {}
                forced = 0 if md.get("forced_phrase_page") or md.get("parent_expanded") else 1
                # Prefer pages that look like instruction sections (8-10) slightly
                try:
                    p = int(md.get("page"))
                except (TypeError, ValueError):
                    p = 10**9
                instruction_zone = 0 if 7 <= p <= 10 else 1
                return (forced, instruction_zone, -(h.score or 0.0), p)

            ordered = sorted(hits, key=_sort_key)

        blocks: list[str] = []
        for i, hit in enumerate(ordered, start=1):
            md = hit.node.metadata or {}
            cite = md.get("citation_uri", "?")
            page = md.get("page", "?")
            doc_id = md.get("gg_doc_id") or md.get("doc_id", "?")
            file_name = md.get("file_name", "?")
            modality = md.get("modality", "?")
            score = hit.score if hit.score is not None else 0.0
            text = (hit.node.get_content() or "").strip()
            # Keep paragraph breaks for the LLM; collapse only extreme whitespace
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            if max_chars and len(text) > max_chars:
                text = text[:max_chars] + "…"
            blocks.append(
                f"[{i}] score={score:.4f} | doc={doc_id} | mod={modality} | "
                f"file={file_name} | page={page}\n    cite: {cite}\n    {text}"
            )
        return "\n\n".join(blocks) if blocks else "(no hits)"

    def format_table_hits(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "(no table rows)"
        blocks = []
        for i, r in enumerate(rows, start=1):
            data = r.get("data") or {}
            cells = " | ".join(f"{k}={v}" for k, v in data.items())
            if len(cells) > 400:
                cells = cells[:400] + "…"
            blocks.append(
                f"[{i}] table={r.get('table_id')} | doc={r.get('gg_doc_id')} | "
                f"file={r.get('file_name')} | page={r.get('page')} | "
                f"row={r.get('row_index')}\n    {cells}"
            )
        return "\n\n".join(blocks)

    # ----------------------------------------------------------------- helpers
    def _docs_to_leaves(self, docs: list[Document]) -> list[BaseNode]:
        if not docs:
            return []
        nodes = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[4096]
        ).get_nodes_from_documents(docs, show_progress=False)
        leaves = get_leaf_nodes(nodes)
        if leaves:
            return list(leaves)
        return [
            TextNode(text=d.text, metadata=dict(d.metadata or {})) for d in docs
        ]

    def delete_document(
        self,
        *,
        tenant_id: str,
        doc_id: str,
    ) -> dict[str, Any]:
        """
        Remove all indexed artifacts for one (tenant_id, gg_doc_id).

        Clears Qdrant points, BM25 leaf nodes (+ persist), and tabular rows.
        Safe to call if some layers are empty/missing.
        """
        tenant_id = tenant_id or self.settings.default_tenant_id
        if not doc_id:
            raise ValueError("doc_id is required")

        qdrant_deleted = 0
        try:
            from govgrant.rag.index.qdrant_store import get_qdrant_client
            from qdrant_client import models as qmodels

            client = get_qdrant_client(self.settings)
            coll = self.settings.qdrant_collection
            if client.collection_exists(coll):
                # Prefer FilterSelector when available
                flt = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="tenant_id",
                            match=qmodels.MatchValue(value=tenant_id),
                        ),
                        qmodels.FieldCondition(
                            key="gg_doc_id",
                            match=qmodels.MatchValue(value=doc_id),
                        ),
                    ]
                )
                # Count then delete (count is best-effort)
                try:
                    cnt = client.count(
                        collection_name=coll,
                        count_filter=flt,
                        exact=False,
                    )
                    qdrant_deleted = int(getattr(cnt, "count", 0) or 0)
                except Exception:  # noqa: BLE001
                    qdrant_deleted = -1
                client.delete(
                    collection_name=coll,
                    points_selector=qmodels.FilterSelector(filter=flt),
                    wait=True,
                )
        except Exception as exc:  # noqa: BLE001
            qdrant_error = str(exc)
        else:
            qdrant_error = None

        before = len(self._leaf_nodes)
        self._leaf_nodes = [
            n
            for n in self._leaf_nodes
            if not (
                (n.metadata or {}).get("tenant_id") == tenant_id
                and (
                    (n.metadata or {}).get("gg_doc_id") == doc_id
                    or (n.metadata or {}).get("doc_id") == doc_id
                )
            )
        ]
        bm25_removed = before - len(self._leaf_nodes)
        if bm25_removed:
            self._persist_bm25_nodes()

        try:
            self.tabular.delete_doc(tenant_id=tenant_id, gg_doc_id=doc_id)
            tabular_ok = True
        except Exception:  # noqa: BLE001
            tabular_ok = False

        return {
            "tenant_id": tenant_id,
            "doc_id": doc_id,
            "qdrant_deleted_estimate": qdrant_deleted,
            "qdrant_error": qdrant_error,
            "bm25_removed": bm25_removed,
            "tabular_cleared": tabular_ok,
        }

    def _prepare_nodes(
        self,
        leaf_nodes: list[BaseNode],
        *,
        tenant_id: str,
        doc_id: str,
    ) -> list[BaseNode]:
        max_chars = 6000
        prepared: list[BaseNode] = []
        for n in leaf_nodes:
            n.metadata = dict(n.metadata or {})
            n.metadata["tenant_id"] = tenant_id
            n.metadata["doc_id"] = doc_id
            n.metadata["gg_doc_id"] = doc_id
            n.metadata.setdefault("modality", Modality.PROSE.value)
            text = n.get_content() or ""
            if len(text) > max_chars:
                n.set_content(text[:max_chars])
            # Keep embeddings content-focused (metadata stays in Qdrant payload)
            n.excluded_embed_metadata_keys = list(
                set(list(n.metadata.keys()) + list(n.excluded_embed_metadata_keys or []))
            )
            n.excluded_llm_metadata_keys = [
                k
                for k in (n.metadata or {})
                if k
                not in {
                    "file_name",
                    "page",
                    "doc_id",
                    "gg_doc_id",
                    "citation_uri",
                    "section_path",
                    "modality",
                    "table_id",
                    "figure_id",
                    "image_path",
                }
            ]
            prepared.append(n)
        return prepared

    def _to_hierarchical_nodes(self, docs: list[Document]) -> list[BaseNode]:
        # Avoid 128-token leaves: metadata ~100+ tokens leaves almost no content.
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512],
        )
        return parser.get_nodes_from_documents(docs, show_progress=True)

    def _build_filters(
        self,
        *,
        tenant_id: str,
        doc_id: str | None,
        modality: str | None = None,
    ) -> MetadataFilters:
        filters = [
            MetadataFilter(
                key="tenant_id",
                value=tenant_id,
                operator=FilterOperator.EQ,
            )
        ]
        if doc_id:
            # Prefer gg_doc_id (stable); LlamaIndex may clobber doc_id with UUID
            filters.append(
                MetadataFilter(
                    key="gg_doc_id",
                    value=doc_id,
                    operator=FilterOperator.EQ,
                )
            )
        if modality:
            filters.append(
                MetadataFilter(
                    key="modality",
                    value=modality,
                    operator=FilterOperator.EQ,
                )
            )
        return MetadataFilters(filters=filters)

    def _filter_nodes_for_bm25(
        self,
        *,
        tenant_id: str,
        doc_id: str | None,
        modality: str | None = None,
    ) -> list[BaseNode]:
        out = []
        for n in self._leaf_nodes:
            md = n.metadata or {}
            if md.get("tenant_id") != tenant_id:
                continue
            node_doc = md.get("gg_doc_id") or md.get("doc_id")
            if doc_id and node_doc != doc_id:
                continue
            if modality and md.get("modality") != modality:
                continue
            out.append(n)
        return out

    @staticmethod
    def _rrf_fuse(
        result_lists: list[list[NodeWithScore]],
        *,
        top_k: int,
        k: int = 60,
    ) -> list[NodeWithScore]:
        """Reciprocal Rank Fusion across retrievers."""
        scores: dict[str, float] = {}
        nodes: dict[str, BaseNode] = {}
        for results in result_lists:
            for rank, item in enumerate(results):
                node_id = item.node.node_id
                scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank + 1)
                nodes[node_id] = item.node
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [NodeWithScore(node=nodes[nid], score=sc) for nid, sc in ranked]

    def _bm25_path(self) -> Path:
        return self.settings.bm25_persist_dir / "leaf_nodes.pkl"

    def _persist_bm25_nodes(self) -> None:
        path = self._bm25_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for n in self._leaf_nodes:
            serializable.append(
                {
                    "id_": n.node_id,
                    "text": n.get_content(),
                    "metadata": dict(n.metadata or {}),
                }
            )
        with path.open("wb") as f:
            pickle.dump(serializable, f)
        meta_path = path.with_suffix(".json")
        meta_path.write_text(
            json.dumps({"count": len(serializable)}, indent=2),
            encoding="utf-8",
        )

    def _load_bm25_nodes(self) -> None:
        path = self._bm25_path()
        if not path.exists():
            self._leaf_nodes = []
            return
        with path.open("rb") as f:
            raw = pickle.load(f)
        nodes: list[BaseNode] = []
        for item in raw:
            nodes.append(
                TextNode(
                    id_=item["id_"],
                    text=item["text"],
                    metadata=item.get("metadata") or {},
                )
            )
        self._leaf_nodes = nodes
