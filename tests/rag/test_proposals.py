"""Unit tests for tenant-scoped proposal registry (filesystem + sqlite)."""

from __future__ import annotations

from pathlib import Path

import pytest

from govgrant.auth.context import resolve_request_auth
from govgrant.proposals.service import ProposalService
from govgrant.proposals.store import ProposalStore


@pytest.fixture()
def svc(tmp_path: Path) -> ProposalService:
    db = tmp_path / "p.sqlite"
    root = tmp_path / "files"
    store = ProposalStore(db)
    service = ProposalService(store=store, docs=None)
    service.proposals_dir = root
    return service


def test_upload_list_delete(svc: ProposalService, tmp_path: Path):
    pdf = Path("data/fixtures/pdfs/darpa-sbir-sttr-phase-II-instructions.pdf")
    if not pdf.exists():
        pytest.skip("fixture PDF missing")
    auth = resolve_request_auth(require_auth=False)
    result = svc.upload(auth, pdf, index=False)
    assert result.record.tenant_id == auth.tenant_id
    assert result.record.doc_id.startswith("user-proposal-")
    assert result.record.pages >= 1
    assert not result.record.indexed

    listed = svc.list_proposals(auth)
    assert any(r.doc_id == result.record.doc_id for r in listed)

    text = svc.read_draft_text(auth, result.record.doc_id)
    assert len(text) > 100

    assert svc.delete(auth, result.record.doc_id)
    assert svc.get(auth, result.record.doc_id) is None


def test_delete_document_purges_bm25(tmp_path: Path):
    """HybridRAGService.delete_document removes matching BM25 leaves."""
    from llama_index.core.schema import TextNode

    from govgrant.rag.config import get_settings
    from govgrant.rag.index.hybrid import HybridRAGService

    settings = get_settings()
    # Lightweight instance: only exercise BM25 list + tabular path
    svc = HybridRAGService.__new__(HybridRAGService)
    svc.settings = settings
    svc._leaf_nodes = [
        TextNode(
            text="keep",
            metadata={"tenant_id": "t1", "gg_doc_id": "other", "doc_id": "other"},
        ),
        TextNode(
            text="drop",
            metadata={
                "tenant_id": "t1",
                "gg_doc_id": "user-proposal-x",
                "doc_id": "user-proposal-x",
            },
        ),
        TextNode(
            text="other-tenant",
            metadata={
                "tenant_id": "t2",
                "gg_doc_id": "user-proposal-x",
                "doc_id": "user-proposal-x",
            },
        ),
    ]
    svc._persist_bm25_nodes = lambda: None  # type: ignore[method-assign]
    # tabular stub
    class _Tab:
        def delete_doc(self, **kwargs):
            self.called = kwargs

    svc.tabular = _Tab()
    # Avoid real Qdrant by forcing client path to fail softly: monkeypatch delete body
    # Call method with qdrant blocked via patching get_qdrant_client inside method
    import govgrant.rag.index.hybrid as hybrid_mod

    class Boom:
        def collection_exists(self, *a, **k):
            raise RuntimeError("no qdrant in unit test")

    original = hybrid_mod.get_qdrant_client if hasattr(hybrid_mod, "get_qdrant_client") else None

    def _fake_client(*a, **k):
        return Boom()

    # delete_document imports get_qdrant_client from qdrant_store
    import govgrant.rag.index.qdrant_store as qs

    prev = qs.get_qdrant_client
    qs.get_qdrant_client = _fake_client  # type: ignore[assignment]
    try:
        info = HybridRAGService.delete_document(
            svc, tenant_id="t1", doc_id="user-proposal-x"
        )
    finally:
        qs.get_qdrant_client = prev  # type: ignore[assignment]

    assert info["bm25_removed"] == 1
    assert len(svc._leaf_nodes) == 2
    assert svc.tabular.called["gg_doc_id"] == "user-proposal-x"


def test_store_tenant_isolation(tmp_path: Path):
    store = ProposalStore(tmp_path / "x.sqlite")
    from govgrant.proposals.store import ProposalStore as PS

    a = PS.new_record(
        doc_id="user-proposal-a",
        tenant_id="t-a",
        file_name="a.pdf",
        stored_path="/tmp/a.pdf",
        pages=1,
        chars=10,
        indexed=False,
    )
    b = PS.new_record(
        doc_id="user-proposal-b",
        tenant_id="t-b",
        file_name="b.pdf",
        stored_path="/tmp/b.pdf",
        pages=1,
        chars=10,
        indexed=False,
    )
    store.upsert(a)
    store.upsert(b)
    assert len(store.list_for_tenant("t-a")) == 1
    assert store.list_for_tenant("t-a")[0].doc_id == "user-proposal-a"
    assert store.get("t-a", "user-proposal-b") is None
