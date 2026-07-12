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
