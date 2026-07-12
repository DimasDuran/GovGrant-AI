"""
Orchestrate proposal upload: copy → extract → optional RAG index → registry.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from govgrant.auth.context import AuthContext
from govgrant.compliance.proposal import extract_proposal_text, proposal_doc_id
from govgrant.proposals.store import ProposalRecord, ProposalStore
from govgrant.rag.config import REPO_ROOT, Settings, get_settings
from govgrant.rag.index.hybrid import HybridRAGService


@dataclass
class ProposalUploadResult:
    record: ProposalRecord
    extract_parser: str
    index_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "extract_parser": self.extract_parser,
            "index_info": self.index_info,
        }


class ProposalService:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: ProposalStore | None = None,
        docs: HybridRAGService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        root = Path(
            getattr(self.settings, "proposals_dir", None)
            or (REPO_ROOT / "data" / "indexes" / "proposals")
        )
        self.proposals_dir = root
        self.proposals_dir.mkdir(parents=True, exist_ok=True)
        db_path = Path(
            getattr(self.settings, "proposals_db_path", None)
            or (self.proposals_dir / "proposals.sqlite")
        )
        self.store = store or ProposalStore(db_path)
        self.docs = docs

    def list_proposals(self, auth: AuthContext, *, limit: int = 50) -> list[ProposalRecord]:
        return self.store.list_for_tenant(auth.tenant_id, limit=limit)

    def get(self, auth: AuthContext, doc_id: str) -> ProposalRecord | None:
        auth.filter_doc_id(doc_id)  # public always ok; private must be allowed
        rec = self.store.get(auth.tenant_id, doc_id)
        if rec is None:
            return None
        # Tenant isolation: never return another tenant's row
        if rec.tenant_id != auth.tenant_id:
            return None
        return rec

    def upload(
        self,
        auth: AuthContext,
        source_pdf: Path | str,
        *,
        index: bool = True,
        notes: str = "",
    ) -> ProposalUploadResult:
        """
        Register a proposal PDF for this tenant.

        - Copies into data/indexes/proposals/{tenant_id}/
        - Extracts text for draft scoring
        - Optionally indexes into hybrid RAG under doc_id=user-proposal-*
        """
        source = Path(source_pdf).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)

        doc_id = proposal_doc_id(source.name)
        # Allow-list: only unrestricted tenants (allowed_doc_ids is None) may add
        # private proposal docs. Explicit lists are public-only or fixed IDs.
        if auth.allowed_doc_ids is not None:
            if (
                doc_id not in auth.allowed_doc_ids
                and doc_id not in auth.public_doc_ids
            ):
                from govgrant.auth import AuthError

                raise AuthError(
                    f"Tenant {auth.tenant_id!r} cannot register private proposal "
                    f"{doc_id!r} (not in allowed_doc_ids)"
                )

        tenant_dir = self.proposals_dir / auth.tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)
        dest = tenant_dir / source.name
        if dest.resolve() != source.resolve():
            shutil.copy2(source, dest)

        extracted = extract_proposal_text(dest)
        index_info: dict[str, Any] | None = None
        indexed = False
        if index:
            docs = self.docs or HybridRAGService(self.settings)
            index_info = docs.ingest_pdf(
                dest,
                tenant_id=auth.tenant_id,
                doc_id=doc_id,
                use_llamaparse=False,
                extract_tables=True,
                extract_figures=False,
                use_vision=False,
            )
            indexed = True

        rec = ProposalStore.new_record(
            doc_id=doc_id,
            tenant_id=auth.tenant_id,
            file_name=source.name,
            stored_path=str(dest),
            pages=extracted.pages,
            chars=extracted.chars,
            indexed=indexed,
            notes=notes,
        )
        self.store.upsert(rec)
        return ProposalUploadResult(
            record=rec,
            extract_parser=extracted.parser,
            index_info=index_info,
        )

    def delete(self, auth: AuthContext, doc_id: str, *, remove_file: bool = True) -> bool:
        rec = self.store.get(auth.tenant_id, doc_id)
        if rec is None or rec.tenant_id != auth.tenant_id:
            return False
        ok = self.store.delete(auth.tenant_id, doc_id)
        if ok and remove_file:
            path = Path(rec.stored_path)
            if path.is_file() and self.proposals_dir in path.resolve().parents:
                try:
                    path.unlink()
                except OSError:
                    pass
        return ok

    def read_draft_text(self, auth: AuthContext, doc_id: str) -> str:
        rec = self.get(auth, doc_id)
        if rec is None:
            raise FileNotFoundError(f"Proposal not found: {doc_id}")
        return extract_proposal_text(rec.stored_path).text
