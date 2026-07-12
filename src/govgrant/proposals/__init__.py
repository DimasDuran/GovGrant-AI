"""Tenant-scoped user proposal registry and upload service."""

from govgrant.proposals.service import ProposalService, ProposalUploadResult
from govgrant.proposals.store import ProposalRecord, ProposalStore

__all__ = [
    "ProposalRecord",
    "ProposalService",
    "ProposalStore",
    "ProposalUploadResult",
]
