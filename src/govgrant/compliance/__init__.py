"""Proposal compliance checklist against agency instructions."""

from govgrant.compliance.checklist import (
    ChecklistItem,
    ChecklistResult,
    ChecklistRun,
    run_checklist,
    run_darpa_phase2_checklist,
)
from govgrant.compliance.proposal import (
    ProposalExtract,
    extract_proposal_text,
    proposal_doc_id,
)

# draft_llm is optional (ChatLLM); import lazily via checklist.use_llm_draft

__all__ = [
    "ChecklistItem",
    "ChecklistResult",
    "ChecklistRun",
    "ProposalExtract",
    "extract_proposal_text",
    "proposal_doc_id",
    "run_checklist",
    "run_darpa_phase2_checklist",
]
