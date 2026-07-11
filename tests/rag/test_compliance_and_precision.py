"""Unit tests for compliance checklist helpers and answer precision filter."""

from __future__ import annotations

from govgrant.agent.llm import strip_unsolicited_digressions
from govgrant.compliance.checklist import (
    _score_corpus,
    _score_draft,
    all_items,
    darpa_items,
    sba_items,
    sf424_items,
)


def test_strip_volume5_when_not_asked():
    q = (
        "Compare SBIR vs STTR work-share and OT milestones and commercialization strategy"
    )
    answer = """# Work-share

SBIR 50%. STTR 40% and 30%.

## Volume 5 Supporting Documents

You can also upload Subcontract Pricing in Volume 5.

## CCR Volume 4

Upload CCR.
"""
    out = strip_unsolicited_digressions(answer, query=q)
    assert "Work-share" in out or "50%" in out
    assert "Volume 5" not in out
    assert "CCR Volume 4" not in out


def test_keep_volume5_when_asked():
    q = "What goes in Volume 5 Supporting Documents?"
    answer = """# Volume 5 Supporting Documents

Include subcontract pricing and data rights assertions.
"""
    out = strip_unsolicited_digressions(answer, query=q)
    assert "Volume 5" in out
    assert "subcontract" in out.lower()


def test_checklist_items_cover_multi_agency():
    ids = {i.id for i in all_items()}
    for need in (
        "DARPA-WS-SBIR",
        "DARPA-OT",
        "SBA-WS-SBIR-II",
        "SBA-FOREIGN",
        "SF424-RS-REQUIRED",
        "SF424-UEI",
    ):
        assert need in ids
    assert len(darpa_items()) >= 8
    assert len(sba_items()) >= 5
    assert len(sf424_items()) >= 5


def test_score_corpus_pass_on_evidence():
    item = next(i for i in darpa_items() if i.id == "DARPA-WS-SBIR")
    evidence = (
        "[1] score=0.9 | page=8\n"
        "THE FOLLOWING PERTAINS TO SBIR ONLY: A minimum of one-half of the "
        "research work in Phase II must be carried out by the proposer."
    )
    status, found, missing, _ = _score_corpus(item, evidence)
    assert status == "pass"
    assert "one-half" in found
    assert not missing or "SBIR" in found or "SBIR" in missing


def test_draft_signals_detect_workshare():
    item = next(i for i in darpa_items() if i.id == "DARPA-WS-SBIR")
    draft = (
        "Our SBIR Phase II team will perform at least 50% of the research "
        "in-house. University subcontractors will perform the remainder."
    )
    status, hits = _score_draft(item, draft)
    assert status == "draft_ok"
    assert hits


def test_draft_gap_when_empty():
    item = next(i for i in sba_items() if i.id == "SBA-FOREIGN")
    status, hits = _score_draft(item, "We propose a novel sensor architecture.")
    assert status == "draft_gap"
