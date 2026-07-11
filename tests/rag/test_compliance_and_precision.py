"""Unit tests for compliance checklist helpers and answer precision filter."""

from __future__ import annotations

from govgrant.agent.llm import strip_unsolicited_digressions
from govgrant.compliance.checklist import darpa_phase2_items, _score_item


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


def test_checklist_items_cover_core_controls():
    ids = {i.id for i in darpa_phase2_items()}
    for need in (
        "WS-SBIR",
        "WS-STTR",
        "FFRDC",
        "SIMILAR",
        "OT-MILESTONES",
        "COMM-STRAT",
        "COST-MAX",
    ):
        assert need in ids


def test_score_item_pass_on_evidence():
    item = next(i for i in darpa_phase2_items() if i.id == "WS-SBIR")
    evidence = (
        "[1] score=0.9 | page=8\n"
        "THE FOLLOWING PERTAINS TO SBIR ONLY: A minimum of one-half of the "
        "research work in Phase II must be carried out by the proposer."
    )
    r = _score_item(item, evidence, program="sbir")
    assert r.status == "pass"
    assert "one-half" in r.facts_found
