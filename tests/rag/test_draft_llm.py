"""Tests for LLM draft scoring parser and merge (no API required)."""

from __future__ import annotations

from govgrant.compliance.checklist import darpa_items
from govgrant.compliance.draft_llm import (
    DraftJudgment,
    merge_draft_scores,
    parse_llm_draft_json,
)


def test_parse_llm_draft_json_clean():
    raw = """
    [
      {"id": "DARPA-WS-SBIR", "status": "draft_ok", "rationale": "States 50% in-house."},
      {"id": "DARPA-OT", "status": "draft_gap", "rationale": "No milestones listed."}
    ]
    """
    out = parse_llm_draft_json(raw)
    assert out["DARPA-WS-SBIR"].status == "draft_ok"
    assert out["DARPA-OT"].status == "draft_gap"
    assert out["DARPA-WS-SBIR"].method == "llm"


def test_parse_llm_draft_json_with_fences_and_noise():
    raw = """Here is the result:
```json
[{"id":"X","status":"draft_ok","rationale":"ok"}]
```
thanks
"""
    out = parse_llm_draft_json(raw)
    assert "X" in out
    assert out["X"].status == "draft_ok"


def test_parse_rejects_invalid_status():
    raw = '[{"id":"A","status":"maybe","rationale":"nope"}]'
    assert parse_llm_draft_json(raw) == {}


def test_merge_prefers_llm_over_keywords():
    items = [i for i in darpa_items() if i.id in {"DARPA-WS-SBIR", "DARPA-OT"}]
    keyword = {
        "DARPA-WS-SBIR": ("draft_gap", []),
        "DARPA-OT": ("draft_gap", []),
    }
    llm = {
        "DARPA-WS-SBIR": DraftJudgment(
            "DARPA-WS-SBIR", "draft_ok", "Mentions half the work", "llm"
        ),
    }
    merged = merge_draft_scores(items, keyword, llm)
    assert merged["DARPA-WS-SBIR"].method == "llm"
    assert merged["DARPA-WS-SBIR"].status == "draft_ok"
    assert merged["DARPA-OT"].method == "keyword"
    assert merged["DARPA-OT"].status == "draft_gap"
