"""
Optional LLM-assisted draft scoring for compliance controls.

Design:
  - One batched Haiku call for all selected controls (cost/latency).
  - Strict JSON schema; invalid/missing answers fall back to keyword scoring.
  - Never invents corpus rules — only judges whether the draft addresses a given rule.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from govgrant.compliance.checklist import ChecklistItem


class SupportsComplete(Protocol):
    available: bool

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class DraftJudgment:
    item_id: str
    status: str  # draft_ok | draft_gap
    rationale: str
    method: str  # llm | keyword


_SYSTEM = (
    "You are a careful SBIR/STTR proposal compliance reviewer.\n"
    "For each control, decide if the PROPOSAL DRAFT clearly addresses the rule.\n"
    "Rules:\n"
    "1. Use ONLY the draft text. Do not assume missing content is present.\n"
    "2. draft_ok = the draft explicitly discusses or satisfies the control.\n"
    "3. draft_gap = the draft is silent, contradictory, or only tangentially related.\n"
    "4. For forbidden patterns (e.g. hyperlinks when not allowed), draft_gap if present.\n"
    "5. Reply with a single JSON array only — no markdown fences, no prose.\n"
    "Schema per element: "
    '{"id":"<control id>","status":"draft_ok"|"draft_gap","rationale":"<one short sentence>"}\n'
)


def _controls_payload(items: list[ChecklistItem]) -> list[dict[str, str]]:
    return [
        {
            "id": it.id,
            "title": it.title,
            "rule": it.guidance,
            "severity": it.severity,
        }
        for it in items
    ]


def parse_llm_draft_json(raw: str) -> dict[str, DraftJudgment]:
    """Parse model output into judgments; ignore malformed entries."""
    text = (raw or "").strip()
    if not text:
        return {}
    # Strip accidental fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Extract outermost array if model added chatter
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, list):
        return {}
    out: dict[str, DraftJudgment] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        iid = str(row.get("id") or "").strip()
        status = str(row.get("status") or "").strip().lower()
        rationale = str(row.get("rationale") or "").strip()
        if not iid or status not in {"draft_ok", "draft_gap"}:
            continue
        out[iid] = DraftJudgment(
            item_id=iid,
            status=status,
            rationale=rationale[:300],
            method="llm",
        )
    return out


def score_drafts_with_llm(
    items: list[ChecklistItem],
    draft: str,
    llm: SupportsComplete,
    *,
    max_draft_chars: int = 60_000,
) -> dict[str, DraftJudgment]:
    """
    Batch-score draft against controls using the chat LLM.

    Returns partial map: missing ids should be filled by keyword fallback.
    """
    if not items or not (draft or "").strip():
        return {}
    if not getattr(llm, "available", False):
        return {}

    draft_clip = draft[:max_draft_chars]
    if len(draft) > max_draft_chars:
        draft_clip += "\n\n[truncated]"

    user = (
        "Controls to judge:\n"
        f"{json.dumps(_controls_payload(items), ensure_ascii=False)}\n\n"
        "PROPOSAL DRAFT:\n"
        f"{draft_clip}\n\n"
        "Return JSON array with one object per control id listed above."
    )
    try:
        raw = llm.complete(
            system=_SYSTEM,
            user=user,
            temperature=0.0,
            max_tokens=2200,
        )
    except Exception:  # noqa: BLE001 — fall back to keywords
        return {}
    return parse_llm_draft_json(raw)


def merge_draft_scores(
    items: list[ChecklistItem],
    keyword_scores: dict[str, tuple[str, list[str]]],
    llm_scores: dict[str, DraftJudgment],
) -> dict[str, DraftJudgment]:
    """Prefer valid LLM judgments; otherwise keep keyword scores."""
    out: dict[str, DraftJudgment] = {}
    for it in items:
        if it.id in llm_scores:
            out[it.id] = llm_scores[it.id]
            continue
        status, hits = keyword_scores.get(it.id, ("draft_gap", []))
        rationale = (
            f"Keyword signals: {', '.join(hits[:6])}"
            if hits
            else "No clear keyword signals in draft."
        )
        out[it.id] = DraftJudgment(
            item_id=it.id,
            status=status,
            rationale=rationale,
            method="keyword",
        )
    return out
