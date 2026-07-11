"""
DARPA Phase II compliance checklist.

Uses the same hybrid retrieval stack as chat, but grades fixed control points
(work-share, FFRDC, similar proposals, OT milestones, commercialization, etc.)
against retrieved evidence — the same fact model as the golden dataset.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from govgrant.rag.config import Settings, get_settings
from govgrant.rag.eval.runner import _fact_present
from govgrant.rag.index.hybrid import HybridRAGService

Status = Literal["pass", "fail", "warn", "info", "unknown"]

DOC_ID = "darpa-sbir-sttr-phase-II-instructions"


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    section: str
    title: str
    question: str
    """Retrieval query for this control."""
    facts_required: list[str]
    """Must appear in evidence for 'covered'."""
    severity: Literal["critical", "high", "medium", "low"] = "high"
    applies_to: tuple[str, ...] = ("sbir", "sttr")
    """Program types this item applies to."""
    needs_ot: bool = False
    """Only when user requests OT."""
    source_pages: list[int] = field(default_factory=list)
    guidance: str = ""
    """Human-readable rule summary shown in UI."""


@dataclass
class ChecklistResult:
    id: str
    section: str
    title: str
    status: Status
    severity: str
    guidance: str
    evidence_hits: int
    facts_found: list[str]
    facts_missing: list[str]
    citation: str
    detail: str
    source_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChecklistRun:
    program: str
    use_ot: bool
    doc_id: str
    items: list[ChecklistResult]
    summary: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program,
            "use_ot": self.use_ot,
            "doc_id": self.doc_id,
            "summary": self.summary,
            "items": [i.to_dict() for i in self.items],
        }

    def to_markdown(self) -> str:
        icon = {
            "pass": "✅",
            "fail": "❌",
            "warn": "⚠️",
            "info": "ℹ️",
            "unknown": "❓",
        }
        lines = [
            f"# DARPA Phase II compliance checklist",
            f"",
            f"**Program:** `{self.program.upper()}` · "
            f"**OT requested:** `{'yes' if self.use_ot else 'no'}` · "
            f"**Doc:** `{self.doc_id}`",
            f"",
            f"| Status | Count |",
            f"|--------|------:|",
        ]
        for k in ("pass", "fail", "warn", "unknown", "info"):
            if self.summary.get(k):
                lines.append(f"| {icon.get(k, k)} {k} | {self.summary[k]} |")
        lines.append("")
        current_section = ""
        for it in self.items:
            if it.section != current_section:
                current_section = it.section
                lines.append(f"## {current_section}")
                lines.append("")
            lines.append(
                f"### {icon.get(it.status, '•')} {it.title} "
                f"(`{it.id}` · {it.severity})"
            )
            lines.append(f"- **Status:** {it.status}")
            lines.append(f"- **Rule:** {it.guidance}")
            if it.facts_found:
                lines.append(f"- **Found in corpus:** {', '.join(it.facts_found)}")
            if it.facts_missing:
                lines.append(f"- **Missing from retrieve:** {', '.join(it.facts_missing)}")
            if it.citation:
                lines.append(f"- **Citation:** {it.citation}")
            if it.detail:
                lines.append(f"- **Note:** {it.detail}")
            lines.append("")
        lines.append(
            "_This checklist verifies that the **instruction corpus** covers each "
            "control and surfaces the rule. It does not grade your draft proposal text "
            "unless you paste it into Chat with a cross-check question._"
        )
        return "\n".join(lines)


def darpa_phase2_items() -> list[ChecklistItem]:
    return [
        ChecklistItem(
            id="WS-SBIR",
            section="Work-share",
            title="SBIR Phase II in-house minimum",
            question=(
                "SBIR Phase II minimum percentage of research analytical work "
                "performed by the proposer one-half direct and indirect costs"
            ),
            facts_required=["one-half", "SBIR"],
            severity="critical",
            applies_to=("sbir",),
            source_pages=[8],
            guidance=(
                "SBIR: proposer must perform at least one-half (50%) of Phase II "
                "research/analytical work (direct + indirect costs), unless Contracting "
                "Officer approves otherwise in writing."
            ),
        ),
        ChecklistItem(
            id="WS-STTR",
            section="Work-share",
            title="STTR Phase II proposer + RI minima",
            question=(
                "STTR Phase II minimum 40% proposer 30% Research Institution work share"
            ),
            facts_required=["40%", "30%", "Research Institution"],
            severity="critical",
            applies_to=("sttr",),
            source_pages=[9],
            guidance=(
                "STTR: ≥40% by the proposer and ≥30% by the Research Institution (RI), "
                "measured by direct and indirect costs."
            ),
        ),
        ChecklistItem(
            id="FFRDC",
            section="Subcontractors",
            title="FFRDC / Federal Laboratory use",
            question=(
                "FFRDC Federal Laboratory subcontract waiver Cover Sheet certification "
                "cannot send SBIR STTR funding directly"
            ),
            facts_required=["Cover Sheet", "waiver"],
            severity="high",
            applies_to=("sbir", "sttr"),
            source_pages=[8, 9],
            guidance=(
                "Federal Labs/FFRDCs allowed without a waiver if certified on the Cover "
                "Sheet. Agency cannot send funding directly to the lab/FFRDC; SBC contracts "
                "with them. Other federal organizations: not permitted."
            ),
        ),
        ChecklistItem(
            id="FED-ORG",
            section="Subcontractors",
            title="No subcontracts with other federal organizations",
            question=(
                "subcontracts with other federal organizations are not permitted SBIR STTR"
            ),
            facts_required=["not permitted", "federal"],
            severity="high",
            applies_to=("sbir", "sttr"),
            source_pages=[8, 9],
            guidance="Subcontracts with other federal organizations are not permitted.",
        ),
        ChecklistItem(
            id="SIMILAR",
            section="Prior / similar proposals",
            title="Essentially equivalent proposals / awards",
            question=(
                "Prior Current Pending Support similar proposals essentially equivalent "
                "work permissible unlawful disclose before award"
            ),
            facts_required=["permissible", "unlawful", "before award"],
            severity="critical",
            applies_to=("sbir", "sttr"),
            source_pages=[9],
            guidance=(
                "Submitting essentially equivalent proposals is permissible; awards "
                "requiring essentially equivalent effort are unlawful. Disclose any "
                "question to the soliciting agency **before award**."
            ),
        ),
        ChecklistItem(
            id="TECH-FMT",
            section="Technical Volume",
            title="Technical Volume format & page limit",
            question=(
                "Technical Volume 20 pages 10-point font one-inch margins page 1 "
                "marketing material not evaluated encrypt"
            ),
            facts_required=["20 pages", "10-point"],
            severity="high",
            applies_to=("sbir", "sttr"),
            source_pages=[6],
            guidance=(
                "Default max 20 pages (or topic instructions). ≥10-point type, 8.5×11, "
                "1-inch margins, consecutive pages from page 1. No marketing material, "
                "no encryption/password, no embedded video/active media."
            ),
        ),
        ChecklistItem(
            id="COST-MAX",
            section="Cost Volume",
            title="Cost Volume maximum amount & duration",
            question=(
                "Phase II Cost Volume maximum 1800000 36 months typical 1000000 base "
                "800000 option Cost Volume template Excel"
            ),
            facts_required=["1,800,000", "36 months"],
            severity="critical",
            applies_to=("sbir", "sttr"),
            source_pages=[10],
            guidance=(
                "Max $1,800,000 and 36 months including Options. Typical structure "
                "$1,000,000 Base (18–24 mo) + $800,000 Option (6–12 mo). Must use DARPA "
                "Cost Volume Excel template."
            ),
        ),
        ChecklistItem(
            id="COST-SHARE",
            section="Cost Volume",
            title="Cost sharing not required / not scored",
            question=(
                "cost sharing is permitted but not required nor evaluation factor Phase II"
            ),
            facts_required=["not required", "evaluation factor"],
            severity="medium",
            applies_to=("sbir", "sttr"),
            source_pages=[10],
            guidance=(
                "Cost sharing is permitted but not required and is not an evaluation factor."
            ),
        ),
        ChecklistItem(
            id="COMM-STRAT",
            section="Commercialization",
            title="Transition & Commercialization Strategy placement",
            question=(
                "Transition and Commercialization Strategy end of Technical Volume "
                "Volume 2 should not exceed 5 pages will NOT count against page limit"
            ),
            facts_required=["5 pages", "Volume 2", "NOT count"],
            severity="high",
            applies_to=("sbir", "sttr"),
            source_pages=[9],
            guidance=(
                "Include at end of Volume 2 (Technical Volume), max 5 pages, does NOT "
                "count against the proposal page limit. Not the same as TCSP agency program."
            ),
        ),
        ChecklistItem(
            id="COMM-LETTERS",
            section="Commercialization",
            title="Optional Advocacy / LOI letters",
            question=(
                "Advocacy Letters Letters of Intent Commitment optional do NOT count "
                "against page limit commercialization claims"
            ),
            facts_required=["Advocacy Letters", "optional", "do NOT count"],
            severity="medium",
            applies_to=("sbir", "sttr"),
            source_pages=[9],
            guidance=(
                "Advocacy Letters and Letters of Intent/Commitment are optional, do not "
                "count against page limits, and should only substantiate commercialization "
                "claims. Government endorsement letters are not accepted (ethics rules)."
            ),
        ),
        ChecklistItem(
            id="OT-MILESTONES",
            section="Other Transaction",
            title="OT Milestone Plan required fields",
            question=(
                "Other Transaction OT Milestone Plan each milestone must include "
                "description Completion Exit criteria Due date Payment funding schedule "
                "Government data rights"
            ),
            facts_required=[
                "Milestone description",
                "Due date",
                "data rights",
            ],
            severity="critical",
            applies_to=("sbir", "sttr"),
            needs_ot=True,
            source_pages=[9],
            guidance=(
                "OT requesters must include a detailed Milestone Plan. Each milestone: "
                "description, completion/exit criteria, due date, payment/funding schedule, "
                "Government data rights per data deliverable. No proprietary data in plan."
            ),
        ),
        ChecklistItem(
            id="CLASSIFIED",
            section="Security / submission",
            title="Classified proposals not accepted",
            question=(
                "Classified proposals are not accepted under the DoW SBIR STTR Program"
            ),
            facts_required=["Classified proposals are not accepted"],
            severity="critical",
            applies_to=("sbir", "sttr"),
            source_pages=[12],
            guidance="Classified proposals are not accepted under the DoW SBIR/STTR Program.",
        ),
        ChecklistItem(
            id="VOLUMES",
            section="Proposal package",
            title="Complete Phase II volume set",
            question=(
                "complete Phase II proposal Volume 1 Cover Sheet Volume 2 Technical "
                "Volume 3 Cost Volume 4 Company Commercialization Report Volume 5 "
                "Supporting Documents Volume 6 Fraud Volume 7 Foreign Affiliations"
            ),
            facts_required=["Volume 1", "Volume 2", "Volume 3", "Volume 4"],
            severity="high",
            applies_to=("sbir", "sttr"),
            source_pages=[5],
            guidance=(
                "A complete package includes Volumes 1–7 (Cover, Technical, Cost, CCR, "
                "Supporting Docs, FWA training, Foreign Affiliations disclosures)."
            ),
        ),
        ChecklistItem(
            id="TCSP",
            section="Post-award support",
            title="TCSP available at no cost after award",
            question=(
                "Transition and Commercialization Support Program TCSP Phase II awardees "
                "at no cost upon contract execution"
            ),
            facts_required=["TCSP", "no cost"],
            severity="low",
            applies_to=("sbir", "sttr"),
            source_pages=[5],
            guidance=(
                "DARPA provides TCSP to Phase II awardees upon contract execution at no cost "
                "(agency support program — distinct from the 5-page strategy in Volume 2)."
            ),
        ),
    ]


def _extract_citation(evidence: str) -> str:
    pages = sorted({int(m) for m in re.findall(r"\bpage[=:\s]+(\d+)\b", evidence, re.I)})
    files = sorted(set(re.findall(r"file=([^\s|]+)", evidence)))
    parts = []
    if files:
        parts.append(files[0].split("/")[-1])
    if pages:
        parts.append("p." + ",".join(str(p) for p in pages[:6]))
    return " · ".join(parts) if parts else ""


def _score_item(
    item: ChecklistItem,
    evidence: str,
    *,
    program: str,
) -> ChecklistResult:
    found: list[str] = []
    missing: list[str] = []
    for fact in item.facts_required:
        if _fact_present(evidence, fact):
            found.append(fact)
        else:
            missing.append(fact)

    covered = len(found)
    need = max(1, int(len(item.facts_required) * 0.6 + 0.999))
    has_hits = bool(re.search(r"score=", evidence or "")) or len(evidence) > 80

    if covered >= need:
        status: Status = "pass"
        detail = "Instruction corpus covers this control."
    elif covered > 0:
        status = "warn"
        detail = "Partially covered in retrieve pack — review citation."
    elif has_hits:
        status = "unknown"
        detail = "Retrieve returned text but required facts not matched."
    else:
        status = "fail"
        detail = "No usable evidence retrieved for this control."

    # Program-specific emphasis in detail
    if item.id == "WS-SBIR" and program == "sbir" and status == "pass":
        detail = "SBIR work-share rule is present in instructions (apply ≥50%/one-half)."
    if item.id == "WS-STTR" and program == "sttr" and status == "pass":
        detail = "STTR work-share rule is present (apply ≥40% proposer / ≥30% RI)."

    return ChecklistResult(
        id=item.id,
        section=item.section,
        title=item.title,
        status=status,
        severity=item.severity,
        guidance=item.guidance,
        evidence_hits=covered,
        facts_found=found,
        facts_missing=missing,
        citation=_extract_citation(evidence),
        detail=detail,
        source_pages=list(item.source_pages),
    )


def run_darpa_phase2_checklist(
    *,
    program: str = "sbir",
    use_ot: bool = False,
    doc_id: str = DOC_ID,
    tenant_id: str | None = None,
    docs: HybridRAGService | None = None,
    settings: Settings | None = None,
    top_k: int = 6,
) -> ChecklistRun:
    """
    Run checklist against indexed instructions (not against a user proposal draft).

    program: 'sbir' | 'sttr'
    use_ot: include Other Transaction milestone controls
    """
    program = (program or "sbir").strip().lower()
    if program not in {"sbir", "sttr"}:
        program = "sbir"

    settings = settings or get_settings()
    docs = docs or HybridRAGService(settings)
    tenant_id = tenant_id or settings.default_tenant_id

    results: list[ChecklistResult] = []
    for item in darpa_phase2_items():
        if program not in item.applies_to:
            continue
        if item.needs_ot and not use_ot:
            continue
        hits = docs.retrieve(
            item.question,
            tenant_id=tenant_id,
            doc_id=doc_id,
            top_k=top_k,
        )
        evidence = docs.format_hits(hits)
        results.append(_score_item(item, evidence, program=program))

    summary: dict[str, int] = {}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1

    return ChecklistRun(
        program=program,
        use_ot=use_ot,
        doc_id=doc_id,
        items=results,
        summary=summary,
    )
