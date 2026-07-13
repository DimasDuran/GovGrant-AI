"""
Multi-agency compliance checklist + optional draft scoring.

Packages:
  - darpa: DARPA Phase II instructions
  - sba: SBA SBIR/STTR Policy Directive
  - sf424: NIH SF424 (R&R) Application Guide

Modes:
  - corpus: is the rule grounded in the indexed instructions?
  - draft: does the user's pasted proposal text address the control?
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from govgrant.rag.config import Settings, get_settings
from govgrant.rag.eval.runner import _fact_present
from govgrant.rag.index.hybrid import HybridRAGService

Status = Literal["pass", "fail", "warn", "info", "unknown", "draft_ok", "draft_gap"]

DOC_DARPA = "darpa-sbir-sttr-phase-II-instructions"
DOC_SBA = "SBA SBIR_STTR_POLICY_DIRECTIVE_May2023"
DOC_SF424 = "SF424 SBIR_STTR Application Guide"

PACKAGES = ("darpa", "sba", "sf424")


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    package: str
    section: str
    title: str
    question: str
    facts_required: list[str]
    doc_id: str
    severity: Literal["critical", "high", "medium", "low"] = "high"
    applies_to: tuple[str, ...] = ("sbir", "sttr")
    needs_ot: bool = False
    source_pages: list[int] = field(default_factory=list)
    guidance: str = ""
    draft_signals: list[str] = field(default_factory=list)
    """Keywords/phrases that suggest the draft addresses this control."""


@dataclass
class ChecklistResult:
    id: str
    package: str
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
    draft_status: str | None = None
    draft_signals_found: list[str] = field(default_factory=list)
    draft_rationale: str | None = None
    draft_method: str | None = None  # keyword | llm

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChecklistRun:
    program: str
    use_ot: bool
    packages: list[str]
    items: list[ChecklistResult]
    summary: dict[str, int]
    draft_provided: bool = False
    draft_summary: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program,
            "use_ot": self.use_ot,
            "packages": self.packages,
            "draft_provided": self.draft_provided,
            "summary": self.summary,
            "draft_summary": self.draft_summary,
            "items": [i.to_dict() for i in self.items],
        }

    def to_markdown(self) -> str:
        icon = {
            "pass": "✅",
            "fail": "❌",
            "warn": "⚠️",
            "info": "\u2139\ufe0f",
            "unknown": "❓",
            "draft_ok": "📝✅",
            "draft_gap": "📝⚠️",
        }
        lines = [
            "# Compliance checklist",
            "",
            f"**Program:** `{self.program.upper()}` · "
            f"**OT:** `{'yes' if self.use_ot else 'no'}` · "
            f"**Packages:** `{', '.join(self.packages)}`",
            "",
            "## Corpus coverage (instruction rules found)",
            "",
            "| Status | Count |",
            "|--------|------:|",
        ]
        for k in ("pass", "fail", "warn", "unknown", "info"):
            if self.summary.get(k):
                lines.append(f"| {icon.get(k, k)} {k} | {self.summary[k]} |")
        if self.draft_provided and self.draft_summary:
            lines += [
                "",
                "## Draft coverage (your proposal text)",
                "",
                "| Status | Count |",
                "|--------|------:|",
            ]
            for k, label in (
                ("draft_ok", "addressed in draft"),
                ("draft_gap", "not found in draft"),
            ):
                if self.draft_summary.get(k):
                    lines.append(f"| {icon.get(k, k)} {label} | {self.draft_summary[k]} |")
        lines.append("")
        current = ""
        for it in self.items:
            head = f"{it.package.upper()} · {it.section}"
            if head != current:
                current = head
                lines.append(f"## {head}")
                lines.append("")
            lines.append(f"### {icon.get(it.status, '•')} {it.title} (`{it.id}` · {it.severity})")
            lines.append(f"- **Corpus status:** {it.status}")
            if it.draft_status:
                method = f" · {it.draft_method}" if it.draft_method else ""
                lines.append(f"- **Draft status:** {it.draft_status}{method}")
                if it.draft_rationale:
                    lines.append(f"- **Draft note:** {it.draft_rationale}")
                if it.draft_signals_found:
                    lines.append(f"- **Draft signals:** {', '.join(it.draft_signals_found)}")
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
            "_Corpus mode verifies that rules exist in indexed instructions. "
            "Draft mode checks whether your pasted text appears to address each "
            "control (keyword signals — not a legal determination)._"
        )
        return "\n".join(lines)


def darpa_items() -> list[ChecklistItem]:
    d = DOC_DARPA
    return [
        ChecklistItem(
            id="DARPA-WS-SBIR",
            package="darpa",
            section="Work-share",
            title="SBIR Phase II in-house minimum",
            question="SBIR Phase II minimum one-half research work proposer direct indirect costs",
            facts_required=["one-half", "SBIR"],
            doc_id=d,
            severity="critical",
            applies_to=("sbir",),
            source_pages=[8],
            guidance="SBIR: proposer must perform ≥ one-half (50%) of Phase II work (direct+indirect).",
            draft_signals=[
                "50%",
                "one-half",
                "work-share",
                "work share",
                "in-house",
                "proposer will perform",
            ],
        ),
        ChecklistItem(
            id="DARPA-WS-STTR",
            package="darpa",
            section="Work-share",
            title="STTR Phase II proposer + RI minima",
            question="STTR Phase II 40% proposer 30% Research Institution",
            facts_required=["40%", "30%", "Research Institution"],
            doc_id=d,
            severity="critical",
            applies_to=("sttr",),
            source_pages=[9],
            guidance="STTR: ≥40% proposer and ≥30% Research Institution.",
            draft_signals=["40%", "30%", "research institution", "STTR"],
        ),
        ChecklistItem(
            id="DARPA-FFRDC",
            package="darpa",
            section="Subcontractors",
            title="FFRDC / Federal Laboratory use",
            question="FFRDC Federal Laboratory waiver Cover Sheet cannot send funding directly",
            facts_required=["Cover Sheet", "waiver"],
            doc_id=d,
            severity="high",
            source_pages=[8, 9],
            guidance="FFRDC/Lab allowed without waiver if Cover Sheet certified; no direct Agency funding to lab.",
            draft_signals=["FFRDC", "federal laboratory", "cover sheet", "subcontract"],
        ),
        ChecklistItem(
            id="DARPA-SIMILAR",
            package="darpa",
            section="Prior / similar proposals",
            title="Essentially equivalent proposals / awards",
            question="essentially equivalent work permissible unlawful disclose before award",
            facts_required=["permissible", "unlawful", "before award"],
            doc_id=d,
            severity="critical",
            source_pages=[9],
            guidance="Equivalent submissions OK; dual awards for equivalent effort unlawful; disclose before award.",
            draft_signals=[
                "essentially equivalent",
                "similar proposal",
                "other agency",
                "disclosure",
            ],
        ),
        ChecklistItem(
            id="DARPA-TECH-FMT",
            package="darpa",
            section="Technical Volume",
            title="Technical Volume format & page limit",
            question="Technical Volume 20 pages 10-point font one-inch margins",
            facts_required=["20 pages", "10-point"],
            doc_id=d,
            severity="high",
            source_pages=[6],
            guidance="Default max 20 pages; ≥10-point type; 1-inch margins; no encryption/active media.",
            draft_signals=["technical volume", "20 page", "page limit"],
        ),
        ChecklistItem(
            id="DARPA-COST-MAX",
            package="darpa",
            section="Cost Volume",
            title="Cost Volume maximum amount & duration",
            question="Phase II Cost Volume maximum 1800000 36 months",
            facts_required=["1,800,000", "36 months"],
            doc_id=d,
            severity="critical",
            source_pages=[10],
            guidance="Max $1,800,000 and 36 months including Options.",
            draft_signals=["1,800,000", "1800000", "1.8", "36 month", "cost volume"],
        ),
        ChecklistItem(
            id="DARPA-COMM",
            package="darpa",
            section="Commercialization",
            title="Transition & Commercialization Strategy",
            question="Transition Commercialization Strategy Volume 2 5 pages NOT count",
            facts_required=["5 pages", "Volume 2", "NOT count"],
            doc_id=d,
            severity="high",
            source_pages=[9],
            guidance="Strategy at end of Volume 2, max 5 pages, does not count against page limit.",
            draft_signals=["commercialization strategy", "transition", "volume 2", "5 page"],
        ),
        ChecklistItem(
            id="DARPA-OT",
            package="darpa",
            section="Other Transaction",
            title="OT Milestone Plan required fields",
            question="Other Transaction Milestone Plan description exit criteria due date payment data rights",
            facts_required=["Milestone description", "Due date", "data rights"],
            doc_id=d,
            severity="critical",
            needs_ot=True,
            source_pages=[9],
            guidance="OT requires Milestone Plan: description, exit criteria, due date, payment, data rights.",
            draft_signals=["milestone", "other transaction", "OT", "exit criteria", "data rights"],
        ),
        ChecklistItem(
            id="DARPA-CLASSIFIED",
            package="darpa",
            section="Security",
            title="Classified proposals not accepted",
            question="Classified proposals are not accepted DoW SBIR STTR",
            facts_required=["Classified proposals are not accepted"],
            doc_id=d,
            severity="critical",
            source_pages=[12],
            guidance="Classified proposals are not accepted under DoW SBIR/STTR.",
            draft_signals=["unclassified", "classified", "security"],
        ),
    ]


def sba_items() -> list[ChecklistItem]:
    d = DOC_SBA
    return [
        ChecklistItem(
            id="SBA-WS-SBIR-I",
            package="sba",
            section="Performance of work",
            title="SBIR Phase I work minimum (two-thirds)",
            question="SBIR Phase I at least two-thirds 66 2/3 percent of the research",
            facts_required=["two-thirds", "Phase I"],
            doc_id=d,
            severity="critical",
            applies_to=("sbir",),
            source_pages=[120],
            guidance="SBA Policy Directive: SBIR Phase I awardee performs at least two-thirds (66 2/3%) of the research.",
            draft_signals=["two-thirds", "66", "phase i", "2/3"],
        ),
        ChecklistItem(
            id="SBA-WS-SBIR-II",
            package="sba",
            section="Performance of work",
            title="SBIR Phase II work minimum (half)",
            question="SBIR Phase II at least half 50% of the research",
            facts_required=["half", "50%", "Phase II"],
            doc_id=d,
            severity="critical",
            applies_to=("sbir",),
            source_pages=[120],
            guidance="SBA Policy Directive: SBIR Phase II awardee performs at least half (50%) of the research.",
            draft_signals=["50%", "half", "phase ii", "work"],
        ),
        ChecklistItem(
            id="SBA-WS-STTR",
            package="sba",
            section="Performance of work",
            title="STTR work minimum (40%)",
            question="STTR Phase I or Phase II at least forty percent 40% of the research",
            facts_required=["forty percent", "40%"],
            doc_id=d,
            severity="critical",
            applies_to=("sttr",),
            source_pages=[120],
            guidance="SBA Policy Directive: STTR Phase I or II awardee performs at least forty percent (40%) of the research.",
            draft_signals=["40%", "forty", "STTR", "research institution"],
        ),
        ChecklistItem(
            id="SBA-EQUIV",
            package="sba",
            section="Definitions / integrity",
            title="Essentially Equivalent Work definition",
            question="Essentially Equivalent Work substantially the same research more than one proposal",
            facts_required=["Essentially Equivalent Work", "substantially the same"],
            doc_id=d,
            severity="high",
            source_pages=[10],
            guidance="Essentially Equivalent Work = substantially the same research proposed for funding in more than one application.",
            draft_signals=["essentially equivalent", "similar work", "duplicate", "other agency"],
        ),
        ChecklistItem(
            id="SBA-FOREIGN",
            package="sba",
            section="Foreign disclosure",
            title="Foreign affiliations / countries of concern disclosures",
            question="disclosure foreign country of concern malign foreign talent recruitment SBIR.gov foreign_disclosures",
            facts_required=["countries of concern", "foreign"],
            doc_id=d,
            severity="critical",
            source_pages=[156],
            guidance=(
                "Applicants must answer foreign disclosure questions (malign talent programs, "
                "parents/subsidiaries in countries of concern, etc.). List on SBIR.gov/foreign_disclosures."
            ),
            draft_signals=[
                "foreign",
                "country of concern",
                "malign",
                "disclosure",
                "affiliation",
                "ownership",
            ],
        ),
        ChecklistItem(
            id="SBA-US-PERF",
            package="sba",
            section="Performance location",
            title="R/R&D performed in the United States",
            question="R/R&D will be performed in the United States unless deviation approved",
            facts_required=["United States", "deviation"],
            doc_id=d,
            severity="high",
            source_pages=[120],
            guidance="R/R&D must be performed in the United States unless a written deviation is approved.",
            draft_signals=["united states", "U.S.", "domestic", "performance location"],
        ),
    ]


def sf424_items() -> list[ChecklistItem]:
    d = DOC_SF424
    return [
        ChecklistItem(
            id="SF424-RS-REQUIRED",
            package="sf424",
            section="Research Plan",
            title="Research Strategy attachment required",
            question="Research Strategy attachment is required PHS 398 page limits NIH Table",
            facts_required=["Research Strategy", "required"],
            doc_id=d,
            severity="critical",
            source_pages=[149, 210],
            guidance="PHS 398 Research Strategy attachment is required; follow NIH Table of Page Limits (or FOA).",
            draft_signals=["research strategy", "significance", "innovation", "approach"],
        ),
        ChecklistItem(
            id="SF424-AIMS",
            package="sf424",
            section="Research Plan",
            title="Specific Aims required + page limits",
            question="Specific Aims attachment required page limits NIH Table of Page Limits",
            facts_required=["Specific Aims", "page limit"],
            doc_id=d,
            severity="critical",
            source_pages=[148, 169],
            guidance="Specific Aims attachment required unless FOA says otherwise; exceeding page limit is an error.",
            draft_signals=["specific aims", "aim 1", "objectives"],
        ),
        ChecklistItem(
            id="SF424-PROGRESS",
            package="sf424",
            section="Research Plan",
            title="Progress Report counts toward Research Strategy pages",
            question="Progress Report falls within the Research Strategy page limits",
            facts_required=["Progress Report", "page limits", "Research Strategy"],
            doc_id=d,
            severity="high",
            source_pages=[153],
            guidance="Progress Report is inside Research Strategy and counts against its page limits.",
            draft_signals=["progress report", "previous project", "renewal"],
        ),
        ChecklistItem(
            id="SF424-UEI",
            package="sf424",
            section="SF424 R&R Form",
            title="Unique Entity Identifier (UEI) required",
            question="Unique Entity Identifier UEI required replaced DUNS applicant organization",
            facts_required=["UEI", "required"],
            doc_id=d,
            severity="critical",
            source_pages=[11, 32],
            guidance="UEI replaced DUNS; UEI field is required and must match eRA Commons IPF.",
            draft_signals=["UEI", "unique entity", "SAM"],
        ),
        ChecklistItem(
            id="SF424-INDIRECT",
            package="sf424",
            section="Budget",
            title="Indirect costs fields on budget form",
            question="Indirect Cost Type Rate established cognizant federal office budget",
            facts_required=["Indirect Cost", "Rate"],
            doc_id=d,
            severity="high",
            source_pages=[139],
            guidance="Budget form requires indirect cost type/rate (cognizant federal office or for-profit guidance).",
            draft_signals=["indirect cost", "F&A", "overhead", "cognizant"],
        ),
        ChecklistItem(
            id="SF424-AUTH-RESOURCES",
            package="sf424",
            section="Other attachments",
            title="Authentication of key biological/chemical resources",
            question="Authentication of Key Biological and/or Chemical Resources identity validity",
            facts_required=["Authentication", "biological", "chemical"],
            doc_id=d,
            severity="medium",
            source_pages=[222],
            guidance="If applicable, attach methods ensuring identity/validity of key biological/chemical resources.",
            draft_signals=["authentication", "biological", "chemical resources", "validity"],
        ),
        ChecklistItem(
            id="SF424-HYPERLINKS",
            package="sf424",
            section="Formatting",
            title="Hyperlinks generally not allowed in Research Strategy",
            question="Use of hyperlinks and URLs is not allowed unless specified in the FOA Research Strategy",
            facts_required=["hyperlinks", "not allowed"],
            doc_id=d,
            severity="medium",
            source_pages=[149, 210],
            guidance="Hyperlinks/URLs not allowed in Research Strategy unless the FOA explicitly permits them.",
            draft_signals=["http", "www.", "https://"],
        ),
    ]


def all_items() -> list[ChecklistItem]:
    return darpa_items() + sba_items() + sf424_items()


def _extract_citation(evidence: str) -> str:
    pages = sorted({int(m) for m in re.findall(r"\bpage[=:\s]+(\d+)\b", evidence, re.I)})
    files = sorted(set(re.findall(r"file=([^\s|]+)", evidence)))
    parts = []
    if files:
        parts.append(files[0].split("/")[-1])
    if pages:
        parts.append("p." + ",".join(str(p) for p in pages[:6]))
    return " · ".join(parts) if parts else ""


def _score_corpus(item: ChecklistItem, evidence: str) -> tuple[Status, list[str], list[str], str]:
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
    return status, found, missing, detail


def _score_draft(item: ChecklistItem, draft: str) -> tuple[str, list[str]]:
    if not draft.strip():
        return "draft_gap", []
    signals = item.draft_signals or item.facts_required
    hits = [s for s in signals if _fact_present(draft, s)]
    # Special case: hyperlinks control — presence of URLs is a GAP
    if item.id == "SF424-HYPERLINKS":
        if hits:
            return "draft_gap", hits  # found forbidden hyperlinks
        return "draft_ok", []
    # Soft signal match: at least 2 hits, or 1 if the control has few signals
    need = 2 if len(signals) >= 3 else 1
    if len(hits) >= need:
        return "draft_ok", hits
    return "draft_gap", hits


def run_checklist(
    *,
    program: str = "sbir",
    use_ot: bool = False,
    packages: list[str] | None = None,
    draft_text: str | None = None,
    tenant_id: str | None = None,
    docs: HybridRAGService | None = None,
    settings: Settings | None = None,
    top_k: int = 6,
    use_llm_draft: bool = False,
) -> ChecklistRun:
    """
    Run multi-package compliance checklist.

    packages: subset of darpa|sba|sf424 (default all three)
    draft_text: optional proposal draft for draft_ok / draft_gap scoring
    use_llm_draft: if True and ChatLLM is available, batch-judge draft with Haiku
                   (falls back to keyword signals per control)
    """
    program = (program or "sbir").strip().lower()
    if program not in {"sbir", "sttr"}:
        program = "sbir"
    packages = [p.lower() for p in (packages or list(PACKAGES)) if p.lower() in PACKAGES]
    if not packages:
        packages = list(PACKAGES)

    settings = settings or get_settings()
    docs = docs or HybridRAGService(settings)
    tenant_id = tenant_id or settings.default_tenant_id
    draft = draft_text or ""
    draft_provided = bool(draft.strip())

    selected: list[ChecklistItem] = []
    for item in all_items():
        if item.package not in packages:
            continue
        if program not in item.applies_to:
            continue
        if item.needs_ot and not use_ot:
            continue
        selected.append(item)

    # Optional batched LLM draft judgments (one call)
    llm_map: dict[str, Any] = {}
    if draft_provided and use_llm_draft:
        try:
            from govgrant.agent.llm import ChatLLM
            from govgrant.compliance.draft_llm import (
                merge_draft_scores,
                score_drafts_with_llm,
            )

            llm = ChatLLM(settings)
            keyword_scores = {it.id: _score_draft(it, draft) for it in selected}
            llm_raw = score_drafts_with_llm(selected, draft, llm)
            llm_map = merge_draft_scores(selected, keyword_scores, llm_raw)
        except Exception:  # noqa: BLE001
            llm_map = {}

    results: list[ChecklistResult] = []
    for item in selected:
        hits = docs.retrieve(
            item.question,
            tenant_id=tenant_id,
            doc_id=item.doc_id,
            top_k=top_k,
        )
        evidence = docs.format_hits(hits)
        status, found, missing, detail = _score_corpus(item, evidence)

        draft_status = None
        draft_signals_found: list[str] = []
        draft_rationale = None
        draft_method = None
        if draft_provided:
            if item.id in llm_map:
                judgment = llm_map[item.id]
                draft_status = judgment.status
                draft_rationale = judgment.rationale
                draft_method = judgment.method
                draft_signals_found = (
                    [s for s in (item.draft_signals or []) if _fact_present(draft, s)]
                    if judgment.method == "keyword"
                    else []
                )
            else:
                draft_status, draft_signals_found = _score_draft(item, draft)
                draft_method = "keyword"
                draft_rationale = (
                    f"Keyword signals: {', '.join(draft_signals_found[:6])}"
                    if draft_signals_found
                    else "No clear keyword signals in draft."
                )
            if draft_status == "draft_gap":
                detail += " Draft does not clearly address this control."
            elif draft_status == "draft_ok":
                detail += " Draft appears to address this control."
            if draft_rationale:
                detail += f" ({draft_method}: {draft_rationale})"

        results.append(
            ChecklistResult(
                id=item.id,
                package=item.package,
                section=item.section,
                title=item.title,
                status=status,
                severity=item.severity,
                guidance=item.guidance,
                evidence_hits=len(found),
                facts_found=found,
                facts_missing=missing,
                citation=_extract_citation(evidence),
                detail=detail,
                source_pages=list(item.source_pages),
                draft_status=draft_status,
                draft_signals_found=draft_signals_found,
                draft_rationale=draft_rationale,
                draft_method=draft_method,
            )
        )

    summary: dict[str, int] = {}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1

    draft_summary: dict[str, int] | None = None
    if draft_provided:
        draft_summary = {}
        for r in results:
            if r.draft_status:
                draft_summary[r.draft_status] = draft_summary.get(r.draft_status, 0) + 1

    return ChecklistRun(
        program=program,
        use_ot=use_ot,
        packages=packages,
        items=results,
        summary=summary,
        draft_provided=draft_provided,
        draft_summary=draft_summary,
    )


# Back-compat alias used by CLI/UI
def run_darpa_phase2_checklist(
    *,
    program: str = "sbir",
    use_ot: bool = False,
    doc_id: str = DOC_DARPA,
    tenant_id: str | None = None,
    docs: HybridRAGService | None = None,
    settings: Settings | None = None,
    top_k: int = 6,
    draft_text: str | None = None,
) -> ChecklistRun:
    """Legacy entrypoint — DARPA package only (doc_id ignored if non-default)."""
    packages = ["darpa"]
    # If caller passed SBA/SF424 doc_id, map package
    if doc_id and "SBA" in doc_id:
        packages = ["sba"]
    elif doc_id and "SF424" in doc_id:
        packages = ["sf424"]
    return run_checklist(
        program=program,
        use_ot=use_ot,
        packages=packages,
        draft_text=draft_text,
        tenant_id=tenant_id,
        docs=docs,
        settings=settings,
        top_k=top_k,
    )
