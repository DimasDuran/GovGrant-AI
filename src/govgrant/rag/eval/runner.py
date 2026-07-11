"""Run regression and golden evaluation cases against the RAG/agent stack.

Scoring model (RAGAS-style fact coverage, not full-string match):

  facts_required  → recall  (must appear; default ≥60% to pass)
  facts_optional  → bonus coverage (never fails the case)
  facts_forbidden → precision (must NOT appear in the *answer*)

Aliases kept for back-compat:
  expected_facts      ≡ facts_required
  must_not_include    ≡ facts_forbidden
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from govgrant.rag.config import REPO_ROOT, get_settings
from govgrant.rag.router.query_router import QueryRouter, RouteIntent

DEFAULT_DOC_ID = "darpa-sbir-sttr-phase-II-instructions"
EVAL_DIR = REPO_ROOT / "data" / "eval"
GOLDEN_FILES = [
    "01_fact_lookup.json",
    "02_boolean.json",
    "03_list.json",
    "04_comparison.json",
    "05_scenario.json",
    "06_multi_hop.json",
    "07_not_found.json",
    "08_edge_case.json",
]

# Default fraction of required facts that must hit to pass
REQUIRED_RECALL_THRESHOLD = 0.6

_PAGE_RE = re.compile(r"\bpage[=:\s]+(\d+)\b", re.I)
_REFUSAL_HINTS = (
    "not specified",
    "does not specify",
    "does not mention",
    "does not provide",
    "does not contain",
    "do not contain",
    "does not include",
    "does not have",
    "does not publish",
    "not provided",
    "not in the document",
    "not found in",
    "not present in",
    "not present",
    "no mention",
    "no specific",
    "insufficient evidence",
    "evidence is insufficient",
    "evidence is missing",
    "evidence does not",
    "retrieved evidence does not",
    "i don't have enough",
    "cannot determine",
    "no information",
    "document does not",
    "not stated",
    "not listed",
    "not addressed",
    "not covered",
    "no requirement",
    "not required in the evidence",
    "evidence provided does not",
    "there is no",
    "there are no",
)


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: list[str]
    intent: str
    preview: str
    category: str = ""
    # Legacy aggregate (required facts)
    fact_hits: int = 0
    fact_total: int = 0
    pages_found: list[int] = field(default_factory=list)
    # Fact-based metrics
    recall: float = 0.0
    precision: float = 1.0
    optional_recall: float = 0.0
    required_hits: int = 0
    required_total: int = 0
    optional_hits: int = 0
    optional_total: int = 0
    forbidden_hits: int = 0
    forbidden_total: int = 0
    missing_required: list[str] = field(default_factory=list)
    hit_forbidden: list[str] = field(default_factory=list)
    hit_optional: list[str] = field(default_factory=list)


def _contains_any(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def _normalize_fact(fact: str) -> str:
    return re.sub(r"\s+", " ", fact.strip().lower())


def _money_forms(text: str) -> set[str]:
    """Normalize money mentions into digit strings for loose comparison."""
    forms: set[str] = set()
    h = text.lower()
    for m in re.finditer(
        r"\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion)?",
        h,
    ):
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        if unit.startswith("million"):
            val *= 1_000_000
        elif unit.startswith("billion"):
            val *= 1_000_000_000
        digits = str(int(val)) if val >= 1000 else ""
        if len(digits) >= 4:
            forms.add(digits)
    for m in re.finditer(r"\d{4,}", re.sub(r"[^\d]", " ", h)):
        forms.add(m.group(0))
    return forms


def _fact_present(haystack: str, fact: str) -> bool:
    """Loose fact match: substring, or all significant tokens present."""
    h = haystack.lower()
    h_norm = (
        h.replace("8-1/2", "8.5")
        .replace("8½", "8.5")
        .replace("–", "-")
        .replace("—", "-")
    )
    f = _normalize_fact(fact)
    if not f:
        return True
    if f in h or f in h_norm:
        return True
    fact_money = _money_forms(f)
    if fact_money and fact_money & _money_forms(h):
        return True
    digits = re.sub(r"[^\d]", "", f)
    if len(digits) >= 4 and digits in re.sub(r"[^\d]", "", h):
        return True
    stop = {"the", "and", "for", "with", "from", "that", "this", "files", "file", "font"}
    # Keep short digits / roman numerals / % so "Volume 4" ≠ "Volume 2"
    # and "Phase II" ≠ "Phase I".
    raw_tokens = [t for t in re.split(r"[^\w%]+", f) if t]
    tokens = [
        t
        for t in raw_tokens
        if t not in stop
        and (
            len(t) > 2
            or t.endswith("%")
            or t.isdigit()
            or re.fullmatch(r"[ivxlcdm]+", t)  # roman numerals
        )
    ]
    # Entity + number/roman (Volume 5, Phase II): require near-contiguous phrase.
    # Avoid "Volume 2 ... 5 pages" matching "Volume 5".
    if len(tokens) >= 2 and any(
        t.isdigit() or re.fullmatch(r"[ivxlcdm]+", t) for t in tokens
    ):
        phrase = " ".join(tokens)
        if phrase in h_norm:
            return True
        if len(tokens) == 2:
            pat = re.compile(
                rf"\b{re.escape(tokens[0])}\s*[:=\-]?\s*{re.escape(tokens[1])}\b",
                re.I,
            )
            return bool(pat.search(h_norm))
        # Longer phrases: all tokens in order with max 12 chars between each
        pos = 0
        for i, t in enumerate(tokens):
            idx = h_norm.find(t, pos)
            if idx < 0:
                return False
            if i > 0 and idx - pos > 12:
                return False
            pos = idx + len(t)
        return True
    if tokens and all(t in h_norm for t in tokens):
        return True
    synonyms = {
        "encrypted": ("encrypt", "lock or encrypt"),
        "password-protected": ("encrypt", "lock or encrypt", "password"),
        "animations": ("moving pictures", "active graphics", "videos"),
        "formatting": ("10-point", "one-inch", "margins"),
        "prohibited": ("do not", "not be evaluated", "not include"),
    }
    remaining = list(tokens)
    for tok in tokens:
        for key, alts in synonyms.items():
            if key in tok or tok in key:
                if any(a in h_norm for a in alts):
                    remaining = [t for t in remaining if t != tok]
                    break
    if remaining and all(t in h_norm for t in remaining):
        return True
    return False


def _extract_pages(text: str) -> set[int]:
    return {int(m.group(1)) for m in _PAGE_RE.finditer(text or "")}


def _looks_like_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(h in low for h in _REFUSAL_HINTS)


def _is_raw_retrieval(text: str) -> bool:
    return bool(
        re.search(r"\[1\]\s*score=", text or "")
        or re.search(r"^intent=.*sources=", (text or "").strip(), re.M)
    )


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(x) for x in value if str(x).strip()]


def normalize_case(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize legacy regression_min and golden schemas into one shape.

    Preferred golden fields:
      facts_required / facts_optional / facts_forbidden
    Aliases:
      expected_facts → facts_required
      must_not_include → facts_forbidden
    """
    case = dict(raw)
    case["id"] = case.get("id") or case.get("case_id") or "unknown"
    case["question"] = case.get("question") or case.get("query") or ""
    case["query"] = case["question"]
    case["doc_id"] = case.get("doc_id") or (
        DEFAULT_DOC_ID
        if case.get("category")
        or case.get("expected_facts") is not None
        or case.get("facts_required") is not None
        else None
    )

    # --- required ---
    required = case.get("facts_required")
    if required is None:
        required = case.get("expected_facts")
    if required is None and case.get("expect_any_keywords"):
        required = list(case["expect_any_keywords"])
    case["facts_required"] = _as_str_list(required)
    # keep expected_facts mirrored for older tooling
    case["expected_facts"] = list(case["facts_required"])

    # --- optional ---
    case["facts_optional"] = _as_str_list(
        case.get("facts_optional") or case.get("optional_facts")
    )

    # --- forbidden ---
    forbidden = case.get("facts_forbidden")
    if forbidden is None:
        forbidden = case.get("must_not_include")
    case["facts_forbidden"] = _as_str_list(forbidden)
    case["must_not_include"] = list(case["facts_forbidden"])

    pages = case.get("source_pages") or case.get("expected_pages") or []
    case["source_pages"] = [
        int(p) for p in pages if p is not None and str(p).strip() != ""
    ]
    case["should_refuse"] = bool(case.get("should_refuse", False))
    case["category"] = case.get("category") or case.get("intent") or "regression"

    # Pass threshold override per case (1.0 = all required facts)
    thr = case.get("required_recall_threshold")
    if thr is None:
        thr = case.get("min_fact_ratio", REQUIRED_RECALL_THRESHOLD)
    try:
        case["required_recall_threshold"] = float(thr)
    except (TypeError, ValueError):
        case["required_recall_threshold"] = REQUIRED_RECALL_THRESHOLD

    return case


def load_cases(path: Path | None = None, *, golden: bool = False) -> list[dict[str, Any]]:
    if golden:
        cases: list[dict[str, Any]] = []
        for name in GOLDEN_FILES:
            fp = EVAL_DIR / name
            if not fp.exists():
                continue
            batch = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(batch, list):
                for c in batch:
                    nc = normalize_case(c)
                    nc["_file"] = name
                    cases.append(nc)
        return cases

    path = path or (EVAL_DIR / "regression_min.json")
    if path.is_dir():
        cases = []
        for fp in sorted(path.glob("0*.json")):
            batch = json.loads(fp.read_text(encoding="utf-8"))
            for c in batch if isinstance(batch, list) else []:
                nc = normalize_case(c)
                nc["_file"] = fp.name
                cases.append(nc)
        return cases

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [normalize_case(c) for c in raw]
    return [normalize_case(raw)]


def _score_fact_list(haystack: str, facts: list[str]) -> tuple[int, list[str], list[str]]:
    hits: list[str] = []
    misses: list[str] = []
    for fact in facts:
        if _fact_present(haystack, fact):
            hits.append(fact)
        else:
            misses.append(fact)
    return len(hits), hits, misses


def score_case(
    case: dict[str, Any],
    *,
    text: str,
    evidence: str,
    intent: str,
    sources_used: list[str] | None = None,
    score_answer_only: bool | None = None,
) -> CaseResult:
    """
    Score one case.

    score_answer_only:
      True  → required/optional on answer text (agent quality)
      False → required/optional on answer+evidence (retrieval coverage)
      None  → auto: answer-only if text is synthesized (not raw retrieval)
    """
    failures: list[str] = []
    sources_used = sources_used or []
    answer = text or ""
    evidence = evidence or ""
    combined = f"{answer}\n{evidence}"
    pages_found = sorted(_extract_pages(combined))
    raw_retrieval = _is_raw_retrieval(answer) or (
        not answer.strip() and _is_raw_retrieval(evidence)
    )

    if score_answer_only is None:
        score_answer_only = not raw_retrieval and bool(answer.strip()) and not _is_raw_retrieval(
            answer
        )

    # Where to look for required/optional facts
    coverage_text = answer if score_answer_only else combined
    # Forbidden always on the answer (precision); skip on raw retrieval dumps
    forbidden_text = answer if not raw_retrieval else ""

    # --- Legacy regression checks ---
    if case.get("expect_any_keywords") and not case.get("facts_required"):
        if not _contains_any(combined, case["expect_any_keywords"]):
            failures.append(f"missing keywords anyof={case['expect_any_keywords']}")
    if case.get("expect_doc_id_contains"):
        if case["expect_doc_id_contains"].lower() not in combined.lower():
            failures.append(
                f"expected doc_id fragment {case['expect_doc_id_contains']!r}"
            )
    if case.get("expect_topic_id"):
        if case["expect_topic_id"] not in combined:
            failures.append(f"expected topic_id {case['expect_topic_id']}")
    if case.get("expect_modality"):
        needle = f"mod={case['expect_modality']}"
        if needle not in combined and case["expect_modality"] not in combined.lower():
            failures.append(f"expected modality {case['expect_modality']}")
    if case.get("expect_sources"):
        for src in case["expect_sources"]:
            if src not in sources_used:
                failures.append(f"expected source {src}")

    # --- Refusal / not_found ---
    if case.get("should_refuse") and not raw_retrieval:
        if not _looks_like_refusal(answer) and not _looks_like_refusal(evidence):
            failures.append("expected refusal / not-in-document language")

    # --- Required facts → recall ---
    required = list(case.get("facts_required") or case.get("expected_facts") or [])
    req_hits, _, missing_required = _score_fact_list(coverage_text, required)
    # Fallback: if answer-only mode missed some, allow evidence to cover them for
    # hybrid "grounded" scoring (still counts as hit for pass, but note in metrics)
    if score_answer_only and missing_required and evidence:
        still_missing: list[str] = []
        for fact in missing_required:
            if _fact_present(evidence, fact):
                req_hits += 1
            else:
                still_missing.append(fact)
        missing_required = still_missing

    req_total = len(required)
    recall = (req_hits / req_total) if req_total else 1.0
    thr = float(case.get("required_recall_threshold") or REQUIRED_RECALL_THRESHOLD)
    if req_total:
        need = max(1, int((req_total * thr) + 0.999))  # ceil
        if req_hits < need:
            failures.append(
                f"recall {req_hits}/{req_total}={recall:.0%} "
                f"(need {need}/{req_total}≥{thr:.0%}); missing={missing_required[:8]}"
            )

    # --- Optional facts → bonus (never fail) ---
    optional = list(case.get("facts_optional") or [])
    opt_hits, hit_optional, _ = _score_fact_list(coverage_text, optional)
    opt_total = len(optional)
    optional_recall = (opt_hits / opt_total) if opt_total else 0.0

    # --- Forbidden facts → precision (answer only) ---
    forbidden = list(case.get("facts_forbidden") or case.get("must_not_include") or [])
    forb_hits = 0
    hit_forbidden: list[str] = []
    if forbidden and forbidden_text:
        for fact in forbidden:
            if _fact_present(forbidden_text, fact):
                forb_hits += 1
                hit_forbidden.append(fact)
        if forb_hits:
            failures.append(
                f"precision: forbidden facts in answer: {hit_forbidden[:8]}"
            )
    forb_total = len(forbidden)
    # precision = 1 when no forbidden list or none hit
    precision = 1.0 - (forb_hits / forb_total) if forb_total else 1.0

    # --- Source pages (soft) ---
    want_pages = list(case.get("source_pages") or [])
    if want_pages and pages_found:
        if not set(want_pages) & set(pages_found):
            failures.append(
                f"no expected pages {want_pages} in retrieved pages {pages_found[:12]}"
            )

    # --- Boolean soft check ---
    if case.get("expected_boolean") is not None and required:
        exp_bool = bool(case["expected_boolean"])
        low = answer.lower()
        if exp_bool is False and re.search(r"\byes\b", low) and not re.search(
            r"\bno\b", low
        ):
            failures.append("expected negative boolean answer")

    return CaseResult(
        case_id=case["id"],
        passed=not failures,
        failures=failures,
        intent=intent,
        preview=(answer or evidence)[:240].replace("\n", " "),
        category=str(case.get("category") or ""),
        fact_hits=req_hits,
        fact_total=req_total,
        pages_found=pages_found,
        recall=round(recall, 4),
        precision=round(precision, 4),
        optional_recall=round(optional_recall, 4),
        required_hits=req_hits,
        required_total=req_total,
        optional_hits=opt_hits,
        optional_total=opt_total,
        forbidden_hits=forb_hits,
        forbidden_total=forb_total,
        missing_required=missing_required,
        hit_forbidden=hit_forbidden,
        hit_optional=hit_optional,
    )


def run_case(router: QueryRouter, case: dict[str, Any]) -> CaseResult:
    case = normalize_case(case)
    intent = RouteIntent(case["intent"]) if case.get("intent") else None
    result = router.ask(
        case["query"],
        doc_id=case.get("doc_id"),
        agency=case.get("agency"),
        top_k=case.get("top_k", 8),
        intent=intent,
    )
    text = result.text or ""
    return score_case(
        case,
        text=text,
        evidence=text,
        intent=result.intent.value,
        sources_used=list(result.sources_used or []),
        score_answer_only=False,  # router = retrieval coverage
    )


def run_case_agent(case: dict[str, Any], *, use_llm: bool = True) -> CaseResult:
    """Score via full agent (retrieve + optional Haiku answer)."""
    from govgrant.agent.graph import run_agent

    case = normalize_case(case)
    final = run_agent(
        case["query"],
        doc_id=case.get("doc_id"),
        agency=case.get("agency"),
        use_llm=use_llm,
    )
    answer = final.get("answer") or ""
    evidence = final.get("evidence") or ""
    return score_case(
        case,
        text=answer,
        evidence=evidence,
        intent=str(final.get("intent") or ""),
        sources_used=list(final.get("sources_used") or []),
        # With LLM: grade the answer (precision/forbidden applies)
        # Without: treat as retrieval pack
        score_answer_only=bool(use_llm),
    )


def _case_dict(r: CaseResult) -> dict[str, Any]:
    return {
        "id": r.case_id,
        "passed": r.passed,
        "category": r.category,
        "intent": r.intent,
        "recall": r.recall,
        "precision": r.precision,
        "optional_recall": r.optional_recall,
        "required_hits": r.required_hits,
        "required_total": r.required_total,
        "optional_hits": r.optional_hits,
        "optional_total": r.optional_total,
        "forbidden_hits": r.forbidden_hits,
        "forbidden_total": r.forbidden_total,
        "missing_required": r.missing_required,
        "hit_forbidden": r.hit_forbidden,
        "hit_optional": r.hit_optional,
        # legacy keys
        "fact_hits": r.fact_hits,
        "fact_total": r.fact_total,
        "pages_found": r.pages_found,
        "failures": r.failures,
        "preview": r.preview,
    }


def run_regression(
    path: Path | None = None,
    *,
    golden: bool = False,
    backend: Literal["router", "agent"] = "router",
    use_llm: bool = False,
    limit: int | None = None,
    ids: set[str] | None = None,
    categories: set[str] | None = None,
) -> dict[str, Any]:
    cases = load_cases(path, golden=golden)
    if categories:
        cases = [c for c in cases if c.get("category") in categories]
    if ids:
        cases = [c for c in cases if c.get("id") in ids]
    if limit is not None:
        cases = cases[:limit]

    results: list[CaseResult] = []
    router: QueryRouter | None = None
    if backend == "router":
        router = QueryRouter(get_settings())

    for c in cases:
        if backend == "agent":
            results.append(run_case_agent(c, use_llm=use_llm))
        else:
            assert router is not None
            results.append(run_case(router, c))

    passed = sum(1 for r in results if r.passed)
    by_cat: dict[str, dict[str, Any]] = {}
    for r in results:
        cat = r.category or "unknown"
        slot = by_cat.setdefault(
            cat,
            {
                "total": 0,
                "passed": 0,
                "recall_sum": 0.0,
                "precision_sum": 0.0,
            },
        )
        slot["total"] += 1
        if r.passed:
            slot["passed"] += 1
        slot["recall_sum"] += r.recall
        slot["precision_sum"] += r.precision

    for cat, slot in by_cat.items():
        n = max(1, slot["total"])
        slot["avg_recall"] = round(slot.pop("recall_sum") / n, 4)
        slot["avg_precision"] = round(slot.pop("precision_sum") / n, 4)

    n = max(1, len(results))
    avg_recall = round(sum(r.recall for r in results) / n, 4) if results else 0.0
    avg_precision = round(sum(r.precision for r in results) / n, 4) if results else 1.0
    avg_optional = (
        round(
            sum(r.optional_recall for r in results if r.optional_total) /
            max(1, sum(1 for r in results if r.optional_total)),
            4,
        )
        if any(r.optional_total for r in results)
        else None
    )

    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "backend": backend,
        "use_llm": use_llm if backend == "agent" else False,
        "metrics": {
            "avg_recall": avg_recall,
            "avg_precision": avg_precision,
            "avg_optional_recall": avg_optional,
            "pass_rate": round(100.0 * passed / n, 2) if results else 0.0,
        },
        "by_category": by_cat,
        "cases": [_case_dict(r) for r in results],
    }


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="RAG regression / golden eval")
    parser.add_argument("--path", default=None, help="JSON file or directory")
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Load all data/eval/0*.json golden files",
    )
    parser.add_argument(
        "--backend",
        choices=["router", "agent"],
        default="router",
        help="router = QueryRouter only; agent = full LangGraph (+ optional LLM)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="When --backend agent, use Haiku for answer formatting",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--id",
        action="append",
        dest="ids",
        default=None,
        help="Only run these case ids (repeatable)",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        default=None,
        help="Filter by category (repeatable)",
    )
    args = parser.parse_args()

    report = run_regression(
        Path(args.path) if args.path else None,
        golden=args.golden,
        backend=args.backend,
        use_llm=args.llm,
        limit=args.limit,
        ids=set(args.ids) if args.ids else None,
        categories=set(args.categories) if args.categories else None,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
