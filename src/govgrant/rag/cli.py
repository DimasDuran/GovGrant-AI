"""CLI entrypoints: ingest PDFs, hybrid query, table structured search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from govgrant.rag.config import get_settings
from govgrant.rag.index.hybrid import HybridRAGService


def ingest_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into hybrid RAG (Qdrant + BM25 + tables)")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="PDF file or directory (default: data/fixtures/pdfs)",
    )
    parser.add_argument("--tenant", default=None, help="tenant_id")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key when AUTH_ENABLED=true (binds tenant)",
    )
    parser.add_argument(
        "--no-llamaparse",
        action="store_true",
        help="Use local pypdf only (no LlamaParse)",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip table extraction",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure/chart extraction (R4)",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Skip Ollama vision captions even if OLLAMA_VISION_MODEL is set",
    )
    args = parser.parse_args(argv)
    auth = _resolve_cli_auth(args)

    settings = get_settings()
    svc = HybridRAGService(settings)
    target = args.path or str(settings.fixtures_pdf_dir)
    use_lp = not args.no_llamaparse
    extract_tables = not args.no_tables
    extract_figures = not args.no_figures
    use_vision = not args.no_vision

    p = Path(target)
    if p.is_file():
        result = svc.ingest_pdf(
            p,
            tenant_id=auth.tenant_id,
            use_llamaparse=use_lp,
            extract_tables=extract_tables,
            extract_figures=extract_figures,
            use_vision=use_vision,
        )
        print(json.dumps(result, indent=2))
    else:
        results = svc.ingest_directory(
            p,
            tenant_id=auth.tenant_id,
            use_llamaparse=use_lp,
            extract_tables=extract_tables,
            extract_figures=extract_figures,
            use_vision=use_vision,
        )
        print(json.dumps(results, indent=2))


def query_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hybrid query against indexed docs")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--tenant", default=None, help="tenant_id filter")
    parser.add_argument("--api-key", default=None, help="API key when AUTH_ENABLED")
    parser.add_argument("--doc-id", default=None, help="gg_doc_id filter")
    parser.add_argument(
        "--modality",
        default=None,
        choices=["prose", "table", "figure", "chart", "form"],
        help="Filter by modality (e.g. table)",
    )
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)
    auth = _resolve_cli_auth(args)
    doc_id = _cli_doc_id(auth, args.doc_id)

    svc = HybridRAGService()
    hits = svc.retrieve(
        args.query,
        tenant_id=auth.tenant_id,
        doc_id=doc_id,
        modality=args.modality,
        top_k=args.top_k,
    )
    print(svc.format_hits(hits))


def tables_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Structured table store commands")
    sub = parser.add_subparsers(dest="action", required=True)

    p_list = sub.add_parser("list", help="List extracted tables")
    p_list.add_argument("--tenant", default=None)
    p_list.add_argument("--doc-id", default=None)

    p_search = sub.add_parser("search", help="Keyword search over table cells")
    p_search.add_argument("query")
    p_search.add_argument("--tenant", default=None)
    p_search.add_argument("--doc-id", default=None)
    p_search.add_argument("--limit", type=int, default=15)

    p_get = sub.add_parser("get", help="Get one table by table_id")
    p_get.add_argument("table_id")

    p_stats = sub.add_parser("stats", help="Table store counts")
    p_stats.add_argument("--tenant", default=None)

    args = parser.parse_args(argv)
    svc = HybridRAGService()

    if args.action == "list":
        rows = svc.list_tables(tenant_id=args.tenant, doc_id=args.doc_id)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    elif args.action == "search":
        rows = svc.search_tables(
            args.query,
            tenant_id=args.tenant,
            doc_id=args.doc_id,
            limit=args.limit,
        )
        print(svc.format_table_hits(rows))
    elif args.action == "get":
        data = svc.tabular.get_table(args.table_id)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif args.action == "stats":
        print(json.dumps(svc.tabular.stats(tenant_id=args.tenant), indent=2))


def agent_main(argv: list[str] | None = None) -> None:
    """R7 LangGraph agent entrypoint (+ optional Anthropic Haiku chat)."""
    from govgrant.agent.graph import run_agent
    from govgrant.agent.llm import ChatLLM

    parser = argparse.ArgumentParser(description="LangGraph agent (R7 + Haiku)")
    parser.add_argument("query")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--api-key", default=None, help="API key when AUTH_ENABLED")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--agency", default=None)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Anthropic Haiku; return raw retrieved evidence",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    auth = _resolve_cli_auth(args)
    doc_id = _cli_doc_id(auth, args.doc_id)

    llm = ChatLLM()
    if not args.no_llm and not llm.available:
        print(
            "[warn] Anthropic chat not available "
            "(check ANTHROPIC_API_KEY / CHAT_ENABLED). Using retrieve-only mode.",
            file=sys.stderr,
        )

    result = run_agent(
        args.query,
        tenant_id=auth.tenant_id,
        doc_id=doc_id,
        agency=args.agency,
        use_llm=not args.no_llm,
    )
    if args.json:
        slim = {
            "query": result.get("query"),
            "intent": result.get("intent"),
            "sources_used": result.get("sources_used"),
            "insufficient": result.get("insufficient"),
            "used_llm": result.get("used_llm"),
            "meta": result.get("meta"),
            "answer": result.get("answer"),
        }
        print(json.dumps(slim, indent=2, ensure_ascii=False))
    else:
        print(result.get("answer") or result.get("evidence") or "(empty)")


def eval_main(argv: list[str] | None = None) -> None:
    """R6 regression + golden runtime evaluation."""
    from govgrant.rag.eval.runner import run_regression

    parser = argparse.ArgumentParser(description="Run RAG regression / golden eval set")
    parser.add_argument(
        "--path",
        default=None,
        help="Path to regression JSON or eval directory (default data/eval/regression_min.json)",
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Load all data/eval/01–08 golden files (unified schema)",
    )
    parser.add_argument(
        "--backend",
        choices=["router", "agent"],
        default="router",
        help="router=QueryRouter only; agent=LangGraph (+ optional LLM)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="With --backend agent, use Haiku to format answers",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only first N cases")
    parser.add_argument(
        "--id",
        action="append",
        dest="ids",
        default=None,
        help="Only these case ids (repeatable)",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        default=None,
        help="Filter by category (repeatable)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write full JSON report to this path (includes per-case results)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full per-case report to stdout (default: summary + failures only)",
    )
    args = parser.parse_args(argv)
    from pathlib import Path

    report = run_regression(
        Path(args.path) if args.path else None,
        golden=args.golden,
        backend=args.backend,
        use_llm=args.llm,
        limit=args.limit,
        ids=set(args.ids) if args.ids else None,
        categories=set(args.categories) if args.categories else None,
    )
    # Compact summary first (fact-based metrics)
    metrics = report.get("metrics") or {}
    summary = {
        "total": report["total"],
        "passed": report["passed"],
        "failed": report["failed"],
        "backend": report.get("backend"),
        "use_llm": report.get("use_llm"),
        "pass_rate": metrics.get(
            "pass_rate",
            (
                round(100.0 * report["passed"] / report["total"], 2)
                if report["total"]
                else 0.0
            ),
        ),
        "avg_recall": metrics.get("avg_recall"),
        "avg_precision": metrics.get("avg_precision"),
        "avg_optional_recall": metrics.get("avg_optional_recall"),
        "by_category": report.get("by_category"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    failed_cases = [c for c in report["cases"] if not c["passed"]]
    if failed_cases:
        print("\n# failures")
        slim = [
            {
                "id": c["id"],
                "recall": c.get("recall"),
                "precision": c.get("precision"),
                "missing_required": c.get("missing_required"),
                "hit_forbidden": c.get("hit_forbidden"),
                "failures": c.get("failures"),
                "preview": (c.get("preview") or "")[:180],
            }
            for c in failed_cases[:50]
        ]
        print(json.dumps(slim, indent=2, ensure_ascii=False))
    if args.full:
        print("\n# full_report")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**report, "summary": summary}
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\n# wrote {out_path}", file=__import__("sys").stderr)
    if report["failed"]:
        raise SystemExit(1)


def ask_main(argv: list[str] | None = None) -> None:
    """R5 multi-source routed question."""
    from govgrant.rag.router.query_router import QueryRouter, RouteIntent

    parser = argparse.ArgumentParser(description="Multi-source routed ask (R5)")
    parser.add_argument("query", help="User question")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--api-key", default=None, help="API key when AUTH_ENABLED")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--agency", default=None, help="SBIR agency filter e.g. DOD")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--intent",
        default=None,
        choices=[i.value for i in RouteIntent],
        help="Force route intent (skip classifier)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON (intent + meta + text)",
    )
    args = parser.parse_args(argv)
    auth = _resolve_cli_auth(args)
    doc_id = _cli_doc_id(auth, args.doc_id)

    router = QueryRouter()
    intent = RouteIntent(args.intent) if args.intent else None
    result = router.ask(
        args.query,
        tenant_id=auth.tenant_id,
        doc_id=doc_id,
        agency=args.agency,
        top_k=args.top_k,
        intent=intent,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"# intent={result.intent.value} sources={result.sources_used}\n")
        print(result.text)


def sbir_main(argv: list[str] | None = None) -> None:
    from govgrant.rag.sbir.service import SBIRTopicService

    parser = argparse.ArgumentParser(description="SBIR.gov topics connector (R3)")
    sub = parser.add_subparsers(dest="action", required=True)

    p_sync = sub.add_parser("sync", help="Sync open topics (API or fixtures fallback)")
    p_sync.add_argument("--keyword", default=None)
    p_sync.add_argument("--agency", default=None, help="e.g. DOD, HHS, NASA")
    p_sync.add_argument(
        "--fixtures",
        action="store_true",
        help="Force load local sample fixtures (skip API)",
    )

    p_search = sub.add_parser("search", help="Hybrid search over indexed SBIR topics")
    p_search.add_argument("query")
    p_search.add_argument("--agency", default=None)
    p_search.add_argument("--program", default=None, help="SBIR or STTR")
    p_search.add_argument("--status", default="open")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument(
        "--no-disclaimer",
        action="store_true",
        help="Omit mandatory SBIR disclaimer footer",
    )

    p_get = sub.add_parser("get", help="Get one topic by topic_id")
    p_get.add_argument("topic_id")
    p_get.add_argument("--no-disclaimer", action="store_true")

    p_list = sub.add_parser("list", help="List topics from structured store")
    p_list.add_argument("--agency", default=None)
    p_list.add_argument("--program", default=None)
    p_list.add_argument("--status", default="open")
    p_list.add_argument("--limit", type=int, default=20)

    args = parser.parse_args(argv)
    svc = SBIRTopicService()

    if args.action == "sync":
        result = svc.sync(
            keyword=args.keyword,
            agency=args.agency,
            force_fixtures=args.fixtures,
        )
        print(json.dumps(result, indent=2))
    elif args.action == "search":
        result = svc.search(
            args.query,
            agency=args.agency,
            program=args.program,
            status=args.status,
            top_k=args.top_k,
            include_disclaimer=not args.no_disclaimer,
        )
        print(result["text"])
        meta = {
            "topic_ids": result["topic_ids"],
            "stale": result["stale"],
            "source": result["source"],
            "last_sync_at": result["last_sync_at"],
        }
        print("\n# meta:", json.dumps(meta))
    elif args.action == "get":
        result = svc.get_topic(
            args.topic_id, include_disclaimer=not args.no_disclaimer
        )
        print(result["text"])
    elif args.action == "list":
        topics = svc.store.list_topics(
            status=args.status,
            agency=args.agency,
            program=args.program,
            limit=args.limit,
        )
        slim = [
            {
                "topic_id": t.topic_id,
                "title": t.topic_title,
                "agency": t.agency,
                "program": t.program,
                "status": t.status,
                "close_date": t.close_date,
                "citation_uri": t.citation_uri,
                "stale": t.stale,
                "source": t.source,
            }
            for t in topics
        ]
        print(json.dumps(slim, indent=2, ensure_ascii=False))


def checklist_main(argv: list[str] | None = None) -> None:
    """Multi-agency compliance checklist (+ optional draft scoring)."""
    from govgrant.compliance.checklist import run_checklist

    parser = argparse.ArgumentParser(
        description="Compliance checklist (DARPA / SBA / SF424) + optional draft"
    )
    parser.add_argument(
        "--program",
        choices=["sbir", "sttr"],
        default="sbir",
        help="Program type (default sbir)",
    )
    parser.add_argument(
        "--ot",
        action="store_true",
        help="Include Other Transaction milestone controls (DARPA)",
    )
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        choices=["darpa", "sba", "sf424"],
        default=None,
        help="Agency package (repeatable). Default: all three.",
    )
    parser.add_argument(
        "--draft-file",
        default=None,
        help="Path to proposal draft text/markdown to score against controls",
    )
    parser.add_argument(
        "--draft-pdf",
        default=None,
        help="Path to proposal PDF (extracted locally for draft scoring)",
    )
    parser.add_argument(
        "--index-proposal",
        action="store_true",
        help="Also ingest the proposal PDF into hybrid RAG for chat Q&A",
    )
    parser.add_argument(
        "--llm-draft",
        action="store_true",
        help="Use Haiku to judge draft vs controls (falls back to keywords)",
    )
    parser.add_argument("--tenant", default=None, help="tenant_id for proposal index")
    parser.add_argument("--api-key", default=None, help="API key when AUTH_ENABLED")
    parser.add_argument("--json", action="store_true", help="Full JSON report")
    args = parser.parse_args(argv)
    auth = _resolve_cli_auth(args)

    draft_text = None
    extract_meta = None
    if args.draft_pdf:
        from govgrant.compliance.proposal import (
            extract_proposal_text,
            proposal_doc_id,
        )

        extract_meta = extract_proposal_text(args.draft_pdf)
        draft_text = extract_meta.text
        if args.index_proposal:
            settings = get_settings()
            rag = HybridRAGService(settings)
            pid = proposal_doc_id(Path(args.draft_pdf).name)
            info = rag.ingest_pdf(
                Path(args.draft_pdf),
                tenant_id=auth.tenant_id,
                doc_id=pid,
                use_llamaparse=False,
                extract_tables=True,
                extract_figures=False,
                use_vision=False,
            )
            print(
                f"# indexed proposal doc_id={info.get('doc_id') or pid} "
                f"leaves={info.get('leaf_nodes')}",
                file=sys.stderr,
            )
            print(
                f"# chat filter: --doc-id {pid}",
                file=sys.stderr,
            )
    elif args.draft_file:
        draft_text = Path(args.draft_file).read_text(encoding="utf-8")

    run = run_checklist(
        program=args.program,
        use_ot=args.ot,
        packages=args.packages,
        draft_text=draft_text,
        use_llm_draft=bool(args.llm_draft),
        tenant_id=auth.tenant_id,
    )
    if args.json:
        payload = run.to_dict()
        if extract_meta is not None:
            payload["proposal_extract"] = {
                "file_name": extract_meta.file_name,
                "pages": extract_meta.pages,
                "chars": extract_meta.chars,
                "parser": extract_meta.parser,
            }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if extract_meta is not None:
            print(extract_meta.summary_markdown())
            print()
        print(run.to_markdown())
    critical_fail = any(
        i.status == "fail" and i.severity == "critical" for i in run.items
    )
    if critical_fail:
        raise SystemExit(1)


def gate_main(argv: list[str] | None = None) -> None:
    """Run a named quality gate from data/eval/THRESHOLDS.json."""
    from govgrant.rag.eval.gates import load_thresholds, run_configured_gate

    parser = argparse.ArgumentParser(
        description="Quality gate runner (thresholds + golden/checklist)"
    )
    parser.add_argument(
        "gate",
        nargs="?",
        default="router",
        help="Gate id: router | hard_llm | checklist_corpus (default router)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured gates and exit",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Path to THRESHOLDS.json (default data/eval/THRESHOLDS.json)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write gate + report JSON under data/eval/reports/",
    )
    args = parser.parse_args(argv)
    thr_path = Path(args.thresholds) if args.thresholds else None

    if args.list:
        cfg = load_thresholds(thr_path)
        print(json.dumps(cfg.get("gates", {}), indent=2, ensure_ascii=False))
        return

    out = Path(args.out) if args.out else None
    if out is None and args.gate in {"router", "hard_llm", "checklist_corpus"}:
        out = (
            Path("data/eval/reports")
            / f"gate_{args.gate}_latest.json"
        )

    result = run_configured_gate(args.gate, thresholds_path=thr_path, out_path=out)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    if not result.passed:
        print("\n# GATE FAILED", file=sys.stderr)
        for c in result.checks:
            if not c.ok:
                print(f"  - {c.message}", file=sys.stderr)
        raise SystemExit(1)
    print("\n# GATE PASSED", file=sys.stderr)


def _resolve_cli_auth(args: argparse.Namespace):
    """Shared --api-key / --tenant resolution for mutating/read paths."""
    from govgrant.auth import AuthError, resolve_request_auth

    api_key = getattr(args, "api_key", None)
    tenant = getattr(args, "tenant", None)
    try:
        return resolve_request_auth(api_key=api_key, tenant_id=tenant)
    except AuthError as exc:
        print(f"auth error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _cli_doc_id(auth, doc_id: str | None) -> str | None:
    from govgrant.auth import AuthError

    try:
        return auth.filter_doc_id(doc_id)
    except AuthError as exc:
        print(f"auth error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def proposals_main(argv: list[str] | None = None) -> None:
    """Tenant-scoped proposal registry CLI."""
    from govgrant.auth import AuthError
    from govgrant.proposals import ProposalService

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tenant", default=None)
    common.add_argument("--api-key", default=None)
    common.add_argument(
        "--json",
        action="store_true",
        help="JSON output",
    )

    parser = argparse.ArgumentParser(description="Manage tenant proposals")
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("list", parents=[common], help="List proposals for this tenant")

    p_up = sub.add_parser(
        "upload", parents=[common], help="Register PDF (+ optional index)"
    )
    p_up.add_argument("path", help="Path to proposal PDF")
    p_up.add_argument(
        "--no-index",
        action="store_true",
        help="Skip hybrid RAG indexing",
    )
    p_up.add_argument("--notes", default="", help="Optional note")

    p_get = sub.add_parser(
        "get", parents=[common], help="Show one proposal metadata"
    )
    p_get.add_argument("doc_id")

    p_del = sub.add_parser(
        "delete",
        parents=[common],
        help="Delete proposal (registry + file + Qdrant/BM25/tables)",
    )
    p_del.add_argument("doc_id")
    p_del.add_argument(
        "--keep-file",
        action="store_true",
        help="Keep PDF on disk",
    )
    p_del.add_argument(
        "--keep-index",
        action="store_true",
        help="Do not purge Qdrant/BM25/tables",
    )

    sub.add_parser("whoami", parents=[common], help="Show resolved auth context")

    args = parser.parse_args(argv)
    auth = _resolve_cli_auth(args)
    svc = ProposalService()

    if args.action == "whoami":
        payload = auth.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(
                f"tenant={auth.tenant_id} roles={list(auth.roles)} "
                f"auth_enabled={auth.auth_enabled} source={auth.source}"
            )
        return

    if args.action == "list":
        rows = [r.to_dict() for r in svc.list_proposals(auth)]
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
            if not rows:
                print(f"(no proposals for tenant {auth.tenant_id})")
                return
            for r in rows:
                print(
                    f"{r['doc_id']}\t{r['file_name']}\tpages={r['pages']}\t"
                    f"indexed={r['indexed']}\t{r['created_at']}"
                )
        return

    if args.action == "get":
        rec = svc.get(auth, args.doc_id)
        if rec is None:
            print(f"not found: {args.doc_id}", file=sys.stderr)
            raise SystemExit(1)
        print(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False))
        return

    if args.action == "upload":
        try:
            result = svc.upload(
                auth,
                args.path,
                index=not args.no_index,
                notes=args.notes or "",
            )
        except (AuthError, FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            r = result.record
            print(
                f"registered {r.doc_id} tenant={r.tenant_id} "
                f"pages={r.pages} indexed={r.indexed} path={r.stored_path}"
            )
        return

    if args.action == "delete":
        # Destructive: require admin when AUTH_ENABLED, else allow owner tenant path
        if auth.auth_enabled and not auth.has_role("admin", "user"):
            # "user" can delete own tenant proposals; admin same for now
            try:
                auth.require_role("admin", "user")
            except AuthError as exc:
                print(f"auth error: {exc}", file=sys.stderr)
                raise SystemExit(2) from exc
        ok = svc.delete(
            auth,
            args.doc_id,
            remove_file=not args.keep_file,
            purge_index=not args.keep_index,
        )
        if not ok:
            print(f"not found: {args.doc_id}", file=sys.stderr)
            raise SystemExit(1)
        purge = getattr(svc, "_last_delete_index", None) or {}
        if args.json:
            print(
                json.dumps(
                    {"deleted": args.doc_id, "index_purge": purge},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(
                f"deleted {args.doc_id} "
                f"bm25={purge.get('bm25_removed')} "
                f"qdrant≈{purge.get('qdrant_deleted_estimate')} "
                f"tables={purge.get('tabular_cleared')}"
            )
        return


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {
        "ingest",
        "query",
        "tables",
        "sbir",
        "ask",
        "eval",
        "agent",
        "checklist",
        "gate",
        "proposals",
    }:
        print(
            "Usage: python -m govgrant.rag.cli "
            "ingest|query|tables|sbir|ask|eval|agent|checklist|gate|proposals ...",
            file=sys.stderr,
        )
        sys.exit(2)
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    if cmd == "ingest":
        ingest_main(rest)
    elif cmd == "query":
        query_main(rest)
    elif cmd == "tables":
        tables_main(rest)
    elif cmd == "sbir":
        sbir_main(rest)
    elif cmd == "ask":
        ask_main(rest)
    elif cmd == "eval":
        eval_main(rest)
    elif cmd == "checklist":
        checklist_main(rest)
    elif cmd == "gate":
        gate_main(rest)
    elif cmd == "proposals":
        proposals_main(rest)
    else:
        agent_main(rest)
