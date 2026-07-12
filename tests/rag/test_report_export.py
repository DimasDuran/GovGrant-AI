"""Tests for checklist report export (local filesystem only)."""

from __future__ import annotations

from pathlib import Path

from govgrant.compliance.checklist import ChecklistResult, ChecklistRun
from govgrant.compliance.report import export_checklist_run


def _sample_run() -> ChecklistRun:
    return ChecklistRun(
        program="sbir",
        use_ot=True,
        packages=["darpa"],
        items=[
            ChecklistResult(
                id="DARPA-WS-SBIR",
                package="darpa",
                section="Work-share",
                title="SBIR minimum",
                status="pass",
                severity="critical",
                guidance="50%",
                evidence_hits=1,
                facts_found=["50%"],
                facts_missing=[],
                citation="p.8",
                detail="ok",
            )
        ],
        summary={"pass": 1},
        draft_provided=False,
    )


def test_export_writes_md_and_json(tmp_path: Path):
    run = _sample_run()
    paths = export_checklist_run(
        run,
        out_dir=tmp_path,
        basename="checklist_test",
        extra={"tenant_id": "local-dev"},
    )
    assert "md" in paths and "json" in paths
    md = Path(paths["md"])
    js = Path(paths["json"])
    assert md.is_file() and js.is_file()
    assert "SBIR" in md.read_text(encoding="utf-8")
    assert "local-dev" in md.read_text(encoding="utf-8")
    assert Path(paths["latest_md"]).is_file()
    assert Path(paths["latest_json"]).is_file()


def test_list_export_history(tmp_path: Path):
    from govgrant.compliance.report import list_export_history, export_history_markdown

    run = _sample_run()
    export_checklist_run(run, out_dir=tmp_path, basename="checklist_a")
    export_checklist_run(run, out_dir=tmp_path, basename="checklist_b")
    hist = list_export_history(out_dir=tmp_path, limit=10)
    assert len(hist) >= 2
    assert all("name" in h for h in hist)
    md = export_history_markdown(out_dir=tmp_path)
    assert "checklist_" in md


def test_admin_required_when_auth_enabled():
    from govgrant.auth.context import AuthContext, AuthError

    user = AuthContext(
        tenant_id="demo-acme",
        roles=("user",),
        api_key_present=True,
        auth_enabled=True,
        allowed_doc_ids=None,
        public_doc_ids=frozenset(),
        source="api_key",
    )
    admin = AuthContext(
        tenant_id="local-dev",
        roles=("admin",),
        api_key_present=True,
        auth_enabled=True,
        allowed_doc_ids=None,
        public_doc_ids=frozenset(),
        source="api_key",
    )
    open_ctx = AuthContext(
        tenant_id="local-dev",
        roles=("user",),
        api_key_present=False,
        auth_enabled=False,
        allowed_doc_ids=None,
        public_doc_ids=frozenset(),
        source="env_default",
    )
    with __import__("pytest").raises(AuthError):
        user.require_admin_for_destructive()
    admin.require_admin_for_destructive()  # no raise
    open_ctx.require_admin_for_destructive()  # open mode OK
