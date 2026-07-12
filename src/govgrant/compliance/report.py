"""
Export compliance checklist runs to local report files.

Reports default under data/eval/reports/ (gitignored). Pure IO helpers —
no network, easy to test with tmp_path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from govgrant.compliance.checklist import ChecklistRun
from govgrant.rag.config import REPO_ROOT

DEFAULT_REPORT_DIR = REPO_ROOT / "data" / "eval" / "reports"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def export_checklist_run(
    run: ChecklistRun,
    *,
    out_dir: Path | str | None = None,
    basename: str | None = None,
    extra: dict[str, Any] | None = None,
    formats: tuple[str, ...] = ("md", "json"),
) -> dict[str, str]:
    """
    Write checklist report files.

    Returns map format → absolute path string.
    """
    directory = Path(out_dir) if out_dir else DEFAULT_REPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    base = basename or f"checklist_{run.program}_{_stamp()}"
    written: dict[str, str] = {}

    if "md" in formats:
        md_path = directory / f"{base}.md"
        body = run.to_markdown()
        if extra:
            body += "\n\n## Export meta\n\n```json\n"
            body += json.dumps(extra, indent=2, ensure_ascii=False)
            body += "\n```\n"
        md_path.write_text(body, encoding="utf-8")
        written["md"] = str(md_path.resolve())

    if "json" in formats:
        json_path = directory / f"{base}.json"
        payload = run.to_dict()
        if extra:
            payload["export_meta"] = extra
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written["json"] = str(json_path.resolve())

    # Always refresh a stable "latest" pointer (md + json)
    if "md" in written:
        latest_md = directory / "checklist_latest.md"
        latest_md.write_text(Path(written["md"]).read_text(encoding="utf-8"), encoding="utf-8")
        written["latest_md"] = str(latest_md.resolve())
    if "json" in written:
        latest_json = directory / "checklist_latest.json"
        latest_json.write_text(
            Path(written["json"]).read_text(encoding="utf-8"), encoding="utf-8"
        )
        written["latest_json"] = str(latest_json.resolve())

    return written
