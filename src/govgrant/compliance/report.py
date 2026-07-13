"""
Export compliance checklist runs to local report files.

Reports default under data/eval/reports/ (gitignored). Pure IO helpers —
no network, easy to test with tmp_path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from govgrant.compliance.checklist import ChecklistRun
from govgrant.rag.config import REPO_ROOT

DEFAULT_REPORT_DIR = REPO_ROOT / "data" / "eval" / "reports"


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


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
        latest_json.write_text(Path(written["json"]).read_text(encoding="utf-8"), encoding="utf-8")
        written["latest_json"] = str(latest_json.resolve())

    return written


def list_export_history(
    *,
    out_dir: Path | str | None = None,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """
    List recent checklist_*.md exports (newest first).

    Skips checklist_latest.md pointer file.
    """
    directory = Path(out_dir) if out_dir else DEFAULT_REPORT_DIR
    if not directory.is_dir():
        return []
    files = sorted(
        (p for p in directory.glob("checklist_*.md") if p.name != "checklist_latest.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for p in files[:limit]:
        st = p.stat()
        json_sib = p.with_suffix(".json")
        out.append(
            {
                "name": p.name,
                "path": str(p.resolve()),
                "json_path": str(json_sib.resolve()) if json_sib.is_file() else None,
                "bytes": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(),
            }
        )
    return out


def export_history_markdown(
    *,
    out_dir: Path | str | None = None,
    limit: int = 15,
) -> str:
    rows = list_export_history(out_dir=out_dir, limit=limit)
    if not rows:
        return "_No checklist exports yet. Run a checklist with export enabled._"
    lines = [
        f"**Export history** (latest {len(rows)})",
        "",
        "| file | size | modified (UTC) |",
        "|------|-----:|----------------|",
    ]
    for r in rows:
        lines.append(f"| `{r['name']}` | {r['bytes']} | {r['mtime']} |")
    latest = (Path(out_dir) if out_dir else DEFAULT_REPORT_DIR) / "checklist_latest.md"
    if latest.is_file():
        lines.append("")
        lines.append(f"Latest pointer: `{latest.resolve()}`")
    return "\n".join(lines)
