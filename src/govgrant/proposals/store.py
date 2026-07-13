"""
SQLite registry of user-uploaded proposals, scoped by tenant_id.

Does not store PDF bytes in SQL — only metadata + filesystem path under
data/indexes/proposals/{tenant_id}/.

Also stores a lightweight audit log (proposal_events) for upload/delete.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ProposalRecord:
    doc_id: str
    tenant_id: str
    file_name: str
    stored_path: str
    pages: int
    chars: int
    indexed: bool
    created_at: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProposalEvent:
    id: int
    tenant_id: str
    doc_id: str
    action: str  # upload | delete | delete_denied
    actor_roles: str  # comma-joined roles at time of action
    detail: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "doc_id": self.doc_id,
            "action": self.action,
            "actor_roles": self.actor_roles,
            "detail": self.detail,
            "created_at": self.created_at,
        }


class ProposalStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proposals (
                    doc_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    pages INTEGER NOT NULL DEFAULT 0,
                    chars INTEGER NOT NULL DEFAULT 0,
                    indexed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (tenant_id, doc_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_prop_tenant "
                "ON proposals(tenant_id, created_at DESC)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proposal_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor_roles TEXT NOT NULL DEFAULT '',
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_prop_events_tenant "
                "ON proposal_events(tenant_id, created_at DESC)"
            )
            conn.commit()

    def upsert(self, rec: ProposalRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO proposals (
                    doc_id, tenant_id, file_name, stored_path, pages, chars,
                    indexed, created_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, doc_id) DO UPDATE SET
                    file_name=excluded.file_name,
                    stored_path=excluded.stored_path,
                    pages=excluded.pages,
                    chars=excluded.chars,
                    indexed=excluded.indexed,
                    notes=excluded.notes
                """,
                (
                    rec.doc_id,
                    rec.tenant_id,
                    rec.file_name,
                    rec.stored_path,
                    rec.pages,
                    rec.chars,
                    1 if rec.indexed else 0,
                    rec.created_at,
                    rec.notes,
                ),
            )
            conn.commit()

    def list_for_tenant(self, tenant_id: str, *, limit: int = 50) -> list[ProposalRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM proposals
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (tenant_id, limit),
            ).fetchall()
        return [self._row(r) for r in rows]

    def get(self, tenant_id: str, doc_id: str) -> ProposalRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM proposals WHERE tenant_id=? AND doc_id=?",
                (tenant_id, doc_id),
            ).fetchone()
        return self._row(row) if row else None

    def delete(self, tenant_id: str, doc_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM proposals WHERE tenant_id=? AND doc_id=?",
                (tenant_id, doc_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def log_event(
        self,
        *,
        tenant_id: str,
        doc_id: str,
        action: str,
        actor_roles: tuple[str, ...] | list[str] | str = (),
        detail: dict[str, Any] | None = None,
    ) -> ProposalEvent:
        """Append an audit event; returns the stored row (with id)."""
        roles_s = (
            actor_roles if isinstance(actor_roles, str) else ",".join(str(r) for r in actor_roles)
        )
        created = _utc_now()
        detail_json = json.dumps(detail or {}, ensure_ascii=False)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO proposal_events (
                    tenant_id, doc_id, action, actor_roles, detail_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (tenant_id, doc_id, action, roles_s, detail_json, created),
            )
            conn.commit()
            event_id = int(cur.lastrowid or 0)
        return ProposalEvent(
            id=event_id,
            tenant_id=tenant_id,
            doc_id=doc_id,
            action=action,
            actor_roles=roles_s,
            detail=detail or {},
            created_at=created,
        )

    def list_events(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
        doc_id: str | None = None,
    ) -> list[ProposalEvent]:
        with self._connect() as conn:
            if doc_id:
                rows = conn.execute(
                    """
                    SELECT * FROM proposal_events
                    WHERE tenant_id = ? AND doc_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (tenant_id, doc_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM proposal_events
                    WHERE tenant_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (tenant_id, limit),
                ).fetchall()
        return [self._event_row(r) for r in rows]

    @staticmethod
    def _event_row(row: sqlite3.Row) -> ProposalEvent:
        raw = row["detail_json"] or "{}"
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            detail = {"raw": raw}
        if not isinstance(detail, dict):
            detail = {"value": detail}
        return ProposalEvent(
            id=int(row["id"]),
            tenant_id=row["tenant_id"],
            doc_id=row["doc_id"],
            action=row["action"],
            actor_roles=row["actor_roles"] or "",
            detail=detail,
            created_at=row["created_at"] or "",
        )

    @staticmethod
    def _row(row: sqlite3.Row) -> ProposalRecord:
        return ProposalRecord(
            doc_id=row["doc_id"],
            tenant_id=row["tenant_id"],
            file_name=row["file_name"],
            stored_path=row["stored_path"],
            pages=int(row["pages"] or 0),
            chars=int(row["chars"] or 0),
            indexed=bool(row["indexed"]),
            created_at=row["created_at"],
            notes=row["notes"] or "",
        )

    @staticmethod
    def new_record(
        *,
        doc_id: str,
        tenant_id: str,
        file_name: str,
        stored_path: str,
        pages: int,
        chars: int,
        indexed: bool,
        notes: str = "",
    ) -> ProposalRecord:
        return ProposalRecord(
            doc_id=doc_id,
            tenant_id=tenant_id,
            file_name=file_name,
            stored_path=stored_path,
            pages=pages,
            chars=chars,
            indexed=indexed,
            created_at=_utc_now(),
            notes=notes,
        )
