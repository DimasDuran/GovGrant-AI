"""
SQLite registry of user-uploaded proposals, scoped by tenant_id.

Does not store PDF bytes in SQL — only metadata + filesystem path under
data/indexes/proposals/{tenant_id}/.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
