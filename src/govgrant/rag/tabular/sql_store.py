"""SQLite structured store for extracted tables (R2 dual path)."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from govgrant.rag.parsers.tables import ExtractedTable

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class TabularStore:
    """
    Registry of tables + JSON rows for flexible schemas.

    Supports:
    - list tables by tenant/doc
    - get table as list[dict]
    - simple keyword search across cell values
    - optional raw SQL on a denormalized view (read-only SELECT)
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA busy_timeout=5000;

                CREATE TABLE IF NOT EXISTS table_registry (
                    table_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    gg_doc_id TEXT NOT NULL,
                    file_name TEXT,
                    page TEXT,
                    section_path TEXT,
                    headers_json TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    markdown TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS table_rows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_id TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    search_blob TEXT NOT NULL,
                    FOREIGN KEY (table_id) REFERENCES table_registry(table_id)
                );
                CREATE INDEX IF NOT EXISTS idx_rows_table ON table_rows(table_id);
                CREATE INDEX IF NOT EXISTS idx_reg_tenant ON table_registry(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_reg_doc ON table_registry(gg_doc_id);
                CREATE INDEX IF NOT EXISTS idx_rows_search ON table_rows(search_blob);
                CREATE VIRTUAL TABLE IF NOT EXISTS rows_fts USING fts5(
                    search_blob,
                    tokenize='porter unicode61'
                );
                """
            )

    def delete_doc(self, *, tenant_id: str, gg_doc_id: str) -> None:
        with self._connect() as conn:
            ids = [
                r["table_id"]
                for r in conn.execute(
                    "SELECT table_id FROM table_registry WHERE tenant_id=? AND gg_doc_id=?",
                    (tenant_id, gg_doc_id),
                )
            ]
            for tid in ids:
                conn.execute(
                    "DELETE FROM rows_fts WHERE rowid IN (SELECT id FROM table_rows WHERE table_id=?)",
                    (tid,),
                )
                conn.execute("DELETE FROM table_rows WHERE table_id=?", (tid,))
                conn.execute("DELETE FROM table_registry WHERE table_id=?", (tid,))

    def upsert_tables(
        self,
        tables: list[ExtractedTable],
        *,
        tenant_id: str,
        gg_doc_id: str,
        file_name: str,
    ) -> int:
        if not tables:
            return 0
        self.delete_doc(tenant_id=tenant_id, gg_doc_id=gg_doc_id)
        with self._connect() as conn:
            for t in tables:
                conn.execute(
                    """
                    INSERT INTO table_registry
                    (table_id, tenant_id, gg_doc_id, file_name, page, section_path,
                     headers_json, row_count, markdown)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        t.table_id,
                        tenant_id,
                        gg_doc_id,
                        file_name,
                        str(t.page) if t.page is not None else None,
                        t.section_path,
                        json.dumps(t.headers, ensure_ascii=False),
                        t.row_count,
                        t.markdown,
                    ),
                )
                for i, row in enumerate(t.to_row_dicts()):
                    blob = " ".join(f"{k} {v}" for k, v in row.items())
                    cursor = conn.execute(
                        """
                        INSERT INTO table_rows (table_id, row_index, data_json, search_blob)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            t.table_id,
                            i,
                            json.dumps(row, ensure_ascii=False),
                            blob.lower(),
                        ),
                    )
                    conn.execute(
                        "INSERT INTO rows_fts(rowid, search_blob) VALUES (?, ?)",
                        (cursor.lastrowid, blob.lower()),
                    )
        return len(tables)

    def list_tables(
        self,
        *,
        tenant_id: str,
        gg_doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM table_registry WHERE tenant_id=?"
        params: list[Any] = [tenant_id]
        if gg_doc_id:
            sql += " AND gg_doc_id=?"
            params.append(gg_doc_id)
        sql += " ORDER BY gg_doc_id, page"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "table_id": r["table_id"],
                    "tenant_id": r["tenant_id"],
                    "gg_doc_id": r["gg_doc_id"],
                    "file_name": r["file_name"],
                    "page": r["page"],
                    "section_path": r["section_path"],
                    "headers": json.loads(r["headers_json"]),
                    "row_count": r["row_count"],
                }
            )
        return out

    def get_table(self, table_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            reg = conn.execute(
                "SELECT * FROM table_registry WHERE table_id=?", (table_id,)
            ).fetchone()
            if not reg:
                return None
            rows = conn.execute(
                "SELECT row_index, data_json FROM table_rows WHERE table_id=? ORDER BY row_index",
                (table_id,),
            ).fetchall()
        return {
            "table_id": reg["table_id"],
            "tenant_id": reg["tenant_id"],
            "gg_doc_id": reg["gg_doc_id"],
            "file_name": reg["file_name"],
            "page": reg["page"],
            "headers": json.loads(reg["headers_json"]),
            "row_count": reg["row_count"],
            "rows": [json.loads(r["data_json"]) for r in rows],
            "markdown": reg["markdown"],
        }

    def search_cells(
        self,
        query: str,
        *,
        tenant_id: str,
        gg_doc_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Keyword search over cell text using FTS5."""
        tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 1]
        if not tokens:
            return []
        # FTS5 query: quoted terms AND-ed together (safe: tokens are \w+)
        fts_query = " AND ".join(f'"{t}"' for t in tokens)
        sql = """
            SELECT r.table_id, r.row_index, r.data_json,
                   t.gg_doc_id, t.file_name, t.page, t.headers_json
            FROM rows_fts f
            JOIN table_rows r ON r.id = f.rowid
            JOIN table_registry t ON t.table_id = r.table_id
            WHERE rows_fts MATCH ?
              AND t.tenant_id = ?
        """
        params: list[Any] = [fts_query, tenant_id]
        if gg_doc_id:
            sql += " AND t.gg_doc_id = ?"
            params.append(gg_doc_id)
        sql += " LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "table_id": r["table_id"],
                "row_index": r["row_index"],
                "data": json.loads(r["data_json"]),
                "gg_doc_id": r["gg_doc_id"],
                "file_name": r["file_name"],
                "page": r["page"],
                "headers": json.loads(r["headers_json"]),
            }
            for r in rows
        ]

    def stats(self, *, tenant_id: str | None = None) -> dict[str, int]:
        with self._connect() as conn:
            if tenant_id:
                tables = conn.execute(
                    "SELECT COUNT(*) AS c FROM table_registry WHERE tenant_id=?",
                    (tenant_id,),
                ).fetchone()["c"]
                rows = conn.execute(
                    """
                    SELECT COUNT(*) AS c FROM table_rows r
                    JOIN table_registry t ON t.table_id=r.table_id
                    WHERE t.tenant_id=?
                    """,
                    (tenant_id,),
                ).fetchone()["c"]
            else:
                tables = conn.execute(
                    "SELECT COUNT(*) AS c FROM table_registry"
                ).fetchone()["c"]
                rows = conn.execute(
                    "SELECT COUNT(*) AS c FROM table_rows"
                ).fetchone()["c"]
        return {"tables": tables, "rows": rows}
