from govgrant.rag.parsers.tables import parse_markdown_tables
from govgrant.rag.tabular.sql_store import TabularStore


def test_parse_markdown_table():
    md = """
Some intro text

| Agency | Phase | Amount |
|--------|-------|--------|
| NIH | I | 275000 |
| DARPA | II | 1800000 |

More text
"""
    tables = parse_markdown_tables(md, page=1, doc_id="demo")
    assert len(tables) == 1
    t = tables[0]
    assert t.headers == ["Agency", "Phase", "Amount"]
    assert t.row_count == 2
    assert "NIH" in t.to_rag_text()
    assert t.to_row_dicts()[1]["Agency"] == "DARPA"


def test_tabular_store_roundtrip(tmp_path):
    md = """
| Code | Limit |
|------|-------|
| SF-424 | yes |
| Phase II | 1800000 |
"""
    tables = parse_markdown_tables(md, page=3, doc_id="d1")
    store = TabularStore(tmp_path / "t.sqlite")
    n = store.upsert_tables(
        tables, tenant_id="local-dev", gg_doc_id="d1", file_name="x.pdf"
    )
    assert n == 1
    listed = store.list_tables(tenant_id="local-dev")
    assert len(listed) == 1
    hits = store.search_cells("1800000 Phase", tenant_id="local-dev")
    assert len(hits) >= 1
    assert hits[0]["data"]["Limit"] == "1800000"
