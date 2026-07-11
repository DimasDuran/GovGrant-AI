from govgrant.rag.contracts import DocumentMeta, Modality, SourceType, build_node_metadata


def test_document_meta_payload():
    meta = DocumentMeta(
        tenant_id="t1",
        doc_id="d1",
        file_name="a.pdf",
        citation_uri="/tmp/a.pdf",
    )
    payload = meta.to_payload()
    assert payload["tenant_id"] == "t1"
    assert payload["source_type"] == SourceType.USER_DOC.value
    assert payload["modality"] == Modality.PROSE.value
    assert payload["gg_doc_id"] == "d1"
    assert payload["doc_id"] == "d1"


def test_build_node_metadata_page():
    meta = DocumentMeta(
        tenant_id="t1",
        doc_id="d1",
        file_name="a.pdf",
        citation_uri="/tmp/a.pdf",
    )
    node = build_node_metadata(meta, page=3, section_path="page:3")
    assert node["page"] == 3
    assert node["doc_id"] == "d1"
