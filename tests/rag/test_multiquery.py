from llama_index.core.schema import NodeWithScore, TextNode

from govgrant.rag.index.hybrid import diversify_by_page, split_subqueries


def test_split_multipart_also_finally():
    q = (
        "What work-share is required for SBIR versus STTR? "
        "Also, what must I disclose for similar proposals? "
        "Finally, what does an OT milestone plan need?"
    )
    parts = split_subqueries(q)
    assert len(parts) >= 2
    assert parts[0] == q or "work-share" in parts[0].lower()


def test_split_seeds_for_darpa_blob():
    q = (
        "university and federally funded research center SBIR versus STTR "
        "similar proposal to another federal agency Other Transaction OT "
        "milestone plan and commercialization strategy"
    )
    parts = split_subqueries(q)
    joined = " ".join(parts).lower()
    assert "40%" in joined or "work share" in joined or "one-half" in joined
    assert "similar" in joined or "equivalent" in joined
    assert "milestone" in joined or "commercialization" in joined
    assert "letters of intent" in joined


def test_also_finally_still_adds_topic_seeds():
    q = (
        "What work-share is required for SBIR versus STTR? "
        "Also, what must I disclose for similar proposals? "
        "Finally, what optional supporting documents and OT milestones are needed?"
    )
    parts = split_subqueries(q)
    joined = " ".join(parts).lower()
    assert len(parts) >= 4  # full + 3 splits + seeds
    assert "letters of intent" in joined


def test_diversify_by_page():
    hits = []
    for page, score in [(9, 0.9), (9, 0.8), (8, 0.7), (10, 0.6)]:
        n = TextNode(
            text=f"page {page}",
            metadata={"gg_doc_id": "d", "page": page},
            id_=f"n{page}-{score}",
        )
        hits.append(NodeWithScore(node=n, score=score))
    out = diversify_by_page(hits, top_k=3)
    pages = [(h.node.metadata or {}).get("page") for h in out]
    assert pages[0] == 9
    assert 8 in pages and 10 in pages
