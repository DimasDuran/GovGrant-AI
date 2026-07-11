from govgrant.agent.graph import build_agent_graph, run_agent


def test_build_graph():
    app = build_agent_graph()
    assert app is not None


def test_run_agent_doc_query():
    # Keep unit tests offline (no Anthropic call)
    out = run_agent(
        "What foreign ownership disclosures are required?",
        use_llm=False,
    )
    assert out.get("intent") == "doc_qa"
    assert out.get("answer")
    assert "user_docs" in (out.get("sources_used") or [])
