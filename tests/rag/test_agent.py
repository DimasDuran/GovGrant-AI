from govgrant.agent.graph import (
    build_agent_graph,
    is_non_substantive_query,
    run_agent,
)


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


def test_greeting_is_not_chatbot_intro():
    assert is_non_substantive_query("Hola")
    assert is_non_substantive_query("hello!")
    assert is_non_substantive_query("quién eres")
    assert not is_non_substantive_query(
        "What is the maximum DARPA Phase II cost volume?"
    )
    out = run_agent("Hola", use_llm=False)
    ans = out.get("answer") or ""
    assert "chatbot" in ans.lower() or "corpus" in ans.lower()
    assert "Soy GovGrant AI" not in ans
    assert out.get("meta", {}).get("skip_reason") == "non_substantive_query"
