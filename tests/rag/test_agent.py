from govgrant.agent.graph import (
    build_agent_graph,
    is_conversational_turn,
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


def test_greeting_is_friendly_vertical_assistant():
    assert is_conversational_turn("Hola")
    assert is_conversational_turn("¡Hola! 👋")
    assert is_conversational_turn("hello!")
    assert is_conversational_turn("quién eres")
    assert not is_conversational_turn("What is the maximum DARPA Phase II cost volume?")
    out = run_agent("Hola", use_llm=False)
    assert out.get("meta", {}).get("mode") == "conversation"
    assert out.get("intent") == "chat"
    assert out.get("used_llm") is False
