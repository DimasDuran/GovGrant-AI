from govgrant.rag.config import REPO_ROOT
from govgrant.rag.sbir.disclaimer import with_disclaimer
from govgrant.rag.sbir.normalizer import load_fixture_json, normalize_solicitations


def test_normalize_fixture_topics():
    path = REPO_ROOT / "data/fixtures/sbir/open_solicitations.sample.json"
    raw = load_fixture_json(path)
    topics = normalize_solicitations(raw, source="fixture", stale=True)
    assert len(topics) >= 5
    ids = {t.topic_id for t in topics}
    assert "12799" in ids  # MDA thermal batteries
    thermal = next(t for t in topics if t.topic_id == "12799")
    assert thermal.agency == "DOD"
    assert "thermal battery" in thermal.topic_description.lower()
    assert thermal.citation_uri.endswith("/topics/12799")


def test_disclaimer_includes_links():
    text = with_disclaimer("Hello", topic_ids=["12799", "12814"])
    assert "Disclaimer" in text
    assert "https://www.sbir.gov/topics/12799" in text
    assert "https://www.sbir.gov/topics/12814" in text
