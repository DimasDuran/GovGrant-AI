from govgrant.rag.router.query_router import QueryRouter, RouteIntent


def test_classify_topic():
    r = QueryRouter.__new__(QueryRouter)
    assert r.classify("What open topics match thermal batteries?") == RouteIntent.TOPIC_SEARCH


def test_classify_table():
    r = QueryRouter.__new__(QueryRouter)
    assert r.classify("Show the budget table for indirect costs") == RouteIntent.TABLE


def test_classify_figure():
    r = QueryRouter.__new__(QueryRouter)
    assert r.classify("Describe the chart on the funding planner") == RouteIntent.FIGURE


def test_classify_cross():
    r = QueryRouter.__new__(QueryRouter)
    assert (
        r.classify("Does my proposal abstract align with open MDA topics?")
        == RouteIntent.CROSS_CHECK
    )


def test_classify_doc():
    r = QueryRouter.__new__(QueryRouter)
    assert r.classify("What foreign ownership disclosures are required?") == RouteIntent.DOC_QA
