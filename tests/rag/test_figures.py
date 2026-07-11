from llama_index.core.schema import Document

from govgrant.rag.contracts import Modality
from govgrant.rag.parsers.figures import (
    extract_figures_from_markdown,
    _guess_modality,
)


def test_guess_chart_modality():
    assert _guess_modality("Bar chart of Phase I awards") == Modality.CHART
    assert _guess_modality("System architecture diagram") == Modality.FIGURE


def test_extract_markdown_figures():
    docs = [
        Document(
            text=(
                "Intro text\n\n"
                "![Funding planner timeline](images/planner.png)\n\n"
                "Figure 2: Award amounts by agency over time.\n"
            ),
            metadata={"page": 3},
        )
    ]
    figs = extract_figures_from_markdown(docs, doc_id="demo")
    assert len(figs) >= 2
    captions = " ".join(f.caption for f in figs).lower()
    assert "funding planner" in captions or "award amounts" in captions
    assert any(f.page == 3 for f in figs)
