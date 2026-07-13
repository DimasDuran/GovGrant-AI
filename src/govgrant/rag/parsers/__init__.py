"""Modality-isolated parsers."""

from govgrant.rag.parsers.base import BaseModalityParser
from govgrant.rag.parsers.figures import FigureChartParser
from govgrant.rag.parsers.prose import ProsePDFParser
from govgrant.rag.parsers.tables import TableMarkdownParser, parse_markdown_tables

__all__ = [
    "BaseModalityParser",
    "FigureChartParser",
    "ProsePDFParser",
    "TableMarkdownParser",
    "parse_markdown_tables",
]
