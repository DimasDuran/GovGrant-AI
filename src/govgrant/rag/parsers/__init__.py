"""Modality-isolated parsers."""

from govgrant.rag.parsers.base import BaseModalityParser
from govgrant.rag.parsers.prose import ProsePDFParser
from govgrant.rag.parsers.tables import TableMarkdownParser, parse_markdown_tables
from govgrant.rag.parsers.figures import FigureChartParser

__all__ = [
    "BaseModalityParser",
    "ProsePDFParser",
    "TableMarkdownParser",
    "parse_markdown_tables",
    "FigureChartParser",
]
