"""Runtime configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Repo root: .../GovGrant-AI
REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "user_docs")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))

    llamaparse_api_key: str | None = os.getenv("LLAMAPARSE_API_KEY") or None

    default_tenant_id: str = os.getenv("DEFAULT_TENANT_ID", "local-dev")
    fixtures_pdf_dir: Path = REPO_ROOT / "data" / "fixtures" / "pdfs"
    bm25_persist_dir: Path = REPO_ROOT / "data" / "indexes" / "bm25"
    # User proposals (tenant-scoped uploads)
    proposals_dir: Path = Path(
        os.getenv(
            "PROPOSALS_DIR",
            str(REPO_ROOT / "data" / "indexes" / "proposals"),
        )
    )
    proposals_db_path: Path = Path(
        os.getenv(
            "PROPOSALS_DB_PATH",
            str(REPO_ROOT / "data" / "indexes" / "proposals" / "proposals.sqlite"),
        )
    )
    tabular_db_path: Path = Path(
        os.getenv(
            "TABULAR_DB_PATH",
            str(REPO_ROOT / "data" / "indexes" / "tabular" / "tables.sqlite"),
        )
    )

    # SBIR Topics (R3)
    sbir_api_base_url: str = os.getenv(
        "SBIR_API_BASE_URL", "https://api.www.sbir.gov/public/api"
    )
    sbir_api_key: str | None = os.getenv("SBIR_API_KEY") or None
    sbir_qdrant_collection: str = os.getenv("SBIR_QDRANT_COLLECTION", "sbir_topics")
    sbir_use_fixtures_on_fail: bool = (
        os.getenv("SBIR_USE_FIXTURES_ON_FAIL", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    sbir_timeout_seconds: float = float(os.getenv("SBIR_TIMEOUT_SECONDS", "60"))
    sbir_fixture_path: Path = Path(
        os.getenv(
            "SBIR_FIXTURE_PATH",
            str(REPO_ROOT / "data" / "fixtures" / "sbir" / "open_solicitations.sample.json"),
        )
    )
    sbir_db_path: Path = Path(
        os.getenv(
            "SBIR_DB_PATH",
            str(REPO_ROOT / "data" / "indexes" / "sbir" / "topics.sqlite"),
        )
    )
    sbir_bm25_dir: Path = Path(
        os.getenv(
            "SBIR_BM25_DIR",
            str(REPO_ROOT / "data" / "indexes" / "sbir_bm25"),
        )
    )

    # Figures / charts (R4)
    figures_dir: Path = Path(
        os.getenv(
            "FIGURES_DIR",
            str(REPO_ROOT / "data" / "indexes" / "figures"),
        )
    )
    figures_max_per_doc: int = int(os.getenv("FIGURES_MAX_PER_DOC", "40"))
    # Optional local vision model via Ollama, e.g. llava, moondream, llama3.2-vision
    ollama_vision_model: str | None = os.getenv("OLLAMA_VISION_MODEL") or None
    ollama_vision_timeout: float = float(os.getenv("OLLAMA_VISION_TIMEOUT", "120"))

    # Chat LLM (Anthropic Haiku for agent answers)
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    chat_model: str = os.getenv("CHAT_MODEL", "claude-haiku-4-5-20251001")
    chat_max_tokens: int = int(os.getenv("CHAT_MAX_TOKENS", "1024"))
    chat_enabled: bool = (
        os.getenv("CHAT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    )

    # Hybrid retrieve defaults (higher recall for multi-section compliance docs)
    similarity_top_k: int = int(os.getenv("SIMILARITY_TOP_K", "12"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "12"))
    fusion_top_k: int = int(os.getenv("FUSION_TOP_K", "10"))


def get_settings() -> Settings:
    return Settings()
