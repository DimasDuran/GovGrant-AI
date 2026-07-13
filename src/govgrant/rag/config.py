"""Runtime configuration from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Repo root: .../GovGrant-AI
REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY") or None)
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "user_docs")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    )
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768")))
    llamaparse_api_key: str | None = field(
        default_factory=lambda: os.getenv("LLAMAPARSE_API_KEY") or None
    )
    default_tenant_id: str = field(
        default_factory=lambda: os.getenv("DEFAULT_TENANT_ID", "local-dev")
    )
    fixtures_pdf_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "fixtures" / "pdfs")
    proposals_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROPOSALS_DIR", str(REPO_ROOT / "data" / "indexes" / "proposals"))
        )
    )
    proposals_db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "PROPOSALS_DB_PATH",
                str(REPO_ROOT / "data" / "indexes" / "proposals" / "proposals.sqlite"),
            )
        )
    )
    tabular_db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "TABULAR_DB_PATH", str(REPO_ROOT / "data" / "indexes" / "tabular" / "tables.sqlite")
            )
        )
    )
    sbir_api_base_url: str = field(
        default_factory=lambda: os.getenv(
            "SBIR_API_BASE_URL", "https://api.www.sbir.gov/public/api"
        )
    )
    sbir_api_key: str | None = field(default_factory=lambda: os.getenv("SBIR_API_KEY") or None)
    sbir_qdrant_collection: str = field(
        default_factory=lambda: os.getenv("SBIR_QDRANT_COLLECTION", "sbir_topics")
    )
    sbir_use_fixtures_on_fail: bool = field(
        default_factory=lambda: (
            os.getenv("SBIR_USE_FIXTURES_ON_FAIL", "true").lower() in {"1", "true", "yes", "on"}
        )
    )
    sbir_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("SBIR_TIMEOUT_SECONDS", "60"))
    )
    sbir_fixture_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "SBIR_FIXTURE_PATH",
                str(REPO_ROOT / "data" / "fixtures" / "sbir" / "open_solicitations.sample.json"),
            )
        )
    )
    sbir_db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "SBIR_DB_PATH", str(REPO_ROOT / "data" / "indexes" / "sbir" / "topics.sqlite")
            )
        )
    )
    figures_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("FIGURES_DIR", str(REPO_ROOT / "data" / "indexes" / "figures"))
        )
    )
    figures_max_per_doc: int = field(
        default_factory=lambda: int(os.getenv("FIGURES_MAX_PER_DOC", "40"))
    )
    ollama_vision_model: str | None = field(
        default_factory=lambda: os.getenv("OLLAMA_VISION_MODEL") or None
    )
    ollama_vision_timeout: float = field(
        default_factory=lambda: float(os.getenv("OLLAMA_VISION_TIMEOUT", "120"))
    )
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None
    )
    chat_model: str = field(
        default_factory=lambda: os.getenv("CHAT_MODEL", "claude-haiku-4-5-20251001")
    )
    chat_max_tokens: int = field(default_factory=lambda: int(os.getenv("CHAT_MAX_TOKENS", "1024")))
    chat_enabled: bool = field(
        default_factory=lambda: (
            os.getenv("CHAT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        )
    )
    similarity_top_k: int = field(default_factory=lambda: int(os.getenv("SIMILARITY_TOP_K", "12")))
    fusion_top_k: int = field(default_factory=lambda: int(os.getenv("FUSION_TOP_K", "10")))


def get_settings() -> Settings:
    return Settings()
