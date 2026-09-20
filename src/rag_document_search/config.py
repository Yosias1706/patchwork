"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    hashing_dimensions: int
    embedding_provider: str
    llm_provider: str
    openai_base_url: str
    openai_api_key: str | None
    embedding_model: str
    chat_model: str
    request_timeout_seconds: int
    github_api_url: str
    github_graphql_url: str
    github_token: str | None
    github_oauth_client_id: str | None
    github_oauth_client_secret: str | None
    github_oauth_redirect_url: str | None
    github_oauth_scopes: tuple[str, ...]
    token_encryption_key: str | None
    patchwork_web_url: str
    patchwork_default_pull_limit: int
    sync_worker_poll_seconds: int
    sync_lease_seconds: int

    @classmethod
    def from_environment(cls) -> Settings:
        chunk_size = _positive_int("RAG_CHUNK_SIZE", 900)
        chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "RAG_CHUNK_OVERLAP must be non-negative and smaller than RAG_CHUNK_SIZE"
            )

        embedding_provider = os.getenv("RAG_EMBEDDING_PROVIDER", "hashing").lower()
        llm_provider = os.getenv("RAG_LLM_PROVIDER", "extractive").lower()
        if embedding_provider not in {"hashing", "openai_compatible"}:
            raise ValueError(f"Unsupported RAG_EMBEDDING_PROVIDER: {embedding_provider}")
        if llm_provider not in {"extractive", "openai_compatible"}:
            raise ValueError(f"Unsupported RAG_LLM_PROVIDER: {llm_provider}")

        patchwork_default_pull_limit = _positive_int("PATCHWORK_DEFAULT_PULL_LIMIT", 300)
        if patchwork_default_pull_limit > 300:
            raise ValueError("PATCHWORK_DEFAULT_PULL_LIMIT must be at most 300")

        # Render provides its public URL at runtime. Explicit Patchwork settings
        # still win, which keeps custom-domain and local deployments predictable.
        public_url = (
            os.getenv("PATCHWORK_WEB_URL")
            or os.getenv("PATCHWORK_PUBLIC_URL")
            or os.getenv("RENDER_EXTERNAL_URL")
            or "http://127.0.0.1:5173"
        ).rstrip("/")
        oauth_host = (
            os.getenv("PATCHWORK_PUBLIC_URL") or os.getenv("RENDER_EXTERNAL_URL") or ""
        ).rstrip("/")
        oauth_redirect_url = os.getenv("PATCHWORK_GITHUB_OAUTH_REDIRECT_URL") or (
            f"{oauth_host}/auth/github/callback" if oauth_host else None
        )

        return cls(
            database_url=os.getenv(
                "RAG_DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag_document_search"
            ),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=_positive_int("RAG_TOP_K", 5),
            hashing_dimensions=_positive_int("RAG_HASHING_DIMENSIONS", 768),
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            openai_base_url=os.getenv("RAG_OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_api_key=os.getenv("RAG_OPENAI_API_KEY") or None,
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            chat_model=os.getenv("RAG_CHAT_MODEL", "gpt-4.1-mini"),
            request_timeout_seconds=_positive_int("RAG_REQUEST_TIMEOUT_SECONDS", 45),
            github_api_url=os.getenv("PATCHWORK_GITHUB_API_URL", "https://api.github.com").rstrip("/"),
            github_graphql_url=os.getenv(
                "PATCHWORK_GITHUB_GRAPHQL_URL", "https://api.github.com/graphql"
            ).rstrip("/"),
            github_token=os.getenv("PATCHWORK_GITHUB_TOKEN") or None,
            github_oauth_client_id=os.getenv("PATCHWORK_GITHUB_OAUTH_CLIENT_ID") or None,
            github_oauth_client_secret=os.getenv("PATCHWORK_GITHUB_OAUTH_CLIENT_SECRET") or None,
            github_oauth_redirect_url=oauth_redirect_url,
            github_oauth_scopes=tuple(
                scope
                for scope in os.getenv("PATCHWORK_GITHUB_OAUTH_SCOPES", "read:user repo").split()
                if scope
            ),
            token_encryption_key=os.getenv("PATCHWORK_TOKEN_ENCRYPTION_KEY") or None,
            patchwork_web_url=public_url,
            patchwork_default_pull_limit=patchwork_default_pull_limit,
            sync_worker_poll_seconds=_positive_int("PATCHWORK_SYNC_WORKER_POLL_SECONDS", 2),
            sync_lease_seconds=_positive_int("PATCHWORK_SYNC_LEASE_SECONDS", 900),
        )
