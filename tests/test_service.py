from pathlib import Path

from rag_document_search.config import Settings
from rag_document_search.models import SourceDocument
from rag_document_search.service import RAGService
from tests.fakes import InMemoryIndex


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url="postgresql://rag:rag@localhost:5432/test_rag_document_search",
        chunk_size=120,
        chunk_overlap=20,
        top_k=3,
        hashing_dimensions=256,
        embedding_provider="hashing",
        llm_provider="extractive",
        openai_base_url="https://example.test/v1",
        openai_api_key=None,
        embedding_model="unused",
        chat_model="unused",
        request_timeout_seconds=5,
        github_api_url="https://api.github.com",
        github_graphql_url="https://api.github.com/graphql",
        github_token=None,
        github_oauth_client_id=None,
        github_oauth_client_secret=None,
        github_oauth_redirect_url=None,
        github_oauth_scopes=("read:user", "repo"),
        token_encryption_key=None,
        patchwork_web_url="http://127.0.0.1:5173",
        patchwork_default_pull_limit=15,
        sync_worker_poll_seconds=2,
        sync_lease_seconds=900,
    )


def test_ingest_search_answer_and_replace(tmp_path: Path) -> None:
    service = RAGService(make_settings(tmp_path), index=InMemoryIndex())
    document = SourceDocument(
        source="policy.md",
        text=(
            "Customer support conversations are retained for 24 months. "
            "Legal holds prevent deletion."
        ),
        content_type="text/markdown",
        metadata={"content_sha256": "v1"},
    )

    report = service.ingest_documents([document])
    assert report.documents_indexed == 1
    assert report.chunks_indexed == 1
    assert service.ingest_documents([document]).documents_unchanged == 1

    hits = service.search("How long are support conversations retained?")
    assert len(hits) == 1
    assert hits[0].source == "policy.md"
    assert "24 months" in service.ask("What is the retention period?").text

    replacement = SourceDocument(
        source="policy.md",
        text="Customer support conversations are retained for 36 months.",
        content_type="text/markdown",
        metadata={"content_sha256": "v2"},
    )
    assert service.ingest_documents([replacement]).documents_indexed == 1
    assert "36 months" in service.ask("What is the retention period?").text
    assert len(service.list_documents()) == 1


def test_empty_query_is_rejected(tmp_path: Path) -> None:
    service = RAGService(make_settings(tmp_path), index=InMemoryIndex())
    try:
        service.search("  ")
    except ValueError as error:
        assert "must not be empty" in str(error)
    else:
        raise AssertionError("Expected ValueError")
