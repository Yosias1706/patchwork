from rag_document_search.config import Settings
from rag_document_search.investigator import InvestigationAgent
from rag_document_search.models import SourceDocument
from rag_document_search.service import RAGService
from tests.fakes import InMemoryIndex


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://rag:rag@localhost:5432/test_patchwork",
        chunk_size=800,
        chunk_overlap=100,
        top_k=5,
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


def test_investigation_agent_uses_multiple_scoped_searches_and_fuses_sources() -> None:
    service = RAGService(_settings(), index=InMemoryIndex())
    service.ingest_documents(
        [
            SourceDocument(
                source="https://github.com/example/api/pull/42",
                text=(
                    "Historical bug fix for ConnectionResetError in src/stream.py. "
                    "The cleanup task kept writing after the client disconnected. "
                    "Changed tests/test_stream.py to cover disconnect cleanup."
                ),
                content_type="text/x-patchwork-card",
                metadata={
                    "content_sha256": "repo-a-fix",
                    "patchwork_repository_id": "repo-a",
                    "test_files": ["tests/test_stream.py"],
                },
            ),
            SourceDocument(
                source="https://github.com/other/api/pull/9",
                text="ConnectionResetError has a fix in another repository.",
                content_type="text/x-patchwork-card",
                metadata={
                    "content_sha256": "repo-b-fix",
                    "patchwork_repository_id": "repo-b",
                    "test_files": ["tests/test_other.py"],
                },
            ),
        ]
    )

    outcome = InvestigationAgent(service).run(
        "repo-a",
        "ConnectionResetError occurs after disconnect in src/stream.py while writing a stream.",
    )

    purposes = {directive.purpose for directive in outcome.retrieval.plan.directives}
    assert {"incident_report", "error_signature", "code_location"}.issubset(purposes)
    assert [hit.source for hit in outcome.retrieval.citations] == [
        "https://github.com/example/api/pull/42"
    ]
    assert outcome.retrieval.confidence == "high"
    assert "ConnectionResetError" in outcome.answer
