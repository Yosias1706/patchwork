from __future__ import annotations

import urllib.parse
from datetime import UTC, datetime

import pytest
from cryptography.fernet import Fernet

from rag_document_search.config import Settings
from rag_document_search.github_oauth import GitHubOAuthError, GitHubOAuthService


class FakeGitHubOAuthIndex:
    def __init__(self) -> None:
        self.states: dict[str, str] = {}
        self.connections: dict[str, dict[str, object]] = {}

    def create_github_oauth_state(
        self, *, state_hash: str, user_id: str, expires_at: object
    ) -> None:
        assert expires_at > datetime.now(UTC)
        self.states[state_hash] = user_id

    def consume_github_oauth_state(self, state_hash: str) -> str | None:
        return self.states.pop(state_hash, None)

    def upsert_github_connection(self, **values: object) -> dict[str, object]:
        user_id = str(values.pop("user_id"))
        connection = {
            **values,
            "created_at": "now",
            "updated_at": "now",
        }
        self.connections[user_id] = connection
        return connection

    def get_github_connection(self, user_id: str) -> dict[str, object] | None:
        return self.connections.get(user_id)

    def delete_github_connection(self, user_id: str) -> bool:
        return self.connections.pop(user_id, None) is not None


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
        github_oauth_client_id="client-id",
        github_oauth_client_secret="client-secret",
        github_oauth_redirect_url="http://localhost:8000/auth/github/callback",
        github_oauth_scopes=("read:user", "repo"),
        token_encryption_key=Fernet.generate_key().decode("ascii"),
        patchwork_web_url="http://localhost:5173",
        patchwork_default_pull_limit=15,
        sync_worker_poll_seconds=2,
        sync_lease_seconds=900,
    )


def test_github_oauth_persists_only_encrypted_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    index = FakeGitHubOAuthIndex()
    oauth = GitHubOAuthService(index, _settings())  # type: ignore[arg-type]
    url = oauth.authorization_url("patchwork-user")
    state = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["state"][0]

    monkeypatch.setattr(
        oauth,
        "_exchange_code",
        lambda _code: {"access_token": "github-access-token", "scope": "read:user,repo"},
    )
    monkeypatch.setattr(
        "rag_document_search.github_oauth.GitHubClient.viewer",
        lambda _client: {"id": 17, "login": "patchwork-engineer"},
    )

    result = oauth.complete_callback(code="temporary-code", state=state)

    assert result["github_login"] == "patchwork-engineer"
    assert "github-access-token" not in str(index.connections["patchwork-user"])
    assert oauth.access_token_for_user("patchwork-user") == "github-access-token"
    assert oauth.status("patchwork-user")["connected"] is True

    with pytest.raises(GitHubOAuthError, match="expired"):
        oauth.complete_callback(code="temporary-code", state=state)
