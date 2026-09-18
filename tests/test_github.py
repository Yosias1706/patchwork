import pytest

from rag_document_search.github import (
    GitHubClient,
    GitHubError,
    GitHubIssueLink,
    GitHubRepository,
    _closing_issue_numbers,
    _patch_card,
    parse_github_repository_url,
)


def test_parse_github_repository_url() -> None:
    assert parse_github_repository_url("https://github.com/fastapi/fastapi/") == (
        "fastapi",
        "fastapi",
    )
    assert parse_github_repository_url("https://github.com/pallets/flask.git") == (
        "pallets",
        "flask",
    )


def test_parse_github_repository_url_rejects_non_repository_urls() -> None:
    with pytest.raises(GitHubError, match="repository"):
        parse_github_repository_url("https://github.com/fastapi/fastapi/pulls")


def test_patch_card_keeps_diff_and_linked_issue_metadata() -> None:
    repository = GitHubRepository(
        github_url="https://github.com/example/api",
        full_name="example/api",
        owner="example",
        name="api",
        default_branch="main",
        description=None,
        is_private=False,
    )
    pull = {
        "number": 42,
        "title": "Fix disconnect cleanup race",
        "body": "Fixes #17 when a streaming client disconnects.",
        "html_url": "https://github.com/example/api/pull/42",
        "merged_at": "2026-01-01T00:00:00Z",
        "merge_commit_sha": "abcdef123456",
        "labels": [{"name": "bug"}],
    }
    files = [
        {
            "filename": "src/streaming.py",
            "status": "modified",
            "patch": "- cleanup()\n+ await cleanup()",
        },
        {
            "filename": "tests/test_streaming.py",
            "status": "added",
            "patch": "+ def test_disconnect_cleanup(): ...",
        },
    ]
    issue = {
        "number": 17,
        "title": "Stream hangs after disconnect",
        "body": "A disconnect leaves the request pending.",
        "html_url": "https://github.com/example/api/issues/17",
    }

    card = _patch_card(repository, pull, files, GitHubIssueLink(issue, "github_closing_reference"))

    assert card.linked_issue_number == 17
    assert card.document.metadata["pull_number"] == 42
    assert card.document.metadata["test_files"] == ["tests/test_streaming.py"]
    assert card.document.metadata["benchmark_provenance"] == "github_closing_reference"
    assert "Relevant diff hunks" in card.document.text
    assert "Stream hangs after disconnect" not in card.document.text


def test_closing_issue_numbers_uses_only_closing_references() -> None:
    assert _closing_issue_numbers("Related to #4. Fixes #9 and resolves #11.") == [9, 11]


def test_github_timeout_becomes_a_recoverable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed_out(*_args: object, **_kwargs: object) -> object:
        raise TimeoutError("network stalled")

    monkeypatch.setattr("rag_document_search.github.urllib.request.urlopen", timed_out)
    monkeypatch.setattr("rag_document_search.github.time.sleep", lambda _seconds: None)

    with pytest.raises(GitHubError, match="did not respond"):
        GitHubClient("https://api.github.com", timeout=1)._get("/repos/example/api")
