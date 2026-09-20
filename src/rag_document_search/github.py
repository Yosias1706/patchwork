"""Small GitHub REST client and patch-card builder for Patchwork."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .models import SourceDocument

_REPAIR_WORDS = re.compile(
    r"\b("
    r"bug|fix(?:ed|es|ing)?|crash|regression|deadlock|error|panic|"
    r"hang|timeout|leak|race|retry|recover|prevent|avoid|guard|"
    r"correct|incorrect|invalid|failure|broken|cleanup|validate|sanitize"
    r")\b",
    re.I,
)
_CLOSING_ISSUE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+(?:[\w-]+/[\w.-]+)?#(\d+)\b", re.I
)


class GitHubError(RuntimeError):
    """A recoverable error from a GitHub API request or repository URL."""


class GitHubRateLimited(GitHubError):
    """GitHub asked Patchwork to pause before continuing a durable sync."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = max(30, retry_after_seconds)
        super().__init__(
            "GitHub rate-limited this sync. Patchwork will resume automatically after "
            f"about {self.retry_after_seconds // 60 or 1} minute(s)."
        )


@dataclass(frozen=True, slots=True)
class GitHubRepository:
    github_url: str
    full_name: str
    owner: str
    name: str
    default_branch: str
    description: str | None
    is_private: bool

    def as_storage_details(self) -> dict[str, object]:
        return {
            "github_url": self.github_url,
            "full_name": self.full_name,
            "owner": self.owner,
            "name": self.name,
            "default_branch": self.default_branch,
            "description": self.description,
            "is_private": self.is_private,
        }


@dataclass(frozen=True, slots=True)
class PatchCard:
    document: SourceDocument
    pull_number: int
    linked_issue_number: int | None
    linked_issue_provenance: str | None


@dataclass(frozen=True, slots=True)
class GitHubIssueLink:
    issue: dict[str, Any]
    provenance: str


def parse_github_repository_url(value: str) -> tuple[str, str]:
    """Return owner/repository from a canonical public GitHub repository URL."""
    parsed = urllib.parse.urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != "github.com":
        raise GitHubError(
            "Enter a GitHub repository URL such as https://github.com/owner/repository"
        )
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        raise GitHubError("The URL must point to a repository, not a file, pull request, or issue")
    owner, repository = parts
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository:
        raise GitHubError("The GitHub repository URL is incomplete")
    return owner, repository


class GitHubClient:
    """Read-only GitHub REST client for public and token-authorized repositories."""

    def __init__(
        self,
        api_url: str,
        *,
        token: str | None = None,
        timeout: int = 45,
        graphql_url: str | None = None,
        minimum_request_interval_seconds: float = 0.8,
        oauth_client_id: str | None = None,
        oauth_client_secret: str | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.graphql_url = (graphql_url or f"{self.api_url}/graphql").rstrip("/")
        self.minimum_request_interval_seconds = max(0.0, minimum_request_interval_seconds)
        self._next_request_at = 0.0
        self.oauth_client_id = oauth_client_id
        self.oauth_client_secret = oauth_client_secret

    def repository(self, github_url: str) -> GitHubRepository:
        owner, name = parse_github_repository_url(github_url)
        payload = self._get(f"/repos/{owner}/{name}")
        full_name = str(payload["full_name"])
        return GitHubRepository(
            github_url=f"https://github.com/{full_name}",
            full_name=full_name,
            owner=str(payload["owner"]["login"]),
            name=str(payload["name"]),
            default_branch=str(payload.get("default_branch") or "main"),
            description=_optional_text(payload.get("description")),
            is_private=bool(payload.get("private", False)),
        )

    def viewer(self) -> dict[str, Any]:
        """Return the GitHub account associated with this client's bearer token."""
        if not self.token:
            raise GitHubError("A GitHub access token is required to identify the connected account")
        payload = self._get("/user")
        if not isinstance(payload, dict) or not payload.get("id") or not payload.get("login"):
            raise GitHubError("GitHub did not return a valid user profile")
        return payload

    def patch_cards(
        self, repository: GitHubRepository, *, limit: int
    ) -> tuple[int, list[PatchCard]]:
        """Build repair-history cards from bug-labelled and repair-like merged PRs."""
        if not 1 <= limit <= 10_000:
            raise ValueError("Patchwork supports between 1 and 10000 historical fixes per sync")
        all_pulls = self._closed_pull_history(repository)
        labelled_pulls = [
            pull for pull in all_pulls if _has_bug_label(pull) and pull.get("number")
        ]
        pulls = _balanced_repair_candidates(labelled_pulls, all_pulls, limit)

        cards: list[PatchCard] = []
        for summary in pulls:
            card = self.patch_card(repository, int(summary["number"]))
            if card:
                cards.append(card)
        return len(all_pulls), cards

    def closed_pull_page(
        self, repository: GitHubRepository, *, page: int
    ) -> list[dict[str, Any]]:
        """Return one stable, 100-PR page of a repository's closed history."""
        if not 1 <= page <= 100:
            raise ValueError("Patchwork supports up to 10,000 closed pull requests per sync")
        pulls = self._get(
            f"/repos/{repository.owner}/{repository.name}/pulls",
            {
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
                "page": page,
            },
        )
        if not isinstance(pulls, list):
            raise GitHubError("GitHub returned an unexpected pull-request listing")
        return [pull for pull in pulls if isinstance(pull, dict)]

    def patch_card(
        self, repository: GitHubRepository, pull_number: int
    ) -> PatchCard | None:
        """Fetch one candidate's evidence, intended for a durable worker batch."""
        pull = self._get(f"/repos/{repository.owner}/{repository.name}/pulls/{pull_number}")
        if not isinstance(pull, dict) or not pull.get("merged_at") or not _looks_like_bug_fix(pull):
            return None
        files = self._get(
            f"/repos/{repository.owner}/{repository.name}/pulls/{pull_number}/files",
            {"per_page": 100},
        )
        if not isinstance(files, list):
            raise GitHubError(f"GitHub returned invalid file data for pull request #{pull_number}")
        linked_issue = self._linked_issue(repository, pull)
        return _patch_card(repository, pull, files, linked_issue)

    def _closed_pull_history(self, repository: GitHubRepository) -> list[dict[str, Any]]:
        """Read all closed PR summaries without fetching every diff up front.

        GitHub serves up to 100 pull-request summaries per page. Detailed PR and
        file calls are reserved for the selected repair candidates, keeping a
        complete history scan practical for large repositories.
        """
        history: list[dict[str, Any]] = []
        for page in range(1, 101):  # 10,000 PRs: a deliberate, bounded ceiling.
            pulls = self.closed_pull_page(repository, page=page)
            history.extend(pulls)
            if len(pulls) < 100:
                break
        return history

    def _linked_issue(
        self, repository: GitHubRepository, pull: dict[str, Any]
    ) -> GitHubIssueLink | None:
        """Find a GitHub-confirmed issue/PR relationship when credentials permit.

        The GraphQL closing-reference field includes linked issues that are not
        written with a closing keyword in the PR body. Public unauthenticated
        syncs retain the explicit-keyword fallback so the product remains useful.
        """
        if self.token:
            try:
                linked = self._closing_issue_references(repository, int(pull["number"]))
                if linked:
                    return GitHubIssueLink(linked[0], "github_closing_reference")
            except GitHubError:
                # The REST ingestion is still valid if GraphQL is unavailable
                # or an organization restricts this optional field.
                pass
        issue_numbers = _closing_issue_numbers(_optional_text(pull.get("body")) or "")
        if not issue_numbers:
            return None
        try:
            issue = self._get(
                f"/repos/{repository.owner}/{repository.name}/issues/{issue_numbers[0]}"
            )
        except GitHubError:
            return None
        if not isinstance(issue, dict) or "pull_request" in issue:
            return None
        return GitHubIssueLink(issue, "pr_closing_keyword")

    def _closing_issue_references(
        self, repository: GitHubRepository, pull_number: int
    ) -> list[dict[str, Any]]:
        payload = self._post_json(
            self.graphql_url,
            {
                "query": """
                    query PatchworkClosingIssues($owner: String!, $name: String!, $number: Int!) {
                      repository(owner: $owner, name: $name) {
                        pullRequest(number: $number) {
                          closingIssuesReferences(first: 10) {
                            nodes { number title body url state closedAt }
                          }
                        }
                      }
                    }
                """,
                "variables": {
                    "owner": repository.owner,
                    "name": repository.name,
                    "number": pull_number,
                },
            },
        )
        errors = payload.get("errors") if isinstance(payload, dict) else None
        if errors:
            raise GitHubError("GitHub could not read pull-request closing references")
        nodes = (
            payload.get("data", {})
            .get("repository", {})
            .get("pullRequest", {})
            .get("closingIssuesReferences", {})
            .get("nodes", [])
            if isinstance(payload, dict)
            else []
        )
        return [node for node in nodes if isinstance(node, dict) and node.get("number")]

    def _get(self, path: str, query: dict[str, object] | None = None) -> Any:
        url = f"{self.api_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        headers = self._headers()
        request = urllib.request.Request(url, headers=headers)
        retryable_error: GitHubError | None = None
        for attempt in range(2):
            try:
                self._wait_for_request_slot()
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    self._record_request_slot()
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:350]
                if error.code in {403, 429}:
                    raise GitHubRateLimited(_retry_after_seconds(error.headers)) from error
                if error.code == 404:
                    raise GitHubError(
                        "GitHub could not find this repository. If it is private, link your "
                        "GitHub account and retry."
                    ) from error
                retryable_error = GitHubError(f"GitHub returned HTTP {error.code}: {detail}")
                if error.code < 500:
                    raise retryable_error from error
            except (TimeoutError, urllib.error.URLError):
                retryable_error = GitHubError(
                    "GitHub did not respond before the request timeout. Retry the sync shortly."
                )
            if attempt == 0:
                time.sleep(0.25)
        if retryable_error:
            raise retryable_error
        raise RuntimeError("GitHub request ended without a response or an error")

    def _post_json(self, url: str, payload: dict[str, object]) -> Any:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={**self._headers(), "Content-Type": "application/json"},
            method="POST",
        )
        try:
            self._wait_for_request_slot()
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self._record_request_slot()
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:350]
            if error.code in {403, 429}:
                raise GitHubRateLimited(_retry_after_seconds(error.headers)) from error
            raise GitHubError(f"GitHub returned HTTP {error.code}: {detail}") from error
        except (TimeoutError, urllib.error.URLError) as error:
            raise GitHubError("GitHub did not respond before the request timeout") from error

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Patchwork-RAG/0.1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.oauth_client_id and self.oauth_client_secret:
            credentials = f"{self.oauth_client_id}:{self.oauth_client_secret}".encode()
            headers["Authorization"] = f"Basic {base64.b64encode(credentials).decode('ascii')}"
        return headers

    def _wait_for_request_slot(self) -> None:
        delay = self._next_request_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _record_request_slot(self) -> None:
        self._next_request_at = time.monotonic() + self.minimum_request_interval_seconds


def _looks_like_bug_fix(pull: dict[str, Any]) -> bool:
    labels = " ".join(
        str(label.get("name", "")) for label in pull.get("labels", []) if isinstance(label, dict)
    )
    candidate_text = " ".join(
        part
        for part in (str(pull.get("title", "")), _optional_text(pull.get("body")), labels)
        if part
    )
    return bool(_REPAIR_WORDS.search(candidate_text)) or bool(
        _CLOSING_ISSUE.search(_optional_text(pull.get("body")) or "")
    )


def _has_bug_label(pull: dict[str, Any]) -> bool:
    return any(
        str(label.get("name", "")).strip().lower() == "bug"
        for label in pull.get("labels", [])
        if isinstance(label, dict)
    )


def is_repair_candidate(pull: dict[str, Any]) -> bool:
    """Use labels and repair language to keep unrelated feature PRs out of the RAG corpus."""
    return bool(pull.get("number")) and (_has_bug_label(pull) or _looks_like_bug_fix(pull))


def _retry_after_seconds(headers: Any) -> int:
    retry_after = headers.get("Retry-After") if headers else None
    if retry_after and str(retry_after).isdigit():
        return int(retry_after) + 5
    reset = headers.get("X-RateLimit-Reset") if headers else None
    if reset and str(reset).isdigit():
        return max(30, int(reset) - int(time.time()) + 5)
    return 60


def _balanced_repair_candidates(
    labelled_pulls: list[dict[str, Any]], recent_pulls: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Retain labelled bugs while reserving room for unlabelled repair signals."""
    labelled = [pull for pull in labelled_pulls if pull.get("number")]
    repair_like = [
        pull for pull in recent_pulls if _looks_like_bug_fix(pull) and pull.get("number")
    ]
    labelled_quota = max(1, (limit * 3 + 4) // 5)
    chosen: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(candidates: list[dict[str, Any]], maximum: int) -> None:
        for candidate in candidates:
            number = int(candidate["number"])
            if number in seen or len(chosen) >= maximum:
                continue
            chosen.append(candidate)
            seen.add(number)

    add(labelled, labelled_quota)
    add(repair_like, limit)
    add(labelled, limit)
    return chosen[:limit]


def _closing_issue_numbers(body: str) -> list[int]:
    return list(dict.fromkeys(int(number) for number in _CLOSING_ISSUE.findall(body)))


def _patch_card(
    repository: GitHubRepository,
    pull: dict[str, Any],
    files: list[dict[str, Any]],
    linked_issue: GitHubIssueLink | None,
) -> PatchCard:
    pull_number = int(pull["number"])
    issue_numbers = _closing_issue_numbers(_optional_text(pull.get("body")) or "")
    issue = linked_issue.issue if linked_issue else None
    linked_issue_number = int(issue["number"]) if issue else None
    changed_files = [str(file.get("filename")) for file in files if file.get("filename")]
    test_files = [path for path in changed_files if _is_test_file(path)]
    patch_sections = [_file_patch_section(file) for file in files[:12]]
    labels = [
        str(label.get("name"))
        for label in pull.get("labels", [])
        if isinstance(label, dict) and label.get("name")
    ]
    body = _truncate(
        _optional_text(pull.get("body")) or "No pull-request description provided.", 4500
    )
    linked_issues = ", ".join(f"#{number}" for number in issue_numbers) or "not declared"
    changed_tests = "\n".join(f"- {path}" for path in test_files) or "No test file detected."
    text = "\n\n".join(
        part
        for part in [
            f"# Historical bug fix: {pull['title']}",
            (
                f"Repository: {repository.full_name}\n"
                f"Pull request: #{pull_number}\n"
                f"Merged: {pull.get('merged_at')}\n"
                f"Commit: {pull.get('merge_commit_sha')}\n"
                f"Labels: {', '.join(labels) or 'none'}\n"
                f"Linked issues: {linked_issues}"
            ),
            "## Problem and resolution\n" + body,
            "## Changed tests\n" + changed_tests,
            "## Relevant diff hunks\n" + "\n\n".join(patch_sections),
        ]
        if part
    )
    metadata = {
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "patchwork_repository_full_name": repository.full_name,
        "pull_number": pull_number,
        "pull_url": str(pull["html_url"]),
        "merge_commit_sha": _optional_text(pull.get("merge_commit_sha")),
        "merged_at": _optional_text(pull.get("merged_at")),
        "linked_issue_number": linked_issue_number,
        "linked_issue_url": (
            _optional_text(issue.get("html_url") or issue.get("url")) if issue else None
        ),
        "benchmark_query": _benchmark_query(issue),
        "benchmark_provenance": linked_issue.provenance if linked_issue else None,
        "changed_files": changed_files[:60],
        "test_files": test_files[:30],
        "labels": labels,
        "filename": f"PR #{pull_number} · {pull['title']}",
        "suffix": "patch",
    }
    document = SourceDocument(
        source=str(pull["html_url"]),
        text=text,
        content_type="text/x-patchwork-card",
        metadata=metadata,
    )
    return PatchCard(
        document=document,
        pull_number=pull_number,
        linked_issue_number=linked_issue_number,
        linked_issue_provenance=linked_issue.provenance if linked_issue else None,
    )


def _benchmark_query(issue: dict[str, Any] | None) -> str | None:
    if not issue:
        return None
    title = _optional_text(issue.get("title")) or ""
    body = _truncate(_optional_text(issue.get("body")) or "", 2500)
    query = f"{title}\n\n{body}".strip()
    return query or None


def _file_patch_section(file: dict[str, Any]) -> str:
    filename = str(file.get("filename") or "unknown file")
    status = str(file.get("status") or "modified")
    patch = _truncate(_optional_text(file.get("patch")) or "Patch unavailable from GitHub.", 3500)
    return f"### {filename} ({status})\n```diff\n{patch}\n```"


def _is_test_file(path: str) -> bool:
    normalized = path.lower()
    return "/test" in normalized or normalized.startswith("test") or "spec" in normalized


def _optional_text(value: object) -> str | None:
    return str(value) if value is not None else None


def _truncate(value: str, maximum: int) -> str:
    return value if len(value) <= maximum else f"{value[:maximum]}\n… [truncated]"
