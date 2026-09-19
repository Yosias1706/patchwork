"""Patchwork orchestration: repository syncs, investigations, and evaluations."""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from .github import GitHubClient, GitHubError, GitHubRepository, PatchCard
from .github_oauth import GitHubOAuthService
from .investigator import InvestigationAgent
from .models import SourceDocument
from .repository import PostgresIndex
from .service import RAGService


class PatchworkService:
    def __init__(self, rag: RAGService) -> None:
        if not isinstance(rag.index, PostgresIndex):
            raise TypeError("Patchwork requires the PostgreSQL index")
        self.rag = rag
        self.index = rag.index
        self.github_oauth = GitHubOAuthService(self.index, rag.settings)
        self.investigator = InvestigationAgent(rag)

    def connect_repository(
        self, github_url: str, *, user_id: str, pull_limit: int | None = None
    ) -> tuple[dict[str, object], dict[str, object]]:
        repository = self._github_client(user_id).repository(github_url)
        stored_repository = self.index.upsert_patchwork_repository(
            repository.as_storage_details(), user_id=user_id
        )
        requested_limit = pull_limit or self.rag.settings.patchwork_default_pull_limit
        run = self.index.create_patchwork_sync_run(
            str(stored_repository["id"]), requested_limit
        )
        return stored_repository, run

    def sync_repository(self, repository_id: str, run_id: str) -> None:
        repository = self.index.get_patchwork_repository(repository_id)
        if not repository:
            raise KeyError(f"Repository not found: {repository_id}")
        run = self.index.get_patchwork_sync_run(run_id)
        if not run:
            raise KeyError(f"Sync run not found: {run_id}")
        self.index.update_patchwork_sync_run(run_id, status="running")
        try:
            owner_user_id = self.index.get_patchwork_repository_owner(repository_id)
            if not owner_user_id:
                raise KeyError(f"Repository owner not found: {repository_id}")
            github_repository = GitHubRepository(
                github_url=str(repository["github_url"]),
                full_name=str(repository["full_name"]),
                owner=str(repository["owner"]),
                name=str(repository["name"]),
                default_branch=str(repository["default_branch"]),
                description=_string_or_none(repository["description"]),
                is_private=bool(repository["is_private"]),
            )
            seen, cards = self._github_client(owner_user_id).patch_cards(
                github_repository, limit=int(run["requested_pull_limit"])
            )
            report = self.rag.ingest_documents(
                _repository_documents(card.document, repository_id) for card in cards
            )
            self.index.replace_patchwork_evaluation_cases(
                repository_id, _evaluation_manifest(cards, repository_id)
            )
            failures = "; ".join(report.failures) if report.failures else None
            self.index.update_patchwork_sync_run(
                run_id,
                status="completed",
                pull_requests_seen=seen,
                patch_cards_indexed=report.documents_indexed + report.documents_unchanged,
                chunks_indexed=report.chunks_indexed,
                error=failures,
            )
        except (GitHubError, OSError, RuntimeError, ValueError) as error:
            self.index.update_patchwork_sync_run(run_id, status="failed", error=str(error))

    def repository(self, repository_id: str, *, user_id: str) -> dict[str, object] | None:
        repository = self.index.get_patchwork_repository(repository_id, user_id=user_id)
        if repository:
            repository["latest_evaluation"] = self.index.get_latest_patchwork_evaluation(
                repository_id
            )
        return repository

    def repositories(self, *, user_id: str) -> list[dict[str, object]]:
        repositories = self.index.list_patchwork_repositories(user_id=user_id)
        for repository in repositories:
            repository["latest_evaluation"] = self.index.get_latest_patchwork_evaluation(
                str(repository["id"])
            )
        return repositories

    def investigate(
        self, repository_id: str, issue: str, *, user_id: str, limit: int = 5
    ) -> dict[str, object]:
        if not self.index.get_patchwork_repository(repository_id, user_id=user_id):
            raise KeyError(f"Repository not found: {repository_id}")
        retrieval = self.investigator.retrieve(repository_id, issue, limit=limit)
        matches = _deduplicated_patch_matches(retrieval.citations)
        summary = _investigation_summary(matches)
        result = {
            "issue": issue,
            "summary": summary,
            "answer": summary,
            "matches": matches,
            "citations": [asdict(hit) for hit in retrieval.citations],
            "confidence": retrieval.confidence,
            "agent": retrieval.as_api_details(),
        }
        saved = self.index.save_patchwork_investigation(
            repository_id=repository_id,
            user_id=user_id,
            issue=issue,
            result=result,
        )
        return {**result, "investigation_id": saved["id"], "saved_at": saved["created_at"]}

    def evaluate(
        self, repository_id: str, *, user_id: str, limit: int = 5
    ) -> dict[str, object]:
        if not self.index.get_patchwork_repository(repository_id, user_id=user_id):
            raise KeyError(f"Repository not found: {repository_id}")
        run = self.index.create_patchwork_evaluation_run(repository_id)
        try:
            benchmarks = self.index.list_patchwork_evaluation_cases(repository_id)
            if not benchmarks:
                # Existing local documents may contain an issue body in their
                # patch card, so they receive only a clearly-labelled smoke
                # test until a fresh sync creates a leakage-safe manifest.
                benchmarks = _legacy_evaluation_cases(
                    self.index.list_documents_for_repository(repository_id)
                )
            reciprocal_ranks: list[float] = []
            retrieved_at_one = 0
            retrieved_at_three = 0
            failures: list[dict[str, object]] = []
            for benchmark in benchmarks:
                query = str(benchmark["query"])
                source_kind = str(benchmark["provenance"])
                hits = self.investigator.retrieve(repository_id, query, limit=limit).citations
                rank = next(
                    (
                        position
                        for position, hit in enumerate(hits, start=1)
                        if hit.source == benchmark["expected_source"]
                    ),
                    None,
                )
                reciprocal_ranks.append(1 / rank if rank else 0.0)
                retrieved_at_one += int(rank == 1)
                retrieved_at_three += int(rank is not None and rank <= 3)
                if not rank and len(failures) < 10:
                    failures.append(
                        {
                            "expected_pull_number": benchmark.get("expected_pull_number"),
                            "issue_url": benchmark.get("issue_url"),
                            "query": query[:280],
                            "failure_type": "retrieval_miss",
                            "source_kind": source_kind,
                        }
                    )
            total = len(benchmarks)
            verified_cases = sum(
                1
                for benchmark in benchmarks
                if benchmark["provenance"] != "pr_description_smoke_test"
            )
            smoke_test_cases = total - verified_cases
            metrics: dict[str, object] = {
                "recall_at_1": round(retrieved_at_one / total, 3) if total else None,
                "recall_at_3": round(retrieved_at_three / total, 3) if total else None,
                "mrr": round(sum(reciprocal_ranks) / total, 3) if total else None,
                "failure_samples": failures,
                "quality": "verified_issue_to_fix" if verified_cases else "smoke_test",
                "case_breakdown": {
                    "verified_issue_to_fix": verified_cases,
                    "pr_description_smoke_test": smoke_test_cases,
                },
                "protocol": {
                    "query_leakage_guard": (
                        "Linked issue text is stored only in the evaluation manifest, not in "
                        "indexed patch cards."
                    ),
                    "candidate_pool": "Historical patch cards from the selected repository only.",
                    "retrieval_limit": limit,
                },
                "note": _evaluation_note(verified_cases, smoke_test_cases) if total else (
                    "No historical fix descriptions were found. Sync this repository before "
                    "evaluating."
                ),
            }
            return self.index.finish_patchwork_evaluation_run(
                str(run["id"]),
                status="completed",
                benchmark_cases=total,
                metrics=metrics,
            )
        except (OSError, RuntimeError, ValueError) as error:
            return self.index.finish_patchwork_evaluation_run(
                str(run["id"]),
                status="failed",
                benchmark_cases=0,
                error=str(error),
            )

    def investigations(self, repository_id: str, *, user_id: str) -> list[dict[str, object]]:
        if not self.index.get_patchwork_repository(repository_id, user_id=user_id):
            raise KeyError(f"Repository not found: {repository_id}")
        return self.index.list_patchwork_investigations(
            repository_id=repository_id, user_id=user_id
        )

    def delete_repository(self, repository_id: str, *, user_id: str) -> bool:
        return self.index.delete_patchwork_repository(repository_id, user_id=user_id)

    def _github_client(self, user_id: str) -> GitHubClient:
        account_token = self.github_oauth.access_token_for_user(user_id)
        return GitHubClient(
            self.rag.settings.github_api_url,
            token=account_token or self.rag.settings.github_token,
            timeout=self.rag.settings.request_timeout_seconds,
            graphql_url=self.rag.settings.github_graphql_url,
        )


def _repository_documents(document: SourceDocument, repository_id: str) -> SourceDocument:
    return SourceDocument(
        source=f"patchwork://{repository_id}/{document.source}",
        text=document.text,
        content_type=document.content_type,
        metadata={
            **document.metadata,
            "patchwork_repository_id": repository_id,
            "source_url": document.source,
        },
    )


def _evaluation_cases(
    documents: list[dict[str, object]],
) -> list[tuple[dict[str, object], str, str]]:
    """Build transparent evaluation cases from the strongest available GitHub evidence."""
    cases: list[tuple[dict[str, object], str, str]] = []
    for document in documents:
        metadata = document["metadata"]
        if not isinstance(metadata, dict):
            continue
        overview = document.get("overview_text")
        if not isinstance(overview, str):
            continue
        description = _patch_excerpt(overview)
        if len(description) >= 30:
            cases.append((document, description, "pr_description_smoke_test"))
    return cases


def _legacy_evaluation_cases(documents: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "expected_source": document["source"],
            "expected_pull_number": document["metadata"].get("pull_number"),
            "query": query,
            "provenance": source_kind,
            "issue_number": document["metadata"].get("linked_issue_number"),
            "issue_url": document["metadata"].get("linked_issue_url"),
        }
        for document, query, source_kind in _evaluation_cases(documents)
        if isinstance(document.get("metadata"), dict)
    ]


def _evaluation_manifest(cards: list[PatchCard], repository_id: str) -> list[dict[str, object]]:
    """Persist only confirmed GitHub issue-to-fix pairs from this exact sync."""
    cases: list[dict[str, object]] = []
    for card in cards:
        metadata = card.document.metadata
        query = metadata.get("benchmark_query")
        provenance = metadata.get("benchmark_provenance")
        if not isinstance(query, str) or not query.strip() or not isinstance(provenance, str):
            continue
        cases.append(
            {
                "expected_source": f"patchwork://{repository_id}/{card.document.source}",
                "expected_pull_number": card.pull_number,
                "query": query.strip(),
                "provenance": provenance,
                "issue_number": card.linked_issue_number,
                "issue_url": metadata.get("linked_issue_url"),
            }
        )
    return cases


def _evaluation_note(verified_cases: int, smoke_test_cases: int) -> str:
    parts: list[str] = []
    if verified_cases:
        parts.append(f"{verified_cases} verified GitHub issue-to-fix pair(s)")
    if smoke_test_cases:
        parts.append(f"{smoke_test_cases} PR-description smoke check(s)")
    details = " and ".join(parts)
    if verified_cases and smoke_test_cases:
        return (
            f"Measured on {details}. The verified pairs are the reportable metric; smoke checks "
            "only confirm that indexing and repository scoping are functioning."
        )
    if verified_cases:
        return (
            f"Measured on {details}, using an issue report to retrieve its fixing pull request. "
            "The issue text is excluded from the indexed target patch to prevent query leakage."
        )
    return (
        f"Measured on {details}. This is a product-health check, not a trustworthy issue-to-fix "
        "benchmark, because this repository did not expose a linked GitHub issue."
    )


def _deduplicated_patch_matches(citations: tuple[Any, ...]) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    seen_sources: set[str] = set()
    for hit in citations:
        if hit.source in seen_sources:
            continue
        seen_sources.add(hit.source)
        matches.append(
            {
                "pull_number": hit.metadata.get("pull_number"),
                "pull_url": hit.metadata.get("pull_url", hit.source),
                "title": hit.metadata.get("filename", "Historical patch"),
                "score": round(hit.score, 4),
                "merge_commit_sha": hit.metadata.get("merge_commit_sha"),
                "linked_issue_number": hit.metadata.get("linked_issue_number"),
                "linked_issue_url": hit.metadata.get("linked_issue_url"),
                "changed_files": hit.metadata.get("changed_files", []),
                "test_files": hit.metadata.get("test_files", []),
                "excerpt": _patch_excerpt(hit.text),
            }
        )
    return matches


def _string_or_none(value: object) -> str | None:
    return str(value) if value is not None else None


def _investigation_summary(matches: list[dict[str, object]]) -> str:
    if not matches:
        return (
            "No close historical fix was found in this repository yet. Try adding an error "
            "message, affected file, or a shorter description of the behavior."
        )
    leading = matches[0]
    title = re.sub(r"^PR #\d+\s*·\s*", "", str(leading["title"]))
    pull_number = leading.get("pull_number")
    return (
        f"Start with PR #{pull_number}, \u201c{title}.\u201d It is the closest historical "
        "fix in this repository; review its implementation and regression coverage before "
        "applying the same approach."
    )


def _patch_excerpt(text: str) -> str:
    """Return a concise, plain-language PR explanation without template clutter."""
    if "## Problem and resolution" not in text:
        return (
            "Patchwork found related code changes; open this pull request to review the "
            "implementation."
        )
    text = text.split("## Problem and resolution", maxsplit=1)[1]
    text = text.split("## Changed tests", maxsplit=1)[0]
    text = text.split("## Relevant diff hunks", maxsplit=1)[0]
    cleaned = _clean_pull_request_prose(text)
    if not cleaned:
        return "This historical pull request is a potentially relevant implementation reference."
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return " ".join(sentences[:2])[:320].rstrip()


def _clean_pull_request_prose(text: str) -> str:
    """Flatten Markdown while removing common PR-template fields and checklists."""
    text = re.sub(r"<!--[\s\S]*?-->", " ", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"```[\s\S]*?```", " ", text)

    useful_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("```", "<")):
            continue
        normalized = re.sub(r"^#{1,6}\s*", "", line).strip().lower().rstrip(":")
        if normalized in {
            "checklist",
            "testing",
            "test plan",
            "screenshots",
            "type of change",
            "related issues",
            "related issue",
            "additional context",
            "reviewers",
        }:
            continue
        if re.match(r"^[-*]\s*\[[ xX]\]", line):
            continue
        if re.match(r"^(close[sd]?|fixe[sd]?|resolve[sd]?)\s+#?\d+\b", line, re.I):
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = re.sub(r"[*_~]", "", line)
        if line:
            useful_lines.append(line)

    return re.sub(r"\s+", " ", " ".join(useful_lines)).strip(" -:")
