"""Patchwork orchestration: repository syncs, investigations, and evaluations."""

from __future__ import annotations

import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from typing import Any

from .github import (
    PATCH_CARD_VERSION,
    GitHubClient,
    GitHubError,
    GitHubRateLimited,
    GitHubRepository,
    PatchCard,
    is_repair_candidate,
)
from .github_oauth import GitHubOAuthService
from .investigator import InvestigationAgent
from .models import SourceDocument
from .repository import PostgresIndex
from .service import RAGService
from .solution_agent import SolutionAgent, build_solution_generator

_DISCOVERY_PAGES_PER_BATCH = 10
_DETAIL_PULLS_PER_BATCH = 10
_MAX_HISTORY_PAGES = 100


class PatchworkService:
    def __init__(self, rag: RAGService) -> None:
        if not isinstance(rag.index, PostgresIndex):
            raise TypeError("Patchwork requires the PostgreSQL index")
        self.rag = rag
        self.index = rag.index
        self.github_oauth = GitHubOAuthService(self.index, rag.settings)
        self.investigator = InvestigationAgent(rag)
        self.solution_agent = SolutionAgent(build_solution_generator(rag.settings))

    def connect_repository(
        self,
        github_url: str,
        *,
        user_id: str,
        pull_limit: int | None = None,
        ingestion_mode: str = "detailed",
    ) -> tuple[dict[str, object], dict[str, object]]:
        if ingestion_mode != "detailed":
            raise ValueError("Bulk ingestion is not available yet. Use detailed ingestion.")
        if pull_limit is not None and not 1 <= pull_limit <= 10_000:
            raise ValueError("Patchwork supports between 1 and 10000 historical fixes per sync")
        repository = self._github_client(user_id).repository(github_url)
        stored_repository = self.index.upsert_patchwork_repository(
            repository.as_storage_details(), user_id=user_id
        )
        requested_limit = (
            self.rag.settings.patchwork_default_pull_limit if pull_limit is None else pull_limit
        )
        run = self.index.create_patchwork_sync_run(str(stored_repository["id"]), requested_limit)
        return stored_repository, run

    def sync_repository(self, repository_id: str, run_id: str) -> None:
        repository = self.index.get_patchwork_repository(repository_id)
        if not repository:
            raise KeyError(f"Repository not found: {repository_id}")
        run = self.index.get_patchwork_sync_run(run_id)
        if not run:
            raise KeyError(f"Sync run not found: {run_id}")
        state = _sync_state(run.get("state"))
        self.index.update_patchwork_sync_run(run_id, status="running", state=state)
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
            client = self._github_client(owner_user_id)
            if state["phase"] == "discovering":
                self._discover_repair_candidates(run_id, github_repository, run, state, client)
            else:
                self._index_repair_candidates(
                    run_id, repository_id, github_repository, run, state, client
                )
        except GitHubRateLimited as error:
            self.index.update_patchwork_sync_run(
                run_id,
                status="queued",
                state=state,
                retry_after_seconds=error.retry_after_seconds,
                error=str(error),
            )
        except (GitHubError, OSError, RuntimeError, ValueError) as error:
            self.index.update_patchwork_sync_run(run_id, status="failed", error=str(error))

    def _discover_repair_candidates(
        self,
        run_id: str,
        repository: GitHubRepository,
        run: dict[str, object],
        state: dict[str, object],
        client: GitHubClient,
    ) -> None:
        """Discover only as much history as needed and make each page usable early."""
        candidates = state["candidate_pull_numbers"]
        summaries = state.setdefault("candidate_summaries", {})
        known = set(candidates)
        seen = int(run["pull_requests_seen"])
        for _ in range(_DISCOVERY_PAGES_PER_BATCH):
            page = int(state["next_page"])
            pulls = client.closed_pull_page(repository, page=page)
            seen += len(pulls)
            for pull in pulls:
                number = int(pull["number"])
                if (
                    number not in known
                    and len(candidates) < int(run["requested_pull_limit"])
                    and pull.get("merged_at")
                    and is_repair_candidate(pull)
                ):
                    candidates.append(number)
                    known.add(number)
                    summaries[str(number)] = {
                        key: pull.get(key)
                        for key in (
                            "number",
                            "title",
                            "body",
                            "html_url",
                            "merged_at",
                            "merge_commit_sha",
                            "labels",
                            "updated_at",
                        )
                    }
            state["next_page"] = page + 1
            state["history_complete"] = (
                len(candidates) >= int(run["requested_pull_limit"])
                or len(pulls) < 100
                or page == _MAX_HISTORY_PAGES
            )
            if state["history_complete"] or len(candidates) > int(state["next_candidate_index"]):
                state["phase"] = "indexing"
            self.index.update_patchwork_sync_run(
                run_id,
                status="running",
                state=state,
                pull_requests_seen=seen,
            )
            if state["phase"] == "indexing":
                break

        self.index.update_patchwork_sync_run(
            run_id,
            status="queued",
            state=state,
            pull_requests_seen=seen,
            error=None,
        )

    def _index_repair_candidates(
        self,
        run_id: str,
        repository_id: str,
        repository: GitHubRepository,
        run: dict[str, object],
        state: dict[str, object],
        client: GitHubClient,
    ) -> None:
        """Index a bounded set of candidate fixes and checkpoint each completed PR."""
        candidates = state["candidate_pull_numbers"]
        offset = int(state["next_candidate_index"])
        documents = int(run["patch_cards_indexed"])
        chunks = int(run["chunks_indexed"])
        existing = {
            document["metadata"].get("pull_number"): document["metadata"]
            for document in self.index.list_documents_for_repository(repository_id)
            if isinstance(document.get("metadata"), dict)
        }
        stop = min(offset + _DETAIL_PULLS_PER_BATCH, len(candidates))
        with ThreadPoolExecutor(max_workers=self.rag.settings.github_fetch_workers) as executor:
            pending: dict[int, Future[PatchCard | None]] = {}
            unchanged: set[int] = set()
            for candidate_index in range(offset, stop):
                number = candidates[candidate_index]
                summary = state.get("candidate_summaries", {}).get(str(number))
                previous = existing.get(number, {})
                if (
                    summary
                    and summary.get("updated_at")
                    and previous.get("pull_updated_at") == summary["updated_at"]
                    and previous.get("patch_card_version") == PATCH_CARD_VERSION
                ):
                    unchanged.add(candidate_index)
                else:
                    pending[candidate_index] = executor.submit(
                        client.patch_card, repository, number, summary=summary
                    )
            try:
                for candidate_index in range(offset, stop):
                    number = candidates[candidate_index]
                    card = pending[candidate_index].result() if candidate_index in pending else None
                    if candidate_index in unchanged:
                        documents += 1
                    elif card:
                        report = self.rag.ingest_documents(
                            [_repository_documents(card.document, repository_id)]
                        )
                        if report.failures:
                            raise RuntimeError(
                                "Could not index PR #" + str(number) + ": " + report.failures[0]
                            )
                        documents += report.documents_indexed + report.documents_unchanged
                        chunks += report.chunks_indexed
                    # Commit in discovery order. A failure can only resume at the
                    # first PR that has not yet been durably indexed.
                    state.get("candidate_summaries", {}).pop(str(number), None)
                    state["next_candidate_index"] = candidate_index + 1
                    self.index.update_patchwork_sync_run(
                        run_id,
                        status="running",
                        state=state,
                        patch_cards_indexed=documents,
                        chunks_indexed=chunks,
                    )
            except BaseException:
                for future in pending.values():
                    future.cancel()
                raise

        if int(state["next_candidate_index"]) < len(candidates):
            self.index.update_patchwork_sync_run(
                run_id,
                status="queued",
                state=state,
                patch_cards_indexed=documents,
                chunks_indexed=chunks,
                error=None,
            )
            return

        documents_for_repository = self.index.list_documents_for_repository(repository_id)
        self.index.replace_patchwork_evaluation_cases(
            repository_id, _evaluation_manifest_from_documents(documents_for_repository)
        )
        if not state.get("history_complete", True):
            state["phase"] = "discovering"
            self.index.update_patchwork_sync_run(
                run_id,
                status="queued",
                state=state,
                patch_cards_indexed=documents,
                chunks_indexed=chunks,
                error=None,
            )
            return
        self.index.update_patchwork_sync_run(
            run_id,
            status="completed",
            state=state,
            patch_cards_indexed=documents,
            chunks_indexed=chunks,
            error=None,
        )

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
        texts = self.index.patchwork_evidence(
            repository_id, [hit.source for hit in retrieval.citations]
        )
        citations = tuple(
            replace(hit, text=texts.get(hit.source, hit.text)) for hit in retrieval.citations
        )
        matches = _deduplicated_patch_matches(citations)
        summary = _investigation_summary(matches)
        recommended_solution = self.solution_agent.recommend(issue, matches, citations)
        recommended_solution_details = recommended_solution.as_api_details()
        if matches:
            matches[0]["recommended_solution"] = recommended_solution_details
        result = {
            "issue": issue,
            "summary": summary,
            "answer": summary,
            "matches": matches,
            "citations": [asdict(hit) for hit in retrieval.citations],
            "confidence": retrieval.confidence,
            "recommended_solution": recommended_solution_details,
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
        self, repository_id: str, *, user_id: str, limit: int = 5, max_cases: int = 50
    ) -> dict[str, object]:
        if not 1 <= max_cases <= 200 or not 1 <= limit <= 10:
            raise ValueError("Evaluation requires 1–200 cases and a retrieval limit of 1–10")
        if not self.index.get_patchwork_repository(repository_id, user_id=user_id):
            raise KeyError(f"Repository not found: {repository_id}")
        run = self.index.create_patchwork_evaluation_run(repository_id)
        started = time.monotonic()
        try:
            documents = self.index.list_documents_for_repository(repository_id)
            benchmarks = self.index.list_patchwork_evaluation_cases(repository_id)
            if not benchmarks:
                benchmarks = _evaluation_manifest_from_documents(documents)
            if not benchmarks:
                benchmarks = _legacy_evaluation_cases(documents)
            available_cases = len(benchmarks)
            # A reproducible, evenly spread sample keeps the interactive lab bounded.
            if len(benchmarks) > max_cases:
                benchmarks = [
                    benchmarks[index * len(benchmarks) // max_cases] for index in range(max_cases)
                ]
            reciprocal_ranks: list[float] = []
            retrieved_at_one = 0
            retrieved_at_three = 0
            failures: list[dict[str, object]] = []
            results: list[dict[str, object]] = []
            for benchmark in benchmarks:
                query = str(benchmark["query"])
                source_kind = str(benchmark["provenance"])
                hits = self.investigator.retrieve(
                    repository_id, query, limit=max(3, limit)
                ).citations
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
                case_result = {
                    "expected_pull_number": benchmark.get("expected_pull_number"),
                    "expected_pull_url": _source_url(str(benchmark["expected_source"])),
                    "issue_url": benchmark.get("issue_url"),
                    "query": query[:1200],
                    "rank": rank,
                    "failure_type": "retrieval_miss"
                    if rank is None
                    else ("wrong_top_result" if rank > 1 else None),
                    "source_kind": source_kind,
                    "retrieved": [
                        {
                            "pull_number": hit.metadata.get("pull_number"),
                            "pull_url": hit.metadata.get("pull_url", hit.source),
                            "title": hit.metadata.get("pull_title", hit.metadata.get("filename")),
                            "score": hit.metadata.get("agent_relevance_score", hit.score),
                        }
                        for hit in hits
                    ],
                }
                results.append(case_result)
                if rank != 1 and len(failures) < 10:
                    failures.append(case_result)
            total = len(benchmarks)
            verified_cases = sum(
                1
                for benchmark in benchmarks
                if benchmark["provenance"] in {"github_closing_reference", "pr_closing_keyword"}
            )
            user_cases = sum(
                benchmark["provenance"] == "user_labeled_issue" for benchmark in benchmarks
            )
            smoke_test_cases = total - verified_cases - user_cases
            metrics: dict[str, object] = {
                "recall_at_1": round(retrieved_at_one / total, 3) if total else None,
                "recall_at_3": round(retrieved_at_three / total, 3) if total else None,
                "mrr": round(sum(reciprocal_ranks) / total, 3) if total else None,
                "failure_samples": failures,
                "case_results": results,
                "available_cases": available_cases,
                "indexed_fixes": len(documents),
                "duration_seconds": round(time.monotonic() - started, 2),
                "quality": "verified_issue_to_fix"
                if verified_cases
                else ("user_labeled_issue_to_fix" if user_cases else "smoke_test"),
                "case_breakdown": {
                    "verified_issue_to_fix": verified_cases,
                    "user_labeled_issue": user_cases,
                    "pr_description_smoke_test": smoke_test_cases,
                },
                "protocol": {
                    "query_leakage_guard": (
                        "Linked issue text is stored only in the evaluation manifest, not in "
                        "indexed patch cards."
                    ),
                    "candidate_pool": "Historical patch cards from the selected repository only.",
                    "retrieval_limit": max(3, limit),
                    "case_limit": max_cases,
                },
                "note": (
                    f"Tested {user_cases} user-labeled issue-to-fix pair(s). "
                    "These labels were provided by you; inspect the results to confirm them. "
                    + (_evaluation_note(verified_cases, smoke_test_cases) if verified_cases else "")
                )
                if user_cases
                else _evaluation_note(verified_cases, smoke_test_cases)
                if total
                else (
                    "No historical fix descriptions were found. Sync this repository before "
                    "evaluating."
                ),
                "recommendations": _evaluation_recommendations(
                    total, len(documents), verified_cases + user_cases, retrieved_at_one, failures
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

    def add_evaluation_case(
        self, repository_id: str, *, user_id: str, issue: str, expected_pull_number: int
    ) -> dict[str, object]:
        if len(issue.strip()) < 8:
            raise ValueError("Describe the bug report using at least 8 non-whitespace characters")
        if not self.index.get_patchwork_repository(repository_id, user_id=user_id):
            raise KeyError(repository_id)
        for document in self.index.list_documents_for_repository(repository_id):
            metadata = document["metadata"]
            if metadata.get("pull_number") == expected_pull_number:
                case = {
                    "expected_source": document["source"],
                    "expected_pull_number": expected_pull_number,
                    "query": issue.strip(),
                    "provenance": "user_labeled_issue",
                    "issue_number": None,
                    "issue_url": None,
                }
                self.index.save_patchwork_evaluation_case(repository_id, case)
                return case
        raise ValueError("The expected PR must be indexed in this repository. Sync it first.")

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
            minimum_request_interval_seconds=self.rag.settings.github_request_interval_seconds,
            oauth_client_id=self.rag.settings.github_oauth_client_id,
            oauth_client_secret=self.rag.settings.github_oauth_client_secret,
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
        body = metadata.get("pull_body")
        if isinstance(body, str) and body.strip():
            description = _clean_pull_request_prose(body)
        elif "## Problem and resolution" in overview:
            description = _patch_excerpt(overview)
        else:
            description = str(metadata.get("pull_title") or "")
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


def _evaluation_manifest_from_documents(
    documents: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Rebuild the final evaluation manifest after a resumable multi-batch sync."""
    cases: list[dict[str, object]] = []
    for document in documents:
        metadata = document.get("metadata")
        if not isinstance(metadata, dict):
            continue
        query = metadata.get("benchmark_query")
        provenance = metadata.get("benchmark_provenance")
        pull_number = metadata.get("pull_number")
        source = document.get("source")
        if not (
            isinstance(query, str)
            and query.strip()
            and isinstance(provenance, str)
            and isinstance(pull_number, int)
            and isinstance(source, str)
        ):
            continue
        cases.append(
            {
                "expected_source": source,
                "expected_pull_number": pull_number,
                "query": query.strip(),
                "provenance": provenance,
                "issue_number": metadata.get("linked_issue_number"),
                "issue_url": metadata.get("linked_issue_url"),
            }
        )
    return cases


def _sync_state(raw_state: object) -> dict[str, object]:
    """Normalize state from newly-created and pre-batching sync rows."""
    raw = raw_state if isinstance(raw_state, dict) else {}
    phase = raw.get("phase") if raw.get("phase") in {"discovering", "indexing"} else "discovering"
    candidates = [
        int(number)
        for number in raw.get("candidate_pull_numbers", [])
        if isinstance(number, int) or (isinstance(number, str) and number.isdigit())
    ]
    normalized = {
        "phase": phase,
        "next_page": max(1, int(raw.get("next_page", 1))),
        "candidate_pull_numbers": list(dict.fromkeys(candidates)),
        "next_candidate_index": max(0, int(raw.get("next_candidate_index", 0))),
    }
    if isinstance(raw.get("candidate_summaries"), dict):
        normalized["candidate_summaries"] = raw["candidate_summaries"]
    if "history_complete" in raw:
        normalized["history_complete"] = bool(raw["history_complete"])
    return normalized


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


def _evaluation_recommendations(
    total: int,
    documents: int,
    issue_cases: int,
    top_hits: int,
    failures: list[dict[str, object]],
) -> list[str]:
    if not documents:
        return ["Sync a repository before evaluating; there are no indexed fixes to search."]
    advice: list[str] = []
    if not issue_cases:
        advice.append(
            "These PR-description checks only test indexing. Add a real bug report and its "
            "expected PR below to test whether recommendations are useful."
        )
    if not total:
        advice.append("Add a test issue and an indexed expected PR to create your first check.")
    elif top_hits < total:
        advice.append(
            "Review the missed examples below. Compare the returned PRs with the expected fix; "
            "include the incident's error, symbol or affected file when investigating."
        )
    else:
        advice.append(
            "Every sampled check ranked its expected PR first. Try additional reports in your "
            "own words before relying on this result."
        )
    if documents < 30:
        advice.append(
            f"Only {documents} fixes are indexed. A broader sync adds competing fixes and makes "
            "the check more representative."
        )
    return advice


def _source_url(source: str) -> str | None:
    if "https://" in source:
        return "https://" + source.split("https://", 1)[1]
    return None


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
                # This is a calibrated evidence score, not an assertion that a
                # historical PR is certainly the right fix.
                "score": round(float(hit.metadata.get("agent_relevance_score", hit.score)), 4),
                "match_reasons": hit.metadata.get("agent_match_reasons", []),
                "matched_terms": hit.metadata.get("agent_matched_terms", []),
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
    reasons = leading.get("match_reasons")
    reason = _summary_reason(reasons[0]) if isinstance(reasons, list) and reasons else ""
    return (
        f"Start with PR #{pull_number}, \u201c{title}.\u201d It is the closest historical "
        "fix in this repository; review its implementation and regression coverage before "
        f"applying the same approach.{reason}"
    )


def _summary_reason(value: object) -> str:
    """Turn the first retrieval reason into a natural-language summary clause."""
    reason = str(value).rstrip(".")
    if reason.startswith("Matches error: "):
        return f" It matches the reported error, {reason.removeprefix('Matches error: ')}."
    if reason.startswith("Touches "):
        return f" It changes {reason.removeprefix('Touches ')}."
    if reason.startswith("References "):
        return f" It references {reason.removeprefix('References ')}."
    return f" {reason}."


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
    skip_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            heading = re.sub(r"^#{1,6}\s*", "", line).strip().lower().rstrip(":")
            skip_section = heading in {"ai disclaimer", "ai disclosure", "ai usage"}
        if skip_section:
            continue
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
