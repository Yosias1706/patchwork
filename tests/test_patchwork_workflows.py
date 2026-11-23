import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_document_search.github import PATCH_CARD_VERSION, GitHubRateLimited, GitHubRepository
from rag_document_search.models import SearchHit
from rag_document_search.patchwork import PatchworkService, _sync_state


@pytest.fixture
def service() -> PatchworkService:
    service = object.__new__(PatchworkService)
    service.index = Mock()
    service.rag = Mock()
    service.rag.settings.github_fetch_workers = 4
    return service


def test_discovery_stops_once_requested_merged_candidates_are_found(service) -> None:
    client = Mock()
    client.closed_pull_page.return_value = [
        {"number": number, "title": "Fix cleanup", "merged_at": "2026-01-01"}
        for number in range(1, 101)
    ]
    state = _sync_state(None)
    repository = GitHubRepository(
        "https://github.com/example/api", "example/api", "example", "api", "main", None, False
    )
    service._discover_repair_candidates(
        "run", repository, {"pull_requests_seen": 0, "requested_pull_limit": 3}, state, client
    )
    assert client.closed_pull_page.call_count == 1
    assert state["phase"] == "indexing"


def test_evaluation_does_not_turn_missing_descriptions_into_generic_queries(service) -> None:
    service.index.get_patchwork_repository.return_value = {"id": "repo"}
    service.index.create_patchwork_evaluation_run.return_value = {"id": "evaluation"}
    service.index.list_patchwork_evaluation_cases.return_value = []
    service.index.list_documents_for_repository.return_value = [
        {"source": "patchwork://repo/pr/1", "metadata": {}, "overview_text": "PR metadata only"}
    ]
    service.investigator = Mock()
    service.investigator.retrieve.return_value = SimpleNamespace(citations=[])
    service.evaluate("repo", user_id="user")
    service.investigator.retrieve.assert_not_called()
    assert service.index.finish_patchwork_evaluation_run.call_args.kwargs["benchmark_cases"] == 0


def test_evaluation_reports_actual_rank_and_misses(service) -> None:
    service.index.get_patchwork_repository.return_value = {"id": "repo"}
    service.index.create_patchwork_evaluation_run.return_value = {"id": "evaluation"}
    service.index.list_documents_for_repository.return_value = []
    service.index.list_patchwork_evaluation_cases.return_value = [
        {"expected_source": "source-a", "expected_pull_number": 1,
         "query": "A cancelled request leaks resources", "provenance": "user_labeled_issue"},
        {"expected_source": "source-b", "expected_pull_number": 2,
         "query": "An upload loses its position", "provenance": "user_labeled_issue"},
    ]
    hit = SearchHit("chunk", "doc", "source-a", "cleanup", 0.8, 0, 0, 7,
                    {"pull_number": 1, "pull_url": "https://github.com/example/api/pull/1"})
    service.investigator = Mock()
    service.investigator.retrieve.side_effect = [
        SimpleNamespace(citations=[hit]), SimpleNamespace(citations=[])
    ]
    service.evaluate("repo", user_id="user")
    metrics = service.index.finish_patchwork_evaluation_run.call_args.kwargs["metrics"]
    assert metrics["recall_at_1"] == 0.5
    assert metrics["mrr"] == 0.5
    assert metrics["case_results"][0]["rank"] == 1
    assert metrics["case_results"][1]["failure_type"] == "retrieval_miss"
    assert metrics["quality"] == "user_labeled_issue_to_fix"
    assert metrics["case_results"][0]["retrieved"][0]["pull_number"] == 1


def test_user_case_must_target_a_pr_in_the_owned_repository(service) -> None:
    service.index.list_documents_for_repository.return_value = []
    with pytest.raises(ValueError, match="indexed"):
        service.add_evaluation_case(
            "repo", user_id="user", issue="Cancelled requests leak", expected_pull_number=99
        )
    service.index.save_patchwork_evaluation_case.assert_not_called()


def test_sync_skips_network_reads_for_unchanged_prs(service) -> None:
    service.index.list_documents_for_repository.return_value = [{
        "source": "source", "metadata": {"pull_number": 42, "pull_updated_at": "yesterday",
                                            "patch_card_version": PATCH_CARD_VERSION}
    }]
    state = _sync_state({"phase": "indexing", "candidate_pull_numbers": [42],
                         "candidate_summaries": {"42": {"updated_at": "yesterday"}}})
    client = Mock()
    repository = GitHubRepository("url", "example/api", "example", "api", "main", None, False)
    service._index_repair_candidates(
        "run", "repo", repository, {"patch_cards_indexed": 0, "chunks_indexed": 0}, state, client
    )
    client.patch_card.assert_not_called()
    assert state["next_candidate_index"] == 1
    assert state["candidate_summaries"] == {}
    assert service.index.update_patchwork_sync_run.call_args.kwargs["status"] == "completed"


def test_sync_fetches_prs_concurrently_and_checkpoints_each_one(service) -> None:
    service.index.list_documents_for_repository.return_value = []
    barrier = threading.Barrier(4, timeout=2)
    client = Mock()
    client.patch_card.side_effect = lambda *_args, **_kwargs: (barrier.wait(), None)[1]
    state = _sync_state({"phase": "indexing", "candidate_pull_numbers": [1, 2, 3, 4]})
    repository = GitHubRepository("url", "example/api", "example", "api", "main", None, False)
    service._index_repair_candidates(
        "run", "repo", repository, {"patch_cards_indexed": 0, "chunks_indexed": 0}, state, client
    )
    assert client.patch_card.call_count == 4
    assert state["next_candidate_index"] == 4
    assert service.index.update_patchwork_sync_run.call_count == 5


def test_parallel_rate_limit_preserves_first_unfinished_checkpoint(service) -> None:
    service.index.list_documents_for_repository.return_value = []
    client = Mock()

    def fetch(_repo, number, **_kwargs):
        if number == 2:
            raise GitHubRateLimited(90)
        return None

    client.patch_card.side_effect = fetch
    state = _sync_state({"phase": "indexing", "candidate_pull_numbers": [1, 2, 3, 4]})
    repository = GitHubRepository("url", "example/api", "example", "api", "main", None, False)
    with pytest.raises(GitHubRateLimited):
        service._index_repair_candidates(
            "run", "repo", repository, {"patch_cards_indexed": 0, "chunks_indexed": 0},
            state, client,
        )
    assert state["next_candidate_index"] == 1
    assert service.index.update_patchwork_sync_run.call_count == 1
