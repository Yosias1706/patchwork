from rag_document_search.patchwork import _evaluation_manifest_from_documents, _sync_state


def test_sync_state_recovers_an_interrupted_discovery_run() -> None:
    state = _sync_state(
        {
            "phase": "discovering",
            "next_page": 7,
            "candidate_pull_numbers": [12, "34", 12, "not-a-number"],
            "next_candidate_index": 3,
        }
    )

    assert state == {
        "phase": "discovering",
        "next_page": 7,
        "candidate_pull_numbers": [12, 34],
        "next_candidate_index": 3,
    }


def test_batch_evaluation_manifest_uses_only_confirmed_issue_pairs() -> None:
    manifest = _evaluation_manifest_from_documents(
        [
            {
                "source": "patchwork://repository-id/https://github.com/example/api/pull/42",
                "metadata": {
                    "pull_number": 42,
                    "benchmark_query": "Disconnecting a client leaves a task pending.",
                    "benchmark_provenance": "github_closing_reference",
                    "linked_issue_number": 17,
                    "linked_issue_url": "https://github.com/example/api/issues/17",
                },
            },
            {"source": "patchwork://repository-id/invalid", "metadata": {}},
        ]
    )

    assert manifest == [
        {
            "expected_source": "patchwork://repository-id/https://github.com/example/api/pull/42",
            "expected_pull_number": 42,
            "query": "Disconnecting a client leaves a task pending.",
            "provenance": "github_closing_reference",
            "issue_number": 17,
            "issue_url": "https://github.com/example/api/issues/17",
        }
    ]
