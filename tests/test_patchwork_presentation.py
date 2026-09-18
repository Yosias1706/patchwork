from rag_document_search.github import PatchCard
from rag_document_search.models import SourceDocument
from rag_document_search.patchwork import (
    _evaluation_cases,
    _evaluation_manifest,
    _investigation_summary,
    _patch_excerpt,
)


def test_patch_excerpt_never_returns_a_raw_diff_fragment() -> None:
    assert _patch_excerpt("@@ -1,2 +1,2 @@\n- old\n+ new") == (
        "Patchwork found related code changes; open this pull request to review the implementation."
    )


def test_investigation_summary_uses_a_readable_pull_request_title() -> None:
    summary = _investigation_summary(
        [
            {
                "pull_number": 42,
                "title": "PR #42 · Fix disconnect cleanup race",
                "excerpt": "A disconnected client left cleanup work pending.",
                "changed_files": ["src/stream.py"],
                "test_files": ["tests/test_stream.py"],
            }
        ]
    )

    assert "PR #42): Fix disconnect cleanup race" in summary
    assert "Regression coverage was added" in summary


def test_evaluation_uses_fix_description_when_a_linked_issue_is_unavailable() -> None:
    cases = _evaluation_cases(
        [
            {
                "source": "patchwork://repo/pull/42",
                "metadata": {
                    "pull_number": 42,
                    "benchmark_query": "A legacy linked issue must not become an evaluation query.",
                },
                "overview_text": (
                    "# Historical bug fix: Fix cleanup\n\n## Problem and resolution\n"
                    "A disconnected client left cleanup work pending. The fix closes the task.\n\n"
                    "## Changed tests\n- tests/test_cleanup.py"
                ),
            }
        ]
    )

    assert cases[0][1].startswith("A disconnected client")
    assert cases[0][2] == "pr_description_smoke_test"


def test_evaluation_manifest_keeps_issue_query_outside_the_patch_card() -> None:
    card = PatchCard(
        document=SourceDocument(
            source="https://github.com/example/api/pull/42",
            text="# Historical bug fix: Close cleanup race",
            content_type="text/x-patchwork-card",
            metadata={
                "benchmark_query": "Disconnecting a client leaves the task pending.",
                "benchmark_provenance": "github_closing_reference",
                "linked_issue_url": "https://github.com/example/api/issues/17",
            },
        ),
        pull_number=42,
        linked_issue_number=17,
        linked_issue_provenance="github_closing_reference",
    )

    manifest = _evaluation_manifest([card], "repository-id")

    assert manifest == [
        {
            "expected_source": "patchwork://repository-id/https://github.com/example/api/pull/42",
            "expected_pull_number": 42,
            "query": "Disconnecting a client leaves the task pending.",
            "provenance": "github_closing_reference",
            "issue_number": 17,
            "issue_url": "https://github.com/example/api/issues/17",
        }
    ]
    assert manifest[0]["query"] not in card.document.text
