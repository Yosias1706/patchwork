from rag_document_search.models import SearchHit
from rag_document_search.solution_agent import SolutionAgent, SolutionGenerator


def test_evidence_guided_solution_references_files_and_tests() -> None:
    match = {
        "pull_number": 42,
        "pull_url": "https://github.com/example/api/pull/42",
        "title": "Fix disconnect cleanup race",
        "excerpt": "Cleanup continued after disconnect.",
        "changed_files": ["src/stream.py"],
        "test_files": ["tests/test_stream.py"],
    }
    hit = SearchHit(
        chunk_id="chunk",
        document_id="doc",
        source=str(match["pull_url"]),
        text="Historical fix for disconnect cleanup.",
        score=0.82,
        ordinal=0,
        start_offset=0,
        end_offset=40,
        metadata={},
    )

    solution = SolutionAgent(SolutionGenerator()).recommend(
        "Disconnects leave work pending", [match], [hit]
    )

    assert solution.mode == "evidence_guided"
    assert "PR #42" in solution.assessment
    assert "src/stream.py" in " ".join(solution.implementation_steps)
    assert "tests/test_stream.py" in " ".join(solution.verification)


def test_scoped_citation_produces_a_specific_recommendation() -> None:
    match = {
        "pull_number": 42,
        "pull_url": "https://github.com/example/api/pull/42",
        "title": "Fix disconnect cleanup race",
        "excerpt": "Cleanup continued after disconnect. Await cleanup before closing the stream.",
        "changed_files": ["src/stream.py"],
        "test_files": ["tests/test_stream.py"],
    }
    hit = SearchHit(
        chunk_id="chunk",
        document_id="doc",
        source="patchwork://repo-a/https://github.com/example/api/pull/42",
        text=("## Relevant diff hunks\n### src/stream.py\n"
              "```diff\n- cleanup()\n+ await cleanup()\n```"),
        score=0.82,
        ordinal=1,
        start_offset=900,
        end_offset=1000,
        metadata={"pull_url": match["pull_url"]},
    )
    solution = SolutionAgent(SolutionGenerator()).recommend(
        "Disconnects leave work pending", [match], [hit]
    )
    assert solution.source_pull_numbers == (42,)
    assert "Await cleanup" in solution.assessment
    assert "src/stream.py" in solution.assessment
    assert "await cleanup()" in (solution.code_guidance or "")
