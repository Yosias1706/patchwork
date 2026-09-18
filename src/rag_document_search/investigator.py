"""Bounded, evidence-first retrieval agent for Patchwork investigations.

The agent intentionally uses deterministic planning and explicit tools.  It can
therefore explain which searches it ran, remain useful without an LLM, and
never performs repository writes.  A configured answer model is used only
after the agent has assembled its citation set.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass

from .models import SearchHit
from .service import RAGService

_ERROR_NAME = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Failure|Panic)\b")
_PATH = re.compile(
    r"(?<![\w.-])(?:[\w.-]+/)+[\w.-]+\.(?:py|pyi|js|jsx|ts|tsx|go|rs|java|rb|php|cs|cpp|c|h)"
)
_SYMBOL = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\s*\(")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_STOP_WORDS = frozenset(
    {
        "after",
        "before",
        "because",
        "client",
        "could",
        "during",
        "error",
        "fails",
        "failure",
        "from",
        "have",
        "issue",
        "when",
        "with",
        "would",
    }
)


@dataclass(frozen=True, slots=True)
class SearchDirective:
    """A single bounded retrieval-tool invocation chosen by the planner."""

    purpose: str
    query: str
    weight: float


@dataclass(frozen=True, slots=True)
class InvestigationPlan:
    """Inspectable plan generated from the incident text, with no model call."""

    directives: tuple[SearchDirective, ...]


@dataclass(frozen=True, slots=True)
class ToolTrace:
    purpose: str
    query: str
    candidate_count: int


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    plan: InvestigationPlan
    citations: tuple[SearchHit, ...]
    trace: tuple[ToolTrace, ...]
    confidence: str
    confidence_reason: str

    def as_api_details(self) -> dict[str, object]:
        return {
            "plan": [asdict(directive) for directive in self.plan.directives],
            "tool_trace": [asdict(event) for event in self.trace],
            "confidence_reason": self.confidence_reason,
        }


@dataclass(frozen=True, slots=True)
class InvestigationOutcome:
    answer: str
    retrieval: RetrievalResult


class InvestigationAgent:
    """Plans multi-query retrieval, fuses evidence, and generates a grounded answer.

    The planner has a strict four-query budget.  Each search is scoped to the
    connected repository, and the rank-fusion stage returns at most one best
    citation per historical pull request.  This makes results stable, prevents
    a noisy trace from dominating retrieval, and keeps the final model context
    small and fully attributable.
    """

    def __init__(self, rag: RAGService) -> None:
        self.rag = rag

    def retrieve(
        self, repository_id: str, issue: str, *, limit: int = 5
    ) -> RetrievalResult:
        if not issue.strip():
            raise ValueError("Issue description must not be empty")
        if not 1 <= limit <= 10:
            raise ValueError("Investigation limit must be between 1 and 10")
        plan = self.plan(issue)
        per_tool_limit = min(max(limit * 3, 8), 30)
        grouped: dict[str, list[tuple[SearchHit, int, float, str]]] = defaultdict(list)
        trace: list[ToolTrace] = []
        for directive in plan.directives:
            candidates = self.rag.search(
                directive.query, limit=per_tool_limit, repository_id=repository_id
            )
            trace.append(
                ToolTrace(
                    purpose=directive.purpose,
                    query=directive.query,
                    candidate_count=len(candidates),
                )
            )
            for rank, hit in enumerate(candidates, start=1):
                grouped[hit.source].append((hit, rank, directive.weight, directive.purpose))

        ranked = _fuse_by_source(grouped, limit)
        confidence, reason = _confidence(ranked, grouped)
        return RetrievalResult(
            plan=plan,
            citations=tuple(ranked),
            trace=tuple(trace),
            confidence=confidence,
            confidence_reason=reason,
        )

    def run(self, repository_id: str, issue: str, *, limit: int = 5) -> InvestigationOutcome:
        retrieval = self.retrieve(repository_id, issue, limit=limit)
        return InvestigationOutcome(
            answer=self.rag.generator.answer(issue, retrieval.citations), retrieval=retrieval
        )

    def plan(self, issue: str) -> InvestigationPlan:
        normalized = " ".join(issue.strip().split())[:1200]
        directives = [SearchDirective("incident_report", normalized, 1.0)]
        error_names = _unique(_ERROR_NAME.findall(normalized))[:2]
        paths = _unique(_PATH.findall(normalized))[:2]
        symbols = _unique(match.rstrip("(").strip() for match in _SYMBOL.findall(normalized))[:2]
        symptoms = _symptom_terms(normalized)

        if error_names:
            terms = " ".join([*error_names, *symptoms[:4]])
            directives.append(SearchDirective("error_signature", terms, 1.35))
        if paths:
            directives.append(SearchDirective("code_location", " ".join(paths), 1.2))
        if symbols:
            directives.append(SearchDirective("code_symbol", " ".join(symbols), 1.1))
        if len(directives) == 1 and symptoms:
            directives.append(SearchDirective("symptom_terms", " ".join(symptoms[:7]), 1.15))
        return InvestigationPlan(directives=tuple(_unique_directives(directives)[:4]))


def _fuse_by_source(
    grouped: dict[str, list[tuple[SearchHit, int, float, str]]], limit: int
) -> list[SearchHit]:
    scored: list[tuple[float, SearchHit]] = []
    for source, entries in grouped.items():
        fused_score = sum(weight / (60 + rank) for _, rank, weight, _ in entries)
        strongest = max(entries, key=lambda entry: entry[0].score)[0]
        overview_candidates = [
            hit
            for hit, _, _, _ in entries
            if hit.ordinal == 0 or "# Historical bug fix:" in hit.text
        ]
        display = max(overview_candidates or [strongest], key=lambda hit: hit.score)
        metadata = {
            **strongest.metadata,
            "agent_fusion_score": round(fused_score, 5),
            "agent_query_support": len({purpose for _, _, _, purpose in entries}),
        }
        scored.append(
            (
                fused_score,
                SearchHit(
                    chunk_id=display.chunk_id,
                    document_id=display.document_id,
                    source=source,
                    text=display.text,
                    score=strongest.score,
                    ordinal=display.ordinal,
                    start_offset=display.start_offset,
                    end_offset=display.end_offset,
                    metadata=metadata,
                ),
            )
        )
    return [hit for _, hit in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]


def _confidence(
    ranked: list[SearchHit], grouped: dict[str, list[tuple[SearchHit, int, float, str]]]
) -> tuple[str, str]:
    if not ranked:
        return "no_similar_fix_found", "No repository-scoped search produced evidence."
    leading = ranked[0]
    supporting_queries = len({purpose for _, _, _, purpose in grouped[leading.source]})
    has_tests = bool(leading.metadata.get("test_files"))
    if supporting_queries >= 2 and has_tests:
        return "high", "The leading patch was supported by multiple planned searches and has tests."
    if leading.score >= 0.18:
        return (
            "moderate",
            "The leading patch has meaningful lexical-semantic overlap with the incident.",
        )
    return "low", "Evidence was found, but it came from a weak or narrow signal."


def _symptom_terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in _unique(_WORD.findall(text))
        if token.lower() not in _STOP_WORDS and not token.isdigit()
    ]


def _unique(values: list[str] | tuple[str, ...] | object) -> list[str]:
    result: list[str] = []
    for value in values:  # type: ignore[union-attr]
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _unique_directives(directives: list[SearchDirective]) -> list[SearchDirective]:
    unique: list[SearchDirective] = []
    seen: set[str] = set()
    for directive in directives:
        key = directive.query.lower()
        if key not in seen:
            unique.append(directive)
            seen.add(key)
    return unique
