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

from .embeddings import _normalize_token
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
        "the",
        "and",
        "are",
        "does",
        "not",
        "this",
        "that",
        "then",
        "there",
        "using",
        "instead",
        "into",
        "should",
        "report",
        "please",
        "find",
        "similar",
        "fix",
        "fixed",
        "fixes",
        "can",
        "cannot",
        "has",
        "our",
        "its",
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


@dataclass(frozen=True, slots=True)
class _IncidentSignals:
    """Concrete incident details used to validate semantic retrieval results."""

    errors: tuple[str, ...]
    paths: tuple[str, ...]
    symbols: tuple[str, ...]
    symptoms: tuple[str, ...]


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

    def retrieve(self, repository_id: str, issue: str, *, limit: int = 5) -> RetrievalResult:
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

        ranked = _fuse_by_source(grouped, limit, _incident_signals(issue))
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
    grouped: dict[str, list[tuple[SearchHit, int, float, str]]],
    limit: int,
    signals: _IncidentSignals,
) -> list[SearchHit]:
    scored: list[tuple[float, SearchHit]] = []
    corpora = {
        source: _signal_tokens(_hit_corpus(*(entry[0] for entry in entries)))
        for source, entries in grouped.items()
    }
    distinctive_terms = {
        term
        for term in signals.symptoms
        if sum(_signal_token(term) in tokens for tokens in corpora.values())
        <= max(1, len(corpora) // 2)
    }
    has_distinctive_evidence = any(
        _signal_token(term) in tokens for term in distinctive_terms for tokens in corpora.values()
    )
    for source, entries in grouped.items():
        fused_score = sum(weight / (60 + rank) for _, rank, weight, _ in entries)
        strongest = max(entries, key=lambda entry: entry[0].score)[0]
        overview_candidates = [
            hit
            for hit, _, _, _ in entries
            if hit.ordinal == 0 or "# Historical bug fix:" in hit.text
        ]
        display = max(overview_candidates or [strongest], key=lambda hit: hit.score)
        support_count = len({purpose for _, _, _, purpose in entries})
        direct_signal_score, reasons, matched_terms = _evidence_for_hit(
            display, strongest, signals, support_count
        )
        concrete = any(
            term in matched_terms for term in (*signals.errors, *signals.paths, *signals.symbols)
        )
        if not concrete and len(matched_terms) < 2:
            continue
        if (
            has_distinctive_evidence
            and not concrete
            and not distinctive_terms.intersection(matched_terms)
        ):
            continue
        semantic_score = _bounded_score(strongest.score)
        support_score = min(support_count / 3, 1.0)
        # Retrieval must remain semantic, but concrete repository evidence should
        # break ties and prevent generic PR language from outranking an exact
        # error, path, or symbol match.
        relevance_score = 0.32 * semantic_score + 0.28 * support_score + 0.40 * direct_signal_score
        metadata = {
            **strongest.metadata,
            "agent_fusion_score": round(fused_score, 5),
            "agent_query_support": support_count,
            "agent_direct_signal_score": round(direct_signal_score, 4),
            "agent_relevance_score": round(relevance_score, 4),
            "agent_match_reasons": reasons,
            "agent_matched_terms": matched_terms,
        }
        scored.append(
            (
                relevance_score,
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
    direct_evidence = float(leading.metadata.get("agent_direct_signal_score", 0))
    if supporting_queries >= 2 and direct_evidence >= 0.25:
        return (
            "high",
            "The leading patch is supported by multiple searches and concrete repository evidence.",
        )
    if direct_evidence >= 0.15 or leading.score >= 0.18:
        return (
            "moderate",
            "The leading patch has meaningful overlap with the reported behavior.",
        )
    return "low", "Evidence was found, but it came from a weak or narrow signal."


def _incident_signals(issue: str) -> _IncidentSignals:
    normalized = " ".join(issue.strip().split())[:1200]
    return _IncidentSignals(
        errors=tuple(_unique(_ERROR_NAME.findall(normalized))[:3]),
        paths=tuple(_unique(_PATH.findall(normalized))[:3]),
        symbols=tuple(
            _unique(match.rstrip("(").strip() for match in _SYMBOL.findall(normalized))[:3]
        ),
        symptoms=tuple(_symptom_terms(normalized)[:24]),
    )


def _evidence_for_hit(
    display: SearchHit,
    strongest: SearchHit,
    signals: _IncidentSignals,
    support_count: int,
) -> tuple[float, list[str], list[str]]:
    corpus = _hit_corpus(display, strongest)
    matched_errors = _matching_terms(signals.errors, corpus)
    matched_paths = _matching_terms(signals.paths, corpus)
    matched_symbols = _matching_symbols(signals.symbols, corpus)
    corpus_tokens = _signal_tokens(corpus)
    matched_symptoms = [term for term in signals.symptoms if _signal_token(term) in corpus_tokens]
    has_tests = bool(strongest.metadata.get("test_files"))

    score = min(0.45, 0.25 * len(matched_errors))
    score += min(0.30, 0.18 * len(matched_paths))
    score += min(0.18, 0.12 * len(matched_symbols))
    score += min(0.14, 0.025 * len(matched_symptoms))
    if has_tests:
        score += 0.08
    score = min(score, 1.0)

    reasons: list[str] = []
    reasons.extend(f"Matches error: {term}" for term in matched_errors[:1])
    reasons.extend(f"Touches {term}" for term in matched_paths[:1])
    reasons.extend(f"References {term}()" for term in matched_symbols[:1])
    if support_count > 1:
        reasons.append(f"Found by {support_count} independent retrieval signals")
    if has_tests:
        reasons.append("Includes regression test coverage")
    if matched_symptoms:
        reasons.append("Shares incident terms: " + ", ".join(matched_symptoms[:4]))
    if not reasons:
        reasons.append("Ranks highly across repository-scoped repair history")
    return (
        score,
        reasons[:4],
        [*matched_errors, *matched_paths, *matched_symbols, *matched_symptoms],
    )


def _hit_corpus(*hits: SearchHit) -> str:
    values: list[str] = []
    for hit in hits:
        values.append(hit.text)
        for key in ("filename", "pull_title", "pull_body", "changed_files", "test_files"):
            value = hit.metadata.get(key)
            if isinstance(value, (list, tuple)):
                values.extend(str(item) for item in value)
            elif value:
                values.append(str(value))
    return "\n".join(values).casefold()


def _matching_terms(terms: tuple[str, ...], corpus: str) -> list[str]:
    return [term for term in terms if term.casefold() in corpus]


def _matching_symbols(symbols: tuple[str, ...], corpus: str) -> list[str]:
    return [
        symbol for symbol in symbols if re.search(rf"\b{re.escape(symbol.casefold())}\s*\(", corpus)
    ]


def _bounded_score(score: float) -> float:
    return min(max(score, 0.0), 1.0)


def _signal_token(token: str) -> str:
    normalized = _normalize_token(token)
    if normalized.startswith(("cancel", "disconnect")):
        return "cancel" if normalized.startswith("cancel") else "disconnect"
    return normalized


def _signal_tokens(text: str) -> set[str]:
    return {_signal_token(token) for token in re.findall(r"[a-z0-9]+", text.casefold())}


def _symptom_terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in _unique([word.lower() for word in _WORD.findall(text)])
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
