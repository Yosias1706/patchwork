"""Grounded implementation advice built from Patchwork retrieval evidence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .config import Settings
from .embeddings import _post_json
from .models import SearchHit


@dataclass(frozen=True, slots=True)
class RecommendedSolution:
    """A proposal that is useful to review but never treated as an automatic patch."""

    assessment: str
    implementation_steps: tuple[str, ...]
    code_guidance: str | None
    verification: tuple[str, ...]
    cautions: tuple[str, ...]
    source_pull_numbers: tuple[int, ...]
    mode: str

    def as_api_details(self) -> dict[str, object]:
        return asdict(self)


class SolutionAgent:
    """Turn retrieved repair history into an evidence-bounded implementation plan.

    Retrieval happens before this agent runs. The agent receives only citations
    from the active repository, so it cannot search other repositories or make
    repository changes. A model-backed generator can enhance the plan when one
    is configured; the deterministic plan remains useful without an API key.
    """

    def __init__(self, generator: SolutionGenerator) -> None:
        self.generator = generator

    def recommend(
        self,
        issue: str,
        matches: Sequence[dict[str, object]],
        citations: Sequence[SearchHit],
    ) -> RecommendedSolution:
        evidence = _evidence(matches, citations)
        fallback = _evidence_guided_solution(issue, evidence)
        try:
            return self.generator.propose(issue, evidence, fallback)
        except (RuntimeError, ValueError, KeyError, TypeError):
            # A temporary model failure must never make a completed retrieval
            # unusable. The UI explicitly identifies this fallback mode.
            return fallback


class SolutionGenerator:
    """Boundary for optional model-assisted recommendation generation."""

    def propose(
        self,
        issue: str,
        evidence: Sequence[dict[str, object]],
        fallback: RecommendedSolution,
    ) -> RecommendedSolution:
        return fallback


class OpenAICompatibleSolutionGenerator(SolutionGenerator):
    """Generate a constrained proposal from the citations already selected by RAG."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("RAG_OPENAI_API_KEY is required for model-assisted solutions")
        self.base_url = settings.openai_base_url
        self.api_key = settings.openai_api_key
        self.model = settings.chat_model
        self.timeout = settings.request_timeout_seconds

    def propose(
        self,
        issue: str,
        evidence: Sequence[dict[str, object]],
        fallback: RecommendedSolution,
    ) -> RecommendedSolution:
        if not evidence:
            return fallback
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Patchwork's implementation advisor. Use only the supplied "
                        "repository evidence. Return JSON with assessment, implementation_steps, "
                        "code_guidance, verification, and cautions. Keep every field concise. "
                        "A code_guidance value may be a small illustrative sketch, but it must not "
                        "invent APIs, paths, or behavior not supported by the evidence. "
                        "If evidence is incomplete, set code_guidance to null and state the "
                        "uncertainty in cautions."
                    ),
                },
                {
                    "role": "user",
                    "content": "Issue report:\n"
                    + issue[:3000]
                    + "\n\nRetrieved repair evidence:\n"
                    + json.dumps(list(evidence), ensure_ascii=False),
                },
            ],
        }
        response = _post_json(
            f"{self.base_url}/chat/completions", payload, self.api_key, self.timeout
        )
        try:
            content = response["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("Model returned an invalid solution proposal") from error
        if not isinstance(parsed, dict):
            raise RuntimeError("Model returned an invalid solution proposal")
        return RecommendedSolution(
            assessment=_text(parsed.get("assessment"), fallback.assessment, 700),
            implementation_steps=_text_list(
                parsed.get("implementation_steps"), fallback.implementation_steps, 5
            ),
            code_guidance=_optional_text(parsed.get("code_guidance"), 1800),
            verification=_text_list(parsed.get("verification"), fallback.verification, 4),
            cautions=_text_list(parsed.get("cautions"), fallback.cautions, 3),
            source_pull_numbers=fallback.source_pull_numbers,
            mode="model_assisted",
        )


def build_solution_generator(settings: Settings) -> SolutionGenerator:
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleSolutionGenerator(settings)
    return SolutionGenerator()


def _evidence(
    matches: Sequence[dict[str, object]], citations: Sequence[SearchHit]
) -> list[dict[str, object]]:
    by_pull_url = {str(match.get("pull_url")): match for match in matches}
    entries: list[dict[str, object]] = []
    for hit in citations:
        match = by_pull_url.get(hit.source)
        if not match or any(entry["pull_url"] == hit.source for entry in entries):
            continue
        entries.append(
            {
                "pull_number": match.get("pull_number"),
                "pull_url": hit.source,
                "title": str(match.get("title") or "Historical fix"),
                "excerpt": str(match.get("excerpt") or ""),
                "changed_files": [str(path) for path in match.get("changed_files", [])[:5]],
                "test_files": [str(path) for path in match.get("test_files", [])[:3]],
                "context": hit.text[:1800],
            }
        )
        if len(entries) == 3:
            break
    return entries


def _evidence_guided_solution(
    issue: str, evidence: Sequence[dict[str, object]]
) -> RecommendedSolution:
    if not evidence:
        return RecommendedSolution(
            assessment=(
                "No close historical fix was found, so Patchwork cannot responsibly suggest an "
                "implementation yet."
            ),
            implementation_steps=(
                "Narrow the report with an error message, affected symbol, or file path.",
                "Sync more repository history, then investigate again.",
            ),
            code_guidance=None,
            verification=(),
            cautions=("Do not infer a fix from unrelated repository changes.",),
            source_pull_numbers=(),
            mode="evidence_guided",
        )

    lead = evidence[0]
    pull_number = lead.get("pull_number")
    files = [str(path) for path in lead.get("changed_files", [])]
    tests = [str(path) for path in lead.get("test_files", [])]
    inspection_target = ", ".join(files[:3]) or "the files changed by the reference PR"
    steps = [
        f"Compare the reported behavior with PR #{pull_number} and inspect {inspection_target}.",
        (
            "Apply the same narrow control-flow or validation pattern only where the current "
            "code has the matching failure mode."
        ),
    ]
    if tests:
        steps.append(f"Use {', '.join(tests[:2])} as the starting point for a regression test.")
    source_numbers = tuple(
        int(entry["pull_number"]) for entry in evidence if isinstance(entry.get("pull_number"), int)
    )
    return RecommendedSolution(
        assessment=(
            f"PR #{pull_number} is the closest available implementation reference for this report. "
            "Use it as a pattern to verify, not as a patch to copy blindly."
        ),
        implementation_steps=tuple(steps),
        code_guidance=None,
        verification=tuple(
            [f"Run or extend the regression coverage in {path}." for path in tests[:2]]
            or [
                "Add a regression test that reproduces the reported behavior before and after "
                "the change."
            ]
        ),
        cautions=(
            "Repository history is evidence, not proof that the current code has the same root "
            "cause.",
        ),
        source_pull_numbers=source_numbers,
        mode="evidence_guided",
    )


def _text(value: object, fallback: str, maximum: int) -> str:
    text = str(value).strip() if isinstance(value, str) else ""
    return text[:maximum] if text else fallback


def _optional_text(value: object, maximum: int) -> str | None:
    text = str(value).strip() if isinstance(value, str) else ""
    return text[:maximum] if text else None


def _text_list(value: object, fallback: Sequence[str], maximum_items: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return tuple(fallback)
    entries = tuple(str(item).strip()[:500] for item in value if str(item).strip())
    return entries[:maximum_items] or tuple(fallback)
