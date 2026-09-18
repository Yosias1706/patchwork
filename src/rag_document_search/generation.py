"""Grounded answer generation with a local fallback and compatible API provider."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from .config import Settings
from .embeddings import _post_json
from .models import SearchHit

_WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


class AnswerGenerator(Protocol):
    def answer(self, question: str, hits: Sequence[SearchHit]) -> str: ...


class ExtractiveAnswerGenerator:
    """Select relevant source sentences and never invent facts outside the corpus."""

    def answer(self, question: str, hits: Sequence[SearchHit]) -> str:
        if not hits:
            return "I couldn't find relevant information in the indexed documents."
        terms = set(_terms(question))
        candidates: list[tuple[float, str, int]] = []
        for source_number, hit in enumerate(hits, start=1):
            for sentence in _SENTENCE_PATTERN.split(hit.text):
                sentence = sentence.strip()
                if sentence:
                    overlap = len(terms.intersection(_terms(sentence)))
                    candidates.append((overlap + hit.score, sentence, source_number))
        if not candidates:
            return "I found relevant passages, but no readable answer could be extracted."
        selected: list[str] = []
        used_sources: set[int] = set()
        for _, sentence, source_number in sorted(candidates, reverse=True):
            if sentence in selected:
                continue
            selected.append(sentence)
            used_sources.add(source_number)
            if len(selected) == 3:
                break
        citations = " ".join(f"[{number}]" for number in sorted(used_sources))
        return f"{' '.join(selected)} {citations}".strip()


class OpenAICompatibleAnswerGenerator:
    """Chat-completions adapter for OpenAI-compatible API servers."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError(
                "RAG_OPENAI_API_KEY is required for openai_compatible answer generation"
            )
        self.base_url = settings.openai_base_url
        self.api_key = settings.openai_api_key
        self.model = settings.chat_model
        self.timeout = settings.request_timeout_seconds

    def answer(self, question: str, hits: Sequence[SearchHit]) -> str:
        if not hits:
            return "I couldn't find relevant information in the indexed documents."
        context = "\n\n".join(
            f"[{number}] Source: {hit.source}\n{hit.text}"
            for number, hit in enumerate(hits, start=1)
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied context. "
                        "If the answer is not present, say so. "
                        "Cite supporting passages using their bracketed source numbers."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        }
        response = _post_json(
            f"{self.base_url}/chat/completions", payload, self.api_key, self.timeout
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Unexpected chat completion response from compatible API") from error
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Compatible API returned an empty answer")
        return content.strip()


def build_answer_generator(settings: Settings) -> AnswerGenerator:
    if settings.llm_provider == "extractive":
        return ExtractiveAnswerGenerator()
    return OpenAICompatibleAnswerGenerator(settings)


def _terms(text: str) -> list[str]:
    return [word.lower() for word in _WORD_PATTERN.findall(text) if len(word) > 2]
