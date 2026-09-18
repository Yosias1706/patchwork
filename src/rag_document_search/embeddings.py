"""Embedding providers with a dependable local default."""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from .config import Settings

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_STOP_WORDS = frozenset(
    (
        "a an and are as at be by for from has have if in is it its "
        "of on or that the to was were with"
    ).split()
)
_NORMALIZED_TERMS = {
    "retention": "retain",
    "retained": "retain",
    "retaining": "retain",
    "deletion": "delete",
    "deleted": "delete",
    "deleting": "delete",
}


class Embedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """A deterministic, dependency-free feature hashing embedder.

    It is a strong lexical baseline and keeps this project fully runnable offline.
    Use a semantic API provider when retrieval must understand paraphrases.
    """

    def __init__(self, dimensions: int) -> None:
        if dimensions < 64:
            raise ValueError("Hashing embedding dimensions must be at least 64")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        tokens = [
            _normalize_token(token)
            for token in _TOKEN_PATTERN.findall(text)
            if token.lower() not in _STOP_WORDS
        ]
        vector = [0.0] * self.dimensions
        counts = Counter(tokens)
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class OpenAICompatibleEmbedder:
    """Minimal client for OpenAI-compatible `/embeddings` endpoints."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("RAG_OPENAI_API_KEY is required for openai_compatible embeddings")
        self.base_url = settings.openai_base_url
        self.api_key = settings.openai_api_key
        self.model = settings.embedding_model
        self.timeout = settings.request_timeout_seconds
        self._dimensions: int | None = None

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise RuntimeError("Embedding dimensions are unknown until the first embedding request")
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": list(texts)}
        response = _post_json(f"{self.base_url}/embeddings", payload, self.api_key, self.timeout)
        try:
            items = sorted(response["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in items]
            self._dimensions = len(vectors[0]) if vectors else self._dimensions
            if not vectors or any(len(vector) != self._dimensions for vector in vectors):
                raise ValueError("Embedding API returned inconsistent vectors")
            return [[float(value) for value in vector] for vector in vectors]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Unexpected embeddings response from compatible API") from error


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hashing":
        return HashingEmbedder(settings.hashing_dimensions)
    return OpenAICompatibleEmbedder(settings)


def _normalize_token(token: str) -> str:
    normalized = token.lower()
    if normalized in _NORMALIZED_TERMS:
        return _NORMALIZED_TERMS[normalized]
    if len(normalized) > 4 and normalized.endswith("ies"):
        return f"{normalized[:-3]}y"
    if len(normalized) > 4 and normalized.endswith("s"):
        return normalized[:-1]
    return normalized


def _post_json(
    url: str, payload: dict[str, object], api_key: str, timeout: int
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Compatible API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach compatible API: {error.reason}") from error
