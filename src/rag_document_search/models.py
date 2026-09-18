"""Small immutable domain models shared by the application layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceDocument:
    source: str
    text: str
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TextChunk:
    id: str
    document_id: str
    ordinal: int
    text: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    document_id: str
    source: str
    text: str
    score: float
    ordinal: int
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IngestReport:
    documents_seen: int
    documents_indexed: int
    documents_unchanged: int
    chunks_indexed: int
    failures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[SearchHit, ...]
