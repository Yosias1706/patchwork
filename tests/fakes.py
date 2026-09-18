"""In-memory index used to exercise the service without a running Postgres server."""

from __future__ import annotations

import math
from collections.abc import Sequence

from rag_document_search.models import SearchHit, SourceDocument, TextChunk


class InMemoryIndex:
    def __init__(self) -> None:
        self._documents: dict[str, tuple[SourceDocument, str, str]] = {}
        self._records: list[tuple[TextChunk, list[float], SourceDocument]] = []

    def source_is_current(self, source: str, content_sha256: str) -> bool:
        record = self._documents.get(source)
        return record is not None and record[2] == content_sha256

    def replace_document(
        self,
        document: SourceDocument,
        document_id: str,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        self._records = [record for record in self._records if record[2].source != document.source]
        self._documents[document.source] = (
            document,
            document_id,
            str(document.metadata["content_sha256"]),
        )
        self._records.extend(
            (chunk, list(vector), document) for chunk, vector in zip(chunks, vectors, strict=True)
        )

    def search(
        self,
        query_vector: Sequence[float],
        limit: int,
        *,
        repository_id: str | None = None,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for chunk, vector, document in self._records:
            if repository_id and document.metadata.get("patchwork_repository_id") != repository_id:
                continue
            score = _cosine(query_vector, vector)
            if score > 0:
                hits.append(
                    SearchHit(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        source=document.source,
                        text=chunk.text,
                        score=score,
                        ordinal=chunk.ordinal,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                        metadata={**document.metadata, **chunk.metadata},
                    )
                )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]

    def list_documents(self) -> list[dict[str, object]]:
        return [
            {
                "id": document_id,
                "source": document.source,
                "content_type": document.content_type,
                "metadata": document.metadata,
                "created_at": "test",
                "chunk_count": sum(
                    1 for chunk, _, _ in self._records if chunk.document_id == document_id
                ),
            }
            for document, document_id, _ in self._documents.values()
        ]

    def delete_document(self, document_id: str) -> bool:
        source = next(
            (
                source
                for source, (_, stored_id, _) in self._documents.items()
                if stored_id == document_id
            ),
            None,
        )
        if source is None:
            return False
        del self._documents[source]
        self._records = [record for record in self._records if record[0].document_id != document_id]
        return True


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
