"""Application service coordinating parsing, indexing, search, and answering."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from .chunking import TextSplitter
from .config import Settings
from .embeddings import Embedder, build_embedder
from .generation import AnswerGenerator, build_answer_generator
from .loaders import iter_supported_paths, load_path
from .models import Answer, IngestReport, SearchHit, SourceDocument
from .repository import PostgresIndex, SearchIndex


class RAGService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        index: SearchIndex | None = None,
        embedder: Embedder | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.settings = settings or Settings.from_environment()
        self.index = index or PostgresIndex(self.settings.database_url)
        self.embedder = embedder or build_embedder(self.settings)
        self.generator = generator or build_answer_generator(self.settings)
        self.splitter = TextSplitter(self.settings.chunk_size, self.settings.chunk_overlap)

    def ingest_directory(self, directory: Path, *, continue_on_error: bool = True) -> IngestReport:
        failures: list[str] = []
        documents: list[SourceDocument] = []
        for path in iter_supported_paths(directory):
            try:
                documents.append(load_path(path))
            except (OSError, RuntimeError, ValueError) as error:
                if not continue_on_error:
                    raise
                failures.append(f"{path}: {error}")
        report = self.ingest_documents(documents)
        return IngestReport(
            documents_seen=report.documents_seen,
            documents_indexed=report.documents_indexed,
            documents_unchanged=report.documents_unchanged,
            chunks_indexed=report.chunks_indexed,
            failures=tuple(failures) + report.failures,
        )

    def ingest_file(self, path: Path, *, source: str | None = None) -> IngestReport:
        return self.ingest_documents([load_path(path, source=source)])

    def ingest_documents(self, documents: Iterable[SourceDocument]) -> IngestReport:
        seen = indexed = unchanged = chunks_indexed = 0
        failures: list[str] = []
        for document in documents:
            seen += 1
            generated_hash = hashlib.sha256(document.text.encode()).hexdigest()
            content_hash = str(document.metadata.get("content_sha256") or generated_hash)
            if self.index.source_is_current(document.source, content_hash):
                unchanged += 1
                continue
            try:
                document_id = hashlib.sha256(document.source.encode("utf-8")).hexdigest()
                chunks = self.splitter.split(document_id, document.text)
                vectors = self.embedder.embed([chunk.text for chunk in chunks])
                metadata = {**document.metadata, "content_sha256": content_hash}
                normalized_document = SourceDocument(
                    source=document.source,
                    text=document.text,
                    content_type=document.content_type,
                    metadata=metadata,
                )
                self.index.replace_document(normalized_document, document_id, chunks, vectors)
                indexed += 1
                chunks_indexed += len(chunks)
            except (OSError, RuntimeError, ValueError) as error:
                failures.append(f"{document.source}: {error}")
        return IngestReport(seen, indexed, unchanged, chunks_indexed, tuple(failures))

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        repository_id: str | None = None,
    ) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        vectors = self.embedder.embed([query])
        return self.index.search(
            vectors[0], limit or self.settings.top_k, repository_id=repository_id
        )

    def ask(
        self,
        question: str,
        *,
        limit: int | None = None,
        repository_id: str | None = None,
    ) -> Answer:
        hits = self.search(question, limit=limit, repository_id=repository_id)
        return Answer(text=self.generator.answer(question, hits), citations=tuple(hits))

    def list_documents(self) -> list[dict[str, object]]:
        return self.index.list_documents()

    def delete_document(self, document_id: str) -> bool:
        return self.index.delete_document(document_id)
