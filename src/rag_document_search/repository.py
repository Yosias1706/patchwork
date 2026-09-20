"""PostgreSQL persistence for documents, chunks, and embedding vectors."""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .models import SearchHit, SourceDocument, TextChunk


class SearchIndex(Protocol):
    """Persistence boundary used by the application service."""

    def source_is_current(self, source: str, content_sha256: str) -> bool: ...

    def replace_document(
        self,
        document: SourceDocument,
        document_id: str,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    def search(
        self,
        query_vector: Sequence[float],
        limit: int,
        *,
        repository_id: str | None = None,
    ) -> list[SearchHit]: ...

    def list_documents(self) -> list[dict[str, object]]: ...

    def delete_document(self, document_id: str) -> bool: ...


class PostgresIndex:
    """PostgreSQL index storing metadata in JSONB and embeddings as real arrays.

    Similarity is calculated in Python to keep a plain PostgreSQL deployment
    sufficient. Replace `search` with pgvector SQL when corpus size warrants it.
    """

    def __init__(self, database_url: str) -> None:
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("RAG_DATABASE_URL must be a PostgreSQL connection URL")
        self.database_url = database_url
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        connection = psycopg.connect(self.database_url, row_factory=dict_row)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL UNIQUE,
                    content_sha256 TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding REAL[] NOT NULL,
                    embedding_dimensions INTEGER NOT NULL,
                    UNIQUE(document_id, ordinal)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patchwork_users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patchwork_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES patchwork_users(id) ON DELETE CASCADE,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patchwork_github_connections (
                    user_id TEXT PRIMARY KEY REFERENCES patchwork_users(id) ON DELETE CASCADE,
                    github_user_id TEXT NOT NULL UNIQUE,
                    github_login TEXT NOT NULL,
                    encrypted_access_token TEXT NOT NULL,
                    encrypted_refresh_token TEXT,
                    access_token_expires_at TIMESTAMPTZ,
                    scopes TEXT[] NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patchwork_github_oauth_states (
                    state_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES patchwork_users(id) ON DELETE CASCADE,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patchwork_repositories (
                    id TEXT PRIMARY KEY,
                    owner_user_id TEXT REFERENCES patchwork_users(id) ON DELETE CASCADE,
                    github_url TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    name TEXT NOT NULL,
                    default_branch TEXT NOT NULL,
                    description TEXT,
                    is_private BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                ALTER TABLE patchwork_repositories
                ADD COLUMN IF NOT EXISTS owner_user_id TEXT
                REFERENCES patchwork_users(id) ON DELETE CASCADE
                """
            )
            connection.execute(
                "ALTER TABLE patchwork_repositories "
                "DROP CONSTRAINT IF EXISTS patchwork_repositories_github_url_key"
            )
            connection.execute(
                "ALTER TABLE patchwork_repositories "
                "DROP CONSTRAINT IF EXISTS patchwork_repositories_full_name_key"
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_patchwork_repositories_owner_url
                ON patchwork_repositories(owner_user_id, github_url)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patchwork_sync_runs (
                    id TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL REFERENCES patchwork_repositories(id)
                        ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    requested_pull_limit INTEGER NOT NULL,
                    pull_requests_seen INTEGER NOT NULL DEFAULT 0,
                    patch_cards_indexed INTEGER NOT NULL DEFAULT 0,
                    chunks_indexed INTEGER NOT NULL DEFAULT 0,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    state JSONB NOT NULL DEFAULT '{}'::jsonb,
                    available_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    error TEXT,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "ALTER TABLE patchwork_sync_runs "
                "ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0"
            )
            connection.execute(
                "ALTER TABLE patchwork_sync_runs "
                "ADD COLUMN IF NOT EXISTS state JSONB NOT NULL DEFAULT '{}'::jsonb"
            )
            connection.execute(
                "ALTER TABLE patchwork_sync_runs "
                "ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ NOT NULL "
                "DEFAULT CURRENT_TIMESTAMP"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patchwork_evaluation_runs (
                    id TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL REFERENCES patchwork_repositories(id)
                        ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    benchmark_cases INTEGER NOT NULL DEFAULT 0,
                    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
                    error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMPTZ
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patchwork_evaluation_cases (
                    repository_id TEXT NOT NULL REFERENCES patchwork_repositories(id)
                        ON DELETE CASCADE,
                    expected_source TEXT NOT NULL,
                    expected_pull_number INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    issue_number INTEGER,
                    issue_url TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (repository_id, expected_source)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patchwork_investigations (
                    id TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL REFERENCES patchwork_repositories(id)
                        ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES patchwork_users(id) ON DELETE CASCADE,
                    issue TEXT NOT NULL,
                    result JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_patchwork_sync_runs_repository
                ON patchwork_sync_runs(repository_id, created_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_patchwork_sync_runs_claim
                ON patchwork_sync_runs(status, created_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_patchwork_investigations_repository
                ON patchwork_investigations(repository_id, created_at DESC)
                """
            )

    def source_is_current(self, source: str, content_sha256: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM documents WHERE source = %s AND content_sha256 = %s",
                (source, content_sha256),
            ).fetchone()
        return row is not None

    def replace_document(
        self,
        document: SourceDocument,
        document_id: str,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding")
        if not chunks:
            raise ValueError("Cannot index a document without chunks")
        dimensions = len(vectors[0])
        if not dimensions or any(len(vector) != dimensions for vector in vectors):
            raise ValueError("Embeddings must have a shared, non-zero dimension")

        content_hash = str(document.metadata["content_sha256"])
        with self._connection() as connection:
            connection.execute("DELETE FROM documents WHERE source = %s", (document.source,))
            connection.execute(
                """
                INSERT INTO documents (id, source, content_sha256, content_type, metadata)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    document.source,
                    content_hash,
                    document.content_type,
                    Jsonb(document.metadata),
                ),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO chunks (
                        id, document_id, ordinal, text, start_offset, end_offset, metadata,
                        embedding, embedding_dimensions
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            chunk.id,
                            document_id,
                            chunk.ordinal,
                            chunk.text,
                            chunk.start_offset,
                            chunk.end_offset,
                            Jsonb(chunk.metadata),
                            list(vector),
                            dimensions,
                        )
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ],
                )

    def search(
        self,
        query_vector: Sequence[float],
        limit: int,
        *,
        repository_id: str | None = None,
    ) -> list[SearchHit]:
        if limit <= 0 or not query_vector:
            return []
        dimensions = len(query_vector)
        with self._connection() as connection:
            query = """
                SELECT chunks.id AS chunk_id,
                       chunks.document_id,
                       chunks.ordinal,
                       chunks.text,
                       chunks.start_offset,
                       chunks.end_offset,
                       chunks.metadata AS chunk_metadata,
                       chunks.embedding,
                       documents.source,
                       documents.metadata AS document_metadata
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.embedding_dimensions = %s
            """
            parameters: tuple[object, ...] = (dimensions,)
            if repository_id:
                query += " AND documents.metadata ->> 'patchwork_repository_id' = %s"
                parameters += (repository_id,)
            rows = connection.execute(query, parameters).fetchall()

        scored: list[SearchHit] = []
        for row in rows:
            score = _cosine(query_vector, row["embedding"])
            if score > 0:
                metadata = dict(row["document_metadata"])
                metadata.update(row["chunk_metadata"])
                scored.append(
                    SearchHit(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        source=row["source"],
                        text=row["text"],
                        score=score,
                        ordinal=row["ordinal"],
                        start_offset=row["start_offset"],
                        end_offset=row["end_offset"],
                        metadata=metadata,
                    )
                )
        return sorted(scored, key=lambda hit: hit.score, reverse=True)[:limit]

    def list_documents(self) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT documents.id,
                       documents.source,
                       documents.content_type,
                       documents.metadata,
                       documents.created_at,
                       COUNT(chunks.id)::INTEGER AS chunk_count
                FROM documents
                LEFT JOIN chunks ON chunks.document_id = documents.id
                GROUP BY documents.id, documents.source, documents.content_type,
                         documents.metadata, documents.created_at
                ORDER BY documents.source
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "source": row["source"],
                "content_type": row["content_type"],
                "metadata": row["metadata"],
                "created_at": row["created_at"].isoformat(),
                "chunk_count": row["chunk_count"],
            }
            for row in rows
        ]

    def delete_document(self, document_id: str) -> bool:
        with self._connection() as connection:
            result = connection.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        return result.rowcount > 0

    def create_user(self, email: str, password_hash: str) -> dict[str, object]:
        user_id = str(uuid.uuid4())
        try:
            with self._connection() as connection:
                row = connection.execute(
                    """
                    INSERT INTO patchwork_users (id, email, password_hash)
                    VALUES (%s, %s, %s)
                    RETURNING *
                    """,
                    (user_id, email, password_hash),
                ).fetchone()
        except psycopg.errors.UniqueViolation as error:
            raise ValueError("An account already exists for that email address") from error
        return _user_row(row, include_password_hash=True)

    def get_user_by_email(self, email: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM patchwork_users WHERE email = %s", (email,)
            ).fetchone()
        return _user_row(row, include_password_hash=True) if row else None

    def claim_unowned_repositories(self, user_id: str) -> int:
        """One-time migration path for repositories indexed before account support."""
        with self._connection() as connection:
            result = connection.execute(
                """
                UPDATE patchwork_repositories
                SET owner_user_id = %s
                WHERE owner_user_id IS NULL
                """,
                (user_id,),
            )
        return result.rowcount

    def create_session(
        self, user_id: str, token_hash: str, expires_at: object
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO patchwork_sessions (token_hash, user_id, expires_at)
                VALUES (%s, %s, %s)
                """,
                (token_hash, user_id, expires_at),
            )

    def get_user_for_session(self, token_hash: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT users.*
                FROM patchwork_sessions AS sessions
                JOIN patchwork_users AS users ON users.id = sessions.user_id
                WHERE sessions.token_hash = %s AND sessions.expires_at > CURRENT_TIMESTAMP
                """,
                (token_hash,),
            ).fetchone()
        return _user_row(row) if row else None

    def revoke_session(self, token_hash: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM patchwork_sessions WHERE token_hash = %s", (token_hash,)
            )

    def create_github_oauth_state(
        self, *, state_hash: str, user_id: str, expires_at: object
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM patchwork_github_oauth_states WHERE expires_at <= CURRENT_TIMESTAMP"
            )
            connection.execute(
                """
                INSERT INTO patchwork_github_oauth_states (state_hash, user_id, expires_at)
                VALUES (%s, %s, %s)
                """,
                (state_hash, user_id, expires_at),
            )

    def consume_github_oauth_state(self, state_hash: str) -> str | None:
        """Atomically consume a live OAuth state to prevent replay attacks."""
        with self._connection() as connection:
            row = connection.execute(
                """
                DELETE FROM patchwork_github_oauth_states
                WHERE state_hash = %s AND expires_at > CURRENT_TIMESTAMP
                RETURNING user_id
                """,
                (state_hash,),
            ).fetchone()
        return str(row["user_id"]) if row else None

    def upsert_github_connection(
        self,
        *,
        user_id: str,
        github_user_id: str,
        github_login: str,
        encrypted_access_token: str,
        encrypted_refresh_token: str | None,
        access_token_expires_at: object | None,
        scopes: list[str],
    ) -> dict[str, object]:
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO patchwork_github_connections (
                    user_id, github_user_id, github_login, encrypted_access_token,
                    encrypted_refresh_token, access_token_expires_at, scopes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    github_user_id = EXCLUDED.github_user_id,
                    github_login = EXCLUDED.github_login,
                    encrypted_access_token = EXCLUDED.encrypted_access_token,
                    encrypted_refresh_token = EXCLUDED.encrypted_refresh_token,
                    access_token_expires_at = EXCLUDED.access_token_expires_at,
                    scopes = EXCLUDED.scopes,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING *
                """,
                (
                    user_id,
                    github_user_id,
                    github_login,
                    encrypted_access_token,
                    encrypted_refresh_token,
                    access_token_expires_at,
                    scopes,
                ),
            ).fetchone()
        return _github_connection_row(row)

    def get_github_connection(self, user_id: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM patchwork_github_connections WHERE user_id = %s", (user_id,)
            ).fetchone()
        return _github_connection_row(row, include_tokens=True) if row else None

    def delete_github_connection(self, user_id: str) -> bool:
        with self._connection() as connection:
            result = connection.execute(
                "DELETE FROM patchwork_github_connections WHERE user_id = %s", (user_id,)
            )
        return result.rowcount > 0

    def list_documents_for_repository(self, repository_id: str) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT documents.id,
                       documents.source,
                       documents.metadata,
                       (
                           SELECT chunks.text
                           FROM chunks
                           WHERE chunks.document_id = documents.id
                           ORDER BY chunks.ordinal
                           LIMIT 1
                       ) AS overview_text
                FROM documents
                WHERE documents.metadata ->> 'patchwork_repository_id' = %s
                ORDER BY documents.source
                """,
                (repository_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "source": row["source"],
                "metadata": row["metadata"],
                "overview_text": row["overview_text"],
            }
            for row in rows
        ]

    def delete_documents_for_repository(self, repository_id: str) -> int:
        with self._connection() as connection:
            result = connection.execute(
                "DELETE FROM documents WHERE metadata ->> 'patchwork_repository_id' = %s",
                (repository_id,),
            )
        return result.rowcount

    def upsert_patchwork_repository(
        self, details: dict[str, object], *, user_id: str
    ) -> dict[str, object]:
        repository_id = str(uuid.uuid4())
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO patchwork_repositories (
                    id, owner_user_id, github_url, full_name, owner, name, default_branch,
                    description, is_private
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (owner_user_id, github_url) DO UPDATE SET
                    full_name = EXCLUDED.full_name,
                    owner = EXCLUDED.owner,
                    name = EXCLUDED.name,
                    default_branch = EXCLUDED.default_branch,
                    description = EXCLUDED.description,
                    is_private = EXCLUDED.is_private,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING *
                """,
                (
                    repository_id,
                    user_id,
                    details["github_url"],
                    details["full_name"],
                    details["owner"],
                    details["name"],
                    details["default_branch"],
                    details.get("description"),
                    details.get("is_private", False),
                ),
            ).fetchone()
        return _repository_row(row)

    def get_patchwork_repository(
        self, repository_id: str, *, user_id: str | None = None
    ) -> dict[str, object] | None:
        with self._connection() as connection:
            if user_id:
                row = connection.execute(
                    "SELECT * FROM patchwork_repositories WHERE id = %s AND owner_user_id = %s",
                    (repository_id, user_id),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM patchwork_repositories WHERE id = %s", (repository_id,)
                ).fetchone()
        return _repository_row(row) if row else None

    def get_patchwork_repository_owner(self, repository_id: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT owner_user_id FROM patchwork_repositories WHERE id = %s", (repository_id,)
            ).fetchone()
        return str(row["owner_user_id"]) if row and row["owner_user_id"] else None

    def list_patchwork_repositories(self, *, user_id: str) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT repositories.*,
                       sync_runs.id AS latest_sync_id,
                       sync_runs.status AS latest_sync_status,
                       sync_runs.pull_requests_seen AS latest_sync_pulls_seen,
                       sync_runs.patch_cards_indexed AS latest_sync_patch_cards,
                       sync_runs.chunks_indexed AS latest_sync_chunks,
                       sync_runs.error AS latest_sync_error,
                       sync_runs.created_at AS latest_sync_created_at,
                       sync_runs.finished_at AS latest_sync_finished_at
                FROM patchwork_repositories AS repositories
                LEFT JOIN LATERAL (
                    SELECT * FROM patchwork_sync_runs
                    WHERE repository_id = repositories.id
                    ORDER BY created_at DESC
                    LIMIT 1
                ) AS sync_runs ON TRUE
                WHERE repositories.owner_user_id = %s
                ORDER BY repositories.updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [_repository_row(row, include_sync=True) for row in rows]

    def create_patchwork_sync_run(
        self, repository_id: str, requested_pull_limit: int
    ) -> dict[str, object]:
        run_id = str(uuid.uuid4())
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO patchwork_sync_runs (
                    id, repository_id, status, requested_pull_limit, state
                ) VALUES (%s, %s, 'queued', %s, %s)
                RETURNING *
                """,
                (run_id, repository_id, requested_pull_limit, Jsonb(_initial_sync_state())),
            ).fetchone()
        return _sync_run_row(row)

    def claim_next_patchwork_sync_run(self, *, lease_seconds: int) -> dict[str, object] | None:
        """Claim one queued (or abandoned) run with PostgreSQL row locking.

        This permits one or more worker processes to safely share a durable
        queue. A lease makes a run recoverable after an API process terminates.
        """
        with self._connection() as connection:
            row = connection.execute(
                """
                WITH candidate AS (
                    SELECT id
                    FROM patchwork_sync_runs
                    WHERE (status = 'queued' AND available_at <= CURRENT_TIMESTAMP)
                       OR (
                           status = 'running'
                           AND started_at < CURRENT_TIMESTAMP - (%s * INTERVAL '1 second')
                       )
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE patchwork_sync_runs AS runs
                SET status = 'running',
                    attempt_count = runs.attempt_count + 1,
                    started_at = CURRENT_TIMESTAMP,
                    finished_at = NULL,
                    error = NULL
                FROM candidate
                WHERE runs.id = candidate.id
                RETURNING runs.*
                """,
                (lease_seconds,),
            ).fetchone()
        return _sync_run_row(row) if row else None

    def update_patchwork_sync_run(
        self,
        run_id: str,
        *,
        status: str,
        pull_requests_seen: int | None = None,
        patch_cards_indexed: int | None = None,
        chunks_indexed: int | None = None,
        state: dict[str, object] | None = None,
        retry_after_seconds: int | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        started_at = "CURRENT_TIMESTAMP" if status == "running" else "started_at"
        finished_at = "CURRENT_TIMESTAMP" if status in {"completed", "failed"} else "finished_at"
        with self._connection() as connection:
            row = connection.execute(
                f"""
                UPDATE patchwork_sync_runs SET
                    status = %s,
                    pull_requests_seen = COALESCE(%s, pull_requests_seen),
                    patch_cards_indexed = COALESCE(%s, patch_cards_indexed),
                    chunks_indexed = COALESCE(%s, chunks_indexed),
                    state = COALESCE(%s, state),
                    available_at = CASE
                        WHEN %s::INTEGER IS NULL THEN CURRENT_TIMESTAMP
                        ELSE CURRENT_TIMESTAMP + (%s * INTERVAL '1 second')
                    END,
                    error = %s,
                    started_at = {started_at},
                    finished_at = {finished_at}
                WHERE id = %s
                RETURNING *
                """,
                (
                    status,
                    pull_requests_seen,
                    patch_cards_indexed,
                    chunks_indexed,
                    Jsonb(state) if state is not None else None,
                    retry_after_seconds,
                    retry_after_seconds,
                    error,
                    run_id,
                ),
            ).fetchone()
        if not row:
            raise KeyError(f"Sync run not found: {run_id}")
        return _sync_run_row(row)

    def replace_patchwork_evaluation_cases(
        self, repository_id: str, cases: Sequence[dict[str, object]]
    ) -> None:
        """Persist an evaluation manifest produced during the same repository sync."""
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM patchwork_evaluation_cases WHERE repository_id = %s",
                (repository_id,),
            )
            if not cases:
                return
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO patchwork_evaluation_cases (
                        repository_id, expected_source, expected_pull_number, query,
                        provenance, issue_number, issue_url
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            repository_id,
                            case["expected_source"],
                            case["expected_pull_number"],
                            case["query"],
                            case["provenance"],
                            case.get("issue_number"),
                            case.get("issue_url"),
                        )
                        for case in cases
                    ],
                )

    def list_patchwork_evaluation_cases(self, repository_id: str) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM patchwork_evaluation_cases
                WHERE repository_id = %s
                ORDER BY expected_pull_number
                """,
                (repository_id,),
            ).fetchall()
        return [_evaluation_case_row(row) for row in rows]

    def get_patchwork_sync_run(self, run_id: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM patchwork_sync_runs WHERE id = %s", (run_id,)
            ).fetchone()
        return _sync_run_row(row) if row else None

    def create_patchwork_evaluation_run(self, repository_id: str) -> dict[str, object]:
        run_id = str(uuid.uuid4())
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO patchwork_evaluation_runs (id, repository_id, status)
                VALUES (%s, %s, 'running')
                RETURNING *
                """,
                (run_id, repository_id),
            ).fetchone()
        return _evaluation_run_row(row)

    def finish_patchwork_evaluation_run(
        self,
        run_id: str,
        *,
        status: str,
        benchmark_cases: int,
        metrics: dict[str, object] | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        with self._connection() as connection:
            row = connection.execute(
                """
                UPDATE patchwork_evaluation_runs SET
                    status = %s,
                    benchmark_cases = %s,
                    metrics = %s,
                    error = %s,
                    finished_at = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING *
                """,
                (status, benchmark_cases, Jsonb(metrics or {}), error, run_id),
            ).fetchone()
        if not row:
            raise KeyError(f"Evaluation run not found: {run_id}")
        return _evaluation_run_row(row)

    def get_latest_patchwork_evaluation(self, repository_id: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM patchwork_evaluation_runs
                WHERE repository_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (repository_id,),
            ).fetchone()
        return _evaluation_run_row(row) if row else None

    def save_patchwork_investigation(
        self,
        *,
        repository_id: str,
        user_id: str,
        issue: str,
        result: dict[str, object],
    ) -> dict[str, object]:
        investigation_id = str(uuid.uuid4())
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO patchwork_investigations (
                    id, repository_id, user_id, issue, result
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING *
                """,
                (investigation_id, repository_id, user_id, issue, Jsonb(result)),
            ).fetchone()
        return _investigation_row(row)

    def list_patchwork_investigations(
        self, *, repository_id: str, user_id: str, limit: int = 20
    ) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM patchwork_investigations
                WHERE repository_id = %s AND user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (repository_id, user_id, limit),
            ).fetchall()
        return [_investigation_row(row) for row in rows]

    def delete_patchwork_repository(self, repository_id: str, *, user_id: str) -> bool:
        with self._connection() as connection:
            result = connection.execute(
                """
                DELETE FROM patchwork_repositories
                WHERE id = %s AND owner_user_id = %s
                """,
                (repository_id, user_id),
            )
        deleted = result.rowcount > 0
        if deleted:
            self.delete_documents_for_repository(repository_id)
        return deleted


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _timestamp(value: object) -> str | None:
    return value.isoformat() if value is not None else None


def _user_row(
    row: dict[str, object], *, include_password_hash: bool = False
) -> dict[str, object]:
    user = {
        "id": row["id"],
        "email": row["email"],
        "created_at": _timestamp(row["created_at"]),
    }
    if include_password_hash:
        user["password_hash"] = row["password_hash"]
    return user


def _repository_row(row: dict[str, object], *, include_sync: bool = False) -> dict[str, object]:
    repository = {
        "id": row["id"],
        "github_url": row["github_url"],
        "full_name": row["full_name"],
        "owner": row["owner"],
        "name": row["name"],
        "default_branch": row["default_branch"],
        "description": row["description"],
        "is_private": row["is_private"],
        "created_at": _timestamp(row["created_at"]),
        "updated_at": _timestamp(row["updated_at"]),
    }
    if include_sync:
        repository["latest_sync"] = (
            {
                "id": row["latest_sync_id"],
                "status": row["latest_sync_status"],
                "pull_requests_seen": row["latest_sync_pulls_seen"],
                "patch_cards_indexed": row["latest_sync_patch_cards"],
                "chunks_indexed": row["latest_sync_chunks"],
                "error": row["latest_sync_error"],
                "created_at": _timestamp(row["latest_sync_created_at"]),
                "finished_at": _timestamp(row["latest_sync_finished_at"]),
            }
            if row["latest_sync_id"]
            else None
        )
    return repository


def _sync_run_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "id": row["id"],
        "repository_id": row["repository_id"],
        "status": row["status"],
        "requested_pull_limit": row["requested_pull_limit"],
        "pull_requests_seen": row["pull_requests_seen"],
        "patch_cards_indexed": row["patch_cards_indexed"],
        "chunks_indexed": row["chunks_indexed"],
        "attempt_count": row["attempt_count"],
        "state": row["state"],
        "error": row["error"],
        "created_at": _timestamp(row["created_at"]),
        "started_at": _timestamp(row["started_at"]),
        "finished_at": _timestamp(row["finished_at"]),
    }


def _initial_sync_state() -> dict[str, object]:
    """State for a sync that can safely resume after a worker restart."""
    return {
        "phase": "discovering",
        "next_page": 1,
        "candidate_pull_numbers": [],
        "next_candidate_index": 0,
    }


def _evaluation_run_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "id": row["id"],
        "repository_id": row["repository_id"],
        "status": row["status"],
        "benchmark_cases": row["benchmark_cases"],
        "metrics": row["metrics"],
        "error": row["error"],
        "created_at": _timestamp(row["created_at"]),
        "finished_at": _timestamp(row["finished_at"]),
    }


def _investigation_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "id": row["id"],
        "repository_id": row["repository_id"],
        "issue": row["issue"],
        "result": row["result"],
        "created_at": _timestamp(row["created_at"]),
    }


def _github_connection_row(
    row: dict[str, object], *, include_tokens: bool = False
) -> dict[str, object]:
    connection = {
        "github_user_id": row["github_user_id"],
        "github_login": row["github_login"],
        "access_token_expires_at": _timestamp(row["access_token_expires_at"]),
        "scopes": list(row["scopes"]),
        "created_at": _timestamp(row["created_at"]),
        "updated_at": _timestamp(row["updated_at"]),
    }
    if include_tokens:
        connection["encrypted_access_token"] = row["encrypted_access_token"]
        connection["encrypted_refresh_token"] = row["encrypted_refresh_token"]
    return connection


def _evaluation_case_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "expected_source": row["expected_source"],
        "expected_pull_number": row["expected_pull_number"],
        "query": row["query"],
        "provenance": row["provenance"],
        "issue_number": row["issue_number"],
        "issue_url": row["issue_url"],
        "created_at": _timestamp(row["created_at"]),
    }
