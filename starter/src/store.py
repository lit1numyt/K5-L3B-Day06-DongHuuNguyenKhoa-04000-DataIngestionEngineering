"""Persistent PostgreSQL + pgvector storage for chunks and metadata."""
import json
from typing import Any

import psycopg
from psycopg import sql

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        statement = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS rag_chunks (
                strategy TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({dimensions}) NOT NULL,
                PRIMARY KEY (strategy, chunk_id)
            )
            """
        ).format(dimensions=sql.Literal(self.dimensions))
        with psycopg.connect(self.database_url) as connection:
            connection.execute(statement)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS rag_chunks_strategy_idx ON rag_chunks (strategy)"
            )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace one strategy atomically so stale chunks cannot survive re-ingest."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if any(len(vector) != self.dimensions for vector in vectors):
            raise ValueError("one or more vectors has the wrong dimension")

        statement = """
            INSERT INTO rag_chunks
                (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                source = EXCLUDED.source,
                page = EXCLUDED.page,
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
        """
        with psycopg.connect(self.database_url) as connection:
            connection.execute("DELETE FROM rag_chunks WHERE strategy = %s", (strategy,))
            for chunk, vector in zip(chunks, vectors):
                metadata = dict(chunk.metadata)
                metadata.setdefault("document_id", chunk.document_id)
                metadata.setdefault("source", chunk.source)
                metadata.setdefault("page", chunk.page)
                connection.execute(
                    statement,
                    (
                        strategy,
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.source,
                        chunk.page,
                        chunk.text,
                        json.dumps(metadata, ensure_ascii=False),
                        self._vector_literal(vector),
                    ),
                )

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Search by cosine similarity and apply whitelisted metadata filters."""
        if len(query_vector) != self.dimensions:
            raise ValueError("query vector has the wrong dimension")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        allowed_filters = {
            "section",
            "year",
            "version",
            "category",
            "audience",
            "language",
            "document_type",
        }
        filters = filters or {}
        unknown = set(filters) - allowed_filters
        if unknown:
            raise ValueError(f"unsupported metadata filters: {sorted(unknown)}")

        conditions = [sql.SQL("strategy = %s")]
        parameters: list[Any] = [self._vector_literal(query_vector), strategy]
        for key, value in filters.items():
            conditions.append(sql.SQL("metadata ->> %s = %s"))
            parameters.extend([key, str(value)])
        parameters.extend([self._vector_literal(query_vector), top_k])
        query = sql.SQL(
            """
            SELECT chunk_id, document_id, text, source, page, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM rag_chunks
            WHERE {conditions}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
        ).format(conditions=sql.SQL(" AND ").join(conditions))
        with psycopg.connect(self.database_url) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "text": row[2],
                "source": row[3],
                "page": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        return "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"
