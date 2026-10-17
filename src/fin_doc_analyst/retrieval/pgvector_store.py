"""Postgres + pgvector dense search, fused with in-process BM25.

BM25 stays in process because stock pgvector images do not ship a BM25 extension.
Dense search is the part that needs the database. Both lists are fused with the
same reciprocal-rank function as the memory backend.
"""

from __future__ import annotations

import threading
from typing import Any

from fin_doc_analyst.retrieval.embeddings import Embedder
from fin_doc_analyst.retrieval.hybrid import ScoredChunk, rerank, rrf_fuse
from fin_doc_analyst.schemas import Chunk, DocumentInfo
from fin_doc_analyst.textutil import expand, query_tokens, tokenize

try:
    import psycopg
    from pgvector.psycopg import register_vector
except ImportError:  # pragma: no cover - declared dependency
    psycopg = None
    register_vector = None


class PgVectorStore:
    def __init__(self, database_url: str, embedder: Embedder) -> None:
        if psycopg is None or register_vector is None:
            raise RuntimeError("psycopg and pgvector are required for the pgvector backend")
        if not database_url:
            raise RuntimeError("DATABASE_URL is required for the pgvector backend")
        self.embedder = embedder
        self._lock = threading.Lock()
        self._conn = psycopg.connect(database_url, autocommit=True)
        register_vector(self._conn)
        self._ensure_schema()
        self._chunks: list[Chunk] = []
        self._reload()

    def _ensure_schema(self) -> None:
        dim = int(self.embedder.dim)
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_name TEXT NOT NULL,
                pages INTEGER NOT NULL,
                chunks INTEGER NOT NULL
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                doc_name TEXT NOT NULL,
                page INTEGER NOT NULL,
                section TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding vector({dim}) NOT NULL
            )
            """,
        ]
        with self._lock:
            with self._conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
                try:
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
                        ON chunks USING hnsw (embedding vector_cosine_ops)
                        """
                    )
                except Exception:
                    # Older pgvector builds may lack HNSW. Exact search still works.
                    # autocommit is on, so the failed statement does not abort the connection.
                    pass

    def _reload(self) -> None:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, doc_name, page, section, text
                    FROM chunks
                    ORDER BY chunk_id
                    """
                )
                rows = cur.fetchall()
        self._chunks = [
            Chunk(
                chunk_id=row[0],
                doc_id=row[1],
                doc_name=row[2],
                page=row[3],
                section=row[4],
                text=row[5],
            )
            for row in rows
        ]

    def add(self, info: DocumentInfo, chunks: list[Chunk]) -> None:
        embeddings = (
            self.embedder.embed_documents([chunk.text for chunk in chunks]) if chunks else []
        )
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE doc_id = %s", (info.doc_id,))
                cur.execute("DELETE FROM documents WHERE doc_id = %s", (info.doc_id,))
                cur.execute(
                    """
                    INSERT INTO documents (doc_id, doc_name, pages, chunks)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (info.doc_id, info.doc_name, info.pages, info.chunks),
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    cur.execute(
                        """
                        INSERT INTO chunks
                            (chunk_id, doc_id, doc_name, page, section, text, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.doc_name,
                            chunk.page,
                            chunk.section,
                            chunk.text,
                            embedding,
                        ),
                    )
        self._reload()

    def delete(self, doc_id: str) -> None:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
        self._reload()

    def list_documents(self) -> list[DocumentInfo]:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT doc_id, doc_name, pages, chunks FROM documents ORDER BY doc_name"
                )
                rows = cur.fetchall()
        return [
            DocumentInfo(doc_id=row[0], doc_name=row[1], pages=row[2], chunks=row[3])
            for row in rows
        ]

    def all_chunks(self, doc_ids: list[str] | None = None) -> list[Chunk]:
        if not doc_ids:
            return list(self._chunks)
        allowed = set(doc_ids)
        return [chunk for chunk in self._chunks if chunk.doc_id in allowed]

    def search(self, query: str, k: int = 4, doc_ids: list[str] | None = None) -> list[ScoredChunk]:
        dense_ids = self._dense_ids(query, doc_ids)
        bm25_ids = self._bm25_ids(query, doc_ids)
        fused = rrf_fuse([bm25_ids, dense_ids])
        by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        ranked = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)[:20]
        scored = [
            ScoredChunk(by_id[chunk_id], fused[chunk_id])
            for chunk_id in ranked
            if chunk_id in by_id
        ]
        return rerank(query, scored, k)

    def _dense_ids(self, query: str, doc_ids: list[str] | None) -> list[str]:
        vector = self.embedder.embed_query(query)
        sql = """
            SELECT chunk_id
            FROM chunks
            {where}
            ORDER BY embedding <=> %s
            LIMIT 20
        """
        where = ""
        params: list[Any] = []
        if doc_ids:
            where = "WHERE doc_id = ANY(%s)"
            params.append(doc_ids)
        params.append(vector)
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(sql.format(where=where), params)
                return [row[0] for row in cur.fetchall()]

    def _bm25_ids(self, query: str, doc_ids: list[str] | None) -> list[str]:
        from rank_bm25 import BM25Okapi

        chunks = self.all_chunks(doc_ids)
        if not chunks:
            return []
        tokenized = [tokenize(chunk.text) or ["empty"] for chunk in chunks]
        index = BM25Okapi(tokenized)
        tokens = tokenize(" ".join(sorted(expand(query_tokens(query))))) or tokenize(query)
        scores = index.get_scores(tokens or ["empty"])
        order = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
        return [chunks[i].chunk_id for i in order if float(scores[i]) > 0]

    def ping(self) -> bool:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
        return bool(row and row[0] == 1)
