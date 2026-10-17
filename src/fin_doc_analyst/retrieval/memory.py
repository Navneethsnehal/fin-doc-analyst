"""Process-local hybrid store. This is the default backend and the CI backend."""

from __future__ import annotations

import threading

from fin_doc_analyst.retrieval.embeddings import Embedder
from fin_doc_analyst.retrieval.hybrid import HybridIndex, ScoredChunk
from fin_doc_analyst.schemas import Chunk, DocumentInfo


class MemoryStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._docs: dict[str, DocumentInfo] = {}
        self._index = HybridIndex([], embedder)
        self._lock = threading.Lock()

    def add(self, info: DocumentInfo, chunks: list[Chunk]) -> None:
        with self._lock:
            self._chunks = [chunk for chunk in self._chunks if chunk.doc_id != info.doc_id]
            self._chunks.extend(chunks)
            self._docs[info.doc_id] = info
            self._index = HybridIndex(self._chunks, self.embedder)

    def delete(self, doc_id: str) -> None:
        with self._lock:
            self._chunks = [chunk for chunk in self._chunks if chunk.doc_id != doc_id]
            self._docs.pop(doc_id, None)
            self._index = HybridIndex(self._chunks, self.embedder)

    def list_documents(self) -> list[DocumentInfo]:
        with self._lock:
            return list(self._docs.values())

    def all_chunks(self, doc_ids: list[str] | None = None) -> list[Chunk]:
        with self._lock:
            if not doc_ids:
                return list(self._chunks)
            allowed = set(doc_ids)
            return [chunk for chunk in self._chunks if chunk.doc_id in allowed]

    def search(self, query: str, k: int = 4, doc_ids: list[str] | None = None) -> list[ScoredChunk]:
        with self._lock:
            return self._index.search(query, k=k, doc_ids=doc_ids)

    def ping(self) -> bool:
        return True
