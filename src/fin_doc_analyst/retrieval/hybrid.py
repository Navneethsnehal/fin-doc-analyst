"""BM25 + dense fusion and a lexical rerank step.

Reciprocal rank fusion keeps a rare token such as "overdraft" from being
washed out by the embedding, and lets a paraphrase still surface through
the dense list. The reranker then prefers passages that actually contain
the discriminative query terms.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from fin_doc_analyst.retrieval.embeddings import Embedder
from fin_doc_analyst.schemas import Chunk
from fin_doc_analyst.textutil import expand, query_tokens, tokenize

RRF_K = 60


class ScoredChunk:
    def __init__(self, chunk: Chunk, score: float) -> None:
        self.chunk = chunk
        self.score = score


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] += 1.0 / (k + rank + 1)
    return dict(scores)


class HybridIndex:
    """In-process BM25 + embedding index over a chunk list."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks = list(chunks)
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        tokenized = [tokenize(chunk.text) or ["empty"] for chunk in self.chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        if self.chunks:
            matrix = np.asarray(
                embedder.embed_documents([c.text for c in self.chunks]), dtype=np.float64
            )
        else:
            matrix = np.zeros((0, embedder.dim), dtype=np.float64)
        self._matrix = matrix

    def search(self, query: str, k: int = 4, doc_ids: list[str] | None = None) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        allowed = set(doc_ids) if doc_ids else None
        candidate_indexes = [
            index
            for index, chunk in enumerate(self.chunks)
            if allowed is None or chunk.doc_id in allowed
        ]
        if not candidate_indexes:
            return []

        bm25_order = self._bm25_rank(query, candidate_indexes)
        dense_order = self._dense_rank(query, candidate_indexes)
        fused = rrf_fuse([bm25_order, dense_order])
        ranked_ids = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)[:20]
        scored = [
            ScoredChunk(self._by_id[chunk_id], fused[chunk_id])
            for chunk_id in ranked_ids
            if chunk_id in self._by_id
        ]
        return rerank(query, scored, k)

    def _bm25_rank(self, query: str, indexes: list[int]) -> list[str]:
        if self._bm25 is None:
            return []
        tokens = tokenize(" ".join(sorted(expand(query_tokens(query))))) or tokenize(query)
        scores = self._bm25.get_scores(tokens or ["empty"])
        ordered = sorted(indexes, key=lambda index: float(scores[index]), reverse=True)
        return [self.chunks[index].chunk_id for index in ordered if float(scores[index]) > 0]

    def _dense_rank(self, query: str, indexes: list[int]) -> list[str]:
        query_vector = np.asarray(self.embedder.embed_query(query), dtype=np.float64)
        if self._matrix.size == 0 or float(np.linalg.norm(query_vector)) == 0:
            return []
        scores = self._matrix @ query_vector
        ordered = sorted(indexes, key=lambda index: float(scores[index]), reverse=True)
        nonzero = [index for index in ordered if float(scores[index]) > 0]
        return [self.chunks[index].chunk_id for index in nonzero]


def rerank(query: str, items: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    """Lexical rerank over the fused candidate list.

    This is intentionally model-free so CI does not download a cross-encoder.
    The callable shape matches a cross-encoder reranker if one is swapped in.
    """
    if not items:
        return []
    focus = query_tokens(query)
    peak = max(item.score for item in items) or 1.0
    rescored: list[ScoredChunk] = []
    for item in items:
        haystack = set(tokenize(item.chunk.text))
        lexical = 0.0
        if focus:
            lexical = sum(1 for token in focus if expand([token]) & haystack) / len(focus)
        phrase_boost = 0.12 if focus and set(focus) <= haystack else 0.0
        fused_norm = item.score / peak
        score = 0.62 * lexical + 0.38 * fused_norm + phrase_boost
        rescored.append(ScoredChunk(item.chunk, score))
    rescored.sort(key=lambda item: item.score, reverse=True)
    if not rescored:
        return []
    best = rescored[0].score or 1.0
    kept = [item for item in rescored if item.score >= best * 0.55]
    return kept[:top_k]
