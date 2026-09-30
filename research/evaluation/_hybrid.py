"""Hybrid lexical + semantic retrieval via Reciprocal Rank Fusion (RRF).

Composes an already-fitted TfidfRetriever and an already-loaded
EmbeddingRetriever - it does not reimplement either algorithm. Fuses
their full-corpus rankings by RANK POSITION only (standard RRF), never
by blending raw scores: TF-IDF's sparse cosine and Granite's dense
cosine live on different scales and are never directly comparable.
"""

from _tfidf import TfidfRetriever  # moved out of src/knowledge_system/retrieval/ - archived, not production
from knowledge_system.models import DocumentChunk, SearchResult
from knowledge_system.retrieval.embedding import EmbeddingRetriever

_DEFAULT_K = 60


class HybridRetriever:
    """Fuse an already-fit TfidfRetriever and an already-loaded EmbeddingRetriever via RRF.

    Usage:
        tfidf = TfidfRetriever(); tfidf.fit(chunks)
        embedding = EmbeddingRetriever(); embedding.load_index(chunks, embeddings_path, manifest_path)
        hybrid = HybridRetriever(tfidf, embedding, chunks, k=60)
        results = hybrid.search("some question", top_k=3)

    RRF score for a chunk = 1/(k + lexical_rank) + 1/(k + semantic_rank),
    using 1-based ranks from each retriever's FULL-corpus ranking (not
    just its own top_k). A chunk missing from one side's ranking simply
    contributes 0 from that side - it is not treated as an error, since
    every chunk in the corpus should appear in both full rankings anyway.
    """

    def __init__(
        self,
        lexical_retriever: TfidfRetriever,
        semantic_retriever: EmbeddingRetriever,
        chunks: list[DocumentChunk],
        k: int = _DEFAULT_K,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")

        chunk_indexes = [chunk.chunk_index for chunk in chunks]
        if len(chunk_indexes) != len(set(chunk_indexes)):
            raise ValueError("duplicate chunk_index values found - RRF fusion requires unique chunk identities")

        # Both underlying retrievers must already be ready - HybridRetriever
        # composes them, it does not fit/load them itself.
        if getattr(lexical_retriever, "_chunks", None) is None:
            raise RuntimeError("lexical_retriever must be fit() before constructing HybridRetriever")
        if getattr(semantic_retriever, "_chunks", None) is None:
            raise RuntimeError("semantic_retriever must be fit()/load_index()ed before constructing HybridRetriever")

        expected_indexes = set(chunk_indexes)
        lexical_indexes = {c.chunk_index for c in lexical_retriever._chunks}
        semantic_indexes = {c.chunk_index for c in semantic_retriever._chunks}
        if lexical_indexes != expected_indexes or semantic_indexes != expected_indexes:
            raise ValueError(
                "lexical_retriever/semantic_retriever were fit against a different chunk "
                "collection than the `chunks` passed to HybridRetriever"
            )

        self._lexical = lexical_retriever
        self._semantic = semantic_retriever
        self._chunks_by_index = {chunk.chunk_index: chunk for chunk in chunks}
        self._k = k

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        """Return up to `top_k` chunks ranked by RRF-fused score, highest first."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        corpus_size = len(self._chunks_by_index)
        # Rank the FULL corpus from each side, not just top_k - RRF needs
        # every chunk's rank position, and 235 chunks makes this trivial.
        lexical_results = self._lexical.search(question, top_k=corpus_size)
        semantic_results = self._semantic.search(question, top_k=corpus_size)

        lexical_ranks = {r.chunk.chunk_index: rank for rank, r in enumerate(lexical_results, start=1)}
        semantic_ranks = {r.chunk.chunk_index: rank for rank, r in enumerate(semantic_results, start=1)}

        rrf_scores: dict[int, float] = {}
        for chunk_index in self._chunks_by_index:
            score = 0.0
            if chunk_index in lexical_ranks:
                score += 1.0 / (self._k + lexical_ranks[chunk_index])
            if chunk_index in semantic_ranks:
                score += 1.0 / (self._k + semantic_ranks[chunk_index])
            rrf_scores[chunk_index] = score

        # Sort by RRF score descending; ties broken by chunk_index
        # ascending only - never by the underlying raw TF-IDF/cosine
        # scores, which aren't comparable across the two retrievers.
        ranked_indexes = sorted(rrf_scores, key=lambda idx: (-rrf_scores[idx], idx))

        return [
            SearchResult(chunk=self._chunks_by_index[idx], score=rrf_scores[idx])
            for idx in ranked_indexes[:top_k]
        ]
