"""TF-IDF lexical retrieval baseline: DocumentChunk[] -> ranked SearchResult[].

A simple, understandable baseline against which later retrieval
improvements get compared. Knows nothing about how chunks were produced
(Markdown vs PDF, chunking strategy) - it only operates on the chunk
list handed to `fit()`.
"""

from sklearn.feature_extraction.text import TfidfVectorizer  # the TF-IDF implementation itself
from sklearn.metrics.pairwise import cosine_similarity  # scores query against every fitted chunk in one call

from knowledge_system.models import DocumentChunk, SearchResult


def _searchable_text(chunk: DocumentChunk) -> str:
    """Build the text TF-IDF indexes for one chunk, without mutating chunk.text.

    When section hierarchy is available, prepending it gives the
    vectorizer extra vocabulary to match on (e.g. a question containing
    "rail travel" can match a chunk whose body never says that phrase, but
    whose section heading does). Falls back to the chunk body alone when
    there's no section metadata.
    """
    if chunk.section:
        return f"{chunk.section}\n\n{chunk.text}"
    return chunk.text


class TfidfRetriever:
    """Fit once on a chunk collection, then search it repeatedly.

    Usage:
        retriever = TfidfRetriever()
        retriever.fit(chunks)
        results = retriever.search("some question", top_k=3)
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self._chunks: list[DocumentChunk] | None = None
        self._matrix = None  # sparse TF-IDF matrix, one row per chunk, set by fit()

    def fit(self, chunks: list[DocumentChunk]) -> None:
        """Store the chunks (in the given order) and fit the vectorizer against their searchable text."""
        self._chunks = list(chunks)
        searchable_texts = [_searchable_text(chunk) for chunk in self._chunks]
        self._matrix = self._vectorizer.fit_transform(searchable_texts)

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        """Return up to `top_k` chunks ranked by cosine similarity to `question`, highest first."""
        if self._chunks is None or self._matrix is None:
            raise RuntimeError("TfidfRetriever.search() called before fit()")
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self._vectorizer.transform([question])
        # cosine_similarity(query, matrix) returns a (1, n_chunks) array;
        # [0] flattens it to one similarity score per chunk.
        scores = cosine_similarity(query_vector, self._matrix)[0]

        # Sort by score descending; ties broken by chunk_index ascending so
        # results are reproducible across runs instead of depending on
        # whatever order an unstable sort happens to leave equal scores in.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], self._chunks[i].chunk_index),
        )

        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in ranked_indices[:top_k]
        ]
