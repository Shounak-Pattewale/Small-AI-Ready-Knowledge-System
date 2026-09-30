"""Dense sentence-embedding retrieval: DocumentChunk[] -> ranked SearchResult[].

A semantic counterpart to TfidfRetriever, kept API-compatible with it so
the two can be evaluated head-to-head. Knows nothing about how chunks
were produced - it only operates on the chunk list handed to `fit()`/
`load_index()`.
"""

import numpy as np
from sentence_transformers import SentenceTransformer  # official recommended usage for the Granite model card

from knowledge_system.embeddings import load_embeddings
from knowledge_system.models import DocumentChunk, SearchResult

_DEFAULT_MODEL_NAME = "ibm-granite/granite-embedding-small-english-r2"


def searchable_text(chunk: DocumentChunk) -> str:
    """Build the text the embedding model encodes for one chunk, without mutating chunk.text.

    Same contextual-information principle as TfidfRetriever._searchable_text
    (section heading + body when available, body alone otherwise) - keeping
    both retrievers' inputs equivalent is what makes a head-to-head
    evaluation between them fair. Public (not underscore-prefixed) because
    knowledge_system.build_embeddings needs the exact same construction -
    one implementation, no risk of the offline builder and this retriever
    subtly drifting apart.
    """
    if chunk.section:
        return f"{chunk.section}\n\n{chunk.text}"
    return chunk.text


class EmbeddingRetriever:
    """Fit/load once on a chunk collection, then search it repeatedly, using dense sentence embeddings.

    Two ways to populate document embeddings:
        retriever.fit(chunks)                                    # encodes chunks now (tests/experimentation)
        retriever.load_index(chunks, embeddings_path, manifest_path)  # loads a validated, precomputed index

    Either way, search() only ever encodes the query with the model -
    document chunks are never re-encoded during search().
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)  # HF model ID -> normal download/cache flow
        self._chunks: list[DocumentChunk] | None = None
        self._embeddings: np.ndarray | None = None  # normalized, one row per chunk

    def fit(self, chunks: list[DocumentChunk]) -> None:
        """Store the chunks (in the given order) and encode their searchable text into normalized embeddings.

        Recomputes document embeddings every call - useful for tests and
        experimentation, but not what runtime/evaluation should use once a
        persisted index exists (see load_index()).
        """
        self._chunks = list(chunks)
        searchable_texts = [searchable_text(chunk) for chunk in self._chunks]
        self._embeddings = np.asarray(self._model.encode(searchable_texts, normalize_embeddings=True))

    def load_index(self, chunks: list[DocumentChunk], embeddings_path, manifest_path) -> None:
        """Load precomputed, validated document embeddings instead of encoding chunks.

        The Granite model itself is still used for query encoding in
        search() - only document-side encoding is skipped here.
        """
        self._chunks = list(chunks)
        self._embeddings = load_embeddings(self._chunks, self._model_name, embeddings_path, manifest_path)

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        """Return up to `top_k` chunks ranked by cosine similarity to `question`, highest first."""
        if self._chunks is None or self._embeddings is None:
            raise RuntimeError("EmbeddingRetriever.search() called before fit()/load_index()")
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = np.asarray(self._model.encode([question], normalize_embeddings=True))[0]
        # Both document and query embeddings are unit-normalized, so a dot
        # product already IS cosine similarity - no separate division needed.
        scores = self._embeddings @ query_embedding

        # Sort by score descending; ties broken by chunk_index ascending,
        # same deterministic tie-break rule as TfidfRetriever.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], self._chunks[i].chunk_index),
        )

        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in ranked_indices[:top_k]
        ]
