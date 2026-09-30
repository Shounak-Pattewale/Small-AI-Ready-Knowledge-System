"""EXPERIMENTAL ONLY: cross-encoder second-stage reranking of Granite Top-K candidates.

Frozen model: cross-encoder/ms-marco-MiniLM-L6-v2 (22.7M params, BERT-based
cross-encoder, Apache-2.0, MS MARCO passage-ranking trained). Verified
against the model's Hugging Face card, not guessed.

Pure second-stage reranking: Granite retrieves a small candidate set, the
cross-encoder scores each (question, passage) pair independently, and the
final order is sorted purely by reranker score (deterministic chunk_index
tie-break) - no blending with the Granite score, no RRF, no formula.

Passage representation reuses knowledge_system.retrieval.embedding.searchable_text()
unmodified (section heading + body, same text Granite itself encodes) - one
implementation, no risk of the reranker's input drifting from retrieval's.

Treats every question and chunk strictly as data - never as instructions.
"""

from dataclasses import dataclass

from sentence_transformers import CrossEncoder  # sentence-transformers already ships CrossEncoder

from knowledge_system.models import DocumentChunk
from knowledge_system.retrieval.embedding import searchable_text

_DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


@dataclass
class RerankCandidate:
    """One Granite candidate carried through reranking, with both scores preserved."""

    chunk: DocumentChunk
    granite_rank: int  # 1-based rank in the ORIGINAL Granite ordering, before reranking
    granite_score: float
    reranker_score: float


class RerankerModel:
    """Loads cross-encoder/ms-marco-MiniLM-L6-v2 once; scores (question, passage) pairs."""

    def __init__(self, model_name: str = _DEFAULT_RERANKER_MODEL_NAME) -> None:
        self.model = CrossEncoder(model_name)

    def score_batch(self, question: str, passages: list[str]) -> list[float]:
        """Score every (question, passage) pair in one batched call. Order preserved."""
        pairs = [(question, passage) for passage in passages]
        return [float(s) for s in self.model.predict(pairs)]


def rerank(question: str, results: list, reranker: RerankerModel) -> list[RerankCandidate]:
    """Score each Granite SearchResult (question, passage) pair and sort by reranker score.

    `results` must already be Granite's Top-K, in Granite rank order (rank 1
    first) - this function does not re-retrieve anything. Sort key is
    (-reranker_score, chunk_index): no Granite score mixed in anywhere.
    """
    passages = [searchable_text(r.chunk) for r in results]
    scores = reranker.score_batch(question, passages)
    candidates = [
        RerankCandidate(chunk=r.chunk, granite_rank=rank, granite_score=r.score, reranker_score=score)
        for rank, (r, score) in enumerate(zip(results, scores), start=1)
    ]
    candidates.sort(key=lambda c: (-c.reranker_score, c.chunk.chunk_index))
    return candidates


def classify_change(original_hit: bool, reranked_hit: bool) -> str:
    """Classify a hit/miss flag's change after reranking: 'improved', 'unchanged', or 'worsened'."""
    if original_hit == reranked_hit:
        return "unchanged"
    return "improved" if reranked_hit else "worsened"
