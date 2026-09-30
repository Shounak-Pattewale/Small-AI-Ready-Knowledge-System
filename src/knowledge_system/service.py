"""Production knowledge services: question -> Granite Top-1 -> KnowledgeAnswer.

Two answer paths share the same Granite retrieval:
- KnowledgeService: extractive QA (MiniLM) gated by the frozen QA_THRESHOLD - the original
  non-LLM baseline, kept for comparison (see docs/EVALUATION_HISTORY.md, Experiments QA1/QA2).
- LLMKnowledgeService: Ollama Cloud grounded answerability - the preferred path as of the
  production-integration step (see docs/ARCHITECTURE.md and EVALUATION_HISTORY.md Section 27).

Application/domain logic only - no Flask, no terminal I/O, no HTML, no HTTP
concerns, no threshold tuning, no evaluation metrics, no dataset labels, no
Docling, no ingestion. Granite similarity is used only to select the Top-1
chunk (retrieval); it is never used as an answerability signal in either path.
"""

from pathlib import Path

from knowledge_system.llm import OllamaCloudClient
from knowledge_system.models import KnowledgeAnswer
from knowledge_system.qa import ExtractiveQA
from knowledge_system.retrieval.embedding import EmbeddingRetriever
from knowledge_system.storage import load_chunks

QA_THRESHOLD = -5.7906  # frozen after blind evaluation - do not tune here or anywhere in production code

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CHUNKS_PATH = _PROJECT_ROOT / "storage" / "chunks.json"
_DEFAULT_EMBEDDINGS_PATH = _PROJECT_ROOT / "storage" / "embeddings.npy"
_DEFAULT_MANIFEST_PATH = _PROJECT_ROOT / "storage" / "embeddings_manifest.json"


class KnowledgeService:
    """Load once, then call ask() repeatedly - the retriever and QA model are reused, never reloaded per question.

    Usage:
        service = KnowledgeService.load()
        answer = service.ask("Can I carry over unused holiday?")

    Constructor takes an already-ready retriever/QA pair (dependency
    injection), so tests can pass fakes without loading real models -
    KnowledgeService.load() is the one path that loads the real ones.
    """

    def __init__(self, retriever: EmbeddingRetriever, qa: ExtractiveQA) -> None:
        self._retriever = retriever
        self._qa = qa

    @classmethod
    def load(
        cls,
        chunks_path: Path = _DEFAULT_CHUNKS_PATH,
        embeddings_path: Path = _DEFAULT_EMBEDDINGS_PATH,
        manifest_path: Path = _DEFAULT_MANIFEST_PATH,
    ) -> "KnowledgeService":
        """Load the persisted Granite index and the QA model once. No document re-encoding, no Docling."""
        chunks = load_chunks(chunks_path)
        retriever = EmbeddingRetriever()
        retriever.load_index(chunks, embeddings_path, manifest_path)  # precomputed - chunks are NOT re-encoded
        qa = ExtractiveQA()
        return cls(retriever, qa)

    def ask(self, question: str) -> KnowledgeAnswer:
        """Answer one question, or raise ValueError for empty/whitespace-only input."""
        if not question.strip():
            raise ValueError("question must not be empty")

        results = self._retriever.search(question, top_k=1)  # Top-1 only - the validated architecture
        chunk = results[0].chunk  # only the query was embedded here; chunk embeddings are already persisted

        qa_answer = self._qa.answer(question, chunk.text)

        # Deterministic safeguard, not a second model: a signal that clears
        # the threshold but extracted only whitespace/empty text isn't a
        # real answer - treat it the same as insufficient evidence rather
        # than surfacing a blank answer as answered=True.
        if qa_answer.signal >= QA_THRESHOLD and qa_answer.text.strip():
            return KnowledgeAnswer(
                answered=True,
                answer=qa_answer.text,
                source=chunk.source,
                section=chunk.section,
                page=chunk.page,
            )

        # Insufficient evidence: a passage was retrieved, but the QA model
        # did not find enough support in it. Deliberately return no source/
        # section/page here - the retrieved chunk did NOT answer the
        # question, so citing it as if it were the answer's provenance
        # would be misleading. (The retrieved chunk is discarded, not
        # persisted anywhere - there is currently no internal diagnostic
        # channel that needs it; see service.py docstring / final report.)
        return KnowledgeAnswer(answered=False, answer=None, source=None, section=None, page=None)


_TOP_K = 3  # Granite retrieves this many candidates once per question; validated in the Top-3
# evidence-block experiment (docs/EVALUATION_HISTORY.md Section 29) and promoted from Top-1.


class LLMKnowledgeService:
    """Preferred production answer path: question -> Granite Top-3 -> Ollama Cloud grounded
    answerability -> KnowledgeAnswer (see docs/ARCHITECTURE.md and docs/EVALUATION_HISTORY.md
    Sections 27/29, "KEEP OLLAMA CLOUD DIRECTION" then promoted from Top-1 to Top-3).

    Retrieval reuses the same Granite index as KnowledgeService, requesting the top 3 candidates
    in ONE call instead of KnowledgeService's Top-1. The LLM decides answerable/not-answerable and,
    when answerable, supplies the answer text plus which evidence block (1/2/3) primarily supports
    it (`evidence_id`) - the application then maps that back to the corresponding SearchResult to
    build provenance (source/section/page). Provenance NEVER comes from the LLM: `GroundedLLMResult`
    has no filename/section/page fields at all, so there is nothing for the model to override even
    if it tried.
    """

    def __init__(self, retriever: EmbeddingRetriever, llm_client: OllamaCloudClient) -> None:
        self._retriever = retriever
        self._llm_client = llm_client

    @classmethod
    def load(
        cls,
        chunks_path: Path = _DEFAULT_CHUNKS_PATH,
        embeddings_path: Path = _DEFAULT_EMBEDDINGS_PATH,
        manifest_path: Path = _DEFAULT_MANIFEST_PATH,
        llm_client: OllamaCloudClient | None = None,
    ) -> "LLMKnowledgeService":
        """Construct the Ollama Cloud client FIRST (reads OLLAMA_API_KEY/OLLAMA_MODEL/
        OLLAMA_BASE_URL/OLLAMA_TIMEOUT_SECONDS unless a client is injected) so a missing/invalid
        OLLAMA_API_KEY raises LLMConfigError immediately, before the slower Granite index load."""
        client = llm_client or OllamaCloudClient()
        chunks = load_chunks(chunks_path)
        retriever = EmbeddingRetriever()
        retriever.load_index(chunks, embeddings_path, manifest_path)
        return cls(retriever, client)

    def ask(self, question: str) -> KnowledgeAnswer:
        """Answer one question, or raise ValueError for empty/whitespace-only input.

        Raises LLMProviderError/LLMResponseError (knowledge_system.llm) on a provider/network
        failure or malformed model output - these are never silently converted into an
        abstention, since that would disguise a broken system as "insufficient evidence".
        """
        if not question.strip():
            raise ValueError("question must not be empty")

        results = self._retriever.search(question, top_k=_TOP_K)  # ONE retrieval call for all evidence blocks
        evidences = [r.chunk.text for r in results]

        result = self._llm_client.answer(question, evidences)

        if not result.answerable:
            return KnowledgeAnswer(answered=False, answer=None, source=None, section=None, page=None)

        # evidence_id is already validated by the client as an int in [1, len(evidences)] -
        # deterministic 1-indexed mapping back to the SearchResult that actually supports the answer.
        selected = results[result.evidence_id - 1]
        return KnowledgeAnswer(
            answered=True,
            answer=result.answer,
            source=selected.chunk.source,  # provenance from the SELECTED retrieved chunk, never from the LLM
            section=selected.chunk.section,
            page=selected.chunk.page,
        )
