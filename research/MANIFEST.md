# Archive Manifest

Compact map of every archived file: purpose and production status. See
`docs/EVALUATION_HISTORY.md` for the full narrative each row summarizes.

## research/evaluation/ - shared helper modules (private, underscore-prefixed)

| file | purpose | production status |
|---|---|---|
| `_shared.py` | Source/topic scoring helpers shared by the retrieval evaluators | Rejected direction's support code |
| `_answerability.py` | Generic threshold-sweep/confusion-matrix statistics (cosine answerability) | Rejected direction |
| `_tfidf.py` | `TfidfRetriever` - moved out of `src/knowledge_system/retrieval/` | Rejected retrieval strategy |
| `_hybrid.py` | `HybridRetriever` (RRF) - moved out of `src/knowledge_system/retrieval/` | Rejected retrieval strategy |
| `_extractive_qa.py` | Experimental mirror of production `qa.py`'s span-extraction logic | Superseded by production `src/knowledge_system/qa.py` |
| `_nli.py` | NLI Experiment 1 infrastructure (question-to-hypothesis conversion) | Rejected direction |
| `_claim.py` | NLI Experiment 2 infrastructure (QA-aware claim construction) | Rejected direction |
| `_verifier.py` | FLAN-T5 question-native verifier | Rejected direction |
| `_reranker.py` | Cross-encoder reranker (`ms-marco-MiniLM-L6-v2`) | Rejected for V1 |
| `_sentence_selection.py` | Granite sentence-level answer selection | Non-LLM presentation baseline, not adopted as final mechanism |
| `_ollama_grounded_qa.py` | Experimental Ollama Cloud grounded-QA client (local-daemon transport) | Superseded by production `src/knowledge_system/llm.py` |

## research/evaluation/ - evaluation runner scripts

| file | purpose | production status |
|---|---|---|
| `evaluate_tfidf.py` | TF-IDF retrieval evaluation | Rejected |
| `evaluate_embeddings.py` | Granite retrieval evaluation | Underlying model KEPT (production retrieval); script itself archived |
| `evaluate_hybrid.py` | Hybrid RRF retrieval evaluation | Rejected |
| `analyze_answerability.py` | Cosine-similarity answerability diagnostic | Rejected |
| `evaluate_extractive_qa.py` | Development-only extractive QA experiment | Superseded (production `qa.py` frozen after this) |
| `evaluate_blind_answerability.py` | Final blind evaluation of frozen cosine vs. QA rules | Historical validation of the (now-baseline) QA gate |
| `evaluate_nli_answerability.py` | NLI Experiment 1 | Rejected |
| `evaluate_nli_v2_answerability.py` | NLI Experiment 2 (QA-aware claims) | Rejected |
| `evaluate_verifier.py` | FLAN-T5 verifier experiment | Rejected |
| `evaluate_reranker.py` | Cross-encoder reranker experiment | Rejected for V1 |
| `evaluate_evidence_vs_qa.py` | Direct evidence vs. extractive QA span comparison | Motivated the sentence-selection and LLM-answer-presentation work |
| `evaluate_sentence_answers.py` | Sentence-level semantic answer selection experiment | Non-LLM presentation baseline |
| `evaluate_combined_pipeline.py` | QA gate + Granite sentence answer combined pipeline | Superseded by the Ollama Cloud direction |
| `evaluate_ollama_cloud.py` | Ollama Cloud grounded-answerability formal evaluation (local daemon) | **Selected direction** - production reimplemented separately with direct-cloud API auth (`src/knowledge_system/llm.py`) |

## research/datasets/

| file | purpose | production status |
|---|---|---|
| `questions.json` | 65-question development/heldout evaluation set (fixed, unmodified) | Reference dataset, not loaded by production |
| `fresh_blind_test_questions.json` | 30-question sealed blind-origin test set (fixed, unmodified) | Reference dataset, not loaded by production |

## research/scripts/

| file | purpose | production status |
|---|---|---|
| `inspect_docling_pdf.py` | Read-only diagnostic of Docling's structured PDF output | Predates `evaluation/`; informed the (kept) Docling parsing decision |
| `inspect_granite_embeddings.py` | Read-only sanity check of Granite embedding similarity | Predates `evaluation/`; informed the (kept) Granite retrieval decision |

## research/tests/

One test file per corresponding `research/evaluation/` module/script above (`test_answerability.py`,
`test_blind_answerability.py`, `test_claim.py`, `test_combined_pipeline.py`,
`test_evaluate_no_docling.py`, `test_evidence_vs_qa.py`, `test_extractive_qa.py`, `test_hybrid.py`,
`test_nli.py`, `test_ollama_cloud_evaluation.py`, `test_reranker.py`, `test_sentence_answers.py`,
`test_tfidf.py`, `test_verifier.py`), plus a copy of `tests/conftest.py` (shared fixtures such as
`kb_dir`, `mock_pdf`, `mock_embedding_model` - duplicated rather than shared across `tests/` and
`research/tests/` to keep the two suites independently runnable).

## What replaced what, in production

| archived component | production replacement |
|---|---|
| `_extractive_qa.py` | `src/knowledge_system/qa.py` (`ExtractiveQA`) |
| `_ollama_grounded_qa.py` (local Ollama daemon, `localhost:11434`) | `src/knowledge_system/llm.py` (`OllamaCloudClient`, direct `https://ollama.com/api/chat`, API-key auth) |
| `evaluate_combined_pipeline.py` (experimental gate+sentence combination) | `src/knowledge_system/service.py` (`LLMKnowledgeService`) |
| `_tfidf.py` / `_hybrid.py` (`src/knowledge_system/retrieval/tfidf.py`/`hybrid.py`) | `src/knowledge_system/retrieval/embedding.py` (`EmbeddingRetriever`, Granite-only) |
