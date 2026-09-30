# Research / Evaluation Archive

## 1. Purpose

This directory contains archived experiments and evaluation tooling used to select the production
architecture of the Small AI-Ready Knowledge System. It is historical evidence, not a supported
application component.

## 2. Production does NOT depend on this directory

`src/knowledge_system/` has zero imports from `research/`. Nothing here is required to run the
CLI, `KnowledgeService`, `LLMKnowledgeService`, or the offline ingestion/embedding-build commands.
Everything the production application needs lives under `src/`, `data/`, `storage/`, and `docs/`.

## 3. Why it is preserved

- **Reproducibility** - every experiment can still be re-run against the persisted production
  corpus (`storage/chunks.json`, `storage/embeddings.npy`).
- **Engineering decision history** - `docs/EVALUATION_HISTORY.md` explains *why* each architecture
  choice was made; this directory is the code that produced that evidence.
- **Interview/reviewer reference** - a reviewer can see the actual comparisons, not just the
  conclusions.
- **Future fallback** - if a production assumption changes (e.g. Ollama Cloud becomes unavailable),
  the rejected non-LLM alternatives are still here, fully implemented and tested, not deleted.

## 4. Final production decisions

| aspect | decision |
|---|---|
| Parsing | Docling structured ingestion (PDF), heading-hierarchy Markdown parser |
| Chunking | structure-aware sections/chunks, recursive fallback for oversized sections |
| Retrieval | Granite embeddings (`ibm-granite/granite-embedding-small-english-r2`) |
| Retrieval depth | Top-1 |
| Primary answer path | Ollama Cloud grounded answerability + answer generation (`LLMKnowledgeService`) |
| Provenance | application-controlled, always from the retrieved chunk, never from the LLM |
| Baseline | MiniLM extractive QA / non-generative path retained for comparison (`KnowledgeService`) |

See `docs/ARCHITECTURE.md` for the full design record.

## 5. Major rejected directions

| Approach | Outcome |
|---|---|
| TF-IDF | Baseline only - inferior to Granite semantic retrieval on every metric |
| Hybrid TF-IDF+Granite (RRF) | Rejected - measured worse than Granite alone |
| Cosine-similarity threshold as answerability | Rejected - retrieval relevance doesn't reliably indicate answer existence |
| NLI verifier (2 experiments) | Rejected - improved rejection but reduced recall / added complexity without net gain |
| FLAN-T5 question-native verifier | Rejected - balanced accuracy below QA-only, near-unconditional acceptance |
| Cross-encoder reranking | Rejected for V1 - improved topic ranking but regressed source accuracy |
| Raw extractive QA spans as final answer | Rejected for presentation - sentence selection materially improved faithfulness |
| Granite sentence selection | Kept as a useful non-LLM presentation baseline (not the final answer mechanism) |
| Ollama Cloud grounded answerability | **Selected as the primary production direction** |

All metrics above are copied from `docs/EVALUATION_HISTORY.md` - see that document for full
confusion matrices, per-split results, and reasoning. Nothing here is invented.

## 6. Directory structure

```
research/
├── README.md          - this file
├── MANIFEST.md         - file-by-file archive map
├── evaluation/          - experimental scripts + their private helper modules (_shared.py, etc.)
├── datasets/            - fixed evaluation question sets (development/heldout/blind, unmodified)
├── tests/                - tests for the code in evaluation/ (not discovered by `pytest` at the repo root)
└── scripts/              - one-off diagnostic scripts predating the evaluation/ framework
```

## 7. Running archived code

Scripts here were written to be run from the **project root**, e.g.:

```bash
PYTHONPATH=src venv/bin/python research/evaluation/evaluate_tfidf.py
```

Most scripts reuse the persisted production corpus (`storage/chunks.json`,
`storage/embeddings.npy`) - they never re-parse documents or rebuild embeddings.

Archived tests can be run explicitly:

```bash
pytest research/tests
```

They are **not** part of the normal `pytest` run at the repo root (`pytest.ini` scopes discovery
to `tests/` only) - the production suite tests the supported application; this suite validates
historical experiments. Some scripts/tests import private sibling modules (e.g. `_shared.py`,
`_tfidf.py`) via `sys.path` manipulation at the top of the file, matching how they were originally
written - this was preserved as-is rather than refactored into a package, since the priority here
is preserving history, not maintaining a second production-quality suite.

## 8. Dependencies

`scikit-learn` in the project's `requirements.txt` is used **only** by `research/evaluation/_tfidf.py`
and `_hybrid.py` (and their tests) - it is not imported anywhere under `src/knowledge_system/` or
`tests/`. It was kept in `requirements.txt` so this archive remains runnable; see
`docs/ARCHITECTURE.md` for the full dependency accounting.
