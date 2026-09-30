# Architecture - Small AI-Ready Knowledge System

Consolidated design, kept current as the system evolves. Sections 1-17 record the V1 design
(Granite + extractive QA, no LLM), frozen after the retrieval and answerability experiments in
[`EVALUATION_HISTORY.md`](EVALUATION_HISTORY.md) and [`CORPUS_HISTORY.md`](CORPUS_HISTORY.md).
**Update (production integration step):** following the Ollama Cloud grounded-answerability
experiment (`EVALUATION_HISTORY.md` Section 27, decision **KEEP OLLAMA CLOUD DIRECTION**), an
LLM-backed answer path (`LLMKnowledgeService`) has been added to production as the **preferred**
answer path; the original non-LLM path (`KnowledgeService`) is retained as a baseline. See
Sections 18-23 for what changed and why. This document is the design record; the history files
are the evidence.

## 1. Problem

Employees need to ask natural-language questions about internal company policy and get a useful
answer, with a citable source, or a clear "I don't know" - without the system inventing information
that isn't actually in the knowledge base. Generative AI is explicitly not required.

## 2. Design goals

- Retrieve the right passage for a question.
- Extract a real answer from that passage, not a fabricated one.
- Cite the source (and section/page, when available).
- Decline to answer honestly, rather than guess, when the evidence is insufficient.
- Stay small enough to explain end-to-end in an interview - no component whose presence can't be
  justified by a measured result.

## 3. Final architecture

```
                     OFFLINE (run explicitly, ahead of time)
 data/*.md, *.pdf
        |
        v
   loaders.py (Markdown / Docling)
        |
        v
   chunking.py (structure-aware, recursive fallback)
        |
        v
   storage/chunks.json  <-- persisted, git-tracked
        |
        v
   build_embeddings.py (Granite encodes every chunk)
        |
        v
   storage/embeddings.npy + embeddings_manifest.json  <-- persisted, git-tracked


                     RUNTIME - two answer paths, same Granite index, different retrieval depth

 question
        |
        +---------------------------------------------+
        |                                               |
        v                                               v
  PREFERRED: LLMKnowledgeService              BASELINE: KnowledgeService
        |                                               |
        v                                               v
  Granite Top-3 (one call, 3 chunks)         Granite Top-1 (one call, 1 chunk)
        |                                               |
        v                                               v
  Ollama Cloud (gemma4, api.ollama.com)      Extractive QA (MiniLM SQuAD2.0)
  3 numbered evidence blocks ->              signal >= QA_THRESHOLD (-5.7906)
  {answerable, answer, evidence_id}          AND extracted text non-blank
        |                                               |
   answerable? --no--> abstain                 yes? -----+-----> no?
        |                                      |                |
       yes                                     v                v
        |                              answer + source +   "I couldn't find enough
        v                              section + page      information..."
   map evidence_id -> that SearchResult
        |
        v
   answer + source + section + page
   (provenance ALWAYS from the SELECTED
    retrieved chunk, never from the LLM)
```

Each path does exactly one retrieval call - the preferred path just asks for 3 candidates instead
of 1. The baseline is one QA call and one deterministic threshold comparison (no second retrieval
pass, no second local model). The LLM path replaces the QA call with one Ollama Cloud request
carrying all 3 evidence blocks; see Section 19 for its exact request/response shape. Neither path
does a second retrieval pass, reranking, or multi-document context - Top-3 is a wider single
retrieval call, not an additional one.

## 4. Offline ingestion pipeline

`knowledge_system.ingest.run_ingestion()` (`python -m knowledge_system.ingest`):

1. `loaders.load_directory()` - Markdown via a heading-hierarchy parser, PDF via Docling, grouped by
   level-1 heading.
2. `chunking.chunk_sections()` - structure-aware chunking (400 words, 60-word overlap, recursive
   section/paragraph/sentence/word fallback for oversized sections).
3. `storage.save_chunks()` - atomic write to `storage/chunks.json`.

`knowledge_system.build_embeddings.build_embeddings()` (`python -m knowledge_system.build_embeddings`):

4. Loads the persisted chunks (never re-parses documents), encodes each with Granite, and
   `embeddings.save_embeddings()` atomically writes `storage/embeddings.npy` plus a manifest
   (model ID, dimension, chunk count, a SHA-256 fingerprint of the chunk collection).

Both steps are explicit, separate commands. Nothing at runtime triggers either of them.

## 5. Runtime query pipeline

`KnowledgeService.load()` (called once, e.g. at process/CLI startup):

1. `storage.load_chunks()` - load the persisted chunk list.
2. `EmbeddingRetriever().load_index()` - load the persisted embedding matrix, **validated** against
   the current chunks (model ID, schema version, chunk count, fingerprint, shape, finite values).
   Document chunks are never re-encoded here.
3. `ExtractiveQA()` - load the MiniLM QA model/tokenizer.

`KnowledgeService.ask(question)` (called once per question, retriever/QA reused every time):

1. Reject empty/whitespace-only input (`ValueError`).
2. Encode the question with Granite; retrieve the Top-1 chunk by cosine similarity.
3. Run extractive QA on `(question, chunk.text)`.
4. If `signal >= QA_THRESHOLD` **and** the extracted text is non-blank, return the answer with its
   source/section/page. Otherwise return `answered=False` with no provenance.

## 6. Why Docling

Docling gives structure-aware PDF parsing (heading levels, not just raw page text), which is what
makes section-aware chunking and citation possible for PDF sources at all. It runs only in the
offline ingestion path - never a runtime dependency. See `EVALUATION_HISTORY.md` P1-P3 for the
parsing-strategy experiments that led here (page-based and every-heading grouping were both
rejected in favor of grouping by level-1 heading).

## 7. Why structure-aware chunks

Chunking by document structure (not fixed-size windows) keeps each chunk's `section` heading path
accurate, which is what lets the system cite a specific section, not just a filename, and is what
`topic_matches()` evaluation depends on. The recursive fallback (paragraph -> sentence -> word)
only engages for sections that exceed 400 words, so most chunks are whole, coherent sections.

## 8. Why Granite embeddings

`ibm-granite/granite-embedding-small-english-r2` was evaluated against TF-IDF and a TF-IDF+Granite
RRF hybrid on the same question set. Granite alone won on every metric (Source Top-1 97.50% vs.
TF-IDF's weaker baseline; the hybrid was worse than Granite alone, not better - see
`EVALUATION_HISTORY.md` Sections 5, 7). It is also the model already validated across every later
experiment (QA, NLI, verifier, reranker all built on top of Granite Top-1/Top-3 retrieval), so
switching it now would invalidate that entire evidence chain for no measured benefit.

## 9. Why extractive QA

`deepset/minilm-uncased-squad2` gives near-total answerable recall (Section 16-17: 96.67% dev,
100% heldout, 100% blind-origin) with a small, well-understood scoring mechanism (SQuAD2.0
null-vs-best-span logits) rather than a black-box generative step. Its known weakness - extracting
a plausible-looking span from a related-but-insufficient passage - was investigated three separate
times (cosine similarity, NLI x2, a FLAN-T5 verifier, then cross-encoder reranking as a retrieval-side
fix) and none of those additions improved on QA-only enough to justify their cost. QA remains the
answer-extraction mechanism because nothing tested beat it, not because its limitation was solved.

*(Update: extractive QA remains the non-LLM baseline path (`KnowledgeService`), still exactly as
described above. The Ollama Cloud LLM path (`LLMKnowledgeService`, Section 18) has since been
evaluated and adopted as the preferred production path - see Section 19.)*

## 10. Abstention strategy

**The frozen QA threshold (`QA_THRESHOLD = -5.7906`) remains the only semantic abstention signal.**
Every alternative or supplement tested - cosine similarity, NLI (twice), a dedicated
question-native verifier model, cross-encoder reranking - was rejected on measured evidence (see
Section 14's decision table). There is no better learned signal available without introducing a new
model, which this task explicitly rules out.

On top of that frozen threshold, V1 adds exactly one deterministic, non-semantic safeguard: an
extracted answer that is empty or whitespace-only is treated as insufficient evidence even if its
signal happens to clear the threshold (`service.py`, `qa_answer.text.strip()`). This isn't a second
intelligence layer - it's a sanity check on the one component we already trust, guarding a genuine
edge case (a token span that maps to non-content characters) with no new model, no new threshold,
and no corpus-specific rule. No other deterministic rule was added: an empty retrieved chunk already
drives the QA signal to `-inf` (see `qa.py`'s null-vs-best-span search), which is always below
threshold, so a separate "empty chunk" check would be redundant dead code. A genuine model/runtime
failure (e.g. a crash inside Granite or QA) is **not** caught and converted into an abstention
message - it propagates as an exception, so a broken system is never disguised as "I couldn't find
enough information" (see Section 13).

**Honest limitation, stated plainly:** the QA signal is a raw logit difference, not a calibrated
probability, and does not perfectly distinguish "answer present" from "related but insufficient
evidence." Three separate attempts to build a better signal (Sections 19, 20, 21) did not succeed
well enough to replace it. V1 ships with this known, documented gap rather than a worse-performing
"fix."

*(Update: this remains true for the baseline `KnowledgeService` path exactly as written. The
preferred `LLMKnowledgeService` path uses a different, LLM-based answerability decision instead of
`QA_THRESHOLD` - see Section 18/19 - measured to substantially improve on this exact limitation.)*

## 11. Source provenance

An answered question returns the chunk's `source` (filename), `section` (heading path, if any,
e.g. `"IT Support & Security > 3. Lost or stolen devices"`), and `page` (PDF page, if any). An
abstained question returns none of these - the retrieved chunk did not actually answer the
question, so citing it as if it were the answer's provenance would be misleading (`service.py`
already implements this deliberately).

## 12. Persistence

`storage/chunks.json` (chunk text + metadata) and `storage/embeddings.npy` +
`storage/embeddings_manifest.json` (Granite vectors + a fingerprint of the chunk collection they
were built from) are both git-tracked, atomically written (`tempfile` + `os.replace()`), and
validated on load - a stale or mismatched index fails loudly rather than being silently used or
silently rebuilt.

## 13. Error handling

| condition | behavior |
|---|---|
| empty/whitespace question | `ValueError`, no model call made |
| missing `chunks.json` | `FileNotFoundError` naming the fix (`python -m knowledge_system.ingest`) |
| missing `embeddings.npy` / manifest | `FileNotFoundError` naming the fix (`python -m knowledge_system.build_embeddings`) |
| manifest schema/model/chunk-count/fingerprint mismatch | `ValueError` naming exactly what's stale |
| embedding matrix shape mismatch or non-finite values | `ValueError` |
| model load failure (Granite/QA) | propagates as-is (e.g. from `sentence_transformers`/`transformers`) - not caught |
| retrieval/QA inference failure at query time | propagates as-is - not caught, not disguised as abstention |
| extracted answer is empty/whitespace | treated as `answered=False` (Section 10 safeguard) |

Startup/configuration failures and per-question infrastructure failures are always distinguishable
from "the system searched and found insufficient evidence" - the former are real exceptions, the
latter is the only path that returns `answered=False`.

## 14. Approaches tested and rejected

| Component | Decision | Reason |
|---|---|---|
| Markdown loader | **KEEP** | Simple, correct, heading-hierarchy aware; no issues found |
| Docling (PDF parsing) | **KEEP** | Only structure-aware PDF parser evaluated; grouping by level-1 heading beat page-based and every-heading grouping |
| Structure-aware chunking | **KEEP** | Preserves section citation accuracy; recursive fallback handles oversized sections without a second strategy |
| TF-IDF | **REJECT** | Weaker than Granite semantic retrieval on every metric |
| Granite embeddings | **KEEP** | Best retrieval result of anything tested (97.50% Source Top-1); validated across every later experiment |
| Hybrid TF-IDF+Granite (RRF) | **REJECT** | Measured worse than Granite alone - added complexity, no benefit |
| Extractive QA (MiniLM) | **KEEP** | Near-total answerable recall; smallest, most explainable answer-extraction mechanism tested |
| Cosine-similarity answerability | **REJECT** | Retrieval relevance does not reliably indicate answer existence; threshold unstable on blind data |
| NLI (Experiment 1) | **REJECT** | Improved rejection but reduced recall; brittle question-to-hypothesis conversion |
| NLI (Experiment 2, QA-aware claims) | **REJECT** | More complexity than Experiment 1, did not outperform it, zero coverage on the motivating manual questions |
| Question-native verifier (FLAN-T5) | **REJECT** | Balanced accuracy (48.33% dev, 35.00% heldout, 50.00% blind) below QA-only; near-unconditional acceptance |
| Cross-encoder reranking | **REJECT for V1** | Improved topic ranking but regressed source accuracy at the tested primary depth; fixed the motivating laptop case but broke others |
| Vector database | **NOT NEEDED FOR V1** | 48 chunks fits comfortably in memory; a flat NumPy matrix + cosine similarity is exact, fast, and simpler to reason about than an ANN index at this corpus size |
| Ollama Cloud LLM (gemma4, grounded answerability) | **KEEP - production, preferred path** | Not needed for V1's own bar, but explicitly evaluated afterward (109 real calls, 0 failures) and materially improved balanced accuracy on every split, including blind (73.33% -> 83.33%) - see `EVALUATION_HISTORY.md` Section 27 and Sections 18-19 below. Cloud dependency; baseline retained. |

None of the rejected experiments were engineering failures - each was implemented correctly,
evaluated honestly, and did not clear the bar its own decision criteria set. The evidence for each
is preserved in `research/evaluation/` and `EVALUATION_HISTORY.md`, not deleted.

## 15. Known limitations

1. Granite can retrieve a topically-related section instead of the single best section (Topic
   Top-1 is 67.50%, well below Source Top-1's 97.50%) - the system usually finds the right
   *document* but not always the right *section* on the first try.
2. Extractive QA can return a plausible-looking span from a passage that doesn't actually contain
   the requested information (negation, missing amount/deadline/exception) - this is the one
   weakness three separate experiments (NLI x2, verifier) tried and failed to fix well enough to
   adopt.
3. `QA_THRESHOLD` is an empirically useful cutoff on a raw logit difference, not a calibrated
   probability - it should never be presented to a user as a confidence percentage.
4. The corpus is synthetic (Corpus V2, `CORPUS_HISTORY.md`) - the assessment did not supply real
   company documents.
5. The corpus is tiny (7 documents, 48 chunks) - retrieval-quality conclusions at this scale may
   not directly generalize to a much larger corpus.
6. There is no authorization/access-control model - every question can retrieve from every
   document.
7. There is no document versioning or change-detection beyond the chunk fingerprint that
   invalidates a stale embedding index at load time - there's no process yet for *noticing* a
   source document changed and re-running ingestion.
8. Models (Granite + MiniLM QA) run locally and are loaded fresh at process startup; startup has
   non-trivial cost (multiple seconds) before the first question can be answered.

## 16. Why the system intentionally remains simple

Every additional component considered (hybrid retrieval, NLI, a verifier model, reranking) was
actually built and measured, not just discussed - and none improved the outcome enough to justify
its cost. The resulting architecture (Granite Top-1 -> extractive QA -> one frozen threshold) is
the smallest design the evidence actually supports, not a default. Simplicity here is a measured
conclusion, not a shortcut.

## 17. First production-scale improvements (~1,000 employees)

Corpus size, not employee count, drives most infrastructure decisions here - 1,000 employees asking
questions against a 48-chunk corpus is a concurrency/ops problem, not a retrieval-scale problem. In
rough priority order:

1. **Real company documents and a repeatable ingestion pipeline.** Everything here was validated
   against a synthetic corpus; the first real task is re-running ingestion/evaluation against
   actual policy documents and re-checking every number in this document.
2. **Evaluation with real employee questions.** The current 65-question set (plus the 30-question
   blind set) is hand-written; real query logs would reveal failure modes this corpus can't.
3. **Authentication and document-level permissions.** Not every employee should be able to query
   every document (e.g. HR-only or leadership-only policies) - this is a correctness/compliance
   gap, not a performance one, and matters before scale does.
4. **Incremental ingestion.** Right now, ingestion re-processes the whole `data/` directory;
   at real document-set sizes, only re-parsing changed files (tracked via the existing chunk
   fingerprint mechanism) avoids unnecessary Docling/Granite work.
5. **Document versioning / change detection.** Know when a source document changed and
   automatically trigger re-ingestion + re-embedding, instead of relying on someone remembering to
   run the two offline commands.
6. **Observability.** Log every question, retrieved source/section, QA signal, and
   answered/abstained outcome (internally - never surfaced to the user) so abstention and failure
   rates are visible, not just correctness on a static eval set.
7. **Model serving / deployment.** Keep Granite and QA loaded once in a long-running process (already
   true for `KnowledgeService`) behind whatever web layer comes next, rather than reloading per
   request; consider a model server if request volume grows enough to need multiple worker
   processes.
8. **Caching.** Identical or near-identical questions (e.g. "how many holiday days can I carry
   over") are common in a 1,000-person company - caching Granite/QA results for repeated exact
   questions is cheap and safe, since answers don't depend on who's asking.
9. **Auditability.** Retain a record of what was answered, from what source, so incorrect or
   misleading answers can be traced back and the underlying document/chunk fixed.
10. **A vector database - only if the corpus grows enough to need one.** At real company scale (a
    few hundred to a few thousand documents) a flat matrix may still be fine; this should be
    revisited against measured corpus size and query latency, not assumed up front just because
    employee count is 1,000.

Deliberately not listed as "first": Redis, Celery, a general-purpose task queue, or a generative
LLM - none of these are implied by going from 1 evaluator to 1,000 employees; they'd be solving
problems this system doesn't have yet.

## 18. Production LLM integration (Ollama Cloud)

Following the experiment in `EVALUATION_HISTORY.md` Section 27 (**KEEP OLLAMA CLOUD DIRECTION** -
balanced accuracy improved on every split, blind-origin 73.33% -> 83.33%, zero hallucinations
observed across 109 real calls), a production Ollama Cloud client and service were added:

- `src/knowledge_system/llm.py` - `OllamaCloudClient`, `OllamaConfig`, `GroundedLLMResult`, and a
  small typed error hierarchy (`LLMConfigError`, `LLMProviderError`, `LLMResponseError`). Calls
  Ollama Cloud **directly** over HTTPS (`https://ollama.com/api/chat` by default) - no local
  Ollama daemon, no `ollama signin`, no dependency on `~/.ollama/`. Uses only the Python standard
  library (`urllib`) for the HTTP request; no SDK was added (see Section 20).
- `src/knowledge_system/service.py` - `LLMKnowledgeService`, alongside the existing
  `KnowledgeService` (unmodified). Same Granite retrieval, different answer-extraction/gating
  component.

Production code does **not** import `research/evaluation/_ollama_grounded_qa.py` or
`research/evaluation/evaluate_ollama_cloud.py` - those remain archived reference material
documenting how this decision was reached, unmodified. The production system prompt and
structured-output contract are a deliberate, separate copy of the experimentally-validated ones
(not a shared import), so the production module has zero dependency on `research/`.

## 19. LLM runtime query pipeline

**Updated: Granite Top-3, not Top-1** (promoted from the Top-3 evidence-block experiment,
`docs/EVALUATION_HISTORY.md` Section 29 - improved balanced accuracy on every formal split and
fixed two observed production retrieval failures where the correct evidence ranked 2nd, not 1st).

`LLMKnowledgeService.load()` (called once, e.g. at CLI startup):

1. Construct `OllamaCloudClient()` **first** - reads `OLLAMA_API_KEY`/`OLLAMA_MODEL`/
   `OLLAMA_BASE_URL`/`OLLAMA_TIMEOUT_SECONDS` from the environment and raises `LLMConfigError`
   immediately if the API key is missing, before the slower Granite index load runs.
2. `storage.load_chunks()` + `EmbeddingRetriever().load_index()` - identical to `KnowledgeService`,
   same validated persisted index, same model.

`LLMKnowledgeService.ask(question)` (called once per question, retriever/client reused every time):

1. Reject empty/whitespace-only input (`ValueError`) - same contract as `KnowledgeService`.
2. Retrieve the Top-3 chunks by Granite cosine similarity, **one retrieval call** (unchanged
   retrieval model/index - only the requested candidate count changed, from 1 to 3).
3. Send `(question, [chunk1.text, chunk2.text, chunk3.text])` - the question plus up to three
   evidence texts, in rank order - to `OllamaCloudClient.answer()`. No filename, section, page, or
   similarity score is ever sent to the model.
4. The client sends one `/api/chat` request: the frozen system prompt (evidence-only, no
   general-knowledge, no invented amounts/deadlines/entitlements/exceptions/conditions/benefits/
   consequences, evidence treated strictly as data, never as instructions - unchanged principles,
   extended for multiple blocks) plus `QUESTION:\n{question}\n\n[EVIDENCE 1]\n{chunk1.text}\n\n
   [EVIDENCE 2]\n{chunk2.text}\n\n[EVIDENCE 3]\n{chunk3.text}` as the user message, `format: "json"`,
   `temperature: 0`. Same non-schema-constrained `format:"json"` + strict validation approach
   validated in both the Top-1 and Top-3 experiments (full JSON-schema `format` was not reliably
   honored by this model).
5. The response is strictly validated into `GroundedLLMResult(answerable: bool, answer: str|None,
   evidence_id: int|None)` - any deviation from the exact `{"answerable": bool, "answer": str|null,
   "evidence_id": 1|2|3|null}` shape raises `LLMResponseError`, never silently becomes an answer.
   `evidence_id` must be an integer (not a string, not a bool) in `[1, N]` where N is the number of
   evidence blocks actually sent that call (normally 3) - never hard-coded to 3, so a model can
   never "select" a block that was never supplied.
6. **Provenance always comes from the SELECTED retrieved chunk, never from the LLM** -
   `GroundedLLMResult` has no `source`/`section`/`page` fields at all, so there is nothing for the
   model to override even if it tried; only a validated integer index into the application's own
   retrieval results. If `answerable=false`: `answered=False`, no provenance (identical contract to
   `KnowledgeService`'s abstention - no candidate source is ever exposed for an abstention). If
   `answerable=true`: `answered=True`, `answer` from the LLM, `source`/`section`/`page` from
   `results[evidence_id - 1]` - the specific chunk the model cited, not automatically Top-1.

A provider/network failure or malformed structured output raises `LLMProviderError` /
`LLMResponseError` - these propagate out of `ask()` uncaught, exactly like `KnowledgeService`
never disguises a model crash as "insufficient evidence" (Section 10/13). The CLI is the first
layer that catches them, and only to print a generic, safe message (Section 21). Flask's `/demo`
route (unchanged by this promotion) never exposes `evidence_id`, retrieval rank, or similarity -
only the same `answered`/`answer`/`source`/`section`/`page` contract it already had.

**Cost:** three evidence blocks instead of one increased measured prompt-token usage by
approximately 1.81x (see Section 29) - explicitly accepted for this project given the balanced-
accuracy improvement on every evaluated split. No chunk truncation, summarization, or conditional
Top-K was introduced to offset this; the validated experimental behavior was promoted as-is.

## 20. Configuration

| variable | default | purpose |
|---|---|---|
| `OLLAMA_API_KEY` | *(none - required)* | Bearer token for Ollama Cloud. `LLMConfigError` if missing/blank. Never logged, printed, or included in any exception message or `repr()`. |
| `OLLAMA_MODEL` | `gemma4` | Ollama Cloud model identifier. The direct-cloud API name (no local-daemon `:cloud` suffix). |
| `OLLAMA_BASE_URL` | `https://ollama.com` | Ollama Cloud API base URL; the client calls `{base_url}/api/chat`. |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Per-request timeout (seconds). Validated: must parse as a positive number. |

Configuration is centralized in `OllamaConfig.from_env()` (`llm.py`) - the base URL/endpoint is
never hard-coded elsewhere. Read directly from `os.environ`; no `.env`-parsing dependency was
added (`python-dotenv` etc.) - for local development, export the variables in the shell; Docker
Compose will inject them later. `.env.example` documents the four variables with placeholder
values only (`OLLAMA_API_KEY=` is blank). `.gitignore` ignores `.env`/`.env.*` but explicitly
un-ignores `.env.example` (`!.env.example`) so the placeholder file itself is trackable.

**Retry policy:** at most one retry, only for a transient, clearly-retryable failure (HTTP
429/5xx, network error, timeout), with a small fixed delay (no exponential backoff). Authentication
failures (401/403) and malformed model output are never retried - retrying an auth failure wastes
a request cycle for an error that won't resolve itself, and retrying malformed output risks
masking a real prompt/model problem behind an apparently-successful second attempt.

## 21. CLI

```
python cli.py                  # LLM mode (Ollama Cloud) - the default
python cli.py --mode llm       # same, explicit
python cli.py --mode baseline  # MiniLM extractive QA (the non-LLM baseline)
```

The CLI knows nothing about HTTP, the API key, Ollama's JSON format, the prompt, or retrieval
internals - it calls `service.ask(question)` and formats a `KnowledgeAnswer`, identically for
either mode (`format_answer()` is mode-agnostic, since both services return the same contract).
This is the same service boundary Flask will reuse later (Section 23).

Startup: for LLM mode, `LLMKnowledgeService.load()` validates `OLLAMA_API_KEY` before the Granite
index loads; a missing key prints a concise message (`OLLAMA_API_KEY is not configured`) and exits
cleanly - no stack trace, no key ever printed. Granite/QA/the Ollama client are all loaded exactly
once at startup, never per question, in both modes. `exit`/`quit`/Ctrl+C/EOF and empty-input
handling are unchanged from the pre-LLM CLI.

**Output:**

| outcome | CLI text |
|---|---|
| answered | `Answer:` / `Source:` / `Section:` / `Page:` (metadata lines omitted when `None`) |
| abstained | `I couldn't find enough information in the knowledge base to answer that.` |
| provider failure (LLM mode only) | `The language model service is temporarily unavailable.` - no provider diagnostics, no stack trace, no credential |

## 22. Security boundary

- The API key lives only in `OLLAMA_API_KEY` (process environment) and inside `OllamaConfig`,
  whose `__repr__` explicitly redacts it (`api_key=<redacted>`) so an accidental `print(config)` or
  log line cannot leak it. It appears in exactly one place at runtime: the `Authorization: Bearer
  <key>` HTTP header of the outgoing request.
- No exception type in `llm.py` (`LLMConfigError`, `LLMProviderError`, `LLMResponseError`) ever
  includes the key or the raw provider response body in its message.
- Retrieved evidence is treated strictly as data: the system prompt explicitly instructs the model
  to ignore any instructions that appear inside `EVIDENCE`, and the application never executes or
  follows anything a document says.
- `.env` and `.env.*` are git-ignored; `.env.example` (placeholders only, no real value) is
  explicitly un-ignored so it can be committed as documentation.

## 23. Future Flask reuse

`LLMKnowledgeService` and `KnowledgeService` are the exact service boundary a future Flask app will
call - `service.ask(question) -> KnowledgeAnswer`, with no HTTP/CLI/terminal concerns baked into
either class. Flask will construct one service instance at app startup (same "load once" pattern
already used by the CLI) and call `.ask()` per request; no changes to `service.py` or `llm.py` are
anticipated for that step. Flask and Docker are explicitly out of scope for this step (see
`EVALUATION_HISTORY.md` for the evaluation trail and this document's own history above for why
each component was chosen).

## 24. Web application layer (Flask)

The AI/backend architecture (Sections 1-23) is unchanged by this step. Flask is a thin layer on
top of it - no retrieval, no LLM calls, no prompt logic live in `app.py`.

```
Browser
   |
   v
Flask (app.py, create_app())
   |
   +-- GET /        renders templates/index.html - no retrieval, no LLM call
   +-- GET /health  {"status": "ok"}, HTTP 200 - no retrieval, no LLM call
   +-- POST /demo   request validation
                        |
                        v
                    10/min/IP rate limiter (ratelimit.RateLimiter)
                        |  (rejected request -> HTTP 429, ZERO cloud calls)
                        v
                    LLMKnowledgeService.ask(question)  <- the SAME production service, unchanged
                        |                                  by Flask (its internal retrieval depth
                        v                                  changed separately - see Section 19)
                    Granite Top-3 -> Ollama Cloud -> KnowledgeAnswer
                        |
                        v
                    JSON response (application-controlled provenance,
                                    evidence_id never exposed)
```

**Public routes:** exactly `GET /`, `GET /health`, `POST /demo` (plus Flask's own `/static/...`
handling). No `/api/ask`, `/chat`, `/compare`, `/baseline`, `/debug`, `/admin`, or `/metrics` -
verified by a dedicated test that enumerates the app's URL map.

**GET /health does not call the LLM.** It returns a static `{"status": "ok"}` with no Granite
retrieval, no Ollama call, no document parsing, no embedding rebuild - cheap enough for future
Docker health checks or Traefik/uptime monitoring, and it exposes no API keys, environment values,
filesystem paths, or model internals.

**Rate limiting happens before the paid cloud call.** `ratelimit.RateLimiter` is a small,
dependency-free, thread-safe, fixed-60-second-window limiter (max 10 accepted requests per IP per
window), keyed by `request.remote_addr` - Werkzeug's own address resolution, not a blindly-trusted
`X-Forwarded-For` header. No Redis, no Flask-Limiter, no external service. It sweeps stale
per-IP entries on every call so memory stays bounded as distinct IPs come and go. An injectable
clock makes it fully testable without real sleeps. **Production proxy/IP-trust configuration
(e.g. `ProxyFix` trusted-hop count once Cloudflare -> Traefik -> Gunicorn is in place) is
explicitly deferred to the Docker/deployment step** - this app-level limiter is one layer;
Cloudflare will provide an outer layer later.

**Request validation** (all before the rate-limit check and before `.ask()` is ever called):
`Content-Type: application/json`, body must be a JSON object, `question` must be a non-empty
string, `question` max length 1000 characters. Any violation returns `HTTP 400` with a generic
safe message - never a stack trace, never silently truncated. `MAX_CONTENT_LENGTH` is set to 16KB
(`app.config`), comfortably above a 1000-character question but small enough to reject an
oversized/unrelated payload before Flask even finishes parsing it (`HTTP 413`).

**Provenance stays application-controlled.** The JSON response's `source`/`section`/`page` come
directly from the `KnowledgeAnswer` `LLMKnowledgeService.ask()` already returns - the same
guarantee documented in Section 19 (the LLM never supplies provenance) - `app.py` only serializes
that object, it never lets the model influence which fields are shown.

**Error mapping:** an `LLMProviderError`/`LLMResponseError` (provider/network failure or malformed
structured output) -> `HTTP 503`, generic message `"The knowledge service is temporarily
unavailable."`. Any other unexpected exception -> `HTTP 500`, generic message `"Something went
wrong."`. Neither response includes the provider's raw body, the API key, the Authorization
header, or a Python traceback; server-side logs record a short, non-secret diagnostic line only.

**Service initialization happens once per process**, inside `create_app()` - `LLMKnowledgeService`
(and therefore Granite + the Ollama Cloud client) is loaded exactly once at app startup and reused
across every request, the same "load once" discipline the CLI already follows.
`create_app(service=None, rate_limiter=None)` accepts dependency injection so tests never load a
real model or require `OLLAMA_API_KEY`.

**Frontend:** Jinja (`templates/index.html`) + plain CSS (`static/css/style.css`) + vanilla JS
(`static/js/app.js`) - no frontend framework, no build step. The browser only ever talks to Flask;
it never calls Ollama Cloud directly and never sees `OLLAMA_API_KEY`. Each submitted question is
one independent `POST /demo` request - no chat history is sent to the backend, avoiding token
growth and an extra prompt-injection surface. Model/user text is inserted via `textContent`, never
`innerHTML`, and no Markdown-to-HTML rendering is performed (plain text + CSS `white-space:
pre-wrap` for line breaks) - both are deliberate choices against an injection surface, not an
oversight.
