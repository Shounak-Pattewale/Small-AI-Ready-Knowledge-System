# Small AI-Ready Knowledge System

An internal-company knowledge assistant that answers policy and support questions
from a small document set, using semantic retrieval and grounded LLM generation.
Built as a NALITS AI Solution Engineer take-home submission.

## What It Does

- Answers employee questions about company policy, benefits, and IT support.
- Retrieves relevant document sections with semantic search (no keyword matching).
- Generates a grounded answer from retrieved evidence only, via Ollama Cloud.
- Returns the source document, section, and page alongside every answer.
- Abstains (`answered: false`) when the retrieved evidence does not support an answer,
  rather than falling back on the model's general knowledge.

## Demo / Interface

A minimal Flask web UI (`GET /`) lets you type a question and see the answer with its
source. A terminal CLI (`cli.py`) offers the same question/answer loop without a browser,
in either LLM or baseline mode (see [Running the CLI](#running-the-cli)).

## Knowledge Base

The original assessment repository did not include any knowledge-base documents, so a
synthetic internal-company corpus was created, modeled on the domains the challenge
describes. **These documents are synthetic placeholders, not real NALITS material.**

PDFs (`data/`):
- `employee_handbook.pdf`
- `holiday_leave_policy.pdf`
- `flexible_remote_working.pdf`
- `career_development.pdf`

Markdown (`data/`):
- `it_support_security.md`
- `expenses_business_travel.md`
- `learning_training.md`

## Architecture

```
Documents
    ↓
Offline ingestion + structure-aware chunking
    ↓
Persisted chunks (storage/chunks.json)
    ↓
Granite embeddings (storage/embeddings.npy)
    ↓
Top-3 semantic retrieval
    ↓
Ollama Cloud grounded generation
    ↓
Evidence-ID provenance mapping
    ↓
Flask / CLI response
```

Ingestion (Docling for PDFs, a heading-aware parser for Markdown) and chunking run
offline, producing persisted chunks and embeddings. The Flask app and CLI load these
persisted artifacts at startup and never parse documents or recompute embeddings on a
request path.

Chunking is structure-aware: section → paragraph → sentence → word fallback, targeting
~400 words per chunk with a 60-word overlap kept within a section. The current corpus
produces 48 sections and 48 chunks.

## Retrieval and Grounding

Retrieval uses `ibm-granite/granite-embedding-small-english-r2` via SentenceTransformers,
with normalized embeddings and Top-3 cosine-similarity search. Similarity score is used
only to rank candidates - it is not exposed as, or treated as, a confidence score.

The three retrieved chunks are passed to Ollama Cloud as three numbered evidence blocks.
The model returns structured JSON: `answerable`, `answer`, and `evidence_id` (the number
of the evidence block it used, or `null` if none support an answer). The application -
not the model - maps `evidence_id` back to the corresponding retrieved chunk and reads
its source, section, and page from there. The LLM never supplies provenance directly.

## Why Top-3?

Production initially used Top-1 retrieval. Manual testing found two real questions where
the correct section ranked second, not first (a travel-expenses question and a
remote-work question). Rather than changing production on the spot, a frozen benchmark
compared Top-1 against Top-3 across development, heldout, and blind question sets.
Top-3 scored 90% balanced accuracy on all three splits, fixed both rank-2 failures with
6 net improvements and 1 conservative regression, at a measured ~1.81x increase in
prompt tokens. That tradeoff was accepted for a demo of this size. Full detail is in
[docs/EVALUATION_HISTORY.md](docs/EVALUATION_HISTORY.md).

## Project Structure

```
app.py                      Flask app (GET /, GET /health, POST /demo)
cli.py                      Terminal CLI (LLM or baseline mode)
ratelimit.py                In-memory per-IP rate limiter
data/                       Synthetic knowledge-base source documents
storage/                    Persisted chunks + Granite embeddings (tracked, required at runtime)
src/knowledge_system/       Ingestion, chunking, embeddings, retrieval, LLM client, service layer
templates/, static/         Flask web UI
tests/                      Production test suite (pytest)
manual_tests/                Live smoke-test script against a running server (not run by pytest)
research/                   Archived experiments/evaluation that informed the production design
docs/                       Architecture and evaluation history
requirements.txt            Production dependencies
research/requirements.txt   Extra dependency needed only to run archived research code
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OLLAMA_API_KEY` (required - the app refuses to start LLM mode
without it). `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, and `OLLAMA_TIMEOUT_SECONDS` are optional
and default to `gemma4`, `https://ollama.com`, and `30` seconds respectively.

## Running the Web App

```bash
python3 app.py
```

Open `http://127.0.0.1:5000` in a browser.

## Running the CLI

```bash
python3 cli.py                  # LLM mode (Ollama Cloud grounded answerability, default)
python3 cli.py --mode baseline  # MiniLM extractive QA, no LLM call
```

## Deployment

Deployment target (not yet live/verified): `https://knowledge-system-demo.shounakdev.tech`

```
Cloudflare
    ↓
Traefik (existing shared stack)
    ↓
Docker (vps-network)
    ↓
Gunicorn :8000
    ↓
Flask
```

The application is packaged as a Docker image (`Dockerfile`) run behind an existing
Traefik reverse proxy via Docker Compose (`docker-compose.yml`), on the same shared
external `vps-network` used by other services on the VPS. Traefik handles HTTPS/Let's
Encrypt and routing; Cloudflare sits in front of the VPS. This project's Compose file
does not add another reverse proxy, TLS terminator, or Cloudflare configuration.

- The container runs Gunicorn (`app:create_app()`), not Flask's development server,
  bound to `0.0.0.0:8000` inside the container. Port 8000 is internal only - it is not
  published to the VPS host; Traefik reaches it over `vps-network`.
- `OLLAMA_API_KEY`, `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, and `OLLAMA_TIMEOUT_SECONDS` are
  runtime environment variables supplied via a deployment-only `.env` file read by
  Compose - never baked into the image, the Dockerfile, or the Compose file itself.
- The Granite embedding model is downloaded once at image build time, not at container
  startup, so there is no first-request download latency and no cache volume to manage.
- `GET /health` is used as the container healthcheck; it does not call retrieval or
  Ollama.

```bash
docker compose up -d --build
```

## API

**`GET /health`** - liveness check, no model calls.

**`POST /demo`** - the only LLM-backed route. JSON body, 16 KB max request size,
`question` must be a non-blank string of at most 1000 characters. Rate-limited to
10 requests/minute/IP.

Request:
```json
{"question": "How many days of annual leave do I get?"}
```

Answered response:
```json
{"answered": true, "answer": "Employees accrue 25 days of annual leave per year.", "source": "holiday_leave_policy.pdf", "section": "2. Annual Leave Entitlement", "page": 1}
```

Abstention response:
```json
{"answered": false, "answer": null, "source": null, "section": null, "page": null}
```

The public API intentionally does not expose `evidence_id` or any similarity score.

## Testing

```bash
pytest
```

Runs the production suite in `tests/` (pytest.ini scopes discovery there). Archived
research tests are not part of this run and can be invoked explicitly, e.g.:

```bash
pytest research/tests/test_ollama_top3.py
```

A live smoke test against a running server exists at `manual_tests/smoke_test_demo.py`.
It is a manual diagnostic script, not a unit test - it requires the Flask app to be
running and a valid `OLLAMA_API_KEY`, and makes real Ollama Cloud calls:

```bash
python3 manual_tests/smoke_test_demo.py
```

## Evaluation

The frozen Top-3 benchmark measured 90% balanced accuracy on development, heldout, and
blind question splits (see [Why Top-3?](#why-top-3) and
[docs/EVALUATION_HISTORY.md](docs/EVALUATION_HISTORY.md)). A subsequent live smoke test
against the running app passed 8/8 expected-answer questions and 6/6 expected-abstention
questions, with 0 HTTP errors and 0 unexpected 429s. This is a small synthetic evaluation
intended to validate this demo's design choices - it is not evidence of accuracy at
production scale.

## Design Decisions

- Offline ingestion and chunking, not runtime PDF parsing, so requests never pay
  Docling's cost.
- Persisted embeddings, loaded once at startup, rather than recomputed per request.
- Top-3 retrieval over Top-1, accepted after a measured benchmark, not a guess.
- Provenance (source/section/page) is application-controlled, read from the retrieved
  chunk, never generated by the LLM.
- Abstention is preferred over an unsupported answer.
- A single Flask process, no database - appropriate at this corpus size, not a
  long-term scaling decision.

## Assumptions and Limitations

- The knowledge base is synthetic and small (7 documents, 48 chunks).
- Each answer cites a single source chunk; there is no multi-source citation.
- Broad questions spanning several policies may correctly abstain rather than partially
  answer.
- The rate limiter is in-memory and process-local, not shared across workers/instances.
- No authentication or authorization.
- No conversation memory - each question is answered independently.
- No database.
- The system depends on external Ollama Cloud availability.
- Retrieval similarity is not a calibrated confidence score.
- Current evaluation is limited and synthetic, not production-scale validation.

## First Five Changes for ~1,000 Employees

The current system is a working demo, not something to deploy unchanged at this scale.

1. **Authentication/SSO and document-level authorization** - so answers only draw on
   documents a given employee is entitled to see.
2. **Durable, shared rate limiting and observability** - replace the in-memory limiter
   with a shared store (e.g. Redis), and add request/latency/error metrics and logging
   suitable for on-call use.
3. **A real document ingestion and versioning pipeline** - automated re-ingestion when
   source documents change, with change tracking instead of a manually rebuilt corpus.
4. **Scalable retrieval storage** - a vector database/search index instead of a single
   in-process NumPy matrix, so the corpus can grow and retrieval can scale horizontally.
5. **Production hardening** - larger and ongoing evaluation, security review of the
   generation path, and a real deployment pipeline (the current app runs via
   `python3 app.py` only).

## Security / Safety

- `OLLAMA_API_KEY` stays server-side; the client and CLI never see or log it.
- `.env` is git-ignored; `.env.example` documents required/optional variables without
  real values.
- Request size (16 KB) and question length (1000 characters) are enforced before any
  model call.
- Answers are grounded in retrieved document text; the model is not asked to use
  outside knowledge.
- Retrieved document text is treated as evidence to quote from, not as instructions to
  the model or the application.
- Error responses are generic; raw provider errors/stack traces are never returned to
  the client.

## AI Assistance

AI tools were used during development. Details and decisions will be documented in
`AI_USAGE.md`.

## Future Improvements

- Multi-source citation when an answer draws on more than one section.
- More automated ingestion (change detection, incremental re-embedding).
- Shared/distributed rate limiting.
- Authentication and per-user access control.
- Structured observability (metrics, tracing) around the retrieval and generation calls.
