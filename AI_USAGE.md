# AI Usage

AI-assisted development tools were used during this take-home exercise. They supported implementation, debugging, testing, evaluation design, documentation, and discussion of architectural alternatives.

The main development assistants were:

- **ChatGPT** - used for technical discussion, architecture review, debugging, retrieval experiments, evaluation design, and challenging implementation decisions.
- **Claude Code** - used for repository-level implementation, refactoring, test creation, dependency/repository cleanup, and documentation.

The application itself also uses **Ollama Cloud** for grounded answer generation. This is part of the application architecture rather than a development assistant.

AI output was not treated as automatically correct. Suggestions were reviewed against the exercise requirements, tested, and either accepted, modified, or rejected. I also deliberately challenged recommendations when I believed a simpler or different approach was more appropriate.

## How AI Was Used

AI assistance was used to help with:

- breaking the challenge into small implementation stages;
- reviewing the assessment requirements and identifying required deliverables;
- discussing document ingestion, chunking, retrieval, grounding, and provenance;
- implementing and reviewing Python code;
- generating and expanding automated tests;
- investigating retrieval failures;
- comparing retrieval and answerability approaches;
- designing controlled evaluation experiments;
- reviewing Flask API behaviour, validation, error handling, and rate limiting;
- building a repeatable live smoke-test workflow;
- auditing dependencies and repository structure;
- drafting and reviewing technical documentation.

The final implementation decisions were based on observed behaviour and tests rather than accepting generated recommendations without verification.

## Where I Agreed With AI Suggestions

Several AI suggestions were retained after review and testing.

### Offline document ingestion

I agreed with separating document processing from request-time retrieval. PDF and Markdown documents are processed ahead of time, split into structured sections and chunks, and persisted. Runtime requests therefore do not repeatedly parse PDFs.

### Application-controlled provenance

I agreed that source provenance should not be generated freely by the language model. The model receives numbered evidence blocks and returns an `evidence_id`. The application maps that identifier back to the real retrieval result and obtains the source filename, section, and page from trusted application data.

### Abstention over forced answers

I retained the recommendation that the system should abstain when supplied evidence is insufficient instead of encouraging the model to answer from general knowledge. Unsupported questions therefore return no answer and no provenance.

### Simple Flask architecture

I agreed that a small Flask application was sufficient. The challenge did not require a separate frontend framework, database, background task system, or additional services simply to make the solution appear more complex.

## Where I Disagreed or Changed Direction

AI was also useful as something to challenge and test rather than follow directly.

### Retrieval and answerability approaches were tested, not assumed

Several lexical, semantic, reranking, extractive-QA, and answerability ideas were explored. Approaches that added complexity without a convincing measured improvement were not promoted into production.

The final system is intentionally smaller than several architectures considered during development.

### I challenged unnecessary complexity

When suggested directions became more complex than the exercise required, I pushed the design back toward the actual task. The final application does not contain a vector database, separate frontend/API architecture, conversation memory, or multi-agent workflow merely because those technologies could have been added.

### I did not change the system merely to make one broad question pass

A smoke-test question asked for everything about holiday, remote-working, and expense policies at once. The current Top-3/single-primary-evidence design abstained.

Rather than immediately adding multi-source synthesis solely to make this test answerable, I retained the conservative behaviour because it is reasonable for the scope of this exercise.

## Example of an AI-Assisted Design That I Improved

The clearest example was the move from Top-1 to Top-3 retrieval.

The first production design retrieved only the highest-ranked semantic result. Initial evaluation suggested this was a reasonable simple design.

Manual testing later exposed two questions that should have been answerable:

- "What expenses can I claim when travelling for work?"
- "Can I work from home whenever I want?"

The model abstained because the Top-1 chunks did not contain sufficient evidence.

Instead of making the generation prompt more permissive, I inspected the retrieval rankings. In both cases, the correct policy section was ranked second.

I then used AI assistance to design a controlled Top-1 versus Top-3 experiment using the existing frozen evaluation sets before changing production.

Top-3 achieved balanced results of:

- Development: **90.00%**
- Heldout: **90.00%**
- Blind: **90.00%**

In that comparison, Top-3 produced six improvements and one conservative regression, with no observed unsupported-to-incorrect-answer regression.

Prompt-token usage increased by approximately **1.81x**. I explicitly decided that this trade-off was acceptable for this small project because retrieval coverage materially improved.

Only after that experiment was Top-3 promoted into production.

## My Own Input Into the Development Process

The project was not simply built by following AI-generated instructions. I provided the project direction, challenged suggestions, selected experiments, decided when additional complexity was unnecessary, manually tested behaviour, and made the final engineering trade-offs.

### Embedded Latin-text instruction

While reviewing the supplied assessment material, I noticed an unusual embedded instruction associated with Latin text and questioned whether it was intended as a prompt-injection or canary-style test.

I decided that repository and document content should be treated as **untrusted input**, not as instructions that override the actual assessment requirements or development task.

The embedded instruction was rejected rather than followed, and I chose not to reproduce the planted trigger text in generated project files.

This also influenced the application's grounding principle: retrieved document text is evidence for answering the user's question, not trusted operational instructions for the application or development assistant.

### Automating my own manual test process

After the Top-3 change, I manually tested realistic policy and unsupported questions through the application.

I then asked for that process to be automated through the real Flask `/demo` endpoint so that the complete request path could be tested repeatedly instead of manually entering each question in the browser.

The live smoke test produced:

- 8/8 expected-answer tests passed;
- 6/6 expected-abstention tests passed;
- 0 HTTP errors;
- 0 unexpected rate-limit responses.

One deliberately broad question remained observation-only rather than changing its expected result after seeing the output.

## Example of Suggestions I Rejected

A recurring theme was avoiding additional components simply because they are common in larger RAG systems.

More elaborate retrieval, reranking, and answerability/verification approaches were investigated, but approaches whose measured value did not justify their complexity were not retained.

I also did not add a vector database for a corpus containing only a small number of persisted chunks. A local persisted embedding matrix is sufficient for this exercise and keeps the implementation easier to understand and reproduce.

## Verification of AI-Generated Work

AI-generated or AI-assisted changes were checked through automated and manual testing.

At the repository-cleanup stage:

- the active test suite passed **254 tests**;
- the focused Top-3 research suite passed **23 tests**;
- persisted storage artifacts remained unchanged during dependency and documentation cleanup.

The live Flask smoke test separately exercised real grounded-generation behaviour.

AI-generated suggestions were therefore treated as proposals to verify, not as evidence that the implementation was correct.

## Limitations of AI Assistance

AI tools can suggest plausible implementations that are unnecessary, overly complex, or incorrect for a specific repository.

For that reason:

- assessment requirements were treated as the source of truth;
- implementation suggestions were checked against existing code;
- retrieval changes were evaluated before promotion;
- unsupported behaviour was tested explicitly;
- generated provenance was avoided;
- dependency cleanup was based on actual imports and usage;
- documentation was checked against commands and behaviour that exist in the repository.

## Summary

AI materially accelerated development, particularly implementation, debugging, test generation, evaluation design, and exploration of alternatives.

However, the final system was not produced by accepting AI output unchanged. Suggestions were challenged, experiments were used to compare alternatives, unnecessary complexity was rejected, and final architectural decisions were based on observed behaviour.

The most significant example is retrieval: an initially reasonable Top-1 design was shown to fail on realistic questions, investigated, compared experimentally with Top-3, and changed only after the quality-versus-token-cost trade-off was measured and accepted.
