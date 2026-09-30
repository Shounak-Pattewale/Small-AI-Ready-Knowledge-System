# Corpus History

This file records the history of the synthetic knowledge-base documents in `data/`, separately from `docs/EVALUATION_HISTORY.md`, which records retrieval/answerability experiment results. Corpus changes and experiment results are two different kinds of history - a corpus rewrite is a data-quality change, not a model or threshold experiment.

## Corpus V1

The assessment repository supplied no knowledge-base documents, so a synthetic internal knowledge base was created from scratch: four PDFs (`employee_handbook.pdf`, `holiday_leave_policy.pdf`, `flexible_remote_working.pdf`, `career_development.pdf`) and three Markdown files (`it_support_security.md`, `expenses_business_travel.md`, `learning_training.md`), covering the intended policy domains - IT support/security, expenses/business travel, learning/training, holiday/leave, flexible/remote working, and career development.

Corpus V1 had broad topic coverage and supported the full evaluation question set (development, heldout, and blind splits) reasonably well at the source level - Granite semantic retrieval reached 96% source Top-1 accuracy against it (see `docs/EVALUATION_HISTORY.md`, Section 5).

However, later manual CLI testing exposed a structural weakness: every major policy section had been generated with the same fixed subsection template - "Standard process," "Exceptions and cross-policy considerations," "Example scenarios" - and those subsections' body text was heavily generic across unrelated topics, differing mainly in a single topic-name phrase. Concretely:

> Question: "I lost my work laptop on the train, what should I do?"
> Question: "My work laptop was stolen. Who should I report it to?"
>
> Granite retrieval ranked **"3. Laptop faults"** above **"4. Lost or stolen devices"** for both questions (similarity 0.8287 vs 0.8171, and 0.8844 vs 0.8825 - gaps of 0.0116 and 0.0019). Diagnostic inspection of the persisted chunks showed both sections' body text shared the near-identical sentence *"Employees should use the designated internal process and provide enough information for the request, incident or decision to be assessed accurately"* - differing only in the opening topic-description clause.

This was diagnosed as a synthetic-corpus generation artifact, not a retrieval model defect: Granite was being asked to distinguish two policies whose actual stored text was templated and nearly identical, which is a fundamentally harder (and in places impossible) discrimination task regardless of embedding quality. Full diagnostic detail (Top-5 rankings, exact chunk indexes, QA behaviour on both chunks) is preserved in the session record of that diagnostic task; it is not duplicated here since this file tracks corpus versions, not experiment results.

## Corpus V2

**Rewritten:** date not recorded (session-local; no systematic corpus-versioning timestamps in this project)

**Why:** to remove the templated-boilerplate weakness identified above, by rewriting each policy section around what that specific topic actually needs to communicate, rather than filling a fixed generic subsection template.

**Design principles applied:**
- Each major section has genuinely distinct vocabulary, procedures, conditions, and examples - not a paraphrase of a shared template.
- No fixed "Standard process / Exceptions / Example scenarios" subsection structure under every heading; subsections are used only where a topic genuinely needs one (e.g. "Becoming a mentor" under Mentoring).
- Concrete, specific language ("Up to five days of unused annual leave may be carried into the next holiday year...") in place of generic language ("Employees should follow the appropriate process").
- No arbitrary invented numbers added merely to make QA easier - concrete values were included only where a realistic policy would naturally state them (e.g. the five-day carry-over limit, the 1 January-31 December holiday year), and were checked against both evaluation files' negative questions before being written down.
- PDFs remain realistic continuous documents: natural heading hierarchy (numbered level-1 topics, one level-2 subsection example), no manual page breaks, multiple sections flowing together per page, generated with the same library (ReportLab) Corpus V1 used, confirmed via PDF producer metadata.
- Markdown files use `#`/`##` only where a subsection is genuinely warranted, not as a fixed template.

**All seven documents were rewritten:**
- `data/it_support_security.md` - Service desk, Laptop faults, Lost or stolen devices, Phishing, Security incidents, Equipment away from the office. Laptop faults and Lost or stolen devices (the two sections implicated in the diagnosed failure) now share no boilerplate sentence at all - see the duplication audit below.
- `data/expenses_business_travel.md` - Submitting claims, Rail travel, Other travel and subsistence, Training courses and conferences.
- `data/learning_training.md` - Internal learning, External courses, Professional certifications, Conferences, Records.
- `data/career_development.pdf` - Career conversations, Internal vacancies, Mentoring (with a "Becoming a mentor" subsection), Promotion, Development records.
- `data/employee_handbook.pdf` - Welcome and scope, Employment principles, Working hours and flexible working, Pay and benefits, Conduct and respect at work, Health/safety/security, Raising concerns, Leaving the company. Rewritten as genuinely high-level guidance with concise cross-references to the specialist policies, rather than duplicating their detailed procedures.
- `data/flexible_remote_working.pdf` - Hybrid and home working, Working from another UK location, Flexible start and finish times, Company equipment away from the office, Overseas remote work.
- `data/holiday_leave_policy.pdf` - Annual leave entitlement, Requesting and approving leave, Carry-over, Leaving employment, Sickness during leave.

**Evaluation labels preserved:** every `answerable: true` question in both `evaluation/questions.json` and `evaluation/fresh_blind_test_questions.json` was checked against the rewritten text for its `expected_source`/`expected_topic` pair before and after writing; every `answerable: false` question's `negative_type` and wording were checked to confirm the rewrite did not introduce the specific unsupported fact it asks about (exact amounts, limits, guarantees, deadlines, entitlements, exceptions). See the session's answerability audit for the full per-question check; no evaluation file was modified.

**Boilerplate removed:** the fixed "Standard process / Exceptions and cross-policy considerations / Example scenarios" subsection template is gone from every document. A textual duplication scan (exact-sentence matching, and a word-overlap near-duplicate check across paragraph pairs from different documents) found zero substantive duplicated policy content - the only repeated text across documents is the one-line "Fictional demonstration policy" disclaimer, which is front-matter, not policy content.

**Corpus size:** Corpus V2 is intentionally more compact than Corpus V1 - the four PDFs now run one page each (except the handbook, at two), and the Markdown files are similarly leaner, reflecting removed boilerplate rather than removed topic coverage.

**Ingestion status at initial authoring:** `storage/chunks.json`, `storage/embeddings.npy`, and `storage/embeddings_manifest.json` were NOT yet regenerated against Corpus V2 at the time this section was first written. Docling had not been run against it. See the "Corpus V2 ingested" update below for what happened next - this paragraph is left as the original historical record of that intermediate state, not edited to match the update.

## Corpus V2 ingested

- **Ingestion date:** date not recorded
- Corpus V2 was ingested using the existing, unmodified production pipeline (`python -m knowledge_system.ingest`, then `python -m knowledge_system.build_embeddings`) - no new parser, no chunking/embedding configuration changes.

**Resulting counts:**

| metric | value |
|---|---|
| sections loaded | 48 |
| chunks generated | 48 |
| chunks per source | career_development.pdf 6, employee_handbook.pdf 9, expenses_business_travel.md 6, flexible_remote_working.pdf 6, holiday_leave_policy.pdf 6, it_support_security.md 8, learning_training.md 7 |
| chunk word stats | min 10, max 188, avg 66.29, median 63.5 |

**New persisted artifacts:**

| artifact | value |
|---|---|
| `storage/embeddings.npy` shape | (48, 384), float32, normalized |
| `storage/embeddings_manifest.json` chunk_fingerprint | `8caee2c0fd7ba5d9f7db39ed33c919dfdd49273449f450604f64aefac297d963` |

**Confirmation:** Corpus V2 artifacts were successfully generated and now back `storage/chunks.json`/`storage/embeddings.npy`/`storage/embeddings_manifest.json` in place of the Corpus V1 artifacts referenced above. The Laptop-faults-vs-Lost-or-stolen-devices sections were re-inspected post-ingestion and confirmed to still share zero boilerplate sentences (chunk_index 36 vs 37 in `it_support_security.md`).

Full retrieval/QA regression results for Corpus V2 (TF-IDF, Granite, frozen-threshold QA on development/heldout/blind-origin sets, Corpus V1-vs-V2 comparison, production smoke test) are recorded in `docs/EVALUATION_HISTORY.md`, Section 18 - not duplicated here, since this file tracks corpus versions and ingestion facts, not experiment results. The reason Corpus V2 was created (the diagnosed templated-boilerplate defect) remains recorded above and is not erased by this update.
