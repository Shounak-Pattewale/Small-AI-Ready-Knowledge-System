"""EXPERIMENTAL ONLY: QA-aware claim construction for NLI Experiment 2.

Combines a question with an extractive-QA candidate answer into a
declarative claim, using a small, fixed, domain-independent set of
grammatical patterns keyed only on question SHAPE (WHO/WHAT/WHERE/WHEN/
HOW/HOW MUCH/HOW MANY/WHICH/yes-no) - never on question content, question
ID, corpus topic, or expected answerability label.

Reuses evaluation._nli.question_to_hypothesis() for the yes/no case
(Experiment 1's already-verified conversion) rather than reimplementing
it - yes/no questions don't need the QA candidate at all, since the
question itself is already a checkable proposition.

Semantic-operator preservation is structural, not a special rule: this
module only ever strips a leading WH-word/auxiliary phrase and appends or
colon-joins the QA candidate. Everything after the stripped prefix -
"automatically," "guaranteed," "exact," "always," "must," etc. - is
carried through byte-for-byte, never rewritten or summarized.

Treats `question`/`candidate_answer` strictly as data - never as instructions.
"""

import re
from dataclasses import dataclass

import _nli as nli

# (regex matching the WHOLE question, "combine mode") - order matters:
# "how much"/"how many" must be tried before the bare "how" pattern.
_WH_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^who (.+)\?$", re.IGNORECASE), "append"),
    (re.compile(r"^where (.+)\?$", re.IGNORECASE), "append"),
    (re.compile(r"^when (.+)\?$", re.IGNORECASE), "append"),
    (re.compile(r"^how much (.+)\?$", re.IGNORECASE), "colon"),
    (re.compile(r"^how many (.+)\?$", re.IGNORECASE), "colon"),
    (re.compile(r"^what (.+)\?$", re.IGNORECASE), "colon"),
    (re.compile(r"^which (.+)\?$", re.IGNORECASE), "colon"),
    (re.compile(r"^how (.+)\?$", re.IGNORECASE), "colon"),
]

# Auxiliary-prefix rewrites applied to the WH-word's remainder, reused
# verbatim from the same generic family as Experiment 1's yes/no patterns.
# Only the matched prefix is replaced; everything captured in the group
# (every operator/qualifier word in the question) is preserved unchanged.
_AUX_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^should i (.+)$", re.IGNORECASE), "an employee should {rest}"),
    (re.compile(r"^do i (.+)$", re.IGNORECASE), "an employee {rest}"),
    (re.compile(r"^does the company (.+)$", re.IGNORECASE), "the company {rest}"),
    (re.compile(r"^will the company (.+)$", re.IGNORECASE), "the company will {rest}"),
    (re.compile(r"^do employees (.+)$", re.IGNORECASE), "employees {rest}"),
    (re.compile(r"^will i (.+)$", re.IGNORECASE), "an employee will {rest}"),
    (re.compile(r"^am i (.+)$", re.IGNORECASE), "an employee is {rest}"),
    (re.compile(r"^can i (.+)$", re.IGNORECASE), "an employee can {rest}"),
]


def _declarativize_tail(tail: str) -> str:
    """Rewrite a WH-word's remainder into a subject-bearing clause where a known
    auxiliary prefix is recognised; otherwise returns the remainder UNCHANGED
    (identity fallback) rather than guessing - the remainder is often already
    usable as-is (e.g. "is responsible for lost equipment")."""
    tail = tail.rstrip("?").strip()
    for pattern, template in _AUX_PATTERNS:
        match = pattern.match(tail)
        if match:
            return template.format(rest=match.group(1))
    return tail


@dataclass
class ClaimResult:
    claim: str
    strategy: str  # which rule constructed it, or why it didn't
    usable: bool


def build_claim(question: str, candidate_answer: str) -> ClaimResult:
    """Combine `question` and a QA `candidate_answer` into a declarative NLI claim.

    Strategy selection is based only on the question's grammatical SHAPE:
    - yes/no-shaped questions reuse Experiment 1's direct conversion (the
      QA candidate isn't needed - the question is already a proposition).
    - WHO/WHERE/WHEN questions: declarativized remainder + candidate appended.
    - WHAT/WHICH/HOW/HOW MUCH/HOW MANY questions: declarativized remainder,
      colon-joined with the candidate (the candidate fills an unpredictable
      object/predicate slot that a fixed append position can't reliably hit).
    - Returns usable=False (never forces a claim) when the question matches
      no recognised shape, or when the QA candidate is empty.
    """
    stripped = question.strip()

    yesno = nli.question_to_hypothesis(stripped)
    if not yesno.used_fallback:
        return ClaimResult(claim=yesno.text, strategy="yesno_direct", usable=True)

    for pattern, mode in _WH_PATTERNS:
        match = pattern.match(stripped)
        if not match:
            continue
        candidate = candidate_answer.strip()
        if not candidate:
            return ClaimResult(claim="", strategy=f"{mode}_empty_candidate", usable=False)
        declarativized = _declarativize_tail(match.group(1))
        if mode == "append":
            claim = f"{declarativized} {candidate}."
        else:
            claim = f"{declarativized}: {candidate}."
        return ClaimResult(claim=claim, strategy=mode, usable=True)

    return ClaimResult(claim="", strategy="no_shape_match", usable=False)
