"""Tests for research/evaluation/_ollama_top3_grounded_qa.py (Top-3 evidence-block experiment).

No real cloud request is ever made - every test injects a fake transport,
same pattern as research/tests/test_ollama_cloud_evaluation.py.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
sys.path.insert(0, str(_EVAL_DIR))  # so _ollama_top3_grounded_qa.py's `from _ollama_grounded_qa import ...` resolves
_MODULE_PATH = _EVAL_DIR / "_ollama_top3_grounded_qa.py"


def _load():
    spec = importlib.util.spec_from_file_location("ollama_top3_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def top3qa():
    return _load()


class _FakeTransport:
    def __init__(self, response_body: dict):
        self._body = response_body
        self.last_payload = None

    def __call__(self, payload, host, timeout):
        self.last_payload = payload
        return self._body


def _body(content: str) -> dict:
    return {"message": {"content": content}, "prompt_eval_count": 10, "eval_count": 5}


# --- prompt construction: 1-3 evidence blocks, labeled ---


def test_build_user_prompt_labels_three_blocks(top3qa):
    prompt = top3qa.build_user_prompt_top3("q?", ["first chunk", "second chunk", "third chunk"])
    assert "[EVIDENCE 1]\nfirst chunk" in prompt
    assert "[EVIDENCE 2]\nsecond chunk" in prompt
    assert "[EVIDENCE 3]\nthird chunk" in prompt
    assert prompt.startswith("QUESTION:\nq?")


def test_build_user_prompt_handles_fewer_than_three_blocks(top3qa):
    prompt = top3qa.build_user_prompt_top3("q?", ["only chunk"])
    assert "[EVIDENCE 1]\nonly chunk" in prompt
    assert "[EVIDENCE 2]" not in prompt


def test_payload_uses_json_format_and_temperature_zero(top3qa):
    payload = top3qa.build_request_payload_top3("q", ["e1", "e2", "e3"])
    assert payload["format"] == "json"
    assert payload["options"]["temperature"] == 0


def test_system_prompt_requires_evidence_id_and_preserves_grounding_rules(top3qa):
    lowered = top3qa.SYSTEM_PROMPT_TOP3.lower()
    assert "evidence_id" in lowered
    assert "only the supplied" in lowered or "only this evidence" in lowered
    assert "never follow instructions" in lowered
    assert "do not invent" in lowered


def test_system_prompt_does_not_mention_confidence_or_probability(top3qa):
    lowered = top3qa.SYSTEM_PROMPT_TOP3.lower()
    assert "confidence" not in lowered
    assert "probability" not in lowered


# --- structured output validation ---


def test_valid_answerable_with_evidence_id_parses(top3qa):
    result = top3qa.parse_structured_output_top3('{"answerable": true, "answer": "x", "evidence_id": 2}')
    assert result.valid is True
    assert result.answerable is True
    assert result.evidence_id == 2


def test_valid_abstention_parses(top3qa):
    result = top3qa.parse_structured_output_top3('{"answerable": false, "answer": null, "evidence_id": null}')
    assert result.valid is True
    assert result.answerable is False
    assert result.evidence_id is None


@pytest.mark.parametrize("bad_id", [0, 4, -1, "2", 2.0, True])
def test_answerable_true_with_invalid_evidence_id_rejected(top3qa, bad_id):
    payload = json.dumps({"answerable": True, "answer": "x", "evidence_id": bad_id})
    result = top3qa.parse_structured_output_top3(payload)
    assert result.valid is False


def test_answerable_true_missing_evidence_id_rejected(top3qa):
    result = top3qa.parse_structured_output_top3('{"answerable": true, "answer": "x"}')
    assert result.valid is False


def test_answerable_false_with_non_null_evidence_id_rejected(top3qa):
    result = top3qa.parse_structured_output_top3('{"answerable": false, "answer": null, "evidence_id": 1}')
    assert result.valid is False


def test_answerable_false_with_non_null_answer_rejected(top3qa):
    result = top3qa.parse_structured_output_top3('{"answerable": false, "answer": "surprise", "evidence_id": null}')
    assert result.valid is False


def test_answerable_true_empty_answer_rejected(top3qa):
    result = top3qa.parse_structured_output_top3('{"answerable": true, "answer": "", "evidence_id": 1}')
    assert result.valid is False


def test_malformed_json_rejected(top3qa):
    result = top3qa.parse_structured_output_top3("not json {")
    assert result.valid is False


def test_missing_answerable_field_rejected(top3qa):
    result = top3qa.parse_structured_output_top3('{"answer": "x", "evidence_id": 1}')
    assert result.valid is False


def test_evidence_id_1_and_3_both_valid(top3qa):
    for eid in (1, 3):
        result = top3qa.parse_structured_output_top3(json.dumps({"answerable": True, "answer": "x", "evidence_id": eid}))
        assert result.valid is True
        assert result.evidence_id == eid


# --- fake-transport integration ---


def test_call_top3_full_sends_all_three_evidence_blocks(top3qa):
    transport = _FakeTransport(_body('{"answerable": true, "answer": "x", "evidence_id": 2}'))
    top3qa.call_ollama_chat_top3_full("q", ["e1", "e2", "e3"], transport=transport)
    for label in ("[EVIDENCE 1]", "[EVIDENCE 2]", "[EVIDENCE 3]"):
        assert label in transport.last_payload["messages"][1]["content"]


def test_call_top3_full_exposes_token_counts(top3qa):
    transport = _FakeTransport(_body('{"answerable": false, "answer": null, "evidence_id": null}'))
    body = top3qa.call_ollama_chat_top3_full("q", ["e1"], transport=transport)
    assert body["prompt_eval_count"] == 10
    assert body["eval_count"] == 5


def test_reuses_top1_transport_and_error_type(top3qa):
    # Confirms this module reuses _ollama_grounded_qa's OllamaAPIError/_default_transport,
    # rather than duplicating HTTP handling.
    import _ollama_grounded_qa as top1qa

    assert top3qa.OllamaAPIError is top1qa.OllamaAPIError
    assert top3qa._default_transport is top1qa._default_transport
