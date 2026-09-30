"""Tests for manual_tests/smoke_test_demo.py - the live /demo smoke-test script.

No real HTTP call and no Ollama call is ever made here - urllib.request.urlopen is
monkeypatched to return fake HTTP responses. Deliberately kept OUTSIDE tests/ (pytest.ini
scopes normal `pytest` discovery to tests/ only) - this is a diagnostic script's own
tests, not part of the production suite. Run explicitly:

    pytest manual_tests/test_smoke_test_demo.py
"""

import importlib.util
import io
import json
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent / "smoke_test_demo.py"


def _load():
    spec = importlib.util.spec_from_file_location("smoke_test_demo_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoke():
    return _load()


# --- classify_response: ANSWER / ABSTAIN / ERROR, never looks for evidence_id ---


def test_classify_answer(smoke):
    body = {"answered": True, "answer": "x", "source": "a.md", "section": "s", "page": None}
    assert smoke.classify_response(200, body, None) == "ANSWER"


def test_classify_abstain(smoke):
    body = {"answered": False, "answer": None, "source": None, "section": None, "page": None}
    assert smoke.classify_response(200, body, None) == "ABSTAIN"


def test_classify_ignores_evidence_id_if_present(smoke):
    # The public /demo contract should never include evidence_id, but even if a server
    # accidentally leaked it, classification must not depend on it in any way.
    body = {"answered": True, "answer": "x", "source": "a.md", "section": "s", "page": None, "evidence_id": 2}
    assert smoke.classify_response(200, body, None) == "ANSWER"


def test_classify_malformed_body_is_error(smoke):
    assert smoke.classify_response(200, {"unexpected": "shape"}, None) == "ERROR"


def test_classify_non_200_is_error(smoke):
    assert smoke.classify_response(400, {"error": "bad request"}, None) == "ERROR"


def test_classify_network_error_is_error(smoke):
    assert smoke.classify_response(None, None, "network error: refused") == "ERROR"


def test_classify_non_dict_body_is_error(smoke):
    assert smoke.classify_response(200, ["not", "a", "dict"], None) == "ERROR"


# --- evaluate(): PASS/FAIL/OBSERVE/HTTP_ERROR per the documented rules ---


def test_evaluate_answer_pass(smoke):
    entry = {"id": 1, "question": "q", "expected": "ANSWER"}
    body = {"answered": True, "answer": "Yes.", "source": "a.md", "section": "s", "page": None}
    verdict, _ = smoke.evaluate(entry, 200, body, None)
    assert verdict == "PASS"


def test_evaluate_answer_fail_when_actually_abstained(smoke):
    entry = {"id": 1, "question": "q", "expected": "ANSWER"}
    body = {"answered": False, "answer": None, "source": None, "section": None, "page": None}
    verdict, reason = smoke.evaluate(entry, 200, body, None)
    assert verdict == "FAIL"
    assert "ABSTAIN" in reason


def test_evaluate_answer_fail_when_source_null(smoke):
    entry = {"id": 1, "question": "q", "expected": "ANSWER"}
    body = {"answered": True, "answer": "Yes.", "source": None, "section": "s", "page": None}
    verdict, reason = smoke.evaluate(entry, 200, body, None)
    assert verdict == "FAIL"
    assert "source" in reason


def test_evaluate_answer_fail_when_section_null(smoke):
    entry = {"id": 1, "question": "q", "expected": "ANSWER"}
    body = {"answered": True, "answer": "Yes.", "source": "a.md", "section": None, "page": None}
    verdict, reason = smoke.evaluate(entry, 200, body, None)
    assert verdict == "FAIL"
    assert "section" in reason


def test_evaluate_answer_fail_when_answer_empty(smoke):
    entry = {"id": 1, "question": "q", "expected": "ANSWER"}
    body = {"answered": True, "answer": "   ", "source": "a.md", "section": "s", "page": None}
    verdict, reason = smoke.evaluate(entry, 200, body, None)
    assert verdict == "FAIL"


def test_evaluate_answer_pass_with_null_page(smoke):
    entry = {"id": 1, "question": "q", "expected": "ANSWER"}
    body = {"answered": True, "answer": "Yes.", "source": "a.md", "section": "s", "page": None}
    verdict, _ = smoke.evaluate(entry, 200, body, None)
    assert verdict == "PASS"  # page may legitimately be null


def test_evaluate_abstain_pass(smoke):
    entry = {"id": 9, "question": "q", "expected": "ABSTAIN"}
    body = {"answered": False, "answer": None, "source": None, "section": None, "page": None}
    verdict, _ = smoke.evaluate(entry, 200, body, None)
    assert verdict == "PASS"


def test_evaluate_abstain_fail_when_actually_answered(smoke):
    entry = {"id": 9, "question": "q", "expected": "ABSTAIN"}
    body = {"answered": True, "answer": "Yes.", "source": "a.md", "section": "s", "page": None}
    verdict, reason = smoke.evaluate(entry, 200, body, None)
    assert verdict == "FAIL"
    assert "ANSWER" in reason


def test_evaluate_abstain_fail_when_provenance_leaks():
    smoke = _load()
    entry = {"id": 9, "question": "q", "expected": "ABSTAIN"}
    # answered=False but a source leaked anyway - must be flagged, not silently passed.
    body = {"answered": False, "answer": None, "source": "a.md", "section": None, "page": None}
    verdict, reason = smoke.evaluate(entry, 200, body, None)
    assert verdict == "FAIL"
    assert "leaked" in reason


def test_evaluate_observe_never_pass_or_fail(smoke):
    entry = {"id": 15, "question": "q", "expected": "OBSERVE"}
    body_answered = {"answered": True, "answer": "Yes.", "source": "a.md", "section": "s", "page": None}
    body_abstained = {"answered": False, "answer": None, "source": None, "section": None, "page": None}

    verdict_a, _ = smoke.evaluate(entry, 200, body_answered, None)
    verdict_b, _ = smoke.evaluate(entry, 200, body_abstained, None)

    assert verdict_a == "OBSERVE"
    assert verdict_b == "OBSERVE"


def test_evaluate_http_error_status(smoke):
    entry = {"id": 1, "question": "q", "expected": "ANSWER"}
    verdict, reason = smoke.evaluate(entry, 503, {"error": "unavailable"}, None)
    assert verdict == "HTTP_ERROR"


def test_evaluate_network_error(smoke):
    entry = {"id": 1, "question": "q", "expected": "ANSWER"}
    verdict, reason = smoke.evaluate(entry, None, None, "network error: refused")
    assert verdict == "HTTP_ERROR"
    assert "network error" in reason


def test_evaluate_never_auto_passes_unexpected_answer_for_abstain():
    smoke = _load()
    entry = {"id": 9, "question": "q", "expected": "ABSTAIN"}
    body = {"answered": True, "answer": "Some surprising answer.", "source": "a.md", "section": "s", "page": None}
    verdict, _ = smoke.evaluate(entry, 200, body, None)
    assert verdict == "FAIL"  # never silently reclassified as acceptable


def test_evaluate_never_auto_passes_unexpected_abstention_for_answer():
    smoke = _load()
    entry = {"id": 1, "question": "q", "expected": "ANSWER"}
    body = {"answered": False, "answer": None, "source": None, "section": None, "page": None}
    verdict, _ = smoke.evaluate(entry, 200, body, None)
    assert verdict == "FAIL"


# --- question set structural checks ---


def test_fifteen_questions_defined(smoke):
    assert len(smoke.QUESTIONS) == 15


def test_question_ids_are_1_through_15(smoke):
    assert [q["id"] for q in smoke.QUESTIONS] == list(range(1, 16))


def test_exactly_one_observe_question(smoke):
    observe = [q for q in smoke.QUESTIONS if q["expected"] == "OBSERVE"]
    assert len(observe) == 1
    assert observe[0]["id"] == 15


def test_eight_answer_and_six_abstain_questions(smoke):
    answer_qs = [q for q in smoke.QUESTIONS if q["expected"] == "ANSWER"]
    abstain_qs = [q for q in smoke.QUESTIONS if q["expected"] == "ABSTAIN"]
    assert len(answer_qs) == 8
    assert len(abstain_qs) == 6


# --- HTTP layer: fake transport, no real network/Ollama calls ---


class _FakeHTTPResponse:
    def __init__(self, status: int, body: dict):
        self.status = status
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_http_post_demo_sends_question_only_no_credentials(monkeypatch, smoke):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["data"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(200, {"answered": False, "answer": None, "source": None, "section": None, "page": None})

    monkeypatch.setattr(smoke.urllib.request, "urlopen", fake_urlopen)

    smoke._http_post_demo("http://127.0.0.1:5000", "a question", 5.0)

    assert captured["url"] == "http://127.0.0.1:5000/demo"
    assert captured["data"] == {"question": "a question"}
    assert "Authorization" not in captured["headers"]
    for header_value in captured["headers"].values():
        assert "OLLAMA_API_KEY" not in str(header_value)


def test_http_post_demo_parses_successful_response(monkeypatch, smoke):
    def fake_urlopen(request, timeout):
        return _FakeHTTPResponse(200, {"answered": True, "answer": "Yes.", "source": "a.md", "section": "s", "page": 1})

    monkeypatch.setattr(smoke.urllib.request, "urlopen", fake_urlopen)

    status, body, error, retry_after = smoke._http_post_demo("http://127.0.0.1:5000", "q", 5.0)

    assert status == 200
    assert body["answered"] is True
    assert error is None


def test_health_check_ok(monkeypatch, smoke):
    def fake_urlopen(request, timeout):
        return _FakeHTTPResponse(200, {"status": "ok"})

    monkeypatch.setattr(smoke.urllib.request, "urlopen", fake_urlopen)

    status, body, error = smoke._http_get("http://127.0.0.1:5000/health", 5.0)

    assert status == 200
    assert body == {"status": "ok"}


def test_script_never_imports_or_references_ollama_api_key(smoke):
    # The docstring mentions OLLAMA_API_KEY in prose (explaining what the script does NOT do) -
    # what matters is that it's never used as a string literal / env-var key in actual code.
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert '"OLLAMA_API_KEY"' not in source
    assert "'OLLAMA_API_KEY'" not in source


def test_script_never_calls_ollama_directly(smoke):
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "ollama.com" not in source.lower()
    assert "localhost:11434" not in source
