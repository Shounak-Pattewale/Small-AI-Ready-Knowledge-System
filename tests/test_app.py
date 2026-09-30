"""Tests for app.py (Flask application layer) - the ONLY LLM-backed route is POST /demo.

Every test injects a fake service and/or a fake-clock rate limiter via
create_app()'s dependency injection - no real model is loaded, no real
Ollama Cloud call is made, OLLAMA_API_KEY is never required.
"""

import json

import pytest
from app import create_app
from knowledge_system.llm import LLMProviderError, LLMResponseError
from knowledge_system.models import KnowledgeAnswer
from ratelimit import RateLimiter


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class _FakeService:
    def __init__(self, result: KnowledgeAnswer | None = None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc
        self.ask_calls: list[str] = []

    def ask(self, question: str) -> KnowledgeAnswer:
        self.ask_calls.append(question)
        if self._exc is not None:
            raise self._exc
        return self._result


def _answered(**overrides) -> KnowledgeAnswer:
    defaults = dict(answered=True, answer="Up to five days.", source="holiday_leave_policy.pdf", section="3. Carry-over", page=1)
    defaults.update(overrides)
    return KnowledgeAnswer(**defaults)


def _abstained() -> KnowledgeAnswer:
    return KnowledgeAnswer(answered=False, answer=None, source=None, section=None, page=None)


def _make_client(service=None, limiter=None):
    service = service or _FakeService(result=_answered())
    app = create_app(service=service, rate_limiter=limiter)
    app.config["TESTING"] = True
    return app.test_client(), service


# --- APP ---


def test_create_app_works_with_fake_service():
    client, _ = _make_client()
    assert client is not None


def test_index_returns_200():
    client, _ = _make_client()
    response = client.get("/")
    assert response.status_code == 200


def test_index_renders_expected_app_title():
    client, _ = _make_client()
    response = client.get("/")
    assert b"Knowledge Assistant" in response.data


def test_index_does_not_call_service():
    client, service = _make_client()
    client.get("/")
    assert service.ask_calls == []


# --- HEALTH ---


def test_health_returns_200():
    client, _ = _make_client()
    response = client.get("/health")
    assert response.status_code == 200


def test_health_json_status_ok():
    client, _ = _make_client()
    response = client.get("/health")
    assert response.get_json() == {"status": "ok"}


def test_health_does_not_call_service():
    client, service = _make_client()
    client.get("/health")
    assert service.ask_calls == []


# --- DEMO VALIDATION ---


def test_demo_valid_json_accepted():
    client, _ = _make_client()
    response = client.post("/demo", json={"question": "Can I carry over leave?"})
    assert response.status_code == 200


def test_demo_missing_json_rejected():
    client, _ = _make_client()
    response = client.post("/demo", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_demo_json_array_rejected():
    client, _ = _make_client()
    response = client.post("/demo", json=["question", "x"])
    assert response.status_code == 400


def test_demo_missing_question_rejected():
    client, _ = _make_client()
    response = client.post("/demo", json={})
    assert response.status_code == 400


def test_demo_non_string_question_rejected():
    client, _ = _make_client()
    response = client.post("/demo", json={"question": 12345})
    assert response.status_code == 400


def test_demo_blank_question_rejected():
    client, _ = _make_client()
    response = client.post("/demo", json={"question": "   "})
    assert response.status_code == 400


def test_demo_question_over_1000_chars_rejected():
    client, _ = _make_client()
    response = client.post("/demo", json={"question": "a" * 1001})
    assert response.status_code == 400


def test_demo_oversized_body_rejected():
    client, _ = _make_client()
    huge_payload = {"question": "x", "padding": "y" * (32 * 1024)}
    response = client.post("/demo", json=huge_payload)
    assert response.status_code == 413


def test_invalid_demo_requests_never_call_service():
    client, service = _make_client()
    client.post("/demo", json={})
    client.post("/demo", json={"question": 5})
    client.post("/demo", json={"question": "   "})
    client.post("/demo", json={"question": "a" * 1001})
    assert service.ask_calls == []


# --- ANSWERED ---


def test_answered_returns_correct_json():
    service = _FakeService(result=_answered(answer="Yes, up to five days."))
    client, _ = _make_client(service=service)
    response = client.post("/demo", json={"question": "q"})
    body = response.get_json()
    assert body["answered"] is True
    assert body["answer"] == "Yes, up to five days."


def test_answered_source_preserved():
    service = _FakeService(result=_answered(source="it_support_security.md"))
    client, _ = _make_client(service=service)
    body = client.post("/demo", json={"question": "q"}).get_json()
    assert body["source"] == "it_support_security.md"


def test_answered_section_preserved():
    service = _FakeService(result=_answered(section="3. Lost or stolen devices"))
    client, _ = _make_client(service=service)
    body = client.post("/demo", json={"question": "q"}).get_json()
    assert body["section"] == "3. Lost or stolen devices"


def test_answered_page_preserved():
    service = _FakeService(result=_answered(page=7))
    client, _ = _make_client(service=service)
    body = client.post("/demo", json={"question": "q"}).get_json()
    assert body["page"] == 7


def test_answered_page_null_supported():
    service = _FakeService(result=_answered(page=None))
    client, _ = _make_client(service=service)
    body = client.post("/demo", json={"question": "q"}).get_json()
    assert body["page"] is None


# --- ABSTENTION ---


def test_abstention_returns_200():
    service = _FakeService(result=_abstained())
    client, _ = _make_client(service=service)
    response = client.post("/demo", json={"question": "q"})
    assert response.status_code == 200


def test_abstention_answer_null():
    service = _FakeService(result=_abstained())
    client, _ = _make_client(service=service)
    body = client.post("/demo", json={"question": "q"}).get_json()
    assert body["answer"] is None
    assert body["answered"] is False


def test_abstention_provenance_null():
    service = _FakeService(result=_abstained())
    client, _ = _make_client(service=service)
    body = client.post("/demo", json={"question": "q"}).get_json()
    assert body["source"] is None
    assert body["section"] is None
    assert body["page"] is None


# --- ERRORS ---


def test_provider_error_returns_503():
    service = _FakeService(exc=LLMProviderError("Ollama Cloud unavailable"))
    client, _ = _make_client(service=service)
    response = client.post("/demo", json={"question": "q"})
    assert response.status_code == 503


def test_response_error_returns_503():
    service = _FakeService(exc=LLMResponseError("malformed structured output"))
    client, _ = _make_client(service=service)
    response = client.post("/demo", json={"question": "q"})
    assert response.status_code == 503


def test_unexpected_error_returns_500():
    service = _FakeService(exc=RuntimeError("boom"))
    client, _ = _make_client(service=service)
    response = client.post("/demo", json={"question": "q"})
    assert response.status_code == 500


def test_error_responses_contain_no_secret_or_internal_text():
    service = _FakeService(exc=RuntimeError("boom with secret_key=abc123 in it"))
    client, _ = _make_client(service=service)
    response = client.post("/demo", json={"question": "q"})
    body_text = response.get_data(as_text=True).lower()
    for forbidden in ["secret_key", "traceback", "runtimeerror", "abc123", "/home/", "site-packages"]:
        assert forbidden not in body_text


# --- RATE LIMIT (through the real Flask route) ---


def test_rate_limit_requests_1_to_10_allowed():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    client, _ = _make_client(limiter=limiter)
    for _ in range(10):
        response = client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
        assert response.status_code == 200


def test_rate_limit_request_11_blocked():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    client, _ = _make_client(limiter=limiter)
    for _ in range(10):
        client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    response = client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    assert response.status_code == 429


def test_rate_limited_request_never_calls_service():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    service = _FakeService(result=_answered())
    client, _ = _make_client(service=service, limiter=limiter)
    for _ in range(10):
        client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    assert len(service.ask_calls) == 10
    client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    assert len(service.ask_calls) == 10  # 11th blocked, no new call


def test_rate_limit_separate_ip_independent():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    client, _ = _make_client(limiter=limiter)
    for _ in range(10):
        client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    blocked = client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    assert blocked.status_code == 429
    other_ip = client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "5.6.7.8"})
    assert other_ip.status_code == 200


def test_rate_limit_resets_after_window():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    client, _ = _make_client(limiter=limiter)
    for _ in range(10):
        client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    blocked = client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    assert blocked.status_code == 429

    clock.advance(61.0)
    allowed_again = client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    assert allowed_again.status_code == 200


def test_health_unaffected_by_rate_limit():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    client, _ = _make_client(limiter=limiter)
    for _ in range(10):
        client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})  # 11th, blocked
    response = client.get("/health", environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    assert response.status_code == 200


def test_index_unaffected_by_rate_limit():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    client, _ = _make_client(limiter=limiter)
    for _ in range(10):
        client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})  # 11th, blocked
    response = client.get("/", environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    assert response.status_code == 200


def test_rate_limit_429_body_is_safe_json():
    clock = _FakeClock()
    limiter = RateLimiter(limit=10, window_seconds=60.0, clock=clock)
    client, _ = _make_client(limiter=limiter)
    for _ in range(10):
        client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    response = client.post("/demo", json={"question": "q"}, environ_overrides={"REMOTE_ADDR": "1.2.3.4"})
    body = response.get_json()
    assert body == {"error": "Too many requests. Please wait a moment and try again."}


# --- SERVICE LIFETIME ---


def test_same_injected_service_instance_reused_across_requests():
    service = _FakeService(result=_answered())
    client, _ = _make_client(service=service)
    client.post("/demo", json={"question": "one"})
    client.post("/demo", json={"question": "two"})
    assert service.ask_calls == ["one", "two"]


# --- SECURITY / RESPONSE ---


def test_demo_response_content_type_is_json():
    client, _ = _make_client()
    response = client.post("/demo", json={"question": "q"})
    assert response.content_type.startswith("application/json")


def test_request_size_limit_configured():
    client, _ = _make_client()
    assert client.application.config["MAX_CONTENT_LENGTH"] == 16 * 1024


def test_debug_mode_not_forced_on():
    client, _ = _make_client()
    assert client.application.debug is False


def test_no_secret_exposed_in_health_or_index():
    client, _ = _make_client()
    for path in ("/", "/health"):
        text = client.get(path).get_data(as_text=True).lower()
        assert "ollama_api_key" not in text
        assert "bearer" not in text


def test_only_expected_routes_exist():
    client, _ = _make_client()
    app = client.application
    rules = {rule.rule for rule in app.url_map.iter_rules() if not rule.rule.startswith("/static")}
    assert rules == {"/", "/health", "/demo"}
