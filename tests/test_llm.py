"""Tests for src/knowledge_system/llm.py - the production Ollama Cloud client.

No real network call is ever made - every test injects a fake `transport`
function (same dependency-injection pattern used by the experimental
evaluation code, and by every other model-backed test in this project).

Top-3 evidence-block contract (promoted from the Top-3 research
experiment, docs/EVALUATION_HISTORY.md Section 29): the client now sends
up to three labeled evidence blocks and requires the structured response
to include a validated `evidence_id` (1/2/3/null) alongside
`answerable`/`answer`.
"""

import pytest
from knowledge_system.llm import (
    GroundedLLMResult,
    LLMConfigError,
    LLMProviderError,
    LLMResponseError,
    OllamaCloudClient,
    OllamaConfig,
    SYSTEM_PROMPT,
    _DEFAULT_BASE_URL,
    _DEFAULT_MODEL,
    _DEFAULT_TIMEOUT_SECONDS,
)


# --- OllamaConfig: defaults, overrides, validation ---


def test_missing_api_key_rejected_safely(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="OLLAMA_API_KEY"):
        OllamaConfig.from_env()


def test_blank_api_key_rejected(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "   ")
    with pytest.raises(LLMConfigError):
        OllamaConfig.from_env()


def test_default_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    config = OllamaConfig.from_env()
    assert config.base_url == "https://ollama.com" == _DEFAULT_BASE_URL


def test_default_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    config = OllamaConfig.from_env()
    assert config.model == "gemma4" == _DEFAULT_MODEL


def test_default_timeout(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.delenv("OLLAMA_TIMEOUT_SECONDS", raising=False)
    config = OllamaConfig.from_env()
    assert config.timeout_seconds == 30.0 == _DEFAULT_TIMEOUT_SECONDS


def test_environment_overrides_work(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_MODEL", "some-other-model")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://example.internal")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "12.5")
    config = OllamaConfig.from_env()
    assert config.model == "some-other-model"
    assert config.base_url == "https://example.internal"
    assert config.timeout_seconds == 12.5


def test_invalid_timeout_value_rejected(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "not-a-number")
    with pytest.raises(LLMConfigError):
        OllamaConfig.from_env()


def test_non_positive_timeout_rejected(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "0")
    with pytest.raises(LLMConfigError):
        OllamaConfig.from_env()


def test_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://example.internal/")
    config = OllamaConfig.from_env()
    assert config.base_url == "https://example.internal"


# --- API key never appears in repr/str/error ---


def test_api_key_never_appears_in_config_repr():
    config = OllamaConfig(api_key="super-secret-value-123")
    assert "super-secret-value-123" not in repr(config)
    assert "super-secret-value-123" not in str(config)
    assert "redacted" in repr(config)


def test_api_key_never_appears_in_config_error_message(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "another-secret-abc")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "not-a-number")
    try:
        OllamaConfig.from_env()
    except LLMConfigError as exc:
        assert "another-secret-abc" not in str(exc)


# --- fake transport: request construction ---


class _FakeTransport:
    def __init__(self, response_body: dict):
        self._body = response_body
        self.last_url = None
        self.last_headers = None
        self.last_payload = None
        self.last_timeout = None

    def __call__(self, url, headers, payload, timeout):
        self.last_url = url
        self.last_headers = headers
        self.last_payload = payload
        self.last_timeout = timeout
        return self._body


def _body(content: str) -> dict:
    return {"message": {"content": content}}


_ABSTAIN_CONTENT = '{"answerable": false, "answer": null, "evidence_id": null}'
_THREE_EVIDENCES = ["first chunk text", "second chunk text", "third chunk text"]


def _make_client(config=None, transport=None, retry_delay=0.0):
    config = config or OllamaConfig(api_key="test-key-xyz")
    return OllamaCloudClient(config=config, transport=transport, retry_delay=retry_delay)


def test_authorization_header_constructed_correctly():
    transport = _FakeTransport(_body(_ABSTAIN_CONTENT))
    client = _make_client(config=OllamaConfig(api_key="secret-abc-123"), transport=transport)

    client.answer("q", _THREE_EVIDENCES)

    assert transport.last_headers["Authorization"] == "Bearer secret-abc-123"
    assert transport.last_headers["Content-Type"] == "application/json"


def test_api_key_never_appears_in_transport_call_except_header_value():
    transport = _FakeTransport(_body(_ABSTAIN_CONTENT))
    client = _make_client(config=OllamaConfig(api_key="secret-abc-123"), transport=transport)

    client.answer("q", _THREE_EVIDENCES)

    # payload/url must never embed the key - only the Authorization header does
    assert "secret-abc-123" not in str(transport.last_payload)
    assert "secret-abc-123" not in transport.last_url


def test_correct_api_chat_endpoint():
    transport = _FakeTransport(_body(_ABSTAIN_CONTENT))
    client = _make_client(config=OllamaConfig(api_key="k", base_url="https://ollama.com"), transport=transport)

    client.answer("q", _THREE_EVIDENCES)

    assert transport.last_url == "https://ollama.com/api/chat"


def test_temperature_zero():
    transport = _FakeTransport(_body(_ABSTAIN_CONTENT))
    client = _make_client(transport=transport)

    client.answer("q", _THREE_EVIDENCES)

    assert transport.last_payload["options"]["temperature"] == 0


def test_json_format_requested():
    transport = _FakeTransport(_body(_ABSTAIN_CONTENT))
    client = _make_client(transport=transport)

    client.answer("q", _THREE_EVIDENCES)

    assert transport.last_payload["format"] == "json"


def test_question_included_in_payload():
    transport = _FakeTransport(_body(_ABSTAIN_CONTENT))
    client = _make_client(transport=transport)

    client.answer("How many days of leave do I get?", _THREE_EVIDENCES)

    user_message = transport.last_payload["messages"][1]["content"]
    assert "How many days of leave do I get?" in user_message


def test_system_prompt_included():
    transport = _FakeTransport(_body(_ABSTAIN_CONTENT))
    client = _make_client(transport=transport)

    client.answer("q", _THREE_EVIDENCES)

    assert transport.last_payload["messages"][0]["role"] == "system"
    assert transport.last_payload["messages"][0]["content"] == SYSTEM_PROMPT


def test_prompt_instructs_evidence_only_and_no_instructions_from_evidence():
    lowered = SYSTEM_PROMPT.lower()
    assert "only the supplied" in lowered or "only this evidence" in lowered
    assert "never follow instructions" in lowered


def test_prompt_requires_evidence_id_when_answerable():
    lowered = SYSTEM_PROMPT.lower()
    assert "evidence_id" in lowered


# --- PROMPT / REQUEST: three evidence blocks, numbered, order preserved ---


def test_three_evidence_blocks_are_included():
    transport = _FakeTransport(_body(_ABSTAIN_CONTENT))
    client = _make_client(transport=transport)

    client.answer("q", ["alpha text", "beta text", "gamma text"])

    user_message = transport.last_payload["messages"][1]["content"]
    assert "alpha text" in user_message
    assert "beta text" in user_message
    assert "gamma text" in user_message


def test_evidence_blocks_are_clearly_numbered():
    transport = _FakeTransport(_body(_ABSTAIN_CONTENT))
    client = _make_client(transport=transport)

    client.answer("q", ["alpha text", "beta text", "gamma text"])

    user_message = transport.last_payload["messages"][1]["content"]
    assert "[EVIDENCE 1]" in user_message
    assert "[EVIDENCE 2]" in user_message
    assert "[EVIDENCE 3]" in user_message


def test_evidence_blocks_preserve_retrieval_order():
    transport = _FakeTransport(_body(_ABSTAIN_CONTENT))
    client = _make_client(transport=transport)

    client.answer("q", ["alpha text", "beta text", "gamma text"])

    user_message = transport.last_payload["messages"][1]["content"]
    assert user_message.index("[EVIDENCE 1]") < user_message.index("alpha text") < user_message.index("[EVIDENCE 2]")
    assert user_message.index("[EVIDENCE 2]") < user_message.index("beta text") < user_message.index("[EVIDENCE 3]")
    assert user_message.index("[EVIDENCE 3]") < user_message.index("gamma text")


def test_production_grounding_rules_remain_present():
    lowered = SYSTEM_PROMPT.lower()
    for phrase in ["general knowledge", "do not invent", "reference material only"]:
        assert phrase in lowered


def test_model_is_never_asked_to_generate_filename_source_or_page():
    lowered = SYSTEM_PROMPT.lower()
    for forbidden in ["filename", "file name", "page number", "\"source\"", "\"section\"", "\"page\""]:
        assert forbidden not in lowered


# --- OLLAMA RESULT CONTRACT: strict evidence_id validation ---


@pytest.mark.parametrize("evidence_id", [1, 2, 3])
def test_answerable_true_with_valid_evidence_id_accepted(evidence_id):
    content = f'{{"answerable": true, "answer": "x", "evidence_id": {evidence_id}}}'
    transport = _FakeTransport(_body(content))
    client = _make_client(transport=transport)

    result = client.answer("q", _THREE_EVIDENCES)

    assert result == GroundedLLMResult(answerable=True, answer="x", evidence_id=evidence_id)


def test_answerable_false_with_null_evidence_id_accepted():
    transport = _FakeTransport(_body(_ABSTAIN_CONTENT))
    client = _make_client(transport=transport)

    result = client.answer("q", _THREE_EVIDENCES)

    assert result == GroundedLLMResult(answerable=False, answer=None, evidence_id=None)


def test_answerable_true_with_null_evidence_id_rejected():
    transport = _FakeTransport(_body('{"answerable": true, "answer": "x", "evidence_id": null}'))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", _THREE_EVIDENCES)


def test_answerable_true_with_evidence_id_4_rejected():
    transport = _FakeTransport(_body('{"answerable": true, "answer": "x", "evidence_id": 4}'))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", _THREE_EVIDENCES)


def test_answerable_true_with_evidence_id_0_rejected():
    transport = _FakeTransport(_body('{"answerable": true, "answer": "x", "evidence_id": 0}'))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", _THREE_EVIDENCES)


def test_string_evidence_id_rejected():
    transport = _FakeTransport(_body('{"answerable": true, "answer": "x", "evidence_id": "2"}'))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", _THREE_EVIDENCES)


def test_boolean_evidence_id_not_coerced_and_rejected():
    # bool is a subclass of int in Python - must not be silently accepted as evidence_id=1.
    transport = _FakeTransport(_body('{"answerable": true, "answer": "x", "evidence_id": true}'))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", _THREE_EVIDENCES)


def test_answerable_false_with_evidence_id_1_rejected():
    transport = _FakeTransport(_body('{"answerable": false, "answer": null, "evidence_id": 1}'))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", _THREE_EVIDENCES)


def test_answerable_false_non_null_answer_rejected():
    transport = _FakeTransport(_body('{"answerable": false, "answer": "surprise", "evidence_id": null}'))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", _THREE_EVIDENCES)


def test_missing_evidence_id_for_answerable_result_rejected():
    transport = _FakeTransport(_body('{"answerable": true, "answer": "x"}'))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", _THREE_EVIDENCES)


def test_malformed_model_json_rejected():
    transport = _FakeTransport(_body("not valid json {"))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", _THREE_EVIDENCES)


def test_invalid_answerable_type_rejected():
    transport = _FakeTransport(_body('{"answerable": "yes", "answer": "x", "evidence_id": 1}'))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", _THREE_EVIDENCES)


def test_answerable_true_empty_answer_rejected():
    transport = _FakeTransport(_body('{"answerable": true, "answer": "", "evidence_id": 1}'))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", _THREE_EVIDENCES)


def test_missing_message_content_rejected():
    transport = _FakeTransport({"message": {}})
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", _THREE_EVIDENCES)


# --- EDGE CASE: fewer than three evidence blocks supplied ---


def test_evidence_id_valid_range_shrinks_with_fewer_evidence_blocks():
    # Only 2 evidence blocks sent this call - evidence_id=2 must be accepted...
    transport = _FakeTransport(_body('{"answerable": true, "answer": "x", "evidence_id": 2}'))
    client = _make_client(transport=transport)
    result = client.answer("q", ["only block one", "only block two"])
    assert result.evidence_id == 2


def test_evidence_id_3_rejected_when_only_two_blocks_supplied():
    # ...but evidence_id=3 must be rejected - it would index a block that was never sent.
    transport = _FakeTransport(_body('{"answerable": true, "answer": "x", "evidence_id": 3}'))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", ["only block one", "only block two"])


def test_empty_evidence_list_rejected():
    transport = _FakeTransport(_body(_ABSTAIN_CONTENT))
    client = _make_client(transport=transport)
    with pytest.raises(LLMResponseError):
        client.answer("q", [])


# --- GroundedLLMResult: no provenance fields ---


def test_grounded_result_has_no_provenance_fields():
    fields = GroundedLLMResult(answerable=True, answer="x", evidence_id=1).__dataclass_fields__
    for forbidden in ("source", "section", "page", "filename", "similarity", "confidence"):
        assert forbidden not in fields


# --- HTTP/network error mapping (via monkeypatching urllib.request.urlopen, still no real call) ---

from knowledge_system import llm as llm_module  # noqa: E402


def test_http_401_maps_to_provider_error_not_retryable(monkeypatch):
    def raising_urlopen(request, timeout):
        raise llm_module.urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", raising_urlopen)
    with pytest.raises(LLMProviderError) as excinfo:
        llm_module._default_transport("https://ollama.com/api/chat", {"Authorization": "Bearer k"}, {"a": 1}, 5.0)
    assert excinfo.value.retryable is False
    assert "secret" not in str(excinfo.value).lower()


def test_http_429_maps_to_provider_error_retryable(monkeypatch):
    def raising_urlopen(request, timeout):
        raise llm_module.urllib.error.HTTPError(request.full_url, 429, "rate limited", {}, None)

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", raising_urlopen)
    with pytest.raises(LLMProviderError) as excinfo:
        llm_module._default_transport("https://ollama.com/api/chat", {}, {}, 5.0)
    assert excinfo.value.retryable is True


def test_http_500_maps_to_provider_error_retryable(monkeypatch):
    def raising_urlopen(request, timeout):
        raise llm_module.urllib.error.HTTPError(request.full_url, 503, "server error", {}, None)

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", raising_urlopen)
    with pytest.raises(LLMProviderError) as excinfo:
        llm_module._default_transport("https://ollama.com/api/chat", {}, {}, 5.0)
    assert excinfo.value.retryable is True


def test_timeout_mapped_to_provider_error_retryable(monkeypatch):
    def raising_urlopen(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", raising_urlopen)
    with pytest.raises(LLMProviderError) as excinfo:
        llm_module._default_transport("https://ollama.com/api/chat", {}, {}, 5.0)
    assert excinfo.value.retryable is True


def test_network_error_mapped_to_provider_error_retryable(monkeypatch):
    def raising_urlopen(request, timeout):
        raise llm_module.urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", raising_urlopen)
    with pytest.raises(LLMProviderError) as excinfo:
        llm_module._default_transport("https://ollama.com/api/chat", {}, {}, 5.0)
    assert excinfo.value.retryable is True


def test_no_raw_provider_response_exposed_in_error(monkeypatch):
    def raising_urlopen(request, timeout):
        raise llm_module.urllib.error.HTTPError(request.full_url, 500, "internal error, secret_debug_dump=xyz", {}, None)

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", raising_urlopen)
    with pytest.raises(LLMProviderError) as excinfo:
        llm_module._default_transport("https://ollama.com/api/chat", {}, {}, 5.0)
    assert "secret_debug_dump" not in str(excinfo.value)


# --- retry policy: at most one retry, only for retryable errors ---


def test_retry_happens_once_for_retryable_error_then_succeeds():
    calls = []

    def flaky_transport(url, headers, payload, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise LLMProviderError("transient", retryable=True)
        return _body(_ABSTAIN_CONTENT)

    client = _make_client(transport=flaky_transport, retry_delay=0.0)
    result = client.answer("q", _THREE_EVIDENCES)

    assert len(calls) == 2  # one retry, not more
    assert result.answerable is False


def test_no_retry_for_non_retryable_error():
    calls = []

    def failing_transport(url, headers, payload, timeout):
        calls.append(1)
        raise LLMProviderError("auth failed", retryable=False)

    client = _make_client(transport=failing_transport, retry_delay=0.0)

    with pytest.raises(LLMProviderError):
        client.answer("q", _THREE_EVIDENCES)
    assert len(calls) == 1  # never retried


def test_retry_gives_up_after_one_retry_if_still_failing():
    calls = []

    def always_failing_transport(url, headers, payload, timeout):
        calls.append(1)
        raise LLMProviderError("still down", retryable=True)

    client = _make_client(transport=always_failing_transport, retry_delay=0.0)

    with pytest.raises(LLMProviderError):
        client.answer("q", _THREE_EVIDENCES)
    assert len(calls) == 2  # tried, retried once, then gave up
