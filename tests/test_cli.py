"""Tests for cli.py - only the pure formatting function, per instructions
("test formatting functions separately rather than simulating a terminal
session"). No KnowledgeService, no models, no I/O.
"""

import importlib.util
from pathlib import Path

import pytest

_CLI_PATH = Path(__file__).resolve().parent.parent / "cli.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("cli_under_test", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _answer(cli, **overrides):
    from knowledge_system.models import KnowledgeAnswer

    defaults = dict(answered=True, answer="up to five days", source="holiday_leave_policy.pdf", section="4. Carry-over", page=3)
    defaults.update(overrides)
    return KnowledgeAnswer(**defaults)


def test_format_answer_shows_all_present_fields(cli):
    text = cli.format_answer(_answer(cli))

    assert "Answer:" in text
    assert "up to five days" in text
    assert "Source:" in text
    assert "holiday_leave_policy.pdf" in text
    assert "Section:" in text
    assert "4. Carry-over" in text
    assert "Page:" in text
    assert "3" in text


def test_format_answer_omits_none_section_and_page(cli):
    text = cli.format_answer(_answer(cli, section=None, page=None))

    assert "Section:" not in text
    assert "Page:" not in text
    assert "Source:" in text  # still present since it wasn't None


def test_format_answer_insufficient_information(cli):
    answer = _answer(cli, answered=False, answer=None, source=None, section=None, page=None)

    text = cli.format_answer(answer)

    assert text == "I couldn't find enough information in the knowledge base to answer that."
    assert "Source:" not in text
    assert "Answer:" not in text


def test_format_answer_never_shows_raw_scores(cli):
    text = cli.format_answer(_answer(cli))

    # KnowledgeAnswer has no score/signal/similarity fields at all, so this
    # is really a guard against someone adding one later and wiring it in.
    for forbidden in ("confidence", "signal", "similarity", "%"):
        assert forbidden not in text.lower()


def test_exit_commands_defined(cli):
    assert cli._EXIT_COMMANDS == {"exit", "quit"}


# --- --mode argument parsing ---


def test_default_mode_is_llm(cli):
    args = cli.parse_args([])
    assert args.mode == "llm"


def test_mode_llm_explicit(cli):
    args = cli.parse_args(["--mode", "llm"])
    assert args.mode == "llm"


def test_mode_baseline_explicit(cli):
    args = cli.parse_args(["--mode", "baseline"])
    assert args.mode == "baseline"


def test_invalid_mode_rejected(cli):
    with pytest.raises(SystemExit):
        cli.parse_args(["--mode", "nonsense"])


# --- mode -> service class mapping (pure, no I/O) ---


def test_llm_mode_selects_llm_knowledge_service(cli):
    from knowledge_system.service import LLMKnowledgeService

    assert cli.service_class_for_mode("llm") is LLMKnowledgeService


def test_baseline_mode_selects_knowledge_service(cli):
    from knowledge_system.service import KnowledgeService

    assert cli.service_class_for_mode("baseline") is KnowledgeService


# --- provider-failure formatting is a safe, generic message ---


def test_provider_unavailable_message_is_generic_and_safe(cli):
    assert cli._PROVIDER_UNAVAILABLE_MESSAGE == "The language model service is temporarily unavailable."
    lowered = cli._PROVIDER_UNAVAILABLE_MESSAGE.lower()
    for forbidden in ["api_key", "authorization", "bearer", "traceback", "exception"]:
        assert forbidden not in lowered


# --- security: no credential ever appears in cli.py's own source ---


def test_cli_source_contains_no_hardcoded_credential():
    source = _CLI_PATH.read_text(encoding="utf-8")
    for forbidden in ["OLLAMA_API_KEY=", "Bearer ey", "sk-"]:
        assert forbidden not in source
