"""Tests for app.py's local-dev .env auto-loading (load_dotenv, override=False).

Verifies the dotenv mechanics directly (via tmp_path .env files), not by
re-importing app.py (which would trigger real LLMKnowledgeService.load()
at module scope if OLLAMA_API_KEY happened to be unset in the test
environment) - the same load_dotenv() call app.py makes is exercised here
in isolation, with no risk of a real model/service load.
"""

from pathlib import Path

import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts with OLLAMA_API_KEY unset, regardless of the outer shell's env."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)


def test_env_values_loaded_for_local_startup(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_API_KEY=from-dotenv-file\n")

    load_dotenv(env_file, override=False)

    assert __import__("os").environ["OLLAMA_API_KEY"] == "from-dotenv-file"


def test_already_set_environment_variable_takes_precedence_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "real-production-value")
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_API_KEY=from-dotenv-file\n")

    load_dotenv(env_file, override=False)

    assert __import__("os").environ["OLLAMA_API_KEY"] == "real-production-value"


def test_missing_dotenv_file_does_not_raise(tmp_path):
    nonexistent = tmp_path / "does-not-exist.env"
    load_dotenv(nonexistent, override=False)  # dotenv silently no-ops on a missing file
    assert "OLLAMA_API_KEY" not in __import__("os").environ


def test_app_resolves_dotenv_path_from_project_root_not_cwd():
    """app.py must compute the .env path from its own file location, not the shell's cwd."""
    source = Path(__file__).resolve().parent.parent.joinpath("app.py").read_text(encoding="utf-8")
    assert "Path(__file__).resolve().parent" in source
    assert 'load_dotenv(_PROJECT_ROOT / ".env"' in source


def test_app_loads_dotenv_with_override_false():
    source = Path(__file__).resolve().parent.parent.joinpath("app.py").read_text(encoding="utf-8")
    assert "override=False" in source


def test_app_loads_dotenv_before_importing_knowledge_system_service():
    """load_dotenv() must appear before the knowledge_system.service import, so OllamaConfig.from_env()
    (called later, inside create_app()) always sees whatever .env provided."""
    source = Path(__file__).resolve().parent.parent.joinpath("app.py").read_text(encoding="utf-8")
    dotenv_call_index = source.index("load_dotenv(")
    service_import_index = source.index("from knowledge_system.service import")
    assert dotenv_call_index < service_import_index


def test_app_never_prints_or_logs_the_loaded_api_key_value():
    """Structural guard: no print()/logger call in app.py references OLLAMA_API_KEY or an
    environment-derived secret value."""
    source = Path(__file__).resolve().parent.parent.joinpath("app.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        code = line.split("#", 1)[0].strip()  # ignore comment text, only check actual code
        if code.startswith("print(") or ".log" in code or "logger." in code:
            assert "OLLAMA_API_KEY" not in code
            assert "os.environ" not in code
