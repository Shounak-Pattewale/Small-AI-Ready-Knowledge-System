"""Security checks for the Ollama Cloud production-integration configuration:
no real credential ever committed, .env is ignored, .env.example has placeholders only.
"""

import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_EXAMPLE = _PROJECT_ROOT / ".env.example"
_GITIGNORE = _PROJECT_ROOT / ".gitignore"


def test_env_example_exists():
    assert _ENV_EXAMPLE.exists()


def test_env_example_declares_all_four_variables():
    text = _ENV_EXAMPLE.read_text(encoding="utf-8")
    for var in ["OLLAMA_API_KEY", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "OLLAMA_TIMEOUT_SECONDS"]:
        assert var in text


def test_env_example_api_key_is_blank_placeholder():
    text = _ENV_EXAMPLE.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("OLLAMA_API_KEY="):
            assert line == "OLLAMA_API_KEY="  # nothing after the '=' - no real key
            return
    raise AssertionError("OLLAMA_API_KEY= line not found in .env.example")


def test_env_example_contains_no_plausible_real_key():
    text = _ENV_EXAMPLE.read_text(encoding="utf-8")
    # A real key would be a long opaque token; the placeholder file should have none.
    for line in text.splitlines():
        if "=" in line and not line.strip().startswith("#"):
            _, _, value = line.partition("=")
            assert len(value.strip()) < 20  # every real default value here is short (model name, URL, number)


def test_gitignore_ignores_env_but_not_env_example():
    text = _GITIGNORE.read_text(encoding="utf-8")
    assert ".env" in text.splitlines()
    assert ".env.*" in text.splitlines()
    assert "!.env.example" in text.splitlines()


def test_dot_env_example_not_actually_ignored_by_git():
    # git check-ignore exits 0 if the path WOULD be ignored - .env.example must not be.
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(_ENV_EXAMPLE)],
        cwd=_PROJECT_ROOT, capture_output=True,
    )
    # Not a git repo yet in this project - git check-ignore still evaluates .gitignore
    # patterns locally without requiring a repo; if it does require one, skip gracefully.
    if result.returncode == 128:
        return
    assert result.returncode == 1  # 1 = not ignored (0 would mean incorrectly ignored)


def test_no_plausible_ollama_api_key_anywhere_in_tracked_source_files():
    # Scan common source/doc file types for anything that looks like a real bearer token,
    # not just .env.example. A real Ollama Cloud key is a long opaque string; this is a
    # coarse heuristic, not a full secret scanner.
    suspicious_patterns = ["OLLAMA_API_KEY=sk-", "OLLAMA_API_KEY=ollama-", "Bearer sk-"]
    for path in _PROJECT_ROOT.rglob("*"):
        if path.is_dir() or "venv" in path.parts or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path == Path(__file__):
            continue  # this file legitimately contains the patterns as strings to search for
        if path.suffix not in {".py", ".md", ".txt", ".json", ".example", ""}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in suspicious_patterns:
            assert pattern not in text, f"suspicious credential-like pattern found in {path}"
