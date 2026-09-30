#!/usr/bin/env python3
"""Flask web application for the Small AI-Ready Knowledge System.

Application layer only - all retrieval/LLM logic lives in
knowledge_system.service.LLMKnowledgeService, loaded once per process and
reused across requests. This module never talks to Granite or Ollama
directly: question in -> service.ask(question) -> KnowledgeAnswer out.

Three public routes: GET / (chat UI), GET /health (cheap liveness check,
no model calls), POST /demo (the only LLM-backed endpoint, rate-limited
and validated before the service is ever called).
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Local-dev convenience only: populate os.environ from the root .env file BEFORE any
# knowledge_system module reads config (OllamaConfig.from_env() etc). override=False means a
# real environment variable already set (Docker/Compose/VPS/CI) always wins over .env - this
# never overrides production config, it only fills gaps for `python3 app.py` on a laptop.
load_dotenv(_PROJECT_ROOT / ".env", override=False)

from flask import Flask, jsonify, render_template, request  # noqa: E402

from knowledge_system.llm import LLMConfigError, LLMProviderError, LLMResponseError  # noqa: E402
from knowledge_system.service import LLMKnowledgeService  # noqa: E402
from ratelimit import RateLimiter  # noqa: E402

_MAX_QUESTION_LENGTH = 1000
_MAX_CONTENT_LENGTH_BYTES = 16 * 1024  # 16 KB - comfortably above a 1000-char question, still small
_RATE_LIMIT_PER_MINUTE = 10
_RATE_LIMIT_WINDOW_SECONDS = 60.0

_TOO_MANY_REQUESTS_MESSAGE = "Too many requests. Please wait a moment and try again."
_PROVIDER_UNAVAILABLE_MESSAGE = "The knowledge service is temporarily unavailable."
_GENERIC_ERROR_MESSAGE = "Something went wrong."


def create_app(service=None, rate_limiter: RateLimiter | None = None) -> Flask:
    """Build the Flask app once. `service`/`rate_limiter` are dependency-injected (tests pass a
    fake service and/or a limiter with a fake clock); production passes neither, which loads the
    real LLMKnowledgeService (Granite + Ollama Cloud client) and a real-time limiter here, exactly
    once - never per request."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = _MAX_CONTENT_LENGTH_BYTES

    knowledge_service = service if service is not None else LLMKnowledgeService.load()
    limiter = rate_limiter if rate_limiter is not None else RateLimiter(limit=_RATE_LIMIT_PER_MINUTE, window_seconds=_RATE_LIMIT_WINDOW_SECONDS)

    @app.get("/")
    def index():
        # Static render only - no retrieval, no LLM call. The one initial assistant
        # message is baked into the template, not fetched from the backend.
        return render_template("index.html")

    @app.get("/health")
    def health():
        # Deliberately cheap: no Ollama call, no Granite retrieval, no filesystem/model checks.
        return jsonify(status="ok"), 200

    @app.post("/demo")
    def demo():
        # --- request validation (before the rate-limit check, before the service call) ---
        if not request.is_json:
            return jsonify(error="Request must be JSON."), 400
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Request body must be a JSON object."), 400

        question = payload.get("question")
        if not isinstance(question, str):
            return jsonify(error="'question' must be a string."), 400
        stripped = question.strip()
        if not stripped:
            return jsonify(error="'question' must not be empty."), 400
        if len(question) > _MAX_QUESTION_LENGTH:
            return jsonify(error=f"'question' must be at most {_MAX_QUESTION_LENGTH} characters."), 400

        # --- rate limit (before the service call - a rejected request makes zero cloud calls) ---
        client_ip = request.remote_addr or "unknown"  # Werkzeug's own resolution, no X-Forwarded-For trust here
        if not limiter.allow(client_ip):
            return jsonify(error=_TOO_MANY_REQUESTS_MESSAGE), 429

        # --- service call ---
        try:
            answer = knowledge_service.ask(stripped)
        except (LLMProviderError, LLMResponseError):
            app.logger.warning("LLM provider/response failure on /demo")  # no secrets, no raw provider body
            return jsonify(error=_PROVIDER_UNAVAILABLE_MESSAGE), 503
        except Exception:
            app.logger.exception("Unexpected error on /demo")
            return jsonify(error=_GENERIC_ERROR_MESSAGE), 500

        return jsonify(
            answered=answer.answered,
            answer=answer.answer,
            source=answer.source,
            section=answer.section,
            page=answer.page,
        ), 200

    return app


if __name__ == "__main__":
    try:
        flask_app = create_app()
    except LLMConfigError as exc:
        print(str(exc))  # e.g. "OLLAMA_API_KEY is not configured" - never a stack trace, never the key
        sys.exit(1)
    flask_app.run(debug=False)
