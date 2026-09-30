#!/usr/bin/env python3
"""Small interactive terminal CLI for the Small AI-Ready Knowledge System.

Loads a production KnowledgeService once, then answers repeated questions
from the terminal. No Flask, no web UI - this is a manual verification
step / the same service layer Flask will reuse later.

Two modes, both backed by the same Granite retrieval:
    --mode llm       Ollama Cloud grounded answerability (default; requires OLLAMA_API_KEY)
    --mode baseline  MiniLM extractive QA (the original non-LLM baseline)

Run from the project root:
    venv/bin/python cli.py                  # LLM mode
    venv/bin/python cli.py --mode baseline   # baseline mode
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from knowledge_system.llm import LLMConfigError, LLMProviderError, LLMResponseError  # noqa: E402
from knowledge_system.models import KnowledgeAnswer  # noqa: E402
from knowledge_system.service import KnowledgeService, LLMKnowledgeService  # noqa: E402

_EXIT_COMMANDS = {"exit", "quit"}
_INSUFFICIENT_INFO_MESSAGE = "I couldn't find enough information in the knowledge base to answer that."
_PROVIDER_UNAVAILABLE_MESSAGE = "The language model service is temporarily unavailable."


def format_answer(answer: KnowledgeAnswer) -> str:
    """Render a KnowledgeAnswer as CLI text. Never shows raw similarity/QA-signal/confidence."""
    if not answer.answered:
        return _INSUFFICIENT_INFO_MESSAGE

    lines = ["Answer:", answer.answer or "", ""]
    if answer.source:
        lines += ["Source:", answer.source, ""]
    if answer.section:
        lines += ["Section:", answer.section, ""]
    if answer.page is not None:
        lines += ["Page:", str(answer.page), ""]
    return "\n".join(lines).rstrip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small AI-Ready Knowledge System")
    parser.add_argument(
        "--mode", choices=["llm", "baseline"], default="llm",
        help="answer pipeline: 'llm' = Ollama Cloud (default), 'baseline' = MiniLM extractive QA",
    )
    return parser.parse_args(argv)


def service_class_for_mode(mode: str) -> type:
    """Pure mapping, no I/O - both classes expose a no-arg-required .load() classmethod."""
    return LLMKnowledgeService if mode == "llm" else KnowledgeService


def main() -> None:
    args = parse_args()

    print("Small AI-Ready Knowledge System")
    print("Type a question about company policies.")
    print("Type 'exit' or 'quit' to stop.\n")

    service_class = service_class_for_mode(args.mode)
    try:
        service = service_class.load()  # loads Granite (+ MiniLM or the Ollama Cloud client) once
    except LLMConfigError as exc:
        print(str(exc))  # e.g. "OLLAMA_API_KEY is not configured" - never a stack trace, never the key
        return
    print(f"Mode: {'LLM (Ollama Cloud)' if args.mode == 'llm' else 'baseline (MiniLM extractive QA)'}\n")

    while True:
        try:
            question = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()  # clean newline so the shell prompt doesn't land mid-line
            break

        stripped = question.strip()
        if not stripped:
            continue  # empty input never reaches the models
        if stripped.lower() in _EXIT_COMMANDS:
            break

        try:
            answer = service.ask(stripped)
        except ValueError:
            continue  # shouldn't happen given the strip() check above, but never show a stack trace either way
        except (LLMProviderError, LLMResponseError):
            print()
            print(_PROVIDER_UNAVAILABLE_MESSAGE)  # provider/network failure - never disguised as "no info found"
            print()
            continue

        print()
        print(format_answer(answer))
        print()


if __name__ == "__main__":
    main()
