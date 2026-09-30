"""Verifies evaluation/evaluate_tfidf.py never triggers raw document parsing.

This is the architectural boundary the ingestion/persistence split exists
to enforce: evaluation must only ever call knowledge_system.storage.load_chunks(),
never knowledge_system.loaders.load_directory() (which is what invokes Docling).
Proven here by making load_directory() raise if called at all, then running
the real evaluation entry point end-to-end against a small persisted corpus.
"""

import importlib.util
import sys
from pathlib import Path

from knowledge_system.models import DocumentChunk
from knowledge_system.storage import save_chunks

_EVALUATE_TFIDF_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "evaluate_tfidf.py"


def _load_evaluate_tfidf_module():
    """Import evaluation/evaluate_tfidf.py as a fresh module (it's a script, not a package)."""
    spec = importlib.util.spec_from_file_location("evaluate_tfidf_under_test", _EVALUATE_TFIDF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_tfidf_never_calls_load_directory(monkeypatch, kb_dir: Path, capsys):
    def load_directory_must_not_be_called(*args, **kwargs):
        raise AssertionError("evaluate_tfidf.main() must not call load_directory() - Docling boundary violated")

    monkeypatch.setattr("knowledge_system.loaders.load_directory", load_directory_must_not_be_called)

    # A tiny persisted corpus with enough vocabulary overlap for every
    # evaluation/questions.json question to get a non-empty ranked result.
    chunks_path = kb_dir / "chunks.json"
    save_chunks(
        [
            DocumentChunk(text="placeholder retrieval content", source="placeholder.md", section=None, page=None, chunk_index=0),
        ],
        chunks_path,
    )

    module = _load_evaluate_tfidf_module()
    module._CHUNKS_PATH = chunks_path  # point the freshly loaded module at our temp corpus
    sys.modules["evaluate_tfidf_under_test"] = module

    module.main()  # would raise via the monkeypatch above if it touched load_directory()

    assert "Loaded 1 chunks" in capsys.readouterr().out
