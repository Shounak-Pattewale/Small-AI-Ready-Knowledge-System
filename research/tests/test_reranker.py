"""Tests for evaluation/_reranker.py's cross-encoder second-stage reranking.

No real Hugging Face model is downloaded - rerank()/classify_change() are
exercised with a fake CrossEncoder standing in for sentence_transformers
(same pattern as tests/test_nli.py and tests/test_verifier.py).
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "_reranker.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("reranker_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reranker():
    return _load_module()


# --- classify_change: pure function ---


@pytest.mark.parametrize(
    "original_hit,reranked_hit,expected",
    [
        (False, True, "improved"),
        (True, False, "worsened"),
        (True, True, "unchanged"),
        (False, False, "unchanged"),
    ],
)
def test_classify_change(reranker, original_hit, reranked_hit, expected):
    assert reranker.classify_change(original_hit, reranked_hit) == expected


# --- fake chunk/SearchResult doubles (avoid importing knowledge_system just for a struct) ---


class _FakeChunk:
    def __init__(self, text, source, section, chunk_index):
        self.text = text
        self.source = source
        self.section = section
        self.chunk_index = chunk_index


class _FakeSearchResult:
    def __init__(self, chunk, score):
        self.chunk = chunk
        self.score = score


class _FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder - predict() returns fixed scores in order."""

    def __init__(self, scores):
        self._scores = scores
        self.last_pairs = None

    def predict(self, pairs):
        self.last_pairs = pairs
        return list(self._scores)


def _make_reranker_model(reranker, scores):
    model = reranker.RerankerModel.__new__(reranker.RerankerModel)  # bypass __init__ (loads real CrossEncoder)
    model.model = _FakeCrossEncoder(scores)
    return model


def _three_candidates():
    return [
        _FakeSearchResult(_FakeChunk("body A", "docA.md", "Section A", chunk_index=0), score=0.90),
        _FakeSearchResult(_FakeChunk("body B", "docB.md", "Section B", chunk_index=1), score=0.85),
        _FakeSearchResult(_FakeChunk("body C", "docC.md", "Section C", chunk_index=2), score=0.80),
    ]


# --- (question, passage) pairs built correctly, with section+body representation ---


def test_score_batch_receives_question_paired_with_each_passage(reranker):
    model = _make_reranker_model(reranker, scores=[0.1, 0.2, 0.3])
    model.score_batch("my question", ["passage 1", "passage 2", "passage 3"])
    assert model.model.last_pairs == [
        ("my question", "passage 1"),
        ("my question", "passage 2"),
        ("my question", "passage 3"),
    ]


def test_rerank_builds_section_plus_body_passage_text(reranker):
    model = _make_reranker_model(reranker, scores=[0.1, 0.1, 0.1])
    results = _three_candidates()
    reranker.rerank("q", results, model)
    passages = [pair[1] for pair in model.model.last_pairs]
    assert passages == ["Section A\n\nbody A", "Section B\n\nbody B", "Section C\n\nbody C"]


# --- deterministic descending score sort + chunk_index tie-break ---


def test_rerank_sorts_descending_by_reranker_score(reranker):
    model = _make_reranker_model(reranker, scores=[0.1, 0.9, 0.5])  # candidate B should win
    results = _three_candidates()
    ranked = reranker.rerank("q", results, model)
    assert [c.chunk.source for c in ranked] == ["docB.md", "docC.md", "docA.md"]


def test_rerank_breaks_ties_by_chunk_index_ascending(reranker):
    model = _make_reranker_model(reranker, scores=[0.5, 0.5, 0.5])  # all tied
    results = _three_candidates()
    ranked = reranker.rerank("q", results, model)
    assert [c.chunk.chunk_index for c in ranked] == [0, 1, 2]


# --- Granite candidate metadata preserved through reranking ---


def test_rerank_preserves_granite_rank_and_score(reranker):
    model = _make_reranker_model(reranker, scores=[0.1, 0.9, 0.5])
    results = _three_candidates()
    ranked = reranker.rerank("q", results, model)
    winner = ranked[0]  # was granite rank 2 (docB.md), granite_score 0.85
    assert winner.granite_rank == 2
    assert winner.granite_score == pytest.approx(0.85)
    assert winner.reranker_score == pytest.approx(0.9)


def test_rerank_does_not_change_the_candidate_set(reranker):
    model = _make_reranker_model(reranker, scores=[0.3, 0.1, 0.2])
    results = _three_candidates()
    ranked = reranker.rerank("q", results, model)
    assert {c.chunk.source for c in ranked} == {r.chunk.source for r in results}
    assert len(ranked) == len(results)


# --- no Granite/reranker score blending: reranker score alone decides order ---


def test_rerank_never_blends_granite_and_reranker_scores(reranker):
    # Granite ranked docA highest (0.90); reranker ranks docC highest (0.99).
    # If any blending occurred, docA (high granite score) could still win - it must not.
    model = _make_reranker_model(reranker, scores=[0.01, 0.01, 0.99])
    results = _three_candidates()
    ranked = reranker.rerank("q", results, model)
    assert ranked[0].chunk.source == "docC.md"


def test_reranked_score_is_exactly_the_reranker_score_not_a_formula(reranker):
    model = _make_reranker_model(reranker, scores=[0.42, 0.1, 0.1])
    results = _three_candidates()
    ranked = reranker.rerank("q", results, model)
    winner = ranked[0]
    assert winner.reranker_score == pytest.approx(0.42)
    assert winner.reranker_score != winner.granite_score  # not blended into a shared number


# --- fake reranker reorders Top-3 deterministically across repeated calls ---


def test_rerank_is_deterministic_across_repeated_calls(reranker):
    model = _make_reranker_model(reranker, scores=[0.2, 0.8, 0.5])
    results = _three_candidates()
    first = reranker.rerank("q", results, model)
    second = reranker.rerank("q", results, model)
    assert [c.chunk.source for c in first] == [c.chunk.source for c in second]


# --- evaluate_reranker.py: metric calculation + rank-movement classification ---

_SCRIPT_MODULE_PATH = Path(__file__).resolve().parent.parent.parent / "research" / "evaluation" / "evaluate_reranker.py"


@pytest.fixture(scope="module")
def evaluate_reranker():
    spec = importlib.util.spec_from_file_location("evaluate_reranker_under_test", _SCRIPT_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(source, section, score):
    return _FakeSearchResult(_FakeChunk("body", source, section, chunk_index=0), score=score)


def test_evaluate_candidate_set_top1_and_topn_hit_fields(evaluate_reranker):
    question = {
        "id": "q1",
        "expected_source": "docA.md",
        "expected_topic": "Section A",
    }
    results = [_result("docB.md", "Section B", 0.5), _result("docA.md", "Section A", 0.4)]
    row = evaluate_reranker.evaluate_candidate_set(question, results)
    # Top-1 is docB.md/Section B - neither the expected source nor topic.
    assert row["top1_source_hit"] is False
    assert row["top1_topic_hit"] is False
    # But the expected source/topic IS present somewhere in the full candidate set.
    assert row["topN_source_hit"] is True
    assert row["topN_topic_hit"] is True


def test_evaluate_candidate_set_is_depth_agnostic(evaluate_reranker):
    # A 5-item candidate list must check all 5 for topN, not just the first 3
    # (this is the exact bug this helper was written to avoid - see module docstring).
    question = {"id": "q1", "expected_source": "docE.md", "expected_topic": "Section E"}
    results = [_result(f"doc{c}.md", f"Section {c}", 0.5) for c in "ABCDE"]
    row = evaluate_reranker.evaluate_candidate_set(question, results)
    assert row["topN_source_hit"] is True
    assert row["topN_topic_hit"] is True


def test_rank_movement_counts_improved_unchanged_worsened(evaluate_reranker):
    original_rows = [
        {"top1_topic_hit": False},  # -> True below: improved
        {"top1_topic_hit": True},  # -> False below: worsened
        {"top1_topic_hit": True},  # -> True: unchanged
    ]
    reranked_rows = [
        {"top1_topic_hit": True},
        {"top1_topic_hit": False},
        {"top1_topic_hit": True},
    ]
    counts = evaluate_reranker.rank_movement(original_rows, reranked_rows, "top1_topic_hit")
    assert counts == {"improved": 1, "unchanged": 1, "worsened": 1}
