"""Tests for the QASPER evaluation harness, on a tiny hand-made file."""

import json

from evaluation import qasper

_LIMITS = "A key limitation of our approach is its reliance on labelled data."
_METHOD = "We propose a sparse attention pattern that grows linearly with length."
_PROBLEM = "Existing models are limited by the quadratic memory cost of attention."

_PAPER = {
    "abstract": "We study attention for long documents and report gains on two corpora.",
    "full_text": [
        {"section_name": "Introduction", "paragraphs": [_PROBLEM]},
        {"section_name": "Methods ::: Model", "paragraphs": [_METHOD]},
        {"section_name": "Discussion", "paragraphs": [_LIMITS]},
    ],
    "qas": [
        {"question": "What are the limitations?",
         "answers": [{"answer": {"unanswerable": False, "evidence": [_LIMITS]}}]},
        {"question": "What is the capital of France?",
         "answers": [{"answer": {"unanswerable": True, "evidence": []}}]},
        {"question": "How does the figure look?",
         "answers": [{"answer": {"unanswerable": False,
                                 "evidence": ["FLOAT SELECTED: Figure 1"]}}]},
    ],
}


def _file(tmp_path):
    path = tmp_path / "qasper.json"
    path.write_text(json.dumps({"p1": _PAPER}), encoding="utf-8")
    return str(path)


def test_only_answerable_text_questions_are_kept(tmp_path):
    (paper,) = qasper.load(_file(tmp_path))
    assert [q.text for q in paper.questions] == ["What are the limitations?"]
    gold = paper.questions[0].gold[0]
    assert paper.paragraphs[gold] == _LIMITS


def test_section_names_map_to_canonical_buckets(tmp_path):
    (paper,) = qasper.load(_file(tmp_path))
    assert paper.kinds == ["abstract", "introduction", "method", "discussion"]


def test_metrics_for_a_ranking():
    got = qasper.per_question([4, 2, 0, 1, 3], gold=[0, 3])
    assert got["MRR"] == 1 / 3
    assert got["hit@1"] == 0.0 and got["hit@3"] == 1.0
    assert got["recall@3"] == 0.5 and got["recall@5"] == 1.0


def test_paired_difference_is_the_mean_of_per_question_gaps():
    rows = {
        "a": [{"MRR": 1.0}, {"MRR": 0.5}, {"MRR": 0.5}, {"MRR": 1.0}],
        "b": [{"MRR": 0.5}, {"MRR": 0.5}, {"MRR": 0.0}, {"MRR": 0.5}],
    }
    mean, lo, hi = qasper.paired_difference(rows, "a", "b")
    assert mean == 0.375
    assert 0.0 < lo <= mean <= hi <= 0.5


def test_section_prior_finds_the_limitation_in_the_discussion(tmp_path):
    papers = qasper.load(_file(tmp_path))
    rows = qasper.evaluate(papers, qasper.build_systems(None))
    assert rows["AVANTI (full)"][0]["hit@1"] == 1.0
    assert set(rows) >= {"BM25 (plain)", "TF-IDF cosine", "AVANTI (full)"}
