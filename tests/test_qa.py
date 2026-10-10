"""Tests for question answering over a paper.

Retrieval is lexical, so the risk is a question whose words the paper never
uses. The suggested questions are the worst case: they are phrased the same
way for every paper ("What problem does this paper address?"), and papers
almost never say "problem" or "address" -- they say "limited by", "a
drawback", "we introduce". These lock down that every suggestion still finds
its answer.
"""

from avanti import qa

# A paper's worth of sentences that answer every suggested question without
# repeating any of the suggestions' own content words.
_SENTENCES = [
    "Existing vision transformers are limited by the quadratic cost of "
    "self attention at high resolution.",
    "We introduce dilated neighborhood attention, which expands the "
    "receptive field at no extra cost.",
    "We evaluate on ImageNet, COCO and ADE20K using top-1 accuracy and "
    "mean IoU.",
    "Our network outperforms Swin Transformer by 1.5 points on object "
    "detection.",
    "A drawback of our design is that the custom kernels require extra "
    "engineering effort.",
    "Compared with earlier convolutional networks, the hierarchy is "
    "learned from data rather than fixed by hand.",
    "The training schedule uses a cosine learning rate with linear warmup "
    "for twenty epochs.",
]


def _index() -> qa.PaperIndex:
    return qa.PaperIndex(
        [qa.Passage(text=s, page=1, section="") for s in _SENTENCES]
    )


class _AllSections:
    """A stand-in Paper whose sections unlock every suggested question."""

    found = ["experiments", "results", "discussion", "related work"]


def test_every_suggested_question_finds_an_answer():
    index = _index()
    suggestions = qa.suggested_questions(_AllSections())
    assert len(suggestions) == 6
    for question in suggestions:
        assert index.ask(question), f"no answer for {question!r}"


def test_problem_question_ranks_the_problem_statement_first():
    """The paper's own drawback is a limitation, not the problem it solves."""
    answers = _index().ask("What problem does this paper address?")
    assert answers
    assert "limited by the quadratic cost" in answers[0].text


def test_unrelated_question_still_finds_nothing():
    """Expansion must not turn every question into a match."""
    assert _index().ask("What is a good recipe for banana bread?") == []
