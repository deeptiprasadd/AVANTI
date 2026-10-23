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


def test_limitations_prefer_the_discussion():
    """Same wording, different section: the discussion is where caveats live."""
    index = qa.PaperIndex([
        qa.Passage("A key limitation of prior systems is their reliance on labels.",
                   page=1, section="Introduction", kind="introduction"),
        qa.Passage("A key limitation of our approach is its reliance on labels.",
                   page=9, section="Discussion", kind="discussion"),
    ])
    assert index.ask("What are the limitations?")[0].section == "Discussion"


def test_a_data_question_finds_the_papers_own_dataset_names():
    index = qa.PaperIndex(
        [qa.Passage("We train every variant on ImageNet-1K for 300 epochs.", 4, ""),
         qa.Passage("Attention layers alternate between local and sparse patterns.", 3, "")],
        datasets=["ImageNet"],
    )
    answers = index.ask("What data was used?")
    assert answers and "ImageNet-1K" in answers[0].text


# ------------------------------------------------------------ build_index


def _doc(*pages):
    from avanti.extract import Document

    return Document(pages=list(pages), pages_processed=len(pages))


def test_every_passage_is_labelled_with_its_page_and_section():
    doc = _doc(
        "A Paper\nAbstract\nWe study how attention scales to very long documents.\n"
        "1 Introduction\nExisting methods are limited by their quadratic memory cost.",
        # A heading at the very top of a page used to be glued to the last
        # line of the page before, and lost.
        "2 Method\nWe propose a sparse attention pattern that grows linearly with length.",
    )
    by_text = {p.text[:20]: p for p in qa.build_index(doc).passages}
    intro = by_text["Existing methods are"]
    method = by_text["We propose a sparse "]
    assert (intro.page, intro.section, intro.kind) == (1, "Introduction", "introduction")
    assert (method.page, method.section, method.kind) == (2, "Method", "method")


def test_the_bibliography_is_not_indexed():
    doc = _doc(
        "1 Introduction\nWe extend layer normalization to sparse attention models.\n"
        "References\n[1] Jimmy Lei Ba and Geoffrey Hinton. Layer normalization "
        "for sparse attention models. arXiv preprint arXiv:1607.06450, 2016."
    )
    texts = [p.text for p in qa.build_index(doc).passages]
    assert texts and not any("arXiv" in t for t in texts)
