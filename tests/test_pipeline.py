"""Tests for the analysis pipeline.

Every stage here is a pure function over text, so it can be tested without a
PDF or a network call. These lock down the specific failures that made the
old version read papers badly -- interleaved columns, run-in phrases read as
headings, thresholds reported as results, duplicated summary sentences.
"""

from avanti import analyze, sections, summarize, textutil


# ----------------------------------------------------------- section split


def test_canonical_headings_are_recognized():
    cases = {
        "Abstract": "abstract",
        "1. Introduction": "introduction",
        "2 Related Work": "related work",
        "III. METHODOLOGY": "method",
        "IV. RESULTS": "results",
        "5 Conclusion": "conclusion",
        "References": "references",
        "Materials and Methods": "method",
    }
    for line, expected in cases.items():
        bucket, _ = sections.classify_heading(line)
        assert bucket == expected, f"{line!r} -> {bucket!r}, wanted {expected!r}"


def test_run_in_phrase_is_not_a_heading():
    """A bold phrase opening a paragraph must not split the paper.

    "Analysis of Layer Responses. Fig. 7 shows ..." starts with the word
    "Analysis" and previously became a spurious Results section.
    """
    bucket, _ = sections.classify_heading("Analysis of Layer Responses. Fig. 7 shows")
    assert bucket is None


def test_full_sentence_is_not_a_heading():
    line = "We propose a residual learning framework to ease the training of networks"
    assert sections.classify_heading(line)[0] is None


def test_numbering_depth_is_reported():
    assert sections.classify_heading("3. Deep Residual Learning")[1] == 1
    assert sections.classify_heading("3.1 Encoder and Decoder Stacks")[1] == 2


def test_depth_follows_marker_style_not_dot_count():
    """IEEE papers nest by marker style, and all of these have one dot.

    Roman numerals are top-level sections, letters are subsections, and
    "1)" marks a point inside one. Counting dots made all three depth 1,
    which promoted every subsection and bullet into its own section.
    """
    assert sections.classify_heading("II. RELATED WORK")[1] == 1
    assert sections.classify_heading("A. Econometric based methods")[1] == 2
    assert sections.classify_heading("1) Inventory data for stock-outs")[1] == 3


def test_ieee_subsections_do_not_become_sections():
    text = "\n".join([
        "I. INTRODUCTION",
        "Price elasticity measures how demand responds to a change in price.",
        "II. RELATED WORK",
        "Earlier studies rely on econometric models of consumer behaviour.",
        "A. Econometric based methods",
        "These approaches estimate demand curves from historical sales data.",
        "B. Machine-learning based methods",
        "More recent work applies gradient boosting to the same problem.",
        "1) Inventory data to account for stock-outs",
        "Stock-outs distort the observed demand signal considerably.",
        "III. CONCLUSION",
        "The proposed approach outperforms the econometric baselines tested.",
    ])
    paper = sections.analyze(text)
    labels = [s.label for s in paper.sections]

    assert "RELATED WORK" in labels
    assert "CONCLUSION" in labels
    # Subsections and list points stay inside their parent.
    assert "Econometric based methods" not in labels
    assert "Inventory data to account for stock-outs" not in labels
    assert len(paper.sections) <= 4


def test_subsections_do_not_split_their_parent():
    """An unrecognized subsection belongs to the section containing it."""
    text = "\n".join([
        "1. Introduction",
        "Neural networks are widely used for many practical tasks today.",
        "3. Deep Residual Learning",
        "We introduce a residual mapping that is easier to optimize in practice.",
        "3.1. Identity Mapping",
        "The identity shortcut adds no parameters to the model whatsoever.",
        "3.2. Network Architectures",
        "We evaluate plain and residual networks of comparable total depth.",
    ])
    paper = sections.analyze(text)
    # "3.1" and "3.2" must not each become their own section.
    labels = [s.label for s in paper.sections]
    assert "Identity Mapping" not in labels
    assert "Network Architectures" not in labels
    assert "Deep Residual Learning" in labels


def test_body_excludes_front_and_back_matter():
    text = "\n".join([
        "Abstract",
        "This paper studies something interesting in considerable depth.",
        "1. Introduction",
        "The introduction explains the background of the problem at hand.",
        "References",
        "[1] Someone. A paper title. Journal, 2020.",
    ])
    paper = sections.analyze(text)
    body = paper.body
    assert "introduction explains" in body
    assert "Someone" not in body          # bibliography excluded
    assert "studies something" not in body  # abstract excluded


# ------------------------------------------------------------- text utils


def test_sentence_split_keeps_abbreviations_intact():
    text = "We used a CNN, e.g. ResNet, for this task. It worked well overall."
    sents = textutil.sent_tokenize(text, min_chars=10)
    assert len(sents) == 2
    assert "e.g. ResNet" in sents[0]


def test_citations_are_stripped():
    out = textutil.strip_citations(
        "CNNs work well [1], [2,3] as shown in Figure 4 by Smith et al., 2020."
    )
    assert "[1]" not in out
    assert "Figure 4" not in out
    assert "CNNs work well" in out


def test_orphan_numbers_are_removed():
    """A stray table cell glued to a sentence must be dropped.

    Borderless tables have no ruling lines for a detector to find, so a cell
    value can end up in front of real prose.
    """
    out = textutil.strip_orphan_numbers("93.3 In rows (B), we observe a drop.")
    assert out.startswith("In rows")


def test_table_rows_are_rejected_as_prose():
    assert not textutil.looks_like_prose("N d d h P 6 512 8 0.1 4.92 25.8")
    assert textutil.looks_like_prose(
        "We trained the model on eight GPUs for one hundred thousand steps."
    )


def test_reference_entries_are_rejected():
    assert textutil.looks_like_reference(
        "Smith, J. and Doe, A. Some title. Journal of Things, 2020."
    )
    assert not textutil.looks_like_reference(
        "The proposed model improves accuracy on the benchmark considerably."
    )


# --------------------------------------------------------------- analysis


def test_metrics_are_extracted_with_values():
    text = (
        "Our model reaches an accuracy of 94.2% on the test set. "
        "We also report an F1 of 0.91 across all folds of the data. "
        "The BLEU score of 28.4 exceeds the previous best published result."
    )
    hits = {h.name.lower(): h.value for h in analyze.find_metrics(text)}
    assert hits.get("accuracy") == "94.2"
    assert hits.get("bleu") == "28.4"


def test_threshold_is_not_reported_as_a_result():
    """"mAP @ IoU = 0.5" states an evaluation setting, not a score."""
    text = (
        "We evaluate using the PASCAL VOC metric (mAP @ IoU = 0.5) throughout "
        "all of the detection experiments reported in this section."
    )
    assert not [h for h in analyze.find_metrics(text) if h.name.lower() == "map"]


def test_probability_parameter_is_not_a_pvalue():
    """A bare "P = 0.1" is usually a dropout rate, not a significance test."""
    text = (
        "We apply dropout to the output of each sub-layer with P = 0.1 for the "
        "base model configuration used in all of these experiments."
    )
    assert not [h for h in analyze.find_metrics(text) if h.name == "p-value"]


def test_real_pvalue_is_found():
    text = (
        "The improvement over the baseline was statistically significant "
        "with p < 0.01 across every one of the evaluation folds."
    )
    assert [h for h in analyze.find_metrics(text) if h.name == "p-value"]


def test_known_datasets_are_recognized():
    text = (
        "We evaluate on ImageNet and CIFAR-10, and additionally report "
        "detection results on the COCO benchmark for completeness."
    )
    found = analyze.find_datasets(text)
    assert "ImageNet" in found
    assert "CIFAR-10" in found
    assert "COCO" in found


def test_cue_extraction_finds_the_contribution():
    text = (
        "Prior approaches rely on recurrence and therefore train slowly. "
        "In this work we propose the Transformer, a new architecture based "
        "entirely on attention mechanisms without any recurrence at all. "
        "The remainder of this section describes the experimental protocol."
    )
    found = analyze.find_by_cues(text, "contribution", limit=2)
    assert found
    assert "we propose the Transformer" in found[0].text


def test_objective_and_contribution_do_not_duplicate():
    """The thesis sentence matches both cue sets; it must appear once."""
    text = (
        "In this work we propose a new framework for estimating demand. "
        "We address the problem of sparse observations in retail data. "
        "The model is trained on a large corpus of historical transactions."
    )
    b = analyze.brief(text, {"introduction": text, "method": text})
    overlap = {f.text for f in b.objective} & {f.text for f in b.contribution}
    assert not overlap


# -------------------------------------------------------------- summarize


def test_summary_has_no_duplicate_sentences():
    """A repeated sentence must not survive, however salient it is."""
    repeated = "Deep learning is central to modern medical image analysis today."
    text = " ".join([
        repeated,
        "Our model was trained on twelve thousand annotated chest radiographs.",
        "We report an accuracy of 94.2 percent on the held-out test partition.",
        repeated,
        "A key limitation is that all data came from a single hospital system.",
        repeated,
    ])
    for diversity in (0.0, 0.45, 0.9):
        out = summarize.summarize_text(text, target_words=200, diversity=diversity)
        assert out.lower().count(repeated.lower()[:40]) <= 1


def test_summary_respects_its_word_budget():
    text = " ".join(
        f"This is sentence number {i} about an entirely distinct research topic "
        f"with its own specific vocabulary and terminology involved."
        for i in range(40)
    )
    out = summarize.summarize_text(text, target_words=60)
    # Selection stops once the budget is met, so allow one sentence of slop.
    assert len(out.split()) <= 110


def test_long_section_gets_more_budget_than_a_short_one():
    """Budget must track content volume, not just the section's label.

    A 190-word "Implementation" section previously outranked a 2,600-word
    "Experiments" section purely because of its canonical weight.
    """
    long_text = " ".join(
        f"Experimental finding {i} describes a distinct measured outcome "
        f"observed under carefully controlled evaluation conditions."
        for i in range(60)
    )
    short_text = " ".join(
        f"Implementation note {i} mentions a configuration detail briefly."
        for i in range(6)
    )
    text = (
        "3. Implementation\n" + short_text
        + "\n4. Experiments\n" + long_text
    )
    paper = sections.analyze(text)
    summary = summarize.summarize_paper(paper, target_words=300)
    by_name = {s.name: s.word_count for s in summary.sections}
    assert by_name.get("experiments", 0) > by_name.get("method", 0)
