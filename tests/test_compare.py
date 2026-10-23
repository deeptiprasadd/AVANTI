"""Tests for comparing two papers."""

from types import SimpleNamespace

from avanti import compare
from avanti.analyze import Finding, MetricHit
from avanti.references import parse_entry


def _analysis(title, body, datasets, metrics, phrases, refs, objective=""):
    brief = SimpleNamespace(
        objective=[Finding("objective", objective)] if objective else [],
        contribution=[Finding("contribution", f"We propose {title}.")],
        limitation=[],
        datasets=datasets,
        metrics=[MetricHit(name=n, value=v, unit="", sentence="", page=1)
                 for n, v in metrics],
        phrases=phrases,
    )
    return SimpleNamespace(
        title=title,
        doc=SimpleNamespace(author_lines=["A. Author"], pages_processed=9,
                            word_count=len(body.split())),
        paper=SimpleNamespace(body=body),
        brief=brief,
        references=[parse_entry(r) for r in refs],
    )


TRANSFORMER = _analysis(
    "Attention Is All You Need",
    "attention translation encoder decoder sequence model BLEU WMT " * 30,
    ["WMT", "Penn Treebank"],
    [("BLEU", "28.4")],
    ["attention", "decoder", "translation"],
    ["[1] D. Bahdanau. Neural machine translation by jointly learning to align and translate. ICLR, 2015.",
     "[2] D. Kingma and J. Ba. Adam: A method for stochastic optimization. ICLR, 2015."],
    objective="Recurrent models preclude parallelization.",
)

FOLLOW_UP = _analysis(
    "A Faster Transformer for Translation",
    "attention translation decoder speed latency BLEU WMT pruning " * 30,
    ["wmt", "IWSLT"],
    [("BLEU", "27.9"), ("latency", "12ms")],
    ["attention", "translation", "pruning"],
    ["[4] A. Vaswani et al. Attention is all you need. NeurIPS, 2017.",
     "[5] D. Bahdanau. Neural machine translation by jointly learning to align and translate. ICLR, 2015."],
)

UNRELATED = _analysis(
    "Protein Folding with Cryo-EM",
    "protein folding microscopy residue structure density map " * 30,
    ["PDB"],
    [("RMSD", "1.2")],
    ["protein", "folding"],
    ["[1] R. Henderson. Cryo-EM of proteins. Nature, 2013."],
)


def test_shared_datasets_and_metrics_line_up():
    c = compare.compare(TRANSFORMER, FOLLOW_UP)
    # Case-insensitive: "WMT" and "wmt" are the same benchmark.
    assert c.shared_datasets == ["WMT"]
    assert c.shared_metrics == [("BLEU", "28.4", "27.9")]
    assert set(c.shared_phrases) == {"attention", "translation"}


def test_prior_work_both_cite_is_found():
    c = compare.compare(TRANSFORMER, FOLLOW_UP)
    assert c.shared_references == [
        "Neural machine translation by jointly learning to align and translate"
    ]


def test_detects_which_paper_cites_the_other():
    c = compare.compare(TRANSFORMER, FOLLOW_UP)
    assert c.b_cites_a and not c.a_cites_b


def test_related_papers_score_higher_than_unrelated_ones():
    related = compare.compare(TRANSFORMER, FOLLOW_UP)
    unrelated = compare.compare(TRANSFORMER, UNRELATED)
    assert related.similarity > unrelated.similarity
    assert unrelated.relation == "Different topics"
    assert not unrelated.shared_datasets and not unrelated.shared_metrics


def test_sides_carry_each_papers_read_out():
    c = compare.compare(TRANSFORMER, FOLLOW_UP)
    assert c.a.problem == ["Recurrent models preclude parallelization."]
    assert c.b.problem == []
    assert c.a.reference_count == 2
    assert c.b.metrics == {"BLEU": "27.9", "latency": "12ms"}


def test_empty_text_has_zero_similarity():
    assert compare.similarity("", "anything at all") == 0.0
