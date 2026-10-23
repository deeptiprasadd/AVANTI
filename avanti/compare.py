"""Compare two analyzed papers side by side.

The structured read-out already says, for each paper, what problem it takes
on, what it proposes, what data it uses and what it reports. Comparing two
papers is then mostly lining those up -- plus the things only a pair can
show: how close the two are in topic, which datasets and metrics they share
(where their numbers can be read against each other), which prior work they
both build on, and whether one cites the other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import references as refs_mod


@dataclass
class Side:
    """One paper's column in a comparison."""

    title: str
    authors: str
    pages: int
    words: int
    problem: List[str]
    method: List[str]
    limitations: List[str]
    datasets: List[str]
    metrics: Dict[str, str]          # metric name -> first reported value
    reference_count: int


@dataclass
class Comparison:
    a: Side
    b: Side
    similarity: float                # 0..1 topical overlap of the two bodies
    shared_datasets: List[str] = field(default_factory=list)
    shared_metrics: List[Tuple[str, str, str]] = field(default_factory=list)
    shared_phrases: List[str] = field(default_factory=list)
    shared_references: List[str] = field(default_factory=list)
    a_cites_b: bool = False
    b_cites_a: bool = False

    @property
    def relation(self) -> str:
        """The similarity in words, for a reader rather than a statistician.

        Two documents give no corpus to learn which words are distinctive,
        so even unrelated papers in one field share their vocabulary: three
        deep-learning papers on translation, vision and optimization all
        scored 0.10-0.14 against each other. The bands are set for that.
        """
        if self.similarity >= 0.35:
            return "Closely related"
        if self.similarity >= 0.2:
            return "Related"
        if self.similarity >= 0.08:
            return "Same broad field"
        return "Different topics"


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _side(an) -> Side:
    b = an.brief
    metrics: Dict[str, str] = {}
    for m in b.metrics:
        metrics.setdefault(m.name, f"{m.value}{m.unit}")
    return Side(
        title=an.title,
        authors=(an.doc.author_lines or [""])[0],
        pages=an.doc.pages_processed,
        words=an.doc.word_count,
        problem=[f.text for f in b.objective[:2]],
        method=[f.text for f in b.contribution[:2]],
        limitations=[f.text for f in b.limitation[:2]],
        datasets=list(b.datasets),
        metrics=metrics,
        reference_count=len(an.references),
    )


def similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity of two texts' TF-IDF vectors, 0..1."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    try:
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                              ngram_range=(1, 2), min_df=1)
        m = vec.fit_transform([text_a, text_b])
    except ValueError:
        return 0.0
    return float(cosine_similarity(m[0], m[1])[0, 0])


def _cites(citing, cited_title: str) -> bool:
    """Whether a paper's bibliography contains a work with this title."""
    target = _norm(cited_title)
    if len(target.split()) < 3:
        return False
    return any(
        target in _norm(r.title) or (r.title and _norm(r.title) in target
                                     and len(_norm(r.title).split()) >= 3)
        for r in citing.references
    )


def compare(a, b) -> Comparison:
    """Compare two avanti.pipeline.Analysis objects."""
    side_a, side_b = _side(a), _side(b)

    lower_b = {d.lower(): d for d in side_b.datasets}
    shared_datasets = [d for d in side_a.datasets if d.lower() in lower_b]

    metrics_b = {k.lower(): (k, v) for k, v in side_b.metrics.items()}
    shared_metrics = [
        (name, value, metrics_b[name.lower()][1])
        for name, value in side_a.metrics.items()
        if name.lower() in metrics_b
    ]

    phrases_b = {p.lower() for p in b.brief.phrases}
    shared_phrases = [p for p in a.brief.phrases if p.lower() in phrases_b]

    shared_refs = [
        pair["a"].title or pair["a"].text[:120]
        for pair in refs_mod.shared(a.references, b.references)
    ]

    return Comparison(
        a=side_a,
        b=side_b,
        similarity=similarity(a.paper.body, b.paper.body),
        shared_datasets=shared_datasets,
        shared_metrics=shared_metrics,
        shared_phrases=shared_phrases,
        shared_references=shared_refs,
        a_cites_b=_cites(a, b.title),
        b_cites_a=_cites(b, a.title),
    )
