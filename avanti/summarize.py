"""Section-aware extractive summarization.

Summarizing a whole paper as one pool of sentences lets the introduction
dominate, because introductions restate the paper's themes and therefore
score highest on TF-IDF. The result reads like an abstract of the abstract
and says nothing about what was actually done or measured.

Instead each section gets its own word budget, weighted by how much a reader
needs from it, and sentences are selected within the section. The method and
results sections then get the space they deserve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .textutil import clean_sentences, to_paragraphs

# Cosine similarity above which two sentences count as the same sentence.
# Without a hard ceiling, a repeated sentence that scores highly enough on
# relevance still gets admitted twice.
REDUNDANCY_CEILING = 0.92

# How much of the summary each section is worth to a reader. Sections absent
# from a paper have their share redistributed across the rest.
SECTION_WEIGHTS: Dict[str, float] = {
    "abstract": 0.00,          # already a summary; shown separately
    "introduction": 0.14,
    "background": 0.04,
    "related work": 0.05,
    "method": 0.28,
    "experiments": 0.14,
    "results": 0.22,
    "discussion": 0.08,
    "conclusion": 0.05,
    "other": 0.10,
    "references": 0.00,
    "acknowledgements": 0.00,
    "appendix": 0.00,
}


def select_sentences(
    sentences: List[str],
    target_words: int,
    diversity: float = 0.4,
) -> List[int]:
    """Pick sentence indices up to a word budget, using MMR.

    Salience is normalized to 0..1 so that it is comparable with cosine
    similarity; otherwise the two terms sit on different scales and the
    diversity setting has almost no effect.
    """
    if not sentences:
        return []
    if len(sentences) == 1:
        return [0]

    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                              max_df=0.9, sublinear_tf=True)
        X = vec.fit_transform(sentences)
    except ValueError:
        # Too little usable text to vectorize; fall back to document order.
        return list(range(min(len(sentences), max(1, target_words // 25))))

    energy = X.power(2).sum(axis=1).A1
    lo, hi = float(energy.min()), float(energy.max())
    sal = (energy - lo) / (hi - lo) if hi > lo else energy * 0 + 1.0

    order = sal.argsort()[::-1]
    selected: List[int] = []
    vectors: List = []
    words = 0

    for idx in order:
        if words >= target_words:
            break
        cand = X[idx]

        if vectors:
            red = max(
                (cosine_similarity(cand, v)[0, 0] for v in vectors), default=0.0
            )
            # A near-duplicate is never worth including, however salient.
            if red >= REDUNDANCY_CEILING:
                continue
            mmr = (1 - diversity) * float(sal[idx]) - diversity * float(red)
            if mmr <= 0.0:
                continue

        selected.append(int(idx))
        vectors.append(cand)
        words += len(sentences[idx].split())

    # A strict threshold can starve a short section; backfill with the next
    # most salient sentences that are not near-duplicates.
    if words < target_words * 0.5:
        chosen = set(selected)
        for idx in order:
            if words >= target_words or len(selected) >= len(sentences):
                break
            if int(idx) in chosen:
                continue
            cand = X[idx]
            red = max(
                (cosine_similarity(cand, v)[0, 0] for v in vectors), default=0.0
            )
            if red < REDUNDANCY_CEILING:
                selected.append(int(idx))
                vectors.append(cand)
                chosen.add(int(idx))
                words += len(sentences[idx].split())

    return sorted(selected)


def summarize_text(
    text: str, target_words: int = 200, diversity: float = 0.4
) -> str:
    """Summarize a single block of text into readable paragraphs."""
    sents = clean_sentences(text)
    if not sents:
        return ""
    picked = select_sentences(sents, target_words, diversity)
    return to_paragraphs([sents[i] for i in picked], per_para=3)


@dataclass
class SectionSummary:
    """The summary of one section."""

    name: str
    label: str
    text: str
    source_words: int = 0

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Summary:
    """A whole-paper summary, section by section."""

    sections: List[SectionSummary] = field(default_factory=list)
    abstract: str = ""
    key_points: List[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """The summary as one plain-text document, for download."""
        parts: List[str] = []
        if self.abstract:
            parts.append("ABSTRACT (as published)\n" + self.abstract)
        if self.key_points:
            parts.append(
                "KEY POINTS\n" + "\n".join(f"- {p}" for p in self.key_points)
            )
        for s in self.sections:
            if s.text:
                parts.append(s.label.upper() + "\n" + s.text)
        return "\n\n".join(parts)

    @property
    def word_count(self) -> int:
        return sum(s.word_count for s in self.sections)


def _allocate(present: Dict[str, int], total_words: int) -> Dict[str, int]:
    """Split the word budget across the sections a paper actually has."""
    weights = {
        name: SECTION_WEIGHTS.get(name, 0.06)
        for name in present
        if SECTION_WEIGHTS.get(name, 0.06) > 0
    }
    if not weights:
        return {}

    total_weight = sum(weights.values())
    budget: Dict[str, int] = {}
    for name, w in weights.items():
        share = int(total_words * w / total_weight)
        # Never budget more than the section actually contains, and give
        # every included section enough room for a couple of sentences.
        budget[name] = max(40, min(share, present[name]))
    return budget


def summarize_paper(
    paper, target_words: int = 450, diversity: float = 0.4
) -> Summary:
    """Summarize a sectioned paper, section by section.

    `paper` is an avanti.sections.Paper.
    """
    summary = Summary()

    abstract_section = paper.get("abstract")
    if abstract_section:
        # The abstract is already a summary -- show it verbatim, trimmed.
        text = " ".join(abstract_section.text.split())
        summary.abstract = text[:1600] + ("..." if len(text) > 1600 else "")

    summarizable = [
        s for s in paper.sections
        if SECTION_WEIGHTS.get(s.name, 0.06) > 0 and s.word_count > 60
    ]

    if not summarizable:
        # No usable structure -- summarize whatever body text exists.
        body = paper.body
        if body.strip():
            summary.sections.append(
                SectionSummary(
                    name="paper",
                    label="Summary",
                    text=summarize_text(body, target_words, diversity),
                    source_words=len(body.split()),
                )
            )
        summary.key_points = key_points(paper.body)
        return summary

    # Sections can repeat a canonical name ("other"); budget by position.
    present = {}
    for i, s in enumerate(summarizable):
        present[f"{s.name}#{i}"] = s.word_count

    # Blend the section's importance with how much text it actually holds.
    # Weight alone misallocates badly: a 190-word "Implementation" section
    # lands in the high-weight `method` slot and outranks a 2,658-word
    # "Experiments" section, so the summary covers the wrong ground. The
    # square root keeps the effect sub-linear, so a long section gets more
    # room without a short important one being squeezed out entirely.
    weights_by_key = {
        f"{s.name}#{i}": SECTION_WEIGHTS.get(s.name, 0.06) * (s.word_count ** 0.5)
        for i, s in enumerate(summarizable)
    }
    total_weight = sum(weights_by_key.values()) or 1.0

    for i, s in enumerate(summarizable):
        key = f"{s.name}#{i}"
        share = target_words * weights_by_key[key] / total_weight
        # No single section may dominate the whole summary.
        share = min(share, target_words * 0.35)
        budget = max(45, min(int(share), s.word_count))
        text = summarize_text(s.text, budget, diversity)
        if text:
            summary.sections.append(
                SectionSummary(
                    name=s.name,
                    label=s.label,
                    text=text,
                    source_words=s.word_count,
                )
            )

    summary.key_points = key_points(paper.body)
    return summary


def key_points(text: str, n: int = 6) -> List[str]:
    """The n most informative standalone sentences from the paper body."""
    sents = clean_sentences(text, min_chars=50)
    if not sents:
        return []
    # Budget by sentence count rather than words, then trim to n.
    picked = select_sentences(sents, target_words=n * 30, diversity=0.55)
    return [sents[i] for i in picked[:n]]
