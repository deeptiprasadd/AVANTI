"""Ask questions about a paper and get answers with page citations.

This is retrieval, not generation: the answer is always sentences the paper
actually contains, each labelled with its page and section. That means no
API key, no hallucination, and an answer the reader can verify against the
PDF in front of them.

The index is built once per document and cached, so follow-up questions are
effectively instant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .textutil import normalize_spaces, sent_tokenize

# Words a question carries that say nothing about which sentence answers it.
_QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
    "should", "the", "a", "an", "of", "in", "on", "for", "to", "and", "or",
    "this", "that", "these", "those", "paper", "authors", "study", "they",
    "it", "its", "their", "there", "about", "tell", "me", "explain", "describe",
}


@dataclass
class Passage:
    """One indexed sentence and where it sits in the paper."""

    text: str
    page: int
    section: str


@dataclass
class Answer:
    """A retrieved passage with its relevance score."""

    text: str
    page: int
    section: str
    score: float


class PaperIndex:
    """A searchable index over one paper's sentences."""

    def __init__(self, passages: List[Passage]):
        self.passages = passages
        self._vec: Optional[TfidfVectorizer] = None
        self._matrix = None

        if passages:
            try:
                self._vec = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                )
                self._matrix = self._vec.fit_transform(
                    [p.text for p in passages]
                )
            except ValueError:
                self._vec, self._matrix = None, None

    def __len__(self) -> int:
        return len(self.passages)

    @property
    def ready(self) -> bool:
        return self._matrix is not None and len(self.passages) > 0

    def ask(self, question: str, k: int = 5, min_score: float = 0.015) -> List[Answer]:
        """Return the k passages that best answer the question."""
        if not self.ready or not question.strip():
            return []

        query = self._expand(question)
        try:
            qv = self._vec.transform([query])
        except ValueError:
            return []

        scores = cosine_similarity(qv, self._matrix)[0]
        order = scores.argsort()[::-1]

        answers: List[Answer] = []
        seen_text = set()
        for idx in order:
            score = float(scores[idx])
            if score < min_score or len(answers) >= k:
                break
            p = self.passages[idx]
            fingerprint = p.text[:70].lower()
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)
            answers.append(
                Answer(text=p.text, page=p.page, section=p.section, score=score)
            )
        return answers

    @staticmethod
    def _expand(question: str) -> str:
        """Strip question scaffolding so content words dominate the match."""
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]+|\d+", question.lower())
        content = [t for t in tokens if t not in _QUESTION_WORDS]
        # Repeat content words so they outweigh any scaffolding left behind.
        return " ".join(content * 2) if content else question


def build_index(doc, paper) -> PaperIndex:
    """Build a PaperIndex from an extracted Document and its sections.

    `doc` is an avanti.extract.Document, `paper` an avanti.sections.Paper.
    Page numbers come from the document; section labels from the split.
    """
    # Map a character offset in the full text to its section label.
    spans: List[Tuple[int, int, str]] = []
    full = doc.text
    cursor = 0
    for s in paper.sections:
        # Locate each section's text in the document to recover its offsets.
        head = s.text[:80]
        if not head:
            continue
        found = full.find(head, cursor)
        if found == -1:
            found = full.find(head)
        if found == -1:
            continue
        spans.append((found, found + len(s.text), s.label))
        cursor = found + len(s.text)

    def section_at(offset: int) -> str:
        for start, end, label in spans:
            if start <= offset < end:
                return label
        return ""

    passages: List[Passage] = []
    for page_no, page_text in enumerate(doc.pages, start=1):
        page_offset = full.find(page_text) if page_text else -1
        for sent in sent_tokenize(page_text, min_chars=40):
            text = normalize_spaces(sent)
            if len(text.split()) < 6:
                continue
            offset = (
                full.find(text[:60], page_offset)
                if page_offset >= 0
                else -1
            )
            passages.append(
                Passage(
                    text=text,
                    page=page_no,
                    section=section_at(offset) if offset >= 0 else "",
                )
            )

    return PaperIndex(passages)


def suggested_questions(paper) -> List[str]:
    """Starter questions tailored to what the paper actually contains."""
    base = [
        "What problem does this paper address?",
        "What method or model do the authors propose?",
    ]
    found = set(paper.found)
    if {"experiments", "results"} & found:
        base.append("What datasets and metrics were used?")
        base.append("What were the main results?")
    if {"discussion", "conclusion"} & found:
        base.append("What are the limitations?")
    if "related work" in found:
        base.append("How does this compare to prior work?")
    return base[:6]
