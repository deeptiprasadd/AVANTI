"""Ask questions about a paper and get answers with page citations.

This is retrieval, not generation: the answer is always sentences the paper
actually contains, each labelled with its page and section. That means no
API key, no hallucination, and an answer the reader can verify against the
PDF in front of them.

Ranking is BM25 over the paper's sentences, steered by what the question is
asking for. Lexical match alone is not enough: asked "What problem does
this paper address?", it ranked a stray "limited" in the method section
above the paper's actual problem statement. So a question is also read for
its *intent* -- problem, method, data, results, limitations, comparison --
and each intent prefers the sections that answer it (limitations live in
the discussion, not the introduction) and the phrasing that signals it
("we aim to", "we propose").

The index is built once per document and cached, so follow-up questions are
effectively instant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from . import analyze, sections
from .textutil import normalize_spaces, sent_tokenize

# Words a question carries that say nothing about which sentence answers it.
_QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
    "should", "the", "a", "an", "of", "in", "on", "for", "to", "and", "or",
    "this", "that", "these", "those", "paper", "authors", "study", "they",
    "it", "its", "their", "there", "about", "tell", "me", "explain", "describe",
    # Qualifiers a question adds that a paper uses for something else: "the
    # main results" matched "our main model" ahead of any actual result.
    "main", "key", "major", "primary", "overall", "used", "use", "using",
}

# Words a paper actually uses for what a question asks about. Papers rarely
# echo the question: "What problem does this address?" is answered by
# "existing methods are limited by..." -- a sentence sharing no words with
# it. Each question word pulls in its usual answers' vocabulary, at a lower
# weight than the asker's own words, so it widens the net without steering.
_RELATED = {
    # Not "drawback": a paper's drawbacks are its own limitations, not the
    # problem it sets out to solve -- that is usually other work's failing.
    "problem": "challenge limited lack suffer issue difficult difficulty gap "
               "bottleneck existing",
    "address": "tackle aim goal solve focus",
    "method": "propose proposed introduce present approach framework "
              "architecture algorithm technique",
    "model": "propose proposed introduce network architecture framework",
    "propose": "proposed introduce present approach framework",
    "contribution": "propose introduce present novel",
    "dataset": "benchmark corpus evaluate evaluated trained training",
    "metric": "accuracy precision recall score f1 auc bleu error iou map",
    "result": "outperform achieve improve improvement gain state-of-the-art",
    # Not "future": future-work sentences answered "What are the
    # limitations?" ahead of the limitation the paper actually admits.
    "limitation": "limited drawback weakness shortcoming fail restricted",
    "compare": "compared comparison baseline outperform previous existing "
               "earlier",
    "prior": "previous existing earlier baseline",
}

# What a question can be asking for. Each intent has the words that signal
# it, a boost for the sections that usually answer it, and the cue category
# (from analyze) whose phrasing marks an answering sentence.
_INTENTS: Dict[str, Tuple[Set[str], Dict[str, float], Optional[str]]] = {
    "problem": (
        {"problem", "challenge", "motivation", "motivate", "gap", "address",
         "issue", "aim", "goal", "objective", "purpose"},
        {"abstract": 1.6, "introduction": 1.6, "background": 1.2},
        "objective",
    ),
    "method": (
        {"method", "approach", "model", "propose", "proposed", "architecture",
         "technique", "algorithm", "framework", "contribution", "novel"},
        {"abstract": 1.4, "introduction": 1.25, "method": 1.5},
        "contribution",
    ),
    "data": (
        {"dataset", "data", "benchmark", "metric", "evaluate", "evaluation",
         "corpus", "measure"},
        {"experiments": 1.6, "results": 1.25, "method": 1.1},
        "dataset",
    ),
    "results": (
        {"result", "performance", "accuracy", "finding", "achieve", "score",
         "outperform", "improvement"},
        {"results": 1.5, "experiments": 1.3, "abstract": 1.3,
         "conclusion": 1.3},
        "result",
    ),
    "limitations": (
        {"limitation", "weakness", "drawback", "shortcoming", "future",
         "fail", "caveat"},
        {"discussion": 1.7, "conclusion": 1.5, "results": 1.1},
        "limitation",
    ),
    "comparison": (
        {"compare", "comparison", "prior", "previous", "baseline", "existing",
         "related", "versus"},
        {"related work": 1.5, "results": 1.3, "experiments": 1.2},
        None,
    ),
}

# Weight of a related (expansion) term relative to the asker's own words.
_RELATED_WEIGHT = 0.35

# Weight of intent-signalling phrasing relative to the best lexical match.
_CUE_WEIGHT = 0.8

# Sections whose sentences are not answers: a bibliography entry shares a
# lot of vocabulary with the paper and used to surface as one.
_SKIP_SECTIONS = {"references", "acknowledgements"}

_BIBLIOGRAPHIC = re.compile(
    r"\barXiv\b|\bdoi\b|https?://|\bIn Proceedings\b|\bProc\.|\bpp\.\s*\d",
    re.I,
)

_STOP = frozenset(ENGLISH_STOP_WORDS)


def _stem(word: str) -> str:
    """Fold plurals and third-person -s, the inflections that matter here.

    Deliberately crude: "datasets" must match "dataset" and "outperforms"
    must match "outperform", and anything cleverer starts merging words
    that mean different things.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _words(text: str) -> List[str]:
    return [
        _stem(w) for w in re.findall(r"[a-z0-9]{2,}", text.lower())
        if w not in _STOP
    ]


def _terms(text: str) -> List[str]:
    """Index terms: stemmed words plus adjacent pairs, so phrases score higher."""
    words = _words(text)
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


_RELATED_TERMS = {
    _stem(key): [_stem(w) for w in related.split()]
    for key, related in _RELATED.items()
}


@dataclass
class Passage:
    """One indexed sentence and where it sits in the paper."""

    text: str
    page: int
    section: str        # the heading as printed, for display
    kind: str = ""      # canonical section bucket, for ranking


@dataclass
class Answer:
    """A retrieved passage with its relevance score."""

    text: str
    page: int
    section: str
    score: float


class PaperIndex:
    """A searchable index over one paper's sentences."""

    K1 = 1.5
    B = 0.75

    def __init__(self, passages: List[Passage], datasets: Optional[List[str]] = None):
        self.passages = passages
        # Dataset names this paper mentions. "What data did they use?" shares
        # no words with "We train on ImageNet-1K", so a data question also
        # searches for the paper's own dataset names.
        self.datasets = datasets or []
        self._vec: Optional[CountVectorizer] = None
        self._tf = None
        self._cues: Dict[str, np.ndarray] = {}

        if passages:
            try:
                self._vec = CountVectorizer(analyzer=_terms)
                # Column-major, so each query term's postings are one slice.
                self._tf = self._vec.fit_transform(
                    [p.text for p in passages]
                ).tocsc()
            except ValueError:  # nothing indexable at all
                self._vec, self._tf = None, None

        if self._tf is not None:
            n = self._tf.shape[0]
            df = np.diff(self._tf.indptr)
            self._idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
            lengths = np.asarray(self._tf.sum(axis=1)).ravel()
            avg = max(float(lengths.mean()), 1e-9)
            self._norm = self.K1 * (1 - self.B + self.B * lengths / avg)

    def __len__(self) -> int:
        return len(self.passages)

    @property
    def ready(self) -> bool:
        return self._tf is not None and len(self.passages) > 0

    # ---------------------------------------------------------- scoring

    def _bm25(self, weights: Dict[str, float]) -> np.ndarray:
        scores = np.zeros(len(self.passages))
        vocab = self._vec.vocabulary_
        for term, weight in weights.items():
            j = vocab.get(term)
            if j is None:
                continue
            lo, hi = self._tf.indptr[j], self._tf.indptr[j + 1]
            rows, tf = self._tf.indices[lo:hi], self._tf.data[lo:hi]
            scores[rows] += (
                weight * self._idf[j] * tf * (self.K1 + 1) / (tf + self._norm[rows])
            )
        return scores

    def _cue_strength(self, kind: str) -> np.ndarray:
        """Per-passage cue score for one category, 0..1, computed once."""
        if kind not in self._cues:
            self._cues[kind] = np.array(
                [min(analyze.cue_score(p.text, kind), 3.0) / 3.0
                 for p in self.passages]
            )
        return self._cues[kind]

    @staticmethod
    def _query(question: str) -> Tuple[Dict[str, float], List[str]]:
        """Weighted query terms and detected intents for a question."""
        tokens = re.findall(r"[a-z0-9]{2,}", question.lower())
        content = [t for t in tokens if t not in _QUESTION_WORDS]
        stems = [_stem(t) for t in content]

        weights: Dict[str, float] = {}
        for stem in stems:
            for related in _RELATED_TERMS.get(stem, ()):
                weights[related] = _RELATED_WEIGHT
        # The asker's own words always outweigh the expansion.
        for term in _terms(" ".join(content)):
            weights[term] = 1.0

        intents = [
            name for name, (signals, _, _) in _INTENTS.items()
            if signals & set(stems)
        ]
        return weights, intents

    def ask(self, question: str, k: int = 5, min_score: float = 0.25) -> List[Answer]:
        """Return the k passages that best answer the question.

        Answers scoring below `min_score` times the best answer's score are
        dropped rather than shown as padding.
        """
        if not self.ready or not question.strip():
            return []

        weights, intents = self._query(question)
        if not weights:
            return []
        if "data" in intents:
            for name in self.datasets:
                for term in _words(name):
                    weights.setdefault(term, _RELATED_WEIGHT)

        lexical = self._bm25(weights)
        top = float(lexical.max())
        relevance = lexical / top if top > 0 else lexical

        boost = np.ones(len(self.passages))
        for name in intents:
            _, section_boost, cue_kind = _INTENTS[name]
            boost = np.maximum(
                boost,
                np.array([section_boost.get(p.kind, 1.0) for p in self.passages]),
            )
            if cue_kind:
                relevance = relevance + _CUE_WEIGHT * self._cue_strength(cue_kind)

        # A passage must share a word with the question, or -- for a question
        # whose intent is known -- be phrased as that kind of answer.
        if not np.any(relevance > 0):
            return []
        final = relevance * boost
        best = float(final.max())

        answers: List[Answer] = []
        seen_text = set()
        for idx in np.argsort(-final, kind="stable"):
            score = float(final[idx])
            if score <= 0 or score < min_score * best or len(answers) >= k:
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


def build_index(doc) -> PaperIndex:
    """Build a PaperIndex from an extracted Document.

    `doc` is an avanti.extract.Document. The text is walked line by line with
    both the page and the section known for every line, so each sentence is
    labelled exactly. (Locating sentences afterwards by string search failed
    for most of them: the sentence had been unwrapped and the text had not.)
    """
    lines = sections.split_lines(doc.text)
    breaks = re.findall(r"[\n\f]", doc.text)

    # Section of every line, from the unmerged split: merging sections
    # discards where each piece sat. Heading lines themselves get None.
    label: List[Optional[Tuple[str, str]]] = [("", "")] * len(lines)
    found = sections.split(doc.text).sections
    for i, s in enumerate(found):
        end = found[i + 1].start_line if i + 1 < len(found) else len(lines)
        label[s.start_line] = None
        for ln in range(s.start_line + 1, end):
            label[ln] = (s.name, s.label)

    # Group consecutive lines that share a page and a section.
    segments: List[Tuple[int, Tuple[str, str], List[str]]] = []
    page = 1
    for i, line in enumerate(lines):
        where = label[i]
        if where is not None and where[0] not in _SKIP_SECTIONS:
            if segments and segments[-1][0] == page and segments[-1][1] == where:
                segments[-1][2].append(line)
            else:
                segments.append((page, where, [line]))
        if i < len(breaks) and breaks[i] == "\f":
            page += 1

    passages: List[Passage] = []
    for page_no, (kind, heading), seg_lines in segments:
        for sent in sent_tokenize("\n".join(seg_lines), min_chars=40):
            text = normalize_spaces(sent)
            if len(text.split()) < 6 or _BIBLIOGRAPHIC.search(text):
                continue
            passages.append(
                Passage(text=text, page=page_no, section=heading, kind=kind)
            )

    return PaperIndex(passages, datasets=analyze.find_datasets(doc.text))


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
