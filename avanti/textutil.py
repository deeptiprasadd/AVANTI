"""Shared sentence-level text utilities."""

from __future__ import annotations

import re
from typing import List

# Abbreviations that must not be treated as a sentence end. Each lookbehind
# must include the abbreviation's own trailing period: the split point sits
# immediately after that period, so a pattern without it never matches and
# the guard silently does nothing ("e.g. ResNet" would split in two).
_ABBREV = (
    r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bcf\.)(?<!\bvs\.)(?<!\betc\.)"
    r"(?<!\bal\.)(?<!\bFig\.)(?<!\bFigs\.)(?<!\bEq\.)(?<!\bEqs\.)"
    r"(?<!\bNo\.)(?<!\bvol\.)(?<!\bpp\.)(?<!\bDr\.)(?<!\bProf\.)"
    r"(?<!\bSt\.)(?<!\bTab\.)(?<!\bSec\.)(?<!\bref\.)(?<!\bapprox\.)"
)

_SENT_SPLIT = re.compile(_ABBREV + r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9(\[])")


def normalize_spaces(text: str) -> str:
    """Collapse whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def sent_tokenize(text: str, min_chars: int = 30) -> List[str]:
    """Split text into sentences.

    Line wrapping inside a PDF paragraph is joined first, so a sentence that
    spans three physical lines still comes out as one sentence.
    """
    text = text.replace("\r", " ")
    text = re.sub(r"-\s*\n\s*", "", text)     # de-hyphenate across breaks
    text = re.sub(r"\s*\n\s*", " ", text)     # unwrap lines
    parts = _SENT_SPLIT.split(text)
    return [normalize_spaces(p) for p in parts if len(p.strip()) >= min_chars]


def strip_citations(sentence: str) -> str:
    """Remove citation markers and figure/table pointers from a sentence."""
    s = sentence
    s = re.sub(r"\[[0-9,\s\-;]{1,24}\]", " ", s)                  # [1], [2,3], [1-5]
    s = re.sub(r"\((?:[A-Z][A-Za-z\-]+(?:\s+(?:et\s+al\.?|and|&)\s*"
               r"[A-Za-z\-]*)*,?\s*\d{4}[a-z]?(?:;\s*)?)+\)", " ", s)
    s = re.sub(r"\(\d{4}[a-z]?\)", " ", s)                        # (2020)
    s = re.sub(r"\b(?:Figures?|Figs?\.?|Tables?|Eqs?\.?|Equations?|Sections?|Secs?\.?)"
               r"\s*\d+[a-z]?\b", " ", s, flags=re.I)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return normalize_spaces(s)


def looks_like_reference(sentence: str) -> bool:
    """True if a sentence looks like a bibliography entry, not prose."""
    s = sentence.lower()
    if any(x in s for x in ("doi.org", "http://", "https://", "arxiv:",
                            "isbn", " pp. ", "preprint")):
        return True
    # "Smith, J. A. (2020)" style author-year runs
    if re.search(r"[A-Z][a-z]+,\s+[A-Z]\.", sentence) and re.search(r"\b(19|20)\d{2}\b", s):
        return True
    # Mostly initials and years, little prose
    letters = sum(c.isalpha() for c in sentence)
    if letters and sum(c.isdigit() for c in sentence) / max(1, letters) > 0.35:
        return True
    return False


# A token that is only a number, possibly with punctuation: "93.3", "(4)".
_ORPHAN_NUMBER = re.compile(r"^[\d.,%()\[\]:;=<>+/-]+$")


def strip_orphan_numbers(sentence: str) -> str:
    """Drop stray numeric tokens at a sentence's edges.

    Borderless tables have no ruling lines for a table detector to find, so
    a stray cell value can end up glued to the front of a real sentence
    ("93.3 In rows (B), we observe that ..."). The sentence itself is sound
    once the orphan number is removed.
    """
    tokens = sentence.split()
    while tokens and _ORPHAN_NUMBER.match(tokens[0]):
        tokens.pop(0)
    while tokens and _ORPHAN_NUMBER.match(tokens[-1]):
        tokens.pop()
    return " ".join(tokens)


def looks_like_prose(sentence: str) -> bool:
    """True if a sentence reads like prose rather than flattened table rows."""
    tokens = sentence.split()
    if len(tokens) < 5:
        return False

    numeric = sum(1 for t in tokens if _ORPHAN_NUMBER.match(t))
    if numeric / len(tokens) > 0.22:
        return False

    # A run of short capitalized tokens is a table header, not a sentence.
    run = 0
    for t in tokens:
        if t[:1].isupper() and len(t) <= 4:
            run += 1
            if run >= 4:
                return False
        else:
            run = 0

    lower = sum(1 for t in tokens if t.islower())
    return lower / len(tokens) >= 0.35


def clean_sentences(text: str, min_chars: int = 30) -> List[str]:
    """Sentences with citation noise, bibliography and table debris removed."""
    out: List[str] = []
    for s in sent_tokenize(text, min_chars=min_chars):
        if looks_like_reference(s):
            continue
        cleaned = strip_orphan_numbers(strip_citations(s))
        if len(cleaned) < min_chars or not looks_like_prose(cleaned):
            continue
        out.append(cleaned)
    return out


def to_paragraphs(sentences: List[str], per_para: int = 3) -> str:
    """Group sentences into readable paragraphs."""
    paras = [
        " ".join(sentences[i : i + per_para])
        for i in range(0, len(sentences), per_para)
    ]
    return "\n\n".join(p for p in paras if p.strip())
