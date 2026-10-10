"""Split a paper into its standard sections.

The old code only ever found "Abstract" and "References" and summarized
everything in between as one undifferentiated blob. Knowing where the
methods end and the results begin is what makes the rest of the analysis
possible: you can budget the summary per section, search within a section,
and pull limitations from the discussion rather than the introduction.

Headings in papers come in many shapes -- "3 Method", "III. METHODOLOGY",
"3.1. Experimental Setup", or a bare "Discussion" -- so each candidate line
is matched against canonical patterns and mapped into one standard bucket.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Canonical buckets, in the order they normally appear in a paper.
CANONICAL_ORDER = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "acknowledgements",
    "references",
    "appendix",
]

# Which words map into which bucket. Checked longest-first per bucket.
_ALIASES: Dict[str, Tuple[str, ...]] = {
    "abstract": ("abstract", "summary"),
    "introduction": ("introduction", "overview"),
    "background": ("background", "preliminaries", "notation", "problem statement",
                   "problem formulation", "motivation"),
    "related work": ("related work", "related works", "literature review",
                     "prior work", "state of the art", "previous work"),
    "method": ("method", "methods", "methodology", "approach", "proposed method",
               "proposed approach", "model", "architecture", "our method",
               "materials and methods", "framework", "system design",
               "implementation", "algorithm", "model architecture"),
    "experiments": ("experiment", "experiments", "experimental setup",
                    "experimental results", "evaluation", "experimental design",
                    "dataset", "datasets", "data", "setup", "training details",
                    "training"),
    "results": ("result", "results", "findings", "performance",
                "ablation", "ablation study", "ablation studies",
                "comparison", "quantitative results"),
    "discussion": ("discussion", "limitations", "threats to validity",
                   "future work", "interpretation"),
    "conclusion": ("conclusion", "conclusions", "concluding remarks",
                   "conclusion and future work", "summary and conclusion"),
    "acknowledgements": ("acknowledgement", "acknowledgements",
                         "acknowledgment", "acknowledgments", "funding"),
    "references": ("references", "bibliography", "works cited"),
    "appendix": ("appendix", "appendices", "supplementary material",
                 "supplementary", "supplemental material"),
}

# A numbering prefix: "3", "3.", "3.1", "III.", "A.", "(3)"
_NUMBER_PREFIX = re.compile(
    r"^\s*(?:"
    r"\(?\d+(?:\.\d+)*\)?[.)]?"          # 3, 3., 3.1, (3)
    r"|[IVXLC]{1,5}[.)]"                 # III.
    r"|[A-H][.)]"                        # A.
    r")\s+"
)

# An internal sentence break: what separates a heading from a bold run-in
# phrase that opens a paragraph ("Analysis of Layer Responses. Fig. 7 shows").
_INTERNAL_BREAK = re.compile(r"\.\s+\S")


# Markers whose nesting level is set by style rather than by dot count.
_ROMAN_MARKER = re.compile(r"^[IVXLC]{1,5}[.)]$")
_LETTER_MARKER = re.compile(r"^[A-H][.)]$")
_PAREN_NUMBER = re.compile(r"^\(?\d{1,2}\)$")


def _strip_number(line: str) -> Tuple[str, bool, int]:
    """Remove a leading section number.

    Returns (rest, had_number, depth), where depth is the nesting level.

    Depth comes from the *style* of the marker, not from counting dots. An
    IEEE-format paper numbers its top-level sections with Roman numerals,
    its subsections with letters, and the points inside them with "1)" --
    all of which have a single dot and would otherwise read as depth 1.
    That promoted every subsection and bullet to a top-level section: one
    six-page paper came out with eighteen of them.
    """
    m = _NUMBER_PREFIX.match(line)
    if not m:
        return line.strip(), False, 0

    token = m.group(0).strip()
    rest = line[m.end():].strip()

    if _ROMAN_MARKER.match(token):        # I.  II.  III.   -> top level
        depth = 1
    elif _LETTER_MARKER.match(token):     # A.  B.  C.      -> subsection
        depth = 2
    elif _PAREN_NUMBER.match(token):      # 1)  2)  (3)     -> point in a list
        depth = 3
    else:                                 # 3 / 3. / 3.1 / 3.1.2
        depth = len([p for p in token.strip("().").split(".") if p])

    return rest, True, max(1, depth)


def classify_heading(line: str) -> Tuple[Optional[str], int]:
    """Classify a line as a section heading.

    Returns (bucket, depth). A bucket of None means the line is not a
    heading; "other" means it is clearly a heading but not one we recognize.
    """
    raw = line.strip()
    if not raw or len(raw) > 90:
        return None, 0

    body, numbered, depth = _strip_number(raw)
    if not body:
        return None, 0

    # A heading is short and does not read as a sentence.
    if len(body.split()) > 9:
        return None, 0

    # A real heading has no internal sentence break.
    if _INTERNAL_BREAK.search(body):
        return None, 0

    if body.endswith((".", ",", ";")) and not numbered:
        return None, 0

    probe = re.sub(r"\s+", " ", body.lower().strip(" .:;-"))

    for bucket in CANONICAL_ORDER:
        for alias in sorted(_ALIASES[bucket], key=len, reverse=True):
            if probe == alias or probe == alias + "s":
                return bucket, depth
            # Allow a few trailing words ("results and discussion"), but
            # never a whole sentence that merely starts with the alias.
            if probe.startswith(alias + " "):
                if len(probe[len(alias):].split()) <= 4:
                    return bucket, depth

    # Clearly formatted, but not a section we know by name.
    if numbered and 1 <= len(body.split()) <= 7 and body[:1].isupper():
        return "other", depth

    return None, 0


@dataclass
class Section:
    """One section of the paper."""

    name: str                    # canonical bucket, or "other"
    heading: str                 # the heading text as printed
    text: str = ""
    start_line: int = 0

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def label(self) -> str:
        """What to show a reader: the real heading, minus its numbering."""
        body, _, _ = _strip_number(self.heading)
        return (body or self.heading).strip(" .:;-") or self.name


@dataclass
class Paper:
    """A paper split into sections."""

    sections: List[Section] = field(default_factory=list)
    unsectioned: str = ""        # text before the first heading

    def get(self, name: str) -> Optional[Section]:
        """First section matching a canonical name."""
        for s in self.sections:
            if s.name == name:
                return s
        return None

    def text_of(self, *names: str) -> str:
        """Concatenated text of the named sections."""
        return "\n\n".join(s.text for s in self.sections if s.name in names)

    @property
    def found(self) -> List[str]:
        """Canonical names detected, in document order, without duplicates."""
        seen, out = set(), []
        for s in self.sections:
            if s.name not in seen:
                seen.add(s.name)
                out.append(s.name)
        return out

    @property
    def body(self) -> str:
        """Everything except front matter and back matter.

        This is what should be summarized: it excludes the abstract (already
        a summary), the bibliography, and the appendices.
        """
        skip = {"abstract", "references", "acknowledgements", "appendix"}
        parts = [s.text for s in self.sections if s.name not in skip]
        return "\n\n".join(p for p in parts if p.strip()) or self.unsectioned


def split(text: str) -> Paper:
    """Split a paper's text into sections."""
    lines = text.split("\n")
    paper = Paper()

    marks: List[Tuple[int, str, str]] = []  # (line index, bucket, heading)
    for i, ln in enumerate(lines):
        bucket, depth = classify_heading(ln)
        if not bucket:
            continue
        # An unrecognized *sub*-heading ("3.1 Residual Learning") belongs to
        # the section it sits inside, not a new one. Treating these as their
        # own sections collapsed most of a paper into a single blob.
        if bucket == "other" and depth >= 2:
            continue
        marks.append((i, bucket, lines[i].strip()))

    if not marks:
        paper.unsectioned = text.strip()
        return paper

    paper.unsectioned = "\n".join(lines[: marks[0][0]]).strip()

    for idx, (line_no, bucket, heading) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        body = "\n".join(lines[line_no + 1 : end]).strip()
        if body:
            paper.sections.append(
                Section(name=bucket, heading=heading, text=body, start_line=line_no)
            )

    return paper


def merge_duplicates(paper: Paper) -> Paper:
    """Combine sections that landed in the same canonical bucket."""
    out = Paper(unsectioned=paper.unsectioned)
    by_name: Dict[str, Section] = {}

    for s in paper.sections:
        # Unrecognized sections keep their own identity -- merging them would
        # fuse unrelated parts of the paper together.
        if s.name == "other":
            out.sections.append(Section(s.name, s.heading, s.text, s.start_line))
            continue
        if s.name in by_name:
            by_name[s.name].text += "\n\n" + s.text
        else:
            new = Section(s.name, s.heading, s.text, s.start_line)
            by_name[s.name] = new
            out.sections.append(new)

    out.sections.sort(key=lambda s: s.start_line)
    return out


def analyze(text: str) -> Paper:
    """Split and normalize in one call."""
    return merge_duplicates(split(text))
