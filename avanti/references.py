"""Read a paper's bibliography into structured entries.

Each entry comes out with its authors, title, year and -- where the entry
carries one -- an arXiv id or DOI, so it can be linked. Entries are then
matched against the paper's own in-text citations, which says which prior
work the paper leans on most: a reference cited eleven times is the work
being built on; one cited once is background.

Two citation styles cover nearly every paper:

  numbered     "[12] A. Author, B. Author. Title. Venue, 2019."
               cited in the text as [12], [3, 12] or [10-14]
  author-year  "Author, A. and Author, B. Title. Venue, 2019."
               cited in the text as (Author et al., 2019) or Author (2019)

The bibliography is read from `Document.full_text`, because most papers
set their references a size below the body text and the analysis text
drops small print.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import quote_plus

from . import sections
from .textutil import normalize_spaces

# "[12] ..." at the start of a line.
_NUMBERED = re.compile(r"^\s*\[(\d{1,3})\]\s*")

# An author-year entry opens with a surname and a comma: "Kingma, Diederik"
# or "Jacob Devlin, Ming-Wei Chang". A continuation line can start the same
# way ("He, Xiaodong, Williams, Jason"), so an entry only breaks where the
# previous line also ended one -- on a full stop.
_AUTHOR_START = re.compile(
    r"^[A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+){0,2},\s+[A-Z]"
)

_YEAR = re.compile(r"\b((?:19|20)\d{2})[a-z]?\b")
_ARXIV = re.compile(
    r"(?:arXiv:\s*|abs/|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?",
    re.I,
)
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)

# CVPR-style back-references: the pages that cite an entry, printed after it
# ("... (CVPR), 2017. 10, 11"). They are not part of the reference.
_BACKREF = re.compile(r"\.\s+\d{1,3}(?:,\s*\d{1,3})*\s*$")

# A segment break inside an entry is a full stop, except after a few
# abbreviations. ("et al." is a break: in a bibliography it ends the
# author list.) Initials are judged separately -- see _segments.
_NOT_A_BREAK = {"jr", "vol", "pp", "eds", "ed", "no", "dr", "st"}

# "E", "G.E", "R.R": an initial, as it reads before its final full stop.
_INITIAL = re.compile(r"^(?:[A-Z]\.)*[A-Z]$")

# An entry this long has swallowed whatever followed the bibliography --
# appendix figures, usually.
_MAX_ENTRY = 600


@dataclass
class Reference:
    """One bibliography entry."""

    text: str                    # the entry as printed, whitespace-normalized
    number: Optional[int] = None # [n] in numbered styles
    authors: str = ""
    title: str = ""
    year: Optional[int] = None
    year_label: str = ""         # as printed: "2012b" tells it from "2012a"
    arxiv_id: str = ""
    doi: str = ""
    cited: int = 0               # in-text citations found in the body

    @property
    def url(self) -> str:
        """A direct link, if the entry carries an identifier."""
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        return ""

    @property
    def search_url(self) -> str:
        """A scholarly search for the entry, for when there is no identifier."""
        query = self.title or self.text[:160]
        return f"https://scholar.google.com/scholar?q={quote_plus(query)}"

    @property
    def first_author_surname(self) -> str:
        first = re.split(r",| and ", self.authors, maxsplit=1)[0].strip()
        if not first:
            return ""
        words = first.split()
        # "Kingma, Diederik" puts the surname first; "Jacob Devlin" last.
        if "," in self.authors and len(words) == 1:
            return words[0]
        return words[-1].strip(".")

    @property
    def keys(self) -> List[str]:
        """Identities for matching the same work across two bibliographies.

        Every identity counts, not just the strongest: one paper may cite a
        work by its arXiv id and another by title alone.
        """
        out = []
        if self.arxiv_id:
            out.append(f"arxiv:{self.arxiv_id}")
        if self.doi:
            out.append(f"doi:{self.doi.lower()}")
        words = re.findall(r"[a-z0-9]+", self.title.lower())
        if len(words) >= 3:
            out.append("title:" + " ".join(words))
        return out


# ------------------------------------------------------------- splitting


def _entries_numbered(lines: List[str]) -> List[str]:
    entries: List[str] = []
    for line in lines:
        if _NUMBERED.match(line):
            entries.append(line)
        elif entries:
            entries[-1] += "\n" + line
    return entries


def _entries_author_year(lines: List[str]) -> List[str]:
    entries: List[str] = []
    for line in lines:
        starts = bool(_AUTHOR_START.match(line))
        if entries and not (starts and entries[-1].rstrip().endswith(".")):
            entries[-1] += "\n" + line
        elif starts:
            entries.append(line)
    return entries


def split_entries(text: str) -> List[str]:
    """Split a bibliography's text into one string per entry."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    numbered = sum(1 for ln in lines if _NUMBERED.match(ln))
    # Most entries open on a fresh line, so a numbered bibliography shows a
    # marker on a good share of its lines.
    if numbered >= 3 and numbered >= len(lines) * 0.15:
        entries = _entries_numbered(lines)
    else:
        entries = _entries_author_year(lines)

    out = []
    for entry in entries:
        entry = re.sub(r"([a-z])-\n([a-z])", r"\1\2", entry)
        entry = normalize_spaces(entry)
        entry = _BACKREF.sub(".", entry)
        if len(entry) > _MAX_ENTRY:
            # Keep up to the first year that closes a sentence, which is
            # where a reference ends in nearly every style.
            end = re.search(r"\b(?:19|20)\d{2}[a-z]?\.", entry)
            entry = entry[: end.end()] if end else entry[:_MAX_ENTRY]
        if len(entry) >= 20:
            out.append(entry)
    return out


# --------------------------------------------------------------- parsing


def _ends_authors_at_initial(after: str) -> bool:
    """Whether the text after an initial's full stop begins the title.

    "Geoffrey E. Hinton." and "K. He, X. Zhang" carry on with a surname --
    a word closed by a full stop or comma -- or with "and" or another
    initial. Anything else ("Matthew D. Adadelta: An adaptive...") is the
    title, and the author list ended at the initial.
    """
    tokens = after.split(" ", 2)
    token = tokens[0]
    if not token or token[0].islower():
        return False
    # Another initial needs its own full stop: a bare "A" opens a title
    # ("Andrew W. A fast natural newton method").
    if re.fullmatch(r"(?:[A-Z]\.)+", token):
        return False
    # "A. Vaswani et al." and "J. Smith and K. Lee": a surname, still authors.
    if len(tokens) > 1 and tokens[1] in ("et", "and", "&"):
        return False
    return not token.endswith((".", ","))


def _segments(entry: str) -> List[str]:
    """Split an entry at the full stops (or ? and !) that end a part."""
    parts, start = [], 0
    for m in re.finditer(r"([.?!])\s+", entry):
        if m.group(1) == ".":
            before = entry[start : m.start()].rsplit(" ", 1)[-1]
            if _INITIAL.match(before) and not _ends_authors_at_initial(entry[m.end():]):
                continue
            if before.lower() in _NOT_A_BREAK:
                continue
            parts.append(entry[start : m.start()].strip())
        else:
            # A question is a title's own punctuation; keep it.
            parts.append(entry[start : m.start() + 1].strip())
        start = m.end()
    tail = entry[start:].strip().rstrip(".")
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def parse_entry(entry: str) -> Reference:
    """Pull authors, title, year and identifiers out of one entry."""
    ref = Reference(text=entry)

    m = _NUMBERED.match(entry)
    body = entry
    if m:
        ref.number = int(m.group(1))
        body = entry[m.end():]

    if a := _ARXIV.search(body):
        ref.arxiv_id = a.group(1)
    if d := _DOI.search(body):
        ref.doi = d.group(1).rstrip(".,;)")
    years = list(_YEAR.finditer(body))
    if years:
        ref.year = int(years[-1].group(1))
        ref.year_label = years[-1].group(0)

    # IEEE style puts the title in quotes; trust that when it is there.
    quoted = re.search(r"[\"“](.{8,250}?)[,.]?[\"”]", body)
    parts = _segments(body)
    if quoted:
        ref.title = quoted.group(1).strip(" ,.")
        ref.authors = body[: quoted.start()].strip(" ,.")
    elif parts:
        ref.authors = parts[0]
        rest = parts[1:]
        # ACL style puts the year right after the authors: "Names. 2019. Title."
        if rest and re.fullmatch(r"\(?(?:19|20)\d{2}[a-z]?\)?", rest[0]):
            rest = rest[1:]
        if rest:
            ref.title = rest[0]
    return ref


# ------------------------------------------------------------- citations


def _numbered_citations(body: str) -> Counter:
    """How often each [n] is cited, with ranges like [10-14] expanded."""
    counts: Counter = Counter()
    for group in re.findall(r"\[(\d{1,3}(?:\s*[,\-]\s*\d{1,3})*)\]", body):
        for part in group.split(","):
            if "-" in part:
                lo, _, hi = part.partition("-")
                lo, hi = int(lo), int(hi)
                if 0 < hi - lo <= 30:
                    counts.update(range(lo, hi + 1))
                    continue
            counts[int(part.split("-")[0])] += 1
    return counts


def _author_year_citations(ref: Reference, body: str) -> int:
    surname = ref.first_author_surname
    if not surname or not ref.year or len(surname) < 2:
        return 0
    # "(Duchi et al., 2011)", "Duchi et al. (2011)", "(Duchi & Singer, 2011a)".
    # A lettered year must match its letter, or 2012a and 2012b both claim
    # every citation of either.
    if ref.year_label and ref.year_label[-1].isalpha():
        year = re.escape(ref.year_label)
    else:
        year = rf"{ref.year}[a-z]?"
    pattern = re.compile(
        rf"\b{re.escape(surname)}\b[^()\[\]]{{0,40}}?\(?{year}\b"
    )
    return len(pattern.findall(body))


def count_citations(refs: List[Reference], body: str) -> None:
    """Fill in `cited` on each reference from the paper's body text."""
    if any(r.number is not None for r in refs):
        counts = _numbered_citations(body)
        for r in refs:
            r.cited = counts.get(r.number, 0) if r.number is not None else 0
    else:
        for r in refs:
            r.cited = _author_year_citations(r, body)


# ------------------------------------------------------------ public API


def extract_references(doc, body: str = "") -> List[Reference]:
    """The paper's bibliography, parsed, with in-text citation counts.

    `doc` is an avanti.extract.Document; `body` is the text to count
    citations in (normally the paper's body, without the bibliography).
    """
    paper = sections.analyze(doc.full_text)
    section = paper.get("references")
    if section is None:
        return []

    refs = [parse_entry(e) for e in split_entries(section.text)]
    count_citations(refs, body or doc.text)
    return refs


def most_cited(refs: List[Reference], limit: int = 5) -> List[Reference]:
    """The references the paper cites most, most-cited first."""
    cited = [r for r in refs if r.cited > 0]
    return sorted(cited, key=lambda r: (-r.cited, r.number or 0))[:limit]


def shared(a: List[Reference], b: List[Reference]) -> List[Dict[str, Reference]]:
    """Works that two bibliographies both cite, matched by id or title."""
    index = {key: r for r in b for key in r.keys}
    out = []
    for r in a:
        match = next((index[k] for k in r.keys if k in index), None)
        if match is not None:
            out.append({"a": r, "b": match})
    return out
