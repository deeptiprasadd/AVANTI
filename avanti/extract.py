"""PDF text extraction that respects page layout.

The naive `page.extract_text()` reads a two-column paper straight across the
page. That interleaves the left and right columns, splices figure captions
into the middle of sentences, and fuses words together -- and every analysis
step downstream inherits the damage.

This module reconstructs the real reading order instead:

  1. pull words with positions and font sizes, at a tolerance that does not
     fuse neighbouring words together
  2. discard rotated margin text (the arXiv stamp down the left edge)
  3. work out the body font size across the whole document, then drop text
     far from it: captions, footnotes, table cells, page numbers
  4. detect a column gutter and read each column in turn
  5. strip repeated running heads, de-hyphenate, and fix glyph damage

Title and authors come from font size rather than keyword guessing: on a
paper's first page the title is simply the largest upright text near the top,
which is far more reliable than scoring lines for title-ish words.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------- backends

try:
    import pdfplumber

    HAS_PDFPLUMBER = True
except ImportError:  # pragma: no cover - depends on install
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader

    HAS_PYPDF = True
except ImportError:  # pragma: no cover - depends on install
    HAS_PYPDF = False


# pdfplumber's default x_tolerance of 3 fuses adjacent words in many
# academic PDFs (one page of the Transformer paper collapses from 430 words
# to 132, with single tokens 88 characters long). 1.5 splits correctly.
X_TOLERANCE = 1.5

# Vertical tolerance for deciding two words share a text line.
Y_TOLERANCE = 2.5


def backend_name() -> str:
    """Which extraction backend is active, for display in the UI."""
    if HAS_PDFPLUMBER:
        return "pdfplumber (layout-aware)"
    if HAS_PYPDF:
        return "pypdf (no layout support)"
    return "none"


# ---------------------------------------------------------------- cleaning

# Ligatures and lookalike characters that PDF fonts emit as single glyphs.
_GLYPH_FIXES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "­": "",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "﻿": "",
}

# Unmapped glyphs come through as (cid:123); they are pure noise.
_CID = re.compile(r"\(cid:\d+\)")


def clean_glyphs(text: str) -> str:
    """Normalize ligatures, quotes and dashes; drop unmapped-glyph markers."""
    for bad, good in _GLYPH_FIXES.items():
        text = text.replace(bad, good)
    return _CID.sub("", text)


def dehyphenate(text: str) -> str:
    """Join words split across a line break ("rep-" + "resentation")."""
    # Only join when a lowercase letter follows, so genuine hyphenated
    # compounds sitting at a line end ("state-of-the-art") survive.
    return re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    """Squeeze runs of spaces and blank lines without losing paragraphs."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_junk_line(line: str) -> bool:
    """True for equation fragments and stray glyphs, not real prose.

    Inline math is laid out as separate positioned glyphs, so it arrives as
    short lines like "x", "(x)+x" or "F (x) relu x". Those wreck sentence
    splitting, and they carry no summarizable content.
    """
    s = line.strip()
    if not s:
        return True
    letters = sum(c.isalpha() for c in s)
    if len(s) <= 6 and letters <= 2:
        return True
    if len(s) < 25 and letters / len(s) < 0.45:
        return True
    return False


# ------------------------------------------------------------ layout logic


def _page_words(page) -> List[dict]:
    """Upright words on a page, at a tolerance that keeps words separate."""
    try:
        words = page.extract_words(
            extra_attrs=["size", "upright"],
            x_tolerance=X_TOLERANCE,
            keep_blank_chars=False,
        )
    except Exception:
        return []
    # Rotated text is marginalia -- the arXiv identifier runs vertically up
    # the left edge and otherwise lands inside the title.
    return [w for w in words if w.get("upright", True)]


def _table_regions(page) -> List[Tuple[float, float, float, float]]:
    """Bounding boxes of tables on a page.

    Table cells are often set at the body font size, so a size filter alone
    keeps them -- and once flattened into lines they fuse into pseudo-prose
    like "Transformer (4 layers) In rows (B), we observe that reducing ...".
    Excluding the table regions outright fixes that, and lets the tables be
    reported separately as data.
    """
    try:
        return [tuple(t.bbox) for t in page.find_tables()]
    except Exception:
        return []


def _in_any_region(
    word: dict, regions: List[Tuple[float, float, float, float]]
) -> bool:
    """True if a word's centre falls inside one of the regions."""
    cx = (word["x0"] + word["x1"]) / 2.0
    cy = (word["top"] + word["bottom"]) / 2.0
    for x0, top, x1, bottom in regions:
        if x0 <= cx <= x1 and top <= cy <= bottom:
            return True
    return False


def extract_tables(page) -> List[List[List[str]]]:
    """Tables on a page as lists of rows, for separate display."""
    out: List[List[List[str]]] = []
    try:
        for table in page.find_tables():
            rows = table.extract()
            clean = [
                [
                    clean_glyphs((cell or "").strip().replace(chr(10), " "))
                    for cell in row
                ]
                for row in rows
                if any((cell or "").strip() for cell in row)
            ]
            if len(clean) >= 2:
                out.append(clean)
    except Exception:
        pass
    return out


def document_body_size(pages_words: List[List[dict]]) -> float:
    """The body text font size, measured across the whole document.

    Measured per page this misfires: on a table-heavy page the small table
    font can outnumber body text, and the filter then keeps the junk and
    drops the prose. Weighting by character count across all pages is
    stable -- for a paper with 10pt body text, 10pt wins by an order of
    magnitude.
    """
    weighted: Counter = Counter()
    for words in pages_words:
        for w in words:
            weighted[round(w.get("size", 0) * 2) / 2] += len(w["text"])
    if not weighted:
        return 0.0
    return weighted.most_common(1)[0][0]


def _detect_gutter(words: List[dict], page_width: float) -> Optional[float]:
    """Find the x of a two-column gutter, or None for a single column.

    A real gutter is a vertical line almost no word crosses, with
    substantial text on both sides of it.
    """
    if len(words) < 40:
        return None

    spans = [(w["x0"], w["x1"]) for w in words]
    total = len(spans)
    best: Optional[Tuple[int, float]] = None

    # Only look near the middle -- that is where an academic gutter sits.
    for pct in range(40, 61):
        split = page_width * pct / 100.0
        crossing = sum(1 for x0, x1 in spans if x0 < split - 2 and x1 > split + 2)
        left = sum(1 for _, x1 in spans if x1 <= split)
        right = sum(1 for x0, _ in spans if x0 >= split)

        if crossing > max(2, int(0.015 * total)):
            continue
        if left < 0.25 * total or right < 0.25 * total:
            continue
        if best is None or crossing < best[0]:
            best = (crossing, split)

    return best[1] if best else None


def _group_into_lines(words: List[dict], y_tol: float = Y_TOLERANCE) -> List[str]:
    """Cluster words into text lines by vertical position, then read L->R."""
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    groups: List[List[dict]] = [[ordered[0]]]

    for w in ordered[1:]:
        if abs(w["top"] - groups[-1][0]["top"]) <= y_tol:
            groups[-1].append(w)
        else:
            groups.append([w])

    lines: List[str] = []
    for group in groups:
        group.sort(key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in group).strip()
        if text:
            lines.append(text)
    return lines


def _page_lines(
    page,
    words: List[dict],
    body_size: float,
    drop_junk: bool = True,
    exclude_tables: bool = True,
) -> Tuple[List[str], bool]:
    """One page's lines in true reading order. Returns (lines, two_column)."""
    if not words:
        return [], False

    # Drop words inside tables: their cells flatten into fake sentences.
    if exclude_tables:
        regions = _table_regions(page)
        if regions:
            words = [w for w in words if not _in_any_region(w, regions)]
            if not words:
                return [], False

    # Keep only body-sized text. Captions, footnotes, table cells and page
    # numbers are what splice fragments into the middle of sentences.
    if body_size > 0:
        words = [w for w in words if w.get("size", body_size) >= body_size * 0.94]
    if not words:
        return [], False

    gutter = _detect_gutter(words, float(page.width))

    if gutter is None:
        lines, two_col = _group_into_lines(words), False
    else:
        # Two columns: read the left one fully, then the right one.
        left = [w for w in words if w["x1"] <= gutter]
        right = [w for w in words if w["x0"] >= gutter]
        lines = _group_into_lines(left) + _group_into_lines(right)
        two_col = True

    if drop_junk:
        lines = [ln for ln in lines if not is_junk_line(ln)]
    return lines, two_col


# ------------------------------------------------------- header / footer


def _normalize_for_repeat(line: str) -> str:
    """Blur out page numbers so repeated running heads compare equal."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _strip_running_heads(pages: List[List[str]]) -> List[List[str]]:
    """Remove lines that repeat across pages (journal name, page numbers)."""
    if len(pages) < 3:
        return pages

    candidates: Counter = Counter()
    for lines in pages:
        for ln in lines[:2] + lines[-2:]:
            if len(ln) < 100:
                candidates[_normalize_for_repeat(ln)] += 1

    threshold = max(3, int(0.4 * len(pages)))
    repeated = {k for k, n in candidates.items() if n >= threshold and k}

    cleaned: List[List[str]] = []
    for lines in pages:
        keep = [
            ln
            for i, ln in enumerate(lines)
            if not (
                (i < 2 or i >= len(lines) - 2)
                and _normalize_for_repeat(ln) in repeated
            )
        ]
        cleaned.append(keep)
    return cleaned


# ------------------------------------------------------ title and authors

_ABSTRACT_LINE = re.compile(r"^\s*(abstract|summary)\b", re.I)
_EMAIL = re.compile(r"[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}")


def _extract_title_and_authors(page, words: List[dict], body_size: float):
    """Read title and authors from font sizes on the first page."""
    if not words:
        return "", []

    top_zone = float(page.height) * 0.45
    head = [w for w in words if w["top"] <= top_zone]
    if not head:
        return "", []

    sizes = sorted({round(w.get("size", 0) * 2) / 2 for w in head}, reverse=True)

    title, title_size = "", None
    for size in sizes:
        at_size = [
            w for w in head if abs(round(w.get("size", 0) * 2) / 2 - size) < 0.25
        ]
        text = " ".join(_group_into_lines(at_size)).strip()
        n_words = len(text.split())
        # Bigger than body text, and a plausible title length -- this skips a
        # single huge word (a journal logo) and the body text itself.
        if size > body_size * 1.12 and 3 <= n_words <= 40:
            title, title_size = text, size
            break

    if not title:
        for ln in _group_into_lines(head):
            if len(ln.split()) >= 4:
                title = ln
                break

    # Authors: text below the title, above the abstract, at least body size.
    authors: List[str] = []
    if title_size is not None:
        below = [
            w
            for w in head
            if round(w.get("size", 0) * 2) / 2 < title_size - 0.25
            and w.get("size", 0) >= body_size * 0.9
        ]
        for ln in _group_into_lines(below):
            if _ABSTRACT_LINE.match(ln):
                break
            ln = _EMAIL.sub("", ln).strip(" ,;")
            # Removing an email or a superscript marker can leave a gap
            # before the separating comma ("Ali Hassani , Humphrey Shi").
            ln = re.sub(r"\s+([,;])", r"\1", ln)
            ln = re.sub(r"\s{2,}", " ", ln)
            if ln and not is_junk_line(ln):
                authors.append(ln)

    return clean_glyphs(title), [clean_glyphs(a) for a in authors[:6]]


# ------------------------------------------------------------- public API


@dataclass
class Document:
    """A parsed paper: clean text plus what we learned while parsing it."""

    pages: List[str] = field(default_factory=list)
    page_count: int = 0
    pages_processed: int = 0
    title: str = ""
    author_lines: List[str] = field(default_factory=list)
    two_column_pages: int = 0
    body_font_size: float = 0.0
    tables: List[List[List[str]]] = field(default_factory=list)
    used_ocr: bool = False
    backend: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """All pages joined, with form feeds marking page boundaries."""
        return "\f".join(self.pages)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def layout(self) -> str:
        if not self.pages_processed:
            return "unknown"
        if self.two_column_pages >= self.pages_processed * 0.5:
            return "two-column"
        return "single-column"

    def page_of_offset(self, char_index: int) -> int:
        """Which 1-based page a character offset into `text` falls on."""
        running = 0
        for i, page in enumerate(self.pages, start=1):
            running += len(page) + 1  # +1 for the form feed separator
            if char_index < running:
                return i
        return max(1, len(self.pages))


def _clean_page(lines: List[str]) -> str:
    text = clean_glyphs("\n".join(lines))
    return collapse_whitespace(dehyphenate(text))


# Below this many words per page, a PDF is treated as having no usable text
# layer -- i.e. it is a scan. A digital paper runs 300-600 words per page.
OCR_TRIGGER_WORDS_PER_PAGE = 25


def extract(
    file_bytes: bytes,
    max_pages: Optional[int] = None,
    ocr_mode: str = "auto",
    ocr_progress=None,
) -> Document:
    """Parse a PDF into a Document with layout-correct text.

    `ocr_mode` controls optical recognition for scanned pages:
      "auto"   OCR only when the text layer is effectively empty (default)
      "always" OCR regardless, ignoring any text layer
      "never"  never OCR; a scan will simply come back empty
    """
    doc = Document(backend=backend_name())

    if ocr_mode == "always":
        return _extract_via_ocr(file_bytes, max_pages, doc)

    if HAS_PDFPLUMBER:
        doc = _extract_pdfplumber(file_bytes, max_pages, doc, ocr_progress)
    elif HAS_PYPDF:
        doc = _extract_pypdf(file_bytes, max_pages, doc)
    else:
        doc.warnings.append("No PDF library installed. Install pdfplumber.")
        return doc

    if ocr_mode == "auto" and _needs_ocr(doc):
        return _extract_via_ocr(file_bytes, max_pages, doc, ocr_progress)
    return doc


def _needs_ocr(doc: Document) -> bool:
    """True if the text layer is too sparse to analyze."""
    if not doc.pages_processed:
        return True
    return doc.word_count < OCR_TRIGGER_WORDS_PER_PAGE * doc.pages_processed


def _extract_via_ocr(
    file_bytes: bytes,
    max_pages: Optional[int],
    doc: Document,
    progress=None,
) -> Document:
    """Read a scanned PDF optically, reusing the normal cleaning pipeline."""
    from . import ocr as ocr_module

    if not ocr_module.available():
        doc.warnings.append(
            "This looks like a scanned PDF with no text layer. "
            + ocr_module.INSTALL_HINT
        )
        return doc

    result = ocr_module.ocr_pdf(
        file_bytes, max_pages=max_pages, progress=progress
    )
    if not result.word_count:
        doc.warnings.extend(result.warnings)
        return doc

    # OCR output goes through the same glyph, hyphen and whitespace cleanup
    # as a digital PDF, and the same junk-line filter.
    fresh = Document(backend=f"OCR: {result.backend}")
    fresh.page_count = doc.page_count or result.pages_processed
    fresh.pages = [
        _clean_page([ln for ln in page.split("\n") if not is_junk_line(ln)])
        for page in result.pages
    ]
    fresh.pages_processed = len(fresh.pages)
    fresh.used_ocr = True
    fresh.warnings = list(result.warnings)

    # A scan carries no font metadata, so type size cannot identify the
    # title. Among the first few header lines the title is normally the
    # longest -- an author list is usually shorter and comma-separated --
    # so pick by length rather than taking the first line that qualifies.
    if fresh.pages:
        head = [
            ln.strip()
            for ln in fresh.pages[0].split("\n")[:8]
            if 4 <= len(ln.split()) <= 30
        ]
        # Skip lines that read as an author list or an affiliation.
        candidates = [
            ln for ln in head
            if ln.count(",") <= 2
            and not re.search(r"@|\b(?:University|Institute|Department|Research)\b", ln)
        ] or head
        if candidates:
            fresh.title = max(candidates, key=lambda ln: len(ln.split()))

    return fresh


def _extract_pdfplumber(
    file_bytes: bytes,
    max_pages: Optional[int],
    doc: Document,
    ocr_progress=None,
) -> Document:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        doc.page_count = len(pdf.pages)
        limit = min(max_pages or doc.page_count, doc.page_count)
        pages = list(pdf.pages[:limit])

        # First pass: gather words so the body font size can be measured
        # across the whole document before any filtering happens.
        per_page_words = [_page_words(p) for p in pages]
        doc.body_font_size = document_body_size(per_page_words)

        raw_pages: List[List[str]] = []
        for i, (page, words) in enumerate(zip(pages, per_page_words)):
            lines, two_col = _page_lines(page, words, doc.body_font_size)
            raw_pages.append(lines)
            doc.tables.extend(extract_tables(page))
            if two_col:
                doc.two_column_pages += 1
            if i == 0:
                doc.title, doc.author_lines = _extract_title_and_authors(
                    page, words, doc.body_font_size
                )

    raw_pages = _strip_running_heads(raw_pages)
    doc.pages = [_clean_page(lines) for lines in raw_pages]
    doc.pages_processed = len(doc.pages)

    # A sparse text layer is not warned about here: extract() checks for it
    # and routes the document through OCR instead.
    return doc


def _extract_pypdf(
    file_bytes: bytes, max_pages: Optional[int], doc: Document
) -> Document:
    """Fallback with no layout awareness -- flagged so the user knows."""
    reader = PdfReader(io.BytesIO(file_bytes))
    doc.page_count = len(reader.pages)
    limit = min(max_pages or doc.page_count, doc.page_count)

    for page in reader.pages[:limit]:
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        doc.pages.append(collapse_whitespace(dehyphenate(clean_glyphs(raw))))

    doc.pages_processed = len(doc.pages)
    doc.warnings.append(
        "pdfplumber is not installed, so column layout could not be detected. "
        "Text from two-column papers may be interleaved."
    )
    return doc
