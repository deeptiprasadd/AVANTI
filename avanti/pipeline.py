"""One call from PDF bytes to a complete analysis.

Both front ends -- the Streamlit app and the HTTP API -- run exactly this, so
a paper reads the same whichever one it is uploaded to, and a new stage is
added in one place rather than two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import analyze, extract, qa, references, sections, summarize

# Weights for an overall progress bar. Extraction dominates when OCR runs,
# which is exactly when a progress bar matters, so it gets most of the span.
_STAGE_SPAN = {
    "extracting": (0.02, 0.70),
    "sectioning": (0.70, 0.75),
    "analyzing": (0.75, 0.85),
    "summarizing": (0.85, 0.92),
    "references": (0.92, 0.96),
    "indexing": (0.96, 0.99),
}

# progress(stage, overall fraction 0..1, human-readable detail)
Progress = Callable[[str, float, str], None]


@dataclass
class Analysis:
    """Everything derived from one PDF."""

    doc: extract.Document
    paper: sections.Paper
    brief: analyze.Brief
    summary: summarize.Summary
    index: qa.PaperIndex
    references: List[references.Reference] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.doc.title or "Untitled paper"


def _span(stage: str, fraction: float) -> float:
    lo, hi = _STAGE_SPAN.get(stage, (0.0, 1.0))
    return lo + (hi - lo) * max(0.0, min(1.0, fraction))


def run(
    data: bytes,
    *,
    max_pages: int = 30,
    target_words: int = 450,
    ocr_mode: str = "auto",
    progress: Optional[Progress] = None,
) -> Analysis:
    """Extract, segment, analyze, summarize and index a PDF."""

    def report(stage: str, fraction: float, detail: str) -> None:
        if progress:
            progress(stage, _span(stage, fraction), detail)

    report("extracting", 0.0, "Reading the PDF")

    def on_ocr_page(done: int, total: int) -> None:
        report(
            "extracting",
            done / max(1, total),
            f"No text layer - reading page {done} of {total} optically",
        )

    doc = extract.extract(
        data, max_pages=max_pages, ocr_mode=ocr_mode, ocr_progress=on_ocr_page
    )

    report("sectioning", 0.5, "Finding sections")
    paper = sections.analyze(doc.text)
    section_map = {s.name: s.text for s in paper.sections}

    def page_lookup(sentence: str) -> int:
        return doc.page_of_offset(doc.text.find(sentence[:60]))

    report("analyzing", 0.3, "Extracting claims and metrics")
    brief = analyze.brief(paper.body, section_map, page_lookup)

    report("summarizing", 0.4, "Summarizing by section")
    summary = summarize.summarize_paper(paper, target_words=target_words)

    report("references", 0.4, "Reading the bibliography")
    refs = references.extract_references(doc, paper.body)

    report("indexing", 0.5, "Indexing for questions")
    index = qa.build_index(doc)

    return Analysis(doc, paper, brief, summary, index, refs)
