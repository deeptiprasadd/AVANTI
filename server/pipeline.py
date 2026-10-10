"""Run the analysis engine and turn its output into JSON.

The engine returns dataclasses built for Python callers. This module is the
only place that knows how to flatten them for the wire, so the engine stays
free of any serialization concerns and the API shape can change without
touching `avanti/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from avanti import analyze, extract, qa, sections, summarize

# Weights for the overall progress bar. Extraction dominates when OCR runs,
# which is exactly when a progress bar matters, so it gets most of the span.
_STAGE_SPAN = {
    "extracting": (0.02, 0.70),
    "sectioning": (0.70, 0.76),
    "analyzing": (0.76, 0.88),
    "summarizing": (0.88, 0.95),
    "indexing": (0.95, 0.99),
}


@dataclass
class Analysis:
    """Everything derived from one PDF, kept together for the session."""

    doc: Any
    paper: Any
    brief: Any
    summary: Any
    index: Any


def _span(stage: str, fraction: float) -> float:
    lo, hi = _STAGE_SPAN.get(stage, (0.0, 1.0))
    return lo + (hi - lo) * max(0.0, min(1.0, fraction))


def run(
    data: bytes,
    *,
    max_pages: int = 30,
    target_words: int = 450,
    ocr_mode: str = "auto",
    progress=None,
) -> Tuple[Analysis, dict]:
    """Analyze a PDF. Returns (live analysis, JSON-safe result)."""

    def report(stage: str, fraction: float, detail: str = "") -> None:
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
        data,
        max_pages=max_pages,
        ocr_mode=ocr_mode,
        ocr_progress=on_ocr_page,
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

    report("indexing", 0.5, "Indexing for questions")
    index = qa.build_index(doc, paper)

    analysis = Analysis(doc, paper, brief, summary, index)
    return analysis, to_json(analysis)


# --------------------------------------------------------- serialization


def _finding(f) -> dict:
    return {"text": f.text, "page": f.page, "score": round(f.score, 3)}


def _metric(m) -> dict:
    return {
        "name": m.name,
        "value": m.value,
        "unit": m.unit,
        "display": f"{m.value}{m.unit}",
        "sentence": m.sentence,
        "page": m.page,
    }


def document_json(doc) -> dict:
    return {
        "title": doc.title,
        "authors": doc.author_lines,
        "pageCount": doc.page_count,
        "pagesProcessed": doc.pages_processed,
        "wordCount": doc.word_count,
        "layout": doc.layout,
        "usedOcr": doc.used_ocr,
        "backend": doc.backend,
        "tableCount": len(doc.tables),
        "warnings": doc.warnings,
    }


def to_json(a: Analysis) -> dict:
    """Flatten a full analysis for the wire."""
    b = a.brief

    return {
        "document": document_json(a.doc),
        "brief": {
            "objective": [_finding(f) for f in b.objective],
            "contribution": [_finding(f) for f in b.contribution],
            "limitation": [_finding(f) for f in b.limitation],
            "future": [_finding(f) for f in b.future],
            "datasets": b.datasets,
            "metrics": [_metric(m) for m in b.metrics],
            "phrases": b.phrases,
        },
        "summary": {
            "abstract": a.summary.abstract,
            "keyPoints": a.summary.key_points,
            "sections": [
                {
                    "name": s.name,
                    "label": s.label,
                    "text": s.text,
                    "sourceWords": s.source_words,
                    "words": s.word_count,
                }
                for s in a.summary.sections
            ],
            "markdown": a.summary.full_text,
        },
        "sections": [
            {"name": s.name, "label": s.label, "words": s.word_count}
            for s in a.paper.sections
        ],
        "tables": [
            {"rows": t[:40], "rowCount": len(t)} for t in a.doc.tables[:8]
        ],
        "suggestedQuestions": qa.suggested_questions(a.paper),
        "passageCount": len(a.index),
    }


def answers_json(answers) -> List[dict]:
    return [
        {
            "text": ans.text,
            "page": ans.page,
            "section": ans.section,
            "score": round(ans.score, 4),
        }
        for ans in answers
    ]
