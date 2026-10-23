"""Run the analysis engine and turn its output into JSON.

The engine returns dataclasses built for Python callers. This module is the
only place that knows how to flatten them for the wire, so the engine stays
free of any serialization concerns and the API shape can change without
touching `avanti/`.
"""

from __future__ import annotations

from typing import List, Tuple

from avanti import pipeline as engine
from avanti import qa, references
from avanti.pipeline import Analysis


def run(
    data: bytes,
    *,
    max_pages: int = 30,
    target_words: int = 450,
    ocr_mode: str = "auto",
    progress=None,
) -> Tuple[Analysis, dict]:
    """Analyze a PDF. Returns (live analysis, JSON-safe result)."""
    analysis = engine.run(
        data,
        max_pages=max_pages,
        target_words=target_words,
        ocr_mode=ocr_mode,
        progress=progress,
    )
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


def reference_json(r: references.Reference) -> dict:
    return {
        "number": r.number,
        "text": r.text,
        "authors": r.authors,
        "title": r.title,
        "year": r.year,
        "arxivId": r.arxiv_id,
        "doi": r.doi,
        "url": r.url,
        "searchUrl": r.search_url,
        "cited": r.cited,
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
        "references": {
            "items": [reference_json(r) for r in a.references],
            "mostCited": [
                reference_json(r) for r in references.most_cited(a.references)
            ],
        },
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


def _side_json(s) -> dict:
    return {
        "title": s.title,
        "authors": s.authors,
        "pages": s.pages,
        "words": s.words,
        "problem": s.problem,
        "method": s.method,
        "limitations": s.limitations,
        "datasets": s.datasets,
        "metrics": s.metrics,
        "referenceCount": s.reference_count,
    }


def comparison_json(c) -> dict:
    return {
        "a": _side_json(c.a),
        "b": _side_json(c.b),
        "similarity": round(c.similarity, 3),
        "relation": c.relation,
        "sharedDatasets": c.shared_datasets,
        "sharedMetrics": [
            {"name": n, "a": va, "b": vb} for n, va, vb in c.shared_metrics
        ],
        "sharedPhrases": c.shared_phrases,
        "sharedReferences": c.shared_references,
        "aCitesB": c.a_cites_b,
        "bCitesA": c.b_cites_a,
    }
