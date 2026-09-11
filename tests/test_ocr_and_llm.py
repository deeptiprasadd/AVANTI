"""Tests for OCR reading order and Gemini model selection.

Neither of these needs a network call or a PDF: model ranking is a pure
function over a list of names, and OCR reading order is a pure function
over box coordinates.
"""

from avanti import llm, ocr


# --------------------------------------------------------- model selection


def test_version_ranking_prefers_newer():
    assert llm._version_key("gemini-3.8-flash") > llm._version_key("gemini-3.5-flash")
    assert llm._version_key("gemini-3.5-flash") > llm._version_key("gemini-2.5-flash")
    assert llm._version_key("gemini-3-flash") > llm._version_key("gemini-2.9-flash")


def test_candidate_order_prefers_alias_then_newest(monkeypatch):
    """The retired model must not be chosen first.

    ListModels returns retired models, and in its own order -- it listed
    `gemini-2.5-flash` first, which 404s for new keys. Selection must rank
    the listing rather than trust it.
    """
    listing = {
        "models": [
            # Deliberately in the same misleading order the real API uses.
            {"name": "models/gemini-2.5-flash",
             "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-2.5-flash-preview-tts",
             "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-flash-latest",
             "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.5-flash",
             "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.8-flash",
             "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-001",
             "supportedGenerationMethods": ["embedContent"]},
        ]
    }

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return listing

    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: FakeResponse())
    llm.candidate_models.cache_clear()

    order = llm.candidate_models("test-key-not-real")

    assert order[0] == "gemini-flash-latest", f"got {order}"
    # The newest plain model outranks the retired one.
    assert order.index("gemini-3.8-flash") < order.index("gemini-3.5-flash")
    # Specialized and embedding-only models are never candidates.
    assert not any("tts" in n for n in order)
    assert not any("embedding" in n for n in order)


def test_candidate_models_falls_back_when_listing_fails(monkeypatch):
    def boom(*a, **k):
        raise llm.requests.RequestException("network down")

    monkeypatch.setattr(llm.requests, "get", boom)
    llm.candidate_models.cache_clear()

    assert llm.candidate_models("test-key-not-real") == (llm.FALLBACK_MODEL,)


def test_missing_key_returns_setup_notice(monkeypatch):
    monkeypatch.setattr(llm, "get_key", lambda: "")
    out = llm.deep_analysis("Some paper text.", "Title", "Professional")
    assert "No API key configured" in out


# ------------------------------------------------------- OCR reading order


def box(text, x0, x1, top, bottom=None):
    return ocr.Box(text, x0, x1, top, bottom if bottom is not None else top + 10)


def test_lines_group_by_vertical_position():
    boxes = [
        box("world", 120, 180, 50),
        box("Hello", 10, 100, 52),      # same line, a couple of pixels off
        box("Second", 10, 90, 90),
    ]
    lines = ocr._group_lines(boxes)
    assert lines == ["Hello world", "Second"]


def test_two_column_scan_is_not_interleaved():
    """A two-column scan must read down one column, then the other."""
    boxes = []
    for i in range(8):
        boxes.append(box(f"L{i}", 40, 260, 100 + i * 20))     # left column
        boxes.append(box(f"R{i}", 340, 560, 100 + i * 20))    # right column
    text = ocr._page_text(boxes, width=600.0)
    order = text.replace("\n", " ").split()

    assert order[0] == "L0"
    # Every left-column token must precede every right-column one.
    assert max(order.index(f"L{i}") for i in range(8)) < min(
        order.index(f"R{i}") for i in range(8)
    )


def test_single_column_scan_reads_straight_down():
    boxes = [box(f"line{i}", 40, 500, 100 + i * 20) for i in range(8)]
    text = ocr._page_text(boxes, width=600.0)
    assert text.split("\n")[0] == "line0"
    assert text.split("\n")[-1] == "line7"


def test_gutter_not_detected_in_full_width_text():
    """Text spanning the page must not be split into phantom columns."""
    boxes = [box(f"wide{i}", 40, 560, 100 + i * 18) for i in range(20)]
    assert ocr._detect_gutter(boxes, 600.0) is None


def test_backend_reports_consistently():
    """available() and backend() must agree."""
    assert ocr.available() == (ocr.backend() is not None)
    if not ocr.available():
        assert ocr.backend_label() == "not installed"


def test_ocr_without_engine_returns_install_hint(monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: False)
    result = ocr.ocr_pdf(b"%PDF-1.4 not really a pdf", max_pages=1)
    assert result.pages == []
    assert any("requirements-ocr" in w for w in result.warnings)
