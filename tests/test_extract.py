"""Tests for title and author recovery on a paper's first page."""

from avanti import extract


def _word(text, x0, x1, size, bottom=100.0):
    return {"text": text, "x0": x0, "x1": x1, "size": size,
            "bottom": bottom, "top": bottom - size}


def test_small_caps_title_is_detected():
    assert extract._looks_small_caps("A : A M S O")
    assert not extract._looks_small_caps("Attention Is All You Need")


def test_small_caps_title_is_rebuilt_from_its_baseline():
    """Each word is a full-size capital touching a smaller rest: "A"+"DAM"."""
    words = [
        _word("A", 0, 12, 17), _word("DAM", 12.4, 50, 14), _word(":", 50.3, 54, 17),
        _word("A", 60, 72, 17),
        _word("M", 78, 92, 17), _word("ETHOD", 92.3, 150, 14),
        _word("Diederik Kingma", 0, 120, 10, bottom=130),   # an author line below
    ]
    big = [w for w in words if w["size"] == 17]
    title = extract._title_on_baselines(words, big, body_size=10)
    assert title == "ADAM: A METHOD"


def test_figure_legends_are_not_author_lines():
    assert extract._is_author_line("Ali Hassani, Humphrey Shi")
    assert extract._is_author_line("University of Amsterdam, OpenAI University of Toronto")
    assert not extract._is_author_line("NA / DiNA (dilation = 1) DiNA (dilation = 2)")
