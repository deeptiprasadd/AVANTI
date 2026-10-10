"""OCR for scanned PDFs.

A scanned paper has no text layer at all -- every page is an image, so
`extract_words()` returns nothing and the whole pipeline has nothing to work
with. This module rasterizes those pages and reads them optically.

Two backends are supported, tried in order:

  rapidocr-onnxruntime   pip-installable, bundles its own models, needs no
                         system packages -- the default
  pytesseract            wraps the Tesseract binary, which must be
                         installed separately

Reading order gets the same treatment as digital PDFs: recognized boxes are
grouped into lines by vertical position, and a two-column scan is split at
its gutter so the columns do not interleave.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# ------------------------------------------------------------- rasterizing

try:
    import pypdfium2 as pdfium

    HAS_RASTERIZER = True
except ImportError:  # pragma: no cover - depends on install
    HAS_RASTERIZER = False

# --------------------------------------------------------------- backends

try:
    from rapidocr_onnxruntime import RapidOCR

    HAS_RAPIDOCR = True
except ImportError:  # pragma: no cover - depends on install
    HAS_RAPIDOCR = False

try:
    import pytesseract
    from PIL import Image  # noqa: F401  (pytesseract needs PIL)

    HAS_PYTESSERACT = True
except ImportError:  # pragma: no cover - depends on install
    HAS_PYTESSERACT = False


# Rendering resolution. Measured on a 2-column scan, 110/150/220 DPI all
# recover the same text (589/598/577 words), because the detector rescales
# internally anyway -- so there is nothing to gain from rendering larger.
DEFAULT_DPI = int(os.environ.get("AVANTI_OCR_DPI", "150"))

# Tuning, measured on one page of a dense two-column scan:
#
#   baseline                    38.2s
#   no angle classifier         23.6s
#   + recognition batch of 16   10.4s
#
# All three produced identical text (598 words), so this is a 3.7x speedup
# for free. The angle classifier decides whether a text box is upside down,
# which is wasted work on a paper; batching lets the recognition model run
# over many boxes per inference call instead of six.
_ENGINE_CONFIG = {
    "Global.use_angle_cls": False,
    "Rec.rec_batch_num": 16,
    "Cls.cls_batch_num": 16,
}

INSTALL_HINT = (
    "OCR needs an extra package. Install it with:\n\n"
    "    pip install rapidocr-onnxruntime\n\n"
    "That pulls in rapidocr-onnxruntime, which bundles its own models and "
    "needs no system packages."
)


def _enabled() -> bool:
    """OCR can be switched off per deployment.

    The app idles at about 270MB, but an OCR pass adds a page bitmap plus
    the model's working set on top. On a 512MB free tier that is close
    enough to the ceiling to risk the container being killed mid-request,
    so a host can opt out with AVANTI_ENABLE_OCR=0 and the app degrades to
    reporting scanned PDFs rather than reading them.
    """
    return os.environ.get("AVANTI_ENABLE_OCR", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def backend() -> Optional[str]:
    """Which OCR backend is usable, or None."""
    if not _enabled() or not HAS_RASTERIZER:
        return None
    if HAS_RAPIDOCR:
        return "rapidocr"
    if HAS_PYTESSERACT and _tesseract_binary_present():
        return "pytesseract"
    return None


def backend_label() -> str:
    """Human-readable backend name for the UI."""
    name = backend()
    if name:
        return {
            "rapidocr": "RapidOCR (bundled models)",
            "pytesseract": "Tesseract",
        }[name]
    # Distinguish "switched off here" from "never installed" -- otherwise a
    # deployment that deliberately disabled OCR looks like a broken install.
    if not _enabled():
        return "disabled on this deployment"
    return "not installed"


def available() -> bool:
    return backend() is not None


def _tesseract_binary_present() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ------------------------------------------------------------ engine cache

_engine = None


def _get_engine():
    """RapidOCR loads ONNX models on first use, so build it once."""
    global _engine
    if _engine is None and HAS_RAPIDOCR:
        try:
            _engine = RapidOCR(**_ENGINE_CONFIG)
        except TypeError:
            # An older build may not accept these overrides; correctness
            # does not depend on them, only speed.
            _engine = RapidOCR()
    return _engine


# --------------------------------------------------------- reading order


@dataclass
class Box:
    """One recognized text box."""

    text: str
    x0: float
    x1: float
    top: float
    bottom: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0


def _detect_gutter(boxes: List[Box], width: float) -> Optional[float]:
    """Find a two-column gutter among recognized boxes, or None."""
    if len(boxes) < 12:
        return None

    total = len(boxes)
    best: Optional[Tuple[int, float]] = None
    for pct in range(40, 61):
        split = width * pct / 100.0
        crossing = sum(1 for b in boxes if b.x0 < split - 4 and b.x1 > split + 4)
        left = sum(1 for b in boxes if b.x1 <= split)
        right = sum(1 for b in boxes if b.x0 >= split)
        # OCR boxes are noisier than PDF words, so allow a little crossing.
        if crossing > max(2, int(0.06 * total)):
            continue
        if left < 0.25 * total or right < 0.25 * total:
            continue
        if best is None or crossing < best[0]:
            best = (crossing, split)
    return best[1] if best else None


def _group_lines(boxes: List[Box], y_tol: float = 10.0) -> List[str]:
    """Group boxes into lines by vertical position, reading left to right."""
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda b: (b.top, b.x0))
    groups: List[List[Box]] = [[ordered[0]]]
    for b in ordered[1:]:
        # Compare against the line's average top, since OCR boxes on one
        # line can be offset by a few pixels.
        ref = sum(x.top for x in groups[-1]) / len(groups[-1])
        if abs(b.top - ref) <= y_tol:
            groups[-1].append(b)
        else:
            groups.append([b])

    lines: List[str] = []
    for group in groups:
        group.sort(key=lambda b: b.x0)
        text = " ".join(b.text for b in group).strip()
        if text:
            lines.append(text)
    return lines


def _page_text(boxes: List[Box], width: float) -> str:
    """Reconstruct a page's text from recognized boxes."""
    if not boxes:
        return ""
    gutter = _detect_gutter(boxes, width)
    if gutter is None:
        lines = _group_lines(boxes)
    else:
        left = [b for b in boxes if b.x1 <= gutter]
        right = [b for b in boxes if b.x0 >= gutter]
        lines = _group_lines(left) + _group_lines(right)
    return "\n".join(lines)


# ---------------------------------------------------------------- running


def _recognize(image) -> List[Box]:
    """Run the active backend on one page image."""
    name = backend()

    if name == "rapidocr":
        import numpy as np

        engine = _get_engine()
        result, _ = engine(np.asarray(image))
        boxes: List[Box] = []
        for entry in result or []:
            quad, text = entry[0], entry[1]
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
            if text and text.strip():
                boxes.append(
                    Box(text.strip(), min(xs), max(xs), min(ys), max(ys))
                )
        return boxes

    if name == "pytesseract":
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        boxes = []
        for i, text in enumerate(data.get("text", [])):
            if not text or not text.strip():
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, KeyError, IndexError):
                conf = 0.0
            if conf < 40:      # drop low-confidence noise
                continue
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            boxes.append(Box(text.strip(), x, x + w, y, y + h))
        return boxes

    return []


@dataclass
class OcrResult:
    """Text recovered by OCR, page by page."""

    pages: List[str] = field(default_factory=list)
    backend: str = ""
    pages_processed: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(p.split()) for p in self.pages)


def ocr_pdf(
    file_bytes: bytes,
    max_pages: Optional[int] = None,
    dpi: int = DEFAULT_DPI,
    progress: Optional[Callable[[int, int], None]] = None,
) -> OcrResult:
    """Rasterize a PDF and read every page optically.

    `progress` is called with (page_done, page_total) so a UI can show it.
    """
    result = OcrResult(backend=backend_label())

    if not HAS_RASTERIZER:
        result.warnings.append(
            "pypdfium2 is not installed, so pages cannot be rendered for OCR."
        )
        return result
    if not available():
        result.warnings.append(INSTALL_HINT)
        return result

    try:
        pdf = pdfium.PdfDocument(file_bytes)
    except Exception as e:
        result.warnings.append(f"Could not open the PDF for rendering: {e}")
        return result

    try:
        total = len(pdf)
        limit = min(max_pages or total, total)

        for i in range(limit):
            try:
                page = pdf[i]
                bitmap = page.render(scale=dpi / 72.0)
                image = bitmap.to_pil()
                boxes = _recognize(image)
                result.pages.append(_page_text(boxes, float(image.width)))
            except Exception as e:
                # One unreadable page must not lose the whole document.
                result.pages.append("")
                result.warnings.append(f"Page {i + 1} could not be read: {e}")
            if progress:
                progress(i + 1, limit)

        result.pages_processed = len(result.pages)
    finally:
        try:
            pdf.close()
        except Exception:
            pass

    if result.word_count < 20:
        result.warnings.append(
            "OCR found almost no text. The scan may be too low-resolution, "
            "blank, or rotated."
        )
    return result
