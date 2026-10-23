"""The two deployments install from different files; their pins must agree.

Streamlit Community Cloud reads requirements.txt and cannot be pointed at
another file. The Docker image (Render) installs requirements-server.txt,
which adds the HTTP API and OCR. Both run the same engine, so the engine's
pins have to match, or the two live apps quietly read papers differently.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pins(name: str) -> dict:
    pins = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s;]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def test_engine_pins_match_across_deployments():
    streamlit_app, server = _pins("requirements.txt"), _pins("requirements-server.txt")
    shared = set(streamlit_app) & set(server)
    assert {"pdfplumber", "pypdf", "scikit-learn", "requests"} <= shared
    for package in shared:
        assert streamlit_app[package] == server[package], package


def test_dev_installs_the_same_api_versions_as_the_server():
    dev, server = _pins("requirements-dev.txt"), _pins("requirements-server.txt")
    for package in ("fastapi", "uvicorn", "python-multipart"):
        assert dev[package] == server[package], package


def test_streamlit_cloud_does_not_install_ocr():
    """rapidocr pulls in ~100MB and broke the Streamlit Cloud build."""
    pins = _pins("requirements.txt")
    assert not any("rapidocr" in p or "onnxruntime" in p or "opencv" in p for p in pins)
