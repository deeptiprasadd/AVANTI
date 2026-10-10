# AVANTI -- research paper analyzer.
#
# One image serving the API and (once built) the web UI. Sized for a free
# CPU host: no Streamlit, so none of pandas/pyarrow/altair come along.
#
#   docker build -t avanti .
#   docker run -p 7860:7860 avanti
#
# Hugging Face Spaces runs this as-is; it expects port 7860.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl is only here for the healthcheck below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# requirements-ocr.txt pulls in requirements.txt via its own -r line.
COPY requirements.txt requirements-ocr.txt ./

# rapidocr depends on opencv-python, which needs libGL and friends that a
# slim image does not have -- the classic "ImportError: libGL.so.1" on
# deploy. Swapping in the headless build fixes that and is ~30MB smaller.
# Left unpinned because the right version tracks whatever rapidocr pulls;
# pin it once your first build succeeds if you want byte-identical images.
RUN pip install -r requirements-ocr.txt \
 && pip uninstall -y opencv-python \
 && pip install opencv-python-headless

COPY avanti/ ./avanti/
COPY server/ ./server/

# Hugging Face Spaces expects 7860; other hosts inject their own $PORT.
ENV PORT=7860 \
    HOME=/app \
    MPLCONFIGDIR=/tmp \
    OMP_NUM_THREADS=2

EXPOSE 7860

# Spaces runs containers as a non-root user; match that locally so the two
# behave the same.
RUN useradd --create-home --uid 1000 app \
 && chown -R app:app /app
USER app

HEALTHCHECK --interval=60s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

# Shell form so ${PORT} is expanded by the host's environment.
CMD uvicorn server.main:app --host 0.0.0.0 --port ${PORT} --workers 1
