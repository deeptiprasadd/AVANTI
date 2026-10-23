# AVANTI -- research paper analyzer.
#
# One image serving the API and the web UI. Sized for a free CPU host: no
# Streamlit, so none of pandas/pyarrow/altair come along.
#
#   docker build -t avanti .
#   docker run -p 7860:7860 avanti
#
# Render builds this as-is (see render.yaml) and injects its own $PORT.

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

# requirements.txt is the Streamlit app's list. The image installs its own,
# which adds the HTTP API and OCR and leaves Streamlit out.
COPY requirements-server.txt ./

# rapidocr depends on opencv-python, which needs libGL and friends that a
# slim image does not have -- the classic "ImportError: libGL.so.1" on
# deploy. Swapping in the headless build fixes that and is ~30MB smaller.
# Left unpinned because the right version tracks whatever rapidocr pulls;
# pin it once your first build succeeds if you want byte-identical images.
RUN pip install -r requirements-server.txt \
 && pip uninstall -y opencv-python \
 && pip install opencv-python-headless

COPY avanti/ ./avanti/
COPY server/ ./server/

# 7860 is the default for a local `docker run`; Render overrides $PORT.
ENV PORT=7860 \
    HOME=/app \
    MPLCONFIGDIR=/tmp \
    OMP_NUM_THREADS=2

EXPOSE 7860

# Run as a non-root user: the app needs no privileges, so it gets none.
RUN useradd --create-home --uid 1000 app \
 && chown -R app:app /app
USER app

HEALTHCHECK --interval=60s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

# Shell form so ${PORT} is expanded by the host's environment.
CMD uvicorn server.main:app --host 0.0.0.0 --port ${PORT} --workers 1
