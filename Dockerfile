FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .

# CPU-only torch: the VPS has no GPU, but the default PyPI "torch" wheel pulls in
# multi-GB CUDA libraries that would never be used for inference here. torchvision
# is pulled in transitively (docling-ibm-models/transformers) and must come from the
# same CPU index in the same command, or pip resolves a CUDA-built torchvision that
# mismatches the CPU torch below and fails to import. Installed first so the unpinned
# "torch" line in requirements.txt is already satisfied and pip skips reinstalling it.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY templates/ templates/
COPY static/ static/
COPY storage/ storage/
COPY app.py ratelimit.py ./

# Bake the Granite embedding model into the image at build time (not downloaded at
# runtime, no cache volume needed): reproducible builds, no first-request latency,
# and the model is small enough that this is simpler and more reliable than a
# runtime-download + persisted-volume strategy for a demo of this size.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('ibm-granite/granite-embedding-small-english-r2')"

EXPOSE 8000

# Stdlib-only healthcheck (no curl installed just for this): hits the deliberately
# cheap GET /health route, which does no retrieval/embedding/Ollama work.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# 1 worker: this process loads the Granite embedding model and the persisted index
# once at boot; more workers would each load their own copy of the model in memory
# for no benefit at this traffic scale, and keeps the existing in-memory rate
# limiter's per-process behaviour predictable. Timeout 60s stays safely above the
# ~30s Ollama Cloud request timeout so Gunicorn never kills a still-in-flight request.
CMD ["gunicorn", "app:create_app()", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "60"]
