# aarav: the model, the OpenAI-compatible API and the web app in one small CPU image (~1 GB, mostly PyTorch).
# The model files are mounted, not baked in (see docker-compose.yml), so a new model is a file change, not an image rebuild.
# NOT BUILT OR RUN on the development machine (no Docker there): treat the first build as a test.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 AARAV_LM_HOME=/app HOME=/tmp
WORKDIR /app

# CPU-only PyTorch: a 5-15M parameter model decodes faster on CPU than on a GPU (docs/PERFORMANCE.md), so no CUDA is needed.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -e ".[api]"
COPY scripts ./scripts
COPY artifacts ./artifacts
COPY models ./models

RUN useradd --system --uid 10001 --home-dir /tmp aarav && mkdir -p /data && chown aarav /data
USER aarav
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# --trust-proxy: the address and scheme come from the reverse proxy in front (Caddy); never expose this container's port directly.
# --cookie-secure: session cookies are only ever sent over HTTPS.
ENTRYPOINT ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000", "--db", "/data/accounts.db", "--trust-proxy", "--cookie-secure"]
CMD ["--checkpoint", "models/tiny-tinystories-v1-sft-int8.pt"]
