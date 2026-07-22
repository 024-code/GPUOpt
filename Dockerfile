FROM python:3.12-slim AS builder

LABEL org.opencontainers.image.title="GPUOpt Backend" \
      org.opencontainers.image.description="Kubernetes-connected GPU cluster environment checker" \
      org.opencontainers.image.vendor="anomalyco" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/anomalyco/GPUOpt" \
      org.opencontainers.image.base.name="python:3.12-slim"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --upgrade pip && \
    pip install .[postgres,notifications] gunicorn && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="GPUOpt Backend" \
      org.opencontainers.image.description="Kubernetes-connected GPU cluster environment checker" \
      org.opencontainers.image.vendor="anomalyco" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/anomalyco/GPUOpt"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GPUOPT_ENV=production

RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 gpuopt && useradd --uid 10001 --gid gpuopt --create-home gpuopt

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app
WORKDIR /app

USER 10001
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health/ready')"

ENTRYPOINT ["gunicorn", "gpuopt.main:app", "-c", "gunicorn.conf.py"]
