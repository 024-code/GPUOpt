from __future__ import annotations

import multiprocessing
import os

bind = os.getenv("GPUOPT_API_HOST", "0.0.0.0") + ":" + os.getenv("GPUOPT_API_PORT", "8080")
workers = int(os.getenv("GPUOPT_API_WORKERS", str(multiprocessing.cpu_count() * 2 + 1)))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = int(os.getenv("GPUOPT_GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(os.getenv("GPUOPT_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GPUOPT_KEEPALIVE", "5"))
max_requests = int(os.getenv("GPUOPT_MAX_REQUESTS", "10000"))
max_requests_jitter = int(os.getenv("GPUOPT_MAX_REQUESTS_JITTER", "1000"))
preload_app = True
reload = os.getenv("GPUOPT_ENV", "production") != "production"
accesslog = os.getenv("GPUOPT_ACCESS_LOG", "-")
errorlog = os.getenv("GPUOPT_ERROR_LOG", "-")
loglevel = os.getenv("GPUOPT_LOG_LEVEL", "info").lower()
