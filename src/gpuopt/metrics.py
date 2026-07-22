from __future__ import annotations

import time
from collections.abc import Callable

from prometheus_client import Counter, Gauge, Histogram, generate_latest

REQUEST_COUNT = Counter("gpuopt_http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQUEST_DURATION = Histogram("gpuopt_http_request_duration_seconds", "HTTP request duration", ["method", "path"])
ACTIVE_REQUESTS = Gauge("gpuopt_http_requests_active", "Active HTTP requests")
PREDICTIONS_TOTAL = Counter("gpuopt_predictions_total", "Total failure predictions", ["result"])
SCHEDULED_JOBS = Counter("gpuopt_scheduled_jobs_total", "Total scheduled jobs", ["status"])
REMEDIATIONS_TOTAL = Counter("gpuopt_remediations_total", "Total auto-remediations", ["action"])
POLICY_EVOLUTIONS = Counter("gpuopt_policy_evolutions_total", "Total policy evolutions")
CLUSTER_GPU_COUNT = Gauge("gpuopt_cluster_gpu_total", "Total GPUs per cluster", ["cluster_id"])
CLUSTER_GPU_UTILIZATION = Gauge("gpuopt_cluster_gpu_utilization", "GPU utilization per cluster", ["cluster_id"])
HEALER_RUNNING = Gauge("gpuopt_healer_running", "Auto-healer monitor running (1=yes)")


def instrument_app(app: object) -> None:
    from fastapi import FastAPI
    from fastapi.routing import APIRoute

    if not isinstance(app, FastAPI):
        return

    for route in app.routes:
        if isinstance(route, APIRoute) and route.path != "/metrics":
            original = route.endpoint

            async def decorator(endpoint: Callable) -> Callable:
                async def wrapper(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
                    method = list(route.methods)[0] if route.methods else "UNKNOWN"
                    path = route.path
                    ACTIVE_REQUESTS.inc()
                    start = time.perf_counter()
                    try:
                        response = await endpoint(*args, **kwargs)
                        REQUEST_COUNT.labels(method=method, path=path, status="200").inc()
                        return response
                    except Exception:
                        REQUEST_COUNT.labels(method=method, path=path, status="500").inc()
                        raise
                    finally:
                        REQUEST_DURATION.labels(method=method, path=path).observe(time.perf_counter() - start)
                        ACTIVE_REQUESTS.dec()

                return wrapper

            if hasattr(original, "__call__"):
                route.endpoint = decorator(original)

    from starlette.responses import Response

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")
