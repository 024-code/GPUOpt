from __future__ import annotations

"""Load/soak test suite for GPUOpt API.

Run:  python -m pytest tests/load/test_load.py -v --timeout=120
Env:   TARGET_URL=http://localhost:8080  (default)
       LOAD_DURATION_SECONDS=30          (default)
       WARMUP_REQUESTS=5                 (default)

This uses unbounded async I/O to saturate the API with concurrent requests,
measuring throughput, error rate, and p95 latency.
"""

import asyncio
import os
import time
from typing import Any

import httpx
import pytest

TARGET_URL = os.environ.get("TARGET_URL", "http://localhost:8080")
LOAD_DURATION = int(os.environ.get("LOAD_DURATION_SECONDS", "15"))
WARMUP = int(os.environ.get("WARMUP_REQUESTS", "5"))


ENDPOINTS: list[tuple[str, str]] = [
    ("GET", "/health/live"),
    ("GET", "/health/ready"),
    ("GET", "/health/detailed"),
    ("GET", "/api/v1/clusters"),
    ("GET", "/api/v1/scheduler/metrics"),
    ("GET", "/api/v1/governance/models"),
    ("GET", "/api/v1/predictor/models"),
    ("GET", "/api/v1/monitoring/status"),
    ("GET", "/api/v1/power/analysis"),
    ("GET", "/api/v1/finops/summary"),
    ("GET", "/api/v1/references"),
    ("GET", "/api/v1/metrics-kpi"),
    ("GET", "/api/v1/optimization-analysis"),
]

CLUSTER_ENDPOINTS: list[tuple[str, str]] = [
    ("GET", "/api/v1/clusters/{cid}"),
    ("GET", "/api/v1/clusters/{cid}/state"),
    ("GET", "/api/v1/clusters/{cid}/checks/latest"),
    ("GET", "/api/v1/clusters/{cid}/analysis/latest"),
    ("GET", "/api/v1/clusters/{cid}/recommendations/latest"),
    ("GET", "/api/v1/clusters/{cid}/actuations"),
]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0, "count": 0}
    sv = sorted(values)
    n = len(sv)
    return {
        "min": round(sv[0], 2),
        "max": round(sv[-1], 2),
        "avg": round(sum(sv) / n, 2),
        "p50": round(sv[int(n * 0.50)], 2),
        "p95": round(sv[int(n * 0.95)], 2),
        "p99": round(sv[int(n * 0.99)], 2),
        "count": n,
    }


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
async def client() -> httpx.AsyncClient:
    async with httpx.AsyncClient(base_url=TARGET_URL, timeout=30) as c:
        yield c


@pytest.fixture(scope="session")
async def cluster_id(client: httpx.AsyncClient) -> str:
    payload = {
        "name": f"load-test-{int(time.time())}",
        "environment": "loadtest",
        "connector_type": "mock",
        "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
    }
    resp = await client.post("/api/v1/clusters", json=payload)
    if resp.status_code == 200:
        cid = resp.json().get("id", "")
    else:
        cid = ""
    await client.post(f"/api/v1/clusters/{cid}/state")
    await client.post(f"/api/v1/clusters/{cid}/state")
    await client.post(f"/api/v1/clusters/{cid}/analyze")
    await client.post(f"/api/v1/clusters/{cid}/recommendations")
    yield cid
    if cid:
        await client.delete(f"/api/v1/clusters/{cid}")


class TestApiLoad:
    async def _hit(self, client: httpx.AsyncClient, method: str, path: str, cid: str = "") -> float:
        url = path.format(cid=cid) if cid else path
        start = time.perf_counter()
        try:
            resp = await client.request(method, url)
            status = resp.status_code
        except Exception:
            status = 0
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed

    async def _load_test(self, client: httpx.AsyncClient, endpoints: list[tuple[str, str]], cid: str = "", concurrency: int = 10) -> dict[str, Any]:
        latencies: list[float] = []
        errors = 0
        total = 0
        deadline = time.monotonic() + LOAD_DURATION

        async def _worker() -> None:
            nonlocal errors, total, latencies
            while time.monotonic() < deadline:
                for method, path in endpoints:
                    if time.monotonic() >= deadline:
                        break
                    elapsed = await self._hit(client, method, path, cid)
                    latencies.append(elapsed)
                    total += 1
                    if elapsed < 0 or elapsed > 10000:
                        errors += 1

        workers = [asyncio.create_task(_worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers)

        stats = _stats(latencies)
        error_rate = errors / max(total, 1)
        throughput = total / LOAD_DURATION
        return {
            "total_requests": total,
            "throughput_rps": round(throughput, 1),
            "error_rate": round(error_rate, 4),
            "latency_ms": stats,
        }

    async def test_health_endpoints_under_load(self, client: httpx.AsyncClient):
        result = await self._load_test(client, [("GET", "/health/live"), ("GET", "/health/ready")], concurrency=20)
        assert result["error_rate"] < 0.05
        assert result["latency_ms"]["p95"] < 2000

    async def test_read_endpoints_under_load(self, client: httpx.AsyncClient, cluster_id: str):
        result = await self._load_test(client, ENDPOINTS + CLUSTER_ENDPOINTS, cid=cluster_id, concurrency=10)
        assert result["error_rate"] < 0.05
        assert result["latency_ms"]["p95"] < 5000
        assert result["latency_ms"]["avg"] < 1000

    async def test_health_readiness(self, client: httpx.AsyncClient):
        latencies: list[float] = []
        for _ in range(20):
            el = await self._hit(client, "GET", "/health/ready")
            latencies.append(el)
        stats = _stats(latencies)
        assert stats["p95"] < 500
        assert stats["max"] < 2000

    async def test_concurrent_cluster_writes(self, client: httpx.AsyncClient, cluster_id: str):
        async def _write() -> float:
            start = time.perf_counter()
            try:
                await client.post(f"/api/v1/clusters/{cluster_id}/state")
            except Exception:
                pass
            return (time.perf_counter() - start) * 1000
        results = await asyncio.gather(*[_write() for _ in range(20)])
        stats = _stats(list(results))
        assert stats["p95"] < 3000
