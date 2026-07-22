from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gpuopt.versioning import (
    V1_PREFIX,
    V2_PREFIX,
    V1DeprecationMiddleware,
    build_v1_deprecation_header,
    create_v2_router,
    mount_v2,
)


class TestVersioningCore:
    def test_v1_prefix(self):
        assert V1_PREFIX == "/api/v1"

    def test_v2_prefix(self):
        assert V2_PREFIX == "/api/v2"

    def test_build_deprecation_header(self):
        headers = build_v1_deprecation_header()
        assert "X-API-Version" in headers
        assert "X-API-Deprecation" in headers
        assert "v1 will be removed" in headers["X-API-Deprecation"]
        assert "/api/v2" in headers["X-API-Deprecation"]

    def test_create_v2_router(self):
        router = create_v2_router()
        assert router.prefix == V2_PREFIX
        assert len(router.routes) > 0


class TestV2API:
    def test_health_endpoint(self, client: TestClient):
        resp = client.get("/api/v2/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_version_endpoint(self, client: TestClient):
        resp = client.get("/api/v2/version")
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_version"] == "2"
        assert "version" in data

    def test_clusters_list_empty(self, client: TestClient):
        resp = client.get("/api/v2/clusters")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_clusters_get_nonexistent(self, client: TestClient):
        resp = client.get("/api/v2/clusters/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 200
        assert resp.json()["error"] == "not_found"

    def test_schedule_no_nodes(self, client: TestClient):
        resp = client.post("/api/v2/schedule", json={"id": "j1", "required_gpus": 1, "nodes": []})
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_schedule_with_nodes(self, client: TestClient):
        resp = client.post("/api/v2/schedule", json={
            "id": "j1", "required_gpus": 1, "nodes": [{"id": "n1", "available_gpus": 8, "total_gpus": 8, "free_memory_gb": 64, "temperature": 45, "gpu_model": "A100", "status": "ready"}],
        })
        assert resp.status_code == 200
        assert resp.json()["status"] in ("scheduled", "queued")

    def test_scheduler_metrics(self, client: TestClient):
        resp = client.get("/api/v2/scheduler-metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "q_table_size" in data

    def test_train_scheduler(self, client: TestClient):
        resp = client.post("/api/v2/train-scheduler?episodes=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "training_complete"

    def test_domain_counts(self, client: TestClient):
        resp = client.get("/api/v2/domains")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_domain_collect(self, client: TestClient):
        resp = client.post("/api/v2/domains/collect", json={"cluster_id": "c1", "node": "n1", "gpu_count": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert "gpu_telemetry" in data
        assert "fabric_telemetry" in data

    def test_domain_query_invalid(self, client: TestClient):
        resp = client.get("/api/v2/domains/invalid_type?limit=5")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_domain_query_valid(self, client: TestClient):
        resp = client.get("/api/v2/domains/gpu?limit=5")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_predict_untrained(self, client: TestClient):
        resp = client.post("/api/v2/predict", json={"temperature": 50, "gpu_utilization": 50})
        assert resp.status_code == 200
        data = resp.json()
        assert "failure_predicted" in data

    def test_policy_evolve(self, client: TestClient):
        resp = client.post("/api/v2/policy/evolve", json=[{"gpu_utilization": 0.7, "failure_rate": 0.05}])
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "evolution_complete"

    def test_policy_best(self, client: TestClient):
        resp = client.get("/api/v2/policy/best")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("success", "no_policy_evolved_yet")

    def test_healing_check(self, client: TestClient):
        resp = client.post("/api/v2/healing/check", json={"temperature": 50})
        assert resp.status_code == 200
        data = resp.json()
        assert "health_status" in data

    def test_healing_execute(self, client: TestClient):
        resp = client.post("/api/v2/healing/execute?node_id=n1", json={"temperature": 90, "ecc_errors": 25})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "executed"

    def test_healing_history(self, client: TestClient):
        resp = client.get("/api/v2/healing/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_governance_models_list(self, client: TestClient):
        resp = client.get("/api/v2/governance/models")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_optimize_endpoint(self, client: TestClient):
        resp = client.post("/api/v2/optimize", json={
            "workloads": [],
            "nodes": [{"id": "n1", "gpu_model": "A100", "available_gpus": 8, "total_gpus": 8, "free_memory_gb": 64, "temperature": 45, "status": "ready"}],
            "candidates": [],
            "tenant_profiles": {},
        })
        assert resp.status_code == 200


class TestV1DeprecationHeaders:
    def test_v1_endpoint_has_deprecation_header(self, client: TestClient):
        resp = client.get("/api/v1/domains/counts")
        assert resp.status_code == 200
        assert "X-API-Version" in resp.headers
        assert resp.headers["X-API-Version"] == "2"
        assert "X-API-Deprecation" in resp.headers
        assert "v1 will be removed" in resp.headers["X-API-Deprecation"]

    def test_v2_endpoint_no_deprecation_header(self, client: TestClient):
        resp = client.get("/api/v2/health")
        assert resp.status_code == 200
        assert "X-API-Version" not in resp.headers
        assert "X-API-Deprecation" not in resp.headers

    def test_non_api_endpoint_no_deprecation_header(self, client: TestClient):
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "X-API-Deprecation" not in resp.headers

    def test_v1_post_endpoint_has_deprecation_header(self, client: TestClient):
        resp = client.post("/api/v1/policy/evolve", json=[{"gpu_utilization": 0.7}])
        assert resp.status_code == 200
        assert resp.headers["X-API-Version"] == "2"
        assert "v1 will be removed" in resp.headers["X-API-Deprecation"]
