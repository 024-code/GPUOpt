from __future__ import annotations

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


class TestMLAnomalyDetector:
    def test_detect_anomalies(self, client: TestClient):
        cid = client.post("/api/v1/clusters", json={
            "name": "anomaly-test",
            "environment": "test",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()["id"]
        for _ in range(5):
            client.post(f"/api/v1/clusters/{cid}/state")
        resp = client.post(f"/api/v1/anomaly/detect/{cid}")
        assert resp.status_code == 200
        data = resp.json()
        assert "scores" in data or "anomaly_count" in data

    def test_anomaly_history(self, client: TestClient):
        resp = client.get("/api/v1/anomaly/history")
        assert resp.status_code == 200


class TestWebSocketStreaming:
    def test_ws_status(self, client: TestClient):
        resp = client.get("/api/v1/ws/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_connections" in data

    def test_ws_state_stream(self, client: TestClient):
        cid = client.post("/api/v1/clusters", json={
            "name": "ws-test",
            "environment": "test",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()["id"]
        client.post(f"/api/v1/clusters/{cid}/state")
        with client.websocket_connect(f"/api/v1/ws/state/{cid}") as ws:
            data = ws.receive_json()
            assert data["type"] in ("state_update", "heartbeat")

    def test_ws_metrics_stream(self, client: TestClient):
        cid = client.post("/api/v1/clusters", json={
            "name": "ws-metrics-test",
            "environment": "test",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()["id"]
        client.post(f"/api/v1/clusters/{cid}/state")
        with client.websocket_connect(f"/api/v1/ws/metrics/{cid}") as ws:
            data = ws.receive_json()
            assert data["type"] == "metrics_update"


class TestRemediationEngine:
    def test_create_rule(self, client: TestClient):
        resp = client.post("/api/v1/remediation/rules", json={
            "name": "test-rule",
            "description": "Test",
            "trigger_alert_severity": ["critical"],
            "actions": [{"type": "send_notification", "target": "admin", "params": {"channel": "slack"}}],
            "enabled": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-rule"

    def test_list_rules(self, client: TestClient):
        resp = client.get("/api/v1/remediation/rules")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_rule_not_found(self, client: TestClient):
        resp = client.get("/api/v1/remediation/rules/nonexistent")
        assert resp.status_code == 404

    def test_update_rule(self, client: TestClient):
        resp = client.post("/api/v1/remediation/rules", json={
            "name": "update-test",
            "trigger_alert_severity": ["warning"],
        })
        rid = resp.json()["id"]
        resp = client.patch(f"/api/v1/remediation/rules/{rid}", json={"name": "updated"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "updated"

    def test_delete_rule(self, client: TestClient):
        resp = client.post("/api/v1/remediation/rules", json={
            "name": "delete-test",
            "trigger_alert_severity": ["critical"],
        })
        rid = resp.json()["id"]
        resp = client.delete(f"/api/v1/remediation/rules/{rid}")
        assert resp.status_code == 204

    def test_list_runs(self, client: TestClient):
        resp = client.get("/api/v1/remediation/runs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestRBAC:
    def test_create_user(self, client: TestClient):
        resp = client.post("/api/v1/rbac/users?username=testuser&email=test@test.com&role=viewer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "testuser"
        assert "api_key" in data

    def test_list_users(self, client: TestClient):
        resp = client.get("/api/v1/rbac/users")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_user_not_found(self, client: TestClient):
        resp = client.get("/api/v1/rbac/users/nonexistent")
        assert resp.status_code == 404

    def test_rotate_key(self, client: TestClient):
        resp = client.post("/api/v1/rbac/users?username=keytest&email=key@test.com&role=admin")
        uid = resp.json()["user_id"]
        resp = client.post(f"/api/v1/rbac/users/{uid}/rotate-key")
        assert resp.status_code == 200
        assert "api_key" in resp.json()

    def test_list_roles(self, client: TestClient):
        resp = client.get("/api/v1/rbac/roles")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_permissions(self, client: TestClient):
        resp = client.post("/api/v1/rbac/users?username=permtest&email=perm@test.com&role=admin")
        uid = resp.json()["user_id"]
        resp = client.get(f"/api/v1/rbac/permissions/{uid}")
        assert resp.status_code == 200
        data = resp.json()
        assert "permissions" in data


class TestCloudPricing:
    def test_list_providers(self, client: TestClient):
        resp = client.get("/api/v1/cloud/providers")
        assert resp.status_code == 200
        assert "aws" in resp.json()["providers"]

    def test_get_pricing(self, client: TestClient):
        resp = client.get("/api/v1/cloud/pricing/aws")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_pricing_invalid(self, client: TestClient):
        resp = client.get("/api/v1/cloud/pricing/invalid")
        assert resp.status_code == 400

    def test_compare_gpu(self, client: TestClient):
        resp = client.get("/api/v1/cloud/compare?gpu_model=A100")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_estimate_cost(self, client: TestClient):
        resp = client.get("/api/v1/cloud/estimate?gpu_model=H100&gpu_count=8&provider=aws")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gpu_model"] == "H100"
        assert data["gpu_count"] == 8


class TestBenchmark:
    def test_benchmark(self, client: TestClient):
        cid = client.post("/api/v1/clusters", json={
            "name": "bench-test",
            "environment": "test",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()["id"]
        client.post(f"/api/v1/clusters/{cid}/state")
        resp = client.get("/api/v1/benchmark")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "cluster_count" in data["results"]

    def test_ci_health(self, client: TestClient):
        resp = client.get("/api/v1/ci/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_api_version(self, client: TestClient):
        resp = client.get("/api/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "current_version" in data
