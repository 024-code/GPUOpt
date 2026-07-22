from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from gpuopt.healing.auto_healer import AutoHealer
from gpuopt.healing.router import _get_healer, _monitor_thread, _monitor_stop


class TestHealingMonitorAPI:
    def test_start_monitor(self, client: TestClient):
        resp = client.post("/api/v1/healing/start", json={
            "cluster_id": "test-cluster",
            "interval_seconds": 3600,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healing_started"
        assert data["cluster_id"] == "test-cluster"

    def test_start_monitor_idempotent(self, client: TestClient):
        client.post("/api/v1/healing/start", json={
            "cluster_id": "test",
            "interval_seconds": 3600,
        })
        resp = client.post("/api/v1/healing/start", json={
            "cluster_id": "test",
            "interval_seconds": 3600,
        })
        data = resp.json()
        assert data["status"] == "already_running"

    def test_status_running(self, client: TestClient):
        client.post("/api/v1/healing/start", json={
            "cluster_id": "test",
            "interval_seconds": 3600,
        })
        resp = client.get("/api/v1/healing/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "monitor_running" in data

    def test_status_not_running(self, client: TestClient):
        resp = client.get("/api/v1/healing/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "monitor_running" in data

    def test_stop_monitor(self, client: TestClient):
        client.post("/api/v1/healing/start", json={
            "cluster_id": "test",
            "interval_seconds": 3600,
        })
        resp = client.post("/api/v1/healing/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"

    def test_stop_monitor_not_running(self, client: TestClient):
        resp = client.post("/api/v1/healing/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_running"


class TestHealingMonitorLifecycle:
    def test_monitor_actually_runs(self):
        import gpuopt.healing.router as hr

        hr._monitor_stop.clear()
        hr._monitor_thread = None
        healer = hr._get_healer()
        healer.clear_history()

        from gpuopt.main import app
        with TestClient(app) as c:
            c.post("/api/v1/healing/start", json={
                "cluster_id": "test-monitor",
                "interval_seconds": 1,
            })
            time.sleep(2.5)
            hr._monitor_stop.set()
            if hr._monitor_thread and hr._monitor_thread.is_alive():
                hr._monitor_thread.join(timeout=5)
            hr._monitor_thread = None

        assert len(healer.remediation_history) > 0
