from __future__ import annotations

import json
import random
import threading
import time

import pytest
from fastapi.testclient import TestClient


class TestChaosEndpoints:
    CONCURRENT_REQUESTS = 20

    def test_concurrent_health_checks(self, client: TestClient):
        errors = []

        def hit() -> None:
            try:
                resp = client.get("/health/live")
                if resp.status_code != 200:
                    errors.append(f"status={resp.status_code}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=hit) for _ in range(self.CONCURRENT_REQUESTS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors, f"Concurrent health check errors: {errors}"

    def test_concurrent_predictions(self, client: TestClient):
        errors = []

        def predict() -> None:
            try:
                resp = client.post("/api/v1/predictor/predict", json={
                    "temperature": random.uniform(30, 95),
                    "gpu_utilization": random.uniform(10, 100),
                    "memory_utilization": random.uniform(20, 98),
                    "ecc_errors": random.randint(0, 20),
                    "xid_errors": random.randint(0, 10),
                })
                if resp.status_code != 200:
                    errors.append(f"predict status={resp.status_code}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=predict) for _ in range(self.CONCURRENT_REQUESTS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not errors, f"Concurrent predict errors: {errors}"

    def test_rapid_start_stop_healer(self, client: TestClient):
        for i in range(5):
            start = client.post("/api/v1/healing/start", json={"cluster_id": f"chaos-{i}", "interval_seconds": 60})
            assert start.status_code == 200
            stop = client.post("/api/v1/healing/stop")
            assert stop.status_code == 200

    def test_rapid_start_stop_k8s_controller(self, client: TestClient):
        for i in range(5):
            start = client.post("/api/v1/k8s/controller/start?poll_interval=60")
            assert start.status_code == 200
            stop = client.post("/api/v1/k8s/controller/stop")
            assert stop.status_code == 200

    def test_large_payload_evolution(self, client: TestClient):
        metrics = [{"gpu_utilization": random.uniform(0, 1), "failure_rate": random.uniform(0, 0.5),
                     "temperature": random.uniform(30, 95)} for _ in range(100)]
        resp = client.post("/api/v1/policy/evolve", json=metrics)
        assert resp.status_code == 200

    def test_malformed_requests_return_422(self, client: TestClient):
        bad_inputs = [
            ("/api/v1/predictor/train", {"telemetry_data": "not-a-list", "labels": "not-a-list"}),
            ("/api/v1/healing/check-health", "not-a-dict"),
            ("/api/v1/k8s/actions/execute", {"spec": {"actionType": "invalid"}}),
        ]
        for path, payload in bad_inputs:
            resp = client.post(path, json=payload)
            assert resp.status_code == 422, f"{path} returned {resp.status_code} instead of 422"
