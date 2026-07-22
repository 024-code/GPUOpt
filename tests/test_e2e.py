from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestE2EWorkflow:
    def test_full_workflow(self, client: TestClient):
        cluster_payload = {"name": "e2e-cluster", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=cluster_payload)
        assert created.status_code in (200, 201), created.text
        cluster_id = created.json()["id"]

        health = client.get("/health/live")
        assert health.status_code == 200

        checks = client.post(f"/api/v1/clusters/{cluster_id}/checks")
        assert checks.status_code == 200

        state = client.get(f"/api/v1/clusters/{cluster_id}/state")
        assert state.status_code in (200, 404)

    def test_evolve_and_deploy_policy(self, client: TestClient):
        evolve = client.post("/api/v1/policy/evolve", json=[{"gpu_utilization": 0.7, "failure_rate": 0.05, "temperature": 45}])
        assert evolve.status_code == 200
        assert evolve.json()["status"] == "evolution_complete"

        deploy = client.post("/api/v1/policy/deploy-policy")
        assert deploy.status_code == 200
        assert "template" in deploy.json()

    def test_train_predictor_then_predict(self, client: TestClient):
        import random

        telemetry = []
        labels = []
        for _ in range(150):
            will_fail = random.random() < 0.3
            telemetry.append({
                "gpu_utilization": random.uniform(70, 100) if will_fail else random.uniform(10, 60),
                "memory_utilization": random.uniform(80, 100) if will_fail else random.uniform(20, 70),
                "temperature": random.uniform(70, 95) if will_fail else random.uniform(30, 60),
                "ecc_errors": random.randint(5, 20) if will_fail else random.randint(0, 3),
                "xid_errors": random.randint(2, 10) if will_fail else 0,
            })
            labels.append(1 if will_fail else 0)

        train = client.post("/api/v1/predictor/train", json={"telemetry_data": telemetry, "labels": labels})
        assert train.status_code == 200
        assert train.json()["status"] == "training_complete"

        predict = client.post("/api/v1/predictor/predict", json={"temperature": 50, "gpu_utilization": 50})
        assert predict.status_code == 200
        assert "probability" in predict.json()

    def test_k8s_operator_controller_lifecycle(self, client: TestClient):
        status = client.get("/api/v1/k8s/controller/status")
        assert status.status_code == 200
        assert "running" in status.json()

        start = client.post("/api/v1/k8s/controller/start?poll_interval=60")
        assert start.status_code == 200
        assert start.json()["status"] == "controller_started"

        stop = client.post("/api/v1/k8s/controller/stop")
        assert stop.status_code == 200
        assert stop.json()["status"] == "controller_stopped"

    def test_healer_monitor_lifecycle(self, client: TestClient):
        start = client.post("/api/v1/healing/start", json={"cluster_id": "e2e-cluster", "interval_seconds": 60})
        assert start.status_code == 200
        assert start.json()["status"] == "healing_started"

        status = client.get("/api/v1/healing/status")
        assert status.status_code == 200
        assert status.json()["monitor_running"] is True

        stop = client.post("/api/v1/healing/stop")
        assert stop.status_code == 200
        assert stop.json()["status"] == "stopped"

    def test_rl_scheduler_integration(self, client: TestClient):
        schedule = client.post("/api/v1/scheduler/rl/schedule", json={
            "id": "e2e-job",
            "required_gpus": 2,
            "priority": 5,
            "nodes": [
                {"id": "node-0", "available_gpus": 8, "total_gpus": 8, "free_memory_gb": 64},
                {"id": "node-1", "available_gpus": 4, "total_gpus": 8, "free_memory_gb": 32},
            ],
        })
        assert schedule.status_code == 200
        assert schedule.json()["status"] in ("scheduled", "queued")
