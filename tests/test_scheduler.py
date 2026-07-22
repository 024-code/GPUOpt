from __future__ import annotations


class TestForecast:
    def test_forecast_no_traces(self, client):
        payload = {"name": "sched-fc-no-tr", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        resp = client.post(f"/api/v1/clusters/{cluster_id}/scheduler/forecast")
        assert resp.status_code == 404

    def test_forecast_with_traces(self, client):
        payload = {"name": "sched-fc-tr", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        resp = client.post(f"/api/v1/clusters/{cluster_id}/scheduler/forecast?horizon_hours=6")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_name"] == "sched-fc-tr"
        assert data["trace_count"] == 2
        assert data["horizon_hours"] == 6
        assert len(data["forecast_points"]) == 6
        assert "summary" in data

    def test_forecast_with_data(self, client):
        payload = {
            "name": "sched-fc-data",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        resp = client.post(f"/api/v1/clusters/{cluster_id}/scheduler/forecast?horizon_hours=12")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace_count"] >= 2
        assert data["predicted_avg_utilization_percent"] >= 0
        assert data["predicted_idle_gpus"] >= 0
        assert len(data["forecast_points"]) == 12

    def test_forecast_not_found(self, client):
        resp = client.post("/api/v1/clusters/00000000-0000-0000-0000-000000000000/scheduler/forecast")
        assert resp.status_code == 404


class TestPlacement:
    def test_placement_no_state(self, client):
        payload = {"name": "sched-pl-no-st", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        req = {"gpu_count": 1, "gpu_memory_bytes": 0, "cpu_millicores": 1000, "memory_bytes": 0}
        resp = client.post(f"/api/v1/clusters/{cluster_id}/scheduler/placement", json=req)
        assert resp.status_code == 404

    def test_placement_success(self, client):
        payload = {
            "name": "sched-pl-ok",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        req = {"gpu_count": 1, "gpu_memory_bytes": 0, "cpu_millicores": 1000, "memory_bytes": 0}
        resp = client.post(f"/api/v1/clusters/{cluster_id}/scheduler/placement", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_name"] == "sched-pl-ok"
        assert data["suggested_node"] != ""
        assert data["confidence"] >= 0
        assert "reasoning" in data
        assert isinstance(data["alternative_nodes"], list)
        assert data["score"] >= 0

    def test_placement_with_memory_constraint(self, client):
        payload = {
            "name": "sched-pl-mem",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        req = {"gpu_count": 1, "gpu_memory_bytes": 1024 * 1024 * 1024, "cpu_millicores": 2000, "memory_bytes": 0}
        resp = client.post(f"/api/v1/clusters/{cluster_id}/scheduler/placement", json=req)
        assert resp.status_code == 200
        assert resp.json()["suggested_node"] != ""

    def test_placement_not_found(self, client):
        req = {"gpu_count": 1, "gpu_memory_bytes": 0, "cpu_millicores": 1000, "memory_bytes": 0}
        resp = client.post("/api/v1/clusters/00000000-0000-0000-0000-000000000000/scheduler/placement", json=req)
        assert resp.status_code == 404


class TestSimulate:
    def test_simulate_success(self, client):
        payload = {
            "name": "sched-sim",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        req = {"gpu_count": 1, "gpu_memory_bytes": 0, "cpu_millicores": 1000, "memory_bytes": 0}
        resp = client.post(f"/api/v1/clusters/{cluster_id}/scheduler/simulate", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_name"] == "sched-sim"
        assert data["placement"]["suggested_node"] != ""
        assert data["projected_utilization_delta"] >= 0
        assert data["risk_score"] >= 0
        assert "summary" in data

    def test_simulate_no_state(self, client):
        payload = {"name": "sched-sim-no-st", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        req = {"gpu_count": 1, "gpu_memory_bytes": 0, "cpu_millicores": 1000, "memory_bytes": 0}
        resp = client.post(f"/api/v1/clusters/{cluster_id}/scheduler/simulate", json=req)
        assert resp.status_code == 404

    def test_simulate_not_found(self, client):
        req = {"gpu_count": 1, "gpu_memory_bytes": 0, "cpu_millicores": 1000, "memory_bytes": 0}
        resp = client.post("/api/v1/clusters/00000000-0000-0000-0000-000000000000/scheduler/simulate", json=req)
        assert resp.status_code == 404


class TestSchedulingPlan:
    def test_plan_success(self, client):
        payload = {
            "name": "sched-plan",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        resp = client.get(f"/api/v1/clusters/{cluster_id}/scheduler/plan")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_name"] == "sched-plan"
        assert data["total_gpus"] >= 0
        assert data["free_gpus"] >= 0
        assert data["node_count"] >= 0
        assert "summary" in data

    def test_plan_no_state(self, client):
        payload = {"name": "sched-plan-no", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        resp = client.get(f"/api/v1/clusters/{cluster_id}/scheduler/plan")
        assert resp.status_code == 404

    def test_plan_not_found(self, client):
        resp = client.get("/api/v1/clusters/00000000-0000-0000-0000-000000000000/scheduler/plan")
        assert resp.status_code == 404
