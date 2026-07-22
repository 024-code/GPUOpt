from __future__ import annotations


class TestTelemetryCollection:
    def test_collect_state_mock_cluster(self, client):
        payload = {
            "name": "state-test",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        response = client.post(f"/api/v1/clusters/{cluster_id}/state")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["cluster_id"] == cluster_id
        assert data["cluster_name"] == "state-test"
        assert data["node_count"] >= 1
        assert "collected_at" in data
        assert "generated_at" in data
        assert "nodes" in data

    def test_collect_state_not_found(self, client):
        response = client.post("/api/v1/clusters/00000000-0000-0000-0000-000000000000/state")
        assert response.status_code == 404

    def test_telemetry_includes_gpu_data(self, client):
        payload = {
            "name": "gpu-state-test",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        response = client.post(f"/api/v1/clusters/{cluster_id}/state")
        assert response.status_code == 200
        data = response.json()
        assert data["gpu_count"] >= 0
        if data["gpu_count"] > 0:
            for node in data["nodes"]:
                for gpu in node["gpu_devices"]:
                    assert "index" in gpu
                    assert "model" in gpu
                    assert "memory_total_bytes" in gpu
                    assert "memory_used_bytes" in gpu


class TestClusterStateQuery:
    def test_get_latest_state(self, client):
        payload = {
            "name": "latest-state-test",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")

        response = client.get(f"/api/v1/clusters/{cluster_id}/state")
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_name"] == "latest-state-test"
        assert "telemetry" in data

    def test_get_latest_state_no_data(self, client):
        payload = {
            "name": "no-state-test",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        response = client.get(f"/api/v1/clusters/{cluster_id}/state")
        assert response.status_code == 404

    def test_get_latest_state_not_found(self, client):
        response = client.get("/api/v1/clusters/00000000-0000-0000-0000-000000000000/state")
        assert response.status_code == 404


class TestStateSummary:
    def test_state_summary_empty(self, client):
        response = client.get("/api/v1/state/summary")
        assert response.status_code == 200
        assert response.json() == []

    def test_state_summary_with_data(self, client):
        payload_1 = {
            "name": "summary-test-1",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        payload_2 = {
            "name": "summary-test-2",
            "environment": "development",
            "connector_type": "mock",
            "options": {},
        }
        c1 = client.post("/api/v1/clusters", json=payload_1).json()
        c2 = client.post("/api/v1/clusters", json=payload_2).json()

        client.post(f"/api/v1/clusters/{c1['id']}/state")

        response = client.get("/api/v1/state/summary")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        summaries_by_name = {s["cluster_name"]: s for s in data}
        assert summaries_by_name["summary-test-1"]["status"] in ("fresh", "recent", "stale")
        assert summaries_by_name["summary-test-2"]["status"] == "unchecked"
        assert summaries_by_name["summary-test-1"]["gpu_count"] >= 0


class TestStatePersistence:
    def test_state_survives_multiple_collections(self, client):
        payload = {
            "name": "multi-collect-test",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")

        response = client.get(f"/api/v1/clusters/{cluster_id}/state")
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_name"] == "multi-collect-test"

    def test_state_deleted_with_cluster(self, client):
        payload = {
            "name": "delete-state-test",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.delete(f"/api/v1/clusters/{cluster_id}")

        response = client.get(f"/api/v1/clusters/{cluster_id}/state")
        assert response.status_code == 404
