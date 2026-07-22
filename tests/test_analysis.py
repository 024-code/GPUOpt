from __future__ import annotations


class TestWorkloadAnalysis:
    def test_analyze_no_data(self, client):
        payload = {
            "name": "analysis-no-data",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        response = client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        assert response.status_code == 404

    def test_analyze_with_data(self, client):
        payload = {
            "name": "analysis-with-data",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")

        response = client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_name"] == "analysis-with-data"
        assert data["trace_count"] >= 1
        assert data["gpu_trends"] is not None
        assert data["node_efficiencies"] is not None
        assert data["overall_efficiency_score"] >= 0
        assert "summary" in data

    def test_analyze_not_found(self, client):
        response = client.post(
            "/api/v1/clusters/00000000-0000-0000-0000-000000000000/analyze"
        )
        assert response.status_code == 404


class TestAnalysisResults:
    def test_get_latest_analysis(self, client):
        payload = {
            "name": "analysis-latest",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")

        response = client.get(f"/api/v1/clusters/{cluster_id}/analysis/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_name"] == "analysis-latest"
        assert "gpu_trends" in data
        assert "node_efficiencies" in data

    def test_get_latest_no_analysis(self, client):
        payload = {
            "name": "analysis-no-result",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        response = client.get(f"/api/v1/clusters/{cluster_id}/analysis/latest")
        assert response.status_code == 404

    def test_list_analyses(self, client):
        payload = {
            "name": "analysis-list",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")

        response = client.get(f"/api/v1/clusters/{cluster_id}/analysis/list")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert all("gpu_trends" in a for a in data)


class TestEfficiencyScoring:
    def test_efficiency_score_range(self, client):
        payload = {
            "name": "efficiency-score",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")

        response = client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["overall_efficiency_score"] <= 100
        for ne in data["node_efficiencies"]:
            assert 0 <= ne["efficiency_score"] <= 100
        assert data["total_gpu_hours"] >= 0
        assert data["estimated_power_waste_kwh"] >= 0

    def test_analysis_with_snapshot_gpu_data(self, client):
        payload = {
            "name": "analysis-gpu",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {
                "snapshot_path": "sandbox/mock-clusters/local-kind.json"
            },
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")

        response = client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        assert response.status_code == 200
        data = response.json()
        assert data["gpu_count"] >= 4
        assert len(data["gpu_trends"]) >= 4
        for trend in data["gpu_trends"]:
            assert trend["avg_utilization_percent"] >= 0
            assert trend["memory_total_bytes"] > 0
