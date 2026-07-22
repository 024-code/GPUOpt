from __future__ import annotations


class TestTwinSync:
    def test_sync_twin_no_state(self, client):
        payload = {"name": "twin-no-state", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        resp = client.post(f"/api/v1/clusters/{cluster_id}/twin")
        assert resp.status_code == 404

    def test_sync_twin_success(self, client):
        payload = {"name": "twin-sync", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        resp = client.post(f"/api/v1/clusters/{cluster_id}/twin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_name"] == "twin-sync"
        assert data["node_count"] >= 0
        assert data["gpu_count"] >= 0
        assert data["has_diverged"] is False

    def test_sync_twin_not_found(self, client):
        resp = client.post("/api/v1/clusters/00000000-0000-0000-0000-000000000000/twin")
        assert resp.status_code == 404

    def test_get_twin(self, client):
        payload = {"name": "twin-get", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/twin")
        resp = client.get(f"/api/v1/clusters/{cluster_id}/twin")
        assert resp.status_code == 200
        assert resp.json()["cluster_name"] == "twin-get"

    def test_get_twin_no_data(self, client):
        payload = {"name": "twin-get-none", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        resp = client.get(f"/api/v1/clusters/{cluster_id}/twin")
        assert resp.status_code == 404


class TestTwinCompare:
    def test_compare_no_twin(self, client):
        payload = {"name": "twin-cmp-none", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        resp = client.post(f"/api/v1/clusters/{cluster_id}/twin/compare")
        assert resp.status_code == 404

    def test_compare_twin_identical(self, client):
        payload = {"name": "twin-cmp-id", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/twin")
        resp = client.post(f"/api/v1/clusters/{cluster_id}/twin/compare")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_name"] == "twin-cmp-id"
        assert data["drift_count"] >= 0
        assert "overall_drift_severity" in data
        assert "summary" in data

    def test_compare_not_found(self, client):
        resp = client.post("/api/v1/clusters/00000000-0000-0000-0000-000000000000/twin/compare")
        assert resp.status_code == 404


class TestTwinApply:
    def test_apply_no_twin(self, client):
        payload = {"name": "twin-app-none", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        resp = client.post(
            f"/api/v1/clusters/{cluster_id}/twin/apply/00000000-0000-0000-0000-000000000000"
        )
        assert resp.status_code == 404

    def test_apply_recommendation(self, client):
        payload = {
            "name": "twin-app-rec",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")

        client.post(f"/api/v1/clusters/{cluster_id}/twin")
        gen = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        data = gen.json()

        if data["recommendations"]:
            rec_id = data["recommendations"][0]["id"]
            resp = client.post(f"/api/v1/clusters/{cluster_id}/twin/apply/{rec_id}")
            assert resp.status_code == 200
            assert resp.json()["has_diverged"] is True
            assert "divergence_reason" in resp.json()

    def test_apply_bad_rec(self, client):
        payload = {"name": "twin-app-bad", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/twin")
        resp = client.post(
            f"/api/v1/clusters/{cluster_id}/twin/apply/00000000-0000-0000-0000-000000000000"
        )
        assert resp.status_code == 404


class TestTwinReset:
    def test_reset_twin(self, client):
        payload = {"name": "twin-reset", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/twin")
        resp = client.post(f"/api/v1/clusters/{cluster_id}/twin/reset")
        assert resp.status_code == 200
        assert resp.json()["has_diverged"] is False

    def test_reset_no_twin(self, client):
        payload = {"name": "twin-reset-none", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        resp = client.post(f"/api/v1/clusters/{cluster_id}/twin/reset")
        assert resp.status_code == 404
