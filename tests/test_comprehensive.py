from __future__ import annotations

import uuid


class TestClusterCRUD:
    def test_create_cluster(self, client):
        payload = {
            "name": "test-cluster",
            "environment": "sandbox",
            "connector_type": "mock",
            "description": "Test cluster for validation",
            "options": {},
        }
        response = client.post("/api/v1/clusters", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test-cluster"
        assert data["environment"] == "sandbox"
        assert data["connector_type"] == "mock"
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_cluster_invalid_name(self, client):
        payload = {
            "name": "x",  # Too short (min_length=2)
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        response = client.post("/api/v1/clusters", json=payload)
        assert response.status_code == 422

    def test_create_cluster_invalid_connector_type(self, client):
        payload = {
            "name": "test-cluster",
            "environment": "sandbox",
            "connector_type": "invalid",
            "options": {},
        }
        response = client.post("/api/v1/clusters", json=payload)
        assert response.status_code == 422

    def test_list_clusters_empty(self, client):
        response = client.get("/api/v1/clusters")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_clusters(self, client):
        client.post(
            "/api/v1/clusters",
            json={
                "name": "cluster-1",
                "environment": "sandbox",
                "connector_type": "mock",
                "options": {},
            },
        )
        client.post(
            "/api/v1/clusters",
            json={
                "name": "cluster-2",
                "environment": "development",
                "connector_type": "mock",
                "options": {},
            },
        )
        response = client.get("/api/v1/clusters")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        # Verify ordering (by environment, name)
        assert data[0]["name"] == "cluster-2"
        assert data[1]["name"] == "cluster-1"

    def test_get_cluster(self, client):
        create_response = client.post(
            "/api/v1/clusters",
            json={
                "name": "get-test",
                "environment": "sandbox",
                "connector_type": "mock",
                "options": {},
            },
        )
        cluster_id = create_response.json()["id"]
        response = client.get(f"/api/v1/clusters/{cluster_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "get-test"

    def test_get_cluster_not_found(self, client):
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/clusters/{fake_id}")
        assert response.status_code == 404

    def test_get_cluster_invalid_id(self, client):
        response = client.get("/api/v1/clusters/invalid-id")
        assert response.status_code == 422

    def test_delete_cluster(self, client):
        create_response = client.post(
            "/api/v1/clusters",
            json={
                "name": "delete-test",
                "environment": "sandbox",
                "connector_type": "mock",
                "options": {},
            },
        )
        cluster_id = create_response.json()["id"]
        response = client.delete(f"/api/v1/clusters/{cluster_id}")
        assert response.status_code == 204
        # Verify deletion
        response = client.get(f"/api/v1/clusters/{cluster_id}")
        assert response.status_code == 404

    def test_delete_cluster_not_found(self, client):
        fake_id = str(uuid.uuid4())
        response = client.delete(f"/api/v1/clusters/{fake_id}")
        assert response.status_code == 404


class TestHealthEndpoints:
    def test_liveness(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    def test_readiness(self, client):
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "registered_clusters" in data


class TestMetricsEndpoint:
    def test_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]


class TestCheckEndpoints:
    def test_run_check_on_cluster(self, client):
        create_response = client.post(
            "/api/v1/clusters",
            json={
                "name": "check-test",
                "environment": "sandbox",
                "connector_type": "mock",
                "options": {},
            },
        )
        cluster_id = create_response.json()["id"]
        response = client.post(f"/api/v1/clusters/{cluster_id}/checks")
        assert response.status_code == 200
        data = response.json()
        assert data["overall_status"] == "pass"
        assert len(data["checks"]) >= 7
        assert "cluster_name" in data
        assert "started_at" in data
        assert "completed_at" in data

    def test_run_check_cluster_not_found(self, client):
        fake_id = str(uuid.uuid4())
        response = client.post(f"/api/v1/clusters/{fake_id}/checks")
        assert response.status_code == 404

    def test_get_latest_check(self, client):
        create_response = client.post(
            "/api/v1/clusters",
            json={
                "name": "latest-check-test",
                "environment": "sandbox",
                "connector_type": "mock",
                "options": {},
            },
        )
        cluster_id = create_response.json()["id"]
        # Run a check first
        client.post(f"/api/v1/clusters/{cluster_id}/checks")
        # Get latest check
        response = client.get(f"/api/v1/clusters/{cluster_id}/checks/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["overall_status"] == "pass"

    def test_get_latest_check_no_checks(self, client):
        create_response = client.post(
            "/api/v1/clusters",
            json={
                "name": "no-checks-test",
                "environment": "sandbox",
                "connector_type": "mock",
                "options": {},
            },
        )
        cluster_id = create_response.json()["id"]
        response = client.get(f"/api/v1/clusters/{cluster_id}/checks/latest")
        assert response.status_code == 404

    def test_get_latest_check_cluster_not_found(self, client):
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/clusters/{fake_id}/checks/latest")
        assert response.status_code == 404


class TestEnvironmentEndpoints:
    def test_check_all_environments(self, client):
        # Create multiple clusters
        client.post(
            "/api/v1/clusters",
            json={
                "name": "env-test-1",
                "environment": "sandbox",
                "connector_type": "mock",
                "options": {},
            },
        )
        client.post(
            "/api/v1/clusters",
            json={
                "name": "env-test-2",
                "environment": "development",
                "connector_type": "mock",
                "options": {},
            },
        )
        response = client.post("/api/v1/environments/check-all")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        for report in data:
            assert report["overall_status"] == "pass"

    def test_environment_summary(self, client):
        # Create clusters in different environments
        client.post(
            "/api/v1/clusters",
            json={
                "name": "summary-sandbox",
                "environment": "sandbox",
                "connector_type": "mock",
                "options": {},
            },
        )
        client.post(
            "/api/v1/clusters",
            json={
                "name": "summary-dev",
                "environment": "development",
                "connector_type": "mock",
                "options": {},
            },
        )
        response = client.get("/api/v1/environments/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["clusters"] == 2
        assert "environments" in data
        assert "sandbox" in data["environments"]
        assert "development" in data["environments"]


class TestUpsertCluster:
    def test_upsert_cluster_create(self, client):
        payload = {
            "name": "upsert-new",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        response = client.put("/api/v1/clusters/by-name/upsert-new", json=payload)
        assert response.status_code == 200
        assert response.json()["name"] == "upsert-new"

    def test_upsert_cluster_update(self, client):
        # Create initial cluster
        client.post(
            "/api/v1/clusters",
            json={
                "name": "upsert-existing",
                "environment": "sandbox",
                "connector_type": "mock",
                "options": {},
            },
        )
        # Upsert with updated data
        payload = {
            "name": "upsert-existing",
            "environment": "development",
            "connector_type": "mock",
            "description": "Updated via upsert",
            "options": {},
        }
        response = client.put("/api/v1/clusters/by-name/upsert-existing", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["environment"] == "development"
        assert data["description"] == "Updated via upsert"

    def test_upsert_cluster_name_mismatch(self, client):
        payload = {
            "name": "different-name",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        response = client.put("/api/v1/clusters/by-name/wrong-name", json=payload)
        assert response.status_code == 400
