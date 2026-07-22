from __future__ import annotations


class TestDetailedHealth:
    def test_detailed_health(self, client):
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "system" in data
        assert "configuration" in data
        assert "clusters" in data
        assert data["clusters"]["total"] == 0

    def test_detailed_health_with_clusters(self, client):
        # Create a cluster
        client.post(
            "/api/v1/clusters",
            json={
                "name": "health-test",
                "environment": "sandbox",
                "connector_type": "mock",
                "options": {},
            },
        )
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert data["clusters"]["total"] == 1
        assert "sandbox" in data["clusters"]["by_environment"]
        assert "mock" in data["clusters"]["by_connector_type"]


class TestSystemInfo:
    def test_system_info(self, client):
        response = client.get("/api/v1/info")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "GPUOpt Backend Sandbox"
        assert "version" in data
        assert "environment" in data
        assert data["documentation"] == "/docs"
        assert data["health"] == "/health/detailed"
        assert data["metrics"] == "/metrics"


class TestCorrelationID:
    def test_correlation_id_in_response(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        assert "X-Response-Time" in response.headers

    def test_custom_correlation_id(self, client):
        custom_id = "test-correlation-id-123"
        response = client.get("/health/live", headers={"X-Correlation-ID": custom_id})
        assert response.status_code == 200
        assert response.headers["X-Correlation-ID"] == custom_id


class TestRateLimiting:
    def test_rate_limit_headers(self, client):
        # Health endpoints skip rate limiting, so test with a regular endpoint
        response = client.get("/api/v1/clusters")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers


class TestValidationErrors:
    def test_invalid_uuid_format(self, client):
        response = client.get("/api/v1/clusters/invalid-uuid")
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        response = client.post("/api/v1/clusters", json={})
        assert response.status_code == 422

    def test_invalid_connector_type(self, client):
        response = client.post(
            "/api/v1/clusters",
            json={
                "name": "test",
                "environment": "sandbox",
                "connector_type": "invalid_type",
                "options": {},
            },
        )
        assert response.status_code == 422


class TestErrorHandling:
    def test_404_error_format(self, client):
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/v1/clusters/{fake_id}")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_409_error_format(self, client):
        payload = {
            "name": "duplicate",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        client.post("/api/v1/clusters", json=payload)
        response = client.post("/api/v1/clusters", json=payload)
        assert response.status_code == 409
