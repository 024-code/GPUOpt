from __future__ import annotations


def test_liveness_and_readiness(client):
    assert client.get("/health/live").json() == {"status": "alive"}
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_register_and_check_mock_cluster(client):
    payload = {
        "name": "local-kind",
        "environment": "sandbox",
        "connector_type": "mock",
        "description": "Synthetic GPU development cluster",
        "options": {},
    }
    created = client.post("/api/v1/clusters", json=payload)
    assert created.status_code == 201, created.text
    cluster = created.json()

    report_response = client.post(f"/api/v1/clusters/{cluster['id']}/checks")
    assert report_response.status_code == 200, report_response.text
    report = report_response.json()
    assert report["overall_status"] == "pass"
    assert len(report["checks"]) >= 7

    summary = client.get("/api/v1/environments/summary").json()
    assert summary["clusters"] == 1
    assert summary["healthy"] == 1


def test_duplicate_cluster_rejected(client):
    payload = {
        "name": "duplicate",
        "environment": "sandbox",
        "connector_type": "mock",
        "options": {},
    }
    assert client.post("/api/v1/clusters", json=payload).status_code == 201
    assert client.post("/api/v1/clusters", json=payload).status_code == 409
