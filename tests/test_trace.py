from __future__ import annotations


class TestTraceListing:
    def test_list_traces_empty(self, client):
        payload = {
            "name": "trace-list-empty",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        response = client.get(f"/api/v1/clusters/{cluster_id}/traces")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_traces_with_data(self, client):
        payload = {
            "name": "trace-list-data",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")

        response = client.get(f"/api/v1/clusters/{cluster_id}/traces")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(t["cluster_id"] == cluster_id for t in data)
        assert all("collected_at" in t for t in data)

    def test_list_traces_not_found(self, client):
        response = client.get(
            "/api/v1/clusters/00000000-0000-0000-0000-000000000000/traces"
        )
        assert response.status_code == 404

    def test_get_trace_detail(self, client):
        payload = {
            "name": "trace-detail",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        traces = client.get(f"/api/v1/clusters/{cluster_id}/traces").json()
        trace_id = traces[0]["id"]

        response = client.get(f"/api/v1/clusters/{cluster_id}/traces/{trace_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_id"] == cluster_id
        assert "nodes" in data
        assert "telemetry" in data

    def test_get_trace_not_found(self, client):
        payload = {
            "name": "trace-detail-notfound",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        response = client.get(
            f"/api/v1/clusters/{cluster_id}/traces/nonexistent-trace-id"
        )
        assert response.status_code == 404


class TestTraceReplay:
    def test_replay_latest_trace(self, client):
        payload = {
            "name": "replay-latest",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")

        response = client.post(f"/api/v1/clusters/{cluster_id}/replay")
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_name"] == "replay-latest"
        assert "checks" in data
        assert len(data["checks"]) >= 3
        assert "overall_status" in data
        assert data["node_count"] >= 1

    def test_replay_specific_trace(self, client):
        payload = {
            "name": "replay-specific",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        traces = client.get(f"/api/v1/clusters/{cluster_id}/traces").json()
        trace_id = traces[0]["id"]

        response = client.post(
            f"/api/v1/clusters/{cluster_id}/replay?trace_id={trace_id}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["trace_id"] == trace_id
        assert len(data["checks"]) >= 3

    def test_replay_no_state(self, client):
        payload = {
            "name": "replay-no-state",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        response = client.post(f"/api/v1/clusters/{cluster_id}/replay")
        assert response.status_code == 404

    def test_replay_not_found(self, client):
        response = client.post(
            "/api/v1/clusters/00000000-0000-0000-0000-000000000000/replay"
        )
        assert response.status_code == 404


class TestBaseline:
    def test_set_baseline(self, client):
        payload = {
            "name": "baseline-set",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")

        response = client.post(f"/api/v1/clusters/{cluster_id}/baseline")
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_id"] == cluster_id
        assert data["node_count"] >= 1
        assert "trace_id" in data

    def test_set_baseline_no_state(self, client):
        payload = {
            "name": "baseline-no-state",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        response = client.post(f"/api/v1/clusters/{cluster_id}/baseline")
        assert response.status_code == 404

    def test_get_baseline(self, client):
        payload = {
            "name": "baseline-get",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/baseline")

        response = client.get(f"/api/v1/clusters/{cluster_id}/baseline")
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_id"] == cluster_id

    def test_get_baseline_not_set(self, client):
        payload = {
            "name": "baseline-not-set",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        response = client.get(f"/api/v1/clusters/{cluster_id}/baseline")
        assert response.status_code == 404

    def test_baseline_appears_in_trace_list(self, client):
        payload = {
            "name": "baseline-in-traces",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/baseline")

        traces = client.get(f"/api/v1/clusters/{cluster_id}/traces").json()
        assert len(traces) == 1
        assert traces[0]["has_baseline"] is True


class TestComparison:
    def test_compare_with_baseline(self, client):
        payload = {
            "name": "compare-baseline",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/baseline")
        client.post(f"/api/v1/clusters/{cluster_id}/state")

        response = client.post(f"/api/v1/clusters/{cluster_id}/compare")
        assert response.status_code == 200
        data = response.json()
        assert "baseline_id" in data
        assert "current_id" in data
        assert "gpu_diffs" in data
        assert data["node_count_baseline"] >= 1

    def test_compare_without_baseline(self, client):
        payload = {
            "name": "compare-no-baseline",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")

        response = client.post(f"/api/v1/clusters/{cluster_id}/compare")
        assert response.status_code == 404

    def test_compare_two_traces(self, client):
        payload = {
            "name": "compare-two",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        traces = client.get(f"/api/v1/clusters/{cluster_id}/traces").json()
        assert len(traces) >= 2

        response = client.post(
            f"/api/v1/clusters/{cluster_id}/compare",
            params={"trace_id_a": traces[0]["id"], "trace_id_b": traces[1]["id"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert data["gpu_diffs"] is not None

    def test_compare_trace_against_baseline(self, client):
        payload = {
            "name": "compare-against-baseline",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/baseline")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        traces = client.get(f"/api/v1/clusters/{cluster_id}/traces").json()
        latest_id = traces[0]["id"]

        response = client.post(
            f"/api/v1/clusters/{cluster_id}/compare",
            params={"trace_id_a": latest_id},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["baseline_id"] != data["current_id"]
