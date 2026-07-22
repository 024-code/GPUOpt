from __future__ import annotations


class TestRecommendationGeneration:
    def test_generate_no_data(self, client):
        payload = {
            "name": "rec-no-data",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        response = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        assert response.status_code == 404

    def test_generate_from_state_only(self, client):
        payload = {
            "name": "rec-state-only",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")

        response = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_name"] == "rec-state-only"
        assert data["recommendation_count"] >= 0
        assert isinstance(data["recommendations"], list)
        assert data["critical_count"] >= 0
        assert data["high_count"] >= 0
        assert "summary" in data

    def test_generate_from_state_and_analysis(self, client):
        payload = {
            "name": "rec-full",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")

        response = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        assert response.status_code == 200
        data = response.json()
        assert data["recommendation_count"] >= 0
        assert data["based_on_state_at"] is not None
        assert data["based_on_analysis_at"] is not None

    def test_generate_not_found(self, client):
        response = client.post(
            "/api/v1/clusters/00000000-0000-0000-0000-000000000000/recommendations"
        )
        assert response.status_code == 404


class TestRecommendationStorage:
    def test_get_latest(self, client):
        payload = {
            "name": "rec-latest",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/recommendations")

        response = client.get(f"/api/v1/clusters/{cluster_id}/recommendations/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["cluster_name"] == "rec-latest"

    def test_get_latest_no_data(self, client):
        payload = {
            "name": "rec-no-latest",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        response = client.get(f"/api/v1/clusters/{cluster_id}/recommendations/latest")
        assert response.status_code == 404

    def test_list_recommendations(self, client):
        payload = {
            "name": "rec-list",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        client.post(f"/api/v1/clusters/{cluster_id}/recommendations")

        response = client.get(f"/api/v1/clusters/{cluster_id}/recommendations/list")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1


class TestRecommendationTypes:
    def test_recommendations_with_gpu_snapshot(self, client):
        payload = {
            "name": "rec-gpu",
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
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")

        response = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        assert response.status_code == 200
        data = response.json()
        assert data["recommendation_count"] >= 1

        types_found = set(r["type"] for r in data["recommendations"])
        severities_found = set(r["severity"] for r in data["recommendations"])

        for r in data["recommendations"]:
            assert "title" in r
            assert "reasoning" in r
            assert "description" in r
            assert "expected_impact" in r
            assert 0 <= r["confidence"] <= 1
            assert "actions" in r
            assert 0 <= r["score"] <= 100
            assert r["status"] == "pending"

        assert 0 <= data["avg_score"] <= 100
        assert data["total_estimated_savings_gpu_hours"] >= 0
        assert isinstance(data["top_recommendation"], str)


class TestRecommendationScoring:
    def test_scores_are_assigned(self, client):
        payload = {
            "name": "rec-score-test",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        response = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        assert response.status_code == 200
        data = response.json()

        for r in data["recommendations"]:
            assert 0 <= r["score"] <= 100
            assert r["score"] > 0 or r["confidence"] == 0

    def test_higher_severity_gets_higher_score(self, client):
        payload = {
            "name": "rec-severity-score",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {
                "snapshot_path": "sandbox/mock-clusters/local-kind.json"
            },
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        response = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        data = response.json()

        severity_scores = {"critical": [], "high": [], "medium": [], "low": [], "info": []}
        for r in data["recommendations"]:
            sev = r["severity"]
            if sev in severity_scores:
                severity_scores[sev].append(r["score"])

        for sev in ("critical", "high", "medium", "low", "info"):
            if severity_scores[sev]:
                avg = sum(severity_scores[sev]) / len(severity_scores[sev])
                assert avg > 0, f"{sev} recs should have non-zero avg score"

    def test_recommendations_sorted_by_score(self, client):
        payload = {
            "name": "rec-sort-test",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {
                "snapshot_path": "sandbox/mock-clusters/local-kind.json"
            },
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        response = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        data = response.json()

        scores = [r["score"] for r in data["recommendations"]]
        assert scores == sorted(scores, reverse=True)


class TestRecommendationLifecycle:
    def test_update_status_to_approved(self, client):
        payload = {
            "name": "rec-lifecycle",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        gen = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        data = gen.json()

        if data["recommendations"]:
            rec_id = data["recommendations"][0]["id"]
            resp = client.post(
                f"/api/v1/clusters/{cluster_id}/recommendations/{rec_id}/status",
                json={"status": "approved", "reason": "Looks good"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "approved"
            assert resp.json()["id"] == rec_id

    def test_update_status_to_dismissed(self, client):
        payload = {
            "name": "rec-lifecycle-2",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        gen = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        data = gen.json()

        if data["recommendations"]:
            rec_id = data["recommendations"][0]["id"]
            resp = client.post(
                f"/api/v1/clusters/{cluster_id}/recommendations/{rec_id}/status",
                json={"status": "dismissed", "reason": "Not applicable"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "dismissed"

    def test_update_status_not_found(self, client):
        resp = client.post(
            "/api/v1/clusters/00000000-0000-0000-0000-000000000000/"
            "recommendations/00000000-0000-0000-0000-000000000000/status",
            json={"status": "approved", "reason": "test"},
        )
        assert resp.status_code == 404

    def test_update_status_invalid_value(self, client):
        payload = {
            "name": "rec-invalid-status",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        gen = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        data = gen.json()

        if data["recommendations"]:
            rec_id = data["recommendations"][0]["id"]
            resp = client.post(
                f"/api/v1/clusters/{cluster_id}/recommendations/{rec_id}/status",
                json={"status": "invalid_status", "reason": "test"},
            )
            assert resp.status_code == 422


class TestWhatIfSimulation:
    def test_what_if_no_recommendations(self, client):
        payload = {
            "name": "what-if-no-data",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        resp = client.post(f"/api/v1/clusters/{cluster_id}/recommendations/what-if")
        assert resp.status_code == 404

    def test_what_if_with_recommendations(self, client):
        payload = {
            "name": "what-if-rec",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]

        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/recommendations")

        resp = client.post(f"/api/v1/clusters/{cluster_id}/recommendations/what-if")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_name"] == "what-if-rec"
        assert 0 <= data["projected_gpu_utilization_percent"] <= 100
        assert 0 <= data["projected_efficiency_score"] <= 100
        assert data["projected_idle_gpu_hours_reduction"] >= 0
        assert data["projected_power_savings_kwh"] >= 0
        assert data["estimated_cost_savings_usd"] >= 0
        assert 0 <= data["fragmentation_improvement_percent"] <= 100
        assert data["reservations_freed"] >= 0
        assert 0 <= data["risk_reduction_score"] <= 100
        assert "summary" in data

    def test_what_if_not_found(self, client):
        resp = client.post(
            "/api/v1/clusters/00000000-0000-0000-0000-000000000000/recommendations/what-if"
        )
        assert resp.status_code == 404
