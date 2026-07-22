from __future__ import annotations


class TestActuate:
    def test_actuate_dry_run(self, client):
        payload = {
            "name": "act-dr",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        gen = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        recs = gen.json()["recommendations"]
        assert len(recs) > 0
        rec_id = recs[0]["id"]

        req = {"rec_id": rec_id, "dry_run": True, "reason": "test dry run"}
        resp = client.post(f"/api/v1/clusters/{cluster_id}/actuate", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["dry_run"] is True
        assert "result_summary" in data

    def test_actuate_live(self, client):
        payload = {
            "name": "act-live",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        gen = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        recs = gen.json()["recommendations"]
        assert len(recs) > 0
        rec_id = recs[0]["id"]

        req = {"rec_id": rec_id, "dry_run": False, "reason": "test live actuation"}
        resp = client.post(f"/api/v1/clusters/{cluster_id}/actuate", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("completed", "failed")
        assert data["dry_run"] is False
        assert data["rec_id"] == rec_id
        assert data["cluster_name"] == "act-live"

    def test_actuate_no_recs(self, client):
        payload = {"name": "act-no-rec", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        req = {"rec_id": "00000000-0000-0000-0000-000000000000", "dry_run": False, "reason": ""}
        resp = client.post(f"/api/v1/clusters/{cluster_id}/actuate", json=req)
        assert resp.status_code == 404

    def test_actuate_not_found(self, client):
        req = {"rec_id": "00000000-0000-0000-0000-000000000000", "dry_run": False, "reason": ""}
        resp = client.post("/api/v1/clusters/00000000-0000-0000-0000-000000000000/actuate", json=req)
        assert resp.status_code == 404


class TestActuationList:
    def test_list_actuations_empty(self, client):
        payload = {"name": "act-list-em", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        resp = client.get(f"/api/v1/clusters/{cluster_id}/actuations")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_actuations_with_data(self, client):
        payload = {
            "name": "act-list-d",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        gen = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        rec_id = gen.json()["recommendations"][0]["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/actuate", json={"rec_id": rec_id, "dry_run": True, "reason": ""})
        resp = client.get(f"/api/v1/clusters/{cluster_id}/actuations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    def test_list_not_found(self, client):
        resp = client.get("/api/v1/clusters/00000000-0000-0000-0000-000000000000/actuations")
        assert resp.status_code == 404


class TestActuationDetail:
    def test_get_actuation(self, client):
        payload = {
            "name": "act-det",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        gen = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        rec_id = gen.json()["recommendations"][0]["id"]
        act = client.post(f"/api/v1/clusters/{cluster_id}/actuate", json={"rec_id": rec_id, "dry_run": True, "reason": ""})
        act_id = act.json()["id"]
        resp = client.get(f"/api/v1/clusters/{cluster_id}/actuations/{act_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == act_id

    def test_get_actuation_not_found(self, client):
        payload = {"name": "act-det-nf", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        resp = client.get(f"/api/v1/clusters/{cluster_id}/actuations/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404


class TestActuationRollback:
    def test_rollback_live_actuation(self, client):
        payload = {
            "name": "act-rb",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        gen = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        rec_id = gen.json()["recommendations"][0]["id"]
        act = client.post(f"/api/v1/clusters/{cluster_id}/actuate", json={"rec_id": rec_id, "dry_run": False, "reason": ""})
        act_id = act.json()["id"]

        resp = client.post(f"/api/v1/clusters/{cluster_id}/actuations/{act_id}/rollback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rolled_back"
        assert data["rollback_of"] == act_id

    def test_rollback_not_found(self, client):
        payload = {"name": "act-rb-nf", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        resp = client.post(f"/api/v1/clusters/{cluster_id}/actuations/00000000-0000-0000-0000-000000000000/rollback")
        assert resp.status_code == 404

    def test_rollback_dry_run_no_twin(self, client):
        payload = {
            "name": "act-rb-dr",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        gen = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        rec_id = gen.json()["recommendations"][0]["id"]
        act = client.post(f"/api/v1/clusters/{cluster_id}/actuate", json={"rec_id": rec_id, "dry_run": True, "reason": ""})
        act_id = act.json()["id"]

        resp = client.post(f"/api/v1/clusters/{cluster_id}/actuations/{act_id}/rollback")
        assert resp.status_code == 404


class TestActuationSummary:
    def test_summary_empty(self, client):
        payload = {"name": "act-sum-em", "environment": "sandbox", "connector_type": "mock", "options": {}}
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        resp = client.get(f"/api/v1/clusters/{cluster_id}/actuations/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_actuations"] == 0

    def test_summary_with_data(self, client):
        payload = {
            "name": "act-sum-d",
            "environment": "sandbox",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }
        created = client.post("/api/v1/clusters", json=payload)
        cluster_id = created.json()["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/state")
        client.post(f"/api/v1/clusters/{cluster_id}/analyze")
        gen = client.post(f"/api/v1/clusters/{cluster_id}/recommendations")
        rec_id = gen.json()["recommendations"][0]["id"]
        client.post(f"/api/v1/clusters/{cluster_id}/actuate", json={"rec_id": rec_id, "dry_run": True, "reason": ""})
        client.post(f"/api/v1/clusters/{cluster_id}/actuate", json={"rec_id": rec_id, "dry_run": False, "reason": ""})

        resp = client.get(f"/api/v1/clusters/{cluster_id}/actuations/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_actuations"] >= 2
        assert data["cluster_name"] == "act-sum-d"

    def test_summary_not_found(self, client):
        resp = client.get("/api/v1/clusters/00000000-0000-0000-0000-000000000000/actuations/summary")
        assert resp.status_code == 404
