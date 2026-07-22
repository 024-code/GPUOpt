from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


class TestPolicyEngine:
    def test_create_policy(self, client: TestClient):
        payload = {
            "name": "no-prod-weekends",
            "description": "Block actuations on production during weekends",
            "scope_type": "environment",
            "scope_value": "production",
            "rule_type": "time_window",
            "rule_config": {"start_hour": 8, "end_hour": 18, "allowed_days": [0, 1, 2, 3, 4]},
            "severity": "high",
            "enabled": True,
            "fail_action": "block",
        }
        resp = client.post("/api/v1/guarded/policies", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "no-prod-weekends"
        assert data["rule_type"] == "time_window"
        assert data["enabled"] is True

    def test_list_policies(self, client: TestClient):
        resp = client.get("/api/v1/guarded/policies")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_policy_not_found(self, client: TestClient):
        resp = client.get(f"/api/v1/guarded/policies/{uuid4()}")
        assert resp.status_code == 404

    def test_update_policy(self, client: TestClient):
        payload = {
            "name": "test-update-policy",
            "description": "Test policy",
            "rule_type": "environment_restriction",
            "rule_config": {"denied_environments": ["production"]},
            "severity": "medium",
            "enabled": True,
            "fail_action": "block",
        }
        created = client.post("/api/v1/guarded/policies", json=payload)
        policy_id = created.json()["id"]

        resp = client.patch(f"/api/v1/guarded/policies/{policy_id}", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_delete_policy(self, client: TestClient):
        payload = {
            "name": "test-delete-policy",
            "description": "Policy to delete",
            "rule_type": "environment_restriction",
            "rule_config": {"denied_environments": []},
            "severity": "low",
            "enabled": True,
            "fail_action": "warn",
        }
        created = client.post("/api/v1/guarded/policies", json=payload)
        policy_id = created.json()["id"]

        resp = client.delete(f"/api/v1/guarded/policies/{policy_id}")
        assert resp.status_code == 204

    def test_pre_flight_environment_restriction(self, client: TestClient):
        payload = {
            "name": "prod-restriction",
            "description": "Block production actuations",
            "scope_type": "environment",
            "scope_value": "production",
            "rule_type": "environment_restriction",
            "rule_config": {"denied_environments": ["production"]},
            "severity": "critical",
            "enabled": True,
            "fail_action": "block",
        }
        client.post("/api/v1/guarded/policies", json=payload)

        cluster = client.post("/api/v1/clusters", json={
            "name": "ga-prod-cluster",
            "environment": "production",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()
        cid = cluster["id"]
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post(f"/api/v1/clusters/{cid}/analyze")
        recs = client.post(f"/api/v1/clusters/{cid}/recommendations").json()["recommendations"]
        rec_id = recs[0]["id"]

        resp = client.post(f"/api/v1/guarded/pre-flight/{cid}/{rec_id}",
                           params={"environment": "production"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_passed"] is False
        assert data["blocked_count"] >= 1

    def test_pre_flight_allows_safe_env(self, client: TestClient):
        cluster = client.post("/api/v1/clusters", json={
            "name": "ga-dev-cluster",
            "environment": "development",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()
        cid = cluster["id"]
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post(f"/api/v1/clusters/{cid}/analyze")
        recs = client.post(f"/api/v1/clusters/{cid}/recommendations").json()["recommendations"]
        rec_id = recs[0]["id"]

        resp = client.post(f"/api/v1/guarded/pre-flight/{cid}/{rec_id}",
                           params={"environment": "development"})
        assert resp.status_code == 200
        assert resp.json()["overall_passed"] is True


class TestApprovalWorkflowAPI:
    def test_create_approval(self, client: TestClient):
        cluster = client.post("/api/v1/clusters", json={
            "name": "ga-approval-cluster",
            "environment": "staging",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()
        cid = cluster["id"]
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post(f"/api/v1/clusters/{cid}/analyze")
        recs = client.post(f"/api/v1/clusters/{cid}/recommendations").json()["recommendations"]
        rec_id = recs[0]["id"]

        act = client.post(f"/api/v1/clusters/{cid}/actuate", json={
            "rec_id": rec_id, "dry_run": True, "reason": "approval test",
        }).json()

        resp = client.post("/api/v1/guarded/approvals", json={
            "actuation_id": act["id"],
            "cluster_id": cid,
            "required_approvers": ["alice", "bob"],
            "reason": "Production deployment approval",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert len(data["steps"]) == 2

    def test_approve(self, client: TestClient):
        cluster = client.post("/api/v1/clusters", json={
            "name": "ga-approve-test",
            "environment": "staging",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()
        cid = cluster["id"]
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post(f"/api/v1/clusters/{cid}/analyze")
        recs = client.post(f"/api/v1/clusters/{cid}/recommendations").json()["recommendations"]
        act = client.post(f"/api/v1/clusters/{cid}/actuate", json={
            "rec_id": recs[0]["id"], "dry_run": True,
        }).json()

        approval = client.post("/api/v1/guarded/approvals", json={
            "actuation_id": act["id"],
            "cluster_id": cid,
            "required_approvers": ["admin"],
            "reason": "Test",
        }).json()
        aid = approval["id"]

        resp = client.post(f"/api/v1/guarded/approvals/{aid}/approve",
                           params={"approver": "admin", "reason": "Looks good"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

    def test_reject(self, client: TestClient):
        cluster = client.post("/api/v1/clusters", json={
            "name": "ga-reject-test",
            "environment": "staging",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()
        cid = cluster["id"]
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post(f"/api/v1/clusters/{cid}/analyze")
        recs = client.post(f"/api/v1/clusters/{cid}/recommendations").json()["recommendations"]
        act = client.post(f"/api/v1/clusters/{cid}/actuate", json={
            "rec_id": recs[0]["id"], "dry_run": True,
        }).json()

        approval = client.post("/api/v1/guarded/approvals", json={
            "actuation_id": act["id"],
            "cluster_id": cid,
            "required_approvers": ["reviewer"],
            "reason": "Test",
        }).json()
        aid = approval["id"]

        resp = client.post(f"/api/v1/guarded/approvals/{aid}/reject",
                           params={"approver": "reviewer", "reason": "Not ready"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    def test_approval_not_found(self, client: TestClient):
        resp = client.post(f"/api/v1/guarded/approvals/{uuid4()}/approve",
                           params={"approver": "admin"})
        assert resp.status_code == 404

    def test_list_approvals(self, client: TestClient):
        resp = client.get("/api/v1/guarded/approvals")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestChaosEngineAPI:
    def test_create_experiment(self, client: TestClient):
        payload = {
            "cluster_id": str(uuid4()),
            "name": "node-failure-test",
            "description": "Test node failure handling",
            "fault_type": "node_failure",
            "target": {"target_type": "node", "target_selector": {}, "count": 2},
            "duration_seconds": 120,
            "intensity": 0.7,
        }
        resp = client.post("/api/v1/guarded/chaos-experiments", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "node-failure-test"
        assert data["fault_type"] == "node_failure"
        assert data["status"] == "pending"

    def test_list_experiments(self, client: TestClient):
        resp = client.get("/api/v1/guarded/chaos-experiments")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_experiment_not_found(self, client: TestClient):
        resp = client.get(f"/api/v1/guarded/chaos-experiments/{uuid4()}")
        assert resp.status_code == 404

    def test_run_experiment(self, client: TestClient):
        experiment = client.post("/api/v1/guarded/chaos-experiments", json={
            "cluster_id": str(uuid4()),
            "name": "gpu-failure-test",
            "fault_type": "gpu_failure",
            "target": {"target_type": "gpu", "target_selector": {}, "count": 1},
            "duration_seconds": 60,
            "intensity": 0.5,
        }).json()
        eid = experiment["id"]

        resp = client.post(f"/api/v1/guarded/chaos-experiments/{eid}/run")
        assert resp.status_code == 200
        data = resp.json()
        assert "experiment" in data
        assert "system_resilient" in data
        assert "summary" in data

    def test_delete_experiment(self, client: TestClient):
        experiment = client.post("/api/v1/guarded/chaos-experiments", json={
            "cluster_id": str(uuid4()),
            "name": "delete-test",
            "fault_type": "pod_kill",
            "target": {"target_type": "pod", "target_selector": {}, "count": 1},
            "duration_seconds": 30,
            "intensity": 0.3,
        }).json()
        eid = experiment["id"]

        resp = client.delete(f"/api/v1/guarded/chaos-experiments/{eid}")
        assert resp.status_code == 204


class TestGuardedAutomationRecommendations:
    def test_ga_recommendations(self, client: TestClient):
        cluster = client.post("/api/v1/clusters", json={
            "name": "ga-recs-test",
            "environment": "development",
            "connector_type": "mock",
            "options": {},
        }).json()
        cid = cluster["id"]

        resp = client.get(f"/api/v1/guarded/recommendations/{cid}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 3
        types = {r["recommendation_type"] for r in data}
        assert "policy" in types
        assert "approval" in types
        assert "chaos" in types
