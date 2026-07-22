from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


class TestAlertRules:
    def test_create_rule(self, client: TestClient):
        payload = {
            "name": "high-gpu-util",
            "description": "Alert when GPU utilization too high",
            "cluster_id": str(uuid4()),
            "condition_type": "gpu_utilization",
            "operator": "gt",
            "threshold": 90.0,
            "severity": "warning",
            "enabled": True,
            "cooldown_minutes": 30,
        }
        resp = client.post("/api/v1/alerts/rules", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "high-gpu-util"
        assert data["condition_type"] == "gpu_utilization"

    def test_list_rules(self, client: TestClient):
        resp = client.get("/api/v1/alerts/rules")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_rule_not_found(self, client: TestClient):
        resp = client.get(f"/api/v1/alerts/rules/{uuid4()}")
        assert resp.status_code == 404

    def test_update_rule(self, client: TestClient):
        payload = {
            "name": "temp-alert",
            "cluster_id": str(uuid4()),
            "condition_type": "gpu_temperature",
            "operator": "gt",
            "threshold": 85.0,
            "severity": "critical",
        }
        created = client.post("/api/v1/alerts/rules", json=payload).json()
        rid = created["id"]

        resp = client.patch(f"/api/v1/alerts/rules/{rid}", json={"threshold": 90.0})
        assert resp.status_code == 200
        assert resp.json()["threshold"] == 90.0

    def test_delete_rule(self, client: TestClient):
        payload = {
            "name": "del-rule",
            "cluster_id": str(uuid4()),
            "condition_type": "idle_gpu",
            "operator": "gt",
            "threshold": 50.0,
        }
        created = client.post("/api/v1/alerts/rules", json=payload).json()
        rid = created["id"]

        resp = client.delete(f"/api/v1/alerts/rules/{rid}")
        assert resp.status_code == 204

    def test_evaluate_rules(self, client: TestClient):
        cid = client.post("/api/v1/clusters", json={
            "name": "alert-eval-cluster",
            "environment": "test",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()["id"]
        client.post(f"/api/v1/clusters/{cid}/state")

        client.post("/api/v1/alerts/rules", json={
            "name": "eval-rule",
            "cluster_id": cid,
            "condition_type": "gpu_utilization",
            "operator": "lt",
            "threshold": 10.0,
            "severity": "warning",
        })

        resp = client.post(f"/api/v1/alerts/evaluate/{cid}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestAlertRecords:
    def test_list_alerts(self, client: TestClient):
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_acknowledge_alert(self, client: TestClient):
        cid = client.post("/api/v1/clusters", json={
            "name": "ack-alert-cluster",
            "environment": "test",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()["id"]
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post("/api/v1/alerts/rules", json={
            "name": "ack-rule",
            "cluster_id": cid,
            "condition_type": "gpu_utilization",
            "operator": "gt",
            "threshold": 200.0,
            "severity": "critical",
        })
        client.post(f"/api/v1/alerts/evaluate/{cid}")
        alerts = client.get(f"/api/v1/alerts?cluster_id={cid}").json()
        assert len(alerts) > 0

        resp = client.post(f"/api/v1/alerts/{alerts[0]['id']}/acknowledge",
                           params={"user": "admin"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "acknowledged"

    def test_resolve_alert(self, client: TestClient):
        cid = client.post("/api/v1/clusters", json={
            "name": "resolve-alert-cluster",
            "environment": "test",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()["id"]
        client.post(f"/api/v1/clusters/{cid}/state")
        client.post("/api/v1/alerts/rules", json={
            "name": "resolve-rule",
            "cluster_id": cid,
            "condition_type": "gpu_temperature",
            "operator": "gt",
            "threshold": 200.0,
            "severity": "warning",
        })
        client.post(f"/api/v1/alerts/evaluate/{cid}")
        alerts = client.get(f"/api/v1/alerts?cluster_id={cid}").json()
        assert len(alerts) > 0
        resp = client.post(f"/api/v1/alerts/{alerts[0]['id']}/resolve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "resolved"


class TestNotificationChannels:
    def test_create_channel(self, client: TestClient):
        resp = client.post("/api/v1/notifications/channels", json={
            "name": "slack-alerts",
            "channel_type": "slack",
            "config": {"webhook_url": "https://hooks.slack.com/test"},
        })
        assert resp.status_code == 201
        assert resp.json()["channel_type"] == "slack"

    def test_list_channels(self, client: TestClient):
        resp = client.get("/api/v1/notifications/channels")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_channel_not_found(self, client: TestClient):
        resp = client.get(f"/api/v1/notifications/channels/{uuid4()}")
        assert resp.status_code == 404

    def test_update_channel(self, client: TestClient):
        ch = client.post("/api/v1/notifications/channels", json={
            "name": "email-alerts",
            "channel_type": "email",
            "config": {"to": "admin@example.com"},
        }).json()
        resp = client.patch(f"/api/v1/notifications/channels/{ch['id']}",
                            json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_delete_channel(self, client: TestClient):
        ch = client.post("/api/v1/notifications/channels", json={
            "name": "del-channel",
            "channel_type": "webhook",
            "config": {"url": "https://example.com/hook"},
        }).json()
        resp = client.delete(f"/api/v1/notifications/channels/{ch['id']}")
        assert resp.status_code == 204

    def test_test_channel(self, client: TestClient):
        ch = client.post("/api/v1/notifications/channels", json={
            "name": "test-channel",
            "channel_type": "email",
            "config": {"to": "test@example.com"},
        }).json()
        resp = client.post(f"/api/v1/notifications/channels/{ch['id']}/test")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("sent", "failed")

    def test_list_messages(self, client: TestClient):
        resp = client.get("/api/v1/notifications/messages")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestMultiTenancy:
    def test_create_team(self, client: TestClient):
        resp = client.post("/api/v1/tenants/teams", json={"name": "platform-team"})
        assert resp.status_code == 201
        assert resp.json()["name"] == "platform-team"

    def test_list_teams(self, client: TestClient):
        resp = client.get("/api/v1/tenants/teams")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_team(self, client: TestClient):
        team = client.post("/api/v1/tenants/teams", json={"name": "ml-team"}).json()
        resp = client.get(f"/api/v1/tenants/teams/{team['id']}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "ml-team"

    def test_delete_team(self, client: TestClient):
        team = client.post("/api/v1/tenants/teams", json={"name": "delete-team"}).json()
        resp = client.delete(f"/api/v1/tenants/teams/{team['id']}")
        assert resp.status_code == 204

    def test_create_project(self, client: TestClient):
        team = client.post("/api/v1/tenants/teams", json={"name": "proj-team"}).json()
        resp = client.post("/api/v1/tenants/projects", json={
            "name": "inference-service",
            "team_id": team["id"],
        })
        assert resp.status_code == 201
        assert resp.json()["name"] == "inference-service"

    def test_list_projects(self, client: TestClient):
        resp = client.get("/api/v1/tenants/projects")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_quota(self, client: TestClient):
        team = client.post("/api/v1/tenants/teams", json={"name": "quota-team"}).json()
        project = client.post("/api/v1/tenants/projects", json={
            "name": "quota-project",
            "team_id": team["id"],
        }).json()
        resp = client.get(f"/api/v1/tenants/projects/{project['id']}/quota")
        assert resp.status_code == 200
        data = resp.json()
        assert "max_gpus" in data
        assert "current_gpu_count" in data


class TestCostAnomaly:
    def test_anomaly_single(self, client: TestClient):
        cluster = client.post("/api/v1/clusters", json={
            "name": "anomaly-cluster",
            "environment": "prod",
            "connector_type": "mock",
            "options": {},
        }).json()
        resp = client.get(f"/api/v1/anomaly/cost/{cluster['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert "expected_cost" in data
        assert "actual_cost" in data
        assert "is_anomaly" in data

    def test_anomaly_all(self, client: TestClient):
        resp = client.get("/api/v1/anomaly/cost")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestCompliance:
    def test_compliance_report(self, client: TestClient):
        cluster = client.post("/api/v1/clusters", json={
            "name": "compliance-cluster",
            "environment": "prod",
            "connector_type": "mock",
            "options": {},
        }).json()
        resp = client.get(f"/api/v1/compliance/report/{cluster['id']}?framework=soc2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["framework"] == "soc2"
        assert "controls" in data
        assert data["passed"] >= 0


class TestDashboard:
    def test_dashboard_summary(self, client: TestClient):
        cluster = client.post("/api/v1/clusters", json={
            "name": "dash-cluster",
            "environment": "prod",
            "connector_type": "mock",
            "options": {"snapshot_path": "sandbox/mock-clusters/local-kind.json"},
        }).json()
        cid = cluster["id"]
        client.post(f"/api/v1/clusters/{cid}/state")

        resp = client.get(f"/api/v1/dashboard/{cid}")
        assert resp.status_code == 200
        data = resp.json()
        assert "gpu_count" in data
        assert "metrics" in data

    def test_dashboard_all(self, client: TestClient):
        resp = client.get("/api/v1/dashboard")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_report(self, client: TestClient):
        cluster = client.post("/api/v1/clusters", json={
            "name": "report-cluster",
            "environment": "prod",
            "connector_type": "mock",
            "options": {},
        }).json()
        resp = client.post("/api/v1/reports", json={
            "name": "weekly-cost-report",
            "cluster_ids": [cluster["id"]],
            "format": "pdf",
            "schedule": "weekly",
            "recipients": ["admin@example.com"],
        })
        assert resp.status_code == 201
        assert resp.json()["name"] == "weekly-cost-report"

    def test_list_reports(self, client: TestClient):
        resp = client.get("/api/v1/reports")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_report_not_found(self, client: TestClient):
        resp = client.get(f"/api/v1/reports/{uuid4()}")
        assert resp.status_code == 404

    def test_update_report(self, client: TestClient):
        report = client.post("/api/v1/reports", json={
            "name": "update-report",
            "format": "csv",
            "schedule": "daily",
        }).json()
        resp = client.patch(f"/api/v1/reports/{report['id']}",
                            json={"format": "json"})
        assert resp.status_code == 200
        assert resp.json()["format"] == "json"

    def test_delete_report(self, client: TestClient):
        report = client.post("/api/v1/reports", json={
            "name": "delete-report",
            "format": "pdf",
        }).json()
        resp = client.delete(f"/api/v1/reports/{report['id']}")
        assert resp.status_code == 204

    def test_generate_report(self, client: TestClient):
        cluster = client.post("/api/v1/clusters", json={
            "name": "gen-report-cluster",
            "environment": "test",
            "connector_type": "mock",
            "options": {},
        }).json()
        report = client.post("/api/v1/reports", json={
            "name": "generate-me",
            "cluster_ids": [cluster["id"]],
            "format": "json",
        }).json()
        resp = client.post(f"/api/v1/reports/{report['id']}/generate")
        assert resp.status_code == 200
        data = resp.json()
        assert "report_name" in data
        assert data["report_name"] == "generate-me"
