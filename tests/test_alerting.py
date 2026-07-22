from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from gpuopt.notifications import (
    EmailBackend,
    NotificationBackend,
    NotificationResult,
    NotificationService,
    OpsGenieBackend,
    PagerDutyBackend,
    SlackBackend,
    WebhookBackend,
)
from gpuopt.schemas import (
    AlertConditionType,
    AlertRecord,
    AlertRule,
    AlertSeverity,
    NotificationChannel,
    NotificationChannelType,
    NotificationMessage,
)


# ── Notification Backend Tests ──────────────────────────────────


class TestNotificationBackends:
    def test_slack_backend_no_webhook(self):
        backend = SlackBackend()
        channel = NotificationChannel(name="test", channel_type=NotificationChannelType.SLACK, config={})
        result = backend.send(channel, "test", "body")
        assert not result.success
        assert "No slack webhook_url configured" in result.error

    def test_pagerduty_backend_no_key(self):
        backend = PagerDutyBackend()
        channel = NotificationChannel(name="test", channel_type=NotificationChannelType.PAGERDUTY, config={})
        result = backend.send(channel, "test", "body")
        assert not result.success
        assert "No PagerDuty routing_key configured" in result.error

    def test_opsgenie_backend_no_key(self):
        backend = OpsGenieBackend()
        channel = NotificationChannel(name="test", channel_type=NotificationChannelType.OPSGENIE, config={})
        result = backend.send(channel, "test", "body")
        assert not result.success
        assert "No OpsGenie api_key configured" in result.error

    def test_webhook_backend_no_url(self):
        backend = WebhookBackend()
        channel = NotificationChannel(name="test", channel_type=NotificationChannelType.WEBHOOK, config={})
        result = backend.send(channel, "test", "body")
        assert not result.success
        assert "No webhook url configured" in result.error

    def test_email_backend_no_config(self):
        backend = EmailBackend()
        channel = NotificationChannel(name="test", channel_type=NotificationChannelType.EMAIL, config={})
        result = backend.send(channel, "test", "body")
        assert not result.success
        assert "smtp_host or to_addrs" in result.error


class TestNotificationService:
    def test_unknown_channel_type(self):
        svc = NotificationService()
        del svc._backends[NotificationChannelType.EMAIL]
        result = svc.send(
            NotificationChannel(id=uuid4(), name="x", channel_type=NotificationChannelType.EMAIL, config={}),
            "test", "body",
        )
        assert not result.success
        assert "No backend for channel type" in result.error

    def test_custom_backend_registration(self):
        svc = NotificationService()
        class FakeBackend(NotificationBackend):
            def send(self, channel, subject, body):
                return NotificationResult(True, "fake sent")
        svc.register_backend(NotificationChannelType.WEBHOOK, FakeBackend())
        channel = NotificationChannel(name="f", channel_type=NotificationChannelType.WEBHOOK, config={"url": "http://x"})
        result = svc.send(channel, "t", "b")
        assert result.success
        assert result.message == "fake sent"

    def test_send_test(self):
        svc = NotificationService()
        channel = NotificationChannel(name="t", channel_type=NotificationChannelType.SLACK, config={})
        result = svc.send_test(channel)
        assert not result.success


# ── Alert Rule Tests ────────────────────────────────────────────


class TestAlertRule:
    def test_alert_rule_defaults(self):
        rule = AlertRule(name="test", cluster_id=uuid4())
        assert rule.enabled
        assert rule.severity == AlertSeverity.WARNING
        assert rule.condition_type == AlertConditionType.GPU_UTILIZATION
        assert rule.cooldown_minutes == 60
        assert rule.notification_channel_ids == []

    def test_alert_rule_with_channels(self):
        ch_id = str(uuid4())
        rule = AlertRule(
            name="critical-gpu",
            cluster_id=uuid4(),
            condition_type=AlertConditionType.GPU_TEMPERATURE,
            severity=AlertSeverity.CRITICAL,
            threshold=85.0,
            operator="gt",
            notification_channel_ids=[ch_id],
        )
        assert len(rule.notification_channel_ids) == 1
        assert rule.notification_channel_ids[0] == ch_id
        assert rule.threshold == 85.0


class TestAlertRecord:
    def test_alert_record_defaults(self):
        record = AlertRecord(rule_id=uuid4(), cluster_id=uuid4())
        assert record.status == "firing"
        assert record.severity == AlertSeverity.WARNING

    def test_alert_record_lifecycle(self):
        record = AlertRecord(rule_id=uuid4(), cluster_id=uuid4())
        assert record.status == "firing"
        record.status = "acknowledged"
        assert record.status == "acknowledged"
        record.status = "resolved"
        assert record.status == "resolved"


class TestNotificationChannel:
    def test_channel_defaults(self):
        ch = NotificationChannel(name="slack-alerts", channel_type=NotificationChannelType.SLACK)
        assert ch.enabled
        assert ch.config == {}
        assert ch.channel_type == NotificationChannelType.SLACK

    def test_channel_with_config(self):
        ch = NotificationChannel(
            name="pagerduty-prod",
            channel_type=NotificationChannelType.PAGERDUTY,
            config={"routing_key": "abc123", "severity": "critical"},
        )
        assert ch.config["routing_key"] == "abc123"
        assert ch.config["severity"] == "critical"


class TestNotificationMessage:
    def test_message_defaults(self):
        msg = NotificationMessage(channel_id=uuid4(), subject="test", body="body")
        assert msg.status == "pending"
        assert msg.sent_at is None

    def test_message_lifecycle(self):
        from datetime import datetime, timezone
        msg = NotificationMessage(channel_id=uuid4(), subject="s", body="b")
        assert msg.status == "pending"
        msg.status = "sent"
        msg.sent_at = datetime.now(timezone.utc)
        assert msg.status == "sent"
        assert msg.sent_at is not None


# ── API Integration Tests ───────────────────────────────────────


@pytest.fixture
def client(tmp_path):
    import os
    os.environ["GPUOPT_DATABASE_PATH"] = str(tmp_path / "test_alerting.db")
    from gpuopt.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
    from gpuopt.config import get_settings
    get_settings.cache_clear()


class TestAlertAPI:
    def test_create_alert_rule(self, client):
        cluster_id = str(uuid4())
        resp = client.post("/api/v1/alerts/rules", json={
            "name": "High GPU Temp",
            "description": "Alert when GPU temp exceeds 85C",
            "cluster_id": cluster_id,
            "condition_type": "gpu_temperature",
            "operator": "gt",
            "threshold": 85.0,
            "severity": "critical",
        })
        assert resp.status_code in (200, 201)
        data = resp.json()
        assert data["name"] == "High GPU Temp"
        assert data["severity"] == "critical"
        assert data["threshold"] == 85.0
        assert "id" in data

    def test_list_alert_rules(self, client):
        cluster_id = str(uuid4())
        client.post("/api/v1/alerts/rules", json={
            "name": "Rule 1", "cluster_id": cluster_id,
        })
        client.post("/api/v1/alerts/rules", json={
            "name": "Rule 2", "cluster_id": cluster_id,
        })
        resp = client.get(f"/api/v1/alerts/rules?cluster_id={cluster_id}")
        assert resp.status_code == 200
        data = resp.json()
        filtered = [r for r in data if r["cluster_id"] == cluster_id]
        assert len(filtered) == 2

    def test_get_alert_rule(self, client):
        cluster_id = str(uuid4())
        create = client.post("/api/v1/alerts/rules", json={
            "name": "My Rule", "cluster_id": cluster_id,
        }).json()
        resp = client.get(f"/api/v1/alerts/rules/{create['id']}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "My Rule"

    def test_get_alert_rule_not_found(self, client):
        resp = client.get(f"/api/v1/alerts/rules/{uuid4()}")
        assert resp.status_code == 404

    def test_update_alert_rule(self, client):
        cluster_id = str(uuid4())
        create = client.post("/api/v1/alerts/rules", json={
            "name": "Old Name", "cluster_id": cluster_id,
        }).json()
        resp = client.patch(f"/api/v1/alerts/rules/{create['id']}", json={"name": "New Name", "threshold": 90.0})
        assert resp.status_code == 200
        assert resp.json()["name"] == "New Name"
        assert resp.json()["threshold"] == 90.0

    def test_delete_alert_rule(self, client):
        cluster_id = str(uuid4())
        create = client.post("/api/v1/alerts/rules", json={
            "name": "Delete Me", "cluster_id": cluster_id,
        }).json()
        resp = client.delete(f"/api/v1/alerts/rules/{create['id']}")
        assert resp.status_code in (200, 204)
        if resp.status_code == 200:
            assert resp.json()["status"] == "deleted"
        resp = client.get(f"/api/v1/alerts/rules/{create['id']}")
        assert resp.status_code == 404

    def test_evaluate_rules(self, client):
        cluster_id = str(uuid4())
        client.post("/api/v1/clusters", json={
            "name": "alert-test-cluster",
            "environment": "test",
            "connector_type": "k8s",
        })
        client.post("/api/v1/alerts/rules", json={
            "name": "Eval Rule",
            "cluster_id": cluster_id,
            "condition_type": "gpu_utilization",
            "operator": "lt",
            "threshold": 50,
        })
        resp = client.post(f"/api/v1/alerts/rules/{cluster_id}/evaluate")
        assert resp.status_code == 200
        evaluations = resp.json()
        assert isinstance(evaluations, list)

    def test_list_alerts(self, client):
        resp = client.get("/api/v1/alerts/records")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_acknowledge_alert(self, client):
        cluster_id = str(uuid4())
        client.post("/api/v1/clusters", json={
            "name": "ack-test-cluster",
            "environment": "test",
            "connector_type": "k8s",
        })
        rule_resp = client.post("/api/v1/alerts/rules", json={
            "name": "Ack Rule",
            "cluster_id": cluster_id,
        }).json()
        client.post(f"/api/v1/alerts/rules/{cluster_id}/evaluate")
        alerts = client.get("/api/v1/alerts/records").json()
        if alerts:
            alert_id = alerts[0]["id"]
            resp = client.post(f"/api/v1/alerts/records/{alert_id}/acknowledge", params={"user": "tester"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "acknowledged"

    def test_resolve_alert(self, client):
        cluster_id = str(uuid4())
        client.post("/api/v1/clusters", json={
            "name": "resolve-test-cluster",
            "environment": "test",
            "connector_type": "k8s",
        })
        client.post("/api/v1/alerts/rules", json={
            "name": "Resolve Rule",
            "cluster_id": cluster_id,
        })
        client.post(f"/api/v1/alerts/rules/{cluster_id}/evaluate")
        alerts = client.get("/api/v1/alerts/records").json()
        if alerts:
            alert_id = alerts[0]["id"]
            resp = client.post(f"/api/v1/alerts/records/{alert_id}/resolve")
            assert resp.status_code == 200
            assert resp.json()["status"] == "resolved"


class TestNotificationChannelAPI:
    def test_create_channel_slack(self, client):
        resp = client.post("/api/v1/alerts/channels", json={
            "name": "Slack Alerts",
            "channel_type": "slack",
            "config": {"webhook_url": "https://hooks.slack.com/test"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Slack Alerts"
        assert data["channel_type"] == "slack"
        assert "id" in data

    def test_create_channel_pagerduty(self, client):
        resp = client.post("/api/v1/alerts/channels", json={
            "name": "PagerDuty Prod",
            "channel_type": "pagerduty",
            "config": {"routing_key": "abc123"},
        })
        assert resp.status_code == 200
        assert resp.json()["channel_type"] == "pagerduty"

    def test_create_channel_opsgenie(self, client):
        resp = client.post("/api/v1/alerts/channels", json={
            "name": "OpsGenie Alerts",
            "channel_type": "opsgenie",
            "config": {"api_key": "genie-key-123"},
        })
        assert resp.status_code == 200
        assert resp.json()["channel_type"] == "opsgenie"

    def test_create_channel_email(self, client):
        resp = client.post("/api/v1/alerts/channels", json={
            "name": "Email Alerts",
            "channel_type": "email",
            "config": {
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "to_addrs": ["admin@example.com"],
            },
        })
        assert resp.status_code == 200
        assert resp.json()["channel_type"] == "email"

    def test_list_channels(self, client):
        prev = len(client.get("/api/v1/alerts/channels").json())
        client.post("/api/v1/alerts/channels", json={
            "name": "Chan 1", "channel_type": "slack",
        })
        client.post("/api/v1/alerts/channels", json={
            "name": "Chan 2", "channel_type": "pagerduty",
        })
        resp = client.get("/api/v1/alerts/channels")
        assert resp.status_code == 200
        assert len(resp.json()) == prev + 2

    def test_get_channel(self, client):
        create = client.post("/api/v1/alerts/channels", json={
            "name": "Get Me", "channel_type": "slack",
        }).json()
        resp = client.get(f"/api/v1/alerts/channels/{create['id']}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Get Me"

    def test_get_channel_not_found(self, client):
        resp = client.get(f"/api/v1/alerts/channels/{uuid4()}")
        assert resp.status_code == 404

    def test_update_channel(self, client):
        create = client.post("/api/v1/alerts/channels", json={
            "name": "Old", "channel_type": "slack",
        }).json()
        resp = client.patch(f"/api/v1/alerts/channels/{create['id']}", json={"name": "Updated", "enabled": False})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"
        assert not resp.json()["enabled"]

    def test_delete_channel(self, client):
        create = client.post("/api/v1/alerts/channels", json={
            "name": "Delete", "channel_type": "slack",
        }).json()
        resp = client.delete(f"/api/v1/alerts/channels/{create['id']}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        resp = client.get(f"/api/v1/alerts/channels/{create['id']}")
        assert resp.status_code == 404

    def test_test_channel(self, client):
        create = client.post("/api/v1/alerts/channels", json={
            "name": "Test Chan", "channel_type": "slack", "config": {},
        }).json()
        resp = client.post(f"/api/v1/alerts/channels/{create['id']}/test")
        assert resp.status_code == 200
        data = resp.json()
        assert "message_id" in data
        assert data["status"] in ("sent", "failed")

    def test_list_messages(self, client):
        resp = client.get("/api/v1/alerts/messages")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
