from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gpuopt.healing.auto_healer import AutoHealer, RemediationAction, RemediationResult


@pytest.fixture(autouse=True)
def reset_healer() -> None:
    from gpuopt.healing import router as healing_router_mod
    healing_router_mod._healer = AutoHealer()


def _telemetry(
    temp: float = 50,
    ecc: int = 0,
    mem: float = 50,
    util: float = 50,
    xid: int = 0,
) -> dict:
    return {
        "temperature": temp,
        "ecc_errors": ecc,
        "memory_utilization": mem,
        "gpu_utilization": util,
        "xid_errors": xid,
    }


class TestAutoHealer:
    def test_check_health_healthy(self):
        h = AutoHealer()
        result = h.check_node_health(_telemetry(temp=50, ecc=0, mem=50, xid=0))
        assert result["health_status"] == "healthy"
        assert result["risk_score"] == 0.0
        assert result["issues"] == []

    def test_check_health_critical_temp(self):
        h = AutoHealer()
        result = h.check_node_health(_telemetry(temp=90))
        assert result["health_status"] in ("critical", "degraded")
        assert result["risk_score"] >= 0.4
        assert any("temperature" in i.lower() for i in result["issues"])

    def test_check_health_critical_ecc(self):
        h = AutoHealer()
        result = h.check_node_health(_telemetry(ecc=25))
        assert result["risk_score"] >= 0.4
        assert any("ecc" in i.lower() for i in result["issues"])

    def test_check_health_critical_xid(self):
        h = AutoHealer()
        result = h.check_node_health(_telemetry(xid=15))
        assert result["risk_score"] >= 0.5
        assert any("xid" in i.lower() for i in result["issues"])

    def test_check_health_critical_memory(self):
        h = AutoHealer()
        result = h.check_node_health(_telemetry(mem=98))
        assert result["risk_score"] >= 0.3
        assert any("memory" in i.lower() for i in result["issues"])

    def test_check_health_saturation(self):
        h = AutoHealer()
        result = h.check_node_health(_telemetry(util=98))
        assert any("gpu" in i.lower() for i in result["issues"])

    def test_check_health_combined_risk(self):
        h = AutoHealer()
        result = h.check_node_health(_telemetry(temp=90, ecc=25, mem=98, xid=15))
        assert result["health_status"] == "critical"
        assert result["risk_score"] >= 0.8
        assert len(result["issues"]) >= 3

    @pytest.mark.parametrize("temp,ecc,mem,xid,expected_action", [
        (90, 25, 50, 0, RemediationAction.CORDON),    # 0.4 + 0.4 = 0.8 → cordon
        (80, 25, 50, 0, RemediationAction.DRAIN),      # 0.2 + 0.4 = 0.6 → drain
        (80, 15, 50, 0, RemediationAction.RESTART),    # 0.2 + 0.2 = 0.4 → restart
        (50, 0, 50, 0, None),                           # 0.0 → healthy
    ])
    def test_suggest_remediation(self, temp, ecc, mem, xid, expected_action):
        h = AutoHealer()
        result = h.suggest_remediation(_telemetry(temp=temp, ecc=ecc, mem=mem, xid=xid), "node-1")
        if expected_action is None:
            assert result is None
        else:
            assert result is not None
            assert result.action == expected_action
            assert result.node_id == "node-1"

    def test_execute_remediation_healthy(self):
        h = AutoHealer()
        result = h.execute_remediation(_telemetry(), "node-1")
        assert result.status == "skipped"
        assert result.action == RemediationAction.SCALE_DOWN

    def test_execute_remediation_unhealthy(self):
        h = AutoHealer()
        result = h.execute_remediation(_telemetry(temp=90, ecc=25), "node-1")
        assert result.status == "executed"
        assert result.action in RemediationAction
        assert result.duration_seconds > 0

    def test_history_tracking(self):
        h = AutoHealer()
        assert h.get_history() == []
        h.execute_remediation(_telemetry(temp=90, ecc=25), "node-1")
        h.execute_remediation(_telemetry(temp=88, ecc=20), "node-2")
        history = h.get_history()
        assert len(history) == 2
        assert history[0]["node_id"] == "node-1"
        assert history[0]["status"] == "executed"

    def test_active_remediations(self):
        h = AutoHealer()
        assert h.get_active_remediations() == {}
        h.execute_remediation(_telemetry(temp=90, ecc=25), "node-1")
        active = h.get_active_remediations()
        assert "node-1" in active

    def test_clear_history(self):
        h = AutoHealer()
        h.execute_remediation(_telemetry(temp=90, ecc=25), "node-1")
        assert len(h.get_history()) == 1
        h.clear_history()
        assert h.get_history() == []
        assert h.get_active_remediations() == {}


class TestAutoHealerAPI:
    def test_check_health_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/healing/check-health", json=_telemetry(temp=50))
        assert resp.status_code == 200
        data = resp.json()
        assert data["health_status"] == "healthy"

    def test_check_health_critical_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/healing/check-health", json=_telemetry(temp=90, ecc=25, mem=98, xid=15))
        assert resp.status_code == 200
        assert resp.json()["health_status"] == "critical"

    def test_suggest_endpoint_healthy(self, client: TestClient):
        resp = client.post("/api/v1/healing/suggest?node_id=n1", json=_telemetry())
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_suggest_endpoint_unhealthy(self, client: TestClient):
        resp = client.post("/api/v1/healing/suggest?node_id=n1", json=_telemetry(temp=90, ecc=25))
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "proposed"
        assert data["action"] in ("cordon", "drain", "restart", "reboot", "scale_up", "scale_down")

    def test_execute_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/healing/execute?node_id=n1", json=_telemetry(temp=90, ecc=25))
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "executed"
        assert "duration_seconds" in data

    def test_execute_endpoint_healthy(self, client: TestClient):
        resp = client.post("/api/v1/healing/execute?node_id=n1", json=_telemetry())
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"

    def test_history_endpoint(self, client: TestClient):
        resp = client.get("/api/v1/healing/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_active_endpoint(self, client: TestClient):
        resp = client.get("/api/v1/healing/active")
        assert resp.status_code == 200
        assert "active_remediations" in resp.json()

    def test_clear_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/healing/clear")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"

    def test_start_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/healing/start", json={"cluster_id": "gpu-production-01"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healing_started"
        assert data["cluster_id"] == "gpu-production-01"
