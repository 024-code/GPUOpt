from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from gpuopt.model_governance.approval import ApprovalManager
from gpuopt.model_governance.champion_challenger import ChampionChallenger
from gpuopt.model_governance.drift_monitor import DriftMonitor
from gpuopt.model_governance.fallback import FallbackEngine
from gpuopt.model_governance.governance import ModelGovernor, get_governor
from gpuopt.model_governance.models import (
    ApprovalStatus,
    DriftType,
    FallbackConfig,
    GovernanceConfig,
    ModelActionClass,
    ModelStatus,
    ModelVersion,
)
from gpuopt.model_governance.registry import ModelRegistry


@pytest.fixture(autouse=True)
def reset_governor() -> None:
    get_governor().registry.clear_all()
    get_governor().drift_monitor.reset()
    get_governor().champion_challenger._evaluations.clear()
    get_governor().approval.clear()


@pytest.fixture()
def registry() -> ModelRegistry:
    return get_governor().registry


@pytest.fixture()
def sample_version(registry: ModelRegistry) -> ModelVersion:
    v = ModelVersion(
        model_name="test-model", version="1.0.0",
        action_class=ModelActionClass.RECOMMENDATION_SCORING,
        owner="alice", features=["util_pct", "mem_pct", "queue_depth"],
        training_metrics={"mae": 0.05, "accuracy": 0.92},
        training_data_window_start=datetime.now(timezone.utc) - timedelta(days=30),
        training_data_window_end=datetime.now(timezone.utc) - timedelta(days=1),
    )
    registry.register(v)
    return v


class TestModelRegistry:
    def test_register_and_get(self, registry: ModelRegistry):
        v = ModelVersion(model_name="m", version="1", action_class=ModelActionClass.ANOMALY_DETECTION)
        registry.register(v)
        assert registry.get(v.id) is v

    def test_list_by_action_class(self, registry: ModelRegistry, sample_version: ModelVersion):
        results = registry.list(action_class=ModelActionClass.RECOMMENDATION_SCORING)
        assert len(results) == 1
        assert results[0].id == sample_version.id

    def test_list_by_status(self, registry: ModelRegistry, sample_version: ModelVersion):
        registry.update_status(sample_version.id, ModelStatus.CHAMPION)
        results = registry.list(status=ModelStatus.CHAMPION)
        assert len(results) >= 1

    def test_champion_promotion(self, registry: ModelRegistry, sample_version: ModelVersion):
        registry.update_status(sample_version.id, ModelStatus.CHALLENGER)
        promoted = registry.promote_challenger(ModelActionClass.RECOMMENDATION_SCORING)
        assert promoted is not None
        assert promoted.status == ModelStatus.CHAMPION
        assert registry.get_champion(ModelActionClass.RECOMMENDATION_SCORING) is not None

    def test_metadata(self, registry: ModelRegistry, sample_version: ModelVersion):
        registry.update_status(sample_version.id, ModelStatus.CHAMPION)
        meta = registry.get_metadata("test-model", ModelActionClass.RECOMMENDATION_SCORING)
        assert meta.version_count >= 1
        assert meta.current_champion is not None

    def test_prediction_tracking(self, registry: ModelRegistry, sample_version: ModelVersion):
        registry.record_prediction(sample_version.id, 0.85)
        registry.record_prediction(sample_version.id, 0.90)
        meta = registry.get_metadata("test-model", ModelActionClass.RECOMMENDATION_SCORING)
        assert meta.total_predictions >= 2
        assert meta.avg_confidence > 0


class TestChampionChallenger:
    def test_start_evaluation(self, registry: ModelRegistry):
        champ = ModelVersion(model_name="m", version="1", action_class=ModelActionClass.DEMAND_FORECAST)
        chal = ModelVersion(model_name="m", version="2", action_class=ModelActionClass.DEMAND_FORECAST)
        registry.register(champ)
        registry.register(chal)
        registry.update_status(champ.id, ModelStatus.CHAMPION)
        cc = ChampionChallenger(registry)
        ev = cc.start_evaluation(champ, chal)
        assert ev.champion_id == champ.id
        assert ev.challenger_id == chal.id

    def test_record_and_complete(self, registry: ModelRegistry):
        champ = ModelVersion(model_name="m", version="1", action_class=ModelActionClass.DEMAND_FORECAST)
        chal = ModelVersion(model_name="m", version="2", action_class=ModelActionClass.DEMAND_FORECAST)
        registry.register(champ)
        registry.register(chal)
        registry.update_status(champ.id, ModelStatus.CHAMPION)
        cc = ChampionChallenger(registry)
        ev = cc.start_evaluation(champ, chal)
        for _ in range(100):
            cc.record_result(ev.id, 0.7, 0.9, {"accuracy": {"champion": 0.7, "challenger": 0.9}})
        completed = cc.complete_evaluation(ev.id)
        assert completed.sample_count == 100
        assert completed.challenger_wins > completed.champion_wins

    def test_evaluate_shadow(self, registry: ModelRegistry):
        champ = ModelVersion(model_name="m", version="1", action_class=ModelActionClass.DEMAND_FORECAST)
        chal = ModelVersion(model_name="m", version="2", action_class=ModelActionClass.DEMAND_FORECAST)
        registry.register(champ)
        registry.register(chal)
        registry.update_status(champ.id, ModelStatus.CHAMPION)
        cc = ChampionChallenger(registry)
        ev = cc.start_evaluation(champ, chal)
        result = cc.evaluate_shadow(0.8, 0.85, 0.82, ev.id)
        assert result.sample_count == 1


class TestDriftMonitor:
    def test_feature_drift(self, registry: ModelRegistry, sample_version: ModelVersion):
        dm = DriftMonitor(registry)
        dm.record_features(sample_version, {"util_pct": 50.0, "mem_pct": 60.0})
        for i in range(20):
            dm.record_features(sample_version, {"util_pct": 50.0 + i * 5, "mem_pct": 60.0})
        reports = dm.check_all(sample_version)
        drift = [r for r in reports if r.drift_type == DriftType.FEATURE_DISTRIBUTION and r.drift_score > 0]
        assert len(drift) >= 0

    def test_prediction_error_drift(self, registry: ModelRegistry, sample_version: ModelVersion):
        dm = DriftMonitor(registry)
        for _ in range(60):
            dm.record_prediction_error(sample_version, 0.05)
        for _ in range(60):
            dm.record_prediction_error(sample_version, 0.50)
        reports = dm.check_all(sample_version)
        err_drift = [r for r in reports if r.drift_type == DriftType.PREDICTION_ERROR]
        assert len(err_drift) == 1
        assert err_drift[0].drift_score > 0.1

    def test_data_staleness(self, registry: ModelRegistry):
        old = ModelVersion(
            model_name="stale", version="1", action_class=ModelActionClass.DEMAND_FORECAST,
            training_data_window_end=datetime.now(timezone.utc) - timedelta(days=60),
        )
        registry.register(old)
        dm = DriftMonitor(registry)
        reports = dm.check_all(old)
        staleness = [r for r in reports if r.drift_type == DriftType.DATA_STALENESS]
        assert len(staleness) == 1
        assert staleness[0].drift_score > 0.5

    def test_workload_mix_drift(self, registry: ModelRegistry, sample_version: ModelVersion):
        dm = DriftMonitor(registry)
        dm.record_workload_distribution(sample_version, {"training": 10, "inference": 5})
        dm.record_workload_distribution(sample_version, {"training": 3, "inference": 12})
        reports = dm.check_all(sample_version)
        wl_drift = [r for r in reports if r.drift_type == DriftType.WORKLOAD_MIX]
        assert len(wl_drift) == 1


class TestFallback:
    def test_fallback_triggered_by_confidence(self, registry: ModelRegistry, sample_version: ModelVersion):
        fb = FallbackEngine(registry)
        should, reason = fb.should_fallback(sample_version, 0.1, [])
        assert should
        assert "confidence" in reason.lower()

    def test_fallback_triggered_by_drift(self, registry: ModelRegistry, sample_version: ModelVersion):
        fb = FallbackEngine(registry, FallbackConfig(drift_score_threshold=0.3))
        report = type("DriftReport", (), {"drift_score": 0.8, "drift_type": type("DT", (), {"value": "test"})})()
        report.drift_type.value = "feature_distribution"
        should, reason = fb.should_fallback(sample_version, 0.8, [report])
        assert should
        assert "drift" in reason.lower()

    def test_fallback_triggered_by_staleness(self, registry: ModelRegistry):
        old = ModelVersion(
            model_name="stale", version="1", action_class=ModelActionClass.DEMAND_FORECAST,
            training_data_window_end=datetime.now(timezone.utc) - timedelta(hours=100),
        )
        registry.register(old)
        fb = FallbackEngine(registry, FallbackConfig(max_data_stale_hours=48))
        should, reason = fb.should_fallback(old, 0.8, [])
        assert should
        assert "old" in reason.lower() or "stale" in reason.lower()

    def test_activate_deactivate(self, registry: ModelRegistry):
        fb = FallbackEngine(registry)
        assert not fb.is_fallback_active(ModelActionClass.DEMAND_FORECAST)
        fb.activate_fallback(ModelActionClass.DEMAND_FORECAST, "test")
        assert fb.is_fallback_active(ModelActionClass.DEMAND_FORECAST)
        fb.deactivate_fallback(ModelActionClass.DEMAND_FORECAST)
        assert not fb.is_fallback_active(ModelActionClass.DEMAND_FORECAST)

    def test_deterministic_heuristic(self, registry: ModelRegistry, sample_version: ModelVersion):
        fb = FallbackEngine(registry)
        result = fb.get_heuristic(sample_version, {"heuristic_score": 85.0})
        assert result["fallback"]
        assert result["strategy"] == "deterministic_heuristic"


class TestApproval:
    def test_requires_approval_for_high_impact(self, registry: ModelRegistry, sample_version: ModelVersion):
        am = ApprovalManager(registry)
        assert am.requires_approval(sample_version)

    def test_request_approve_reject(self, registry: ModelRegistry, sample_version: ModelVersion):
        am = ApprovalManager(registry)
        req = am.request_approval(sample_version, requested_by="alice")
        assert req.status == ApprovalStatus.PENDING

        approved = am.approve(req.id, reviewer="bob", certification_days=90)
        assert approved is not None
        assert approved.status == ApprovalStatus.APPROVED
        assert sample_version.approved_by == "bob"

    def test_reject(self, registry: ModelRegistry, sample_version: ModelVersion):
        am = ApprovalManager(registry)
        req = am.request_approval(sample_version)
        rejected = am.reject(req.id, reviewer="bob", notes="not ready")
        assert rejected.status == ApprovalStatus.REJECTED
        assert sample_version.status == ModelStatus.REJECTED

    def test_recertification(self, registry: ModelRegistry, sample_version: ModelVersion):
        am = ApprovalManager(registry)
        req = am.request_approval(sample_version)
        am.approve(req.id, reviewer="bob", certification_days=-1)  # expired immediately
        due = am.check_recertification()
        assert len(due) >= 1


class TestGovernor:
    def test_register_model(self):
        gov = get_governor()
        v = gov.register_model("new-model", "1.0", ModelActionClass.ANOMALY_DETECTION, owner="alice")
        assert v.owner == "alice"
        assert v.status in (ModelStatus.CHALLENGER, ModelStatus.DRAFT)

    def test_register_high_impact_requires_approval(self):
        gov = get_governor()
        v = gov.register_model("hi-model", "1.0", ModelActionClass.RECOMMENDATION_SCORING, owner="alice")
        requests = gov.approval.list_requests()
        assert len(requests) >= 1

    def test_predict_with_fallback(self):
        gov = get_governor()
        v = gov.register_model("pred-model", "1.0", ModelActionClass.DEMAND_FORECAST, owner="alice")
        gov.set_champion(v.id)
        result = gov.predict(ModelActionClass.DEMAND_FORECAST, {}, confidence=0.1)
        assert result["fallback"]

    def test_predict_no_champion(self):
        gov = get_governor()
        result = gov.predict(ModelActionClass.DEMAND_FORECAST, {}, confidence=0.9)
        assert result["fallback"]
        assert "No champion" in result["reason"]

    def test_periodic_checks(self):
        gov = get_governor()
        results = gov.run_periodic_checks()
        assert "recertification_due" in results
        assert "drift_alerts" in results
        assert "fallback_status" in results


class TestGovernanceAPI:
    def test_register_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/governance/models/register?model_name=api-model&version=1.0&action_class=anomaly_detection&owner=bob")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "api-model"
        assert data["version"] == "1.0"

    def test_list_models_endpoint(self, client: TestClient):
        client.post("/api/v1/governance/models/register?model_name=m1&version=1&action_class=demand_forecast")
        resp = client.get("/api/v1/governance/models")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_get_metadata_endpoint(self, client: TestClient):
        resp = client.get("/api/v1/governance/metadata/anomaly_detection")
        assert resp.status_code == 200
        assert resp.json()["action_class"] == "anomaly_detection"

    def test_champion_challenger_endpoint(self, client: TestClient):
        c1 = client.post("/api/v1/governance/models/register?model_name=cc&version=1&action_class=demand_forecast").json()
        c2 = client.post("/api/v1/governance/models/register?model_name=cc&version=2&action_class=demand_forecast").json()
        resp = client.post(f"/api/v1/governance/champion-challenger/start?champion_id={c1['id']}&challenger_id={c2['id']}")
        assert resp.status_code == 200
        ev = resp.json()
        assert ev["champion_id"] == c1["id"]

    def test_drift_reports_endpoint(self, client: TestClient):
        resp = client.get("/api/v1/governance/drift/reports")
        assert resp.status_code == 200

    def test_fallback_status_endpoint(self, client: TestClient):
        resp = client.get("/api/v1/governance/fallback/status/demand_forecast")
        assert resp.status_code == 200
        assert resp.json()["action_class"] == "demand_forecast"

    def test_approval_flow_endpoint(self, client: TestClient):
        v = client.post("/api/v1/governance/models/register?model_name=hi&version=1&action_class=recommendation_scoring").json()
        reqs = client.get("/api/v1/governance/approval/requests").json()
        assert len(reqs) >= 1
        rid = reqs[0]["id"]
        approve = client.post(f"/api/v1/governance/approval/approve/{rid}?reviewer=bob&notes=looks_good&certification_days=90")
        assert approve.status_code == 200
        assert approve.json()["status"] == "approved"

    def test_periodic_check_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/governance/periodic-check")
        assert resp.status_code == 200
        assert "drift_alerts" in resp.json()
