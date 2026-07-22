from __future__ import annotations

from gpuopt.optimization_analysis import OptimizationAnalysisService
from gpuopt.optimization_analysis_schemas import (
    DecisionPattern,
    OptimizationAnalysisResult,
    OptimizationRecommendation,
    SyntheticObservation,
)


# ── Unit: Service ─────────────────────────────────────────────

def test_analyze_default():
    svc = OptimizationAnalysisService()
    result = svc.analyze()
    assert isinstance(result, OptimizationAnalysisResult)
    assert result.observation.gpu_utilization_pct == 37.0
    assert result.observation.memory_utilization_pct == 42.0
    assert result.observation.within_latency_target is True
    assert result.observation.within_throughput_target is True
    assert result.observation.queue_pressure is False


def test_analyze_underutilized_recommendation():
    svc = OptimizationAnalysisService()
    result = svc.analyze(SyntheticObservation(
        gpu_utilization_pct=30.0,
        memory_utilization_pct=40.0,
        within_latency_target=True,
        within_throughput_target=True,
    ))
    priorities = [r.priority for r in result.recommendations]
    assert "P1" in priorities
    categories = [r.category for r in result.recommendations]
    assert "cost" in categories


def test_analyze_memory_pressure():
    svc = OptimizationAnalysisService()
    result = svc.analyze(SyntheticObservation(
        gpu_utilization_pct=60.0,
        memory_utilization_pct=85.0,
    ))
    categories = [r.category for r in result.recommendations]
    assert "capacity" in categories


def test_analyze_latency_violation():
    svc = OptimizationAnalysisService()
    result = svc.analyze(SyntheticObservation(
        gpu_utilization_pct=70.0,
        memory_utilization_pct=50.0,
        within_latency_target=False,
    ))
    categories = [r.category for r in result.recommendations]
    assert "performance" in categories
    assert any("latency" in r.recommended_action.lower() or "batching" in r.recommended_action.lower() for r in result.recommendations)


def test_analyze_throughput_violation():
    svc = OptimizationAnalysisService()
    result = svc.analyze(SyntheticObservation(
        gpu_utilization_pct=90.0,
        within_throughput_target=False,
    ))
    categories = [r.category for r in result.recommendations]
    assert "performance" in categories


def test_analyze_queue_pressure():
    svc = OptimizationAnalysisService()
    result = svc.analyze(SyntheticObservation(
        gpu_utilization_pct=80.0,
        queue_pressure=True,
        within_latency_target=True,
        within_throughput_target=True,
    ))
    categories = [r.category for r in result.recommendations]
    assert "operations" in categories


def test_analyze_multiple_triggers():
    svc = OptimizationAnalysisService()
    result = svc.analyze(SyntheticObservation(
        gpu_utilization_pct=35.0,
        memory_utilization_pct=82.0,
        queue_pressure=True,
        within_latency_target=False,
        within_throughput_target=False,
    ))
    assert len(result.recommendations) >= 3


def test_decision_patterns_loaded():
    svc = OptimizationAnalysisService()
    assert len(svc.DECISION_PATTERNS) == 7


def test_decision_patterns_content():
    svc = OptimizationAnalysisService()
    ids = [p.pattern_id for p in svc.DECISION_PATTERNS]
    assert "opt-underutilized" in ids
    assert "opt-memory-pressure" in ids
    assert "opt-queue-pressure" in ids
    assert "opt-dram-bound" in ids
    assert "opt-compute-bound" in ids
    assert "opt-imbalance" in ids
    assert "opt-reliability-first" in ids


def test_analysis_contains_recommended_experiment():
    svc = OptimizationAnalysisService()
    result = svc.analyze()
    assert "batching" in result.recommended_experiment.lower()
    assert "MIG" in result.recommended_experiment


def test_analysis_summary():
    svc = OptimizationAnalysisService()
    result = svc.analyze()
    assert "37.0%" in result.summary
    assert "42.0%" in result.summary


def test_health():
    svc = OptimizationAnalysisService()
    h = svc.health()
    assert h["status"] == "healthy"
    assert h["decision_patterns_loaded"] == 7


# ── API Tests ─────────────────────────────────────────────────

def test_health_api():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/optimization-analysis/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


def test_analyze_api_default():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/optimization-analysis/analyze")
        assert r.status_code == 200
        data = r.json()
        assert data["observation"]["gpu_utilization_pct"] == 37.0
        assert len(data["recommendations"]) >= 1
        assert len(data["decision_patterns"]) >= 1


def test_analyze_api_custom():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/optimization-analysis/analyze", json={
            "gpu_utilization_pct": 85.0,
            "memory_utilization_pct": 90.0,
            "queue_pressure": True,
            "within_latency_target": False,
            "within_throughput_target": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["observation"]["gpu_utilization_pct"] == 85.0
        assert data["observation"]["memory_utilization_pct"] == 90.0
        assert data["observation"]["queue_pressure"] is True
        assert data["observation"]["within_latency_target"] is False
        assert len(data["recommendations"]) >= 2


def test_analyze_api_low_util():
    from fastapi.testclient import TestClient
    from gpuopt.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/optimization-analysis/analyze", json={
            "gpu_utilization_pct": 25.0,
            "memory_utilization_pct": 30.0,
        })
        assert r.status_code == 200
        data = r.json()
        priorities = [rec["priority"] for rec in data["recommendations"]]
        assert "P1" in priorities
