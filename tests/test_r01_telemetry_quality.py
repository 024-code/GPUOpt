from __future__ import annotations

from datetime import datetime, timezone

from src.gpuopt.schemas import (
    JobTelemetry,
    QueueTelemetry,
    TelemetrySnapshot,
)
from src.gpuopt.telemetry_quality import (
    OnboardingManager,
    TelemetryFallbackHandler,
    TelemetryQualityScorer,
    TelemetryQualityService,
)


def _make_snapshot(gpu_ok: bool = True, queue_ok: bool = True, job_ok: bool = True,
                   age_seconds: float = 0.0, bad_devices: bool = False) -> TelemetrySnapshot:
    gpu = {}
    if gpu_ok:
        devices = [
            {"index": 0, "memory_total_mb": 81920, "memory_used_mb": 40960,
             "utilization_percent": 50.0, "temperature_celsius": 60.0, "power_draw_watts": 300},
        ]
        if bad_devices:
            devices[0]["memory_used_mb"] = 999999
            devices[0]["utilization_percent"] = 150.0
            devices[0]["temperature_celsius"] = 200.0
        gpu = {"total_gpus": 1, "devices": devices, "total_memory_mb": 81920, "used_memory_mb": 40960}
    queues = []
    if queue_ok:
        queues.append(QueueTelemetry(queue_name="main", queue_depth=10, pending_jobs=5, running_jobs=5))
    jobs = []
    if job_ok:
        jobs.append(JobTelemetry(job_id="j1", state="running", gpu_utilization_avg=50.0))
    ts = TelemetrySnapshot(
        cluster_id="test",
        gpu_snapshot=gpu,
        queues=queues,
        jobs=jobs,
    )
    if age_seconds > 0:
        from datetime import timedelta
        ts.collected_at = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat()
    return ts


# ── Quality Scoring Tests ─────────────────────────────────────

def test_quality_score_perfect():
    scorer = TelemetryQualityScorer()
    snap = _make_snapshot()
    score = scorer.score_snapshot(snap)
    assert score.overall > 0.7
    assert len(score.issues) == 0


def test_quality_score_missing_gpu():
    scorer = TelemetryQualityScorer()
    snap = _make_snapshot(gpu_ok=False)
    score = scorer.score_snapshot(snap)
    assert score.completeness < 0.7
    assert any("GPU snapshot missing" in i for i in score.issues)
    assert score.overall < 0.9


def test_quality_score_missing_queues():
    scorer = TelemetryQualityScorer()
    snap = _make_snapshot(queue_ok=False)
    score = scorer.score_snapshot(snap)
    assert any("Queue telemetry missing" in i for i in score.issues)


def test_quality_score_stale():
    scorer = TelemetryQualityScorer()
    snap = _make_snapshot(age_seconds=600)
    score = scorer.score_snapshot(snap)
    assert score.freshness < 0.3
    assert any("Stale" in i for i in score.issues)


def test_quality_score_inconsistent():
    scorer = TelemetryQualityScorer()
    snap = _make_snapshot(bad_devices=True)
    score = scorer.score_snapshot(snap)
    assert score.consistency < 0.7
    assert any("Inconsistent" in i for i in score.issues)


def test_quality_score_empty_snapshot():
    scorer = TelemetryQualityScorer()
    snap = TelemetrySnapshot(cluster_id="empty")
    score = scorer.score_snapshot(snap)
    assert score.completeness < 0.3
    assert any("GPU snapshot missing" in i for i in score.issues)


# ── Fallback Tests ────────────────────────────────────────────

def test_fallback_primary_available():
    handler = TelemetryFallbackHandler()
    data, record = handler.resolve("test", {"gpu_count": 4, "gpu_utilization_percent": 75.0})
    assert not record.using_fallback
    assert data["gpu_count"] == 4


def test_fallback_primary_missing_cache_hit():
    handler = TelemetryFallbackHandler()
    handler.store("test", {"gpu_count": 4, "gpu_utilization_percent": 50.0})
    data, record = handler.resolve("test", None)
    assert record.using_fallback
    assert record.fallback_tier == "cached"
    assert data["gpu_count"] == 4


def test_fallback_primary_missing_no_cache():
    handler = TelemetryFallbackHandler()
    data, record = handler.resolve("nonexistent", None)
    assert record.using_fallback
    assert record.fallback_tier == "default"
    assert data["gpu_count"] == 0
    assert data["gpu_utilization_percent"] == 0.0


def test_fallback_partial_data():
    handler = TelemetryFallbackHandler()
    handler.store("partial", {"temperature_celsius": 45.0})
    data, record = handler.resolve("partial", {"gpu_count": 2},
                                    required_fields=["gpu_count", "temperature_celsius"])
    assert data["gpu_count"] == 2
    assert data["temperature_celsius"] == 45.0


def test_fallback_cache_health():
    handler = TelemetryFallbackHandler()
    health = handler.get_cache_health()
    assert health["cache_size"] >= 0


# ── Onboarding Tests ──────────────────────────────────────────

def test_onboarding_register():
    mgr = OnboardingManager()
    status = mgr.register_source("nvidia-dcgm", "gpu", critical=True)
    assert status.phase == "discovery"
    assert status.tier == "onboarding"
    assert status.contract_valid


def test_onboarding_advance_phase():
    mgr = OnboardingManager()
    mgr.register_source("test-source", "gpu")
    result = mgr.advance_phase("src-1", checks_passed=2)
    assert result["phase_completed"] == "discovery"
    assert result["next_phase"] == "contract"
    assert result["tier"] == "discovery"


def test_onboarding_all_phases():
    mgr = OnboardingManager()
    mgr.register_source("full-source", "gpu")
    sid = "src-1"
    for i in range(6):
        result = mgr.advance_phase(sid)
        if "error" in result:
            break
    status = mgr.get_status(sid)
    assert status is not None
    assert status.tier == "gold"


def test_onboarding_list_sources():
    mgr = OnboardingManager()
    mgr.register_source("s1", "gpu")
    mgr.register_source("s2", "queue")
    sources = mgr.list_sources()
    assert len(sources) == 2


def test_onboarding_quality_update():
    mgr = OnboardingManager()
    mgr.register_source("q-source", "gpu")
    sid = "src-1"
    scorer = TelemetryQualityScorer()
    snap = _make_snapshot()
    score = scorer.score_snapshot(snap)
    mgr.update_quality(sid, score)
    status = mgr.get_status(sid)
    assert status is not None
    assert status.quality is not None


# ── Unified Service Tests ─────────────────────────────────────

def test_quality_service_process():
    svc = TelemetryQualityService()
    snap = _make_snapshot()
    result = svc.process_snapshot(snap)
    assert "quality" in result
    assert "needs_fallback" in result
    assert result["action"] in ("accept", "warn")


def test_quality_service_get_data():
    svc = TelemetryQualityService()
    result = svc.get_data("test-src", {"gpu_count": 4})
    assert "data" in result
    assert "fallback" in result


def test_quality_service_health():
    svc = TelemetryQualityService()
    health = svc.health()
    assert health["status"] == "healthy"
    assert "cache" in health
    assert "sources_onboarded" in health
