from __future__ import annotations

from src.gpuopt.risk_r02_twin_calibration import (
    R02TwinCalibrationService,
    TwinConfidenceCalculator,
    TwinFallbackController,
    WorkloadFamilyCalibrator,
)
from src.gpuopt.risk_r03_canary import CanaryManager
from src.gpuopt.risk_r04_workload_class import WorkloadClassifier
from src.gpuopt.risk_r07_r09_r13_r14 import (
    ContractTestManager,
    DisputeManager,
    SecurityManager,
    TieredOptimizer,
)


# ── R02: Twin Calibration Tests ──────────────────────────────

def test_workload_family_detection():
    cal = WorkloadFamilyCalibrator()
    f1 = cal.get_family({"framework": "pytorch", "gpu_required": 8, "max_duration_minutes": 1440})
    assert f1 in ("llm_training", "cnn_training")
    f2 = cal.get_family({"framework": "pytorch", "type": "llm_inference", "gpu_required": 1})
    assert f2 == "llm_inference"


def test_workload_family_profile():
    cal = WorkloadFamilyCalibrator()
    profile = cal.get_profile("llm_training")
    assert profile.family == "llm_training"
    assert profile.typical_gpu_count == 8
    assert profile.calibration_bias != 0


def test_calibrate_prediction():
    cal = WorkloadFamilyCalibrator()
    cal.record_outcome("llm_training", 0.8, 0.75)
    cal.record_outcome("llm_training", 0.8, 0.72)
    pred, variance = cal.calibrate_prediction("llm_training", 0.8)
    assert isinstance(pred, float)
    assert variance > 0


def test_twin_confidence_limits():
    cal = WorkloadFamilyCalibrator()
    profile = cal.get_profile("llm_training")
    cc = TwinConfidenceCalculator()
    limits = cc.calculate("twin-1", profile)
    assert limits.twin_id == "twin-1"
    assert limits.lower_bound <= limits.upper_bound
    assert limits.confidence_score > 0


def test_twin_fallback_controller():
    cal = WorkloadFamilyCalibrator()
    profile = cal.get_profile("llm_training")
    cc = TwinConfidenceCalculator()
    limits = cc.calculate("twin-1", profile)
    fb = TwinFallbackController()
    mode = fb.evaluate("twin-1", limits, threshold=0.5)
    assert mode.twin_id == "twin-1"
    assert mode.mode in ("full", "recommendation_only", "disabled")


def test_r02_full_service():
    svc = R02TwinCalibrationService()
    result = svc.calibrate_for_workload(
        {"framework": "pytorch", "gpu_required": 8, "type": "llm_training"},
        "twin-demo",
    )
    assert "family" in result
    assert "profile" in result
    assert "confidence_limits" in result
    assert "fallback_mode" in result


# ── R03: Canary Tests ────────────────────────────────────────

def test_canary_create():
    mgr = CanaryManager()
    dep = mgr.create("action-1", "placement")
    assert dep.status == "running"
    assert len(dep.steps) == 5
    assert dep.steps[0].name == "1% canary"


def test_canary_advance():
    mgr = CanaryManager()
    dep = mgr.create("action-2", "scale_down")
    result = mgr.advance_step(dep.deployment_id, {"error_rate": 0.0})
    assert result["status"] in ("advancing", "completed", "rolled_back")


def test_canary_rollback_on_failure():
    mgr = CanaryManager()
    dep = mgr.create("action-3", "placement")
    result = mgr.advance_step(dep.deployment_id, {"error_rate": 0.99})
    assert result["status"] == "rolled_back"


def test_canary_complete():
    mgr = CanaryManager()
    dep = mgr.create("action-4", "placement")
    for _ in range(10):
        result = mgr.advance_step(dep.deployment_id, {"error_rate": 0.0, "latency_p99": 50})
        if result["status"] in ("completed", "rolled_back"):
            break
    assert result["status"] in ("completed", "rolled_back")


# ── R04: Workload Classification Tests ───────────────────────

def test_classify_pytorch():
    clf = WorkloadClassifier()
    result = clf.classify({"job_id": "j1", "framework": "pytorch", "max_duration_minutes": 120})
    assert result.workload_id == "j1"
    assert result.capability.supports_checkpoint is True
    assert "checkpoint" in result.recommended_actions


def test_classify_inference():
    clf = WorkloadClassifier()
    result = clf.classify({"job_id": "j2", "framework": "onnx", "max_duration_minutes": 999})
    assert result.capability.supports_checkpoint is False
    assert "preempt" in result.unsafe_actions


def test_classify_unknown():
    clf = WorkloadClassifier()
    result = clf.classify({"job_id": "j3", "framework": "unknown"})
    assert result.capability.classification in ("fragile", "unknown")
    assert result.capability.supports_elastic is False


def test_is_action_safe():
    clf = WorkloadClassifier()
    safe, msg = clf.is_action_safe({"job_id": "j4", "framework": "pytorch"}, "checkpoint")
    assert safe is True
    unsafe, msg = clf.is_action_safe({"job_id": "j5", "framework": "onnx"}, "preempt")
    assert unsafe is False


# ── R07: Hierarchical Optimization Tests ─────────────────────

def test_tiered_optimizer_prune():
    opt = TieredOptimizer()
    candidates = [{"id": i, "score": i / 10} for i in range(20)]
    result = opt.optimize(candidates, "global")
    assert len(result) <= 50
    for c in result:
        assert c["score"] >= 0.3


def test_tiered_optimizer_cache():
    opt = TieredOptimizer()
    result = {"placement": "node-1"}
    opt.set_cached("key-1", result, ttl_seconds=60)
    cached = opt.get_cached("key-1")
    assert cached == result
    assert opt.get_stats()["hits"] == 1


def test_tiered_optimizer_cache_miss():
    opt = TieredOptimizer()
    cached = opt.get_cached("nonexistent")
    assert cached is None


def test_tiered_optimizer_tiers():
    opt = TieredOptimizer()
    tiers = opt.list_tiers()
    assert len(tiers) == 4
    assert tiers[0]["tier"] == "global"


# ── R09: Security Tests ──────────────────────────────────────

def test_security_add_threat():
    sec = SecurityManager()
    threat = sec.add_threat("authentication", "Weak API keys", "high", "medium", "Enforce key rotation")
    assert threat.threat_id == "T001"
    assert threat.severity == "high"


def test_security_run_checks():
    sec = SecurityManager()
    checks = sec.run_checks()
    assert len(checks) == 7
    assert all(c.check_id.startswith(("auth", "sec")) for c in checks)


def test_security_report():
    sec = SecurityManager()
    report = sec.get_report()
    assert "risk_score" in report
    assert "status" in report


# ── R13: Dispute Tests ───────────────────────────────────────

def test_dispute_create():
    mgr = DisputeManager()
    dispute = mgr.create("tenant-1", "gpu", claimed_usage=8, actual_usage=4, reason="Billing discrepancy")
    assert dispute.tenant_id == "tenant-1"
    assert dispute.status == "open"


def test_dispute_resolve():
    mgr = DisputeManager()
    dispute = mgr.create("tenant-1", "memory", claimed_usage=160, actual_usage=80, reason="Overcharge")
    resolved = mgr.resolve(dispute.dispute_id, "accepted - adjusting usage", accept_claim=True)
    assert resolved is not None
    assert resolved.status == "resolved"


def test_dispute_list():
    mgr = DisputeManager()
    mgr.create("t1", "gpu", 4, 2, "test")
    mgr.create("t2", "memory", 64, 32, "test")
    disputes = mgr.list_disputes("t1")
    assert len(disputes) == 1


def test_dispute_stats():
    mgr = DisputeManager()
    stats = mgr.get_stats()
    assert "total" in stats


# ── R14: Contract Tests ──────────────────────────────────────

def test_contract_add_test():
    mgr = ContractTestManager()
    test = mgr.add_test("kubernetes", "1.28", {"cmd": "get pods"}, {"status": "ok"})
    assert test.adapter_type == "kubernetes"
    assert test.passed is False


def test_contract_run_test():
    mgr = ContractTestManager()
    test = mgr.add_test("slurm", "23.11", {"cmd": "squeue"}, {"jobs": []})
    result = mgr.run_test(test.test_id, {"jobs": []})
    assert result is not None
    assert result.passed is True


def test_contract_run_all():
    mgr = ContractTestManager()
    mgr.add_test("k8s", "1.28", {"a": 1}, {"b": 2})
    mgr.add_test("k8s", "1.29", {"c": 3}, {"d": 4})
    results = mgr.run_all("k8s")
    assert len(results) == 2


def test_version_policy():
    mgr = ContractTestManager()
    mgr.set_version_policy("kubernetes", "1.20", "1.30", "1.28", migration_guide="Upgrade guide at docs/")
    supported, msg = mgr.is_version_supported("kubernetes", "1.28")
    assert supported is True
    supported, msg = mgr.is_version_supported("kubernetes", "1.15")
    assert supported is False
    policy = mgr.get_version_policy("kubernetes")
    assert policy is not None
    assert policy.min_version == "1.20"
