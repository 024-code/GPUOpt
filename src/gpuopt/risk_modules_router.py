from __future__ import annotations

from fastapi import APIRouter

from gpuopt.risk_r02_twin_calibration import (
    R02TwinCalibrationService,
    TwinConfidenceCalculator,
    TwinFallbackController,
    WorkloadFamilyCalibrator,
)
from gpuopt.risk_r03_canary import CanaryManager
from gpuopt.risk_r04_workload_class import WorkloadClassifier
from gpuopt.risk_r07_r09_r13_r14 import (
    ContractTestManager,
    DisputeManager,
    SecurityManager,
    TieredOptimizer,
)
from gpuopt.schemas import (
    CanaryDeployment,
    ContractTestCase,
    DisputeRecord,
    OptimizationTier,
    SecurityChampionCheck,
    SupportedVersion,
    ThreatModelEntry,
    TwinConfidenceLimits,
    TwinFallbackMode,
    WorkloadCapability,
    WorkloadClassResult,
    WorkloadFamilyProfile,
)

router = APIRouter(prefix="/api/v1/risk-modules", tags=["risk_modules"])

_r02_calibrator = WorkloadFamilyCalibrator()
_r02_confidence = TwinConfidenceCalculator()
_r02_fallback = TwinFallbackController()
_r02_service = R02TwinCalibrationService()
_r03_canary = CanaryManager()
_r04_classifier = WorkloadClassifier()
_r07_optimizer = TieredOptimizer()
_r09_security = SecurityManager()
_r13_dispute = DisputeManager()
_r14_contracts = ContractTestManager()


# ── R02: Twin Calibration ────────────────────────────────────────

@router.post("/r02/calibrate", response_model=dict)
def r02_calibrate_workload(workload: dict, twin_id: str = "default") -> dict:
    """Calibrate twin prediction for a workload."""
    return _r02_service.calibrate_for_workload(workload, twin_id)


@router.get("/r02/families", response_model=list[str])
def r02_list_families() -> list[str]:
    """List all known workload families."""
    return _r02_calibrator.list_families()


@router.get("/r02/profile/{family}", response_model=WorkloadFamilyProfile)
def r02_get_profile(family: str) -> WorkloadFamilyProfile:
    """Get profile for a workload family."""
    return _r02_calibrator.get_profile(family)


@router.post("/r02/record-outcome")
def r02_record_outcome(family: str, simulated: float, actual: float) -> dict:
    """Record actual vs simulated outcome for calibration."""
    _r02_calibrator.record_outcome(family, simulated, actual)
    return {"status": "recorded", "family": family}


@router.post("/r02/calibrate-prediction")
def r02_calibrate_prediction(family: str, raw_prediction: float) -> dict:
    """Get calibrated prediction with variance."""
    calibrated, variance = _r02_calibrator.calibrate_prediction(family, raw_prediction)
    return {"calibrated": calibrated, "variance": variance}


@router.post("/r02/confidence", response_model=TwinConfidenceLimits)
def r02_confidence_limits(twin_id: str, family: str, prediction_interval: float = 95.0) -> TwinConfidenceLimits:
    """Calculate confidence limits for a twin."""
    profile = _r02_calibrator.get_profile(family)
    return _r02_confidence.calculate(twin_id, profile, prediction_interval)


@router.post("/r02/fallback", response_model=TwinFallbackMode)
def r02_evaluate_fallback(twin_id: str, family: str, threshold: float = 0.5) -> TwinFallbackMode:
    """Evaluate fallback mode for a twin."""
    profile = _r02_calibrator.get_profile(family)
    limits = _r02_confidence.calculate(twin_id, profile)
    return _r02_fallback.evaluate(twin_id, limits, threshold)


# ── R03: Canary Deployment ───────────────────────────────────────

@router.post("/r03/canary", response_model=CanaryDeployment)
def r03_create_canary(action_id: str, action_type: str = "placement") -> CanaryDeployment:
    """Create a new canary deployment."""
    return _r03_canary.create(action_id, action_type)


@router.post("/r03/canary/{deployment_id}/advance", response_model=dict)
def r03_advance_canary(deployment_id: str, metrics: dict | None = None) -> dict:
    """Advance canary to next step."""
    return _r03_canary.advance_step(deployment_id, metrics or {})


@router.get("/r03/canary/{deployment_id}", response_model=CanaryDeployment | None)
def r03_get_canary(deployment_id: str) -> CanaryDeployment | None:
    """Get canary deployment status."""
    return _r03_canary.get_deployment(deployment_id)


@router.get("/r03/canaries", response_model=list[CanaryDeployment])
def r03_list_canaries() -> list[CanaryDeployment]:
    """List all canary deployments."""
    return _r03_canary.list_deployments()


# ── R04: Workload Classification ─────────────────────────────────

@router.post("/r04/classify", response_model=WorkloadClassResult)
def r04_classify_workload(workload: dict) -> WorkloadClassResult:
    """Classify a workload and get capability assessment."""
    return _r04_classifier.classify(workload)


@router.post("/r04/is-action-safe")
def r04_is_action_safe(workload: dict, action: str) -> dict:
    """Check if an action is safe for a workload."""
    safe, msg = _r04_classifier.is_action_safe(workload, action)
    return {"safe": safe, "message": msg}


# ── R07: Hierarchical Optimization ───────────────────────────────

@router.post("/r07/optimize", response_model=list[dict])
def r07_optimize(candidates: list[dict], tier: str = "cluster") -> list[dict]:
    """Optimize candidates at a specific tier."""
    return _r07_optimizer.optimize(candidates, tier)


@router.get("/r07/cache/{cache_key}", response_model=dict | None)
def r07_get_cached(cache_key: str) -> dict | None:
    """Get cached optimization result."""
    return _r07_optimizer.get_cached(cache_key)


@router.post("/r07/cache/{cache_key}")
def r07_set_cached(cache_key: str, result: dict, ttl_seconds: float = 60.0) -> dict:
    """Set cached optimization result."""
    _r07_optimizer.set_cached(cache_key, result, ttl_seconds)
    return {"status": "cached"}


@router.get("/r07/cache/stats", response_model=dict)
def r07_cache_stats() -> dict:
    """Get cache statistics."""
    return _r07_optimizer.get_stats()


@router.get("/r07/tiers", response_model=list[dict])
def r07_list_tiers() -> list[dict]:
    """List optimization tiers."""
    return _r07_optimizer.list_tiers()


# ── R09: Security Threat Model ───────────────────────────────────

@router.post("/r09/threat", response_model=ThreatModelEntry)
def r09_add_threat(
    category: str, description: str,
    severity: str = "medium", likelihood: str = "medium", mitigation: str = ""
) -> ThreatModelEntry:
    """Add a threat to the threat model."""
    return _r09_security.add_threat(category, description, severity, likelihood, mitigation)


@router.post("/r09/checks", response_model=list[SecurityChampionCheck])
def r09_run_checks() -> list[SecurityChampionCheck]:
    """Run security champion checks."""
    return _r09_security.run_checks()


@router.get("/r09/report", response_model=dict)
def r09_security_report() -> dict:
    """Get security report."""
    return _r09_security.get_report()


# ── R13: Dispute Workflow ────────────────────────────────────────

@router.post("/r13/dispute", response_model=DisputeRecord)
def r13_create_dispute(
    tenant_id: str, resource_type: str,
    claimed_usage: float, actual_usage: float, reason: str
) -> DisputeRecord:
    """Create a new dispute."""
    return _r13_dispute.create(tenant_id, resource_type, claimed_usage, actual_usage, reason)


@router.post("/r13/dispute/{dispute_id}/resolve", response_model=DisputeRecord | None)
def r13_resolve_dispute(dispute_id: str, resolution: str, accept_claim: bool = False) -> DisputeRecord | None:
    """Resolve a dispute."""
    return _r13_dispute.resolve(dispute_id, resolution, accept_claim)


@router.get("/r13/dispute/{dispute_id}", response_model=DisputeRecord | None)
def r13_get_dispute(dispute_id: str) -> DisputeRecord | None:
    """Get a dispute by ID."""
    return _r13_dispute.get_dispute(dispute_id)


@router.get("/r13/disputes", response_model=list[DisputeRecord])
def r13_list_disputes(tenant_id: str | None = None) -> list[DisputeRecord]:
    """List disputes, optionally filtered by tenant."""
    return _r13_dispute.list_disputes(tenant_id)


@router.get("/r13/disputes/stats", response_model=dict)
def r13_dispute_stats() -> dict:
    """Get dispute statistics."""
    return _r13_dispute.get_stats()


# ── R14: Contract Tests & Version Policy ─────────────────────────

@router.post("/r14/contract-test", response_model=ContractTestCase)
def r14_add_contract_test(
    adapter_type: str, version: str,
    input_example: dict, expected_output: dict
) -> ContractTestCase:
    """Add a contract test case."""
    return _r14_contracts.add_test(adapter_type, version, input_example, expected_output)


@router.post("/r14/contract-test/{test_id}/run", response_model=ContractTestCase | None)
def r14_run_contract_test(test_id: str, actual_output: dict | None = None) -> ContractTestCase | None:
    """Run a contract test."""
    return _r14_contracts.run_test(test_id, actual_output)


@router.post("/r14/contract-tests/run-all", response_model=list[ContractTestCase])
def r14_run_all_contract_tests(adapter_type: str | None = None) -> list[ContractTestCase]:
    """Run all contract tests, optionally filtered by adapter type."""
    return _r14_contracts.run_all(adapter_type)


@router.get("/r14/contract-test/{test_id}", response_model=ContractTestCase | None)
def r14_get_contract_test(test_id: str) -> ContractTestCase | None:
    """Get a contract test by ID."""
    return _r14_contracts.get_test(test_id)


@router.post("/r14/version-policy", response_model=SupportedVersion)
def r14_set_version_policy(
    adapter_type: str, min_version: str, max_version: str,
    current_version: str, deprecation_date: str = "",
    sunset_date: str = "", migration_guide: str = ""
) -> SupportedVersion:
    """Set version policy for an adapter type."""
    return _r14_contracts.set_version_policy(
        adapter_type, min_version, max_version, current_version,
        deprecation_date, sunset_date, migration_guide
    )


@router.get("/r14/version-policy/{adapter_type}", response_model=SupportedVersion | None)
def r14_get_version_policy(adapter_type: str) -> SupportedVersion | None:
    """Get version policy for an adapter type."""
    return _r14_contracts.get_version_policy(adapter_type)


@router.get("/r14/version-supported", response_model=tuple[bool, str])
def r14_is_version_supported(adapter_type: str, version: str) -> tuple[bool, str]:
    """Check if a version is supported."""
    return _r14_contracts.is_version_supported(adapter_type, version)