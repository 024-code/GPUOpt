from __future__ import annotations

from fastapi import APIRouter, HTTPException

from gpuopt.risk_r07_r09_r13_r14 import (
    TieredOptimizer,
    SecurityManager,
    DisputeManager,
    ContractTestManager,
)
from gpuopt.schemas import (
    OptimizationTier,
    CachedOptimizationResult,
    ThreatModelEntry,
    SecurityChampionCheck,
    DisputeRecord,
    ContractTestCase,
    SupportedVersion,
)

router = APIRouter(prefix="/api/v1/risk", tags=["risk_r07_r09_r13_r14"])

# R07: Hierarchical Optimization
_optimizer = TieredOptimizer()


@router.post("/r07/optimize")
def r07_optimize(candidates: list[dict], tier: str = "cluster") -> list[dict]:
    return _optimizer.optimize(candidates, tier)


@router.get("/r07/tiers", response_model=list[OptimizationTier])
def r07_list_tiers() -> list[OptimizationTier]:
    from gpuopt.risk_r07_r09_r13_r14 import TieredOptimizer
    return TieredOptimizer.TIERS


@router.get("/r07/cache/stats")
def r07_cache_stats() -> dict:
    return _optimizer.get_stats()


@router.post("/r07/cache/set")
def r07_cache_set(cache_key: str, result: dict, ttl_seconds: float = 60.0) -> dict:
    _optimizer.set_cached(cache_key, result, ttl_seconds)
    return {"status": "cached", "key": cache_key}


@router.get("/r07/cache/get")
def r07_cache_get(cache_key: str) -> dict | None:
    return _optimizer.get_cached(cache_key)


# R09: Security Threat Model
_security = SecurityManager()


@router.post("/r09/threats", response_model=ThreatModelEntry)
def r09_add_threat(
    category: str, description: str, severity: str = "medium",
    likelihood: str = "medium", mitigation: str = "",
) -> ThreatModelEntry:
    return _security.add_threat(category, description, severity, likelihood, mitigation)


@router.get("/r09/checks", response_model=list[SecurityChampionCheck])
def r09_run_checks() -> list[SecurityChampionCheck]:
    return _security.run_checks()


@router.get("/r09/report")
def r09_get_report() -> dict:
    return _security.get_report()


# R13: Dispute Workflow
_disputes = DisputeManager()


@router.post("/r13/disputes", response_model=DisputeRecord)
def r13_create_dispute(
    tenant_id: str, resource_type: str, claimed_usage: float,
    actual_usage: float, reason: str,
) -> DisputeRecord:
    return _disputes.create(tenant_id, resource_type, claimed_usage, actual_usage, reason)


@router.post("/r13/disputes/{dispute_id}/resolve", response_model=DisputeRecord)
def r13_resolve_dispute(dispute_id: str, resolution: str, accept_claim: bool = False) -> DisputeRecord:
    resolved = _disputes.resolve(dispute_id, resolution, accept_claim)
    if not resolved:
        raise HTTPException(status_code=404, detail="Dispute not found")
    return resolved


@router.get("/r13/disputes/{dispute_id}", response_model=DisputeRecord)
def r13_get_dispute(dispute_id: str) -> DisputeRecord:
    dispute = _disputes.get_dispute(dispute_id)
    if not dispute:
        raise HTTPException(status_code=404, detail="Dispute not found")
    return dispute


@router.get("/r13/disputes", response_model=list[DisputeRecord])
def r13_list_disputes(tenant_id: str | None = None) -> list[DisputeRecord]:
    return _disputes.list_disputes(tenant_id)


@router.get("/r13/disputes/stats")
def r13_dispute_stats() -> dict:
    return _disputes.get_stats()


# R14: Contract Tests & Version Policy
_contracts = ContractTestManager()


@router.post("/r14/tests", response_model=ContractTestCase)
def r14_add_test(
    adapter_type: str, version: str, input_example: dict, expected_output: dict,
) -> ContractTestCase:
    return _contracts.add_test(adapter_type, version, input_example, expected_output)


@router.post("/r14/tests/{test_id}/run", response_model=ContractTestCase)
def r14_run_test(test_id: str, actual_output: dict | None = None) -> ContractTestCase:
    result = _contracts.run_test(test_id, actual_output)
    if not result:
        raise HTTPException(status_code=404, detail="Test not found")
    return result


@router.post("/r14/tests/run-all", response_model=list[ContractTestCase])
def r14_run_all(adapter_type: str | None = None) -> list[ContractTestCase]:
    return _contracts.run_all(adapter_type)


@router.get("/r14/tests/{test_id}", response_model=ContractTestCase)
def r14_get_test(test_id: str) -> ContractTestCase:
    test = _contracts.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    return test


@router.post("/r14/versions", response_model=SupportedVersion)
def r14_set_version_policy(
    adapter_type: str, min_version: str, max_version: str, current_version: str,
    deprecation_date: str = "", sunset_date: str = "", migration_guide: str = "",
) -> SupportedVersion:
    return _contracts.set_version_policy(
        adapter_type, min_version, max_version, current_version,
        deprecation_date, sunset_date, migration_guide,
    )


@router.get("/r14/versions/{adapter_type}", response_model=SupportedVersion)
def r14_get_version_policy(adapter_type: str) -> SupportedVersion:
    policy = _contracts.get_version_policy(adapter_type)
    if not policy:
        raise HTTPException(status_code=404, detail="Version policy not found")
    return policy


@router.get("/r14/versions/{adapter_type}/check")
def r14_check_version(adapter_type: str, version: str) -> dict:
    supported, msg = _contracts.is_version_supported(adapter_type, version)
    return {"supported": supported, "message": msg}