from __future__ import annotations

from fastapi import APIRouter

from gpuopt.risk_r02_twin_calibration import R02TwinCalibrationService
from gpuopt.schemas import (
    TwinConfidenceLimits,
    TwinFallbackMode,
    WorkloadFamilyProfile,
)

router = APIRouter(prefix="/api/v1/risk/r02-twin-calibration", tags=["risk_r02"])
_service = R02TwinCalibrationService()


@router.post("/calibrate")
def calibrate_workload(workload: dict, twin_id: str = "default") -> dict:
    return _service.calibrate_for_workload(workload, twin_id)


@router.get("/families")
def list_families() -> list[str]:
    return _service.calibrator.list_families()


@router.get("/families/{family}/profile", response_model=WorkloadFamilyProfile)
def get_profile(family: str) -> WorkloadFamilyProfile:
    return _service.calibrator.get_profile(family)


@router.get("/families/{family}/confidence", response_model=TwinConfidenceLimits)
def get_confidence(family: str, twin_id: str = "default", interval: float = 95.0) -> TwinConfidenceLimits:
    profile = _service.calibrator.get_profile(family)
    return _service.confidence.calculate(twin_id, profile, interval)


@router.get("/families/{family}/fallback", response_model=TwinFallbackMode)
def get_fallback(family: str, twin_id: str = "default", threshold: float = 0.5) -> TwinFallbackMode:
    profile = _service.calibrator.get_profile(family)
    limits = _service.confidence.calculate(twin_id, profile)
    return _service.fallback.evaluate(twin_id, limits, threshold)


@router.post("/families/{family}/record-outcome")
def record_outcome(family: str, simulated: float, actual: float) -> dict:
    _service.calibrator.record_outcome(family, simulated, actual)
    return {"status": "recorded", "family": family, "simulated": simulated, "actual": actual}