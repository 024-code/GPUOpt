from __future__ import annotations

from fastapi import APIRouter

from gpuopt.risk_r04_workload_class import WorkloadClassifier
from gpuopt.schemas import WorkloadClassResult, WorkloadCapability

router = APIRouter(prefix="/api/v1/risk/r04-workload-class", tags=["risk_r04"])
_classifier = WorkloadClassifier()


@router.post("/classify", response_model=WorkloadClassResult)
def classify_workload(workload: dict) -> WorkloadClassResult:
    return _classifier.classify(workload)


@router.post("/check-action")
def check_action_safety(workload: dict, action: str) -> dict:
    safe, msg = _classifier.is_action_safe(workload, action)
    return {"safe": safe, "message": msg}


@router.get("/capabilities/{framework}", response_model=WorkloadCapability)
def get_framework_capabilities(framework: str) -> WorkloadCapability:
    from gpuopt.risk_r04_workload_class import FRAMEWORK_CAPABILITIES
    caps = FRAMEWORK_CAPABILITIES.get(framework.lower(), FRAMEWORK_CAPABILITIES["unknown"])
    return WorkloadCapability(
        workload_type=frame_work if (frame_work := caps.get("classification")) else "unknown",
        **{k: v for k, v in caps.items() if k in WorkloadCapability.model_fields},
    )


@router.get("/capabilities", response_model=dict[str, WorkloadCapability])
def list_all_capabilities() -> dict[str, WorkloadCapability]:
    from gpuopt.risk_r04_workload_class import FRAMEWORK_CAPABILITIES
    return {
        k: WorkloadCapability(
            workload_type=v.get("classification", "unknown"),
            **{k2: v2 for k2, v2 in v.items() if k2 in WorkloadCapability.model_fields},
        )
        for k, v in FRAMEWORK_CAPABILITIES.items()
    }