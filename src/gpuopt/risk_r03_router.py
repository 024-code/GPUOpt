from __future__ import annotations

from fastapi import APIRouter, HTTPException

from gpuopt.risk_r03_canary import CanaryManager
from gpuopt.schemas import CanaryDeployment

router = APIRouter(prefix="/api/v1/risk/r03-canary", tags=["risk_r03"])
_manager = CanaryManager()


@router.post("/create", response_model=CanaryDeployment)
def create_canary(action_id: str, action_type: str = "placement") -> CanaryDeployment:
    return _manager.create(action_id, action_type)


@router.post("/{deployment_id}/advance")
def advance_canary(deployment_id: str, metrics: dict | None = None) -> dict:
    result = _manager.advance_step(deployment_id, metrics)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{deployment_id}", response_model=CanaryDeployment)
def get_canary(deployment_id: str) -> CanaryDeployment:
    dep = _manager.get_deployment(deployment_id)
    if not dep:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return dep


@router.get("/", response_model=list[CanaryDeployment])
def list_canaries() -> list[CanaryDeployment]:
    return _manager.list_deployments()