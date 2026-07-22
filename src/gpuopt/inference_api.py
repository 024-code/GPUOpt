from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from gpuopt.inference_schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    GpuUsageResponse,
)
from gpuopt.inference_services import AnalyzeService
from gpuopt.gpu_usage import GpuInventoryService

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])

_inventory_service = GpuInventoryService()
_analyze_service = AnalyzeService()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(req: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return _analyze_service.analyze(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/clusters/{cluster_id}/gpu-usage", response_model=GpuUsageResponse)
def gpu_usage_endpoint(
    cluster_id: UUID,
    cluster_name: str = Query(default=""),
) -> GpuUsageResponse:
    try:
        return _inventory_service.get_usage(cluster_id, cluster_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
