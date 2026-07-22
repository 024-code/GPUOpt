from __future__ import annotations

from fastapi import APIRouter, Query

from gpuopt.gpu_usage_inventory import GpuUsageInventoryService
from gpuopt.gpu_usage_inventory_schemas import ClusterInventory, DcgmTelemetryResponse

router = APIRouter(prefix="/api/v1/gpu-usage-inventory", tags=["gpu_usage_inventory"])
_service = GpuUsageInventoryService()


@router.get("/health")
def health() -> dict:
    return _service.health()


@router.get("/cluster", response_model=ClusterInventory)
def get_cluster_inventory(cluster_id: str | None = Query(default=None)) -> ClusterInventory:
    return _service.get_inventory(cluster_id)


@router.get("/dcgm-telemetry", response_model=DcgmTelemetryResponse)
def get_dcgm_telemetry(cluster_id: str | None = Query(default=None), num_gpus: int = Query(default=8, ge=1, le=64)) -> DcgmTelemetryResponse:
    return _service.get_dcgm_telemetry(cluster_id, num_gpus)
