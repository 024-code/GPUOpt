from __future__ import annotations

from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

from .rtx_manager import manager

router = APIRouter(prefix="/api/v1/rtx", tags=["RTX 4090 Cluster"])


class JobSubmitRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    required_gpus: int = Field(default=1, ge=1, le=256)
    required_memory_gb: float = Field(default=8.0, ge=0.1)
    estimated_runtime_hours: float = Field(default=1.0, ge=0.1)
    priority: int = Field(default=5, ge=1, le=10)


class JobSubmitResponse(BaseModel):
    success: bool
    job: dict
    message: str


@router.get("/status")
def get_cluster_status() -> dict:
    state = manager.get_cluster_state()
    return {
        "cluster_id": manager.cluster_id,
        "cluster_name": manager.cluster_name,
        "timestamp": state.timestamp.isoformat(),
        "gpus": [
            {
                "name": g.name,
                "index": g.index,
                "memory_total_gb": round(g.memory_total_gb, 1),
                "memory_used_gb": round(g.memory_used_gb, 1),
                "memory_free_gb": round(g.memory_free_gb, 1),
                "utilization_percent": round(g.utilization_percent, 1),
                "temperature_celsius": round(g.temperature_celsius, 1),
                "power_usage_watts": round(g.power_usage_watts, 1),
                "power_limit_watts": round(g.power_limit_watts, 1),
                "driver_version": g.driver_version,
                "cuda_version": g.cuda_version,
                "ecc_errors": g.ecc_errors,
                "pcie_link_gen": g.pcie_link_gen,
                "pcie_link_width": g.pcie_link_width,
                "is_available": g.is_available,
                "partition_id": g.partition_id,
                "physical_gpu_index": g.physical_gpu_index,
            }
            for g in state.gpus
        ],
        "partitions": [
            {
                "id": p.id,
                "node_name": p.node_name,
                "vram_gb": p.gb,
            }
            for p in manager.partitions
        ],
        "cpu": {
            "cores": state.cpu_cores,
            "usage_percent": round(state.cpu_usage_percent, 1),
        },
        "memory": {
            "total_gb": round(state.memory_total_gb, 1),
            "used_gb": round(state.memory_used_gb, 1),
            "free_gb": round(state.memory_free_gb, 1),
        },
        "aggregate": {
            "total_gpu_memory_gb": round(state.total_gpu_memory_gb, 1),
            "total_gpu_usage_percent": round(state.total_gpu_usage_percent, 1),
            "total_power_watts": round(state.total_power_watts, 1),
            "active_jobs": len(state.active_jobs),
        },
    }


@router.get("/gpus")
def list_gpus() -> list[dict]:
    state = manager.get_cluster_state()
    return [
        {
            "index": g.index,
            "name": g.name,
            "memory": {
                "total_gb": round(g.memory_total_gb, 1),
                "used_gb": round(g.memory_used_gb, 1),
                "free_gb": round(g.memory_free_gb, 1),
                "free_percent": round((g.memory_free_gb / g.memory_total_gb) * 100, 1) if g.memory_total_gb > 0 else 0,
            },
            "utilization_percent": round(g.utilization_percent, 1),
            "temperature_celsius": round(g.temperature_celsius, 1),
            "power": {
                "current_watts": round(g.power_usage_watts, 1),
                "limit_watts": round(g.power_limit_watts, 1),
                "usage_percent": round((g.power_usage_watts / g.power_limit_watts) * 100, 1) if g.power_limit_watts > 0 else 0,
            },
            "driver_version": g.driver_version,
            "cuda_version": g.cuda_version,
            "ecc_errors": g.ecc_errors,
            "pcie_link_gen": g.pcie_link_gen,
            "pcie_link_width": g.pcie_link_width,
            "partition_id": g.partition_id,
            "physical_gpu_index": g.physical_gpu_index,
            "health_status": (
                "healthy" if g.temperature_celsius < 80 and g.utilization_percent < 95 else "warning"
            ),
        }
        for g in state.gpus
    ]


@router.get("/partitions")
def list_partitions() -> list[dict]:
    return [
        {
            "id": p.id,
            "node_name": p.node_name,
            "vram_gb": p.gb,
            "status": "active",
        }
        for p in manager.partitions
    ]


@router.post("/submit", response_model=JobSubmitResponse)
def submit_job(req: JobSubmitRequest) -> JobSubmitResponse:
    result = manager.submit_job(req.model_dump())
    return JobSubmitResponse(**result)


@router.get("/jobs")
def list_jobs() -> dict:
    jobs = manager.list_jobs()
    return {"total_jobs": len(jobs), "jobs": jobs}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = manager.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/simulate")
def simulate_job(req: JobSubmitRequest) -> dict:
    return manager.simulate_job(req.model_dump())


@router.post("/detect")
def detect_cluster() -> dict:
    state = manager.get_cluster_state()
    return {
        "cluster_id": manager.cluster_id,
        "detected": {
            "gpus": len(state.gpus),
            "gpu_models": list({g.name for g in state.gpus}),
            "total_gpu_memory_gb": round(state.total_gpu_memory_gb, 1),
            "cuda_version": state.gpus[0].cuda_version if state.gpus else "N/A",
            "driver_version": state.gpus[0].driver_version if state.gpus else "N/A",
        },
        "partitions": [
            {
                "id": p.id,
                "node_name": p.node_name,
                "vram_gb": p.gb,
            }
            for p in manager.partitions
        ],
        "timestamp": state.timestamp.isoformat(),
    }


@router.get("/metrics")
def get_metrics() -> dict:
    state = manager.get_cluster_state()
    running = sum(1 for j in state.active_jobs if j.get("status") == "running")
    queued = sum(1 for j in state.active_jobs if j.get("status") == "queued")
    completed = sum(1 for j in state.active_jobs if j.get("status") == "completed")
    avg_temp = sum(g.temperature_celsius for g in state.gpus) / max(len(state.gpus), 1)

    return {
        "gpu": {
            "count": len(state.gpus),
            "total_memory_gb": round(state.total_gpu_memory_gb, 1),
            "total_utilization": round(state.total_gpu_usage_percent, 1),
            "average_temperature": round(avg_temp, 1),
            "total_power_watts": round(state.total_power_watts, 1),
            "partitions": len(manager.partitions),
            "details": [
                {
                    "gpu": g.index,
                    "utilization": round(g.utilization_percent, 1),
                    "memory_used_gb": round(g.memory_used_gb, 1),
                    "temperature": round(g.temperature_celsius, 1),
                    "power": round(g.power_usage_watts, 1),
                    "partition_id": g.partition_id,
                }
                for g in state.gpus
            ],
        },
        "system": {
            "cpu_cores": state.cpu_cores,
            "cpu_usage": round(state.cpu_usage_percent, 1),
            "memory_total_gb": round(state.memory_total_gb, 1),
            "memory_free_gb": round(state.memory_free_gb, 1),
        },
        "jobs": {
            "total": len(state.active_jobs),
            "running": running,
            "queued": queued,
            "completed": completed,
        },
    }