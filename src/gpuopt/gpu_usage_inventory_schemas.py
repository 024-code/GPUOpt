from __future__ import annotations

from pydantic import BaseModel, Field


class GpuInventorySnapshot(BaseModel):
    total_gpu_capacity: int = 8
    total_allocatable_gpus: int = 8
    allocated_to_all_workloads: int = 6
    allocated_to_inference: int = 2
    estimated_free: int = 2
    allocation_utilization_pct: float = 75.0
    observed_utilization_source: str = "mock_snapshot"
    summary: str = "Allocation is not utilization. K8s shows reserved GPUs via resource limits, not active compute. DCGM telemetry required for engine util, tensor activity, DRAM activity, framebuffer usage, power, temperature, clocks, and XID errors."


class NodeGpuAllocation(BaseModel):
    node_name: str
    gpu_ids: list[int]
    allocated_to: str
    gpu_model: str = "NVIDIA H100-SXM-80GB"
    memory_gib: float = 80.0
    pods: list[str] = Field(default_factory=list)


class ClusterInventory(BaseModel):
    cluster_id: str = "mock-cluster-1"
    snapshot: GpuInventorySnapshot = Field(default_factory=GpuInventorySnapshot)
    nodes: list[NodeGpuAllocation] = Field(default_factory=list)
    dcgm_required_message: str = (
        "DCGM exporter must be deployed to measure true GPU utilization. "
        "Without DCGM, K8s resource limits only reflect allocation, not utilization."
    )


class GpuTelemetrySample(BaseModel):
    gpu_index: int
    engine_util_pct: float
    tensor_activity_pct: float
    dram_activity_pct: float
    framebuffer_used_gib: float
    framebuffer_total_gib: float
    power_draw_watts: float
    gpu_temp_celsius: float
    memory_temp_celsius: float
    pcie_tx_bytes_per_sec: float
    pcie_rx_bytes_per_sec: float
    nvlink_tx_bytes_per_sec: float
    nvlink_rx_bytes_per_sec: float
    xid_errors: int = 0


class DcgmTelemetryResponse(BaseModel):
    cluster_id: str = "mock-cluster-1"
    daemonset_running: bool = True
    samples: list[GpuTelemetrySample] = Field(default_factory=list)
    summary: str = ""
