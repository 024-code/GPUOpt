from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Layer 1: Request Metrics ─────────────────────────────────


class RequestMetrics(BaseModel):
    arrival_rate_req_per_sec: float
    prompt_tokens_avg: float
    output_tokens_avg: float
    queue_time_ms_avg: float
    ttft_ms_avg: float
    tpot_ms_avg: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_rate: float
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


class RequestDecision(BaseModel):
    slo_breached: bool
    slo_breach_reason: str = ""
    recommended_batch_size: int
    recommended_max_batch_size: int
    recommended_replicas: int
    batching_suggestion: str
    primary_decision: str = ""


# ── Layer 2: GPU Metrics ─────────────────────────────────────


class GpuMetricSample(BaseModel):
    gpu_index: int
    engine_util_pct: float
    tensor_activity_pct: float
    dram_activity_pct: float
    framebuffer_used_gib: float
    framebuffer_free_gib: float
    sm_clock_mhz: float
    mem_clock_mhz: float
    pcie_tx_bytes_per_sec: float
    pcie_rx_bytes_per_sec: float


class GpuMetricsResult(BaseModel):
    samples: list[GpuMetricSample]
    avg_engine_util: float
    avg_tensor_activity: float
    avg_dram_activity: float
    total_framebuffer_used_gib: float
    total_framebuffer_free_gib: float


class GpuDecision(BaseModel):
    bottleneck: Literal["compute", "memory", "io", "balanced"]
    bottleneck_reason: str
    is_overcommitted: bool
    recommended_gpu_count: int
    recommended_memory_gib: float
    primary_decision: str = ""


# ── Layer 3: Reliability Metrics ─────────────────────────────


class ReliabilityIncident(BaseModel):
    incident_type: Literal["oom", "xid_error", "pod_restart", "retry_exceeded", "failed_request"]
    count: int
    last_occurrence: datetime | None = None
    affected_resources: list[str] = Field(default_factory=list)


class ReliabilityMetrics(BaseModel):
    oom_count: int
    xid_error_count: int
    pod_restart_count: int
    retry_count: int
    failed_request_count: int
    incidents: list[ReliabilityIncident]


class ReliabilityDecision(BaseModel):
    requires_rollback: bool
    rollback_reason: str = ""
    requires_incident_response: bool
    incident_response_action: str = ""
    recommended_action: str
    priority: Literal["critical", "high", "medium", "low"]
    primary_decision: str = ""


# ── Layer 4: Thermal/Power Metrics ───────────────────────────


class ThermalMetrics(BaseModel):
    gpu_temp_celsius_avg: float
    gpu_temp_celsius_max: float
    memory_temp_celsius_avg: float
    power_draw_watts_avg: float
    power_draw_watts_max: float
    power_limit_watts: float
    throttling_active: bool
    throttling_reason: str = ""


class ThermalDecision(BaseModel):
    recommended_power_cap_watts: float
    requires_cooling_action: bool
    cooling_action: str = ""
    requires_rescheduling: bool
    rescheduling_reason: str = ""
    primary_decision: str = ""


# ── Layer 5: Placement Metrics ───────────────────────────────


class NodeTopologyInfo(BaseModel):
    node_name: str
    gpu_indices: list[int]
    per_gpu_utilization: list[float]
    numa_node: int = 0
    nvlink_active: bool = True
    nvlink_bandwidth_gb_per_sec: float = 600.0
    network_traffic_mbps: float = 0.0


class PlacementMetrics(BaseModel):
    nodes: list[NodeTopologyInfo]
    avg_gpu_utilization: float
    min_gpu_utilization: float
    max_gpu_utilization: float
    nvlink_topology: str = ""
    numa_aware: bool = True


class PlacementDecision(BaseModel):
    recommended_tensor_parallelism: int
    recommended_pipeline_parallelism: int
    recommended_node_count: int
    requires_numa_pinning: bool
    placement_strategy: str
    primary_decision: str = ""


# ── Layer 6: Economics Metrics ───────────────────────────────


class EconomicsMetrics(BaseModel):
    total_gpu_hours: float
    tokens_per_gpu_second: float
    cost_per_million_tokens: float
    idle_gpu_hours: float
    reserved_gpu_hours: float
    total_cost_usd: float
    potential_savings_usd: float
    utilization_effective_pct: float


class EconomicsDecision(BaseModel):
    cost_optimization_action: str
    estimated_savings_usd: float
    recommended_idle_reduction: str
    recommended_reservation_policy: str
    primary_decision: str = ""


# ── Aggregated ───────────────────────────────────────────────


class LayerStatus(BaseModel):
    layer: str
    metrics_count: int
    status: Literal["healthy", "warning", "critical"]
    decisions: list[str]


class MetricsDashboard(BaseModel):
    request: RequestMetrics
    request_decision: RequestDecision
    gpu: GpuMetricsResult
    gpu_decision: GpuDecision
    reliability: ReliabilityMetrics
    reliability_decision: ReliabilityDecision
    thermal: ThermalMetrics
    thermal_decision: ThermalDecision
    placement: PlacementMetrics
    placement_decision: PlacementDecision
    economics: EconomicsMetrics
    economics_decision: EconomicsDecision
    layer_statuses: list[LayerStatus]
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    summary: str = ""
