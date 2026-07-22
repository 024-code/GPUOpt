from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Hard Constraints ──────────────────────────────────────────

class HardConstraint(StrEnum):
    GPU_MEMORY = "gpu_memory"
    GPU_TOPOLOGY = "gpu_topology"
    GPU_COMPATIBILITY = "gpu_compatibility"
    TENANT_QUOTA = "tenant_quota"
    TENANT_PRIORITY = "tenant_priority"
    INFERENCE_SLO_LATENCY = "inference_slo_latency"
    INFERENCE_SLO_ERROR = "inference_slo_error"
    DATA_LOCALITY = "data_locality"
    APPROVED_ZONES = "approved_zones"
    CHECKPOINT_POLICY = "checkpoint_policy"
    PREEMPTION_POLICY = "preemption_policy"
    ACTION_BLAST_RADIUS = "action_blast_radius"


class ConstraintResult(BaseModel):
    constraint: HardConstraint
    passed: bool
    reason: str = ""
    detail: dict[str, Any] = {}


# ── Soft Objectives ───────────────────────────────────────────

class ObjectiveWeight(BaseModel):
    gpu_utilization: float = 1.0
    throughput: float = 1.0
    queue_time_reduction: float = 1.0
    job_completion_time: float = 1.0
    gpu_hours_per_token: float = 1.0
    power_efficiency: float = 1.0
    carbon_footprint: float = 1.0
    fairness: float = 1.0
    starvation_reduction: float = 1.0
    minimal_movement: float = 1.0
    operational_churn: float = 1.0


class ObjectiveScore(BaseModel):
    objective: str
    score: float = 0.0
    weight: float = 1.0
    weighted_score: float = 0.0
    detail: str = ""


# ── Tenant Profile ────────────────────────────────────────────

class TenantObjectiveProfile(BaseModel):
    tenant_id: str
    weights: ObjectiveWeight = Field(default_factory=ObjectiveWeight)
    priority_class: int = 0
    gpu_quota: int = 0
    approved_zones: list[str] = []
    slo_max_latency_ms: float | None = None
    slo_max_error_rate_pct: float | None = None


# ── Workload Spec ─────────────────────────────────────────────

class WorkloadSpec(BaseModel):
    id: str = ""
    tenant_id: str = ""
    job_name: str = ""
    gpu_count: int = 1
    gpu_model: str = ""
    memory_per_gpu_gb: float = 0.0
    required_gpu_topology: str = "any"
    requires_nvlink: bool = False
    requires_co_location: bool = False
    inference_deployment: bool = False
    estimated_runtime_minutes: int = 0
    checkpoint_interval_minutes: int = 0
    preemptible: bool = False
    data_location: str = ""
    approved_zones: list[str] = []
    priority: int = 0
    estimated_tokens_per_step: int = 0
    estimated_samples_per_step: int = 0


class NodeCandidate(BaseModel):
    node_id: str
    cluster_id: str = ""
    zone: str = ""
    gpu_model: str = ""
    total_gpus: int = 0
    free_gpus: int = 0
    gpu_memory_per_gpu_gb: float = 0.0
    has_nvlink: bool = False
    cpu_cores: int = 0
    cpu_memory_gb: float = 0.0
    current_gpu_utilization_pct: float = 0.0
    current_power_watts: float = 0.0
    current_temperature_c: float = 0.0
    current_gpu_hours: float = 0.0
    carbon_intensity_g_per_kwh: float = 0.0
    running_jobs: int = 0
    labels: dict[str, str] = {}


# ── Optimization Request / Result ─────────────────────────────

class OptimizationCandidate(BaseModel):
    workload: WorkloadSpec
    target_node: NodeCandidate | None = None
    target_nodes: list[NodeCandidate] = []
    constraints: list[ConstraintResult] = []
    objective_scores: list[ObjectiveScore] = []
    total_utility: float = 0.0
    feasible: bool = False
    action: str = ""


class OptimizationRequest(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    workloads: list[WorkloadSpec]
    candidates: list[NodeCandidate]
    tenant_profiles: dict[str, TenantObjectiveProfile] = {}
    global_weights: ObjectiveWeight = Field(default_factory=ObjectiveWeight)
    maximize_throughput: bool = True
    minimize_queue_time: bool = True
    minimize_jct: bool = True
    minimize_slo_violations: bool = True
    minimize_cost: bool = True
    minimize_power: bool = True
    minimize_disruption: bool = True
    preserve_fairness: bool = True
    preserve_reliability: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OptimizationResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    candidates: list[OptimizationCandidate] = []
    feasible_count: int = 0
    infeasible_count: int = 0
    best_candidate: OptimizationCandidate | None = None
    summary: str = ""


# ── Default objective weights ─────────────────────────────────

DEFAULT_OBJECTIVE_WEIGHTS: dict[str, float] = {
    "gpu_utilization": 0.15,
    "throughput": 0.15,
    "queue_time_reduction": 0.12,
    "job_completion_time": 0.10,
    "gpu_hours_per_token": 0.10,
    "power_efficiency": 0.08,
    "carbon_footprint": 0.05,
    "fairness": 0.10,
    "starvation_reduction": 0.08,
    "minimal_movement": 0.04,
    "operational_churn": 0.03,
}
