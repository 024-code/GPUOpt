from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


# ── K8s GPU Scheduling ───────────────────────────────────────


class GpuDeviceInfo(BaseModel):
    device_id: str
    gpu_model: str
    memory_gib: float
    memory_used_gib: float = 0.0
    compute_mode: str = "Default"
    mig_enabled: bool = False
    mig_device_id: str | None = None


class NodeGpuInfo(BaseModel):
    node_name: str
    gpu_count_total: int
    gpu_count_allocated: int = 0
    gpu_devices: list[GpuDeviceInfo] = Field(default_factory=list)
    allocatable_gpus: int = 0
    device_plugin_ready: bool = True
    labels: dict[str, str] = Field(default_factory=dict)


class AllocationRequest(BaseModel):
    gpu_count: int = Field(default=1, ge=1, le=256)
    gpu_model: str = ""
    memory_gib_per_gpu: float = Field(default=80.0)
    node_selector: dict[str, str] = Field(default_factory=dict)


class AllocationResult(BaseModel):
    allocated: bool
    gpu_ids: list[str] = Field(default_factory=list)
    node_name: str = ""
    reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class K8sGpuScheduleResponse(BaseModel):
    node_inventory: list[NodeGpuInfo]
    total_gpus: int
    allocated_gpus: int
    available_gpus: int
    device_plugin_status: Literal["ready", "degraded", "unavailable"]
    summary: str
    references: list[str]


# ── DCGM Exporter ─────────────────────────────────────────────


class DcgmMetricName(StrEnum):
    GPU_UTIL = "DCGM_FI_PROF_GPU_UTIL"
    MEM_COPY_UTIL = "DCGM_FI_PROF_MEM_COPY_UTIL"
    SM_OCCUPANCY = "DCGM_FI_PROF_SM_OCCUPANCY"
    MEM_TEMP = "DCGM_FI_DEV_MEMORY_TEMP"
    GPU_TEMP = "DCGM_FI_DEV_GPU_TEMP"
    POWER_DRAW = "DCGM_FI_DEV_POWER_USAGE"
    PCIE_TX = "DCGM_FI_PROF_PCIE_TX_BYTES"
    PCIE_RX = "DCGM_FI_PROF_PCIE_RX_BYTES"
    MEM_FREE = "DCGM_FI_DEV_FB_FREE"
    MEM_USED = "DCGM_FI_DEV_FB_USED"
    CLOCK_SM = "DCGM_FI_DEV_SM_CLOCK"
    CLOCK_MEM = "DCGM_FI_DEV_MEM_CLOCK"


class DcgmMetricSample(BaseModel):
    gpu_index: int
    metric: DcgmMetricName
    value: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class DcgmExporterTarget(BaseModel):
    pod_name: str = ""
    namespace: str = "default"
    node_name: str = ""
    metrics_endpoint: str = ""
    metrics_port: int = 9400
    labels: dict[str, str] = Field(default_factory=dict)


class DcgmMetricsResponse(BaseModel):
    targets: list[DcgmExporterTarget]
    samples: list[DcgmMetricSample]
    daemonset_running: bool
    prometheus_scrape_config: dict[str, Any]
    summary: str
    references: list[str]


# ── Hydro: Surrogate-based HPO ────────────────────────────────


class SurrogateType(StrEnum):
    GAUSSIAN_PROCESS = "gaussian_process"
    RANDOM_FOREST = "random_forest"
    TPE = "tpe"
    BOHAMIANN = "bohamiann"


class AcquisitionFunction(StrEnum):
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"


class HyperparameterRange(BaseModel):
    name: str
    type: Literal["float", "int", "categorical"]
    min_value: float | None = None
    max_value: float | None = None
    categories: list[str] | None = None


class TrialResult(BaseModel):
    trial_id: str
    hyperparameters: dict[str, Any]
    score: float | None = None
    duration_seconds: float | None = None
    status: Literal["pending", "running", "completed", "failed"]


class SurrogateState(BaseModel):
    surrogate_type: SurrogateType
    acquisition_function: AcquisitionFunction
    trials_completed: int
    best_score: float | None = None
    best_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    predictions: list[dict[str, Any]] = Field(default_factory=list)


class HydroHpoResponse(BaseModel):
    search_space: list[HyperparameterRange]
    surrogate: SurrogateState
    suggested_trial: dict[str, Any]
    expected_improvement: float
    uncertainty: float
    summary: str
    references: list[str]


# ── Lyra: Elastic Scheduling ──────────────────────────────────


class ResourcePoolType(StrEnum):
    TRAINING = "training"
    INFERENCE = "inference"
    BATCH = "batch"
    EXPERIMENTAL = "experimental"


class ResourcePool(BaseModel):
    pool_id: str
    pool_type: ResourcePoolType
    total_gpus: int
    allocated_gpus: int
    reserved_gpus: int = 0
    min_gpus: int = 0
    max_gpus: int = 0
    priority: int = Field(default=0, ge=0, le=100)


class CapacitySharingPolicy(BaseModel):
    policy_name: str
    description: str
    overcommit_factor: float = Field(default=1.0, ge=1.0, le=3.0)
    borrowing_enabled: bool = True
    preemption_allowed: bool = False
    priority_threshold: int = 50


class ElasticScalingAction(BaseModel):
    pool_id: str
    action: Literal["scale_up", "scale_down"]
    gpu_delta: int
    reason: str
    estimated_impact: str


class LyraScheduleResponse(BaseModel):
    pools: list[ResourcePool]
    capacity_sharing_policy: CapacitySharingPolicy
    scaling_actions: list[ElasticScalingAction]
    total_gpus: int
    utilization_pct: float
    summary: str
    references: list[str]


# ── JANUS: MoE Inference ──────────────────────────────────────


class MoELayerConfig(BaseModel):
    num_experts: int = Field(default=8, ge=1, le=1024)
    top_k: int = Field(default=2, ge=1, le=8)
    expert_capacity_factor: float = Field(default=1.25, ge=1.0, le=4.0)
    hidden_size: int = 4096
    intermediate_size: int = 14336


class ExpertLoad(BaseModel):
    expert_index: int
    load_pct: float = Field(ge=0, le=100)
    tokens_processed: int = 0
    tokens_dropped: int = 0
    is_balanced: bool = True


class AttentionConfig(BaseModel):
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    attention_dispatched: bool = True


class MoEProvisioningPlan(BaseModel):
    attention_gpus: int
    expert_gpus: int
    total_gpus: int
    attention_vs_expert_ratio: float
    expert_parallelism: int = 1


class MoEBalancingResult(BaseModel):
    expert_loads: list[ExpertLoad]
    balancing_loss: float
    capacity_violation: bool
    tokens_dropped_total: int
    recommended_expert_capacity_factor: float


class JanusMoeResponse(BaseModel):
    moe_config: MoELayerConfig
    attention_config: AttentionConfig
    expert_loads: list[ExpertLoad]
    provisioning: MoEProvisioningPlan
    balancing: MoEBalancingResult
    token_slo_achieved: bool
    summary: str
    references: list[str]


# ── SkipDecode: Adaptive Compute ──────────────────────────────


class SkipDecision(BaseModel):
    layer_index: int
    skipped: bool
    confidence: float = Field(ge=0, le=1)
    reason: str = ""


class SkipConfig(BaseModel):
    enabled: bool = True
    max_skip_layers: int = Field(default=10, ge=0, le=100)
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    fallback_strategy: Literal["all", "none", "selective"] = "selective"


class BatchingCompatibility(BaseModel):
    compatible: bool
    kv_cache_overhead_bytes: int = 0
    requires_recomputation: bool = False
    estimated_speedup: float = 1.0


class ValidationResult(BaseModel):
    perplexity_delta: float = 0.0
    accuracy_delta: float = 0.0
    acceptable: bool = True
    warning: str = ""


class SkipDecodeResponse(BaseModel):
    skip_config: SkipConfig
    decisions: list[SkipDecision]
    layers_skipped: int
    total_layers: int
    estimated_speedup: float
    batching_compatibility: BatchingCompatibility
    quality_validation: ValidationResult
    summary: str
    references: list[str]


# ── Aggregator ────────────────────────────────────────────────


class ReferenceInfo(BaseModel):
    citation_key: str
    title: str
    authors: str
    venue: str
    year: int
    url: str
    concepts_applied: list[str]


class TechnicalBasisHealth(BaseModel):
    status: str
    components: list[str]
    references_loaded: int


class BibliographyResponse(BaseModel):
    references: list[ReferenceInfo]
    total: int
