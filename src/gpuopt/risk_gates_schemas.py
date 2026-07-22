from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class GateResult(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class GateAction(StrEnum):
    ACCEPT = "accept"
    MITIGATE = "mitigate"
    REJECT = "reject"


# ── Risk Gate 1: Memory vs Runtime ───────────────────────────


class MemoryBenchmarkConfig(BaseModel):
    model_name: str = "llama-8b"
    dtype: str = "fp16"
    max_seq_len: int = 8192
    batch_size: int = 4
    gpu_memory_gib: float = 80.0
    estimated_weight_gb: float = 0.0
    estimated_kv_cache_gb: float = 0.0
    estimated_total_gb: float = 0.0


class MemoryGateInput(BaseModel):
    benchmark_config: MemoryBenchmarkConfig
    measured_peak_memory_gib: float = 0.0
    measured_oom_occurred: bool = False


class MemoryGateResult(BaseModel):
    risk: Literal["Memory estimate differs from runtime"]
    control: str = "Benchmark longest context; retain 10-15% memory headroom; zero OOM"
    estimated_memory_gib: float
    measured_peak_memory_gib: float
    headroom_pct: float
    headroom_sufficient: bool
    oom_occurred: bool
    result: GateResult
    action: GateAction
    details: list[str] = Field(default_factory=list)


# ── Risk Gate 2: Mock vs Real Performance ────────────────────


class MockDataGateInput(BaseModel):
    data_source: str = ""
    is_mock_data: bool = True
    has_dcgm_metrics: bool = False
    has_real_endpoint: bool = False


class MockDataGateResult(BaseModel):
    risk: Literal["Mock results mistaken for real performance"]
    control: str = "Label all synthetic data; production sizing requires real endpoint and DCGM"
    data_source: str
    is_mock_data: bool
    has_real_endpoint: bool
    has_dcgm_metrics: bool
    labels_present: bool
    result: GateResult
    action: GateAction
    details: list[str] = Field(default_factory=list)


# ── Risk Gate 3: TP Communication Bound ──────────────────────


class TpBenchmarkConfig(BaseModel):
    num_gpus: int = 4
    tp_size: int = 4
    cross_node_tp: bool = False
    interconnect_bandwidth_gb_per_sec: float = 600.0
    all_reduce_time_us: float = 0.0
    compute_time_us: float = 0.0


class TpGateInput(BaseModel):
    config: TpBenchmarkConfig


class TpGateResult(BaseModel):
    risk: Literal["Tensor parallelism becomes communication-bound"]
    control: str = "Benchmark interconnect; compare TP sizes; avoid unnecessary cross-node TP"
    tp_size: int
    cross_node: bool
    communication_ratio: float
    is_communication_bound: bool
    recommended_tp_size: int
    avoid_cross_node: bool
    result: GateResult
    action: GateAction
    details: list[str] = Field(default_factory=list)


# ── Risk Gate 4: Autoscaling Oscillation ─────────────────────


class AutoscaleConfig(BaseModel):
    cooldown_seconds: int = 300
    hysteresis_pct: float = 10.0
    min_dwell_seconds: int = 600
    scale_up_threshold: float = 70.0
    scale_down_threshold: float = 30.0


class AutoscaleGateInput(BaseModel):
    config: AutoscaleConfig
    observed_oscillations: int = 0
    current_replicas: int = 3
    min_replicas: int = 2
    max_replicas: int = 10


class AutoscaleGateResult(BaseModel):
    risk: Literal["Autoscaling causes oscillation"]
    control: str = "Use cooldown, hysteresis, minimum dwell time, and rollback"
    cooldown_seconds: int
    hysteresis_pct: float
    min_dwell_seconds: int
    oscillation_risk: Literal["low", "medium", "high"]
    has_rollback: bool
    result: GateResult
    action: GateAction
    details: list[str] = Field(default_factory=list)


# ── Risk Gate 5: Optimization Quality Damage ─────────────────


class QualityTestConfig(BaseModel):
    optimization_type: Literal["quantization", "adaptive_compute", "pruning", "distillation"]
    original_perplexity: float = 0.0
    optimized_perplexity: float = 0.0
    original_accuracy: float = 0.0
    optimized_accuracy: float = 0.0
    max_perplexity_degradation: float = 0.5
    max_accuracy_degradation_pct: float = 1.0


class QualityGateInput(BaseModel):
    config: QualityTestConfig


class QualityGateResult(BaseModel):
    risk: Literal["Optimization damages quality"]
    control: str = "Run model-quality regression tests for quantization or adaptive compute"
    optimization_type: str
    perplexity_delta: float
    accuracy_delta_pct: float
    perplexity_acceptable: bool
    accuracy_acceptable: bool
    all_tests_passed: bool
    result: GateResult
    action: GateAction
    details: list[str] = Field(default_factory=list)


# ── Risk Gate 6: Unsafe K8s Mutation ─────────────────────────


class K8sMutationGateInput(BaseModel):
    mutation_type: str = ""
    is_read_only: bool = True
    requires_approval: bool = True
    has_gitops: bool = True


class K8sMutationGateResult(BaseModel):
    risk: Literal["Unsafe Kubernetes mutation"]
    control: str = "Read-only GPUOpt backend; reviewed GitOps applies changes"
    is_read_only: bool
    requires_approval: bool
    has_gitops: bool
    mutation_safe: bool
    result: GateResult
    action: GateAction
    details: list[str] = Field(default_factory=list)


# ── Risk Gate 7: Secrets Exposed ─────────────────────────────


class SecretScanInput(BaseModel):
    uses_secret_references: bool = True
    has_bearer_tokens_in_logs: bool = False
    has_bearer_tokens_in_records: bool = False
    has_api_keys_in_benchmark: bool = False


class SecretScanResult(BaseModel):
    risk: Literal["Secrets exposed in API or logs"]
    control: str = "Use secret references; never persist bearer tokens in benchmark records"
    uses_secret_references: bool
    has_bearer_tokens_in_logs: bool
    has_bearer_tokens_in_records: bool
    has_api_keys_in_benchmark: bool
    secrets_safe: bool
    result: GateResult
    action: GateAction
    details: list[str] = Field(default_factory=list)


# ── Risk Gate 8: MoE Expert Imbalance ────────────────────────


class MoeExpertBalanceInput(BaseModel):
    num_experts: int = 8
    per_gpu_loads: list[float] = Field(default_factory=list)
    top_k: int = 2
    expert_capacity_factor: float = 1.25


class MoeGateResult(BaseModel):
    risk: Literal["MoE expert imbalance"]
    control: str = "Measure per-GPU load; test expert placement and activated-expert balancing"
    num_experts: int
    load_imbalance_ratio: float
    is_balanced: bool
    recommended_capacity_factor: float
    expert_placement_ok: bool
    result: GateResult
    action: GateAction
    details: list[str] = Field(default_factory=list)


# ── Aggregated ───────────────────────────────────────────────


class RiskGateSummary(BaseModel):
    gate_id: str
    risk: str
    result: GateResult
    action: GateAction


class RiskGatesDashboard(BaseModel):
    gates: dict[str, RiskGateSummary]
    all_passed: bool
    can_deploy: bool
    summary: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
