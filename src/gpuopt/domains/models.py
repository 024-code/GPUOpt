from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MetricType(StrEnum):
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class DomainMetadata(BaseModel):
    domain: str
    metric_type: MetricType
    unit: str
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "mock"
    node_id: str | None = None
    cluster_id: str | None = None
    job_id: str | None = None
    tenant_id: str | None = None


# ── 1. GPU & Node ──────────────────────────────────────────────

class GpuNodeMetric(BaseModel):
    metadata: DomainMetadata = Field(default_factory=lambda: DomainMetadata(domain="gpu_node", metric_type=MetricType.GAUGE, unit=""))
    gpu_index: int = 0
    gpu_uuid: str = ""
    gpu_model: str = ""
    utilization_gpu_pct: float = 0.0
    utilization_memory_pct: float = 0.0
    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    power_watts: float = 0.0
    temperature_gpu_c: float = 0.0
    temperature_memory_c: float = 0.0
    clock_sm_mhz: float = 0.0
    clock_mem_mhz: float = 0.0
    clock_graphics_mhz: float = 0.0
    pcie_gen: int = 0
    pcie_link_width: int = 0
    ecc_errors_corrected: int = 0
    ecc_errors_uncorrected: int = 0
    health: str = "healthy"
    mig_enabled: bool = False
    mig_gi_id: int | None = None
    topology: str = ""


class GpuNodeTelemetry(BaseModel):
    cluster_id: str
    node_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    gpus: list[GpuNodeMetric] = []
    cpu_utilization_pct: float = 0.0
    cpu_memory_total_mb: float = 0.0
    cpu_memory_used_mb: float = 0.0
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    nvidia_smi_errors: list[str] = []


# ── 2. Fabric & Storage ────────────────────────────────────────

class NcclEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    collective_type: str = ""  # allreduce, allgather, etc.
    message_size_bytes: int = 0
    duration_us: float = 0.0
    bus_bw_gbps: float = 0.0
    algo_name: str = ""
    n_ranks: int = 0
    rank: int = 0
    error: str | None = None


class NetworkMetric(BaseModel):
    metadata: DomainMetadata = Field(default_factory=lambda: DomainMetadata(domain="fabric_storage", metric_type=MetricType.GAUGE, unit=""))
    interface: str = ""
    throughput_rx_mbps: float = 0.0
    throughput_tx_mbps: float = 0.0
    latency_us: float = 0.0
    packet_loss_pct: float = 0.0
    retransmit_count: int = 0


class StorageMetric(BaseModel):
    metadata: DomainMetadata = Field(default_factory=lambda: DomainMetadata(domain="fabric_storage", metric_type=MetricType.GAUGE, unit=""))
    filesystem: str = ""
    mount_point: str = ""
    read_iops: float = 0.0
    write_iops: float = 0.0
    read_throughput_mbps: float = 0.0
    write_throughput_mbps: float = 0.0
    read_latency_us: float = 0.0
    write_latency_us: float = 0.0
    capacity_total_gb: float = 0.0
    capacity_used_gb: float = 0.0


class FabricStorageTelemetry(BaseModel):
    cluster_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    nccl_events: list[NcclEvent] = []
    network: list[NetworkMetric] = []
    storage: list[StorageMetric] = []


# ── 3. Scheduler & Jobs ────────────────────────────────────────

class SchedulerJobEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    cluster_id: str
    job_id: str
    event_type: str  # submitted, queued, started, completed, failed, cancelled, checkpointed, retried
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    queue: str = ""
    priority: int = 0
    requested_gpus: int = 0
    requested_cpus: int = 0
    requested_memory_mb: int = 0
    requested_walltime_minutes: int = 0
    user: str = ""
    project: str = ""
    partition: str = ""
    nodes_allocated: list[str] = []
    exit_code: int | None = None
    detail: str = ""


class SchedulerState(BaseModel):
    cluster_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    queue_depth: int = 0
    running_jobs: int = 0
    pending_jobs: int = 0
    blocked_jobs: int = 0
    suspended_jobs: int = 0
    total_slots: int = 0
    used_slots: int = 0
    avg_wait_time_seconds: float = 0.0
    avg_run_time_seconds: float = 0.0
    backfill_depth: int = 0
    preemptions: int = 0
    fairshare: dict[str, float] = {}


# ── 4. Training Runtime ────────────────────────────────────────

class TrainingStepMetric(BaseModel):
    metadata: DomainMetadata = Field(default_factory=lambda: DomainMetadata(domain="training", metric_type=MetricType.GAUGE, unit=""))
    job_id: str
    step: int = 0
    epoch: int = 0
    step_time_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    loss: float | None = None
    learning_rate: float | None = None
    gradient_norm: float | None = None
    scale_factor: float | None = None
    global_batch_size: int = 0
    micro_batch_size: int = 0
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    gpu_memory_allocated_gb: float = 0.0
    gpu_memory_reserved_gb: float = 0.0


class TrainingRunSummary(BaseModel):
    job_id: str
    run_id: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None
    total_steps: int = 0
    total_epochs: int = 0
    avg_step_time_ms: float = 0.0
    avg_throughput_samples_per_sec: float = 0.0
    avg_throughput_tokens_per_sec: float = 0.0
    best_loss: float | None = None
    scale_efficiency: float = 0.0  # achieved / ideal speedup
    checkpoint_count: int = 0
    checkpoint_total_size_gb: float = 0.0
    checkpoint_total_time_seconds: float = 0.0
    total_gpu_hours: float = 0.0
    total_cost_usd: float = 0.0
    status: str = ""

    @property
    def checkpoint_overhead_pct(self) -> float:
        if self.checkpoint_total_time_seconds > 0 and self.end_time and self.start_time:
            total = (self.end_time - self.start_time).total_seconds()
            return round(self.checkpoint_total_time_seconds / total * 100, 2) if total > 0 else 0.0
        return 0.0


# ── 5. Inference Runtime ───────────────────────────────────────

class InferenceRequestSample(BaseModel):
    metadata: DomainMetadata = Field(default_factory=lambda: DomainMetadata(domain="inference", metric_type=MetricType.HISTOGRAM, unit=""))
    model_id: str
    deployment_id: str = ""
    request_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_ms: float = 0.0    # time to first token
    tpot_ms: float = 0.0    # time per output token
    total_latency_ms: float = 0.0
    batch_size: int = 1
    kv_cache_usage_pct: float = 0.0
    kv_cache_size_tokens: int = 0
    peak_memory_gb: float = 0.0
    status_code: int = 200
    error: str | None = None
    model_dtype: str = "float16"
    quantization: str = "none"


class InferenceSummary(BaseModel):
    model_id: str
    deployment_id: str = ""
    period_start: datetime | None = None
    period_end: datetime | None = None
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_generated_tokens: int = 0
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    p50_tpot_ms: float = 0.0
    p95_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    avg_batch_size: float = 1.0
    max_batch_size: int = 1
    avg_kv_cache_pct: float = 0.0
    error_rate_pct: float = 0.0
    total_errors: int = 0
    avg_peak_memory_gb: float = 0.0


# ── 6. Tenant & Cost ───────────────────────────────────────────

class TenantQuota(BaseModel):
    tenant_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    gpu_quota: int = 0
    gpu_allocated: int = 0
    gpu_utilization_pct: float = 0.0
    cpu_quota_cores: int = 0
    cpu_allocated_cores: int = 0
    memory_quota_gb: float = 0.0
    memory_allocated_gb: float = 0.0
    priority: int = 0
    fairshare: float = 1.0
    preemptible_jobs: int = 0


class CostAllocation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    period: str = "daily"
    gpu_hours: float = 0.0
    gpu_rate_usd_per_hour: float = 0.0
    gpu_cost_usd: float = 0.0
    storage_gb_hours: float = 0.0
    storage_rate_usd_per_gb_hour: float = 0.0
    storage_cost_usd: float = 0.0
    network_gb: float = 0.0
    network_rate_usd_per_gb: float = 0.0
    network_cost_usd: float = 0.0
    power_kwh: float = 0.0
    power_rate_usd_per_kwh: float = 0.0
    power_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    budget_usd: float = 0.0
    budget_remaining_usd: float = 0.0
    chargeback_code: str = ""
    labels: dict[str, str] = {}


# ── 7. Actions & Outcomes (event-sourced, immutable) ──────────

class ActionType(StrEnum):
    RECOMMENDATION = "recommendation"
    APPROVAL = "approval"
    EXECUTION = "execution"
    ROLLBACK = "rollback"
    VERIFICATION = "verification"


class ActionSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ActionStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ActionEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action_type: ActionType
    status: ActionStatus
    severity: ActionSeverity = ActionSeverity.INFO

    cluster_id: str = ""
    target_resource: str = ""   # node, job, deployment, etc.
    target_id: str = ""          # specific resource identifier
    action_name: str = ""
    action_params: dict[str, Any] = {}

    risk_score: float = 0.0    # 0-1
    expected_impact: str = ""
    actual_impact: str | None = None

    triggered_by: str = "system"
    approval_required: bool = False
    approved_by: str | None = None
    approved_at: datetime | None = None

    parent_event_id: UUID | None = None  # link to recommendation → approval → execution chain

    detail: str = ""


class ActionOutcome(BaseModel):
    event_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: ActionStatus
    realized_effect: dict[str, Any] = {}
    metrics_before: dict[str, float] = {}
    metrics_after: dict[str, float] = {}
    improvement_pct: float | None = None
    rollback_reason: str | None = None
    verification_result: str = ""
    detail: str = ""
