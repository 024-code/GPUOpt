from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConnectorType(StrEnum):
    MOCK = "mock"
    KUBERNETES = "kubernetes"
    SLURM = "slurm"


class CheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class ClusterCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    environment: str = Field(default="development", max_length=40)
    connector_type: ConnectorType
    description: str | None = Field(default=None, max_length=500)
    kube_context: str | None = None
    kubeconfig_path: str | None = None
    in_cluster: bool = False
    credential_ref: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kubeconfig_path")
    @classmethod
    def reject_inline_secret_data(cls, value: str | None) -> str | None:
        if value and "apiVersion:" in value:
            raise ValueError("Provide a kubeconfig path or secret reference, not inline kubeconfig data")
        return value


class ClusterRecord(ClusterCreate):
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CheckItem(BaseModel):
    name: str
    status: CheckStatus
    message: str
    latency_ms: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    remediation: str | None = None


class EnvironmentCheckReport(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    cluster_id: UUID
    cluster_name: str
    environment: str
    started_at: datetime
    completed_at: datetime
    overall_status: CheckStatus
    checks: list[CheckItem]
    summary: dict[str, int]


class EnvironmentSummary(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    clusters: int
    healthy: int
    warning: int
    failing: int
    unchecked: int
    environments: dict[str, dict[str, int]]


# ── R0.2: Telemetry Normalization & Cluster State ─────────────

class GPUDeviceTelemetry(BaseModel):
    """Normalized telemetry for a single GPU device."""
    index: int
    uuid: str = ""
    model: str = ""
    memory_total_bytes: int = 0
    memory_used_bytes: int = 0
    utilization_gpu_percent: float = 0.0
    utilization_memory_percent: float = 0.0
    temperature_gpu_celsius: float = 0.0
    power_draw_watts: float = 0.0
    power_limit_watts: float = 0.0
    ecc_errors_volatile: int = 0
    ecc_errors_aggregate: int = 0
    clock_sm_mhz: int = 0
    clock_mem_mhz: int = 0


class NodeTelemetry(BaseModel):
    """Normalized telemetry for a single cluster node."""
    node_name: str
    status: str = "Unknown"
    cpu_usage_millicores: int = 0
    cpu_capacity_millicores: int = 0
    memory_usage_bytes: int = 0
    memory_capacity_bytes: int = 0
    pod_count: int = 0
    pod_capacity: int = 0
    gpu_devices: list[GPUDeviceTelemetry] = Field(default_factory=list)


class ClusterTelemetry(BaseModel):
    """A cluster-wide telemetry snapshot with per-node and per-GPU metrics."""
    cluster_id: UUID
    cluster_name: str
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node_count: int = 0
    gpu_count: int = 0
    nodes: list[NodeTelemetry] = Field(default_factory=list)
    freshness_seconds: float = 0.0


class GPUDeviceState(BaseModel):
    """GPU device in the cluster state."""
    index: int
    uuid: str = ""
    model: str = ""
    memory_total_bytes: int = 0
    memory_used_bytes: int = 0
    status: str = "unknown"


class NodeState(BaseModel):
    """A node in the cluster state with capacity, labels, and its GPUs."""
    name: str
    status: str = "Unknown"
    capacity: dict[str, Any] = Field(default_factory=dict)
    allocatable: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    gpu_devices: list[GPUDeviceState] = Field(default_factory=list)
    pod_count: int = 0
    pod_capacity: int = 0
    created_at: datetime | None = None


class ClusterStateData(BaseModel):
    """Full cluster state snapshot with telemetry."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node_count: int = 0
    gpu_count: int = 0
    total_gpu_memory_bytes: int = 0
    nodes: list[NodeState] = Field(default_factory=list)
    telemetry: ClusterTelemetry | None = None

    @property
    def freshness_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.collected_at).total_seconds()


class ClusterStateSummary(BaseModel):
    """Condensed cluster state with freshness and health indicators."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    last_collected_at: datetime | None = None
    age_seconds: float = 0.0
    node_count: int = 0
    gpu_count: int = 0
    healthy_nodes: int = 0
    total_gpu_memory_bytes: int = 0
    status: str = "unknown"


# ── R0.3: Trace Replay & Baseline Simulation ────────────────

class TraceListItem(BaseModel):
    """A summary entry in the trace history list."""
    id: str
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    collected_at: datetime
    node_count: int = 0
    gpu_count: int = 0
    has_baseline: bool = False


class TraceReplayResult(BaseModel):
    """Result of replaying a historical trace through the check system."""
    trace_id: str
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    replayed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    original_collected_at: datetime
    node_count: int = 0
    gpu_count: int = 0
    checks: list[CheckItem] = Field(default_factory=list)
    overall_status: CheckStatus = CheckStatus.PASS
    summary: dict[str, int] = Field(default_factory=dict)


class GPUDeviceDiff(BaseModel):
    """Per-GPU comparison between two state snapshots."""
    node: str
    gpu_index: int
    gpu_uuid: str = ""
    gpu_model: str = ""
    baseline_memory_used_bytes: int = 0
    current_memory_used_bytes: int = 0
    baseline_utilization_percent: float = 0.0
    current_utilization_percent: float = 0.0
    baseline_temperature_celsius: float = 0.0
    current_temperature_celsius: float = 0.0
    drift_score: float = 0.0


class StateComparison(BaseModel):
    """Full comparison between two cluster state snapshots."""
    baseline_id: str
    current_id: str
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    compared_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    baseline_collected_at: datetime
    current_collected_at: datetime
    elapsed_hours: float = 0.0
    node_count_baseline: int = 0
    node_count_current: int = 0
    nodes_added: list[str] = Field(default_factory=list)
    nodes_removed: list[str] = Field(default_factory=list)
    gpu_count_baseline: int = 0
    gpu_count_current: int = 0
    gpu_diffs: list[GPUDeviceDiff] = Field(default_factory=list)
    avg_gpu_drift_score: float = 0.0
    max_gpu_drift_score: float = 0.0
    summary: str = ""


class BaselineInfo(BaseModel):
    """Information about a set baseline snapshot."""
    cluster_id: UUID
    trace_id: str
    set_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    collected_at: datetime
    node_count: int = 0
    gpu_count: int = 0


# ── R0.4: Workload Analysis & Resource Profiling ────────────

class GPUUtilizationTrend(BaseModel):
    """Trend analysis for a single GPU across traces."""
    gpu_uuid: str
    node: str
    gpu_index: int
    model: str = ""
    memory_total_bytes: int = 0
    avg_utilization_percent: float = 0.0
    peak_utilization_percent: float = 0.0
    min_utilization_percent: float = 0.0
    avg_memory_used_bytes: float = 0.0
    peak_memory_used_bytes: int = 0
    avg_temperature_celsius: float = 0.0
    peak_temperature_celsius: float = 0.0
    idle_percent: float = 0.0
    memory_pressure_percent: float = 0.0
    sample_count: int = 0


class NodeEfficiency(BaseModel):
    """Efficiency profile for a cluster node."""
    node_name: str
    status: str = "Unknown"
    gpu_count: int = 0
    avg_gpu_utilization_percent: float = 0.0
    gpu_idle_percent: float = 0.0
    avg_memory_utilization_percent: float = 0.0
    pod_count_avg: float = 0.0
    pod_capacity: int = 0
    efficiency_score: float = 0.0
    recommendations: list[str] = Field(default_factory=list)


class WorkloadAnalysisResult(BaseModel):
    """Full workload analysis for a cluster."""
    id: UUID = Field(default_factory=uuid4)
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timeframe_hours: float = 0.0
    trace_count: int = 0
    node_count: int = 0
    gpu_count: int = 0
    total_gpu_hours: float = 0.0
    gpu_trends: list[GPUUtilizationTrend] = Field(default_factory=list)
    node_efficiencies: list[NodeEfficiency] = Field(default_factory=list)
    overall_efficiency_score: float = 0.0
    total_idle_gpu_hours: float = 0.0
    estimated_power_waste_kwh: float = 0.0
    summary: str = ""


class AnalysisSummary(BaseModel):
    """Summary of the latest analysis for quick display."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    generated_at: datetime | None = None
    trace_count: int = 0
    gpu_count: int = 0
    overall_efficiency_score: float = 0.0
    total_idle_gpu_hours: float = 0.0
    estimated_power_waste_kwh: float = 0.0


# ── R0.5: Recommendation MVP ────────────────────────────────

class RecommendationType(StrEnum):
    PLACEMENT = "placement"
    RIGHT_SIZING = "right_sizing"
    SCALING = "scaling"
    RISK_MITIGATION = "risk_mitigation"
    EFFICIENCY = "efficiency"


class RecommendationSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ── R0.6: Recommendation Scoring & Lifecycle ────────────────

class RecommendationStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DISMISSED = "dismissed"
    IMPLEMENTED = "implemented"


class ResourceRecommendation(BaseModel):
    """A single actionable recommendation with explanation."""
    id: UUID = Field(default_factory=uuid4)
    type: RecommendationType
    severity: RecommendationSeverity
    title: str
    description: str
    reasoning: str
    expected_impact: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: str = "low"
    affected_resources: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    estimated_savings: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    score: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall priority score 0-100")
    status: RecommendationStatus = RecommendationStatus.PENDING


class RecommendationSet(BaseModel):
    """A scored and ranked set of recommendations for a cluster."""
    id: UUID = Field(default_factory=uuid4)
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    based_on_state_at: datetime | None = None
    based_on_analysis_at: datetime | None = None
    recommendation_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    recommendations: list[ResourceRecommendation] = Field(default_factory=list)
    summary: str = ""
    avg_score: float = Field(default=0.0, description="Average priority score across all recs")
    total_estimated_savings_gpu_hours: float = 0.0
    top_recommendation: str = ""


class WhatIfProjection(BaseModel):
    """Projected cluster state after applying a set of recommendations."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    recommendation_set_id: UUID
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    projected_gpu_utilization_percent: float = 0.0
    projected_efficiency_score: float = 0.0
    projected_idle_gpu_hours_reduction: float = 0.0
    projected_power_savings_kwh: float = 0.0
    estimated_cost_savings_usd: float = 0.0
    fragmentation_improvement_percent: float = 0.0
    reservations_freed: int = 0
    risk_reduction_score: float = 0.0
    summary: str = ""


class StatusUpdate(BaseModel):
    """Input for updating a recommendation's lifecycle status."""
    status: RecommendationStatus
    reason: str = ""


# ── R0.7: Digital Twin Service ──────────────────────────────

class DriftSeverity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftItem(BaseModel):
    """A single drift observation between twin and actual state."""
    resource: str
    property: str
    twin_value: str = ""
    actual_value: str = ""
    severity: DriftSeverity = DriftSeverity.LOW
    message: str = ""


class TwinComparison(BaseModel):
    """Comparison result between a digital twin and actual cluster state."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    compared_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    twin_synced_at: datetime | None = None
    actual_collected_at: datetime | None = None
    drift_count: int = 0
    critical_drift_count: int = 0
    high_drift_count: int = 0
    medium_drift_count: int = 0
    overall_drift_severity: DriftSeverity = DriftSeverity.NONE
    drifts: list[DriftItem] = Field(default_factory=list)
    summary: str = ""


class TwinState(BaseModel):
    """A digital twin snapshot mirroring or diverging from the actual cluster."""
    id: UUID = Field(default_factory=uuid4)
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    synced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    original_collected_at: datetime | None = None
    node_count: int = 0
    gpu_count: int = 0
    state_json: str = ""
    has_diverged: bool = False
    divergence_reason: str = ""


# ── R0.8: Predictive Scheduling ─────────────────────────────

class NodeResource(BaseModel):
    """Available resources on a node at a point in time."""
    node_name: str
    free_gpu_count: int = 0
    free_gpu_memory_bytes: int = 0
    free_cpu_millicores: int = 0
    free_memory_bytes: int = 0
    gpu_models: list[str] = Field(default_factory=list)
    current_pod_count: int = 0
    pod_capacity: int = 0


class DemandForecastPoint(BaseModel):
    """A single forecast point in time."""
    timestamp: datetime
    predicted_gpu_utilization_percent: float = 0.0
    predicted_gpu_memory_used_bytes: float = 0.0
    predicted_pod_count: float = 0.0
    confidence_lower: float = 0.0
    confidence_upper: float = 0.0


class DemandForecast(BaseModel):
    """Predicted future resource demand for a cluster."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    horizon_hours: int = 0
    trace_count: int = 0
    predicted_idle_gpus: float = 0.0
    predicted_peak_gpu_memory_bytes: float = 0.0
    predicted_avg_utilization_percent: float = 0.0
    forecast_points: list[DemandForecastPoint] = Field(default_factory=list)
    summary: str = ""


class WorkloadRequirements(BaseModel):
    """Resource requirements for a workload to be scheduled."""
    gpu_count: int = 1
    gpu_memory_bytes: int = 0
    cpu_millicores: int = 1000
    memory_bytes: int = 0
    gpu_model_preference: str = ""
    allow_fractional_gpu: bool = False


class PlacementSuggestion(BaseModel):
    """Suggested placement for a workload on the cluster."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    workload: WorkloadRequirements
    suggested_node: str = ""
    alternative_nodes: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    projected_impact: str = ""
    estimated_fragmentation_after: float = 0.0
    score: float = 0.0


class ScheduleSimulation(BaseModel):
    """Result of simulating a scheduling decision."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    placement: PlacementSuggestion
    projected_utilization_delta: float = 0.0
    projected_memory_fragmentation: float = 0.0
    projected_pod_density: float = 0.0
    efficiency_gain: float = 0.0
    risk_score: float = 0.0
    summary: str = ""


class SchedulingPlan(BaseModel):
    """A comprehensive scheduling plan for a cluster."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_gpus: int = 0
    free_gpus: int = 0
    node_count: int = 0
    avg_gpu_utilization: float = 0.0
    suggested_consolidations: int = 0
    suggested_placements: list[str] = Field(default_factory=list)
    recommended_node_counts: int = 0
    estimated_savings_gpu_hours: float = 0.0
    summary: str = ""


# ── R0.9: Actuation Pipeline ──────────────────────────────

class ActuationStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ActuationRequest(BaseModel):
    """Request to actuate (apply) a recommendation."""
    rec_id: UUID
    dry_run: bool = False
    reason: str = ""


class ActuationAction(BaseModel):
    """A single action performed during actuation."""
    action_type: str = ""
    target: str = ""
    value: str = ""
    status: str = ""
    message: str = ""


class ActuationRecord(BaseModel):
    """Full audit record of an actuation attempt."""
    id: UUID = Field(default_factory=uuid4)
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    rec_id: UUID
    rec_title: str = ""
    rec_type: str = ""
    status: ActuationStatus = ActuationStatus.PENDING
    dry_run: bool = False
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    actions: list[ActuationAction] = Field(default_factory=list)
    result_summary: str = ""
    error_message: str = ""
    rollback_of: str = ""
    rolled_back_by: str = ""


class ActuationSummary(BaseModel):
    """Aggregated actuation statistics for a cluster."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    total_actuations: int = 0
    successful: int = 0
    failed: int = 0
    in_progress: int = 0
    rolled_back: int = 0
    pending: int = 0
    latest_actuation: ActuationRecord | None = None


# ── R0.10: Cost Analysis ─────────────────────────────────

class GpuCostBreakdown(BaseModel):
    """Per-GPU cost breakdown."""
    gpu_index: int = 0
    gpu_model: str = ""
    memory_utilization_percent: float = 0.0
    estimated_hourly_cost: float = 0.0
    estimated_monthly_cost: float = 0.0
    estimated_waste_hourly: float = 0.0
    utilization_category: str = ""


class NodeCostBreakdown(BaseModel):
    """Per-node cost breakdown."""
    node_name: str = ""
    gpu_count: int = 0
    gpu_models: list[str] = Field(default_factory=list)
    total_hourly_cost: float = 0.0
    total_monthly_cost: float = 0.0
    waste_hourly_cost: float = 0.0
    gpus: list[GpuCostBreakdown] = Field(default_factory=list)


class CostReport(BaseModel):
    """Current cost report for a cluster."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    gpu_hourly_rate: float = 0.0
    total_gpus: int = 0
    active_gpus: int = 0
    idle_gpus: int = 0
    total_hourly_cost: float = 0.0
    total_daily_cost: float = 0.0
    total_monthly_cost: float = 0.0
    waste_hourly_cost: float = 0.0
    waste_daily_cost: float = 0.0
    waste_monthly_cost: float = 0.0
    efficiency_percent: float = 0.0
    nodes: list[NodeCostBreakdown] = Field(default_factory=list)
    summary: str = ""


class SavingsProjection(BaseModel):
    """Projected savings from optimization recommendations."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    current_monthly_cost: float = 0.0
    projected_monthly_cost: float = 0.0
    monthly_savings: float = 0.0
    annual_savings: float = 0.0
    savings_percent: float = 0.0
    recommendation_count: int = 0
    top_savings_recs: list[str] = Field(default_factory=list)
    summary: str = ""


class CostSummary(BaseModel):
    """Aggregated cost summary for a cluster."""
    cluster_id: UUID
    cluster_name: str
    environment: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_gpus: int = 0
    utilization_rate: float = 0.0
    monthly_cost: float = 0.0
    monthly_waste: float = 0.0
    potential_monthly_savings: float = 0.0
    payback_period_days: float = 0.0
    cost_health: str = ""
    summary: str = ""


# ── S20: FinOps ─────────────────────────────────────────────


class CloudProvider(StrEnum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    EQUINIX = "equinix"
    CUSTOM = "custom"


class GpuPricingTier(StrEnum):
    ONDEMAND = "ondemand"
    SPOT = "spot"
    RESERVED_1YR = "reserved_1yr"
    RESERVED_3YR = "reserved_3yr"


class GpuPricingRow(BaseModel):
    """GPU pricing for a specific model/provider/region/tier."""
    gpu_model: str
    provider: CloudProvider = CloudProvider.CUSTOM
    region: str = "us-east-1"
    tier: GpuPricingTier = GpuPricingTier.ONDEMAND
    hourly_cost: float = 0.0
    monthly_cost: float = 0.0
    spot_savings_percent: float = 0.0
    reserved_1yr_savings_percent: float = 0.0
    reserved_3yr_savings_percent: float = 0.0
    gpu_count_per_instance: int = 1
    instance_type: str = ""
    vcpu_count: int = 0
    system_memory_gb: int = 0


class ProviderCostComparison(BaseModel):
    """Compare GPU costs across providers for a given configuration."""
    gpu_model: str
    gpu_count: int
    providers: list[GpuPricingRow] = Field(default_factory=list)
    cheapest_ondemand: GpuPricingRow | None = None
    cheapest_overall: GpuPricingRow | None = None
    max_potential_savings_percent: float = 0.0
    recommendation: str = ""


class SpotSavingsAnalysis(BaseModel):
    """Analysis of spot/preemptible GPU savings potential."""
    cluster_id: UUID
    cluster_name: str
    total_gpus: int = 0
    ondemand_monthly_cost: float = 0.0
    spot_monthly_cost: float = 0.0
    monthly_savings: float = 0.0
    annual_savings: float = 0.0
    savings_percent: float = 0.0
    spot_viable_gpus: int = 0
    interruption_risk: str = "low"
    recommendations: list[str] = Field(default_factory=list)


class ReservedInstanceRecommendation(BaseModel):
    """Recommendation for reserved/committed use instances."""
    cluster_id: UUID
    cluster_name: str
    current_monthly_cost: float = 0.0
    reserved_1yr_monthly_cost: float = 0.0
    reserved_3yr_monthly_cost: float = 0.0
    monthly_savings_1yr: float = 0.0
    monthly_savings_3yr: float = 0.0
    annual_savings_1yr: float = 0.0
    annual_savings_3yr: float = 0.0
    recommended_term: str = "1yr"
    break_even_months: float = 0.0
    recommendations: list[str] = Field(default_factory=list)


class BudgetAlert(BaseModel):
    """Budget tracking alert."""
    cluster_id: UUID
    cluster_name: str = ""
    monthly_budget: float = 0.0
    current_monthly_spend: float = 0.0
    budget_utilization_percent: float = 0.0
    projected_month_end_spend: float = 0.0
    status: str = "on_track"
    alerts: list[str] = Field(default_factory=list)


class CostAllocationTag(BaseModel):
    """Cost allocation tag for tracking GPU spending."""
    key: str
    value: str
    monthly_cost: float = 0.0
    gpu_count: int = 0
    percentage: float = 0.0


class MultiClusterCostSummary(BaseModel):
    """Aggregated cost summary across multiple clusters."""
    cluster_count: int = 0
    total_gpus: int = 0
    total_monthly_cost: float = 0.0
    total_monthly_waste: float = 0.0
    average_utilization: float = 0.0
    total_potential_monthly_savings: float = 0.0
    total_annual_savings: float = 0.0
    clusters: list[dict[str, Any]] = Field(default_factory=list)
    top_recommendations: list[str] = Field(default_factory=list)


class CostForecastPoint(BaseModel):
    """A single point in a cost forecast."""
    month: str
    projected_cost: float = 0.0
    optimistic_cost: float = 0.0
    pessimistic_cost: float = 0.0
    confidence_upper: float = 0.0
    confidence_lower: float = 0.0


class CostForecast(BaseModel):
    """Cost forecast for a cluster over time."""
    cluster_id: UUID
    cluster_name: str = ""
    current_monthly_cost: float = 0.0
    forecast: list[CostForecastPoint] = Field(default_factory=list)
    projected_annual_cost: float = 0.0
    growth_rate: float = 0.0
    summary: str = ""


class WhatIfCostScenario(BaseModel):
    """A what-if cost scenario for GPU infrastructure changes."""
    scenario_name: str
    description: str = ""
    gpu_count_change: int = 0
    gpu_model_change: str = ""
    provider_change: CloudProvider | None = None
    tier_change: GpuPricingTier | None = None
    utilization_change: float = 0.0
    current_monthly_cost: float = 0.0
    scenario_monthly_cost: float = 0.0
    monthly_difference: float = 0.0
    annual_difference: float = 0.0
    recommendations: list[str] = Field(default_factory=list)


# ── S21: Power Optimization ─────────────────────────────────


class GpuPowerProfile(BaseModel):
    """Power profile for a GPU model."""
    gpu_model: str
    tdp_watts: float = 0.0
    idle_power_watts: float = 0.0
    typical_load_power_watts: float = 0.0
    max_power_watts: float = 0.0
    power_efficiency_tflops_per_watt: float = 0.0
    memory_power_watts: float = 0.0


GPU_POWER_PROFILES: list[GpuPowerProfile] = [
    GpuPowerProfile(gpu_model="h100", tdp_watts=700.0, idle_power_watts=50.0, typical_load_power_watts=450.0, max_power_watts=700.0, power_efficiency_tflops_per_watt=0.29, memory_power_watts=80.0),
    GpuPowerProfile(gpu_model="h200", tdp_watts=700.0, idle_power_watts=50.0, typical_load_power_watts=450.0, max_power_watts=700.0, power_efficiency_tflops_per_watt=0.33, memory_power_watts=80.0),
    GpuPowerProfile(gpu_model="b200", tdp_watts=1000.0, idle_power_watts=60.0, typical_load_power_watts=600.0, max_power_watts=1000.0, power_efficiency_tflops_per_watt=0.45, memory_power_watts=120.0),
    GpuPowerProfile(gpu_model="a100", tdp_watts=400.0, idle_power_watts=40.0, typical_load_power_watts=300.0, max_power_watts=400.0, power_efficiency_tflops_per_watt=0.25, memory_power_watts=60.0),
    GpuPowerProfile(gpu_model="a6000", tdp_watts=300.0, idle_power_watts=30.0, typical_load_power_watts=225.0, max_power_watts=300.0, power_efficiency_tflops_per_watt=0.18, memory_power_watts=40.0),
    GpuPowerProfile(gpu_model="v100", tdp_watts=300.0, idle_power_watts=35.0, typical_load_power_watts=250.0, max_power_watts=300.0, power_efficiency_tflops_per_watt=0.15, memory_power_watts=40.0),
    GpuPowerProfile(gpu_model="t4", tdp_watts=70.0, idle_power_watts=10.0, typical_load_power_watts=60.0, max_power_watts=70.0, power_efficiency_tflops_per_watt=0.14, memory_power_watts=10.0),
    GpuPowerProfile(gpu_model="l40s", tdp_watts=300.0, idle_power_watts=30.0, typical_load_power_watts=250.0, max_power_watts=300.0, power_efficiency_tflops_per_watt=0.32, memory_power_watts=50.0),
    GpuPowerProfile(gpu_model="rtx 4090", tdp_watts=450.0, idle_power_watts=25.0, typical_load_power_watts=350.0, max_power_watts=450.0, power_efficiency_tflops_per_watt=0.19, memory_power_watts=50.0),
    GpuPowerProfile(gpu_model="rtx 6000 ada", tdp_watts=300.0, idle_power_watts=25.0, typical_load_power_watts=250.0, max_power_watts=300.0, power_efficiency_tflops_per_watt=0.28, memory_power_watts=40.0),
]


class PowerAnalysisResult(BaseModel):
    """Power analysis for a cluster."""
    cluster_id: UUID
    cluster_name: str = ""
    total_gpus: int = 0
    total_power_draw_watts: float = 0.0
    total_power_capacity_watts: float = 0.0
    utilization_percent: float = 0.0
    idle_power_watts: float = 0.0
    idle_power_cost_monthly: float = 0.0
    active_power_watts: float = 0.0
    active_power_cost_monthly: float = 0.0
    power_waste_watts: float = 0.0
    power_waste_kwh_daily: float = 0.0
    power_waste_cost_monthly: float = 0.0
    power_efficiency_score: float = 0.0
    recommended_power_cap_watts: float = 0.0
    power_cap_savings_percent: float = 0.0
    estimated_annual_power_cost: float = 0.0
    estimated_annual_carbon_kg: float = 0.0
    recommendations: list[str] = Field(default_factory=list)


class CarbonEmissionsEstimate(BaseModel):
    """Carbon emissions estimate for GPU workloads."""
    cluster_id: UUID
    cluster_name: str = ""
    total_energy_kwh: float = 0.0
    grid_carbon_intensity_g_per_kwh: float = 400.0
    carbon_footprint_kg_co2: float = 0.0
    carbon_footprint_tons_co2: float = 0.0
    equivalent_miles_driven: float = 0.0
    equivalent_homes_energy: float = 0.0
    low_carbon_energy_percent: float = 0.0
    recommended_offset_cost_usd: float = 0.0
    recommendations: list[str] = Field(default_factory=list)


class PowerCapSuggestion(BaseModel):
    """Power capping recommendation for GPUs."""
    gpu_model: str = ""
    gpu_count: int = 0
    current_power_watts: float = 0.0
    current_tdp_percent: float = 100.0
    recommended_cap_watts: float = 0.0
    recommended_cap_percent: float = 100.0
    estimated_performance_impact_percent: float = 0.0
    estimated_power_savings_watts: float = 0.0
    estimated_power_savings_monthly_kwh: float = 0.0
    estimated_cost_savings_monthly: float = 0.0
    estimated_temperature_reduction_c: float = 0.0
    risk_level: str = "low"
    recommendations: list[str] = Field(default_factory=list)


class EnergyTimeSeriesPoint(BaseModel):
    """Single point in an energy time series."""
    timestamp: datetime
    power_watts: float = 0.0
    gpu_utilization: float = 0.0
    temperature_c: float = 0.0


class PowerOptimizationRecommendation(BaseModel):
    """A power optimization recommendation for a cluster."""
    cluster_id: UUID
    cluster_name: str = ""
    power_savings_kwh: float = 0.0
    cost_savings_usd: float = 0.0
    carbon_reduction_kg: float = 0.0
    implementation_effort: str = "low"
    priority: str = "medium"
    recommendation: str = ""
    actions: list[str] = Field(default_factory=list)


# ── S19: Inference Optimization ────────────────────────────


class InferenceFramework(StrEnum):
    VLLM = "vllm"
    TGI = "tgi"
    TRITON = "triton"
    TENSORRTLLM = "tensorrt-llm"
    LLAMACPP = "llama.cpp"
    CUSTOM = "custom"


class InferenceEndpointStatus(StrEnum):
    DEPLOYING = "deploying"
    RUNNING = "running"
    SCALING = "scaling"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


class InferenceEndpoint(BaseModel):
    """A model inference endpoint tracked by GPUOpt."""
    id: UUID = Field(default_factory=uuid4)
    cluster_id: UUID
    endpoint_name: str
    model_name: str
    model_version: str = "latest"
    framework: InferenceFramework = InferenceFramework.CUSTOM
    status: InferenceEndpointStatus = InferenceEndpointStatus.DEPLOYING
    gpu_count: int = 1
    gpu_model: str = ""
    quantisation: str = "fp16"
    max_batch_size: int = 1
    max_input_tokens: int = 4096
    max_output_tokens: int = 1024
    concurrency: int = 1
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_requests_per_sec: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    avg_gpu_utilization: float = 0.0
    peak_gpu_memory_gib: float = 0.0
    kv_cache_utilization: float = 0.0
    cost_per_1k_tokens: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceProfile(BaseModel):
    """Profiling analysis for a model inference endpoint."""
    endpoint: InferenceEndpoint
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_mean: float = 0.0
    throughput_peak: float = 0.0
    tokens_per_second_per_gpu: float = 0.0
    batch_efficiency: float = 0.0
    gpu_compute_efficiency: float = 0.0
    gpu_memory_efficiency: float = 0.0
    kv_cache_efficiency: float = 0.0
    kv_cache_peak_gib: float = 0.0
    estimated_optimal_concurrency: int = 1
    estimated_optimal_batch_size: int = 1
    estimated_optimal_gpu_count: int = 1
    recommended_quantisation: str = ""
    recommended_framework: str = ""
    estimated_speedup: float = 1.0
    potential_cost_savings_per_month: float = 0.0
    recommendations: list[str] = Field(default_factory=list)
    summary: str = ""


class InferenceDeploymentConfig(BaseModel):
    """Recommended deployment configuration for a model."""
    endpoint_name: str = ""
    model_name: str = ""
    model_size_gb: float = 0.0
    context_length: int = 4096
    estimated_required_memory_gb: float = 0.0
    recommended_gpu_model: str = ""
    recommended_gpu_count: int = 1
    recommended_node_count: int = 1
    recommended_quantisation: str = "fp16"
    recommended_framework: InferenceFramework = InferenceFramework.VLLM
    recommended_max_batch_size: int = 1
    recommended_concurrency: int = 1
    estimated_throughput_tokens_per_sec: float = 0.0
    estimated_p50_latency_ms: float = 0.0
    estimated_cost_per_1m_tokens_usd: float = 0.0
    estimated_monthly_cost_usd: float = 0.0
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    summary: str = ""


# ── S15-S18: Training Optimization & Slurm Integration ─────


class TrainingFramework(StrEnum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    CUSTOM = "custom"


class TrainingJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(BaseModel):
    """A training job tracked by GPUOpt."""
    id: UUID = Field(default_factory=uuid4)
    cluster_id: UUID
    job_name: str
    framework: TrainingFramework = TrainingFramework.CUSTOM
    status: TrainingJobStatus = TrainingJobStatus.PENDING
    gpu_count: int = 1
    gpu_model: str = ""
    node_count: int = 1
    batch_size: int = 0
    gradient_accumulation_steps: int = 1
    precision: str = "fp32"
    max_duration_hours: float = 0.0
    elapsed_hours: float = 0.0
    avg_gpu_utilization: float = 0.0
    peak_gpu_memory_gib: float = 0.0
    throughput_samples_per_sec: float = 0.0
    loss_value: float | None = None
    epochs_completed: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingProfile(BaseModel):
    """Profiling analysis for a completed or running training job."""
    job: TrainingJob
    gpu_utilization_mean: float = 0.0
    gpu_utilization_peak: float = 0.0
    gpu_utilization_p5: float = 0.0
    gpu_utilization_p95: float = 0.0
    memory_utilization_mean: float = 0.0
    memory_utilization_peak: float = 0.0
    compute_efficiency: float = 0.0
    memory_efficiency: float = 0.0
    io_bottleneck_score: float = 0.0
    communication_overhead: float = 0.0
    estimated_optimal_batch_size: int = 0
    estimated_optimal_gpu_count: int = 0
    recommended_precision: str = ""
    estimated_speedup: float = 1.0
    recommendations: list[str] = Field(default_factory=list)
    summary: str = ""


class HPOConfig(BaseModel):
    """Hyperparameter optimization configuration."""
    batch_sizes: list[int] = Field(default_factory=lambda: [16, 32, 64, 128, 256])
    learning_rates: list[float] = Field(default_factory=lambda: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
    weight_decays: list[float] = Field(default_factory=lambda: [0.0, 1e-4, 1e-3])
    warmup_steps: list[int] = Field(default_factory=lambda: [0, 100, 500])
    max_trials: int = 10
    parallel_trials: int = 2
    optimization_metric: str = "throughput"


class HPOTrial(BaseModel):
    """A single hyperparameter trial."""
    trial_id: int = 0
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    precision: str = "fp32"
    gpu_count: int = 1
    throughput_samples_per_sec: float = 0.0
    peak_memory_gib: float = 0.0
    final_loss: float | None = None
    epochs_completed: int = 0
    status: TrainingJobStatus = TrainingJobStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None


class HTOResult(BaseModel):
    """Results from a hyperparameter optimization run."""
    job_id: UUID
    best_trial: HPOTrial | None = None
    all_trials: list[HPOTrial] = Field(default_factory=list)
    suggested_batch_size: int = 0
    suggested_learning_rate: float = 0.0
    suggested_weight_decay: float = 0.0
    suggested_precision: str = ""
    estimated_improvement: float = 0.0
    summary: str = ""


class GpuTopologyLink(BaseModel):
    """NVLink/NVSwitch connection between two GPUs."""
    source_gpu: int = 0
    target_gpu: int = 0
    link_type: str = "nvlink"
    bandwidth_gb_per_sec: float = 600.0
    nvlink_count: int = 1


class GpuTopology(BaseModel):
    """GPU interconnect topology for a node."""
    node_name: str = ""
    gpu_count: int = 0
    gpu_model: str = ""
    nvswitch_present: bool = False
    nvlink_per_gpu: int = 0
    links: list[GpuTopologyLink] = Field(default_factory=list)
    numa_affinity: list[int] = Field(default_factory=list)


class NodeTopology(BaseModel):
    """Complete topology for all nodes in a cluster."""
    nodes: list[GpuTopology] = Field(default_factory=list)
    cross_node_bandwidth_gb_per_sec: float = 50.0  # InfiniBand/RoCE typical
    has_nvswitch: bool = False
    recommended_dp_group_size: int = 8
    recommended_tp_group_size: int = 8


class DistributedTrainingConfig(BaseModel):
    """Recommended configuration for distributed training."""
    recommended_node_count: int = 1
    recommended_gpus_per_node: int = 8
    total_gpus: int = 8
    parallelism_strategy: str = "ddp"
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    data_parallel_degree: int = 8
    recommended_batch_size: int = 32
    recommended_precision: str = "bf16"
    estimated_throughput_samples_per_sec: float = 0.0
    estimated_speedup_over_single: float = 1.0
    communication_overhead_estimate: float = 0.0
    topology_aware: bool = False
    tp_within_node: bool = True
    dp_across_nodes: bool = True
    recommendations: list[str] = Field(default_factory=list)
    summary: str = ""


class SlurmNodeInfo(BaseModel):
    """Information about a Slurm node."""
    node_name: str
    state: str = "unknown"
    partitions: list[str] = Field(default_factory=list)
    cpu_count: int = 0
    memory_mb: int = 0
    gpu_count: int = 0
    gpu_model: str = ""
    features: list[str] = Field(default_factory=list)
    weight: int = 1


class SlurmPartitionInfo(BaseModel):
    """Information about a Slurm partition."""
    name: str
    state: str = "up"
    nodes: list[str] = Field(default_factory=list)
    total_cpus: int = 0
    total_gpus: int = 0
    default_time_minutes: int = 60
    max_time_minutes: int = 1440
    gpu_model: str = ""


class SlurmJobInfo(BaseModel):
    """Information about a Slurm job."""
    job_id: int = 0
    job_name: str = ""
    partition: str = ""
    user: str = ""
    state: str = ""
    node_count: int = 0
    gpu_count: int = 0
    cpus: int = 0
    memory_mb: int = 0
    time_limit_minutes: int = 0
    time_used_minutes: int = 0
    nodes: str = ""
    submit_time: datetime | None = None
    start_time: datetime | None = None


class SlurmClusterTelemetry(BaseModel):
    """Slurm-specific cluster telemetry."""
    cluster_id: UUID
    cluster_name: str
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    controller_status: str = "unknown"
    node_count: int = 0
    gpu_count: int = 0
    nodes: list[SlurmNodeInfo] = Field(default_factory=list)
    partitions: list[SlurmPartitionInfo] = Field(default_factory=list)
    pending_jobs: list[SlurmJobInfo] = Field(default_factory=list)
    running_jobs: list[SlurmJobInfo] = Field(default_factory=list)
    total_cpus: int = 0
    allocated_cpus: int = 0
    total_memory_mb: int = 0
    allocated_memory_mb: int = 0
    total_gpus_allocated: int = 0


class SlurmJobSnapshot(BaseModel):
    """A single snapshot of a Slurm job's state over time."""
    timestamp: datetime
    job_id: int
    state: str
    time_used_minutes: int = 0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0


class MonitoringSnapshot(BaseModel):
    """Periodic snapshot of Slurm cluster state for monitoring."""
    cluster_id: UUID
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    running_jobs: list[SlurmJobInfo] = Field(default_factory=list)
    pending_jobs: list[SlurmJobInfo] = Field(default_factory=list)
    total_gpus: int = 0
    free_gpus: int = 0
    avg_cluster_utilization: float = 0.0
    total_waiting_jobs: int = 0
    total_running_jobs: int = 0
    total_gpu_hours_used: float = 0.0
    job_history: list[SlurmJobSnapshot] = Field(default_factory=list)


class JobMonitorConfig(BaseModel):
    """Configuration for monitoring a specific job."""
    job_id: int
    poll_interval_seconds: int = 30
    max_history_points: int = 1000
    alert_on_completion: bool = True
    alert_on_failure: bool = True
    alert_on_stall: bool = True
    stall_threshold_minutes: int = 10


class SlurmReservation(BaseModel):
    """A Slurm reservation for guaranteed resource access."""
    id: str = ""
    name: str = ""
    partition: str = ""
    nodes: list[str] = Field(default_factory=list)
    gpu_count: int = 0
    cpu_count: int = 0
    memory_mb: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_minutes: int = 0
    users: list[str] = Field(default_factory=list)
    accounts: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    state: str = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SlurmJobControlRequest(BaseModel):
    """Request to control a Slurm job (submit, cancel, hold, release)."""
    action: str  # submit, cancel, hold, release, modify
    job_id: int = 0
    job_name: str = ""
    partition: str = ""
    gpu_count: int = 0
    cpu_count: int = 0
    memory_mb: int = 0
    time_limit_minutes: int = 0
    node_count: int = 1
    script: str = ""
    dependency: str = ""


class SlurmJobControlResult(BaseModel):
    """Result of a Slurm job control action."""
    success: bool = False
    action: str = ""
    job_id: int = 0
    message: str = ""


class SlurmReservationRequest(BaseModel):
    """Request to create a Slurm reservation."""
    name: str
    partition: str = ""
    node_count: int = 0
    gpu_count: int = 0
    cpu_count: int = 0
    memory_mb: int = 0
    duration_minutes: int = 120
    users: list[str] = Field(default_factory=list)
    accounts: list[str] = Field(default_factory=list)


# ── S22: Guarded Automation ────────────────────────────────


class PolicySeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyRule(BaseModel):
    """A single guardrail policy that blocks or warns on actuation."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    scope_type: str = "global"  # cluster, environment, global
    scope_value: str = ""       # cluster_id, env name, or empty for global
    rule_type: str = "environment_restriction"  # environment_restriction, time_window, resource_limit, approval_required, maintenance_window, custom
    rule_config: dict[str, Any] = Field(default_factory=dict)
    severity: PolicySeverity = PolicySeverity.MEDIUM
    enabled: bool = True
    fail_action: str = "block"  # block, warn, allow
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PolicyEvaluationResult(BaseModel):
    """Result of evaluating a single policy against an actuation."""
    policy_id: UUID
    policy_name: str
    rule_type: str
    severity: PolicySeverity
    passed: bool
    action: str  # block, warn, allow
    message: str = ""
    details: list[str] = Field(default_factory=list)


class PreFlightCheckResult(BaseModel):
    """Complete pre-flight check for an actuation."""
    actuation_rec_id: UUID
    cluster_id: UUID
    cluster_name: str = ""
    overall_passed: bool = False
    policy_count: int = 0
    passed_count: int = 0
    blocked_count: int = 0
    warned_count: int = 0
    results: list[PolicyEvaluationResult] = Field(default_factory=list)
    requires_approval: bool = False
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ApprovalStep(BaseModel):
    """A single step in a multi-step approval workflow."""
    step_order: int = 1
    approver: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_at: datetime | None = None
    reason: str = ""


class ApprovalWorkflowRequest(BaseModel):
    """Request to create an approval workflow for an actuation."""
    actuation_id: UUID
    cluster_id: UUID | None = None
    required_approvers: list[str] = Field(default_factory=list)
    reason: str = ""


class ApprovalRecord(BaseModel):
    """Record of an approval workflow for an actuation."""
    id: UUID = Field(default_factory=uuid4)
    actuation_id: UUID
    cluster_id: UUID
    cluster_name: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    steps: list[ApprovalStep] = Field(default_factory=list)
    required_approvers: list[str] = Field(default_factory=list)
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decided_at: datetime | None = None
    final_reason: str = ""


class ChaosFaultType(StrEnum):
    NODE_FAILURE = "node_failure"
    GPU_FAILURE = "gpu_failure"
    NETWORK_PARTITION = "network_partition"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_FILL = "disk_fill"
    LATENCY_INJECTION = "latency_injection"
    POD_KILL = "pod_kill"


class ChaosFaultTarget(BaseModel):
    """Target specification for a chaos fault."""
    target_type: str = "node"  # node, gpu, pod, network
    target_selector: dict[str, Any] = Field(default_factory=dict)
    count: int = 1


class ChaosExperiment(BaseModel):
    """A chaos experiment definition."""
    id: UUID = Field(default_factory=uuid4)
    cluster_id: UUID
    cluster_name: str = ""
    name: str
    description: str = ""
    fault_type: ChaosFaultType = ChaosFaultType.NODE_FAILURE
    target: ChaosFaultTarget = Field(default_factory=ChaosFaultTarget)
    duration_seconds: int = 60
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    status: str = "pending"  # pending, running, completed, failed, cancelled
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"


class ChaosExperimentResult(BaseModel):
    """Result of a completed chaos experiment."""
    experiment: ChaosExperiment
    experiment_status: str = ""
    target_impacted: int = 0
    gpu_utilization_drop_percent: float = 0.0
    latency_increase_ms: float = 0.0
    error_count: int = 0
    recovery_time_seconds: float = 0.0
    system_resilient: bool = False
    observations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    summary: str = ""


class GuardedAutomationRecommendation(BaseModel):
    """Recommendation for improving automation safety and governance."""
    cluster_id: UUID
    cluster_name: str = ""
    environment: str = ""
    recommendation: str = ""
    recommendation_type: str = ""  # policy, approval, chaos, monitoring
    priority: str = "medium"
    estimated_risk_reduction: str = ""
    actions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── S23: Observability, Multi-Tenancy, Anomaly, Compliance, Dashboard ──


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertConditionType(StrEnum):
    GPU_UTILIZATION = "gpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    GPU_TEMPERATURE = "gpu_temperature"
    IDLE_GPU = "idle_gpu"
    COST_ANOMALY = "cost_anomaly"
    DRIFT_DETECTED = "drift_detected"
    POWER_EFFICIENCY = "power_efficiency"
    JOB_FAILURE = "job_failure"
    BUDGET_ALERT = "budget_alert"


class NotificationChannelType(StrEnum):
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"


class AlertRule(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    cluster_id: UUID
    condition_type: AlertConditionType = AlertConditionType.GPU_UTILIZATION
    operator: str = "lt"  # lt, gt, eq, lte, gte
    threshold: float = 0.0
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    cooldown_minutes: int = 60
    notification_channel_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AlertRuleEvaluation(BaseModel):
    rule_id: UUID
    rule_name: str = ""
    passed: bool = False
    current_value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AlertRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    rule_id: UUID
    cluster_id: UUID
    cluster_name: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    condition_type: AlertConditionType = AlertConditionType.GPU_UTILIZATION
    current_value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    status: str = "firing"  # firing, resolved, acknowledged
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None
    acknowledged_by: str = ""


class NotificationChannel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    channel_type: NotificationChannelType = NotificationChannelType.EMAIL
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NotificationMessage(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    channel_id: UUID
    channel_name: str = ""
    subject: str = ""
    body: str = ""
    status: str = "pending"  # pending, sent, failed
    sent_at: datetime | None = None
    error_message: str = ""


class Team(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Project(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    team_id: UUID
    cluster_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResourceQuota(BaseModel):
    project_id: UUID
    max_gpus: int = 0
    max_clusters: int = 0
    max_monthly_cost_usd: float = 0.0
    current_gpu_count: int = 0
    current_cluster_count: int = 0
    current_monthly_cost: float = 0.0
    gpu_utilization: float = 0.0
    quota_exceeded: bool = False
    violations: list[str] = Field(default_factory=list)


class CostAnomalyResult(BaseModel):
    cluster_id: UUID
    cluster_name: str = ""
    period: str = ""
    expected_cost: float = 0.0
    actual_cost: float = 0.0
    deviation: float = 0.0
    deviation_percent: float = 0.0
    anomaly_score: float = 0.0
    is_anomaly: bool = False
    contributing_factors: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class AuditLogEntry(BaseModel):
    id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str = "system"
    action: str = ""
    resource_type: str = ""
    resource_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    ip_address: str = ""
    severity: str = "info"


class ComplianceControl(BaseModel):
    id: str = ""
    name: str = ""
    category: str = ""
    status: str = "pass"  # pass, fail, warn, na
    message: str = ""
    remediation: str = ""


class ComplianceReport(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    cluster_id: UUID
    cluster_name: str = ""
    framework: str = "soc2"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    overall_status: str = "fail"  # pass, fail, warn
    controls: list[ComplianceControl] = Field(default_factory=list)
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    summary: str = ""


class DashboardMetric(BaseModel):
    label: str = ""
    value: float = 0.0
    unit: str = ""
    trend: str = "stable"  # up, down, stable
    change_percent: float = 0.0


class DashboardSummary(BaseModel):
    cluster_id: UUID
    cluster_name: str = ""
    environment: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    gpu_count: int = 0
    avg_utilization: float = 0.0
    total_cost_monthly: float = 0.0
    estimated_savings: float = 0.0
    active_alerts: int = 0
    efficiency_score: float = 0.0
    metrics: list[DashboardMetric] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class ScheduledReport(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    cluster_ids: list[str] = Field(default_factory=list)
    format: str = "pdf"  # pdf, csv, json
    schedule: str = "weekly"  # daily, weekly, monthly
    recipients: list[str] = Field(default_factory=list)
    last_sent_at: datetime | None = None
    next_send_at: datetime | None = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Workload Agent Schemas ─────────────────────────────────────


class SystemInfo(BaseModel):
    detected_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hostname: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_usage_percent: float = 0.0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_usage_percent: float = 0.0
    gpu_count: int = 0
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    total_gpu_memory_gb: float = 0.0
    used_gpu_memory_gb: float = 0.0
    free_gpu_memory_gb: float = 0.0
    cluster_id: str = ""


class WorkloadInput(BaseModel):
    name: str
    gpu_required: int = 1
    memory_required_gb: float = 0.0
    cpu_required_cores: float = 0.0
    priority: str = "normal"
    max_duration_minutes: float = 120.0
    framework: str = "pytorch"
    precision: str = "fp32"
    dataset_size_gb: float = 0.0
    model_size_gb: float = 0.0
    cluster_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MLPredictionResult(BaseModel):
    success_probability: float = 0.0
    predicted_duration_minutes: float = 0.0
    predicted_memory_peak_gb: float = 0.0
    predicted_gpu_utilization: float = 0.0
    risk_factors: list[str] = Field(default_factory=list)
    recommendation: str = ""


class DigitalTwinSimulation(BaseModel):
    simulation_id: str = Field(default_factory=lambda: str(uuid4()))
    workload: WorkloadInput
    system: SystemInfo | None = None
    prediction: MLPredictionResult | None = None
    feasible: bool = False
    rejection_reason: str = ""
    assigned_gpu_indices: list[int] = Field(default_factory=list)
    assigned_memory_gb: float = 0.0
    estimated_cost: float = 0.0
    simulated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JobAssignment(BaseModel):
    assignment_id: str = Field(default_factory=lambda: str(uuid4()))
    workload: WorkloadInput
    simulation: DigitalTwinSimulation
    assigned_gpu_indices: list[int] = Field(default_factory=list)
    assigned_memory_gb: float = 0.0
    assigned_node: str = ""
    status: str = "assigned"
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str = ""
    actual_duration_minutes: float = 0.0
    actual_success: bool = False


# ═══════════════════════════════════════════════════════════════
# Domain 1: Telemetry & State (Extended)
# ═══════════════════════════════════════════════════════════════

class EndpointTelemetry(BaseModel):
    endpoint: str = ""
    total_requests: int = 0
    requests_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    throughput_tokens_per_sec: float = 0.0


class ModelServiceTelemetry(BaseModel):
    model_name: str = ""
    model_version: str = ""
    replicas: int = 0
    endpoints: list[EndpointTelemetry] = Field(default_factory=list)
    avg_gpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    cpu_usage_percent: float = 0.0
    collected_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FabricLinkTelemetry(BaseModel):
    link_index: int = 0
    link_type: str = ""  # nvlink, pcie, nvswitch
    bandwidth_usage_percent: float = 0.0
    tx_bytes_per_sec: float = 0.0
    rx_bytes_per_sec: float = 0.0
    errors: int = 0
    crc_errors: int = 0
    link_width: int = 0
    link_gen: int = 0
    is_active: bool = True


class FabricTelemetry(BaseModel):
    node: str = ""
    gpu_index: int = 0
    links: list[FabricLinkTelemetry] = Field(default_factory=list)
    nvlink_bandwidth_utilization: float = 0.0
    pcie_bandwidth_utilization: float = 0.0
    total_nvlink_errors: int = 0


class QueueTelemetry(BaseModel):
    queue_name: str = ""
    queue_depth: int = 0
    pending_jobs: int = 0
    running_jobs: int = 0
    avg_wait_time_seconds: float = 0.0
    max_wait_time_seconds: float = 0.0
    p99_wait_time_seconds: float = 0.0
    submission_rate_per_minute: float = 0.0
    completion_rate_per_minute: float = 0.0
    backlog_growth_rate: float = 0.0
    priority_breaks: int = 0
    starved_jobs: int = 0


class JobTelemetry(BaseModel):
    job_id: str = ""
    job_name: str = ""
    state: str = ""
    priority: int = 0
    gpu_required: int = 0
    memory_required_gb: float = 0.0
    wall_time_seconds: float = 0.0
    wait_time_seconds: float = 0.0
    gpu_utilization_avg: float = 0.0
    memory_utilization_avg: float = 0.0
    oom_killed: bool = False
    exit_code: int = 0
    preempted: bool = False
    submitted_at: str = ""
    started_at: str = ""
    completed_at: str = ""


class TelemetrySnapshot(BaseModel):
    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    cluster_id: str = ""
    collected_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    gpu_snapshot: dict[str, Any] = Field(default_factory=dict)
    model_services: list[ModelServiceTelemetry] = Field(default_factory=list)
    fabric: list[FabricTelemetry] = Field(default_factory=list)
    queues: list[QueueTelemetry] = Field(default_factory=list)
    jobs: list[JobTelemetry] = Field(default_factory=list)


class TelemetryStreamEvent(BaseModel):
    event_type: str  # gpu, model_service, fabric, queue, job
    cluster_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ═══════════════════════════════════════════════════════════════
# Domain 2: Prediction (Extended)
# ═══════════════════════════════════════════════════════════════

class QueuePressureForecast(BaseModel):
    forecast_horizon_hours: float = 1.0
    current_queue_depth: int = 0
    predicted_queue_depth: float = 0.0
    predicted_wait_time_minutes: float = 0.0
    congestion_probability: float = 0.0
    pressure_level: str = "low"
    recommended_actions: list[str] = Field(default_factory=list)


class JCTPrediction(BaseModel):
    job_id: str = ""
    estimated_duration_minutes: float = 0.0
    p50_duration_minutes: float = 0.0
    p95_duration_minutes: float = 0.0
    p99_duration_minutes: float = 0.0
    confidence: float = 0.0
    factors: list[str] = Field(default_factory=list)


class OOMRiskPrediction(BaseModel):
    job_id: str = ""
    gpu_index: int = 0
    current_memory_used_gb: float = 0.0
    peak_memory_predicted_gb: float = 0.0
    available_memory_gb: float = 0.0
    oom_probability: float = 0.0
    risk_level: str = "low"
    recommendation: str = ""


class ThermalRiskPrediction(BaseModel):
    node: str = ""
    gpu_index: int = 0
    current_temperature_c: float = 0.0
    predicted_peak_temperature_c: float = 0.0
    thermal_throttle_probability: float = 0.0
    time_to_throttle_minutes: float = 0.0
    risk_level: str = "low"
    recommendation: str = ""


class DemandBurstDetection(BaseModel):
    burst_detected: bool = False
    burst_start_time: str = ""
    burst_magnitude: float = 0.0
    burst_duration_seconds: float = 0.0
    affected_metrics: list[str] = Field(default_factory=list)
    trigger_threshold: float = 0.0
    severity: str = "info"


class ActionImpactForecast(BaseModel):
    action_type: str = ""
    description: str = ""
    expected_gpu_utilization_change: float = 0.0
    expected_memory_freed_gb: float = 0.0
    expected_cost_savings: float = 0.0
    expected_performance_impact: float = 0.0
    risk_of_disruption: float = 0.0
    confidence: float = 0.0
    recommended: bool = True


class ComprehensivePrediction(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid4()))
    cluster_id: str = ""
    predicted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    queue_forecast: QueuePressureForecast | None = None
    jct_predictions: list[JCTPrediction] = Field(default_factory=list)
    oom_risks: list[OOMRiskPrediction] = Field(default_factory=list)
    thermal_risks: list[ThermalRiskPrediction] = Field(default_factory=list)
    demand_bursts: list[DemandBurstDetection] = Field(default_factory=list)
    action_impacts: list[ActionImpactForecast] = Field(default_factory=list)
    overall_risk_score: float = 0.0
    summary: str = ""


# ═══════════════════════════════════════════════════════════════
# Domain 3: Digital Twin (Extended)
# ═══════════════════════════════════════════════════════════════

class CounterfactualScenario(BaseModel):
    scenario_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    applied_actions: list[ActionImpactForecast] = Field(default_factory=list)
    predicted_utilization: float = 0.0
    predicted_cost_per_hour: float = 0.0
    predicted_power_watts: float = 0.0
    predicted_slo_compliance: float = 0.0
    job_completion_time_impact: float = 0.0
    risk_score: float = 0.0
    feasibility_score: float = 0.0
    recommendation: str = ""


class CandidateActionScore(BaseModel):
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    action_type: str = ""
    target_node: str = ""
    target_gpus: list[int] = Field(default_factory=list)
    feasibility: bool = False
    utility_score: float = 0.0
    cost_score: float = 0.0
    performance_score: float = 0.0
    power_score: float = 0.0
    risk_score: float = 0.0
    overall_score: float = 0.0
    explanation: str = ""


class FullSimulationResult(BaseModel):
    simulation_id: str = Field(default_factory=lambda: str(uuid4()))
    cluster_id: str = ""
    twin_id: str = ""
    simulated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    scenarios: list[CounterfactualScenario] = Field(default_factory=list)
    candidate_scores: list[CandidateActionScore] = Field(default_factory=list)
    baseline_cost: float = 0.0
    baseline_power: float = 0.0
    baseline_slo_compliance: float = 0.0
    optimized_cost: float = 0.0
    optimized_power: float = 0.0
    optimized_slo_compliance: float = 0.0
    savings_percentage: float = 0.0
    summary: str = ""


# ═══════════════════════════════════════════════════════════════
# Domain 4: Optimization (Extended)
# ═══════════════════════════════════════════════════════════════

class ElasticWorkerConfig(BaseModel):
    min_workers: int = 1
    max_workers: int = 16
    current_workers: int = 1
    scale_up_threshold_utilization: float = 70.0
    scale_down_threshold_utilization: float = 30.0
    cooldown_seconds: int = 60
    gpu_per_worker: int = 1
    memory_per_worker_gb: float = 0.0


class GpuTierSelection(BaseModel):
    current_gpu_model: str = ""
    recommended_gpu_model: str = ""
    current_cost_per_hour: float = 0.0
    recommended_cost_per_hour: float = 0.0
    savings_per_hour: float = 0.0
    performance_impact: str = "none"
    confidence: float = 0.0
    reasoning: str = ""


class ConsolidationPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    cluster_id: str = ""
    current_node_count: int = 0
    target_node_count: int = 0
    nodes_to_drain: list[str] = Field(default_factory=list)
    workloads_to_move: list[str] = Field(default_factory=list)
    estimated_cost_savings: float = 0.0
    estimated_power_savings: float = 0.0
    estimated_performance_impact: float = 0.0
    risk_level: str = "low"
    feasibility: bool = True
    steps: list[str] = Field(default_factory=list)


class RecommendationPriority(BaseModel):
    recommendation_id: str = ""
    priority_score: float = 0.0
    urgency: str = "medium"
    impact: str = "medium"
    effort: str = "medium"
    roi: float = 0.0
    dependencies: list[str] = Field(default_factory=list)
    suggested_order: int = 0


# ═══════════════════════════════════════════════════════════════
# Domain 5: Training (Extended)
# ═══════════════════════════════════════════════════════════════

class QueueAwarePlacement(BaseModel):
    job_id: str = ""
    placement_node: str = ""
    placement_gpus: list[int] = Field(default_factory=list)
    queue_wait_predicted_minutes: float = 0.0
    priority: int = 0
    preemptible: bool = False
    checkpoint_available: bool = False
    estimated_duration_minutes: float = 0.0


class ElasticScalingPlan(BaseModel):
    job_id: str = ""
    current_workers: int = 1
    target_workers: int = 1
    scaling_reason: str = ""
    estimated_speedup: float = 1.0
    estimated_cost_impact: float = 0.0
    min_workers: int = 1
    max_workers: int = 64


class CheckpointConfig(BaseModel):
    checkpoint_enabled: bool = True
    checkpoint_interval_minutes: int = 30
    checkpoint_path: str = ""
    last_checkpoint_time: str = ""
    checkpoint_size_gb: float = 0.0
    restore_time_seconds: float = 0.0


class HeterogeneousGpuAssignment(BaseModel):
    job_id: str = ""
    primary_gpu_model: str = ""
    secondary_gpu_model: str = ""
    primary_gpu_count: int = 0
    secondary_gpu_count: int = 0
    strategy: str = "uniform"  # uniform, pipeline, mixed
    expected_speedup: float = 1.0
    compatibility_score: float = 1.0


class HPOJob(BaseModel):
    job_id: str = ""
    search_algorithm: str = "bayesian"
    max_trials: int = 100
    parallel_trials: int = 4
    objective_metric: str = "loss"
    best_trial_id: str = ""
    best_score: float = 0.0
    status: str = "pending"
    trials: list[dict[str, Any]] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Domain 6: Inference (Extended)
# ═══════════════════════════════════════════════════════════════

class ReplicaRightSizing(BaseModel):
    model_name: str = ""
    current_replicas: int = 0
    recommended_replicas: int = 0
    current_gpu_per_replica: int = 0
    recommended_gpu_per_replica: int = 0
    current_latency_p99_ms: float = 0.0
    target_latency_p99_ms: float = 0.0
    current_throughput_tps: float = 0.0
    recommended_throughput_tps: float = 0.0
    estimated_cost_savings: float = 0.0


class SloAwareScalingPolicy(BaseModel):
    policy_id: str = Field(default_factory=lambda: str(uuid4()))
    model_name: str = ""
    min_replicas: int = 1
    max_replicas: int = 32
    target_latency_p99_ms: float = 100.0
    target_throughput_per_replica: float = 100.0
    scale_up_cooldown_seconds: int = 30
    scale_down_cooldown_seconds: int = 60
    current_replicas: int = 1
    current_load_tps: float = 0.0


class ModelInstancePlacement(BaseModel):
    model_name: str = ""
    model_version: str = ""
    instance_id: str = ""
    node: str = ""
    gpu_indices: list[int] = Field(default_factory=list)
    gpu_memory_allocated_gb: float = 0.0
    status: str = "active"
    routing_weight: float = 1.0


class RoutingRecommendation(BaseModel):
    model_name: str = ""
    current_routing: str = "round_robin"
    recommended_routing: str = "latency_based"
    expected_latency_improvement: float = 0.0
    expected_throughput_improvement: float = 0.0
    reasoning: str = ""


class MoEConfig(BaseModel):
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    expert_parallelism: int = 1
    enable_auxiliary_loss: bool = True
    load_balancing_type: str = "auxiliary_loss"
    recommended_routing: str = "top_k"


# ═══════════════════════════════════════════════════════════════
# Domain 7: Governance (Extended)
# ═══════════════════════════════════════════════════════════════

class PolicyEnvelope(BaseModel):
    policy_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    domain: str = "general"
    rules: list[dict[str, Any]] = Field(default_factory=list)
    action: str = "block"
    enabled: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RollbackPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    action_id: str = ""
    action_type: str = ""
    reason: str = ""
    steps: list[str] = Field(default_factory=list)
    estimated_rollback_time_seconds: float = 0.0
    risk_level: str = "low"
    automated: bool = True
    requires_approval: bool = False


class TenantQuota(BaseModel):
    tenant_id: str = ""
    tenant_name: str = ""
    max_gpus: int = 0
    max_memory_gb: float = 0.0
    max_priority: int = 0
    gpus_in_use: int = 0
    memory_in_use_gb: float = 0.0
    priority_surcharge: float = 1.0
    burst_allowed: bool = False
    burst_max_gpus: int = 0
    quota_usage_percent: float = 0.0


class Explanation(BaseModel):
    explanation_id: str = Field(default_factory=lambda: str(uuid4()))
    subject_type: str = ""
    subject_id: str = ""
    summary: str = ""
    factors: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 1.0
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GovernanceReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    report_type: str = "compliance"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cluster_id: str = ""
    tenant_summaries: list[dict[str, Any]] = Field(default_factory=list)
    policy_violations: int = 0
    approval_metrics: dict[str, int] = Field(default_factory=dict)
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Domain 8: Integration (Extended)
# ═══════════════════════════════════════════════════════════════

class PrometheusTarget(BaseModel):
    target_id: str = ""
    endpoint: str = ""
    scrape_interval_seconds: int = 15
    labels: dict[str, str] = Field(default_factory=dict)
    healthy: bool = True


class OpenTelemetryConfig(BaseModel):
    service_name: str = "gpuopt"
    endpoint: str = ""
    protocol: str = "grpc"
    sampling_rate: float = 0.1
    enabled: bool = True


class AiRuntimeInfo(BaseModel):
    runtime_type: str = ""  # pytorch, tensorflow, jax, onnx
    version: str = ""
    gpu_visible: bool = False
    cuda_available: bool = False
    cuda_version: str = ""
    compute_capability: str = ""
    memory_allocated_gb: float = 0.0
    memory_reserved_gb: float = 0.0
    processes: list[dict[str, Any]] = Field(default_factory=list)


class ObjectStoreConfig(BaseModel):
    store_type: str = ""  # s3, gcs, azure_blob, minio
    endpoint: str = ""
    bucket: str = ""
    region: str = ""
    secure: bool = True
    access_key: str = ""
    secret_key: str = ""
    connection_test_passed: bool = False


class IntegrationStatus(BaseModel):
    integration_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""
    type: str = ""
    connected: bool = False
    last_heartbeat: str = ""
    metrics_count: int = 0
    error_count: int = 0
    latency_ms: float = 0.0
