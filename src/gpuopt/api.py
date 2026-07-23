from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__
from .config import get_settings
from .dependencies import (
    get_actuation_service,
    get_alert_manager,
    get_analysis_service,
    get_anomaly_detector,
    get_approval_workflow,
    get_chaos_engine,
    get_check_service,
    get_compliance_engine,
    get_cost_analysis_service,
    get_dashboard_service,
    get_digital_twin,
    get_drift_detector,
    get_finops_service,
    get_forecast_model,
    get_inference_service,
    get_policy_engine,
    get_power_service,
    get_rec_engine,
    get_rec_model,
    get_report_scheduler,
    get_repository,
    get_scheduler_service,
    get_state_service,
    get_tenant_manager,
    get_trace_service,
    get_training_service,
)
from .repository import ClusterRepository, RepositoryError
from .schemas import (
    AlertRecord,
    AlertRule,
    AlertRuleEvaluation,
    AlertSeverity,
    ApprovalRecord,
    ApprovalWorkflowRequest,
    ActuationRecord,
    ActuationRequest,
    ActuationSummary,
    AnalysisSummary,
    BudgetAlert,
    ChaosExperiment,
    ChaosExperimentResult,
    CloudProvider,
    ComplianceReport,
    CostAnomalyResult,
    DashboardSummary,
    NotificationChannel,
    NotificationMessage,
    Project,
    ResourceQuota,
    ScheduledReport,
    Team,
    CostAllocationTag,
    CostForecast,
    CostReport,
    CostSummary,
    GpuPricingTier,
    GuardedAutomationRecommendation,
    SavingsProjection,
    BaselineInfo,
    ClusterCreate,
    ClusterRecord,
    ClusterStateData,
    ClusterStateSummary,
    DemandForecast,
    DistributedTrainingConfig,
    EnvironmentCheckReport,
    EnvironmentSummary,
    HPOConfig,
    HTOResult,
    InferenceDeploymentConfig,
    InferenceEndpoint,
    InferenceFramework,
    InferenceProfile,
    JobMonitorConfig,
    MonitoringSnapshot,
    MultiClusterCostSummary,
    NodeTopology,
    PlacementSuggestion,
    PolicyRule,
    PolicySeverity,
    PreFlightCheckResult,
    PowerAnalysisResult,
    PowerCapSuggestion,
    PowerOptimizationRecommendation,
    ProviderCostComparison,
    RecommendationSet,
    ReservedInstanceRecommendation,
    ResourceRecommendation,
    ScheduleSimulation,
    SchedulingPlan,
    SlurmClusterTelemetry,
    SlurmJobInfo,
    SlurmNodeInfo,
    SlurmPartitionInfo,
    SpotSavingsAnalysis,
    StateComparison,
    StatusUpdate,
    TraceListItem,
    TraceReplayResult,
    TrainingFramework,
    TrainingJob,
    TrainingProfile,
    TwinComparison,
    TwinState,
    WhatIfCostScenario,
    WhatIfProjection,
    WorkloadAnalysisResult,
    WorkloadRequirements,
)
from .agent_protocol import AgentRegistry, AgentStatus, AgentEventType, get_agent_registry
from .dcgm_ingestion import DcgmIngestionPipeline, get_dcgm_pipeline
from .dcgm_quality import DcgmQualityAnalyzer
from .workload_attribution import WorkloadAttributionEngine, get_attribution_engine, TenantIsolationPolicy
from .explanation_service import ExplanationService, get_explanation_service, ExplanationCategory
from .actuation import ActuationService
from .analysis import AnalysisService
from .cost_analysis import CostAnalysisService
from .digital_twin import DigitalTwinService
from .guarded_automation import ApprovalWorkflow, ChaosEngine, PolicyEngine
from .inference.service import InferenceService
from .s23_features import AlertManager, CostAnomalyDetector, ComplianceEngine, DashboardService, ReportScheduler, TenantManager
from .finops import FinOpsService
from .power import PowerService
from .ml.drift_detector import DriftDetector
from .ml.forecast_model import ForecastModel
from .ml.recommendation_model import RecommendationModel
from .recommendations import RecommendationEngine
from .scheduler import SchedulerService
from .services import ClusterStateService, EnvironmentCheckService
from .trace import TraceService
from .training import TrainingService

router = APIRouter()


@router.get("/health/live", tags=["health"])
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready", tags=["health"])
def readiness(repository: ClusterRepository = Depends(get_repository)) -> dict[str, str | int]:
    repository.list_clusters()
    return {"status": "ready", "registered_clusters": len(repository.list_clusters())}


@router.get("/health/detailed", tags=["health"])
def detailed_health(repository: ClusterRepository = Depends(get_repository)) -> dict:
    """Detailed health check with system information."""
    settings = get_settings()
    clusters = repository.list_clusters()

    return {
        "status": "healthy",
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "configuration": {
            "environment": settings.env,
            "database_path": str(settings.database_path),
            "allow_mock_gpu": settings.allow_mock_gpu,
        },
        "clusters": {
            "total": len(clusters),
            "by_environment": _count_by_environment(clusters),
            "by_connector_type": _count_by_connector_type(clusters),
        },
    }


def _count_by_environment(clusters: list[ClusterRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cluster in clusters:
        counts[cluster.environment] = counts.get(cluster.environment, 0) + 1
    return counts


def _count_by_connector_type(clusters: list[ClusterRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cluster in clusters:
        counts[cluster.connector_type] = counts.get(cluster.connector_type, 0) + 1
    return counts


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post(
    "/api/v1/clusters",
    response_model=ClusterRecord,
    status_code=status.HTTP_201_CREATED,
    tags=["clusters"],
)
def create_cluster(
    payload: ClusterCreate,
    repository: ClusterRepository = Depends(get_repository),
) -> ClusterRecord:
    try:
        return repository.create_cluster(payload)
    except RepositoryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put("/api/v1/clusters/by-name/{name}", response_model=ClusterRecord, tags=["clusters"])
def upsert_cluster(
    name: str,
    payload: ClusterCreate,
    repository: ClusterRepository = Depends(get_repository),
) -> ClusterRecord:
    if name != payload.name:
        raise HTTPException(status_code=400, detail="Path name must match payload name")
    return repository.upsert_cluster(payload)


def _compute_cluster_status(cluster_id: UUID, repository: ClusterRepository) -> str:
    from datetime import datetime, timezone
    state = repository.latest_state(cluster_id)
    if state is None:
        return "unknown"
    age = (datetime.now(timezone.utc) - state.collected_at).total_seconds()
    if age < 300:
        return "healthy"
    return "warning"


@router.get("/api/v1/clusters", response_model=list[ClusterRecord], tags=["clusters"])
def list_clusters(repository: ClusterRepository = Depends(get_repository)) -> list[ClusterRecord]:
    clusters = repository.list_clusters()
    for c in clusters:
        c.status = _compute_cluster_status(c.id, repository)
    return clusters


@router.get("/api/v1/clusters/{cluster_id}", response_model=ClusterRecord, tags=["clusters"])
def get_cluster(
    cluster_id: UUID,
    repository: ClusterRepository = Depends(get_repository),
) -> ClusterRecord:
    cluster = repository.get_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    cluster.status = _compute_cluster_status(cluster_id, repository)
    return cluster


@router.delete("/api/v1/clusters/{cluster_id}", status_code=204, tags=["clusters"])
def delete_cluster(
    cluster_id: UUID,
    repository: ClusterRepository = Depends(get_repository),
) -> Response:
    if not repository.delete_cluster(cluster_id):
        raise HTTPException(status_code=404, detail="Cluster not found")
    return Response(status_code=204)


@router.post(
    "/api/v1/clusters/{cluster_id}/checks",
    response_model=EnvironmentCheckReport,
    tags=["environment checks"],
)
def run_cluster_check(
    cluster_id: UUID,
    service: EnvironmentCheckService = Depends(get_check_service),
) -> EnvironmentCheckReport:
    try:
        return service.check_cluster(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/checks/latest",
    response_model=EnvironmentCheckReport,
    tags=["environment checks"],
)
def latest_cluster_check(
    cluster_id: UUID,
    repository: ClusterRepository = Depends(get_repository),
) -> EnvironmentCheckReport:
    if repository.get_cluster(cluster_id) is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    report = repository.latest_report(cluster_id)
    if report is None:
        raise HTTPException(status_code=404, detail="No check report exists")
    return report


@router.post(
    "/api/v1/environments/check-all",
    response_model=list[EnvironmentCheckReport],
    tags=["environment checks"],
)
def check_all_environments(
    service: EnvironmentCheckService = Depends(get_check_service),
) -> list[EnvironmentCheckReport]:
    return service.check_all()


@router.get(
    "/api/v1/environments/summary",
    response_model=EnvironmentSummary,
    tags=["environment checks"],
)
def environment_summary(
    service: EnvironmentCheckService = Depends(get_check_service),
) -> EnvironmentSummary:
    return service.summarize()


@router.post(
    "/api/v1/clusters/{cluster_id}/state",
    response_model=ClusterStateData,
    tags=["cluster state"],
)
def collect_cluster_state(
    cluster_id: UUID,
    service: ClusterStateService = Depends(get_state_service),
) -> ClusterStateData:
    """Collect current telemetry and produce a normalized cluster state snapshot."""
    try:
        return service.collect_state(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/state",
    response_model=ClusterStateData,
    tags=["cluster state"],
)
def get_cluster_state(
    cluster_id: UUID,
    service: ClusterStateService = Depends(get_state_service),
) -> ClusterStateData:
    """Get the latest collected cluster state snapshot."""
    try:
        state = service.get_latest_state(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if state is None:
        raise HTTPException(status_code=404, detail="No cluster state has been collected")
    return state


@router.get(
    "/api/v1/state/summary",
    response_model=list[ClusterStateSummary],
    tags=["cluster state"],
)
def state_summary(
    service: ClusterStateService = Depends(get_state_service),
) -> list[ClusterStateSummary]:
    """Summarize cluster state across all registered clusters with freshness indicators."""
    return service.summarize_all()


@router.get(
    "/api/v1/clusters/{cluster_id}/traces",
    response_model=list[TraceListItem],
    tags=["trace replay"],
)
def list_traces(
    cluster_id: UUID,
    limit: int = 50,
    offset: int = 0,
    service: TraceService = Depends(get_trace_service),
) -> list[TraceListItem]:
    """List historical state snapshots (traces) for a cluster."""
    try:
        return service.list_traces(cluster_id, limit=limit, offset=offset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/traces/{trace_id}",
    response_model=ClusterStateData,
    tags=["trace replay"],
)
def get_trace(
    cluster_id: UUID,
    trace_id: str,
    service: TraceService = Depends(get_trace_service),
) -> ClusterStateData:
    """Get a specific historical trace by ID."""
    try:
        return service.get_trace(cluster_id, trace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/replay",
    response_model=TraceReplayResult,
    tags=["trace replay"],
)
def replay_trace(
    cluster_id: UUID,
    trace_id: str | None = None,
    service: TraceService = Depends(get_trace_service),
) -> TraceReplayResult:
    """Replay a historical trace through the check system.

    If no trace_id is provided, replays the latest state snapshot.
    """
    try:
        if trace_id is None:
            traces = service.list_traces(cluster_id, limit=1)
            if not traces:
                raise HTTPException(status_code=404, detail="No state data available to replay")
            trace_id = traces[0].id
        return service.replay_trace(cluster_id, trace_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/baseline",
    response_model=BaselineInfo,
    tags=["trace replay"],
)
def set_baseline(
    cluster_id: UUID,
    service: TraceService = Depends(get_trace_service),
) -> BaselineInfo:
    """Set the latest collected state as the comparison baseline."""
    try:
        return service.set_baseline(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/baseline",
    response_model=BaselineInfo,
    tags=["trace replay"],
)
def get_baseline(
    cluster_id: UUID,
    service: TraceService = Depends(get_trace_service),
) -> BaselineInfo:
    """Get the current baseline information for a cluster."""
    try:
        return service.get_baseline(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/compare",
    response_model=StateComparison,
    tags=["trace replay"],
)
def compare_traces(
    cluster_id: UUID,
    trace_id_a: str | None = None,
    trace_id_b: str | None = None,
    service: TraceService = Depends(get_trace_service),
) -> StateComparison:
    """Compare two cluster state snapshots.

    If neither trace_id is provided, compares the baseline vs latest state.
    If only trace_id_a is provided, compares baseline vs trace_id_a.
    If both are provided, compares trace_id_a vs trace_id_b.
    """
    try:
        if trace_id_a is None and trace_id_b is None:
            return service.compare_with_baseline(cluster_id)
        if trace_id_b is None:
            baseline = service.get_baseline(cluster_id)
            return service.compare_traces(cluster_id, baseline.trace_id, trace_id_a)
        return service.compare_traces(cluster_id, trace_id_a, trace_id_b)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/analyze",
    response_model=WorkloadAnalysisResult,
    tags=["workload analysis"],
)
def analyze_cluster(
    cluster_id: UUID,
    max_traces: int = 100,
    service: AnalysisService = Depends(get_analysis_service),
) -> WorkloadAnalysisResult:
    """Analyze historical trace data to produce GPU utilization trends, efficiency scores, and recommendations."""
    try:
        return service.analyze_cluster(cluster_id, max_traces=max_traces)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/analysis/latest",
    response_model=WorkloadAnalysisResult,
    tags=["workload analysis"],
)
def latest_analysis(
    cluster_id: UUID,
    service: AnalysisService = Depends(get_analysis_service),
) -> WorkloadAnalysisResult:
    """Get the latest workload analysis for a cluster."""
    try:
        result = service.get_latest_analysis(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="No analysis data available")
    return result


@router.get(
    "/api/v1/clusters/{cluster_id}/analysis/list",
    response_model=list[WorkloadAnalysisResult],
    tags=["workload analysis"],
)
def list_analyses(
    cluster_id: UUID,
    limit: int = 10,
    service: AnalysisService = Depends(get_analysis_service),
) -> list[WorkloadAnalysisResult]:
    """List historical workload analyses for a cluster."""
    try:
        return service.list_analyses(cluster_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/recommendations",
    response_model=RecommendationSet,
    tags=["recommendations"],
)
def generate_recommendations(
    cluster_id: UUID,
    engine: RecommendationEngine = Depends(get_rec_engine),
) -> RecommendationSet:
    """Generate optimization recommendations from current state and analysis data."""
    try:
        return engine.generate(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/recommendations/latest",
    response_model=RecommendationSet,
    tags=["recommendations"],
)
def latest_recommendations(
    cluster_id: UUID,
    engine: RecommendationEngine = Depends(get_rec_engine),
) -> RecommendationSet:
    """Get the latest generated recommendation set."""
    try:
        result = engine.get_latest(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="No recommendations available")
    return result


@router.get(
    "/api/v1/clusters/{cluster_id}/recommendations/list",
    response_model=list[RecommendationSet],
    tags=["recommendations"],
)
def list_recommendations(
    cluster_id: UUID,
    limit: int = 10,
    engine: RecommendationEngine = Depends(get_rec_engine),
) -> list[RecommendationSet]:
    """List historical recommendation sets for a cluster."""
    try:
        return engine.list_recommendations(cluster_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/recommendations/{rec_id}/status",
    response_model=ResourceRecommendation,
    tags=["recommendations"],
)
def update_recommendation_status(
    cluster_id: UUID,
    rec_id: UUID,
    payload: StatusUpdate,
    engine: RecommendationEngine = Depends(get_rec_engine),
) -> ResourceRecommendation:
    """Update the lifecycle status of a specific recommendation."""
    try:
        return engine.update_status(cluster_id, rec_id, payload.status.value, payload.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/recommendations/what-if",
    response_model=WhatIfProjection,
    tags=["recommendations"],
)
def what_if_recommendations(
    cluster_id: UUID,
    engine: RecommendationEngine = Depends(get_rec_engine),
) -> WhatIfProjection:
    """Simulate the projected impact of applying recommendations."""
    try:
        return engine.what_if(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/twin",
    response_model=TwinState,
    tags=["digital twin"],
)
def sync_twin(
    cluster_id: UUID,
    service: DigitalTwinService = Depends(get_digital_twin),
) -> TwinState:
    """Sync the digital twin from the latest actual cluster state."""
    try:
        return service.sync_twin(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/twin",
    response_model=TwinState,
    tags=["digital twin"],
)
def get_twin(
    cluster_id: UUID,
    service: DigitalTwinService = Depends(get_digital_twin),
) -> TwinState:
    """Get the current digital twin state."""
    try:
        twin = service.get_twin(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if twin is None:
        raise HTTPException(status_code=404, detail="No twin has been synced for this cluster")
    return twin


@router.post(
    "/api/v1/clusters/{cluster_id}/twin/compare",
    response_model=TwinComparison,
    tags=["digital twin"],
)
def compare_twin(
    cluster_id: UUID,
    service: DigitalTwinService = Depends(get_digital_twin),
) -> TwinComparison:
    """Compare the digital twin against the actual cluster state to detect drift."""
    try:
        return service.compare_twin(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/twin/apply/{rec_id}",
    response_model=TwinState,
    tags=["digital twin"],
)
def apply_to_twin(
    cluster_id: UUID,
    rec_id: UUID,
    service: DigitalTwinService = Depends(get_digital_twin),
) -> TwinState:
    """Apply a recommendation to the digital twin for what-if analysis."""
    try:
        return service.apply_recommendation(cluster_id, rec_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/twin/reset",
    response_model=TwinState,
    tags=["digital twin"],
)
def reset_twin(
    cluster_id: UUID,
    service: DigitalTwinService = Depends(get_digital_twin),
) -> TwinState:
    """Reset the digital twin to match the actual cluster state."""
    try:
        return service.reset_twin(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/scheduler/forecast",
    response_model=DemandForecast,
    tags=["scheduler"],
)
def forecast_demand(
    cluster_id: UUID,
    horizon_hours: int = Query(24, ge=1, le=168, description="Forecast horizon in hours"),
    scheduler: SchedulerService = Depends(get_scheduler_service),
) -> DemandForecast:
    """Predict future GPU resource demand for a cluster."""
    try:
        return scheduler.forecast_demand(cluster_id, horizon_hours)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/scheduler/placement",
    response_model=PlacementSuggestion,
    tags=["scheduler"],
)
def suggest_placement(
    cluster_id: UUID,
    requirements: WorkloadRequirements,
    scheduler: SchedulerService = Depends(get_scheduler_service),
) -> PlacementSuggestion:
    """Suggest an optimal node placement for a given workload."""
    try:
        return scheduler.suggest_placement(cluster_id, requirements)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/scheduler/simulate",
    response_model=ScheduleSimulation,
    tags=["scheduler"],
)
def simulate_placement(
    cluster_id: UUID,
    requirements: WorkloadRequirements,
    scheduler: SchedulerService = Depends(get_scheduler_service),
) -> ScheduleSimulation:
    """Simulate the impact of placing a workload on the cluster."""
    try:
        return scheduler.simulate_placement(cluster_id, requirements)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/scheduler/plan",
    response_model=SchedulingPlan,
    tags=["scheduler"],
)
def get_scheduling_plan(
    cluster_id: UUID,
    scheduler: SchedulerService = Depends(get_scheduler_service),
) -> SchedulingPlan:
    """Get a comprehensive scheduling plan for the cluster."""
    try:
        return scheduler.get_plan(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/actuate",
    response_model=ActuationRecord,
    tags=["actuation"],
)
def actuate_recommendation(
    cluster_id: UUID,
    payload: ActuationRequest,
    service: ActuationService = Depends(get_actuation_service),
) -> ActuationRecord:
    """Actuate (apply) a recommendation to the cluster twin."""
    try:
        return service.actuate(cluster_id, payload.rec_id, dry_run=payload.dry_run, reason=payload.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/api/v1/clusters/{cluster_id}/actuations/{actuation_id}/rollback",
    response_model=ActuationRecord,
    tags=["actuation"],
)
def rollback_actuation(
    cluster_id: UUID,
    actuation_id: UUID,
    service: ActuationService = Depends(get_actuation_service),
) -> ActuationRecord:
    """Roll back a completed actuation."""
    try:
        return service.rollback(cluster_id, actuation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/actuations/summary",
    response_model=ActuationSummary,
    tags=["actuation"],
)
def actuation_summary(
    cluster_id: UUID,
    service: ActuationService = Depends(get_actuation_service),
) -> ActuationSummary:
    """Get aggregated actuation statistics for a cluster."""
    try:
        return service.summarize(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/actuations",
    response_model=list[ActuationRecord],
    tags=["actuation"],
)
def list_actuations(
    cluster_id: UUID,
    limit: int = Query(20, ge=1, le=100),
    service: ActuationService = Depends(get_actuation_service),
) -> list[ActuationRecord]:
    """List actuation history for a cluster."""
    try:
        return service.list_actuations(cluster_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/actuations/{actuation_id}",
    response_model=ActuationRecord,
    tags=["actuation"],
)
def get_actuation(
    cluster_id: UUID,
    actuation_id: UUID,
    service: ActuationService = Depends(get_actuation_service),
) -> ActuationRecord:
    """Get details of a specific actuation."""
    try:
        record = service.get_actuation(cluster_id, actuation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Actuation not found")
    return record


@router.get(
    "/api/v1/clusters/{cluster_id}/costs/report",
    response_model=CostReport,
    tags=["cost analysis"],
)
def cluster_cost_report(
    cluster_id: UUID,
    service: CostAnalysisService = Depends(get_cost_analysis_service),
) -> CostReport:
    """Generate a detailed cost report for a cluster."""
    try:
        return service.generate_cost_report(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/costs/projections",
    response_model=SavingsProjection,
    tags=["cost analysis"],
)
def cluster_savings_projections(
    cluster_id: UUID,
    service: CostAnalysisService = Depends(get_cost_analysis_service),
) -> SavingsProjection:
    """Project potential savings from optimization recommendations."""
    try:
        return service.project_savings(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/clusters/{cluster_id}/costs/summary",
    response_model=CostSummary,
    tags=["cost analysis"],
)
def cluster_cost_summary(
    cluster_id: UUID,
    service: CostAnalysisService = Depends(get_cost_analysis_service),
) -> CostSummary:
    """Get a high-level cost summary for a cluster."""
    try:
        return service.get_cost_summary(cluster_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/v1/ml/recommendation/status", tags=["ml"])
def ml_recommendation_status(
    model: RecommendationModel = Depends(get_rec_model),
) -> dict:
    return {
        "training_count": model.get_training_count(),
        "feature_importance": model.get_feature_importance(),
    }


@router.post("/api/v1/ml/recommendation/train", tags=["ml"])
def ml_recommendation_train(
    repository: ClusterRepository = Depends(get_repository),
    model: RecommendationModel = Depends(get_rec_model),
    engine: RecommendationEngine = Depends(get_rec_engine),
) -> dict:
    trained = 0
    for cluster in repository.list_clusters():
        rec_set = repository.latest_recommendations(cluster.id)
        if rec_set is None or not rec_set.recommendations:
            continue
        state = repository.latest_state(cluster.id)
        analysis = repository.latest_analysis(cluster.id)
        for rec in rec_set.recommendations:
            outcome_map = {
                "implemented": 95.0,
                "approved": 75.0,
                "pending": 50.0,
                "dismissed": 15.0,
            }
            outcome = outcome_map.get(rec.status.value)
            if outcome is not None:
                model.train_from_feedback(rec, outcome, state, analysis)
                trained += 1
    return {"trained_samples": trained, "total_training_count": model.get_training_count()}


@router.post("/api/v1/ml/recommendation/reset", tags=["ml"])
def ml_recommendation_reset(
    model: RecommendationModel = Depends(get_rec_model),
) -> dict:
    model.reset()
    return {"status": "reset"}


@router.get("/api/v1/ml/forecast/status", tags=["ml"])
def ml_forecast_status(
    model: ForecastModel = Depends(get_forecast_model),
) -> dict:
    return {"training_count": model.get_training_count()}


@router.post("/api/v1/ml/forecast/reset", tags=["ml"])
def ml_forecast_reset(
    model: ForecastModel = Depends(get_forecast_model),
) -> dict:
    model.reset()
    return {"status": "reset"}


@router.get("/api/v1/ml/drift/status", tags=["ml"])
def ml_drift_status(
    detector: DriftDetector = Depends(get_drift_detector),
) -> dict:
    return {
        "baseline_features": list(detector.get_baseline().keys()),
        "baseline_set": len(detector.get_baseline()) > 0,
        "control_limits": detector.get_control_limits(),
    }


@router.post("/api/v1/ml/drift/reset", tags=["ml"])
def ml_drift_reset(
    detector: DriftDetector = Depends(get_drift_detector),
) -> dict:
    detector.reset()
    return {"status": "reset"}


# ── S10: DCGM Ingestion ──────────────────────────────────────────


@router.post("/api/v1/dcgm/scrape", tags=["dcgm"])
def dcgm_scrape(
    endpoint: str = "",
    pipeline: DcgmIngestionPipeline = Depends(get_dcgm_pipeline),
) -> dict:
    discovery = pipeline.scrape(endpoint if endpoint else None)
    return {
        "endpoint": discovery.endpoint,
        "available_metrics": [m.value for m in discovery.available_metrics],
        "sample_count": discovery.sample_count,
        "gpu_count": discovery.gpu_count,
        "gpu_uuids": discovery.gpu_uuids,
        "scrape_duration_ms": discovery.scrape_duration_ms,
        "last_scrape": discovery.last_scrape,
    }


@router.post("/api/v1/dcgm/poll/start", tags=["dcgm"])
def dcgm_poll_start(
    interval: int = 15,
    endpoint: str = "",
    pipeline: DcgmIngestionPipeline = Depends(get_dcgm_pipeline),
) -> dict:
    if endpoint:
        pipeline.set_endpoint(endpoint)
    pipeline.start_polling(interval)
    return {"status": "started", "interval": interval, "endpoint": pipeline._endpoint}


@router.post("/api/v1/dcgm/poll/stop", tags=["dcgm"])
def dcgm_poll_stop(
    pipeline: DcgmIngestionPipeline = Depends(get_dcgm_pipeline),
) -> dict:
    pipeline.stop_polling()
    return {"status": "stopped"}


@router.get("/api/v1/dcgm/samples", tags=["dcgm"])
def dcgm_samples(
    pipeline: DcgmIngestionPipeline = Depends(get_dcgm_pipeline),
) -> list[dict]:
    samples = pipeline.get_latest_samples()
    return [
        {
            "metric": s.metric.value,
            "gpu_index": s.gpu_index,
            "gpu_uuid": s.gpu_uuid,
            "hostname": s.hostname,
            "value": s.value,
            "timestamp": s.timestamp,
            "labels": s.labels,
        }
        for s in samples
    ]


@router.get("/api/v1/dcgm/quality/{cluster_id}", tags=["dcgm"])
def dcgm_quality(
    cluster_id: str = "default",
    pipeline: DcgmIngestionPipeline = Depends(get_dcgm_pipeline),
) -> dict:
    analyzer = DcgmQualityAnalyzer(pipeline)
    return analyzer.analyze_cluster(cluster_id).__dict__


@router.get("/api/v1/dcgm/status", tags=["dcgm"])
def dcgm_status(
    pipeline: DcgmIngestionPipeline = Depends(get_dcgm_pipeline),
) -> dict:
    samples = pipeline.get_latest_samples()
    metrics = set(s.metric.value for s in samples)
    gpus = set(s.gpu_uuid for s in samples)
    return {
        "running": pipeline.is_running,
        "endpoint": pipeline._endpoint,
        "sample_count": len(samples),
        "gpu_count": len(gpus),
        "available_metrics": sorted(metrics),
        "last_scrape": max((s.timestamp for s in samples), default=""),
    }


# ── Training Optimization & Slurm (S15-S18) ──────────────────


@router.post("/api/v1/training/jobs", tags=["training"], status_code=201)
def training_register_job(
    cluster_id: UUID,
    job_name: str,
    framework: TrainingFramework = TrainingFramework.CUSTOM,
    gpu_count: int = 1,
    node_count: int = 1,
    batch_size: int = 0,
    precision: str = "fp32",
    max_duration_hours: float = 0.0,
    metadata: str = "{}",
    training: TrainingService = Depends(get_training_service),
) -> TrainingJob:
    import json
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        meta = {}
    return training.register_job(
        cluster_id=cluster_id,
        job_name=job_name,
        framework=framework,
        gpu_count=gpu_count,
        node_count=node_count,
        batch_size=batch_size,
        precision=precision,
        max_duration_hours=max_duration_hours,
        metadata=meta,
    )


@router.get("/api/v1/training/jobs", tags=["training"])
def training_list_jobs(
    cluster_id: UUID | None = None,
    training: TrainingService = Depends(get_training_service),
) -> list[TrainingJob]:
    return training.list_jobs(cluster_id)


@router.get("/api/v1/training/jobs/{job_id}", tags=["training"])
def training_get_job(
    job_id: UUID,
    training: TrainingService = Depends(get_training_service),
) -> TrainingJob:
    job = training.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Training job not found: {job_id}")
    return job


@router.patch("/api/v1/training/jobs/{job_id}", tags=["training"])
def training_update_job(
    job_id: UUID,
    status: str | None = None,
    loss_value: float | None = None,
    epochs_completed: int | None = None,
    elapsed_hours: float | None = None,
    avg_gpu_utilization: float | None = None,
    peak_gpu_memory_gib: float | None = None,
    throughput_samples_per_sec: float | None = None,
    training: TrainingService = Depends(get_training_service),
) -> TrainingJob:
    status_enum = None
    if status:
        from .schemas import TrainingJobStatus
        status_enum = TrainingJobStatus(status)
    job = training.update_job(
        job_id=job_id,
        status=status_enum,
        loss_value=loss_value,
        epochs_completed=epochs_completed,
        elapsed_hours=elapsed_hours,
        avg_gpu_utilization=avg_gpu_utilization,
        peak_gpu_memory_gib=peak_gpu_memory_gib,
        throughput_samples_per_sec=throughput_samples_per_sec,
    )
    if job is None:
        raise HTTPException(404, f"Training job not found: {job_id}")
    return job


@router.delete("/api/v1/training/jobs/{job_id}", tags=["training"])
def training_delete_job(
    job_id: UUID,
    training: TrainingService = Depends(get_training_service),
) -> dict:
    if not training.delete_job(job_id):
        raise HTTPException(404, f"Training job not found: {job_id}")
    return {"status": "deleted"}


@router.post("/api/v1/training/jobs/{job_id}/profile", tags=["training"])
def training_profile_job(
    job_id: UUID,
    training: TrainingService = Depends(get_training_service),
) -> TrainingProfile:
    profile = training.profile_job(job_id)
    if profile is None:
        raise HTTPException(404, f"Training job not found: {job_id}")
    return profile


@router.post("/api/v1/training/jobs/{job_id}/hpo", tags=["training"])
def training_run_hpo(
    job_id: UUID,
    config: HPOConfig | None = None,
    training: TrainingService = Depends(get_training_service),
) -> HTOResult:
    result = training.run_hpo(job_id, config)
    if result is None:
        raise HTTPException(404, f"Training job not found: {job_id}")
    return result


@router.post("/api/v1/training/distributed-config", tags=["training"])
def training_distributed_config(
    total_gpus: int,
    gpu_model: str = "",
    model_size_gb: float = 0.0,
    per_gpu_memory_gb: float = 80.0,
) -> DistributedTrainingConfig:
    return TrainingService.suggest_distributed_config(
        total_gpus=total_gpus,
        gpu_model=gpu_model,
        model_size_gb=model_size_gb,
        per_gpu_memory_gb=per_gpu_memory_gb,
    )


@router.get("/api/v1/slurm/telemetry/{cluster_id}", tags=["slurm"])
def slurm_telemetry(
    cluster_id: UUID,
    repository: ClusterRepository = Depends(get_repository),
) -> SlurmClusterTelemetry:
    cluster = repository.get_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(404, f"Cluster not found: {cluster_id}")
    from .connectors.factory import build_connector
    from .connectors.slurm import SlurmConnector
    connector = build_connector(cluster)
    if isinstance(connector, SlurmConnector):
        return connector.collect_slurm_telemetry()
    # For mock/non-Slurm clusters, return synthetic telemetry
    state = repository.latest_state(cluster_id)
    from datetime import datetime, timezone
    import random
    node_count = state.node_count if state else 4
    gpu_count = state.gpu_count if state else 8
    gpus_per_node = max(gpu_count // max(node_count, 1), 1)
    partitions_list = [
        SlurmPartitionInfo(
            name="gpu",
            state="up",
            nodes=[f"node-{i}" for i in range(node_count)],
            total_cpus=node_count * 64,
            total_gpus=gpu_count,
            default_time_minutes=60,
            max_time_minutes=43200,
        )
    ]
    mock_nodes = []
    total_alloc_gpus = 0
    for i in range(node_count):
        alloc = random.randint(0, gpus_per_node)
        total_alloc_gpus += alloc
        mock_nodes.append(SlurmNodeInfo(
            node_name=f"node-{i}",
            state="idle" if alloc == 0 else "alloc",
            partitions=["gpu"],
            cpu_count=64,
            memory_mb=1024 * 1024,
            gpu_count=gpus_per_node,
            gpu_model="NVIDIA A100 80GB",
            features=["gpu", "ib"],
            weight=1,
        ))
    running_jobs_list = [
        SlurmJobInfo(
            job_id=1000 + i,
            job_name=f"job-{i}",
            partition="gpu",
            user="gpuopt",
            state="RUNNING",
            node_count=1,
            gpu_count=gpus_per_node // 2,
            cpus=8,
            memory_mb=64 * 1024,
            time_limit_minutes=1440,
            time_used_minutes=random.randint(10, 1200),
            nodes="node-0" if i == 0 else f"node-{i}",
            submit_time=datetime.now(timezone.utc),
            start_time=datetime.now(timezone.utc),
        )
        for i in range(min(3, node_count))
    ]
    return SlurmClusterTelemetry(
        cluster_id=cluster_id,
        cluster_name=cluster.name if hasattr(cluster, 'name') else f"slurm-{cluster_id}",
        collected_at=datetime.now(timezone.utc),
        controller_status="UP",
        node_count=node_count,
        gpu_count=gpu_count,
        nodes=mock_nodes,
        partitions=partitions_list,
        running_jobs=running_jobs_list,
        pending_jobs=[],
        total_cpus=node_count * 64,
        allocated_cpus=random.randint(10, node_count * 60),
        total_memory_mb=node_count * 1024 * 1024,
        allocated_memory_mb=random.randint(100*1024, node_count * 500*1024),
        total_gpus_allocated=total_alloc_gpus,
    )


@router.get("/api/v1/slurm/topology/{cluster_id}", tags=["slurm"])
def slurm_topology(
    cluster_id: UUID,
    repository: ClusterRepository = Depends(get_repository),
) -> NodeTopology:
    cluster = repository.get_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(404, f"Cluster not found: {cluster_id}")
    from .connectors.factory import build_connector
    from .connectors.slurm import SlurmConnector
    connector = build_connector(cluster)
    if isinstance(connector, SlurmConnector):
        return connector.get_cluster_topology()
    # Return synthetic topology for mock clusters
    state = repository.latest_state(cluster_id)
    node_count = state.node_count if state else 4
    gpu_count = state.gpu_count if state else 8
    return NodeTopology(
        nodes=f"node-[0-{node_count-1}]",
        partitions="gpu",
        topology=f"switches: 1, nodes: {node_count}, gpus: {gpu_count}",
        network="InfiniBand 200Gb/s",
        fs="lustre",
    )


@router.get("/api/v1/slurm/monitor/snapshot/{cluster_id}", tags=["slurm"])
def slurm_monitor_snapshot(
    cluster_id: UUID,
    repository: ClusterRepository = Depends(get_repository),
) -> MonitoringSnapshot:
    cluster = repository.get_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(404, f"Cluster not found: {cluster_id}")
    from .connectors.factory import build_connector
    from .connectors.slurm import SlurmConnector
    connector = build_connector(cluster)
    if isinstance(connector, SlurmConnector):
        return connector.collect_monitoring_snapshot()
    # Return synthetic snapshot for mock clusters
    state = repository.latest_state(cluster_id)
    node_count = state.node_count if state else 4
    gpu_count = state.gpu_count if state else 8
    import random
    return MonitoringSnapshot(
        cluster_id=cluster_id,
        node_count=node_count,
        gpu_count=gpu_count,
        cpu_util_pct=random.uniform(30, 80),
        gpu_util_pct=random.uniform(20, 90),
        memory_util_pct=random.uniform(40, 85),
        power_watts=random.uniform(2000, 8000),
        temp_celsius=random.uniform(40, 70),
        running_jobs=random.randint(2, 12),
        pending_jobs=random.randint(0, 5),
        collected_at=datetime.now(timezone.utc),
    )


@router.post("/api/v1/slurm/monitor/start/{cluster_id}/{job_id}", tags=["slurm"])
def slurm_monitor_start(
    cluster_id: UUID,
    job_id: int,
    config: JobMonitorConfig | None = None,
    repository: ClusterRepository = Depends(get_repository),
) -> dict:
    cluster = repository.get_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(404, f"Cluster not found: {cluster_id}")
    from .connectors.factory import build_connector
    from .connectors.slurm import SlurmConnector
    connector = build_connector(cluster)
    if isinstance(connector, SlurmConnector):
        connector.start_job_monitor(job_id, config)
    return {"status": "started", "job_id": job_id, "mode": "mock" if not isinstance(connector, SlurmConnector) else "slurm"}


@router.post("/api/v1/slurm/monitor/stop/{cluster_id}/{job_id}", tags=["slurm"])
def slurm_monitor_stop(
    cluster_id: UUID,
    job_id: int,
    repository: ClusterRepository = Depends(get_repository),
) -> dict:
    cluster = repository.get_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(404, f"Cluster not found: {cluster_id}")
    from .connectors.factory import build_connector
    from .connectors.slurm import SlurmConnector
    connector = build_connector(cluster)
    if isinstance(connector, SlurmConnector):
        connector.stop_job_monitor(job_id)
    return {"status": "stopped", "job_id": job_id}


@router.get("/api/v1/slurm/monitor/history/{cluster_id}/{job_id}", tags=["slurm"])
def slurm_monitor_history(
    cluster_id: UUID,
    job_id: int,
    repository: ClusterRepository = Depends(get_repository),
) -> list:
    cluster = repository.get_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(404, f"Cluster not found: {cluster_id}")
    from .connectors.factory import build_connector
    from .connectors.slurm import SlurmConnector
    connector = build_connector(cluster)
    if isinstance(connector, SlurmConnector):
        history = connector.get_job_history(job_id)
        return [h.model_dump(mode="json") for h in history]
    return []


# ── Inference Optimization (S19) ────────────────────────────


@router.post("/api/v1/inference/endpoints", tags=["inference"], status_code=201)
def inference_register_endpoint(
    cluster_id: UUID,
    endpoint_name: str,
    model_name: str,
    framework: InferenceFramework = InferenceFramework.CUSTOM,
    gpu_count: int = 1,
    gpu_model: str = "",
    quantisation: str = "fp16",
    max_batch_size: int = 1,
    max_input_tokens: int = 4096,
    max_output_tokens: int = 1024,
    concurrency: int = 1,
    model_version: str = "latest",
    metadata: str = "{}",
    inference: InferenceService = Depends(get_inference_service),
) -> InferenceEndpoint:
    import json
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        meta = {}
    return inference.register_endpoint(
        cluster_id=cluster_id,
        endpoint_name=endpoint_name,
        model_name=model_name,
        framework=framework,
        gpu_count=gpu_count,
        gpu_model=gpu_model,
        quantisation=quantisation,
        max_batch_size=max_batch_size,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        concurrency=concurrency,
        model_version=model_version,
        metadata=meta,
    )


@router.get("/api/v1/inference/endpoints", tags=["inference"])
def inference_list_endpoints(
    cluster_id: UUID | None = None,
    inference: InferenceService = Depends(get_inference_service),
) -> list[InferenceEndpoint]:
    return inference.list_endpoints(cluster_id)


@router.get("/api/v1/inference/endpoints/{endpoint_id}", tags=["inference"])
def inference_get_endpoint(
    endpoint_id: UUID,
    inference: InferenceService = Depends(get_inference_service),
) -> InferenceEndpoint:
    ep = inference.get_endpoint(endpoint_id)
    if ep is None:
        raise HTTPException(404, f"Inference endpoint not found: {endpoint_id}")
    return ep


@router.patch("/api/v1/inference/endpoints/{endpoint_id}", tags=["inference"])
def inference_update_endpoint(
    endpoint_id: UUID,
    status: str | None = None,
    avg_latency_ms: float | None = None,
    p99_latency_ms: float | None = None,
    throughput_requests_per_sec: float | None = None,
    throughput_tokens_per_sec: float | None = None,
    avg_gpu_utilization: float | None = None,
    peak_gpu_memory_gib: float | None = None,
    kv_cache_utilization: float | None = None,
    cost_per_1k_tokens: float | None = None,
    inference: InferenceService = Depends(get_inference_service),
) -> InferenceEndpoint:
    status_enum = None
    if status:
        from .schemas import InferenceEndpointStatus
        status_enum = InferenceEndpointStatus(status)
    ep = inference.update_endpoint(
        endpoint_id=endpoint_id,
        status=status_enum,
        avg_latency_ms=avg_latency_ms,
        p99_latency_ms=p99_latency_ms,
        throughput_requests_per_sec=throughput_requests_per_sec,
        throughput_tokens_per_sec=throughput_tokens_per_sec,
        avg_gpu_utilization=avg_gpu_utilization,
        peak_gpu_memory_gib=peak_gpu_memory_gib,
        kv_cache_utilization=kv_cache_utilization,
        cost_per_1k_tokens=cost_per_1k_tokens,
    )
    if ep is None:
        raise HTTPException(404, f"Inference endpoint not found: {endpoint_id}")
    return ep


@router.delete("/api/v1/inference/endpoints/{endpoint_id}", tags=["inference"])
def inference_delete_endpoint(
    endpoint_id: UUID,
    inference: InferenceService = Depends(get_inference_service),
) -> dict:
    if not inference.delete_endpoint(endpoint_id):
        raise HTTPException(404, f"Inference endpoint not found: {endpoint_id}")
    return {"status": "deleted"}


@router.post("/api/v1/inference/endpoints/{endpoint_id}/profile", tags=["inference"])
def inference_profile_endpoint(
    endpoint_id: UUID,
    inference: InferenceService = Depends(get_inference_service),
) -> InferenceProfile:
    profile = inference.profile_endpoint(endpoint_id)
    if profile is None:
        raise HTTPException(404, f"Inference endpoint not found: {endpoint_id}")
    return profile


@router.post("/api/v1/inference/deployment-config", tags=["inference"])
def inference_deployment_config(
    model_name: str = "",
    model_size_gb: float = 0.0,
    context_length: int = 4096,
    target_latency_ms: float = 200.0,
    expected_requests_per_sec: float = 10.0,
    gpu_budget: str = "",
) -> InferenceDeploymentConfig:
    return InferenceService.suggest_deployment_config(
        model_name=model_name,
        model_size_gb=model_size_gb,
        context_length=context_length,
        target_latency_ms=target_latency_ms,
        expected_requests_per_sec=expected_requests_per_sec,
        gpu_budget=gpu_budget,
    )


# ── FinOps (S20) ────────────────────────────────────────────


@router.get("/api/v1/finops/pricing", tags=["finops"])
def finops_pricing(
    gpu_model: str = "",
    provider: str = "",
    region: str = "",
    tier: str = "",
) -> list:
    provider_enum = CloudProvider(provider) if provider else None
    tier_enum = GpuPricingTier(tier) if tier else None
    return [r.model_dump(mode="json") for r in FinOpsService.get_pricing(
        gpu_model=gpu_model,
        provider=provider_enum,
        region=region,
        tier=tier_enum,
    )]


@router.get("/api/v1/finops/compare", tags=["finops"])
def finops_compare(
    gpu_model: str = "h100",
    gpu_count: int = 8,
) -> ProviderCostComparison:
    return FinOpsService.compare_providers(
        gpu_model=gpu_model,
        gpu_count=gpu_count,
    )


@router.get("/api/v1/finops/spot-savings/{cluster_id}", tags=["finops"])
def finops_spot_savings(
    cluster_id: UUID,
    finops: FinOpsService = Depends(get_finops_service),
) -> SpotSavingsAnalysis:
    return finops.analyze_spot_savings(cluster_id)


@router.get("/api/v1/finops/reserved-recs/{cluster_id}", tags=["finops"])
def finops_reserved_recs(
    cluster_id: UUID,
    finops: FinOpsService = Depends(get_finops_service),
) -> ReservedInstanceRecommendation:
    return finops.recommend_reserved_instances(cluster_id)


@router.get("/api/v1/finops/budget/{cluster_id}", tags=["finops"])
def finops_budget(
    cluster_id: UUID,
    monthly_budget: float = 0.0,
    finops: FinOpsService = Depends(get_finops_service),
) -> BudgetAlert:
    return finops.get_budget_alert(cluster_id, monthly_budget)


@router.get("/api/v1/finops/aggregate", tags=["finops"])
def finops_aggregate(
    finops: FinOpsService = Depends(get_finops_service),
) -> MultiClusterCostSummary:
    return finops.aggregate_costs()


@router.get("/api/v1/finops/forecast/{cluster_id}", tags=["finops"])
def finops_forecast(
    cluster_id: UUID,
    months: int = 12,
    growth_rate: float = 0.05,
    finops: FinOpsService = Depends(get_finops_service),
) -> CostForecast:
    return finops.forecast_cost(cluster_id, months=months, growth_rate=growth_rate)


@router.post("/api/v1/finops/what-if", tags=["finops"])
def finops_what_if(
    scenario_name: str,
    description: str = "",
    current_monthly_cost: float = 0.0,
    gpu_count_change: int = 0,
    utilization_change: float = 0.0,
    provider: str = "",
    tier: str = "",
) -> WhatIfCostScenario:
    provider_enum = CloudProvider(provider) if provider else None
    tier_enum = GpuPricingTier(tier) if tier else None
    return FinOpsService.what_if_cost(
        scenario_name=scenario_name,
        description=description,
        current_monthly_cost=current_monthly_cost,
        gpu_count_change=gpu_count_change,
        utilization_change=utilization_change,
        provider_change=provider_enum,
        tier_change=tier_enum,
    )


@router.get("/api/v1/finops/allocation/{cluster_id}", tags=["finops"])
def finops_allocation(
    cluster_id: UUID,
    finops: FinOpsService = Depends(get_finops_service),
) -> list[CostAllocationTag]:
    return finops.get_cost_allocation(cluster_id)


@router.get("/api/v1/finops/recommendations/{cluster_id}", tags=["finops"])
def finops_recommendations(
    cluster_id: UUID,
    finops: FinOpsService = Depends(get_finops_service),
) -> list[ResourceRecommendation]:
    return finops.generate_finops_recommendations(cluster_id)


# ── Power Optimization (S21) ────────────────────────────────


@router.get("/api/v1/power/profiles", tags=["power"])
def power_profiles() -> list:
    return PowerService.list_power_profiles()


@router.get("/api/v1/power/profile/{gpu_model}", tags=["power"])
def power_profile(gpu_model: str) -> dict:
    profile = PowerService.get_power_profile(gpu_model)
    if profile is None:
        raise HTTPException(404, f"Power profile not found: {gpu_model}")
    return profile


@router.get("/api/v1/power/analysis/{cluster_id}", tags=["power"])
def power_analysis(
    cluster_id: UUID,
    power: PowerService = Depends(get_power_service),
) -> PowerAnalysisResult:
    return power.analyze_power(cluster_id)


@router.get("/api/v1/power/carbon/{cluster_id}", tags=["power"])
def power_carbon(
    cluster_id: UUID,
    power: PowerService = Depends(get_power_service),
) -> dict:
    return power.estimate_carbon(cluster_id).model_dump(mode="json")


@router.get("/api/v1/power/cap-suggestion", tags=["power"])
def power_cap_suggestion(
    gpu_model: str = "a100",
    gpu_count: int = 8,
    current_power_watts: float = 0.0,
) -> PowerCapSuggestion:
    return PowerService.suggest_power_cap(
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        current_power_watts=current_power_watts,
    )


@router.get("/api/v1/power/recommendations/{cluster_id}", tags=["power"])
def power_recommendations(
    cluster_id: UUID,
    power: PowerService = Depends(get_power_service),
) -> list[ResourceRecommendation]:
    return power.generate_power_recommendations(cluster_id)


# ── Guarded Automation (S22) ────────────────────────────────


@router.get("/api/v1/guarded/policies", tags=["guarded automation"])
def ga_list_policies(
    engine: PolicyEngine = Depends(get_policy_engine),
) -> list[PolicyRule]:
    return engine.list_policies()


@router.get("/api/v1/guarded/policies/{policy_id}", tags=["guarded automation"])
def ga_get_policy(
    policy_id: UUID,
    engine: PolicyEngine = Depends(get_policy_engine),
) -> PolicyRule:
    policy = engine.get_policy(policy_id)
    if policy is None:
        raise HTTPException(404, f"Policy not found: {policy_id}")
    return policy


@router.post("/api/v1/guarded/policies", tags=["guarded automation"], status_code=201)
def ga_create_policy(
    payload: PolicyRule,
    engine: PolicyEngine = Depends(get_policy_engine),
) -> PolicyRule:
    return engine.create_policy(payload)


@router.patch("/api/v1/guarded/policies/{policy_id}", tags=["guarded automation"])
def ga_update_policy(
    policy_id: UUID,
    payload: dict,
    engine: PolicyEngine = Depends(get_policy_engine),
) -> PolicyRule:
    policy = engine.update_policy(policy_id, payload)
    if policy is None:
        raise HTTPException(404, f"Policy not found: {policy_id}")
    return policy


@router.delete("/api/v1/guarded/policies/{policy_id}", tags=["guarded automation"], status_code=204)
def ga_delete_policy(
    policy_id: UUID,
    engine: PolicyEngine = Depends(get_policy_engine),
) -> Response:
    if not engine.delete_policy(policy_id):
        raise HTTPException(404, f"Policy not found: {policy_id}")
    return Response(status_code=204)


@router.post("/api/v1/guarded/pre-flight/{cluster_id}/{rec_id}", tags=["guarded automation"])
def ga_pre_flight(
    cluster_id: UUID,
    rec_id: UUID,
    environment: str = "",
    engine: PolicyEngine = Depends(get_policy_engine),
) -> PreFlightCheckResult:
    return engine.check_pre_flight(cluster_id, rec_id, environment)


@router.post("/api/v1/guarded/approvals", tags=["guarded automation"], status_code=201)
def ga_create_approval(
    payload: ApprovalWorkflowRequest,
    workflow: ApprovalWorkflow = Depends(get_approval_workflow),
) -> ApprovalRecord:
    try:
        return workflow.create_request(payload)
    except KeyError as exc:
        raise HTTPException(404, detail=str(exc))


@router.get("/api/v1/guarded/approvals", tags=["guarded automation"])
def ga_list_approvals(
    cluster_id: UUID | None = None,
    workflow: ApprovalWorkflow = Depends(get_approval_workflow),
) -> list[ApprovalRecord]:
    return workflow.list_approvals(cluster_id)


@router.get("/api/v1/guarded/approvals/{approval_id}", tags=["guarded automation"])
def ga_get_approval(
    approval_id: UUID,
    workflow: ApprovalWorkflow = Depends(get_approval_workflow),
) -> ApprovalRecord:
    record = workflow.get_approval(approval_id)
    if record is None:
        raise HTTPException(404, f"Approval not found: {approval_id}")
    return record


@router.post("/api/v1/guarded/approvals/{approval_id}/approve", tags=["guarded automation"])
def ga_approve(
    approval_id: UUID,
    approver: str,
    reason: str = "",
    workflow: ApprovalWorkflow = Depends(get_approval_workflow),
) -> ApprovalRecord:
    try:
        return workflow.approve(approval_id, approver, reason)
    except KeyError as exc:
        raise HTTPException(404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(409, detail=str(exc))


@router.post("/api/v1/guarded/approvals/{approval_id}/reject", tags=["guarded automation"])
def ga_reject(
    approval_id: UUID,
    approver: str,
    reason: str = "",
    workflow: ApprovalWorkflow = Depends(get_approval_workflow),
) -> ApprovalRecord:
    try:
        return workflow.reject(approval_id, approver, reason)
    except KeyError as exc:
        raise HTTPException(404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(409, detail=str(exc))


@router.post("/api/v1/guarded/chaos-experiments", tags=["guarded automation"], status_code=201)
def ga_create_experiment(
    payload: ChaosExperiment,
    engine: ChaosEngine = Depends(get_chaos_engine),
) -> ChaosExperiment:
    return engine.create_experiment(payload)


@router.get("/api/v1/guarded/chaos-experiments", tags=["guarded automation"])
def ga_list_experiments(
    cluster_id: UUID | None = None,
    engine: ChaosEngine = Depends(get_chaos_engine),
) -> list[ChaosExperiment]:
    return engine.list_experiments(cluster_id)


@router.get("/api/v1/guarded/chaos-experiments/{experiment_id}", tags=["guarded automation"])
def ga_get_experiment(
    experiment_id: UUID,
    engine: ChaosEngine = Depends(get_chaos_engine),
) -> ChaosExperiment:
    experiment = engine.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(404, f"Chaos experiment not found: {experiment_id}")
    return experiment


@router.post("/api/v1/guarded/chaos-experiments/{experiment_id}/run", tags=["guarded automation"])
def ga_run_experiment(
    experiment_id: UUID,
    engine: ChaosEngine = Depends(get_chaos_engine),
) -> ChaosExperimentResult:
    try:
        return engine.run_experiment(experiment_id)
    except KeyError as exc:
        raise HTTPException(404, detail=str(exc))


@router.delete("/api/v1/guarded/chaos-experiments/{experiment_id}", tags=["guarded automation"], status_code=204)
def ga_delete_experiment(
    experiment_id: UUID,
    engine: ChaosEngine = Depends(get_chaos_engine),
) -> Response:
    if not engine.delete_experiment(experiment_id):
        raise HTTPException(404, f"Chaos experiment not found: {experiment_id}")
    return Response(status_code=204)


@router.get("/api/v1/guarded/recommendations/{cluster_id}", tags=["guarded automation"])
def ga_recommendations(
    cluster_id: UUID,
    engine: ChaosEngine = Depends(get_chaos_engine),
) -> list[GuardedAutomationRecommendation]:
    return engine.generate_ga_recommendations(cluster_id)


# ── S23: Observability & Alerting ──────────────────────────────


@router.post("/api/v1/alerts/rules", tags=["observability"], status_code=201)
def alert_create_rule(
    payload: AlertRule,
    manager: AlertManager = Depends(get_alert_manager),
) -> AlertRule:
    return manager.create_rule(payload)


@router.get("/api/v1/alerts/rules", tags=["observability"])
def alert_list_rules(
    cluster_id: UUID | None = None,
    manager: AlertManager = Depends(get_alert_manager),
) -> list[AlertRule]:
    return manager.list_rules(cluster_id)


@router.get("/api/v1/alerts/rules/{rule_id}", tags=["observability"])
def alert_get_rule(
    rule_id: UUID,
    manager: AlertManager = Depends(get_alert_manager),
) -> AlertRule:
    rule = manager.get_rule(rule_id)
    if rule is None:
        raise HTTPException(404, "Alert rule not found")
    return rule


@router.patch("/api/v1/alerts/rules/{rule_id}", tags=["observability"])
def alert_update_rule(
    rule_id: UUID,
    payload: dict,
    manager: AlertManager = Depends(get_alert_manager),
) -> AlertRule:
    rule = manager.update_rule(rule_id, payload)
    if rule is None:
        raise HTTPException(404, "Alert rule not found")
    return rule


@router.delete("/api/v1/alerts/rules/{rule_id}", tags=["observability"], status_code=204)
def alert_delete_rule(
    rule_id: UUID,
    manager: AlertManager = Depends(get_alert_manager),
) -> Response:
    if not manager.delete_rule(rule_id):
        raise HTTPException(404, "Alert rule not found")
    return Response(status_code=204)


@router.post("/api/v1/alerts/evaluate/{cluster_id}", tags=["observability"])
def alert_evaluate(
    cluster_id: UUID,
    manager: AlertManager = Depends(get_alert_manager),
) -> list[AlertRuleEvaluation]:
    return manager.evaluate_rules(cluster_id)


@router.get("/api/v1/alerts", tags=["observability"])
def alert_list(
    cluster_id: UUID | None = None,
    status: str = "",
    manager: AlertManager = Depends(get_alert_manager),
) -> list[AlertRecord]:
    return manager.list_alerts(cluster_id, status)


@router.post("/api/v1/alerts/{alert_id}/acknowledge", tags=["observability"])
def alert_acknowledge(
    alert_id: UUID,
    user: str = "",
    manager: AlertManager = Depends(get_alert_manager),
) -> AlertRecord:
    alert = manager.acknowledge_alert(alert_id, user)
    if alert is None:
        raise HTTPException(404, "Alert not found")
    return alert


@router.post("/api/v1/alerts/{alert_id}/resolve", tags=["observability"])
def alert_resolve(
    alert_id: UUID,
    manager: AlertManager = Depends(get_alert_manager),
) -> AlertRecord:
    alert = manager.resolve_alert(alert_id)
    if alert is None:
        raise HTTPException(404, "Alert not found")
    return alert


@router.post("/api/v1/notifications/channels", tags=["observability"], status_code=201)
def notification_create_channel(
    payload: NotificationChannel,
    manager: AlertManager = Depends(get_alert_manager),
) -> NotificationChannel:
    return manager.create_channel(payload)


@router.get("/api/v1/notifications/channels", tags=["observability"])
def notification_list_channels(
    manager: AlertManager = Depends(get_alert_manager),
) -> list[NotificationChannel]:
    return manager.list_channels()


@router.get("/api/v1/notifications/channels/{channel_id}", tags=["observability"])
def notification_get_channel(
    channel_id: UUID,
    manager: AlertManager = Depends(get_alert_manager),
) -> NotificationChannel:
    channel = manager.get_channel(channel_id)
    if channel is None:
        raise HTTPException(404, "Notification channel not found")
    return channel


@router.patch("/api/v1/notifications/channels/{channel_id}", tags=["observability"])
def notification_update_channel(
    channel_id: UUID,
    payload: dict,
    manager: AlertManager = Depends(get_alert_manager),
) -> NotificationChannel:
    channel = manager.update_channel(channel_id, payload)
    if channel is None:
        raise HTTPException(404, "Notification channel not found")
    return channel


@router.delete("/api/v1/notifications/channels/{channel_id}", tags=["observability"], status_code=204)
def notification_delete_channel(
    channel_id: UUID,
    manager: AlertManager = Depends(get_alert_manager),
) -> Response:
    if not manager.delete_channel(channel_id):
        raise HTTPException(404, "Notification channel not found")
    return Response(status_code=204)


@router.post("/api/v1/notifications/channels/{channel_id}/test", tags=["observability"])
def notification_test_channel(
    channel_id: UUID,
    manager: AlertManager = Depends(get_alert_manager),
) -> NotificationMessage:
    try:
        return manager.send_test_message(channel_id)
    except KeyError as exc:
        raise HTTPException(404, detail=str(exc))


@router.get("/api/v1/notifications/messages", tags=["observability"])
def notification_list_messages(
    channel_id: UUID | None = None,
    manager: AlertManager = Depends(get_alert_manager),
) -> list[NotificationMessage]:
    return manager.list_messages(channel_id)


# ── S23: Multi-Tenancy & RBAC ───────────────────────────────────


@router.post("/api/v1/tenants/teams", tags=["multi-tenancy"], status_code=201)
def tenant_create_team(
    payload: Team,
    manager: TenantManager = Depends(get_tenant_manager),
) -> Team:
    return manager.create_team(payload)


@router.get("/api/v1/tenants/teams", tags=["multi-tenancy"])
def tenant_list_teams(
    manager: TenantManager = Depends(get_tenant_manager),
) -> list[Team]:
    return manager.list_teams()


@router.get("/api/v1/tenants/teams/{team_id}", tags=["multi-tenancy"])
def tenant_get_team(
    team_id: UUID,
    manager: TenantManager = Depends(get_tenant_manager),
) -> Team:
    team = manager.get_team(team_id)
    if team is None:
        raise HTTPException(404, "Team not found")
    return team


@router.delete("/api/v1/tenants/teams/{team_id}", tags=["multi-tenancy"], status_code=204)
def tenant_delete_team(
    team_id: UUID,
    manager: TenantManager = Depends(get_tenant_manager),
) -> Response:
    if not manager.delete_team(team_id):
        raise HTTPException(404, "Team not found")
    return Response(status_code=204)


@router.post("/api/v1/tenants/projects", tags=["multi-tenancy"], status_code=201)
def tenant_create_project(
    payload: Project,
    manager: TenantManager = Depends(get_tenant_manager),
) -> Project:
    return manager.create_project(payload)


@router.get("/api/v1/tenants/projects", tags=["multi-tenancy"])
def tenant_list_projects(
    team_id: UUID | None = None,
    manager: TenantManager = Depends(get_tenant_manager),
) -> list[Project]:
    return manager.list_projects(team_id)


@router.get("/api/v1/tenants/projects/{project_id}", tags=["multi-tenancy"])
def tenant_get_project(
    project_id: UUID,
    manager: TenantManager = Depends(get_tenant_manager),
) -> Project:
    project = manager.get_project(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


@router.delete("/api/v1/tenants/projects/{project_id}", tags=["multi-tenancy"], status_code=204)
def tenant_delete_project(
    project_id: UUID,
    manager: TenantManager = Depends(get_tenant_manager),
) -> Response:
    if not manager.delete_project(project_id):
        raise HTTPException(404, "Project not found")
    return Response(status_code=204)


@router.get("/api/v1/tenants/projects/{project_id}/quota", tags=["multi-tenancy"])
def tenant_get_quota(
    project_id: UUID,
    manager: TenantManager = Depends(get_tenant_manager),
) -> ResourceQuota:
    return manager.get_quota(project_id)


# ── S11: Workload Attribution ────────────────────────────────────


@router.post("/api/v1/attribution/pods", tags=["workload attribution"], status_code=201)
def attribution_register_pod(
    payload: dict,
    engine: WorkloadAttributionEngine = Depends(get_attribution_engine),
) -> dict:
    ownership = engine.register_pod(
        pod_name=payload.get("pod_name", ""),
        namespace=payload.get("namespace", "default"),
        owner_kind=payload.get("owner_kind", "Job"),
        owner_name=payload.get("owner_name", ""),
        owner_uid=payload.get("owner_uid", ""),
        labels=payload.get("labels"),
        tenant_id=payload.get("tenant_id", ""),
        project_id=payload.get("project_id", ""),
        user_id=payload.get("user_id", ""),
    )
    return {
        "pod_name": ownership.pod_name,
        "resource_owner": ownership.resource_owner.value,
        "tenant_id": ownership.tenant_id,
    }


@router.post("/api/v1/attribution/gpu-allocations", tags=["workload attribution"], status_code=201)
def attribution_record_gpu(
    payload: dict,
    engine: WorkloadAttributionEngine = Depends(get_attribution_engine),
) -> dict:
    record = engine.record_gpu_allocation(
        gpu_uuid=payload.get("gpu_uuid", ""),
        gpu_index=payload.get("gpu_index", 0),
        node_name=payload.get("node_name", ""),
        pod_name=payload.get("pod_name", ""),
        namespace=payload.get("namespace", "default"),
        tenant_id=payload.get("tenant_id", ""),
        project_id=payload.get("project_id", ""),
        owner_kind=payload.get("owner_kind", "Job"),
        owner_name=payload.get("owner_name", ""),
        memory_gib=payload.get("memory_gib", 0.0),
    )
    return {
        "allocation_id": record.allocation_id,
        "gpu_uuid": record.gpu_uuid,
        "tenant_id": record.tenant_id,
    }


@router.post("/api/v1/attribution/gpu-allocations/{allocation_id}/release", tags=["workload attribution"])
def attribution_release_gpu(
    allocation_id: str,
    engine: WorkloadAttributionEngine = Depends(get_attribution_engine),
) -> dict:
    if not engine.release_gpu(allocation_id):
        raise HTTPException(404, "Allocation not found or already released")
    return {"status": "released", "allocation_id": allocation_id}


@router.get("/api/v1/attribution/allocations", tags=["workload attribution"])
def attribution_list_allocations(
    tenant_id: str = "",
    pod_name: str = "",
    owner_name: str = "",
    engine: WorkloadAttributionEngine = Depends(get_attribution_engine),
) -> list[dict]:
    records = engine.get_gpu_allocations(
        tenant_id=tenant_id or None,
        pod_name=pod_name or None,
        owner_name=owner_name or None,
    )
    return [
        {
            "allocation_id": r.allocation_id,
            "gpu_uuid": r.gpu_uuid,
            "gpu_index": r.gpu_index,
            "node_name": r.node_name,
            "pod_name": r.pod_name,
            "tenant_id": r.tenant_id,
            "owner_kind": r.owner_kind,
            "owner_name": r.owner_name,
            "allocated_at": r.allocated_at,
            "released_at": r.released_at,
            "duration_hours": r.duration_hours,
        }
        for r in records
    ]


@router.get("/api/v1/attribution/summary", tags=["workload attribution"])
def attribution_summary(
    tenant_id: str = "",
    engine: WorkloadAttributionEngine = Depends(get_attribution_engine),
) -> dict:
    return engine.get_attribution_summary(tenant_id or None)


@router.get("/api/v1/attribution/pods", tags=["workload attribution"])
def attribution_list_pods(
    tenant_id: str = "",
    owner_kind: str = "",
    owner_name: str = "",
    engine: WorkloadAttributionEngine = Depends(get_attribution_engine),
) -> list[dict]:
    if tenant_id:
        pods = engine.list_pods_by_tenant(tenant_id)
    elif owner_kind and owner_name:
        pods = engine.list_pods_by_owner(owner_kind, owner_name)
    else:
        return []
    return [
        {
            "pod_name": p.pod_name,
            "namespace": p.namespace,
            "owner_kind": p.owner_kind,
            "owner_name": p.owner_name,
            "resource_owner": p.resource_owner.value,
            "tenant_id": p.tenant_id,
            "project_id": p.project_id,
        }
        for p in pods
    ]


# ── S11: Tenant Isolation ────────────────────────────────────────


@router.post("/api/v1/tenants/quota", tags=["multi-tenancy"])
def tenant_set_quota(
    project_id: UUID,
    max_gpus: int = 64,
    max_clusters: int = 10,
    max_monthly_cost: float = 50000.0,
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    manager.set_project_quota(project_id, max_gpus, max_clusters, max_monthly_cost)
    return {"status": "quota_updated", "project_id": str(project_id)}


@router.post("/api/v1/tenants/{team_id}/users", tags=["multi-tenancy"])
def tenant_add_user(
    team_id: str,
    username: str,
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    manager.add_tenant_user(team_id, username)
    return {"status": "added", "team_id": team_id, "username": username}


@router.delete("/api/v1/tenants/{team_id}/users/{username}", tags=["multi-tenancy"])
def tenant_remove_user(
    team_id: str,
    username: str,
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    if not manager.remove_tenant_user(team_id, username):
        raise HTTPException(404, "User not found in team")
    return {"status": "removed"}


@router.get("/api/v1/tenants/{team_id}/users", tags=["multi-tenancy"])
def tenant_list_users(
    team_id: str,
    manager: TenantManager = Depends(get_tenant_manager),
) -> list[str]:
    return manager.get_tenant_users(team_id)


@router.post("/api/v1/tenants/{team_id}/isolation-policy", tags=["multi-tenancy"])
def tenant_set_isolation(
    team_id: str,
    payload: dict,
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    manager.set_isolation_policy(team_id, payload)
    return {"status": "policy_set", "team_id": team_id}


@router.get("/api/v1/tenants/{team_id}/isolation-policy", tags=["multi-tenancy"])
def tenant_get_isolation(
    team_id: str,
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    policy = manager.get_isolation_policy(team_id)
    if policy is None:
        raise HTTPException(404, "Isolation policy not found")
    return policy


@router.get("/api/v1/tenants/projects/{project_id}/namespace", tags=["multi-tenancy"])
def tenant_project_namespace(
    project_id: str,
    manager: TenantManager = Depends(get_tenant_manager),
) -> dict:
    ns = manager.get_namespace_for_project(project_id)
    return {"project_id": project_id, "namespace": ns}


# ── S23: Cost Anomaly Detection ────────────────────────────────


@router.get("/api/v1/anomaly/cost/{cluster_id}", tags=["cost anomaly"])
def anomaly_cost(
    cluster_id: UUID,
    detector: CostAnomalyDetector = Depends(get_anomaly_detector),
) -> CostAnomalyResult:
    cluster = get_repository().get_cluster(cluster_id)
    name = cluster.name if cluster else ""
    return detector.analyze(cluster_id, name)


@router.get("/api/v1/anomaly/cost", tags=["cost anomaly"])
def anomaly_cost_all(
    detector: CostAnomalyDetector = Depends(get_anomaly_detector),
) -> list[CostAnomalyResult]:
    return detector.analyze_all(get_repository())


# ── S23: Compliance & Audit ─────────────────────────────────────


@router.get("/api/v1/compliance/report/{cluster_id}", tags=["compliance"])
def compliance_report(
    cluster_id: UUID,
    framework: str = "soc2",
    engine: ComplianceEngine = Depends(get_compliance_engine),
) -> ComplianceReport:
    return engine.generate_report(cluster_id, framework)


# ── S23: Dashboard & Reporting ─────────────────────────────────


@router.get("/api/v1/dashboard/{cluster_id}", tags=["dashboard"])
def dashboard_summary(
    cluster_id: UUID,
    service: DashboardService = Depends(get_dashboard_service),
) -> DashboardSummary:
    return service.get_summary(cluster_id)


@router.get("/api/v1/dashboard", tags=["dashboard"])
def dashboard_all(
    service: DashboardService = Depends(get_dashboard_service),
) -> list[DashboardSummary]:
    return service.list_summaries()


@router.post("/api/v1/reports", tags=["dashboard"], status_code=201)
def report_create(
    payload: ScheduledReport,
    scheduler: ReportScheduler = Depends(get_report_scheduler),
) -> ScheduledReport:
    return scheduler.create_report(payload)


@router.get("/api/v1/reports", tags=["dashboard"])
def report_list(
    scheduler: ReportScheduler = Depends(get_report_scheduler),
) -> list[ScheduledReport]:
    return scheduler.list_reports()


@router.get("/api/v1/reports/{report_id}", tags=["dashboard"])
def report_get(
    report_id: UUID,
    scheduler: ReportScheduler = Depends(get_report_scheduler),
) -> ScheduledReport:
    report = scheduler.get_report(report_id)
    if report is None:
        raise HTTPException(404, "Report not found")
    return report


@router.patch("/api/v1/reports/{report_id}", tags=["dashboard"])
def report_update(
    report_id: UUID,
    payload: dict,
    scheduler: ReportScheduler = Depends(get_report_scheduler),
) -> ScheduledReport:
    report = scheduler.update_report(report_id, payload)
    if report is None:
        raise HTTPException(404, "Report not found")
    return report


@router.delete("/api/v1/reports/{report_id}", tags=["dashboard"], status_code=204)
def report_delete(
    report_id: UUID,
    scheduler: ReportScheduler = Depends(get_report_scheduler),
) -> Response:
    if not scheduler.delete_report(report_id):
        raise HTTPException(404, "Report not found")
    return Response(status_code=204)


@router.post("/api/v1/reports/{report_id}/generate", tags=["dashboard"])
def report_generate(
    report_id: UUID,
    scheduler: ReportScheduler = Depends(get_report_scheduler),
) -> dict:
    try:
        return scheduler.generate_report_data(report_id, get_repository())
    except KeyError as exc:
        raise HTTPException(404, detail=str(exc))


# ── S20: Explanation Service ──────────────────────────────────────


@router.get("/api/v1/explain/{recommendation_id}", tags=["explanation"])
def explain_recommendation(
    recommendation_id: str,
    service: ExplanationService = Depends(get_explanation_service),
) -> list[dict]:
    explanations = service.get_explanation_by_rec(recommendation_id)
    return [
        {
            "explanation_id": e.explanation_id,
            "category": e.category.value,
            "title": e.title,
            "summary": e.summary,
            "root_cause": e.root_cause,
            "impact": e.impact,
            "evidence": e.evidence,
            "metrics": e.metrics,
            "confidence": e.confidence,
            "severity": e.severity,
            "generated_at": e.generated_at,
            "expires_at": e.expires_at,
        }
        for e in explanations
    ]


@router.post("/api/v1/explain/generate", tags=["explanation"], status_code=201)
def generate_explanation(
    payload: dict,
    service: ExplanationService = Depends(get_explanation_service),
) -> dict:
    explanation = service.generate_explanation(
        recommendation_id=payload.get("recommendation_id", ""),
        category=ExplanationCategory(payload.get("category", "performance")),
        title=payload.get("title", ""),
        summary=payload.get("summary", ""),
        root_cause=payload.get("root_cause", ""),
        impact=payload.get("impact", ""),
        evidence=payload.get("evidence"),
        metrics=payload.get("metrics"),
        confidence=payload.get("confidence", 0.8),
        severity=payload.get("severity", "medium"),
        ttl_hours=payload.get("ttl_hours"),
    )
    return {
        "explanation_id": explanation.explanation_id,
        "expires_at": explanation.expires_at,
    }


@router.get("/api/v1/explain/expiry/{recommendation_id}", tags=["explanation"])
def check_explanation_expiry(
    recommendation_id: str,
    service: ExplanationService = Depends(get_explanation_service),
) -> dict:
    is_expired = service.check_expiry(recommendation_id)
    return {"recommendation_id": recommendation_id, "is_expired": is_expired}


@router.post("/api/v1/explain/extend/{recommendation_id}", tags=["explanation"])
def extend_explanation(
    recommendation_id: str,
    hours: int = 168,
    service: ExplanationService = Depends(get_explanation_service),
) -> dict:
    success = service.extend_expiry(recommendation_id, hours)
    if not success:
        raise HTTPException(400, "Cannot extend expiry (not found or max extensions reached)")
    return {"recommendation_id": recommendation_id, "extended": True, "hours": hours}


@router.get("/api/v1/explain/expired", tags=["explanation"])
def list_expired(
    service: ExplanationService = Depends(get_explanation_service),
) -> list[dict]:
    return service.list_expired_recommendations()


# ── S20: Shadow Deployment ────────────────────────────────────────


@router.post("/api/v1/shadow/create", tags=["shadow"], status_code=201)
def shadow_create(
    payload: dict,
    service: ExplanationService = Depends(get_explanation_service),
) -> dict:
    shadow = service.create_shadow(
        recommendation_id=payload.get("recommendation_id", ""),
        cluster_id=payload.get("cluster_id", ""),
        baseline_metrics=payload.get("baseline_metrics"),
        auto_promote=payload.get("auto_promote", False),
        rollback_on_failure=payload.get("rollback_on_failure", True),
    )
    return {
        "shadow_id": shadow.shadow_id,
        "status": shadow.status,
        "created_at": shadow.created_at,
    }


@router.post("/api/v1/shadow/{shadow_id}/start", tags=["shadow"])
def shadow_start(
    shadow_id: str,
    service: ExplanationService = Depends(get_explanation_service),
) -> dict:
    shadow = service.start_shadow(shadow_id)
    if shadow is None:
        raise HTTPException(404, "Shadow not found or not in pending state")
    return {"shadow_id": shadow_id, "status": shadow.status, "started_at": shadow.started_at}


@router.post("/api/v1/shadow/{shadow_id}/complete", tags=["shadow"])
def shadow_complete(
    shadow_id: str,
    payload: dict,
    service: ExplanationService = Depends(get_explanation_service),
) -> dict:
    shadow = service.complete_shadow(
        shadow_id,
        shadow_metrics=payload.get("shadow_metrics", {}),
        outcome=payload.get("outcome", "success"),
        confidence=payload.get("confidence", 0.0),
    )
    if shadow is None:
        raise HTTPException(404, "Shadow not found")
    result = {
        "shadow_id": shadow_id,
        "status": shadow.status,
        "completed_at": shadow.completed_at,
        "outcome": shadow.outcome,
        "impact_delta": shadow.impact_delta,
    }
    if shadow.auto_promote and shadow.outcome == "success":
        promote = service.promote_shadow(shadow_id)
        result["promotion"] = promote
    return result


@router.post("/api/v1/shadow/{shadow_id}/fail", tags=["shadow"])
def shadow_fail(
    shadow_id: str,
    payload: dict = {},
    service: ExplanationService = Depends(get_explanation_service),
) -> dict:
    shadow = service.fail_shadow(shadow_id, payload.get("reason", ""))
    if shadow is None:
        raise HTTPException(404, "Shadow not found")
    return {"shadow_id": shadow_id, "status": shadow.status, "outcome": shadow.outcome}


@router.get("/api/v1/shadow", tags=["shadow"])
def shadow_list(
    cluster_id: str = "",
    status: str = "",
    service: ExplanationService = Depends(get_explanation_service),
) -> list[dict]:
    shadows = service.list_shadows(cluster_id or None, status or None)
    return [
        {
            "shadow_id": s.shadow_id,
            "recommendation_id": s.recommendation_id,
            "cluster_id": s.cluster_id,
            "status": s.status,
            "created_at": s.created_at,
            "started_at": s.started_at,
            "completed_at": s.completed_at,
            "outcome": s.outcome,
            "confidence": s.confidence,
            "auto_promote": s.auto_promote,
            "impact_delta": s.impact_delta,
        }
        for s in shadows
    ]


@router.get("/api/v1/shadow/{shadow_id}", tags=["shadow"])
def shadow_get(
    shadow_id: str,
    service: ExplanationService = Depends(get_explanation_service),
) -> dict:
    shadow = service.get_shadow(shadow_id)
    if shadow is None:
        raise HTTPException(404, "Shadow not found")
    return {
        "shadow_id": shadow.shadow_id,
        "recommendation_id": shadow.recommendation_id,
        "cluster_id": shadow.cluster_id,
        "status": shadow.status,
        "created_at": shadow.created_at,
        "started_at": shadow.started_at,
        "completed_at": shadow.completed_at,
        "baseline_metrics": shadow.baseline_metrics,
        "shadow_metrics": shadow.shadow_metrics,
        "impact_delta": shadow.impact_delta,
        "outcome": shadow.outcome,
        "confidence": shadow.confidence,
        "auto_promote": shadow.auto_promote,
    }


# ── S7: Agent Protocol ────────────────────────────────────────────


@router.post("/api/v1/agents/register", tags=["agent protocol"], status_code=201)
def agent_register(
    payload: dict,
    registry: AgentRegistry = Depends(get_agent_registry),
) -> dict:
    registration = registry.register(
        hostname=payload.get("hostname", ""),
        version=payload.get("version", "0.1.0"),
        capabilities=payload.get("capabilities", []),
        labels=payload.get("labels", {}),
        public_key_pem=payload.get("public_key_pem", ""),
        api_key_hash=payload.get("api_key_hash", ""),
        mTLS_enabled=payload.get("mtls_enabled", False),
    )
    return {
        "agent_id": registration.agent_id,
        "status": registration.status.value,
        "registered_at": registration.registered_at,
    }


@router.post("/api/v1/agents/{agent_id}/heartbeat", tags=["agent protocol"])
def agent_heartbeat(
    agent_id: str,
    payload: dict,
    registry: AgentRegistry = Depends(get_agent_registry),
) -> dict:
    hb = registry.process_heartbeat(
        agent_id,
        load=payload.get("load"),
        metrics_summary=payload.get("metrics_summary"),
    )
    if hb is None:
        raise HTTPException(404, "Agent not found")
    return {
        "agent_id": agent_id,
        "sequence": hb.sequence,
        "timestamp": hb.timestamp,
        "status": hb.status.value,
    }


@router.post("/api/v1/agents/{agent_id}/telemetry", tags=["agent protocol"])
def agent_telemetry(
    agent_id: str,
    payload: dict,
    registry: AgentRegistry = Depends(get_agent_registry),
) -> dict:
    registry.process_heartbeat(agent_id)
    return {"status": "accepted", "agent_id": agent_id}


@router.get("/api/v1/agents", tags=["agent protocol"])
def agent_list(
    status: str = "",
    registry: AgentRegistry = Depends(get_agent_registry),
) -> list[dict]:
    status_filter = AgentStatus(status) if status else None
    agents = registry.list_agents(status_filter)
    return [
        {
            "agent_id": a.agent_id,
            "hostname": a.hostname,
            "version": a.version,
            "status": a.status.value,
            "capabilities": a.capabilities,
            "labels": a.labels,
            "registered_at": a.registered_at,
            "last_heartbeat": a.last_heartbeat,
        }
        for a in agents
    ]


@router.get("/api/v1/agents/{agent_id}", tags=["agent protocol"])
def agent_get(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry),
) -> dict:
    agent = registry.get_agent(agent_id)
    if agent is None:
        raise HTTPException(404, "Agent not found")
    return {
        "agent_id": agent.agent_id,
        "hostname": agent.hostname,
        "version": agent.version,
        "status": agent.status.value,
        "capabilities": agent.capabilities,
        "labels": agent.labels,
        "registered_at": agent.registered_at,
        "last_heartbeat": agent.last_heartbeat,
    }


@router.patch("/api/v1/agents/{agent_id}/status", tags=["agent protocol"])
def agent_update_status(
    agent_id: str,
    payload: dict,
    registry: AgentRegistry = Depends(get_agent_registry),
) -> dict:
    new_status = AgentStatus(payload.get("status", "active"))
    if not registry.update_status(agent_id, new_status, payload.get("message", ""),
                                  payload.get("details", {})):
        raise HTTPException(404, "Agent not found")
    return {"agent_id": agent_id, "status": new_status.value}


@router.get("/api/v1/agents/{agent_id}/events", tags=["agent protocol"])
def agent_events(
    agent_id: str,
    limit: int = 100,
    registry: AgentRegistry = Depends(get_agent_registry),
) -> list[dict]:
    events = registry.list_events(agent_id, limit)
    return [
        {
            "event_id": e.event_id,
            "event_type": e.event_type.value,
            "timestamp": e.timestamp,
            "previous_status": e.previous_status.value if e.previous_status else None,
            "new_status": e.new_status.value,
            "message": e.message,
            "details": e.details,
        }
        for e in events
    ]


@router.get("/api/v1/agents/summary", tags=["agent protocol"])
def agent_summary(
    registry: AgentRegistry = Depends(get_agent_registry),
) -> dict:
    return {
        "counts": registry.get_agent_count(),
        "heartbeat_stats": registry.get_heartbeat_stats(),
    }


@router.get("/api/v1/info", tags=["system"])
def system_info() -> dict:
    """System information endpoint."""
    settings = get_settings()
    return {
        "name": "GPUOpt Backend Sandbox",
        "version": __version__,
        "environment": settings.env,
        "documentation": "/docs",
        "health": "/health/detailed",
        "metrics": "/metrics",
    }
