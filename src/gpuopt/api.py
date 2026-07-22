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


@router.get("/api/v1/clusters", response_model=list[ClusterRecord], tags=["clusters"])
def list_clusters(repository: ClusterRepository = Depends(get_repository)) -> list[ClusterRecord]:
    return repository.list_clusters()


@router.get("/api/v1/clusters/{cluster_id}", response_model=ClusterRecord, tags=["clusters"])
def get_cluster(
    cluster_id: UUID,
    repository: ClusterRepository = Depends(get_repository),
) -> ClusterRecord:
    cluster = repository.get_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
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
    if not isinstance(connector, SlurmConnector):
        raise HTTPException(400, "Cluster is not a Slurm cluster")
    return connector.collect_slurm_telemetry()


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
    if not isinstance(connector, SlurmConnector):
        raise HTTPException(400, "Cluster is not a Slurm cluster")
    return connector.get_cluster_topology()


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
    if not isinstance(connector, SlurmConnector):
        raise HTTPException(400, "Cluster is not a Slurm cluster")
    return connector.collect_monitoring_snapshot()


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
    if not isinstance(connector, SlurmConnector):
        raise HTTPException(400, "Cluster is not a Slurm cluster")
    connector.start_job_monitor(job_id, config)
    return {"status": "started", "job_id": job_id}


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
    if not isinstance(connector, SlurmConnector):
        raise HTTPException(400, "Cluster is not a Slurm cluster")
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
    if not isinstance(connector, SlurmConnector):
        raise HTTPException(400, "Cluster is not a Slurm cluster")
    history = connector.get_job_history(job_id)
    return [h.model_dump(mode="json") for h in history]


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
