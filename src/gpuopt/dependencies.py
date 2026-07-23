from __future__ import annotations

from functools import lru_cache

from .config import get_settings
from .ml.drift_detector import DriftDetector
from .ml.forecast_model import ForecastModel
from .ml.recommendation_model import RecommendationModel
from .repository import ClusterRepository
from .actuation import ActuationService
from .analysis import AnalysisService
from .cost_analysis import CostAnalysisService
from .digital_twin import DigitalTwinService
from .guarded_automation import ApprovalWorkflow, ChaosEngine, PolicyEngine
from .recommendations import RecommendationEngine
from .s23_features import AlertManager, TenantManager, CostAnomalyDetector, ComplianceEngine, DashboardService, ReportScheduler
from .scheduler import SchedulerService
from .services import ClusterStateService, EnvironmentCheckService
from .trace import TraceService
from .training import TrainingService
from .inference.service import InferenceService
from .finops import FinOpsService
from .power import PowerService
from .agent_protocol import AgentRegistry, get_agent_registry
from .dcgm_ingestion import DcgmIngestionPipeline, get_dcgm_pipeline
from .watch_stream import WatchManager, get_watch_manager
from .workload_attribution import WorkloadAttributionEngine, get_attribution_engine
from .explanation_service import ExplanationService, get_explanation_service
from .rbac import RBACManager


@lru_cache(maxsize=1)
def get_rbac_manager() -> RBACManager:
    return RBACManager()


@lru_cache(maxsize=1)
def get_repository() -> ClusterRepository:
    return ClusterRepository(get_settings().database_path)


@lru_cache(maxsize=1)
def get_rec_model() -> RecommendationModel:
    return RecommendationModel()


@lru_cache(maxsize=1)
def get_forecast_model() -> ForecastModel:
    return ForecastModel()


@lru_cache(maxsize=1)
def get_drift_detector() -> DriftDetector:
    return DriftDetector()


@lru_cache(maxsize=1)
def get_check_service() -> EnvironmentCheckService:
    return EnvironmentCheckService(get_repository())


@lru_cache(maxsize=1)
def get_state_service() -> ClusterStateService:
    return ClusterStateService(get_repository(), get_forecast_model())


@lru_cache(maxsize=1)
def get_trace_service() -> TraceService:
    return TraceService(get_repository())


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    return AnalysisService(get_repository())


@lru_cache(maxsize=1)
def get_rec_engine() -> RecommendationEngine:
    return RecommendationEngine(get_repository(), get_rec_model())


@lru_cache(maxsize=1)
def get_digital_twin() -> DigitalTwinService:
    return DigitalTwinService(get_repository(), get_drift_detector())


@lru_cache(maxsize=1)
def get_scheduler_service() -> SchedulerService:
    return SchedulerService(get_repository(), get_forecast_model())


def get_actuation_service() -> ActuationService:
    return ActuationService(get_repository())


@lru_cache(maxsize=1)
def get_cost_analysis_service() -> CostAnalysisService:
    return CostAnalysisService(get_repository())


@lru_cache(maxsize=1)
def get_training_service() -> TrainingService:
    return TrainingService()


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    return InferenceService()


@lru_cache(maxsize=1)
def get_finops_service() -> FinOpsService:
    return FinOpsService(get_repository())


@lru_cache(maxsize=1)
def get_power_service() -> PowerService:
    return PowerService(get_repository())


@lru_cache(maxsize=1)
def get_policy_engine() -> PolicyEngine:
    return PolicyEngine(get_repository())


@lru_cache(maxsize=1)
def get_approval_workflow() -> ApprovalWorkflow:
    return ApprovalWorkflow(get_repository())


@lru_cache(maxsize=1)
def get_chaos_engine() -> ChaosEngine:
    return ChaosEngine()


@lru_cache(maxsize=1)
def get_alert_manager() -> AlertManager:
    return AlertManager(get_repository())


@lru_cache(maxsize=1)
def get_tenant_manager() -> TenantManager:
    return TenantManager(get_repository())


@lru_cache(maxsize=1)
def get_anomaly_detector() -> CostAnomalyDetector:
    return CostAnomalyDetector()


@lru_cache(maxsize=1)
def get_compliance_engine() -> ComplianceEngine:
    return ComplianceEngine()


@lru_cache(maxsize=1)
def get_dashboard_service() -> DashboardService:
    return DashboardService(get_repository())


@lru_cache(maxsize=1)
def get_report_scheduler() -> ReportScheduler:
    return ReportScheduler()

@lru_cache(maxsize=1)
def get_agent_registry() -> AgentRegistry:
    from .agent_protocol import get_agent_registry as _get_registry
    return _get_registry()
