from __future__ import annotations

from fastapi import APIRouter

from .digital_twin_extended import ExtendedTwinService
from .governance_extended import ExtendedGovernanceService
from .inference_extended import ExtendedInferenceService
from .integration_extended import IntegrationManager
from .optimization_extended import ExtendedOptimizationService
from .prediction_extended import ComprehensivePredictionService
from .telemetry_extended import ExtendedTelemetryService, TelemetryStreamer
from .telemetry_quality import TelemetryQualityService
from .training_extended import ExtendedTrainingService

router = APIRouter(prefix="/api/v1/extended", tags=["extended"])

_twin = ExtendedTwinService()
_gov = ExtendedGovernanceService()
_inf = ExtendedInferenceService()
_integration = IntegrationManager()
_opt = ExtendedOptimizationService()
_pred = ComprehensivePredictionService()
_quality = TelemetryQualityService()
_telemetry = ExtendedTelemetryService()
_streamer = TelemetryStreamer()
_training = ExtendedTrainingService()


@router.get("/telemetry/snapshot")
async def get_telemetry_snapshot(cluster_id: str = ""):
    return _telemetry.collect_full_snapshot(cluster_id).model_dump(mode="json")


@router.post("/telemetry/stream/start")
async def start_streaming(interval_seconds: float = 5.0):
    _streamer.start(interval_seconds)
    return {"status": "started", "interval": interval_seconds}


@router.post("/telemetry/stream/stop")
async def stop_streaming():
    _streamer.stop()
    return {"status": "stopped"}


@router.get("/telemetry/stream/events")
async def get_stream_events(event_type: str | None = None, limit: int = 100):
    return [e.model_dump(mode="json") for e in _streamer.get_events(event_type, limit)]


@router.get("/prediction/comprehensive")
async def comprehensive_prediction(cluster_id: str = ""):
    return _pred.predict_all(cluster_id).model_dump(mode="json")


@router.get("/digital-twin/simulate")
async def simulate_twin(cluster_id: str = ""):
    return _twin.run_comprehensive_simulation(cluster_id).model_dump(mode="json")


@router.post("/digital-twin/what-if")
async def what_if(actions: list[dict]):
    return _twin.what_if(actions).model_dump(mode="json")


@router.post("/digital-twin/score-action")
async def score_action(candidate: dict):
    return _twin.score_action(candidate).model_dump(mode="json")


@router.get("/optimization/consolidation-plan")
async def consolidation_plan(cluster_id: str = ""):
    return _opt.create_consolidation_plan(cluster_id).model_dump(mode="json")


@router.post("/optimization/gpu-tier")
async def select_gpu_tier(workload: dict):
    return _opt.select_gpu_tier(workload).model_dump(mode="json")


@router.post("/optimization/elastic-workers")
async def optimize_workers(job: dict):
    return _opt.optimize_workers(job).model_dump(mode="json")


@router.post("/training/submit")
async def submit_training(job: dict):
    return _training.submit_training_job(job)


@router.post("/training/hpo/create")
async def create_hpo_job(search_algorithm: str = "bayesian", max_trials: int = 100):
    from .training_extended import HPOManager
    hpo = HPOManager()
    return hpo.create_job(search_algorithm, max_trials).model_dump(mode="json")


@router.get("/inference/right-size")
async def right_size(model_name: str = ""):
    return _inf.right_size_replicas(model_name).model_dump(mode="json")


@router.post("/inference/scaling-policy")
async def create_scaling_policy(model_name: str = ""):
    return _inf.create_scaling_policy(model_name).model_dump(mode="json")


@router.post("/inference/place")
async def place_model(model_name: str = "", num_replicas: int = 1):
    return [p.model_dump(mode="json") for p in _inf.place_model(model_name, num_replicas)]


@router.post("/inference/optimize-moe")
async def optimize_moe(model_size_gb: float = 1.0):
    return _inf.optimize_moe(model_size_gb).model_dump(mode="json")


@router.post("/governance/policy/create")
async def create_policy(name: str = "", domain: str = "general", rules: list[dict] | None = None):
    return _gov._enforcer.create_envelope(name, domain, rules or []).model_dump(mode="json")


@router.post("/governance/policy/evaluate")
async def evaluate_policy(policy_id: str, context: dict):
    env = _gov._enforcer._envelopes.get(policy_id)
    if not env:
        return {"error": "Policy not found"}
    return _gov._enforcer.evaluate(env, context)


@router.post("/governance/rollback/create")
async def create_rollback(action_id: str, action_type: str, reason: str = ""):
    return _gov.create_rollback(action_id, action_type, reason).model_dump(mode="json")


@router.post("/governance/quota/set")
async def set_quota(tenant_id: str, tenant_name: str, max_gpus: int, max_memory_gb: float):
    return _gov.manage_tenant_quota(tenant_id, "set", tenant_name=tenant_name,
                                     max_gpus=max_gpus, max_memory_gb=max_memory_gb).model_dump(mode="json")


@router.get("/governance/quota/{tenant_id}")
async def get_quota(tenant_id: str):
    q = _gov.manage_tenant_quota(tenant_id, "get")
    return q.model_dump(mode="json") if q else {"error": "Not found"}


@router.get("/governance/report/{report_type}")
async def generate_report(report_type: str = "compliance"):
    return _gov.generate_report(report_type).model_dump(mode="json")


@router.get("/integration/health")
async def integration_health():
    return _integration.health()


@router.get("/integration/runtimes")
async def detect_runtimes():
    return [r.model_dump(mode="json") for r in _integration.get_runtime_detector().detect()]


@router.post("/integration/prometheus/target")
async def register_prometheus_target(endpoint: str, labels: dict | None = None):
    return _integration.get_prometheus().register_target(endpoint, labels=labels).model_dump(mode="json")


@router.post("/integration/otel/configure")
async def configure_otel(service_name: str = "gpuopt", endpoint: str = ""):
    return _integration.get_opentelemetry().configure(service_name, endpoint).model_dump(mode="json")


@router.post("/integration/storage/configure")
async def configure_storage(store_type: str, endpoint: str, bucket: str, region: str = ""):
    return _integration.get_object_store().configure(store_type, endpoint, bucket, region).model_dump(mode="json")


# ── R01: Telemetry Quality & Onboarding ───────────────────────

@router.post("/telemetry/quality/score")
async def score_telemetry_quality(snapshot: dict):
    from .schemas import TelemetrySnapshot
    obj = TelemetrySnapshot(**snapshot)
    return _quality.process_snapshot(obj)


@router.post("/telemetry/quality/data")
async def resolve_with_fallback(source: str, data: dict | None = None, required_fields: list[str] | None = None):
    return _quality.get_data(source, data, required_fields)


@router.post("/telemetry/onboarding/register")
async def register_source(name: str, source_type: str, required_fields: list[str] | None = None, critical: bool = False):
    return _quality.onboarding.register_source(name, source_type, required_fields, critical).model_dump(mode="json")


@router.post("/telemetry/onboarding/advance")
async def advance_phase(source_id: str, checks_passed: int | None = None):
    return _quality.onboarding.advance_phase(source_id, checks_passed)


@router.get("/telemetry/onboarding/status/{source_id}")
async def get_onboarding_status(source_id: str):
    status = _quality.onboarding.get_status(source_id)
    if not status:
        return {"error": "Source not found"}
    return status.model_dump(mode="json")


@router.get("/telemetry/onboarding/sources")
async def list_onboarding_sources():
    return _quality.onboarding.list_sources()


@router.get("/telemetry/quality/health")
async def quality_health():
    return _quality.health()
