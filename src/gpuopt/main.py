from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import router
from .audit import AuditMiddleware
from .config import get_settings
from .dependencies import get_repository, get_rbac_manager
from .domains.router import domain_router
from .exceptions import register_exception_handlers
from .model_governance.router import governance_router
from .optimizer.router import optimizer_router
from .scheduler import scheduler_router
from .inference.routes import router as inference_router
from .k8s_operator.routes import k8s_router
from .federation.router import federation_router
from .healing import healing_router
from .middleware import AuthMiddleware, RequestLoggingMiddleware
from .policy import policy_router
from .predictor import predictor_router
from .slurm_routes import slurm_router
from .alert_router import alert_router
from .monitoring_router import monitoring_router
from .rtx_routes import router as rtx_router
from .workload_agent_router import router as workload_agent_router
from .extended_router import router as extended_router
from .inference_api import router as inference_api_router
from .mock_inference import router as mock_inference_router
from .references_router import router as references_router
from .metrics_kpi_router import router as metrics_kpi_router
from .risk_gates_router import router as risk_gates_router
from .deployment_workflow_router import router as deployment_workflow_router
from .gpu_usage_inventory_router import router as gpu_usage_inventory_router
from .optimization_analysis_router import router as optimization_analysis_router
from .environment_checks_router import router as environment_checks_router
from .risk_modules_router import router as risk_modules_router
from .risk_r02_router import router as risk_r02_router
from .risk_r03_router import router as risk_r03_router
from .risk_r04_router import router as risk_r04_router
from .ml.router import router as ml_router
from .ollama.router import router as ollama_router
from .deepseek.router import router as deepseek_router
from .intelligence.router import intelligence_router
from .observability.router import router as observability_router
from .ratelimit import RateLimitMiddleware
from .streaming import streaming_router
from .s24_routes import s24_router
from .vllm_router import router as vllm_router
from .versioning import V1DeprecationMiddleware, create_v2_router, mount_v2


_discovery_service: Any = None
_retrain_task: asyncio.Task | None = None


async def _periodic_model_retrain(interval_hours: int = 24) -> None:
    """Background task that retrains ML models on a schedule."""
    import asyncio
    logger.info("Periodic model retrain task started (interval=%dh)", interval_hours)
    while True:
        try:
            from .ml.engine import MLEngine
            engine = MLEngine()
            result = engine.train_ensemble()
            logger.info("Scheduled retrain complete: %s", result)
        except Exception as exc:
            logger.warning("Scheduled retrain failed: %s", exc)
        await asyncio.sleep(interval_hours * 3600)


@asynccontextmanager
async def lifespan(_: FastAPI):
    from .metrics import instrument_app
    instrument_app(_)
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    get_repository()
    rbac = get_rbac_manager()
    if settings.default_admin_key:
        admin_user = rbac.get_user_by_api_key(settings.default_admin_key)
        if admin_user:
            logger.info("Default admin key configured (user: %s)", admin_user.username)
    else:
        admin_users = [u for u in rbac.list_users() if u.username == "admin"]
        if admin_users:
            logger.info("Using default admin user with API key prefix %s", admin_users[0].api_key_prefix)

    global _discovery_service, _retrain_task
    from .discovery_service import AutoDiscoveryService, NodeDiscoveryTarget
    _discovery_service = AutoDiscoveryService(interval_seconds=300)
    discovery_env = os.environ.get("GPUOPT_DISCOVERY_TARGETS", "")
    if discovery_env:
        for entry in discovery_env.split(","):
            entry = entry.strip()
            if entry:
                _discovery_service.add_target(NodeDiscoveryTarget(host=entry))
    if _discovery_service.targets:
        await _discovery_service.start()
        logger.info("Auto-discovery started with %d targets", len(_discovery_service.targets))
    retrain_interval = int(os.environ.get("GPUOPT_RETRAIN_INTERVAL_HOURS", "0"))
    if retrain_interval > 0:
        _retrain_task = asyncio.create_task(_periodic_model_retrain(retrain_interval))
        logger.info("Periodic model retraining started (interval=%dh)", retrain_interval)
    yield
    if _discovery_service and _discovery_service.targets:
        await _discovery_service.stop()
    if _retrain_task:
        _retrain_task.cancel()
        try:
            await _retrain_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="GPUOpt Backend Sandbox",
    version=__version__,
    description=(
        "GPU Cluster Intelligence Platform. Cross-cluster optimization, predictive failure-aware orchestration, "
        "automated idle GPU reclamation, adaptive scheduling, digital twin simulation, "
        "ML-driven workload placement, and FinOps cost optimization for GPU/CPU clusters."
    ),
    lifespan=lifespan,
)

settings = get_settings()

register_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(AuditMiddleware)
app.add_middleware(V1DeprecationMiddleware)

v2_router = create_v2_router()
mount_v2(app, v2_router)

app.include_router(router)
app.include_router(inference_router)
app.include_router(s24_router)
app.include_router(k8s_router)
app.include_router(slurm_router)
app.include_router(domain_router)
app.include_router(optimizer_router)
app.include_router(governance_router)
app.include_router(scheduler_router)
app.include_router(policy_router)
app.include_router(predictor_router)
app.include_router(healing_router)
app.include_router(federation_router)
app.include_router(monitoring_router)
app.include_router(alert_router)
app.include_router(workload_agent_router)
app.include_router(extended_router)
app.include_router(inference_api_router)
app.include_router(mock_inference_router)
app.include_router(references_router)
app.include_router(metrics_kpi_router)
app.include_router(risk_gates_router)
app.include_router(deployment_workflow_router)
app.include_router(gpu_usage_inventory_router)
app.include_router(optimization_analysis_router)
app.include_router(environment_checks_router)
app.include_router(risk_modules_router)
app.include_router(risk_r02_router)
app.include_router(risk_r03_router)
app.include_router(risk_r04_router)
app.include_router(rtx_router)
app.include_router(ml_router)
app.include_router(ollama_router)
app.include_router(deepseek_router)
app.include_router(streaming_router)
app.include_router(intelligence_router)
app.include_router(observability_router)
app.include_router(vllm_router)

frontend_dir = Path(os.environ.get("GPUOPT_FRONTEND_DIR", "/app/frontend"))
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
