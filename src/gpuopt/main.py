from __future__ import annotations

import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
from .ratelimit import RateLimitMiddleware
from .s24_routes import s24_router
from .versioning import V1DeprecationMiddleware, create_v2_router, mount_v2


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
    yield


app = FastAPI(
    title="GPUOpt Backend Sandbox",
    version=__version__,
    description=(
        "Read-only Kubernetes environment registration, GPU platform readiness checks, "
        "telemetry normalization, cluster state querying, trace replay, "
        "baseline comparison, workload analysis, optimization recommendations, "
        "Use this sandbox before implementing predictive scheduling and automated actuation."
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
