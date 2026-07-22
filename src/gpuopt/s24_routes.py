from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from . import __version__
from .actuation import ActuationService
from .cloud_costs import CloudPricingService
from .audit import get_audit_store
from .dependencies import (
    get_actuation_service,
    get_alert_manager,
    get_rbac_manager,
    get_repository,
)
from .ml.anomaly_detector import AnomalyReport, MLAnomalyDetector
from .rbac import (
    Permission,
    RBACManager,
    Role,
    RoleType,
    User,
    require_permission,
)
from .remediation import (
    RemediationAction,
    RemediationActionType,
    RemediationEngine,
    RemediationRule,
    RemediationRun,
    RemediationStatus,
)
from .repository import ClusterRepository
from .s23_features import AlertManager
from .streaming import StreamService, manager as ws_manager

s24_router = APIRouter(tags=["s24"])


# ── ML Anomaly Detector ────────────────────────────────────────

_anomaly_detector: MLAnomalyDetector | None = None


def _get_anomaly_detector() -> MLAnomalyDetector:
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = MLAnomalyDetector()
    return _anomaly_detector


@s24_router.post("/api/v1/anomaly/detect/{cluster_id}")
def detect_anomalies(
    cluster_id: UUID,
    repository: ClusterRepository = Depends(get_repository),
    detector: MLAnomalyDetector = Depends(_get_anomaly_detector),
) -> AnomalyReport:
    state = repository.latest_state(cluster_id)
    if state is None:
        raise HTTPException(404, "No state data available")
    detector.update(state)
    return detector.analyze_state(state)


@s24_router.get("/api/v1/anomaly/history")
def anomaly_history(
    detector: MLAnomalyDetector = Depends(_get_anomaly_detector),
) -> dict:
    return {"status": "ok", "detector": "active"}


# ── WebSocket Streaming ─────────────────────────────────────────

_stream_service: StreamService | None = None


def _get_stream_service() -> StreamService:
    global _stream_service
    if _stream_service is None:
        from .dependencies import get_repository, get_alert_manager
        _stream_service = StreamService(get_repository(), get_alert_manager())
    return _stream_service


@s24_router.websocket("/api/v1/ws/state/{cluster_id}")
async def ws_cluster_state(websocket: WebSocket, cluster_id: UUID):
    svc = _get_stream_service()
    await svc.stream_cluster_state(websocket, cluster_id)


@s24_router.websocket("/api/v1/ws/alerts/{cluster_id}")
async def ws_cluster_alerts(websocket: WebSocket, cluster_id: UUID):
    svc = _get_stream_service()
    await svc.stream_alerts(websocket, cluster_id)


@s24_router.websocket("/api/v1/ws/alerts")
async def ws_all_alerts(websocket: WebSocket):
    svc = _get_stream_service()
    await svc.stream_alerts(websocket)


@s24_router.websocket("/api/v1/ws/metrics/{cluster_id}")
async def ws_cluster_metrics(websocket: WebSocket, cluster_id: UUID):
    svc = _get_stream_service()
    await svc.stream_metrics(websocket, cluster_id)


@s24_router.get("/api/v1/ws/status")
def ws_status() -> dict:
    return {
        "active_connections": ws_manager.active_connections,
        "channels": list(ws_manager._connections.keys()),
    }


# ── Remediation Engine ──────────────────────────────────────────

_remediation_engine: RemediationEngine | None = None


def _get_remediation_engine() -> RemediationEngine:
    global _remediation_engine
    if _remediation_engine is None:
        _remediation_engine = RemediationEngine(get_repository())
    return _remediation_engine


@s24_router.post("/api/v1/remediation/rules")
def create_remediation_rule(rule: RemediationRule) -> RemediationRule:
    engine = _get_remediation_engine()
    return engine.add_rule(rule)


@s24_router.get("/api/v1/remediation/rules")
def list_remediation_rules() -> list[RemediationRule]:
    return _get_remediation_engine().list_rules()


@s24_router.get("/api/v1/remediation/rules/{rule_id}")
def get_remediation_rule(rule_id: str) -> RemediationRule:
    rule = _get_remediation_engine().get_rule(rule_id)
    if rule is None:
        raise HTTPException(404, "Remediation rule not found")
    return rule


@s24_router.patch("/api/v1/remediation/rules/{rule_id}")
def update_remediation_rule(rule_id: str, updates: dict) -> RemediationRule:
    rule = _get_remediation_engine().update_rule(rule_id, updates)
    if rule is None:
        raise HTTPException(404, "Remediation rule not found")
    return rule


@s24_router.delete("/api/v1/remediation/rules/{rule_id}")
def delete_remediation_rule(rule_id: str):
    if not _get_remediation_engine().delete_rule(rule_id):
        raise HTTPException(404, "Remediation rule not found")
    return Response(status_code=204)


@s24_router.get("/api/v1/remediation/runs")
def list_remediation_runs(cluster_id: str = "") -> list[RemediationRun]:
    return _get_remediation_engine().list_runs(cluster_id or None)


@s24_router.post("/api/v1/remediation/evaluate/alert/{alert_id}")
def evaluate_alert_for_remediation(
    alert_id: str,
    repository: ClusterRepository = Depends(get_repository),
    alert_manager: AlertManager = Depends(get_alert_manager),
) -> RemediationRun | None:
    all_alerts = alert_manager.list_alerts()
    target = next((a for a in all_alerts if str(a.id) == alert_id), None)
    if target is None:
        raise HTTPException(404, "Alert not found")
    engine = _get_remediation_engine()
    run = engine.evaluate_alert(target)
    if run is None:
        return JSONResponse(content={"message": "No remediation rule matched", "run": None})
    return run


# ── RBAC ────────────────────────────────────────────────────────


@s24_router.post("/api/v1/rbac/users")
def create_user(
    username: str, email: str, role: RoleType = RoleType.VIEWER,
    rbac: RBACManager = Depends(get_rbac_manager),
    _: None = Depends(require_permission(Permission.RBAC_MANAGE)),
) -> dict:
    user, api_key = rbac.create_user(username, email, role)
    return {"user_id": user.id, "username": user.username, "api_key": api_key, "role": role.value}


@s24_router.get("/api/v1/rbac/users")
def list_users(
    rbac: RBACManager = Depends(get_rbac_manager),
    _: None = Depends(require_permission(Permission.RBAC_MANAGE)),
) -> list[User]:
    return rbac.list_users()


@s24_router.get("/api/v1/rbac/users/{user_id}")
def get_user(
    user_id: str,
    rbac: RBACManager = Depends(get_rbac_manager),
    _: None = Depends(require_permission(Permission.RBAC_MANAGE)),
) -> User:
    user = rbac.get_user(user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    return user


@s24_router.delete("/api/v1/rbac/users/{user_id}")
def delete_user(
    user_id: str,
    rbac: RBACManager = Depends(get_rbac_manager),
    _: None = Depends(require_permission(Permission.RBAC_MANAGE)),
):
    if not rbac.delete_user(user_id):
        raise HTTPException(404, "User not found")
    return Response(status_code=204)


@s24_router.post("/api/v1/rbac/users/{user_id}/rotate-key")
def rotate_api_key(
    user_id: str,
    rbac: RBACManager = Depends(get_rbac_manager),
    _: None = Depends(require_permission(Permission.RBAC_MANAGE)),
) -> dict:
    new_key = rbac.rotate_api_key(user_id)
    if new_key is None:
        raise HTTPException(404, "User not found")
    return {"api_key": new_key}


@s24_router.get("/api/v1/rbac/roles")
def list_roles(
    rbac: RBACManager = Depends(get_rbac_manager),
    _: None = Depends(require_permission(Permission.RBAC_MANAGE)),
) -> list[Role]:
    return rbac.list_roles()


@s24_router.get("/api/v1/rbac/permissions/{user_id}")
def get_user_permissions(
    user_id: str,
    rbac: RBACManager = Depends(get_rbac_manager),
) -> dict:
    user = rbac.get_user(user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    perms = rbac.get_user_permissions(user_id)
    return {"user_id": user_id, "username": user.username, "permissions": sorted(p.value for p in perms)}


# ── Audit Log ──────────────────────────────────────────────────


@s24_router.get("/api/v1/audit/log")
def get_audit_log(
    limit: int = 100, offset: int = 0,
    _: None = Depends(require_permission(Permission.RBAC_MANAGE)),
) -> list:
    store = get_audit_store()
    return [e.model_dump(mode="json") for e in store.list(limit, offset)]


@s24_router.get("/api/v1/audit/stats")
def audit_stats(
    _: None = Depends(require_permission(Permission.RBAC_MANAGE)),
) -> dict:
    store = get_audit_store()
    return {"total_entries": store.count()}


# ── Cloud Provider Pricing ──────────────────────────────────────

_pricing_service: CloudPricingService | None = None


def _get_pricing() -> CloudPricingService:
    global _pricing_service
    if _pricing_service is None:
        _pricing_service = CloudPricingService()
    return _pricing_service


@s24_router.get("/api/v1/cloud/pricing/{provider}")
def get_provider_pricing(provider: str, region: str = "us-east-1") -> list:
    svc = _get_pricing()
    if provider not in svc.get_all_providers():
        raise HTTPException(400, f"Unknown provider: {provider}. Use: {svc.get_all_providers()}")
    return [vars(r) for r in svc.get_pricing(provider, region)]


@s24_router.get("/api/v1/cloud/compare")
def compare_gpu_pricing(gpu_model: str, region: str = "us-east-1") -> list:
    return [vars(r) for r in _get_pricing().compare_gpu(gpu_model, region)]


@s24_router.get("/api/v1/cloud/estimate")
def estimate_cost(gpu_model: str, gpu_count: int = 1, provider: str = "aws", region: str = "us-east-1") -> dict:
    svc = _get_pricing()
    monthly = svc.estimate_monthly_cost(gpu_model, gpu_count, provider, region)
    spot = svc.get_spot_savings(gpu_model, gpu_count, provider, region)
    reserved = svc.get_reserved_savings(gpu_model, gpu_count, provider, region)
    return {
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "provider": provider,
        "region": region,
        "estimated_monthly_cost": monthly,
        "spot_savings": spot,
        "reserved_savings": reserved,
    }


@s24_router.get("/api/v1/cloud/providers")
def list_providers() -> dict:
    svc = _get_pricing()
    return {"providers": svc.get_all_providers()}


# ── Benchmark / Performance ────────────────────────────────────

@s24_router.get("/api/v1/benchmark")
def run_benchmark(
    repository: ClusterRepository = Depends(get_repository),
) -> dict:
    results: dict[str, float] = {}
    clusters = repository.list_clusters()

    t0 = time.perf_counter()
    for c in clusters[:10]:
        repository.get_cluster(c.id)
    results["get_cluster_10x_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    for c in clusters[:10]:
        repository.latest_state(c.id)
    results["latest_state_10x_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    for c in clusters[:10]:
        repository.latest_recommendations(c.id)
    results["latest_recommendations_10x_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    repository.list_clusters()
    results["list_clusters_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    results["cluster_count"] = len(clusters)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "results": results,
    }


@s24_router.get("/api/v1/ci/health")
def ci_health(
    repository: ClusterRepository = Depends(get_repository),
) -> dict:
    try:
        clusters = repository.list_clusters()
        state_count = sum(1 for c in clusters if repository.latest_state(c.id) is not None)
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cluster_count": len(clusters),
            "clusters_with_state": state_count,
            "version": __version__,
        }
    except Exception as exc:
        raise HTTPException(503, detail=f"Health check failed: {exc}")


# ── API Version Info ────────────────────────────────────────────

from fastapi.responses import Response


@s24_router.get("/api/version")
def api_version() -> dict:
    return {
        "current_version": __version__,
        "v1_path": "/api/v1",
        "v2_path": "/api/v2",
        "deprecation": "v1 is stable; v2 is preview",
    }
