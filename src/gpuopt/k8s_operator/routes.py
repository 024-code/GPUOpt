from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_repository
from ..rbac import Permission, require_permission
from ..repository import ClusterRepository
from .controller import GPUOptimizationController
from .models import (
    GPUOptimizationAction,
    GPUOptimizationProfile,
    GPUWorkloadProfile,
    OptimizationProfileSpec,
    ActionSpec,
    ActionType,
    ActionPhase,
    WorkloadProfileSpec,
)
from .adapters import ActionAdapter
from .client import K8sClientWrapper

k8s_router = APIRouter(prefix="/api/v1/k8s", tags=["k8s-integration"])

_client = K8sClientWrapper()
_adapter = ActionAdapter(_client)
_controller = GPUOptimizationController(client=_client, adapter=_adapter)


@k8s_router.get("/profiles")
def list_profiles(
    namespace: str = "",
    _: None = Depends(require_permission(Permission.CLUSTER_READ)),
) -> list:
    items = _client.list_gpuoptimization_profiles(namespace)
    return items


@k8s_router.get("/actions")
def list_actions(
    namespace: str = "",
    _: None = Depends(require_permission(Permission.CLUSTER_READ)),
) -> list:
    items = _client.list_gpuoptimization_actions(namespace)
    return items


@k8s_router.get("/nodes")
def list_nodes(
    label_selector: str = "",
    _: None = Depends(require_permission(Permission.CLUSTER_READ)),
) -> list:
    return _client.list_nodes(label_selector)


@k8s_router.get("/pods")
def list_pods(
    namespace: str = "",
    label_selector: str = "",
    _: None = Depends(require_permission(Permission.CLUSTER_READ)),
) -> list:
    return _client.list_pods(namespace, label_selector)


@k8s_router.post("/reconcile")
def trigger_reconcile(
    _: None = Depends(require_permission(Permission.ACTUATE_LIVE)),
) -> dict:
    return _controller.reconcile_once()


@k8s_router.post("/controller/start")
def start_controller(
    poll_interval: float = Query(30.0, description="Reconciliation interval in seconds"),
    _: None = Depends(require_permission(Permission.ACTUATE_LIVE)),
) -> dict:
    _controller._poll_interval = poll_interval
    _controller.start()
    return {
        "status": "controller_started",
        "poll_interval": poll_interval,
        "running": _controller._running,
    }


@k8s_router.post("/controller/stop")
def stop_controller(
    _: None = Depends(require_permission(Permission.ACTUATE_LIVE)),
) -> dict:
    _controller.stop()
    return {"status": "controller_stopped", "running": _controller._running}


@k8s_router.get("/controller/status")
def controller_status() -> dict:
    thread_alive = _controller._thread is not None and _controller._thread.is_alive()
    return {
        "running": _controller._running,
        "thread_alive": thread_alive,
        "poll_interval": _controller._poll_interval,
    }


@k8s_router.post("/actions/execute")
def execute_action(
    action: GPUOptimizationAction,
    _: None = Depends(require_permission(Permission.ACTUATE_LIVE)),
) -> dict:
    if not action.name:
        action.name = f"manual-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    return _adapter.execute(action)


@k8s_router.post("/actions/from-recommendation/{cluster_id}/{rec_id}")
def create_action_from_recommendation(
    cluster_id: str,
    rec_id: str,
    dry_run: bool = True,
    _: None = Depends(require_permission(Permission.ACTUATE_LIVE)),
    repository: ClusterRepository = Depends(get_repository),
) -> dict:
    from uuid import UUID
    cluster = repository.get_cluster(UUID(cluster_id))
    if cluster is None:
        raise HTTPException(404, "Cluster not found")
    recs = repository.latest_recommendations(UUID(cluster_id))
    if recs is None:
        raise HTTPException(404, "No recommendations found")
    target = next((r for r in recs.recommendations if str(r.id) == rec_id), None)
    if target is None:
        raise HTTPException(404, "Recommendation not found")

    action = GPUOptimizationAction(
        name=f"rec-{rec_id[:8]}",
        namespace="default",
        spec=ActionSpec(
            actionType=ActionType.APPLY_RECOMMENDATION,
            targetCluster=cluster_id,
            recommendationRef=rec_id,
            parameters={
                "reason": f"From recommendation {target.title if target else ''}",
                "dryRun": dry_run,
            },
            approvalRequired=True,
        ),
    )
    result = _adapter.execute(action)
    return {"action": action.model_dump(mode="json"), "result": result}
