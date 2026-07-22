from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..config import get_settings
from ..k8s_operator.client import K8sClientWrapper
from ..registry import get_registry
from .rl_scheduler import Job, Node, RLScheduler

logger = logging.getLogger(__name__)

scheduler_router = APIRouter(prefix="/api/v1/scheduler", tags=["scheduler"])


def _get_scheduler() -> RLScheduler:
    reg = get_registry()
    return reg.get_or_create("scheduler", RLScheduler)


def _get_k8s_client() -> K8sClientWrapper | None:
    reg = get_registry()
    existing = reg.get("k8s_client_adapter")
    if existing is not None:
        return existing
    client = K8sClientWrapper()
    reg.register("k8s_client_adapter", client, force=True)
    return client


# ── RL Scheduler ─────────────────────────────────────────────


@scheduler_router.post("/rl/schedule")
def schedule_job(data: dict[str, Any]) -> dict[str, Any]:
    nodes_data = data.get("nodes", [])
    if not nodes_data:
        return {"status": "error", "reason": "No nodes provided"}
    job = Job(
        id=data.get("id", ""),
        required_gpus=data.get("required_gpus", 1),
        priority=data.get("priority", 5),
        estimated_duration=data.get("estimated_duration", 1.0),
        memory_gb=data.get("memory_gb", 8.0),
        checkpointable=data.get("checkpointable", False),
    )
    nodes = [Node(**n) for n in nodes_data]
    result = _get_scheduler().schedule(job, nodes)
    if result.node:
        return {
            "status": "scheduled",
            "node_id": result.node_id,
            "explanation": result.reasoning,
            "reward": result.reward,
            "q_value": result.q_value,
        }
    return {"status": "queued", "reason": result.reasoning, "message": "Job queued — no eligible node"}


@scheduler_router.post("/rl/train")
def train_scheduler(episodes: int = 100) -> dict[str, Any]:
    if episodes <= 0:
        raise HTTPException(status_code=400, detail="episodes must be positive")
    _get_scheduler().train_from_history(episodes)
    metrics = _get_scheduler().metrics()
    return {
        "status": "training_complete",
        "episodes": episodes,
        "average_reward": metrics["average_reward"],
        "q_table_size": metrics["q_table_size"],
    }


@scheduler_router.get("/rl/metrics")
def scheduler_metrics() -> dict[str, Any]:
    return _get_scheduler().metrics()


@scheduler_router.post("/rl/save")
def save_model() -> dict[str, object]:
    _get_scheduler().save_model()
    return {"status": "saved", "path": str(_get_scheduler().model_path)}


@scheduler_router.post("/rl/load")
def load_model() -> dict[str, object]:
    _get_scheduler().load_model()
    return {"status": "loaded", "path": str(_get_scheduler().model_path), "q_table_size": len(_get_scheduler().q_table)}


# ── Kueue Integration ────────────────────────────────────────


@scheduler_router.get("/kueue/detect")
def detect_kueue() -> dict[str, Any]:
    client = _get_k8s_client()
    if client is None:
        return {"detected": False, "error": "No K8s client available"}
    from .kueue_adapter import KueueAdapter
    adapter = KueueAdapter(client, _get_scheduler())
    return adapter.detect()


@scheduler_router.get("/kueue/queues")
def list_kueue_queues() -> list[dict[str, Any]]:
    client = _get_k8s_client()
    if client is None:
        return []
    from .kueue_adapter import KueueAdapter
    adapter = KueueAdapter(client, _get_scheduler())
    return adapter.list_cluster_queues()


@scheduler_router.post("/kueue/submit")
def submit_kueue_workload(data: dict[str, Any]) -> dict[str, Any]:
    client = _get_k8s_client()
    if client is None:
        return {"status": "error", "reason": "No K8s client available"}
    from .kueue_adapter import KueueAdapter
    adapter = KueueAdapter(client, _get_scheduler())

    job = Job(
        id=data.get("id", ""),
        required_gpus=data.get("required_gpus", 1),
        priority=data.get("priority", 5),
        estimated_duration=data.get("estimated_duration", 1.0),
        memory_gb=data.get("memory_gb", 8.0),
        checkpointable=data.get("checkpointable", False),
    )
    nodes_data = data.get("nodes")
    nodes = [Node(**n) for n in nodes_data] if nodes_data else None

    return adapter.submit_workload(
        job=job,
        cluster_queue=data.get("cluster_queue", "gpu-queue"),
        namespace=data.get("namespace", "default"),
        priority_class=data.get("priority_class", ""),
        nodes=nodes,
    )


@scheduler_router.get("/kueue/workloads")
def list_kueue_workloads(namespace: str = "") -> list[dict[str, Any]]:
    client = _get_k8s_client()
    if client is None:
        return []
    from .kueue_adapter import KueueAdapter
    adapter = KueueAdapter(client, _get_scheduler())
    return adapter.list_workloads(namespace)


# ── Volcano Integration ──────────────────────────────────────


@scheduler_router.get("/volcano/detect")
def detect_volcano() -> dict[str, Any]:
    client = _get_k8s_client()
    if client is None:
        return {"detected": False, "error": "No K8s client available"}
    from .volcano_adapter import VolcanoAdapter
    adapter = VolcanoAdapter(client, _get_scheduler())
    return adapter.detect()


@scheduler_router.get("/volcano/queues")
def list_volcano_queues() -> list[dict[str, Any]]:
    client = _get_k8s_client()
    if client is None:
        return []
    from .volcano_adapter import VolcanoAdapter
    adapter = VolcanoAdapter(client, _get_scheduler())
    return adapter.list_queues()


@scheduler_router.post("/volcano/submit")
def submit_volcano_podgroup(data: dict[str, Any]) -> dict[str, Any]:
    client = _get_k8s_client()
    if client is None:
        return {"status": "error", "reason": "No K8s client available"}
    from .volcano_adapter import VolcanoAdapter
    adapter = VolcanoAdapter(client, _get_scheduler())

    job = Job(
        id=data.get("id", ""),
        required_gpus=data.get("required_gpus", 1),
        priority=data.get("priority", 5),
        estimated_duration=data.get("estimated_duration", 1.0),
        memory_gb=data.get("memory_gb", 8.0),
        checkpointable=data.get("checkpointable", False),
    )
    nodes_data = data.get("nodes")
    nodes = [Node(**n) for n in nodes_data] if nodes_data else None

    return adapter.submit_podgroup(
        job=job,
        queue=data.get("queue", "gpu-queue"),
        namespace=data.get("namespace", "default"),
        priority=data.get("priority", 5),
        nodes=nodes,
    )


@scheduler_router.get("/volcano/podgroups")
def list_volcano_podgroups(namespace: str = "") -> list[dict[str, Any]]:
    client = _get_k8s_client()
    if client is None:
        return []
    from .volcano_adapter import VolcanoAdapter
    adapter = VolcanoAdapter(client, _get_scheduler())
    return adapter.list_podgroups(namespace)
