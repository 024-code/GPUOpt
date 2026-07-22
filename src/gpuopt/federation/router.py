from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..config import get_settings
from ..registry import get_registry
from .models import ClusterHealth, FederatedCluster, FederationRole
from .registry import FederatedClusterRegistry
from .scheduler import FederatedScheduler

logger = logging.getLogger(__name__)

federation_router = APIRouter(prefix="/api/v1/federation", tags=["federation"])


def _get_registry() -> FederatedClusterRegistry:
    reg = get_registry()
    return reg.get_or_create("federation_registry", FederatedClusterRegistry)


def _get_fed_scheduler() -> FederatedScheduler:
    reg = get_registry()
    existing = reg.get("federation_scheduler")
    if existing is not None:
        return existing
    fed = FederatedScheduler(_get_registry())
    reg.register("federation_scheduler", fed, force=True)
    return fed


# ── Cluster Management ───────────────────────────────────────


@federation_router.post("/clusters")
def register_cluster(data: dict[str, Any]) -> dict[str, Any]:
    cluster = _get_registry().register(
        name=data["name"],
        endpoint=data.get("endpoint", ""),
        environment=data.get("environment", "sandbox"),
        region=data.get("region", ""),
        labels=data.get("labels"),
        options=data.get("options"),
    )
    return {"status": "registered", "cluster": cluster.model_dump(mode="json")}


@federation_router.delete("/clusters/{cluster_id}")
def unregister_cluster(cluster_id: str) -> dict[str, Any]:
    if _get_registry().unregister(cluster_id):
        return {"status": "unregistered"}
    raise HTTPException(404, "Cluster not found")


@federation_router.get("/clusters")
def list_clusters() -> list[dict[str, Any]]:
    return [c.model_dump(mode="json") for c in _get_registry().list()]


@federation_router.get("/clusters/{cluster_id}")
def get_cluster(cluster_id: str) -> dict[str, Any]:
    cluster = _get_registry().get(cluster_id)
    if cluster is None:
        raise HTTPException(404, "Cluster not found")
    return cluster.model_dump(mode="json")


@federation_router.put("/clusters/{cluster_id}/health")
def update_cluster_health(cluster_id: str, data: dict[str, Any]) -> dict[str, Any]:
    health = ClusterHealth(data.get("health", "unknown"))
    cluster = _get_registry().update_health(
        cluster_id, health,
        total_gpus=data.get("total_gpus", 0),
        free_gpus=data.get("free_gpus", 0),
        gpu_models=data.get("gpu_models"),
        avg_utilization=data.get("avg_utilization", 0.0),
    )
    if cluster is None:
        raise HTTPException(404, "Cluster not found")
    return {"status": "updated", "cluster": cluster.model_dump(mode="json")}


# ── Cross-Cluster Scheduling ─────────────────────────────────


@federation_router.post("/schedule")
def federated_schedule(data: dict[str, Any]) -> dict[str, Any]:
    return _get_fed_scheduler().schedule_across_clusters(
        required_gpus=data.get("required_gpus", 1),
        gpu_model=data.get("gpu_model", ""),
        priority=data.get("priority", 5),
        memory_gb=data.get("memory_gb", 8.0),
        region=data.get("region", ""),
        workload_name=data.get("workload_name", ""),
    )


@federation_router.get("/workloads")
def federated_workloads() -> list[dict[str, Any]]:
    return [w.model_dump(mode="json") for w in _get_fed_scheduler().list_workloads()]


@federation_router.get("/workloads/{workload_id}")
def federated_workload(workload_id: str) -> dict[str, Any]:
    wl = _get_fed_scheduler().get_workload(workload_id)
    if wl is None:
        raise HTTPException(404, "Workload not found")
    return wl.model_dump(mode="json")


@federation_router.get("/state")
def federation_state() -> dict[str, Any]:
    return _get_fed_scheduler().get_state()
