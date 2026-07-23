from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_repository
from ..repository import ClusterRepository
from ..schemas import WorkloadRequirements
from .adaptive_scheduler import AdaptiveScheduler, SchedulingDecision, SchedulingStrategy
from .cross_cluster_optimizer import CrossClusterOptimizer, CrossClusterResult
from .idle_gpu_reclaimer import IdleGpuReclaimer, ReclamationResult
from .predictive_orchestrator import OrchestrationPlan, PredictiveOrchestrator

logger = logging.getLogger(__name__)

intelligence_router = APIRouter(prefix="/api/v2/intelligence", tags=["intelligence"])


def _cross_cluster_optimizer(repo: ClusterRepository = Depends(get_repository)) -> CrossClusterOptimizer:
    return CrossClusterOptimizer(repo)


def _predictive_orchestrator(repo: ClusterRepository = Depends(get_repository)) -> PredictiveOrchestrator:
    return PredictiveOrchestrator(repo)


def _idle_gpu_reclaimer(repo: ClusterRepository = Depends(get_repository)) -> IdleGpuReclaimer:
    return IdleGpuReclaimer(repo)


def _adaptive_scheduler(repo: ClusterRepository = Depends(get_repository)) -> AdaptiveScheduler:
    return AdaptiveScheduler(repo)


@intelligence_router.post("/cross-cluster/optimize", response_model=dict)
def cross_cluster_optimize(
    requirements: WorkloadRequirements,
    cluster_ids: list[str] | None = Query(None),
    objective: str = Query("balanced", pattern="^(balanced|cost|performance|reliability)$"),
    optimizer: CrossClusterOptimizer = Depends(_cross_cluster_optimizer),
) -> dict:
    cids = [UUID(c) for c in cluster_ids] if cluster_ids else None
    result = optimizer.optimize(requirements, clusters_filter=cids, objective=objective)
    return _dataclass_to_dict(result)


@intelligence_router.post("/orchestrate/plan", response_model=dict)
def orchestrate_plan(
    cluster_id: UUID,
    orchestrator: PredictiveOrchestrator = Depends(_predictive_orchestrator),
) -> dict:
    try:
        plan = orchestrator.analyze_and_plan(cluster_id)
        return _dataclass_to_dict(plan)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@intelligence_router.post("/orchestrate/run-cycle", response_model=dict)
def orchestrate_run_cycle(
    cluster_ids: list[str] | None = Query(None),
    orchestrator: PredictiveOrchestrator = Depends(_predictive_orchestrator),
) -> dict:
    cids = [UUID(c) for c in cluster_ids] if cluster_ids else None
    plans = orchestrator.run_orchestration_cycle(cluster_ids=cids)
    return {
        "plans": [_dataclass_to_dict(p) for p in plans],
        "total_clusters": len(plans),
        "critical_count": sum(1 for p in plans if p.risk_level.value == "critical"),
        "high_count": sum(1 for p in plans if p.risk_level.value == "high"),
    }


@intelligence_router.get("/idle-gpus/scan", response_model=dict)
def scan_idle_gpus(
    cluster_id: UUID | None = Query(None),
    reclaimer: IdleGpuReclaimer = Depends(_idle_gpu_reclaimer),
) -> dict:
    try:
        if cluster_id:
            records = reclaimer.scan_cluster(cluster_id)
            return {
                "cluster_id": str(cluster_id),
                "total_gpus": len(records),
                "idle_gpus": sum(1 for r in records if r.reclaimable),
                "records": [_dataclass_to_dict(r) for r in records],
                "total_monthly_waste": sum(r.monthly_waste for r in records if r.reclaimable),
            }
        else:
            result = reclaimer.scan_all()
            return _dataclass_to_dict(result)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@intelligence_router.post("/idle-gpus/reclaim", response_model=dict)
def reclaim_idle_gpus(
    cluster_id: UUID,
    action_ids: list[int] | None = Query(None),
    dry_run: bool = Query(True),
    reclaimer: IdleGpuReclaimer = Depends(_idle_gpu_reclaimer),
) -> dict:
    try:
        return reclaimer.execute_reclamation(cluster_id, action_ids=action_ids, dry_run=dry_run)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@intelligence_router.post("/adaptive/schedule", response_model=dict)
def adaptive_schedule(
    requirements: WorkloadRequirements,
    strategy: SchedulingStrategy = Query(SchedulingStrategy.BALANCED),
    cluster_ids: list[str] | None = Query(None),
    workload_id: str = Query(""),
    scheduler: AdaptiveScheduler = Depends(_adaptive_scheduler),
) -> dict:
    cids = [UUID(c) for c in cluster_ids] if cluster_ids else None
    decision = scheduler.schedule(
        requirements=requirements,
        strategy=strategy,
        available_cluster_ids=cids,
        workload_id=workload_id,
    )
    return _dataclass_to_dict(decision)


@intelligence_router.get("/health")
def intelligence_health(
    repo: ClusterRepository = Depends(get_repository),
) -> dict:
    clusters = repo.list_clusters()
    total_gpus = 0
    total_nodes = 0
    for c in clusters:
        state = repo.latest_state(c.id)
        if state:
            total_gpus += state.gpu_count
            total_nodes += state.node_count
    return {
        "status": "operational",
        "clusters_monitored": len(clusters),
        "total_gpus_tracked": total_gpus,
        "total_nodes_tracked": total_nodes,
        "modules": ["cross_cluster_optimizer", "predictive_orchestrator", "idle_gpu_reclaimer", "adaptive_scheduler"],
    }


def _dataclass_to_dict(obj: Any) -> dict:
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            val = getattr(obj, field_name)
            if hasattr(val, "__dataclass_fields__"):
                result[field_name] = _dataclass_to_dict(val)
            elif isinstance(val, (list, tuple)):
                result[field_name] = [_dataclass_to_dict(v) if hasattr(v, "__dataclass_fields__") else v for v in val]
            elif isinstance(val, dict):
                result[field_name] = {k: _dataclass_to_dict(v) if hasattr(v, "__dataclass_fields__") else v for k, v in val.items()}
            elif isinstance(val, StrEnum):
                result[field_name] = val.value
            else:
                result[field_name] = val
        return result
    if isinstance(obj, StrEnum):
        return obj.value
    return obj
