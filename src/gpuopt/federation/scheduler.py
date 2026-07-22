from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ..scheduler.rl_scheduler import Job, Node as RLNode, RLScheduler
from .models import ClusterHealth, FederatedCluster, FederatedWorkload, FederationRole, WorkloadState
from .registry import FederatedClusterRegistry

logger = logging.getLogger(__name__)


class FederatedScheduler:
    def __init__(self, registry: FederatedClusterRegistry, rl_scheduler: RLScheduler | None = None) -> None:
        self._registry = registry
        self._rl = rl_scheduler or RLScheduler()
        self._workloads: dict[str, FederatedWorkload] = {}

    def find_best_cluster(self, required_gpus: int, gpu_model: str = "",
                          priority: int = 5, region: str = "") -> FederatedCluster | None:
        clusters = self._registry.list()
        candidates: list[tuple[float, FederatedCluster]] = []

        for c in clusters:
            if c.health != ClusterHealth.ONLINE:
                continue
            if c.role == FederationRole.DRAINING:
                continue
            if region and c.region != region:
                continue
            if c.free_gpus < required_gpus:
                continue
            if gpu_model and gpu_model not in c.gpu_models:
                continue

            score = c.free_gpus / max(c.total_gpus, 1) * 0.6
            score += (1 - c.avg_utilization / 100.0) * 0.3
            score += (priority / 10.0) * 0.1
            candidates.append((score, c))

        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1] if candidates else None

    def schedule_across_clusters(self, required_gpus: int, gpu_model: str = "",
                                 priority: int = 5, memory_gb: float = 8.0,
                                 region: str = "", workload_name: str = "") -> dict[str, Any]:
        best = self.find_best_cluster(required_gpus, gpu_model, priority, region)
        if best is None:
            return {"status": "queued", "reason": "No cluster available", "mode": "federated"}

        job = Job(
            id=str(uuid4()), required_gpus=required_gpus,
            priority=priority, estimated_duration=1.0,
            memory_gb=memory_gb, checkpointable=False,
        )
        nodes = [
            RLNode(
                id=f"{best.name}-node-{i}",
                available_gpus=best.free_gpus,
                total_gpus=best.total_gpus,
                free_memory_gb=max(64, memory_gb * 2),
                gpu_model=gpu_model or (best.gpu_models[0] if best.gpu_models else "A100"),
            )
            for i in range(max(1, best.total_gpus // 8))
        ]
        result = self._rl.schedule(job, nodes)

        wl = FederatedWorkload(
            name=workload_name or f"fed-{uuid4().hex[:8]}",
            required_gpus=required_gpus,
            priority=priority,
            memory_gb=memory_gb,
            state=WorkloadState.PENDING if result.node else WorkloadState.QUEUED,
            assigned_cluster=best.name,
            assigned_node=result.node_id if result.node else "",
            scheduler_type="rl",
        )
        self._workloads[wl.id] = wl

        return {
            "status": "scheduled" if result.node else "queued",
            "cluster": best.name,
            "node": result.node_id if result.node else "",
            "reasoning": result.reasoning,
            "q_value": result.q_value,
            "mode": "federated",
            "workload_id": wl.id,
        }

    def list_workloads(self) -> list[FederatedWorkload]:
        return list(self._workloads.values())

    def get_workload(self, workload_id: str) -> FederatedWorkload | None:
        return self._workloads.get(workload_id)

    def get_state(self) -> dict[str, Any]:
        clusters = self._registry.list()
        workloads = list(self._workloads.values())
        return {
            "clusters": [c.model_dump(mode="json") for c in clusters],
            "workloads": [w.model_dump(mode="json") for w in workloads],
            "total_clusters": len(clusters),
            "total_gpus": sum(c.total_gpus for c in clusters),
            "total_free_gpus": sum(c.free_gpus for c in clusters),
            "pending_workloads": sum(1 for w in workloads if w.state in (WorkloadState.PENDING, WorkloadState.QUEUED)),
            "running_workloads": sum(1 for w in workloads if w.state == WorkloadState.RUNNING),
        }
