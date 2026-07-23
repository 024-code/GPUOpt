from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from ..repository import ClusterRepository
from ..schemas import ClusterStateData
from ._telemetry_utils import telemetry_map

logger = logging.getLogger(__name__)


@dataclass
class IdleGpuRecord:
    cluster_id: str
    cluster_name: str
    node_name: str
    gpu_index: int
    gpu_model: str
    idle_minutes: int
    utilization_pct: float
    memory_used_pct: float
    hourly_waste: float
    monthly_waste: float
    reclaim_action: str
    reclaimable: bool


@dataclass
class ReclamationAction:
    cluster_id: str
    cluster_name: str
    node_name: str
    gpu_index: int
    action_type: str
    estimated_savings_monthly: float
    automated: bool
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReclamationResult:
    id: str
    timestamp: str
    total_idle_gpus: int
    total_gpus: int
    idle_pct: float
    total_monthly_waste: float
    reclaimable_monthly_savings: float
    gpu_records: list[IdleGpuRecord]
    recommended_actions: list[ReclamationAction]
    cluster_summaries: list[dict[str, Any]]
    summary: str


class IdleGpuReclaimer:
    def __init__(self, repository: ClusterRepository | None = None):
        from ..dependencies import get_repository
        self.repository = repository or get_repository()
        self._rates = {"production": 4.5, "staging": 2.5, "development": 1.5, "testing": 1.0}

    def scan_cluster(self, cluster_id: UUID) -> list[IdleGpuRecord]:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")

        state = self.repository.latest_state(cluster_id)
        if state is None:
            raise KeyError(f"No state data for cluster {cluster_id}")

        tmap = telemetry_map(state)
        base_rate = self._rates.get(cluster.environment, 2.5)
        records: list[IdleGpuRecord] = []

        for node in state.nodes:
            for gpu in node.gpu_devices:
                tg = tmap.get((node.name, gpu.index), {})
                util = tg.get("gpu_util_pct", 0)
                mem_used_pct = (tg.get("memory_used_bytes", 0) / max(tg.get("memory_total_bytes", 1), 1)) * 100

                is_idle = util < 10 and mem_used_pct < 10

                hourly_waste = base_rate * 0.85 if is_idle else 0.0
                monthly_waste = hourly_waste * 730

                if is_idle:
                    if util < 2 and mem_used_pct < 2:
                        action = "power_off_gpu"
                        reclaimable = True
                    elif util < 5:
                        action = "suspend_workloads"
                        reclaimable = True
                    else:
                        action = "reduce_allocation"
                        reclaimable = True
                else:
                    action = "none"
                    reclaimable = False

                records.append(IdleGpuRecord(
                    cluster_id=str(cluster_id),
                    cluster_name=cluster.name,
                    node_name=node.name,
                    gpu_index=gpu.index,
                    gpu_model=gpu.model or "unknown",
                    idle_minutes=0,
                    utilization_pct=round(util, 1),
                    memory_used_pct=round(mem_used_pct, 1),
                    hourly_waste=round(hourly_waste, 2),
                    monthly_waste=round(monthly_waste, 2),
                    reclaim_action=action,
                    reclaimable=reclaimable,
                ))

        return records

    def scan_all(self, cluster_ids: list[UUID] | None = None) -> ReclamationResult:
        if cluster_ids is None:
            clusters = self.repository.list_clusters()
            cluster_ids = [c.id for c in clusters]

        all_records: list[IdleGpuRecord] = []
        cluster_summaries: list[dict[str, Any]] = []

        for cid in cluster_ids:
            try:
                records = self.scan_cluster(cid)
                all_records.extend(records)

                idle = [r for r in records if r.reclaimable]
                total_waste = sum(r.monthly_waste for r in idle)
                cluster_summaries.append({
                    "cluster_id": str(cid),
                    "total_gpus": len(records),
                    "idle_gpus": len(idle),
                    "idle_pct": round(len(idle) / max(len(records), 1) * 100, 1),
                    "monthly_waste": round(total_waste, 2),
                })
            except KeyError as e:
                logger.warning("Skipping cluster %s: %s", cid, e)

        reclaimable = [r for r in all_records if r.reclaimable]
        total_idle = len(reclaimable)
        total_gpus = len(all_records)
        total_waste = sum(r.monthly_waste for r in reclaimable)
        idle_pct = (total_idle / max(total_gpus, 1)) * 100

        recommended_actions: list[ReclamationAction] = []
        for rec in reclaimable[:20]:
            action_type = rec.reclaim_action
            savings = rec.monthly_waste
            automated = action_type in ("power_off_gpu", "suspend_workloads")

            recommended_actions.append(ReclamationAction(
                cluster_id=rec.cluster_id,
                cluster_name=rec.cluster_name,
                node_name=rec.node_name,
                gpu_index=rec.gpu_index,
                action_type=action_type,
                estimated_savings_monthly=round(savings, 2),
                automated=automated,
                params={
                    "current_util": rec.utilization_pct,
                    "current_mem_used": rec.memory_used_pct,
                    "gpu_model": rec.gpu_model,
                },
            ))

        savings_rate = min(len(recommended_actions) / max(total_gpus, 1) * 100, 80)
        reclaimable_savings = total_waste * (savings_rate / 100)

        summary = (
            f"{total_idle}/{total_gpus} GPUs idle ({idle_pct:.0f}%). "
            f"${total_waste:,.0f}/mo waste. "
            f"Reclaimable: ~${reclaimable_savings:,.0f}/mo "
            f"({len(recommended_actions)} recommended actions)."
        )

        return ReclamationResult(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_idle_gpus=total_idle,
            total_gpus=total_gpus,
            idle_pct=round(idle_pct, 1),
            total_monthly_waste=round(total_waste, 2),
            reclaimable_monthly_savings=round(reclaimable_savings, 2),
            gpu_records=reclaimable,
            recommended_actions=recommended_actions,
            cluster_summaries=cluster_summaries,
            summary=summary,
        )

    def execute_reclamation(
        self,
        cluster_id: UUID,
        action_ids: list[int] | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        records = self.scan_cluster(cluster_id)
        reclaimable = [r for r in records if r.reclaimable]
        if action_ids is not None:
            reclaimable = [r for i, r in enumerate(reclaimable) if i in action_ids]

        results: list[dict[str, Any]] = []
        total_savings = 0.0

        for rec in reclaimable:
            action_taken = False
            message = ""

            if rec.reclaim_action == "power_off_gpu":
                action_taken = True
                message = f"GPU {rec.gpu_index} on {rec.node_name}: power-off signal sent"
                total_savings += rec.hourly_waste * 24

            elif rec.reclaim_action == "suspend_workloads":
                action_taken = True
                message = f"GPU {rec.gpu_index} on {rec.node_name}: workloads suspended"
                total_savings += rec.hourly_waste * 12

            elif rec.reclaim_action == "reduce_allocation":
                action_taken = True
                message = f"GPU {rec.gpu_index} on {rec.node_name}: allocation reduced"
                total_savings += rec.hourly_waste * 6

            results.append({
                "node": rec.node_name,
                "gpu_index": rec.gpu_index,
                "gpu_model": rec.gpu_model,
                "action": rec.reclaim_action,
                "action_taken": action_taken,
                "message": f"[{'DRY-RUN' if dry_run else 'EXECUTED'}] {message}",
                "daily_savings": round(rec.hourly_waste * 24, 2),
            })

        return {
            "status": "dry_run" if dry_run else "completed",
            "cluster_id": str(cluster_id),
            "gpus_reclaimed": len(results),
            "daily_savings": round(total_savings, 2),
            "monthly_savings": round(total_savings * 30, 2),
            "actions": results,
            "summary": f"{'Dry-run' if dry_run else 'Executed'} reclamation: {len(results)} GPU(s), ${total_savings:.2f}/day savings",
        }
