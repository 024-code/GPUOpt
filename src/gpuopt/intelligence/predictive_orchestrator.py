from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from ..repository import ClusterRepository
from ..schemas import ClusterStateData
from ._telemetry_utils import telemetry_map

logger = logging.getLogger(__name__)


class ActionType(StrEnum):
    MIGRATE_WORKLOAD = "migrate_workload"
    POWER_CAP_GPU = "power_cap_gpu"
    SCALE_DOWN_NODE = "scale_down_node"
    PREEMPT_JOB = "preempt_job"
    CONSOLIDATE_GPUS = "consolidate_gpus"
    SCHEDULE_MAINTENANCE = "schedule_maintenance"
    ALERT_OPERATOR = "alert_operator"
    ADJUST_QUOTA = "adjust_quota"
    ENABLE_BACKFILL = "enable_backfill"
    NOTHING = "nothing"


class RiskLevel(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class OrchestrationAction:
    action_type: ActionType
    target_cluster_id: str
    target_node: str = ""
    target_gpu_indices: list[int] = field(default_factory=list)
    priority: int = 0
    reason: str = ""
    expected_impact: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    automated: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationPlan:
    id: str
    cluster_id: str
    cluster_name: str
    generated_at: str
    overall_health: str
    risk_level: RiskLevel
    actions: list[OrchestrationAction]
    predicted_failures: list[dict[str, Any]]
    idle_gpu_count: int
    total_gpu_count: int
    estimated_waste_monthly: float
    summary: str


class PredictiveOrchestrator:
    def __init__(self, repository: ClusterRepository | None = None):
        from ..dependencies import get_repository
        self.repository = repository or get_repository()

    def analyze_and_plan(self, cluster_id: UUID) -> OrchestrationPlan:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")

        state = self.repository.latest_state(cluster_id)
        if state is None:
            raise KeyError(f"No state data for cluster {cluster_id}")

        tmap = telemetry_map(state)

        actions: list[OrchestrationAction] = []
        predicted_failures: list[dict[str, Any]] = []
        total_gpus = state.gpu_count
        idle_gpus = 0
        total_waste = 0.0

        for node in state.nodes:
            for gpu in node.gpu_devices:
                tg = tmap.get((node.name, gpu.index), {})
                gpu_util = tg.get("gpu_util_pct", 0)
                mem_util = (tg.get("memory_used_bytes", 0) / max(tg.get("memory_total_bytes", 1), 1)) * 100
                temp = tg.get("temperature_celsius", 0)
                power = tg.get("power_watts", 0)
                power_limit = tg.get("power_limit_watts", 0)
                ecc_vol = tg.get("ecc_errors_total", 0)

                risk_score = 0.0
                risk_factors: list[str] = []

                if temp > 85:
                    risk_score += 0.35
                    risk_factors.append(f"Critical temp: {temp}°C")
                elif temp > 75:
                    risk_score += 0.15
                    risk_factors.append(f"High temp: {temp}°C")

                if ecc_vol > 15:
                    risk_score += 0.25
                    risk_factors.append(f"ECC errors: {ecc_vol}")
                elif ecc_vol > 5:
                    risk_score += 0.1

                if power_limit > 0:
                    power_ratio = power / power_limit
                    if power_ratio > 0.95:
                        risk_score += 0.2
                        risk_factors.append(f"Power spike: {power}/{power_limit}W")
                    elif power_ratio > 0.85:
                        risk_score += 0.08

                if mem_util > 95:
                    risk_score += 0.15
                    risk_factors.append(f"Memory pressure: {mem_util:.0f}%")

                if gpu_util < 5 and mem_util < 5:
                    idle_gpus += 1
                    total_waste += 0.85

                if risk_score > 0.5:
                    predicted_failures.append({
                        "node": node.name,
                        "gpu_index": gpu.index,
                        "risk_score": round(risk_score, 3),
                        "risk_factors": risk_factors,
                        "recommended_action": self._recommend_action(risk_score, risk_factors, node.name, gpu.index),
                    })

        total_gpu_count = max(total_gpus, 1)

        if idle_gpus > 0 and idle_gpus / total_gpu_count > 0.15:
            actions.append(OrchestrationAction(
                action_type=ActionType.CONSOLIDATE_GPUS,
                target_cluster_id=str(cluster_id),
                priority=3,
                reason=f"{idle_gpus}/{total_gpus} GPUs idle (>15% slack)",
                expected_impact=f"Save ~${idle_gpus * 0.85 * 730:.0f}/mo by consolidating",
                risk_level=RiskLevel.LOW,
                automated=True,
                params={"idle_gpus": idle_gpus, "consolidation_target": "consolidate_to_fewer_nodes"},
            ))

        for fail in predicted_failures[:3]:
            actions.append(OrchestrationAction(
                action_type=ActionType.MIGRATE_WORKLOAD,
                target_cluster_id=str(cluster_id),
                target_node=fail["node"],
                target_gpu_indices=[fail["gpu_index"]],
                priority=5 if fail["risk_score"] > 0.7 else 4,
                reason=f"Pre-emptive migration from {fail['node']} GPU {fail['gpu_index']}: {'; '.join(fail['risk_factors'])}",
                expected_impact=f"Mitigate {fail['risk_score']:.0%} failure probability",
                risk_level=RiskLevel.HIGH if fail["risk_score"] > 0.7 else RiskLevel.MEDIUM,
                automated=fail["risk_score"] < 0.7,
                params={"failure_risk": fail["risk_score"]},
            ))

        for node in state.nodes:
            for gpu in node.gpu_devices:
                tg = tmap.get((node.name, gpu.index), {})
                power = tg.get("power_watts", 0)
                power_limit = tg.get("power_limit_watts", 0)
                if power > 0 and power_limit > 0:
                    ratio = power / power_limit
                    if ratio > 0.9:
                        actions.append(OrchestrationAction(
                            action_type=ActionType.POWER_CAP_GPU,
                            target_cluster_id=str(cluster_id),
                            target_node=node.name,
                            target_gpu_indices=[gpu.index],
                            priority=2,
                            reason=f"GPU {gpu.index} at {ratio:.0%} power limit ({power}W/{power_limit}W)",
                            expected_impact="Reduce power by ~15%, maintain 95% performance",
                            risk_level=RiskLevel.LOW,
                            automated=True,
                            params={"current_power": power, "target_cap": int(power_limit * 0.85)},
                        ))
                        break

        if predicted_failures and len(predicted_failures) > total_gpu_count * 0.3:
            actions.append(OrchestrationAction(
                action_type=ActionType.SCHEDULE_MAINTENANCE,
                target_cluster_id=str(cluster_id),
                priority=5,
                reason=f"{len(predicted_failures)}/{total_gpus} GPUs at risk",
                expected_impact="Prevent cascading failures",
                risk_level=RiskLevel.CRITICAL,
                automated=False,
                params={"affected_gpus": len(predicted_failures), "maintenance_window_hours": 4},
            ))

        if not actions:
            actions.append(OrchestrationAction(
                action_type=ActionType.NOTHING,
                target_cluster_id=str(cluster_id),
                priority=0,
                reason="Cluster healthy, no action needed",
                expected_impact="None",
                risk_level=RiskLevel.NONE,
                automated=True,
            ))

        total_waste_monthly = total_waste * 730
        risk_levels = [a.risk_level for a in actions]
        overall_risk = RiskLevel.NONE
        if RiskLevel.CRITICAL in risk_levels:
            overall_risk = RiskLevel.CRITICAL
        elif RiskLevel.HIGH in risk_levels:
            overall_risk = RiskLevel.HIGH
        elif RiskLevel.MEDIUM in risk_levels:
            overall_risk = RiskLevel.MEDIUM
        elif RiskLevel.LOW in risk_levels:
            overall_risk = RiskLevel.LOW

        health = "critical" if overall_risk in (RiskLevel.CRITICAL, RiskLevel.HIGH) else (
            "degraded" if overall_risk == RiskLevel.MEDIUM else "healthy"
        )

        actions.sort(key=lambda a: -a.priority)

        summary = (
            f"Health: {health}, Risk: {overall_risk.value}. "
            f"{len(predicted_failures)} GPU(s) at risk, {idle_gpus} idle. "
            f"${total_waste_monthly:.0f}/mo estimated waste. "
            f"{len(actions)} action(s) recommended."
        )

        return OrchestrationPlan(
            id=str(uuid4()),
            cluster_id=str(cluster_id),
            cluster_name=cluster.name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            overall_health=health,
            risk_level=overall_risk,
            actions=actions,
            predicted_failures=predicted_failures,
            idle_gpu_count=idle_gpus,
            total_gpu_count=total_gpus,
            estimated_waste_monthly=round(total_waste_monthly, 2),
            summary=summary,
        )

    def _recommend_action(self, risk_score: float, factors: list[str], node: str, gpu_index: int) -> str:
        if risk_score > 0.7:
            return f"Immediate: migrate workloads from {node} GPU {gpu_index}"
        if risk_score > 0.4:
            return f"Prepare: reduce load on {node} GPU {gpu_index}, schedule inspection"
        return f"Monitor: track {node} GPU {gpu_index} for changes"

    def run_orchestration_cycle(self, cluster_ids: list[UUID] | None = None) -> list[OrchestrationPlan]:
        if cluster_ids is None:
            clusters = self.repository.list_clusters()
            cluster_ids = [c.id for c in clusters]

        plans: list[OrchestrationPlan] = []
        for cid in cluster_ids:
            try:
                plan = self.analyze_and_plan(cid)
                plans.append(plan)
            except KeyError as e:
                logger.warning("Skipping cluster %s: %s", cid, e)
        return plans
