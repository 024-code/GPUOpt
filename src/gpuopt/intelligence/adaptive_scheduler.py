from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from ..repository import ClusterRepository
from ..schemas import ClusterStateData, WorkloadRequirements
from ._telemetry_utils import telemetry_map

logger = logging.getLogger(__name__)


class SchedulingStrategy(StrEnum):
    LEAST_LOADED = "least_loaded"
    RISK_AWARE = "risk_aware"
    COST_OPTIMAL = "cost_optimal"
    POWER_EFFICIENT = "power_efficient"
    THERMAL_AWARE = "thermal_aware"
    BALANCED = "balanced"


@dataclass
class NodeFit:
    cluster_id: str
    cluster_name: str
    node_name: str
    gpu_indices: list[int]
    gpu_model: str
    free_gpus: int
    total_gpus: int
    avg_util_pct: float
    failure_probability: float
    temperature_avg: float
    power_efficiency: float
    cost_per_gpu_hr: float
    fit_score: float
    reasoning: str


@dataclass
class SchedulingDecision:
    id: str
    timestamp: str
    strategy: SchedulingStrategy
    workload_id: str
    gpu_count: int
    gpu_memory_bytes: int
    priority: int
    candidates: list[NodeFit]
    selected: NodeFit | None
    fallback_reason: str | None
    summary: str


class AdaptiveScheduler:
    def __init__(self, repository: ClusterRepository | None = None):
        from ..dependencies import get_repository
        self.repository = repository or get_repository()
        self._rng = random.Random(42)

    def schedule(
        self,
        requirements: WorkloadRequirements,
        strategy: SchedulingStrategy = SchedulingStrategy.BALANCED,
        available_cluster_ids: list[UUID] | None = None,
        workload_id: str = "",
    ) -> SchedulingDecision:
        clusters = self.repository.list_clusters()
        if available_cluster_ids:
            clusters = [c for c in clusters if c.id in available_cluster_ids]

        all_fits: list[NodeFit] = []
        for cluster in clusters:
            state = self.repository.latest_state(cluster.id)
            if not state:
                continue
            fits = self._evaluate_cluster(cluster.id, cluster.name, cluster.environment, state, requirements, strategy)
            all_fits.extend(fits)

        all_fits.sort(key=lambda f: -f.fit_score)

        selected = all_fits[0] if all_fits else None
        fallback_reason = None

        if not selected:
            for cluster in clusters:
                state = self.repository.latest_state(cluster.id)
                if not state:
                    continue
                for node in state.nodes:
                    total_free = sum(
                        1 for g in node.gpu_devices
                        if g.memory_total_bytes - g.memory_used_bytes >= requirements.gpu_memory_bytes
                    )
                    if total_free >= requirements.gpu_count:
                        gpu_indices = [g.index for g in node.gpu_devices if g.memory_total_bytes - g.memory_used_bytes >= requirements.gpu_memory_bytes][:requirements.gpu_count]
                        all_fits.append(NodeFit(
                            cluster_id=str(cluster.id),
                            cluster_name=cluster.name,
                            node_name=node.name,
                            gpu_indices=gpu_indices,
                            gpu_model=node.gpu_devices[0].model if node.gpu_devices else "unknown",
                            free_gpus=total_free,
                            total_gpus=len(node.gpu_devices),
                            avg_util_pct=0,
                            failure_probability=0.5,
                            temperature_avg=0,
                            power_efficiency=0.5,
                            cost_per_gpu_hr=3.5,
                            fit_score=10.0,
                            reasoning="Fallback: minimal capacity match",
                        ))
            all_fits.sort(key=lambda f: -f.fit_score)
            selected = all_fits[0] if all_fits else None
            if selected:
                fallback_reason = "No ideal candidate found; matched by capacity only"

        wid = workload_id or str(uuid4())

        summary_parts = []
        if selected:
            summary_parts.append(f"Selected: {selected.cluster_name}/{selected.node_name}")
            summary_parts.append(f"({len(selected.gpu_indices)} GPU(s))")
            summary_parts.append(f"Strategy: {strategy.value}")
            summary_parts.append(f"Score: {selected.fit_score:.1f}")
        else:
            summary_parts.append("No suitable placement found across all clusters")

        return SchedulingDecision(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy=strategy,
            workload_id=wid,
            gpu_count=requirements.gpu_count,
            gpu_memory_bytes=requirements.gpu_memory_bytes,
            priority=5,
            candidates=all_fits[:10],
            selected=selected,
            fallback_reason=fallback_reason,
            summary=" | ".join(summary_parts),
        )

    def _evaluate_cluster(
        self,
        cluster_id: UUID,
        cluster_name: str,
        environment: str,
        state: ClusterStateData,
        req: WorkloadRequirements,
        strategy: SchedulingStrategy,
    ) -> list[NodeFit]:
        tmap = telemetry_map(state)
        base_rate = {"production": 4.5, "staging": 2.5, "development": 1.5}.get(environment, 2.5)
        fits: list[NodeFit] = []

        for node in state.nodes:
            suitable_gpus: list[dict[str, Any]] = []
            for gpu in node.gpu_devices:
                tg = tmap.get((node.name, gpu.index), {})
                if tg.get("memory_total_bytes", 0) - tg.get("memory_used_bytes", 0) >= req.gpu_memory_bytes:
                    suitable_gpus.append(tg)

            if len(suitable_gpus) < req.gpu_count:
                continue

            gpu_indices = [g["index"] for g in suitable_gpus[:req.gpu_count]]
            gpu_model = suitable_gpus[0].get("model", "unknown") if suitable_gpus else "unknown"
            total_gpus = len(node.gpu_devices)
            free_gpus = len(suitable_gpus)
            sel = suitable_gpus[:req.gpu_count]

            avg_util = sum(g.get("gpu_util_pct", 0) for g in sel) / max(len(sel), 1)
            avg_temp = sum(g.get("temperature_celsius", 0) for g in sel) / max(len(sel), 1)
            avg_power = sum(g.get("power_watts", 0) for g in sel) / max(len(sel), 1)

            failure_prob = self._node_failure_probability(node, tmap)

            power_eff = 0.0
            if avg_power > 0:
                power_eff = min(avg_util / avg_power * 10, 1.0)

            cost = base_rate * req.gpu_count

            util_score = (100 - avg_util) / 100 * 25
            risk_score = (1 - failure_prob) * 20
            temp_score = 0.0
            if avg_temp < 60:
                temp_score = 20
            elif avg_temp < 75:
                temp_score = 12
            elif avg_temp < 85:
                temp_score = 5

            power_score = power_eff * 15
            cost_score = max(0, 1 - (cost - 1) / 6) * 10
            free_score = (free_gpus / max(total_gpus, 1)) * 10

            if strategy == SchedulingStrategy.LEAST_LOADED:
                fit_score = util_score + free_score
            elif strategy == SchedulingStrategy.RISK_AWARE:
                fit_score = risk_score + temp_score
            elif strategy == SchedulingStrategy.COST_OPTIMAL:
                fit_score = cost_score + util_score
            elif strategy == SchedulingStrategy.POWER_EFFICIENT:
                fit_score = power_score + temp_score * 0.5
            elif strategy == SchedulingStrategy.THERMAL_AWARE:
                fit_score = temp_score + risk_score
            else:
                fit_score = util_score + risk_score + temp_score + power_score + cost_score + free_score

            parts = []
            if avg_util < 40:
                parts.append("low utilization")
            if failure_prob < 0.15:
                parts.append("low risk")
            if avg_temp < 60:
                parts.append("cool")
            if power_eff > 0.6:
                parts.append("power efficient")
            if cost < 3:
                parts.append(f"${cost:.1f}/hr")
            reasoning = "; ".join(parts) if parts else "acceptable fit"

            fits.append(NodeFit(
                cluster_id=str(cluster_id),
                cluster_name=cluster_name,
                node_name=node.name,
                gpu_indices=gpu_indices,
                gpu_model=gpu_model,
                free_gpus=free_gpus,
                total_gpus=total_gpus,
                avg_util_pct=round(avg_util, 1),
                failure_probability=round(failure_prob, 4),
                temperature_avg=round(avg_temp, 1),
                power_efficiency=round(power_eff, 4),
                cost_per_gpu_hr=round(cost / max(req.gpu_count, 1), 2),
                fit_score=round(fit_score, 2),
                reasoning=reasoning,
            ))

        return fits

    def _node_failure_probability(self, node: Any, tmap: dict | None = None) -> float:
        prob = 0.0
        gpu_count = len(node.gpu_devices)
        if gpu_count == 0:
            return 0.3

        for gpu in node.gpu_devices:
            tg = (tmap or {}).get((node.name, gpu.index), {})
            if tg.get("temperature_celsius", 0) > 85:
                prob += 0.12
            elif tg.get("temperature_celsius", 0) > 75:
                prob += 0.05

            if tg.get("ecc_errors_total", 0) > 20:
                prob += 0.1
            elif tg.get("ecc_errors_total", 0) > 10:
                prob += 0.04

            if tg.get("ecc_errors_aggregate", 0) > 100:
                prob += 0.06

            pl = tg.get("power_limit_watts", 0)
            pw = tg.get("power_watts", 0)
            if pl > 0:
                ratio = pw / pl
                if ratio > 0.95:
                    prob += 0.08
                elif ratio > 0.85:
                    prob += 0.03

            if tg.get("clock_sm_mhz", 500) < 500:
                prob += 0.03

        return min(prob / max(gpu_count, 1), 0.95)
