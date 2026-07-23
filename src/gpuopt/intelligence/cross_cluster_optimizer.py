from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from ..config import get_settings
from ..repository import ClusterRepository
from ..schemas import ClusterStateData, WorkloadRequirements
from ._telemetry_utils import telemetry_map

logger = logging.getLogger(__name__)


@dataclass
class ClusterScore:
    cluster_id: str
    cluster_name: str
    environment: str
    total_gpus: int
    free_gpus: int
    avg_util_pct: float
    failure_risk: float
    cost_per_gpu_hr: float
    power_efficiency: float
    network_latency_ms: float
    composite_score: float
    rank: int = 0
    reasoning: list[str] = field(default_factory=list)


@dataclass
class PlacementCandidate:
    cluster_id: str
    cluster_name: str
    node_name: str
    gpu_indices: list[int]
    gpu_model: str
    free_memory_bytes: int
    score: float
    reasoning: str
    failure_probability: float
    estimated_cost_hr: float


@dataclass
class CrossClusterResult:
    id: str
    timestamp: str
    workload_description: str
    gpu_count: int
    gpu_memory_bytes: int
    candidates: list[PlacementCandidate]
    ranked_clusters: list[ClusterScore]
    recommended_cluster: str
    recommended_node: str
    estimated_savings_pct: float
    summary: str


class CrossClusterOptimizer:
    def __init__(self, repository: ClusterRepository | None = None):
        from ..dependencies import get_repository
        self.repository = repository or get_repository()
        self._rng = random.Random(42)

    def score_cluster(
        self,
        cluster_id: UUID,
        cluster_name: str,
        environment: str,
        state: ClusterStateData | None,
        requirements: WorkloadRequirements,
    ) -> ClusterScore:
        if state is None:
            return ClusterScore(
                cluster_id=str(cluster_id), cluster_name=cluster_name,
                environment=environment, total_gpus=0, free_gpus=0,
                avg_util_pct=0, failure_risk=0.5, cost_per_gpu_hr=3.5,
                power_efficiency=0.5, network_latency_ms=10,
                composite_score=0, reasoning=["No state data available"],
            )

        tmap = telemetry_map(state)
        total_gpus = state.gpu_count
        free_gpus = sum(
            1 for n in state.nodes for g in n.gpu_devices
            if g.memory_used_bytes < g.memory_total_bytes * 0.1
        )
        used_gpus = total_gpus - free_gpus
        avg_util = (used_gpus / max(total_gpus, 1)) * 100

        util_score = min(avg_util / 80, 1.0) * 25
        free_score = (free_gpus / max(total_gpus, 1)) * 20
        has_capacity = 1.0 if free_gpus >= requirements.gpu_count else 0.2
        capacity_score = has_capacity * 15

        gpu_models = set()
        for n in state.nodes:
            for g in n.gpu_devices:
                if g.model:
                    gpu_models.add(g.model.lower())
        is_premium = any("h100" in m or "h200" in m or "b200" in m for m in gpu_models)
        gpu_quality = 1.2 if is_premium else 1.0

        env_penalty = 0.85 if environment == "production" else 1.0

        unhealthy = 0
        for n in state.nodes:
            if n.status and n.status.lower() != "ready":
                unhealthy += 1
        health_score = max(0, 1 - unhealthy / max(len(state.nodes), 1)) * 15

        failure_risk = round(self._estimate_failure_risk(state, tmap), 4)
        risk_score = (1 - failure_risk) * 10

        base_rate = {"production": 4.5, "staging": 2.5, "development": 1.5}.get(environment, 3.0)
        cost_per_gpu_hr = base_rate * gpu_quality
        cost_score = max(0, 1 - (cost_per_gpu_hr - 1) / 5) * 5

        total_score = (util_score + free_score + capacity_score + health_score + risk_score + cost_score) * env_penalty

        reasoning = []
        if has_capacity >= 1:
            reasoning.append(f"Sufficient free GPUs ({free_gpus}/{total_gpus})")
        else:
            reasoning.append(f"Insufficient free GPUs ({free_gpus}/{total_gpus}, need {requirements.gpu_count})")
        if is_premium:
            reasoning.append(f"Premium GPU hardware ({', '.join(gpu_models)})")
        if avg_util > 70:
            reasoning.append(f"High utilization ({avg_util:.0f}%)")
        if failure_risk > 0.3:
            reasoning.append(f"Elevated failure risk ({failure_risk:.0%})")
        if cost_per_gpu_hr < 3:
            reasoning.append(f"Low cost (${cost_per_gpu_hr:.2f}/GPU-hr)")

        return ClusterScore(
            cluster_id=str(cluster_id), cluster_name=cluster_name,
            environment=environment, total_gpus=total_gpus,
            free_gpus=free_gpus, avg_util_pct=round(avg_util, 1),
            failure_risk=failure_risk, cost_per_gpu_hr=round(cost_per_gpu_hr, 2),
            power_efficiency=round(self._estimate_power_efficiency(state, tmap), 4),
            network_latency_ms=round(self._estimate_latency(state), 1),
            composite_score=round(total_score, 2),
            reasoning=reasoning,
        )

    def optimize(
        self,
        requirements: WorkloadRequirements,
        clusters_filter: list[UUID] | None = None,
        objective: str = "balanced",
    ) -> CrossClusterResult:
        clusters = self.repository.list_clusters()
        if clusters_filter:
            clusters = [c for c in clusters if c.id in clusters_filter]

        scored_clusters: list[ClusterScore] = []
        workload_desc = f"{requirements.gpu_count} GPU(s), {requirements.gpu_memory_bytes / (1024**3):.1f} GB memory"
        if requirements.gpu_model_preference:
            workload_desc += f", model: {requirements.gpu_model_preference}"

        for cluster in clusters:
            state = self.repository.latest_state(cluster.id)
            score = self.score_cluster(cluster.id, cluster.name, cluster.environment, state, requirements)
            scored_clusters.append(score)

        scored_clusters.sort(key=lambda x: -x.composite_score)
        for i, sc in enumerate(scored_clusters):
            sc.rank = i + 1

        candidates: list[PlacementCandidate] = []
        for sc in scored_clusters[:5]:
            state = self.repository.latest_state(UUID(sc.cluster_id))
            if not state:
                continue
            best_node, best_gpus, best_score, reason = self._find_best_node(state, requirements)
            if best_node:
                gpu_model = best_gpus[0].get("model", "unknown") if best_gpus else "unknown"
                free_mem = sum(g.get("memory_total_bytes", 0) - g.get("memory_used_bytes", 0) for g in best_gpus)
                base_rate = {"production": 4.5, "staging": 2.5, "development": 1.5}.get(sc.environment, 3.0)
                cost_hr = base_rate * len(best_gpus)

                candidates.append(PlacementCandidate(
                    cluster_id=sc.cluster_id, cluster_name=sc.cluster_name,
                    node_name=best_node.name, gpu_indices=[g["index"] for g in best_gpus],
                    gpu_model=gpu_model, free_memory_bytes=free_mem,
                    score=round(best_score, 2), reasoning=reason,
                    failure_probability=sc.failure_risk,
                    estimated_cost_hr=round(cost_hr, 2),
                ))

        recommended = candidates[0] if candidates else None
        if recommended:
            best_cluster_score = next((s for s in scored_clusters if s.cluster_id == recommended.cluster_id), None)
            worst_score = scored_clusters[-1] if len(scored_clusters) > 1 else best_cluster_score
            savings = 0.0
            if best_cluster_score and worst_score:
                cost_diff = worst_score.cost_per_gpu_hr - best_cluster_score.cost_per_gpu_hr
                savings = max(0, (cost_diff / max(worst_score.cost_per_gpu_hr, 0.01)) * 100)
        else:
            savings = 0.0

        summary_parts = []
        if recommended:
            summary_parts.append(f"Recommended: {recommended.cluster_name}/{recommended.node_name}")
            summary_parts.append(f"${recommended.estimated_cost_hr:.2f}/hr")
        if savings > 0:
            summary_parts.append(f"~{savings:.0f}% savings vs avg cluster")
        summary_parts.append(f"{len(candidates)} viable candidate(s)")

        return CrossClusterResult(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            workload_description=workload_desc,
            gpu_count=requirements.gpu_count,
            gpu_memory_bytes=requirements.gpu_memory_bytes,
            candidates=candidates,
            ranked_clusters=scored_clusters,
            recommended_cluster=recommended.cluster_name if recommended else "",
            recommended_node=recommended.node_name if recommended else "",
            estimated_savings_pct=round(savings, 1),
            summary="; ".join(summary_parts),
        )

    def _find_best_node(
        self, state: ClusterStateData, req: WorkloadRequirements
    ) -> tuple[Any, list[dict[str, Any]], float, str]:
        tmap = telemetry_map(state)
        best_score = -1.0
        best_node = None
        best_gpus_list: list[dict[str, Any]] = []
        best_reason = ""

        for node in state.nodes:
            free_gpu_dicts: list[dict[str, Any]] = []
            for gpu in node.gpu_devices:
                tg = tmap.get((node.name, gpu.index), {})
                if tg.get("memory_total_bytes", 0) - tg.get("memory_used_bytes", 0) >= req.gpu_memory_bytes:
                    free_gpu_dicts.append(tg)
            if len(free_gpu_dicts) < req.gpu_count:
                continue

            mem_free = sum(g.get("memory_total_bytes", 0) - g.get("memory_used_bytes", 0) for g in free_gpu_dicts)
            mem_score = min(mem_free / max(req.gpu_memory_bytes * req.gpu_count, 1), 2.0) * 30

            idle_bonus = sum(1 for g in free_gpu_dicts if g.get("memory_used_bytes", 0) < g.get("memory_total_bytes", 1) * 0.1)
            idle_score = (idle_bonus / max(len(free_gpu_dicts), 1)) * 25

            temp_penalty = sum(10 for g in free_gpu_dicts if g.get("temperature_celsius", 0) > 80)
            temp_score = max(0, 25 - temp_penalty)

            utils = [g.get("gpu_util_pct", 0) for g in free_gpu_dicts]
            util_variance = (max(utils) - min(utils)) if utils else 0
            balance_penalty = util_variance * 0.2
            balance_score = max(0, 20 - balance_penalty)

            score = mem_score + idle_score + temp_score + balance_score
            if score > best_score:
                best_score = score
                best_node = node
                best_gpus_list = free_gpu_dicts[:req.gpu_count]
                parts = []
                if mem_score > 20:
                    parts.append("abundant memory")
                if idle_bonus >= req.gpu_count:
                    parts.append("idle GPUs")
                if temp_score > 20:
                    parts.append("low temperature")
                if balance_score > 15:
                    parts.append("balanced load")
                best_reason = "; ".join(parts) if parts else "acceptable fit"

        return best_node, best_gpus_list, best_score, best_reason

    def _estimate_failure_risk(self, state: ClusterStateData, tmap: dict | None = None) -> float:
        if tmap is None:
            tmap = telemetry_map(state)
        risk_factors = 0.0
        total_gpus = 0
        for node in state.nodes:
            for gpu in node.gpu_devices:
                total_gpus += 1
                tg = tmap.get((node.name, gpu.index), {})
                if tg.get("temperature_celsius", 0) > 85:
                    risk_factors += 0.15
                if tg.get("ecc_errors_total", 0) > 10:
                    risk_factors += 0.1
                if tg.get("ecc_errors_aggregate", 0) > 50:
                    risk_factors += 0.05
                pl = tg.get("power_limit_watts", 0)
                pw = tg.get("power_watts", 0)
                if pw > 0 and pl > 0 and pw / pl > 0.95:
                    risk_factors += 0.08
        return min(risk_factors / max(total_gpus, 1), 0.95)

    def _estimate_power_efficiency(self, state: ClusterStateData, tmap: dict | None = None) -> float:
        if tmap is None:
            tmap = telemetry_map(state)
        total_util = 0.0
        total_power = 0.0
        for node in state.nodes:
            for gpu in node.gpu_devices:
                tg = tmap.get((node.name, gpu.index), {})
                total_util += tg.get("gpu_util_pct", 0)
                total_power += tg.get("power_watts", 0)
        if total_power == 0:
            return 0.5
        return min(total_util / max(total_power, 1) * 10, 1.0)

    def _estimate_latency(self, state: ClusterStateData) -> float:
        base = self._rng.uniform(1, 20)
        node_count = max(len(state.nodes), 1)
        return round(base + node_count * 0.5, 1)
