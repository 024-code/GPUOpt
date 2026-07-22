from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from gpuopt.ml.forecast_model import ForecastModel
from gpuopt.repository import ClusterRepository
from gpuopt.scheduler.rl_scheduler import RLScheduler
from gpuopt.schemas import (
    ClusterStateData,
    DemandForecast,
    DemandForecastPoint,
    NodeResource,
    PlacementSuggestion,
    ScheduleSimulation,
    SchedulingPlan,
    WorkloadRequirements,
)

logger = logging.getLogger(__name__)


class SchedulerService:
    def __init__(self, repository: ClusterRepository, forecast_model: ForecastModel | None = None,
                 rl_scheduler: RLScheduler | None = None) -> None:
        self.repository = repository
        self.forecast_model = forecast_model or ForecastModel()
        self.rl_scheduler = rl_scheduler or RLScheduler()

    def rl_suggest_placement(self, cluster_id: UUID, requirements: WorkloadRequirements) -> PlacementSuggestion:
        from .rl_scheduler import Job, Node as RLNode
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        state = self.repository.latest_state(cluster_id)
        if state is None:
            raise KeyError("No cluster state available for RL placement")

        nodes = [RLNode(
            id=n.name, available_gpus=sum(1 for g in n.gpu_devices if g.memory_used_bytes == 0),
            total_gpus=len(n.gpu_devices), free_memory_gb=sum(
                (g.memory_total_bytes - g.memory_used_bytes) / (1024**3) for g in n.gpu_devices
            ) / max(len(n.gpu_devices), 1),
            gpu_model=n.gpu_devices[0].model if n.gpu_devices else "unknown",
        ) for n in state.nodes]

        job = Job(
            id=str(uuid4()), required_gpus=requirements.gpu_count,
            priority=5, estimated_duration=1.0, memory_gb=requirements.gpu_memory_bytes / (1024**3),
        )

        result = self.rl_scheduler.schedule(job, nodes)
        if not result.node or not result.success:
            return self.suggest_placement(cluster_id, requirements)

        return PlacementSuggestion(
            cluster_id=cluster_id, cluster_name=cluster.name,
            environment=cluster.environment, workload=requirements,
            suggested_node=result.node_id, alternative_nodes=[],
            confidence=min(1.0, abs(result.q_value)), reasoning=result.reasoning,
            projected_impact=f"RL placed {requirements.gpu_count} GPU(s) on {result.node_id}",
            estimated_fragmentation_after=0.0, score=round(result.reward * 100, 1),
        )

    def forecast_demand(self, cluster_id: UUID, horizon_hours: int = 24) -> DemandForecast:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        traces = self.repository.list_traces(cluster_id, limit=100)
        if not traces:
            raise KeyError("No trace data available for demand forecasting")

        states = [s for _, s in traces]
        trace_count = len(states)

        for s in states:
            self.forecast_model.update_history(s)

        ml_points = self.forecast_model.forecast_gpu_utilization(horizon_hours, steps=horizon_hours)
        predicted_idle = self.forecast_model.predict_idle_gpus(
            sum(s.gpu_count for s in states) / max(trace_count, 1)
        )
        predicted_peak_mem = self.forecast_model.predict_peak_memory()
        predicted_avg_util = self.forecast_model.predict_avg_utilization()

        if not ml_points:
            raise KeyError("Insufficient data for ML forecast")

        summary = (
            f"ML Forecast {horizon_hours}h from {trace_count} trace(s): "
            f"avg util {predicted_avg_util:.0f}%, "
            f"predicted idle GPUs: {predicted_idle:.1f}"
        )

        return DemandForecast(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            horizon_hours=horizon_hours,
            trace_count=trace_count,
            predicted_idle_gpus=round(predicted_idle, 1),
            predicted_peak_gpu_memory_bytes=round(predicted_peak_mem * (1024**3), 1) if predicted_peak_mem > 0 else 0.0,
            predicted_avg_utilization_percent=round(predicted_avg_util, 1),
            forecast_points=ml_points,
            summary=summary,
        )

    def suggest_placement(self, cluster_id: UUID, requirements: WorkloadRequirements) -> PlacementSuggestion:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        state = self.repository.latest_state(cluster_id)
        if state is None:
            raise KeyError("No cluster state available for placement suggestion")

        nodes = self._score_nodes(state, requirements)
        if not nodes:
            raise KeyError("No suitable node found for the given requirements")

        best = nodes[0]
        alts = [n["name"] for n in nodes[1:4]]
        frag = self._estimate_fragmentation(state, requirements, best["name"])

        return PlacementSuggestion(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            workload=requirements,
            suggested_node=best["name"],
            alternative_nodes=alts,
            confidence=best["score"] / 100.0,
            reasoning=best["reasoning"],
            projected_impact=f"Placing {requirements.gpu_count} GPU(s) on {best['name']}",
            estimated_fragmentation_after=round(frag, 1),
            score=round(best["score"], 1),
        )

    def simulate_placement(self, cluster_id: UUID, requirements: WorkloadRequirements) -> ScheduleSimulation:
        placement = self.suggest_placement(cluster_id, requirements)
        state = self.repository.latest_state(cluster_id)
        cluster_info = self.repository.get_cluster(cluster_id)

        if state:
            used_gpus = sum(1 for n in state.nodes for g in n.gpu_devices if g.memory_used_bytes > 0)
            total_gpus = max(sum(1 for n in state.nodes for _ in n.gpu_devices), 1)
            current_util = used_gpus / total_gpus * 100
            projected_util = min(current_util + (requirements.gpu_count / total_gpus) * 20, 100)
            util_delta = projected_util - current_util
        else:
            util_delta = 0.0

        cluster_name = cluster_info.name if cluster_info else placement.cluster_name
        environment = cluster_info.environment if cluster_info else placement.environment

        return ScheduleSimulation(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            environment=environment,
            placement=placement,
            projected_utilization_delta=round(util_delta, 1),
            projected_memory_fragmentation=round(placement.estimated_fragmentation_after, 1),
            projected_pod_density=round(50.0 + util_delta, 1),
            efficiency_gain=round(util_delta * 0.3, 1),
            risk_score=round((1 - placement.confidence) * 100, 1),
            summary=f"Placing {requirements.gpu_count} GPU workload on {placement.suggested_node}: "
                    f"utilization +{util_delta:.1f}%, risk {((1 - placement.confidence) * 100):.0f}%",
        )

    def get_plan(self, cluster_id: UUID) -> SchedulingPlan:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        state = self.repository.latest_state(cluster_id)
        analysis = self.repository.latest_analysis(cluster_id)

        if state is None:
            raise KeyError("No cluster state available for scheduling plan")

        total_gpus = state.gpu_count
        free_gpus = sum(
            1 for n in state.nodes for g in n.gpu_devices
            if g.memory_used_bytes < g.memory_total_bytes * 0.1
        )
        node_count = state.node_count

        avg_util = 0.0
        if analysis:
            avg_util = analysis.overall_efficiency_score

        consolidations = max(free_gpus // 2, 0) if total_gpus > 0 else 0

        suggestions: list[str] = []
        for node in state.nodes:
            free = [g for g in node.gpu_devices if g.memory_used_bytes < g.memory_total_bytes * 0.1]
            if len(free) == len(node.gpu_devices) and len(free) > 0:
                suggestions.append(f"Consolidate workloads from {node.name} ({len(free)} idle GPUs)")

        rec_node_count = max(node_count - consolidations // 4, 1) if consolidations > 0 else node_count

        savings = (node_count - rec_node_count) * 24 * 0.045 if node_count > rec_node_count else 0.0

        summary = (
            f"{free_gpus}/{total_gpus} GPUs free across {node_count} node(s); "
            f"avg util {avg_util:.0f}%; "
            f"recommend {rec_node_count} node(s)"
        )

        return SchedulingPlan(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            total_gpus=total_gpus,
            free_gpus=free_gpus,
            node_count=node_count,
            avg_gpu_utilization=round(avg_util, 1),
            suggested_consolidations=consolidations,
            suggested_placements=suggestions,
            recommended_node_counts=rec_node_count,
            estimated_savings_gpu_hours=round(savings, 1),
            summary=summary,
        )

    @staticmethod
    def _score_nodes(state: ClusterStateData, req: WorkloadRequirements) -> list[dict]:
        scored: list[dict] = []
        for node in state.nodes:
            free_gpus = [
                g for g in node.gpu_devices
                if g.memory_total_bytes - g.memory_used_bytes >= req.gpu_memory_bytes
                   or req.gpu_memory_bytes == 0
            ]
            if len(free_gpus) < req.gpu_count:
                continue

            mem_score = sum(
                g.memory_total_bytes - g.memory_used_bytes for g in free_gpus
            ) / max(len(free_gpus), 1)
            mem_score = min(mem_score / max(req.gpu_memory_bytes, 1), 1.0) * 40 if req.gpu_memory_bytes > 0 else 30

            idle_score = sum(
                1 for g in free_gpus if g.memory_used_bytes < g.memory_total_bytes * 0.1
            ) / max(len(free_gpus), 1) * 30

            pod_score = max(0, 1 - node.pod_count / max(node.pod_capacity, 1)) * 30

            total = mem_score + idle_score + pod_score

            parts = []
            if mem_score > 20:
                parts.append("sufficient free memory")
            if idle_score > 15:
                parts.append("idle GPU capacity")
            if pod_score > 15:
                parts.append("low pod density")

            scored.append({
                "name": node.name,
                "score": total,
                "reasoning": "; ".join(parts) if parts else "acceptable fit",
            })

        scored.sort(key=lambda x: -x["score"])
        return scored

    @staticmethod
    def _estimate_fragmentation(state: ClusterStateData, req: WorkloadRequirements, target_node: str) -> float:
        for node in state.nodes:
            if node.name == target_node:
                free = sum(
                    g.memory_total_bytes - g.memory_used_bytes
                    for g in node.gpu_devices
                )
                total = sum(g.memory_total_bytes for g in node.gpu_devices)
                if total == 0:
                    return 0.0
                after = (free - req.gpu_memory_bytes) / total * 100
                return max(after, 0.0)
        return 0.0
