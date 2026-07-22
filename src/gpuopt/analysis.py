from __future__ import annotations

import logging
from uuid import UUID

from .repository import ClusterRepository
from .schemas import (
    GPUUtilizationTrend,
    NodeEfficiency,
    WorkloadAnalysisResult,
)

logger = logging.getLogger(__name__)

_IDLE_UTILIZATION_THRESHOLD = 10.0
_MEMORY_PRESSURE_THRESHOLD = 80.0
_IDLE_POWER_PER_GPU_W = 45.0
_LOAD_POWER_PER_GPU_W = 250.0


class AnalysisService:
    def __init__(self, repository: ClusterRepository) -> None:
        self.repository = repository

    def analyze_cluster(self, cluster_id: UUID, max_traces: int = 100) -> WorkloadAnalysisResult:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")

        entries = self.repository.list_traces(cluster_id, limit=max_traces)
        if not entries:
            raise KeyError("No trace data available for analysis")

        states = [s for _, s in entries]
        trace_count = len(states)
        timeframe = (
            (states[0].collected_at - states[-1].collected_at).total_seconds() / 3600.0
            if trace_count > 1
            else 0.0
        )

        gpu_map: dict[str, list[dict]] = {}
        node_map: dict[str, list[dict]] = {}

        for state in states:
            for node_state in state.nodes:
                node_key = node_state.name
                if node_key not in node_map:
                    node_map[node_key] = []
                node_map[node_key].append(node_state)

                for gpu in node_state.gpu_devices:
                    gpu_key = gpu.uuid or f"{node_key}/gpu-{gpu.index}"
                    if gpu_key not in gpu_map:
                        gpu_map[gpu_key] = []
                    gpu_map[gpu_key].append({
                        "node": node_key,
                        "index": gpu.index,
                        "uuid": gpu.uuid,
                        "model": gpu.model,
                        "memory_total": gpu.memory_total_bytes,
                        "memory_used": gpu.memory_used_bytes,
                        "utilization": 0.0,
                        "temperature": 0.0,
                        "status": gpu.status,
                    })

        gpu_trends: list[GPUUtilizationTrend] = []
        for gpu_key, samples in gpu_map.items():
            utils = [s["utilization"] for s in samples]
            mems = [s["memory_used"] for s in samples]
            temps = [s["temperature"] for s in samples]
            first = samples[0]
            idle_count = sum(1 for u in utils if u < _IDLE_UTILIZATION_THRESHOLD)
            pressure_count = sum(
                1 for m, t in zip(mems, [s["memory_total"] for s in samples])
                if t > 0 and (m / t) * 100 > _MEMORY_PRESSURE_THRESHOLD
            )
            gpu_trends.append(
                GPUUtilizationTrend(
                    gpu_uuid=gpu_key,
                    node=first["node"],
                    gpu_index=first["index"],
                    model=first["model"],
                    memory_total_bytes=first["memory_total"],
                    avg_utilization_percent=round(sum(utils) / len(utils), 1) if utils else 0.0,
                    peak_utilization_percent=round(max(utils), 1) if utils else 0.0,
                    min_utilization_percent=round(min(utils), 1) if utils else 0.0,
                    avg_memory_used_bytes=round(sum(mems) / len(mems)) if mems else 0,
                    peak_memory_used_bytes=max(mems) if mems else 0,
                    avg_temperature_celsius=round(sum(temps) / len(temps), 1) if temps else 0.0,
                    peak_temperature_celsius=round(max(temps), 1) if temps else 0.0,
                    idle_percent=round((idle_count / len(samples)) * 100, 1) if samples else 0.0,
                    memory_pressure_percent=round((pressure_count / len(samples)) * 100, 1) if samples else 0.0,
                    sample_count=len(samples),
                )
            )

        node_efficiencies: list[NodeEfficiency] = []
        for node_name, node_samples in node_map.items():
            gpu_count = len(gpu_map)
            gpu_utils = []
            mem_utils = []
            pod_counts = []
            for ns in node_samples:
                pod_counts.append(ns.pod_count)
                for gpu in ns.gpu_devices:
                    if gpu.memory_total_bytes > 0:
                        mem_utils.append((gpu.memory_used_bytes / gpu.memory_total_bytes) * 100)
            for trend in gpu_trends:
                if trend.node == node_name:
                    gpu_utils.append(trend.avg_utilization_percent)

            avg_gpu_util = round(sum(gpu_utils) / len(gpu_utils), 1) if gpu_utils else 0.0
            avg_mem_util = round(sum(mem_utils) / len(mem_utils), 1) if mem_utils else 0.0
            idle_pct = round(
                sum(t.idle_percent for t in gpu_trends if t.node == node_name) / max(len([t for t in gpu_trends if t.node == node_name]), 1),
                1,
            )
            avg_pods = round(sum(pod_counts) / len(pod_counts), 1) if pod_counts else 0.0

            score = self._efficiency_score(avg_gpu_util, avg_mem_util, idle_pct)

            recs: list[str] = []
            if idle_pct > 50:
                recs.append(f"GPU idle {idle_pct}% of the time — consider workload consolidation or right-sizing")
            if avg_gpu_util < 30:
                recs.append(f"Low average GPU utilization ({avg_gpu_util}%) — evaluate bin-packing opportunities")
            if avg_mem_util > 85:
                recs.append(f"High memory pressure ({avg_mem_util}%) — risk of OOM, consider GPU with more memory")
            if score < 40:
                recs.append("Overall efficiency low — review GPU allocation and scheduling policies")

            node_efficiencies.append(
                NodeEfficiency(
                    node_name=node_name,
                    status=node_samples[-1].status if node_samples else "Unknown",
                    gpu_count=len([t for t in gpu_trends if t.node == node_name]),
                    avg_gpu_utilization_percent=avg_gpu_util,
                    gpu_idle_percent=idle_pct,
                    avg_memory_utilization_percent=avg_mem_util,
                    pod_count_avg=avg_pods,
                    pod_capacity=node_samples[0].pod_capacity if node_samples else 0,
                    efficiency_score=score,
                    recommendations=recs,
                )
            )

        total_gpu_hours = round(sum(t.sample_count * (timeframe / max(trace_count, 1)) for t in gpu_trends), 2)
        total_idle_hours = round(sum(t.idle_percent / 100.0 * (t.sample_count * timeframe / max(trace_count, 1)) for t in gpu_trends), 2)

        all_scores = [n.efficiency_score for n in node_efficiencies]
        overall_score = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0.0

        idle_power = total_idle_hours * _IDLE_POWER_PER_GPU_W / 1000.0
        load_power = (total_gpu_hours - total_idle_hours) * _LOAD_POWER_PER_GPU_W / 1000.0
        if_full_load = total_gpu_hours * _LOAD_POWER_PER_GPU_W / 1000.0
        waste = round(if_full_load - load_power - idle_power, 2)

        summary_parts = []
        summary_parts.append(f"Analyzed {trace_count} trace(s) over {timeframe:.1f}h")
        summary_parts.append(f"{len(gpu_trends)} GPU(s), overall efficiency {overall_score}/100")
        if total_idle_hours > 0:
            summary_parts.append(f"{total_idle_hours:.1f} GPU-hours idle (est. {waste} kWh waste)")

        result = WorkloadAnalysisResult(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            timeframe_hours=round(timeframe, 2),
            trace_count=trace_count,
            node_count=len(node_efficiencies),
            gpu_count=len(gpu_trends),
            total_gpu_hours=total_gpu_hours,
            gpu_trends=gpu_trends,
            node_efficiencies=node_efficiencies,
            overall_efficiency_score=overall_score,
            total_idle_gpu_hours=total_idle_hours,
            estimated_power_waste_kwh=waste,
            summary="; ".join(summary_parts),
        )

        self.repository.save_analysis(result)
        return result

    def get_latest_analysis(self, cluster_id: UUID) -> WorkloadAnalysisResult | None:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        return self.repository.latest_analysis(cluster_id)

    def list_analyses(self, cluster_id: UUID, limit: int = 10) -> list[WorkloadAnalysisResult]:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        return self.repository.list_analyses(cluster_id, limit=limit)

    @staticmethod
    def _efficiency_score(avg_util: float, avg_mem_util: float, idle_pct: float) -> float:
        util_score = min(avg_util / 80.0, 1.0) * 50
        mem_score = min(avg_mem_util / 80.0, 1.0) * 25
        idle_penalty = max(0, (idle_pct - 20) / 80.0) * 25
        score = util_score + mem_score - idle_penalty
        return round(max(0, min(100, score)), 1)
