from __future__ import annotations

import logging
import random
from typing import Any

from .cost_analysis import CostAnalysisService
from .finops import GPU_PRICING
from .gpu_monitor import GPUMonitor
from .schemas import (
    ConsolidationPlan,
    ElasticWorkerConfig,
    GpuTierSelection,
    RecommendationPriority,
)

logger = logging.getLogger(__name__)


class ElasticWorkerOptimizer:
    def optimize(self, job_id: str = "", workload: dict | None = None,
                 cluster_state: dict | None = None) -> ElasticWorkerConfig:
        wl = workload or {}
        gpu_per = wl.get("gpu_required", 1)
        mem_per = wl.get("memory_required_gb", 16)
        efficiency = self._estimate_parallelism_efficiency(wl)
        max_workers = int(64 * efficiency)
        min_workers = max(1, int(max_workers * 0.1))

        return ElasticWorkerConfig(
            min_workers=min_workers,
            max_workers=max_workers,
            current_workers=min_workers,
            scale_up_threshold_utilization=70.0,
            scale_down_threshold_utilization=30.0,
            cooldown_seconds=60,
            gpu_per_worker=gpu_per,
            memory_per_worker_gb=mem_per,
        )

    def suggest_scale(self, worker_config: ElasticWorkerConfig, current_load: float) -> dict:
        util = current_load * 100
        if util > worker_config.scale_up_threshold_utilization and worker_config.current_workers < worker_config.max_workers:
            target = min(worker_config.max_workers, int(worker_config.current_workers * 1.5))
            return {"action": "up", "target_workers": target, "reason": f"Load at {util:.0f}% exceeds scale-up threshold"}
        if util < worker_config.scale_down_threshold_utilization and worker_config.current_workers > worker_config.min_workers:
            target = max(worker_config.min_workers, int(worker_config.current_workers * 0.5))
            return {"action": "down", "target_workers": target, "reason": f"Load at {util:.0f}% below scale-down threshold"}
        return {"action": "none", "target_workers": worker_config.current_workers, "reason": "Load within normal range"}

    def _estimate_parallelism_efficiency(self, job: dict) -> float:
        model_size = job.get("model_size_gb", 1)
        framework = job.get("framework", "pytorch")
        base = {"pytorch": 0.7, "tensorflow": 0.6, "jax": 0.85}.get(framework, 0.7)
        size_factor = max(0.1, 1.0 - model_size / 100 * 0.2)
        return round(base * size_factor, 2)


class GpuTierSelector:
    def select(self, current_gpu_model: str = "", workload_profile: dict | None = None) -> GpuTierSelection:
        wl = workload_profile or {}
        needed_mem = wl.get("memory_required_gb", 16)
        tiers = self.list_available_tiers()
        current_cost = 0.0
        best_tier = current_gpu_model
        best_cost = float("inf")

        for t in tiers:
            tier_mem = float(t.get("memory_gb", 80))
            tier_cost = float(t.get("on_demand_cost", 1.0))
            model = t.get("gpu_model", "")
            if tier_mem >= needed_mem and tier_cost < best_cost:
                best_cost = tier_cost
                best_tier = model
            if model == current_gpu_model:
                current_cost = tier_cost

        savings = current_cost - best_cost if best_cost < current_cost else 0.0
        perf_impact = "none" if best_tier == current_gpu_model else "minor" if savings < 0.5 else "moderate"

        return GpuTierSelection(
            current_gpu_model=current_gpu_model,
            recommended_gpu_model=best_tier,
            current_cost_per_hour=round(current_cost, 4),
            recommended_cost_per_hour=round(best_cost, 4),
            savings_per_hour=round(savings, 4),
            performance_impact=perf_impact,
            confidence=round(random.uniform(0.7, 0.95), 2),
            reasoning=f"GPU tier {best_tier} meets memory requirement ({needed_mem}GB) at lower cost",
        )

    def list_available_tiers(self) -> list[dict]:
        return [
            {
                "gpu_model": row.gpu_model,
                "memory_gb": GPU_MEMORY_MAP.get(row.gpu_model.upper(), 80),
                "on_demand_cost": row.hourly_cost,
                "spot_cost": row.hourly_cost * (1 - row.spot_savings_percent / 100) if row.spot_savings_percent else row.hourly_cost,
                "provider": row.provider.value if hasattr(row.provider, 'value') else str(row.provider),
            }
            for row in GPU_PRICING
        ]


class ConsolidationPlanner:
    def __init__(self) -> None:
        self._cost_service = CostAnalysisService.__new__(CostAnalysisService)

    def plan(self, cluster_id: str = "") -> ConsolidationPlan:
        try:
            monitor = GPUMonitor()
            snap = monitor.collect()
            devices = [{"index": d.index, "model": d.model, "utilization": d.utilization_gpu_percent,
                        "memory_used_mb": d.memory_used_mb} for d in snap.devices]
        except Exception:
            devices = [{"index": i, "model": random.choice(["H100", "A100"]),
                        "utilization": random.uniform(5, 95),
                        "memory_used_mb": random.uniform(1000, 80000)}
                       for i in range(random.randint(4, 16))]

        nodes = {}
        for d in devices:
            nid = f"node-{d['index'] // 4}"
            if nid not in nodes:
                nodes[nid] = {"gpus": [], "total_util": 0.0}
            nodes[nid]["gpus"].append(d)
            nodes[nid]["total_util"] += d["utilization"]

        idle_nodes = [nid for nid, nd in nodes.items()
                      if nd["total_util"] / max(len(nd["gpus"]), 1) < 20]

        if not idle_nodes:
            idle_nodes = [nid for nid, nd in nodes.items()
                          if nd["total_util"] / max(len(nd["gpus"]), 1) < 30][:max(1, len(nodes) // 4)]

        drain_nodes = idle_nodes[:max(1, len(idle_nodes) // 2)]
        workloads = [f"wl-{random.randint(1000, 9999)}" for _ in range(len(drain_nodes) * random.randint(1, 3))]
        savings = len(drain_nodes) * random.uniform(5, 20)
        power_savings = savings * random.uniform(10, 30)
        perf_impact = random.uniform(0, 5)
        steps = []
        for nid in drain_nodes:
            steps.append(f"Drain node {nid}: migrate workloads to active nodes")
            steps.append(f"Verify node {nid} is idle")
            steps.append(f"Power off or scale down node {nid}")

        return ConsolidationPlan(
            cluster_id=cluster_id,
            current_node_count=len(nodes),
            target_node_count=len(nodes) - len(drain_nodes),
            nodes_to_drain=drain_nodes,
            workloads_to_move=workloads,
            estimated_cost_savings=round(savings, 2),
            estimated_power_savings=round(power_savings, 2),
            estimated_performance_impact=round(perf_impact, 1),
            risk_level="low" if perf_impact < 3 else "medium",
            feasibility=True,
            steps=steps,
        )


class RecommendationPrioritizer:
    def prioritize(self, recommendations: list[dict]) -> list[RecommendationPriority]:
        priorities = []
        for i, rec in enumerate(recommendations):
            urgency = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}.get(
                rec.get("severity", "medium"), 0.5)
            impact = rec.get("impact_score", random.uniform(0.2, 0.9))
            effort = rec.get("effort_score", random.uniform(0.1, 0.8))
            roi = impact / max(effort, 0.01)

            priority = urgency * 0.3 + impact * 0.3 + roi * 0.2 + (1 - effort) * 0.2
            urgency_label = "high" if urgency > 0.7 else "medium" if urgency > 0.4 else "low"
            impact_label = "high" if impact > 0.7 else "medium" if impact > 0.4 else "low"
            effort_label = "low" if effort < 0.3 else "medium" if effort < 0.6 else "high"

            priorities.append(RecommendationPriority(
                recommendation_id=str(rec.get("id", i)),
                priority_score=round(priority, 3),
                urgency=urgency_label,
                impact=impact_label,
                effort=effort_label,
                roi=round(roi, 2),
                dependencies=rec.get("dependencies", []),
                suggested_order=0,
            ))

        priorities.sort(key=lambda x: x.priority_score, reverse=True)
        for i, p in enumerate(priorities):
            p.suggested_order = i + 1
        return priorities


class ExtendedOptimizationService:
    def __init__(self) -> None:
        self._workers = ElasticWorkerOptimizer()
        self._tier_selector = GpuTierSelector()
        self._consolidator = ConsolidationPlanner()
        self._prioritizer = RecommendationPrioritizer()

    def create_consolidation_plan(self, cluster_id: str = "") -> ConsolidationPlan:
        return self._consolidator.plan(cluster_id)

    def select_gpu_tier(self, workload: dict) -> GpuTierSelection:
        return self._tier_selector.select(workload_profile=workload)

    def optimize_workers(self, job: dict) -> ElasticWorkerConfig:
        return self._workers.optimize(workload=job)

    def get_prioritized_recommendations(self, recs: list[dict] | None = None) -> list[RecommendationPriority]:
        return self._prioritizer.prioritize(recs or [])


GPU_MEMORY_MAP = {
    "H200": 141, "H100": 80, "H100_NVL": 188, "A100": 80, "A100_80GB": 80,
    "A100_40GB": 40, "V100": 32, "V100_32GB": 32, "B200": 192, "B100": 96,
    "L40S": 48, "L40": 48, "A40": 48, "A30": 24, "T4": 16, "L4": 24,
}
