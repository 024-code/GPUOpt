from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from .repository import ClusterRepository
from .schemas import (
    CostReport,
    CostSummary,
    GpuCostBreakdown,
    NodeCostBreakdown,
    SavingsProjection,
)

logger = logging.getLogger(__name__)

_GPU_HOURLY_RATE = 0.85
_HOURS_PER_MONTH = 730
_HOURS_PER_DAY = 24


class CostAnalysisService:
    def __init__(self, repository: ClusterRepository) -> None:
        self.repository = repository

    def generate_cost_report(self, cluster_id: UUID) -> CostReport:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        state = self.repository.latest_state(cluster_id)
        if state is None:
            raise KeyError("No cluster state available for cost analysis")

        nodes: list[NodeCostBreakdown] = []
        total_gpus = 0
        active_gpus = 0
        idle_gpus = 0
        total_hourly = 0.0
        total_waste_hourly = 0.0
        total_util_sum = 0.0

        for node in state.nodes:
            gpus_list: list[GpuCostBreakdown] = []
            node_gpu_count = 0
            node_hourly = 0.0
            node_waste = 0.0
            gpu_models: list[str] = []

            for i, gpu in enumerate(node.gpu_devices):
                model = gpu.model or f"GPU-{i}"
                gpu_models.append(model)

                util = 0.0
                if gpu.memory_total_bytes > 0:
                    util = gpu.memory_used_bytes / gpu.memory_total_bytes * 100

                hourly = _GPU_HOURLY_RATE
                waste = hourly * (1 - util / 100) if util < 100 else 0

                if util > 10:
                    cat = "active"
                    active_gpus += 1
                else:
                    cat = "idle"
                    idle_gpus += 1

                gpus_list.append(GpuCostBreakdown(
                    gpu_index=i,
                    gpu_model=model,
                    memory_utilization_percent=round(util, 1),
                    estimated_hourly_cost=round(hourly, 2),
                    estimated_monthly_cost=round(hourly * _HOURS_PER_MONTH, 2),
                    estimated_waste_hourly=round(waste, 2),
                    utilization_category=cat,
                ))

                total_gpus += 1
                node_gpu_count += 1
                node_hourly += hourly
                node_waste += waste
                total_util_sum += util

            nodes.append(NodeCostBreakdown(
                node_name=node.name,
                gpu_count=node_gpu_count,
                gpu_models=list(set(gpu_models)),
                total_hourly_cost=round(node_hourly, 2),
                total_monthly_cost=round(node_hourly * _HOURS_PER_MONTH, 2),
                waste_hourly_cost=round(node_waste, 2),
                gpus=gpus_list,
            ))

            total_hourly += node_hourly
            total_waste_hourly += node_waste

        efficiency = round(total_util_sum / max(total_gpus, 1), 1)
        daily_cost = total_hourly * _HOURS_PER_DAY
        monthly_cost = total_hourly * _HOURS_PER_MONTH
        waste_daily = total_waste_hourly * _HOURS_PER_DAY
        waste_monthly = total_waste_hourly * _HOURS_PER_MONTH

        summary = (
            f"{total_gpus} GPU(s): {active_gpus} active, {idle_gpus} idle. "
            f"${monthly_cost:.0f}/mo (${total_waste_hourly:.1f}/hr waste, {efficiency:.0f}% eff.)"
        )

        return CostReport(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            gpu_hourly_rate=_GPU_HOURLY_RATE,
            total_gpus=total_gpus,
            active_gpus=active_gpus,
            idle_gpus=idle_gpus,
            total_hourly_cost=round(total_hourly, 2),
            total_daily_cost=round(daily_cost, 2),
            total_monthly_cost=round(monthly_cost, 2),
            waste_hourly_cost=round(total_waste_hourly, 2),
            waste_daily_cost=round(waste_daily, 2),
            waste_monthly_cost=round(waste_monthly, 2),
            efficiency_percent=efficiency,
            nodes=nodes,
            summary=summary,
        )

    def project_savings(self, cluster_id: UUID) -> SavingsProjection:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        report = self.generate_cost_report(cluster_id)

        rec_set = self.repository.latest_recommendations(cluster_id)
        rec_count = len(rec_set.recommendations) if rec_set else 0

        total_savings_pct = 0.0
        top_recs: list[str] = []

        if rec_set:
            for r in rec_set.recommendations:
                if r.score and r.score > 0:
                    savings = r.score / 100.0
                    total_savings_pct = max(total_savings_pct, savings)
                    if len(top_recs) < 3:
                        top_recs.append(f"{r.title} ({savings:.0f}%)")

        savings_pct = min(total_savings_pct * 100, 40)
        projected_monthly = report.total_monthly_cost * (1 - savings_pct / 100)
        monthly_savings = report.total_monthly_cost - projected_monthly

        summary = (
            f"Current ${report.total_monthly_cost:.0f}/mo → projected "
            f"${projected_monthly:.0f}/mo (save ${monthly_savings:.0f}/mo, "
            f"{savings_pct:.0f}% reduction)"
        )

        return SavingsProjection(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            current_monthly_cost=report.total_monthly_cost,
            projected_monthly_cost=round(projected_monthly, 2),
            monthly_savings=round(monthly_savings, 2),
            annual_savings=round(monthly_savings * 12, 2),
            savings_percent=round(savings_pct, 1),
            recommendation_count=rec_count,
            top_savings_recs=top_recs,
            summary=summary,
        )

    def get_cost_summary(self, cluster_id: UUID) -> CostSummary:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        report = self.generate_cost_report(cluster_id)
        projection = self.project_savings(cluster_id)

        waste_ratio = report.waste_monthly_cost / max(report.total_monthly_cost, 1)

        if waste_ratio < 0.1:
            health = "good"
        elif waste_ratio < 0.25:
            health = "fair"
        elif waste_ratio < 0.4:
            health = "poor"
        else:
            health = "critical"

        payback = 0.0
        if projection.monthly_savings > 0:
            payback = report.total_monthly_cost / projection.monthly_savings * 30

        summary = (
            f"Cost health: {health}. ${report.total_monthly_cost:.0f}/mo "
            f"(${report.waste_monthly_cost:.0f}/mo waste). "
            f"Potential savings: ${projection.monthly_savings:.0f}/mo"
        )

        return CostSummary(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            total_gpus=report.total_gpus,
            utilization_rate=report.efficiency_percent,
            monthly_cost=report.total_monthly_cost,
            monthly_waste=report.waste_monthly_cost,
            potential_monthly_savings=projection.monthly_savings,
            payback_period_days=round(payback, 1),
            cost_health=health,
            summary=summary,
        )
