from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from gpuopt.schemas import (
    BudgetAlert,
    CloudProvider,
    CostAllocationTag,
    CostForecast,
    CostForecastPoint,
    GpuPricingRow,
    GpuPricingTier,
    MultiClusterCostSummary,
    ProviderCostComparison,
    RecommendationSeverity,
    RecommendationType,
    ReservedInstanceRecommendation,
    ResourceRecommendation,
    SavingsProjection,
    SpotSavingsAnalysis,
    WhatIfCostScenario,
)
from gpuopt.repository import ClusterRepository

logger = logging.getLogger(__name__)

GPU_PRICING: list[GpuPricingRow] = [
    # AWS us-east-1
    GpuPricingRow(gpu_model="h100", provider=CloudProvider.AWS, region="us-east-1", tier=GpuPricingTier.ONDEMAND, hourly_cost=4.82, monthly_cost=3518.6, spot_savings_percent=65.0, reserved_1yr_savings_percent=30.0, reserved_3yr_savings_percent=55.0, gpu_count_per_instance=8, instance_type="p5.48xlarge", vcpu_count=192, system_memory_gb=2048),
    GpuPricingRow(gpu_model="h100", provider=CloudProvider.AWS, region="us-east-1", tier=GpuPricingTier.SPOT, hourly_cost=1.69, monthly_cost=1233.7, spot_savings_percent=65.0, gpu_count_per_instance=8, instance_type="p5.48xlarge"),
    GpuPricingRow(gpu_model="a100", provider=CloudProvider.AWS, region="us-east-1", tier=GpuPricingTier.ONDEMAND, hourly_cost=3.91, monthly_cost=2854.3, spot_savings_percent=60.0, reserved_1yr_savings_percent=28.0, reserved_3yr_savings_percent=52.0, gpu_count_per_instance=8, instance_type="p4d.24xlarge", vcpu_count=96, system_memory_gb=1152),
    GpuPricingRow(gpu_model="a100", provider=CloudProvider.AWS, region="us-east-1", tier=GpuPricingTier.SPOT, hourly_cost=1.56, monthly_cost=1138.8, spot_savings_percent=60.0, gpu_count_per_instance=8, instance_type="p4d.24xlarge"),
    GpuPricingRow(gpu_model="v100", provider=CloudProvider.AWS, region="us-east-1", tier=GpuPricingTier.ONDEMAND, hourly_cost=3.06, monthly_cost=2233.8, spot_savings_percent=60.0, reserved_1yr_savings_percent=28.0, reserved_3yr_savings_percent=50.0, gpu_count_per_instance=8, instance_type="p3.16xlarge", vcpu_count=64, system_memory_gb=488),
    GpuPricingRow(gpu_model="v100", provider=CloudProvider.AWS, region="us-east-1", tier=GpuPricingTier.SPOT, hourly_cost=1.22, monthly_cost=890.6, spot_savings_percent=60.0, gpu_count_per_instance=8, instance_type="p3.16xlarge"),
    GpuPricingRow(gpu_model="t4", provider=CloudProvider.AWS, region="us-east-1", tier=GpuPricingTier.ONDEMAND, hourly_cost=0.94, monthly_cost=686.2, spot_savings_percent=65.0, reserved_1yr_savings_percent=30.0, reserved_3yr_savings_percent=55.0, gpu_count_per_instance=1, instance_type="g4dn.xlarge", vcpu_count=4, system_memory_gb=16),

    # GCP us-central1
    GpuPricingRow(gpu_model="h100", provider=CloudProvider.GCP, region="us-central1", tier=GpuPricingTier.ONDEMAND, hourly_cost=4.50, monthly_cost=3285.0, spot_savings_percent=60.0, reserved_1yr_savings_percent=25.0, reserved_3yr_savings_percent=50.0, gpu_count_per_instance=8, instance_type="a3-highgpu-8g", vcpu_count=208, system_memory_gb=1872),
    GpuPricingRow(gpu_model="h100", provider=CloudProvider.GCP, region="us-central1", tier=GpuPricingTier.SPOT, hourly_cost=1.80, monthly_cost=1314.0, spot_savings_percent=60.0, gpu_count_per_instance=8, instance_type="a3-highgpu-8g"),
    GpuPricingRow(gpu_model="a100", provider=CloudProvider.GCP, region="us-central1", tier=GpuPricingTier.ONDEMAND, hourly_cost=3.67, monthly_cost=2679.1, spot_savings_percent=60.0, reserved_1yr_savings_percent=25.0, reserved_3yr_savings_percent=50.0, gpu_count_per_instance=8, instance_type="a2-highgpu-8g", vcpu_count=96, system_memory_gb=1360),
    GpuPricingRow(gpu_model="a100", provider=CloudProvider.GCP, region="us-central1", tier=GpuPricingTier.SPOT, hourly_cost=1.47, monthly_cost=1073.1, spot_savings_percent=60.0, gpu_count_per_instance=8, instance_type="a2-highgpu-8g"),
    GpuPricingRow(gpu_model="v100", provider=CloudProvider.GCP, region="us-central1", tier=GpuPricingTier.ONDEMAND, hourly_cost=2.48, monthly_cost=1810.4, spot_savings_percent=55.0, reserved_1yr_savings_percent=22.0, reserved_3yr_savings_percent=45.0, gpu_count_per_instance=8, instance_type="n1-standard-96-8-v100", vcpu_count=96, system_memory_gb=360),
    GpuPricingRow(gpu_model="v100", provider=CloudProvider.GCP, region="us-central1", tier=GpuPricingTier.SPOT, hourly_cost=1.12, monthly_cost=817.6, spot_savings_percent=55.0, gpu_count_per_instance=8, instance_type="n1-standard-96-8-v100"),
    GpuPricingRow(gpu_model="t4", provider=CloudProvider.GCP, region="us-central1", tier=GpuPricingTier.ONDEMAND, hourly_cost=0.88, monthly_cost=642.4, spot_savings_percent=60.0, reserved_1yr_savings_percent=25.0, reserved_3yr_savings_percent=50.0, gpu_count_per_instance=1, instance_type="n1-standard-4-1-t4", vcpu_count=4, system_memory_gb=26),

    # Azure eastus
    GpuPricingRow(gpu_model="h100", provider=CloudProvider.AZURE, region="eastus", tier=GpuPricingTier.ONDEMAND, hourly_cost=4.60, monthly_cost=3358.0, spot_savings_percent=60.0, reserved_1yr_savings_percent=27.0, reserved_3yr_savings_percent=52.0, gpu_count_per_instance=8, instance_type="ND H100 v5", vcpu_count=192, system_memory_gb=2048),
    GpuPricingRow(gpu_model="h100", provider=CloudProvider.AZURE, region="eastus", tier=GpuPricingTier.SPOT, hourly_cost=1.84, monthly_cost=1343.2, spot_savings_percent=60.0, gpu_count_per_instance=8, instance_type="ND H100 v5"),
    GpuPricingRow(gpu_model="a100", provider=CloudProvider.AZURE, region="eastus", tier=GpuPricingTier.ONDEMAND, hourly_cost=3.73, monthly_cost=2722.9, spot_savings_percent=55.0, reserved_1yr_savings_percent=25.0, reserved_3yr_savings_percent=48.0, gpu_count_per_instance=8, instance_type="ND A100 v4", vcpu_count=96, system_memory_gb=900),
    GpuPricingRow(gpu_model="a100", provider=CloudProvider.AZURE, region="eastus", tier=GpuPricingTier.SPOT, hourly_cost=1.68, monthly_cost=1226.4, spot_savings_percent=55.0, gpu_count_per_instance=8, instance_type="ND A100 v4"),

    # Equinix on-prem equivalent
    GpuPricingRow(gpu_model="h100", provider=CloudProvider.EQUINIX, region="us-east", tier=GpuPricingTier.ONDEMAND, hourly_cost=2.50, monthly_cost=1825.0, gpu_count_per_instance=8, instance_type="h100-sxm-8gpu", vcpu_count=128, system_memory_gb=2048),
    GpuPricingRow(gpu_model="a100", provider=CloudProvider.EQUINIX, region="us-east", tier=GpuPricingTier.ONDEMAND, hourly_cost=2.10, monthly_cost=1533.0, gpu_count_per_instance=8, instance_type="a100-sxm-8gpu", vcpu_count=96, system_memory_gb=1024),
]


class FinOpsService:
    """FinOps service for GPU cost optimization across cloud providers."""

    def __init__(self, repository: ClusterRepository) -> None:
        self._repository = repository

    @staticmethod
    def get_pricing(
        gpu_model: str = "",
        provider: CloudProvider | None = None,
        region: str = "",
        tier: GpuPricingTier | None = None,
    ) -> list[GpuPricingRow]:
        results = list(GPU_PRICING)
        if gpu_model:
            results = [r for r in results if r.gpu_model == gpu_model.lower()]
        if provider:
            results = [r for r in results if r.provider == provider]
        if region:
            results = [r for r in results if region in r.region]
        if tier:
            results = [r for r in results if r.tier == tier]
        return results

    @staticmethod
    def compare_providers(
        gpu_model: str = "h100",
        gpu_count: int = 8,
    ) -> ProviderCostComparison:
        pricing = [r for r in GPU_PRICING if r.gpu_model == gpu_model.lower()]
        providers: list[GpuPricingRow] = []

        for row in pricing:
            adjusted = row.model_copy(deep=True)
            multiplier = max(math.ceil(gpu_count / row.gpu_count_per_instance), 1)
            adjusted.hourly_cost = round(row.hourly_cost * multiplier, 4)
            adjusted.monthly_cost = round(row.monthly_cost * multiplier, 2)
            providers.append(adjusted)

        ondemand = [p for p in providers if p.tier == GpuPricingTier.ONDEMAND]
        all_tiers = sorted(providers, key=lambda p: p.monthly_cost)
        cheapest_ondemand = min(ondemand, key=lambda p: p.monthly_cost) if ondemand else None
        cheapest_overall = all_tiers[0] if all_tiers else None

        max_savings = 0.0
        if cheapest_ondemand and cheapest_overall:
            max_savings = round((1 - cheapest_overall.monthly_cost / cheapest_ondemand.monthly_cost) * 100, 1) if cheapest_ondemand.monthly_cost > 0 else 0.0

        rec = ""
        if cheapest_overall and cheapest_overall.tier != GpuPricingTier.ONDEMAND:
            rec = f"Save up to {max_savings:.0f}% by using {cheapest_overall.provider.value} {cheapest_overall.tier.value} in {cheapest_overall.region}"
        elif cheapest_ondemand:
            rec = f"Cheapest on-demand: {cheapest_ondemand.provider.value} {cheapest_ondemand.region} at ${cheapest_ondemand.monthly_cost:.0f}/mo"

        return ProviderCostComparison(
            gpu_model=gpu_model,
            gpu_count=gpu_count,
            providers=providers,
            cheapest_ondemand=cheapest_ondemand,
            cheapest_overall=cheapest_overall,
            max_potential_savings_percent=max_savings,
            recommendation=rec,
        )

    def analyze_spot_savings(self, cluster_id: UUID) -> SpotSavingsAnalysis:
        cluster = self._repository.get_cluster(cluster_id)
        state = self._repository.latest_state(cluster_id)
        cluster_name = cluster.name if cluster else "unknown"

        total_gpus = 0
        gpu_model = "a100"
        if state:
            for node in state.nodes:
                for gpu in node.gpu_devices:
                    total_gpus += 1
                    if gpu.model:
                        gpu_model = gpu.model.lower().replace("nvidia ", "")

        pricing = [r for r in GPU_PRICING if r.gpu_model in gpu_model and r.provider == CloudProvider.AWS]
        ondemand_row = next((r for r in pricing if r.tier == GpuPricingTier.ONDEMAND), None)
        spot_row = next((r for r in pricing if r.tier == GpuPricingTier.SPOT), None)

        per_gpu_od = (ondemand_row.hourly_cost / max(ondemand_row.gpu_count_per_instance, 1)) if ondemand_row else 0.85
        per_gpu_spot = (spot_row.hourly_cost / max(spot_row.gpu_count_per_instance, 1)) if spot_row else per_gpu_od * 0.4

        ondemand_monthly = per_gpu_od * total_gpus * 730
        spot_monthly = per_gpu_spot * total_gpus * 730
        monthly_savings = ondemand_monthly - spot_monthly
        annual_savings = monthly_savings * 12
        savings_pct = round((1 - spot_monthly / ondemand_monthly) * 100, 1) if ondemand_monthly > 0 else 0.0

        spot_viable = total_gpus
        risk = "low"

        recs: list[str] = []
        if savings_pct > 50:
            recs.append(f"Spot instances can save {savings_pct:.0f}% (${annual_savings:.0f}/yr) — use spot for non-critical workloads")
        elif savings_pct > 30:
            recs.append(f"Spot pricing offers {savings_pct:.0f}% savings — consider spot for training jobs with checkpointing")
        else:
            recs.append("Spot savings are modest; evaluate interruption tolerance before switching")

        if total_gpus > 16:
            recs.append("With more than 16 GPUs, use a mix of spot (for fault-tolerant jobs) and on-demand (for critical services)")
        recs.append("Enable automatic checkpointing to handle spot instance interruptions")

        return SpotSavingsAnalysis(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            total_gpus=total_gpus,
            ondemand_monthly_cost=round(ondemand_monthly, 2),
            spot_monthly_cost=round(spot_monthly, 2),
            monthly_savings=round(monthly_savings, 2),
            annual_savings=round(annual_savings, 2),
            savings_percent=savings_pct,
            spot_viable_gpus=spot_viable,
            interruption_risk=risk,
            recommendations=recs,
        )

    def recommend_reserved_instances(self, cluster_id: UUID) -> ReservedInstanceRecommendation:
        cluster = self._repository.get_cluster(cluster_id)
        state = self._repository.latest_state(cluster_id)
        cluster_name = cluster.name if cluster else "unknown"

        total_gpus = 0
        gpu_model = "a100"
        if state:
            for node in state.nodes:
                for gpu in node.gpu_devices:
                    total_gpus += 1
                    if gpu.model:
                        gpu_model = gpu.model.lower().replace("nvidia ", "")

        pricing = [r for r in GPU_PRICING if r.gpu_model in gpu_model and r.provider == CloudProvider.AWS and r.tier == GpuPricingTier.ONDEMAND]
        od_row = next(iter(pricing), None)
        per_gpu_od = (od_row.hourly_cost / max(od_row.gpu_count_per_instance, 1)) if od_row else 0.85
        current_monthly = per_gpu_od * total_gpus * 730

        savings_rate_1yr = 0.28
        savings_rate_3yr = 0.52
        if od_row:
            savings_rate_1yr = od_row.reserved_1yr_savings_percent / 100.0
            savings_rate_3yr = od_row.reserved_3yr_savings_percent / 100.0

        reserved_1yr = current_monthly * (1 - savings_rate_1yr)
        reserved_3yr = current_monthly * (1 - savings_rate_3yr)
        monthly_savings_1yr = current_monthly - reserved_1yr
        monthly_savings_3yr = current_monthly - reserved_3yr
        annual_savings_1yr = monthly_savings_1yr * 12
        annual_savings_3yr = monthly_savings_3yr * 12

        upfront_premium_1yr = current_monthly * 2
        break_even_1yr = upfront_premium_1yr / max(monthly_savings_1yr, 1)

        term = "3yr" if annual_savings_3yr > annual_savings_1yr * 1.5 else "1yr"

        recs: list[str] = [
            f"Reserved instances could save ${annual_savings_1yr:.0f}/yr (1yr) or ${annual_savings_3yr:.0f}/yr (3yr)",
        ]
        if term == "3yr":
            recs.append("3-year term offers maximum savings for stable, long-running workloads")
        else:
            recs.append("1-year term balances savings with flexibility for evolving workloads")
        if total_gpus >= 16:
            recs.append("With 16+ GPUs, reserve a baseline and use on-demand/spot for spikes")
        recs.append("Reserved instances are recommended for production inference endpoints and training clusters")

        return ReservedInstanceRecommendation(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            current_monthly_cost=round(current_monthly, 2),
            reserved_1yr_monthly_cost=round(reserved_1yr, 2),
            reserved_3yr_monthly_cost=round(reserved_3yr, 2),
            monthly_savings_1yr=round(monthly_savings_1yr, 2),
            monthly_savings_3yr=round(monthly_savings_3yr, 2),
            annual_savings_1yr=round(annual_savings_1yr, 2),
            annual_savings_3yr=round(annual_savings_3yr, 2),
            recommended_term=term,
            break_even_months=round(break_even_1yr, 1),
            recommendations=recs,
        )

    def get_budget_alert(self, cluster_id: UUID, monthly_budget: float = 0.0) -> BudgetAlert:
        cluster = self._repository.get_cluster(cluster_id)
        cluster_name = cluster.name if cluster else "unknown"
        report = None
        try:
            from gpuopt.cost_analysis import CostAnalysisService
            cost_svc = CostAnalysisService(self._repository)
            report = cost_svc.generate_cost_report(cluster_id)
        except Exception as exc:
            logger.warning("Could not generate cost report for budget alert: %s", exc)

        current_spend = report.total_monthly_cost if report else 0.0
        if monthly_budget <= 0:
            monthly_budget = current_spend * 1.2

        util_pct = round(current_spend / max(monthly_budget, 1) * 100, 1)
        projected = current_spend * 1.05

        if util_pct >= 100:
            status = "over_budget"
        elif util_pct >= 85:
            status = "at_risk"
        elif util_pct >= 70:
            status = "watch"
        else:
            status = "on_track"

        alerts: list[str] = []
        if status == "over_budget":
            alerts.append(f"Monthly spend ${current_spend:.0f} exceeds budget ${monthly_budget:.0f} by ${current_spend - monthly_budget:.0f}")
        elif status == "at_risk":
            alerts.append(f"Spend at {util_pct:.0f}% of budget — projected to exceed by month end")
        if projected > monthly_budget:
            alerts.append(f"Projected month-end spend ${projected:.0f} exceeds budget")

        return BudgetAlert(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            monthly_budget=monthly_budget,
            current_monthly_spend=round(current_spend, 2),
            budget_utilization_percent=util_pct,
            projected_month_end_spend=round(projected, 2),
            status=status,
            alerts=alerts,
        )

    def aggregate_costs(self) -> MultiClusterCostSummary:
        clusters = self._repository.list_clusters()
        totals = MultiClusterCostSummary(cluster_count=len(clusters))

        from gpuopt.cost_analysis import CostAnalysisService
        cost_svc = CostAnalysisService(self._repository)

        for cluster in clusters:
            try:
                report = cost_svc.generate_cost_report(cluster.id)
                summary = cost_svc.get_cost_summary(cluster.id)
                totals.total_gpus += report.total_gpus
                totals.total_monthly_cost += report.total_monthly_cost
                totals.total_monthly_waste += report.waste_monthly_cost
                totals.total_potential_monthly_savings += summary.potential_monthly_savings
                totals.total_annual_savings += summary.potential_monthly_savings * 12
                totals.clusters.append({
                    "cluster_id": str(cluster.id),
                    "name": cluster.name,
                    "environment": cluster.environment,
                    "total_gpus": report.total_gpus,
                    "monthly_cost": round(report.total_monthly_cost, 2),
                    "monthly_waste": round(report.waste_monthly_cost, 2),
                    "efficiency": round(report.efficiency_percent, 1),
                })
            except Exception as exc:
                logger.warning("Could not aggregate costs for cluster %s: %s", cluster.id, exc)

        if totals.total_gpus > 0:
            totals.average_utilization = round(
                sum(c["efficiency"] for c in totals.clusters) / max(len(totals.clusters), 1), 1
            )

        if totals.total_annual_savings > 5000:
            totals.top_recommendations.append(f"Annual savings potential of ${totals.total_annual_savings:.0f} — prioritize right-sizing")
        if totals.total_monthly_waste / max(totals.total_monthly_cost, 1) > 0.2:
            totals.top_recommendations.append("Waste exceeds 20% of total spend — review idle GPU allocation")
        if totals.cluster_count > 1:
            totals.top_recommendations.append("Multi-cluster aggregation enabled — consider centralized GPU pool")

        return totals

    def forecast_cost(
        self,
        cluster_id: UUID,
        months: int = 12,
        growth_rate: float = 0.05,
    ) -> CostForecast:
        cluster = self._repository.get_cluster(cluster_id)
        cluster_name = cluster.name if cluster else "unknown"

        from gpuopt.cost_analysis import CostAnalysisService
        cost_svc = CostAnalysisService(self._repository)
        current_monthly = 0.0
        try:
            report = cost_svc.generate_cost_report(cluster_id)
            current_monthly = report.total_monthly_cost
        except Exception:
            pass

        forecast: list[CostForecastPoint] = []
        for i in range(months):
            month_label = f"Month {i + 1}"
            projected = current_monthly * (1 + growth_rate) ** (i + 1)
            optimistic = projected * 0.9
            pessimistic = projected * 1.15
            confidence_upper = pessimistic * 1.1
            confidence_lower = optimistic * 0.9
            forecast.append(CostForecastPoint(
                month=month_label,
                projected_cost=round(projected, 2),
                optimistic_cost=round(optimistic, 2),
                pessimistic_cost=round(pessimistic, 2),
                confidence_upper=round(confidence_upper, 2),
                confidence_lower=round(confidence_lower, 2),
            ))

        projected_annual = sum(f.projected_cost for f in forecast)

        return CostForecast(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            current_monthly_cost=round(current_monthly, 2),
            forecast=forecast,
            projected_annual_cost=round(projected_annual, 2),
            growth_rate=growth_rate,
            summary=f"Current ${current_monthly:.0f}/mo → ${projected_annual:.0f}/yr projected "
                    f"at {growth_rate*100:.0f}% monthly growth",
        )

    @staticmethod
    def what_if_cost(
        scenario_name: str,
        description: str = "",
        current_monthly_cost: float = 0.0,
        gpu_count_change: int = 0,
        utilization_change: float = 0.0,
        provider_change: CloudProvider | None = None,
        tier_change: GpuPricingTier | None = None,
    ) -> WhatIfCostScenario:
        per_gpu_cost = current_monthly_cost / max(gpu_count_change or 8, 1)
        scenario_cost = current_monthly_cost

        if gpu_count_change != 0:
            scenario_cost += gpu_count_change * per_gpu_cost

        if utilization_change > 0:
            savings_from_util = scenario_cost * (utilization_change / 100.0) * 0.3
            scenario_cost -= savings_from_util

        provider_discount = 0.0
        if provider_change == CloudProvider.EQUINIX:
            provider_discount = 0.35
        elif tier_change == GpuPricingTier.SPOT:
            provider_discount = 0.55
        elif tier_change in (GpuPricingTier.RESERVED_1YR, GpuPricingTier.RESERVED_3YR):
            provider_discount = 0.30 if tier_change == GpuPricingTier.RESERVED_1YR else 0.50

        if provider_discount > 0:
            scenario_cost *= (1 - provider_discount)

        diff = scenario_cost - current_monthly_cost
        annual_diff = diff * 12

        recs: list[str] = []
        if diff < 0:
            recs.append(f"Scenario saves ${abs(diff):.0f}/mo (${abs(annual_diff):.0f}/yr)")
        else:
            recs.append(f"Scenario costs ${diff:.0f}/mo more (${annual_diff:.0f}/yr)")
        if provider_change:
            recs.append(f"Provider change to {provider_change.value} estimated at {provider_discount*100:.0f}% discount")

        return WhatIfCostScenario(
            scenario_name=scenario_name,
            description=description,
            gpu_count_change=gpu_count_change,
            utilization_change=utilization_change,
            provider_change=provider_change,
            tier_change=tier_change,
            current_monthly_cost=round(current_monthly_cost, 2),
            scenario_monthly_cost=round(scenario_cost, 2),
            monthly_difference=round(diff, 2),
            annual_difference=round(annual_diff, 2),
            recommendations=recs,
        )

    def get_cost_allocation(self, cluster_id: UUID) -> list[CostAllocationTag]:
        cluster = self._repository.get_cluster(cluster_id)
        state = self._repository.latest_state(cluster_id)
        tags: dict[str, dict[str, Any]] = {}

        if state:
            for node in state.nodes:
                env = node.labels.get("environment", "default") if hasattr(node, "labels") and node.labels else "default"
                if env not in tags:
                    tags[env] = {"gpu_count": 0, "monthly_cost": 0.0}
                tags[env]["gpu_count"] += len(node.gpu_devices)

        total_cost = 0.0
        from gpuopt.cost_analysis import CostAnalysisService
        try:
            report = CostAnalysisService(self._repository).generate_cost_report(cluster_id)
            total_cost = report.total_monthly_cost
        except Exception:
            pass

        results: list[CostAllocationTag] = []
        for env, data in tags.items():
            pct = round(data["gpu_count"] / max(sum(d["gpu_count"] for d in tags.values()), 1) * 100, 1)
            results.append(CostAllocationTag(
                key="environment",
                value=env,
                monthly_cost=round(total_cost * pct / 100, 2),
                gpu_count=data["gpu_count"],
                percentage=pct,
            ))
        return sorted(results, key=lambda t: t.monthly_cost, reverse=True)

    def generate_finops_recommendations(self, cluster_id: UUID) -> list[ResourceRecommendation]:
        recs: list[ResourceRecommendation] = []
        cluster = self._repository.get_cluster(cluster_id)
        if cluster is None:
            return recs

        spot = self.analyze_spot_savings(cluster_id)
        if spot.annual_savings > 1000:
            recs.append(ResourceRecommendation(
                type=RecommendationType.EFFICIENCY,
                severity=RecommendationSeverity.MEDIUM,
                title="Enable spot/preemptible GPU instances",
                description=f"Using spot instances could save ${spot.annual_savings:.0f}/yr "
                            f"({spot.savings_percent:.0f}% reduction) across {spot.spot_viable_gpus} GPUs.",
                reasoning="Spot instances offer significant cost savings for fault-tolerant workloads. "
                          "Training jobs with checkpointing and batch inference are ideal candidates.",
                expected_impact=f"Up to ${spot.annual_savings:.0f} annual savings.",
                confidence=0.75,
                risk_level="medium",
                affected_resources=[f"cluster/{cluster_id}"],
                actions=[
                    "Identify workloads that can tolerate interruption",
                    "Add checkpointing to training jobs",
                    "Configure spot instance fallback to on-demand",
                ],
                estimated_savings={"annual_spot_savings": spot.annual_savings},
            ))

        ri = self.recommend_reserved_instances(cluster_id)
        if ri.annual_savings_1yr > 2000:
            recs.append(ResourceRecommendation(
                type=RecommendationType.RIGHT_SIZING,
                severity=RecommendationSeverity.LOW,
                title="Purchase reserved/committed use instances",
                description=f"Reserved instances could save ${ri.annual_savings_1yr:.0f}/yr (1yr) "
                            f"or ${ri.annual_savings_3yr:.0f}/yr (3yr).",
                reasoning="For stable, predictable GPU workloads, reserved instances provide "
                          "significant discounts over on-demand pricing.",
                expected_impact=f"${ri.annual_savings_1yr:.0f} annual savings with 1-year commitment.",
                confidence=0.8,
                risk_level="low",
                affected_resources=[f"cluster/{cluster_id}"],
                actions=[
                    f"Commit to {ri.recommended_term} reserved instances",
                    "Reserve baseline capacity, use on-demand for spikes",
                    "Review utilization monthly to optimize coverage",
                ],
                estimated_savings={"annual_reserved_savings": ri.annual_savings_1yr},
            ))

        from gpuopt.cost_analysis import CostAnalysisService
        try:
            report = CostAnalysisService(self._repository).generate_cost_report(cluster_id)
            if report.waste_monthly_cost > report.total_monthly_cost * 0.15:
                recs.append(ResourceRecommendation(
                    type=RecommendationType.EFFICIENCY,
                    severity=RecommendationSeverity.HIGH,
                    title="Reduce GPU waste from idle resources",
                    description=f"${report.waste_monthly_cost:.0f}/mo (${report.waste_monthly_cost*12:.0f}/yr) "
                                f"wasted on idle or underutilized GPUs.",
                    reasoning=f"Cluster has {report.idle_gpus} idle GPUs out of {report.total_gpus} total. "
                              "Rightsizing or consolidating workloads can reduce waste.",
                    expected_impact=f"Up to ${report.waste_monthly_cost*12:.0f} annual savings from eliminating waste.",
                    confidence=0.85,
                    risk_level="low",
                    affected_resources=[f"cluster/{cluster_id}"],
                    actions=[
                        "Identify and terminate idle GPU instances",
                        "Consolidate underutilized workloads",
                        "Implement auto-scaling for variable demand",
                    ],
                    estimated_savings={"annual_waste_reduction": round(report.waste_monthly_cost * 12, 2)},
                ))
        except Exception:
            pass

        return recs
