from __future__ import annotations

import logging
import math
from typing import Any
from uuid import UUID

from gpuopt.schemas import (
    CarbonEmissionsEstimate,
    EnergyTimeSeriesPoint,
    GPU_POWER_PROFILES,
    PowerAnalysisResult,
    PowerCapSuggestion,
    PowerOptimizationRecommendation,
    RecommendationSeverity,
    RecommendationType,
    ResourceRecommendation,
)
from gpuopt.repository import ClusterRepository

logger = logging.getLogger(__name__)

_KWH_PER_WATT_HOUR = 0.001
_HOURS_PER_MONTH = 730
_HOURS_PER_DAY = 24
_ELECTRICITY_COST_PER_KWH = 0.12
_GRID_CARBON_INTENSITY_G_PER_KWH = 400.0
_CARBON_PER_MILE = 0.41  # kg CO2 per mile (avg US car)
_HOMES_ENERGY_PER_YEAR_KWH = 10600.0  # avg US home annual kWh


class PowerService:
    """Power optimization service.

    Analyzes GPU power consumption, estimates carbon emissions,
    suggests power capping, and generates power optimization
    recommendations.
    """

    def __init__(self, repository: ClusterRepository) -> None:
        self._repository = repository

    @staticmethod
    def get_power_profile(gpu_model: str) -> dict[str, Any] | None:
        key = gpu_model.lower().replace("nvidia ", "")
        for p in GPU_POWER_PROFILES:
            if p.gpu_model == key:
                return p.model_dump()
        return None

    @staticmethod
    def list_power_profiles() -> list[dict[str, Any]]:
        return [p.model_dump() for p in GPU_POWER_PROFILES]

    def analyze_power(self, cluster_id: UUID) -> PowerAnalysisResult:
        cluster = self._repository.get_cluster(cluster_id)
        state = self._repository.latest_state(cluster_id)
        cluster_name = cluster.name if cluster else "unknown"

        total_gpus = 0
        total_power_draw = 0.0
        total_power_cap = 0.0
        idle_power = 0.0
        active_power = 0.0
        total_util = 0.0

        if state:
            for node in state.nodes:
                for gpu in node.gpu_devices:
                    total_gpus += 1
                    profile = self.get_power_profile(gpu.model)
                    if profile:
                        load_power = profile["typical_load_power_watts"]
                        idle = profile["idle_power_watts"]
                        cap = profile["max_power_watts"]
                    else:
                        load_power = 250.0
                        idle = 45.0
                        cap = 300.0

                    mem_util = 0.0
                    if gpu.memory_total_bytes > 0:
                        mem_util = gpu.memory_used_bytes / gpu.memory_total_bytes * 100
                    power_draw = idle + (load_power - idle) * min(mem_util / 100, 1.0)

                    total_power_draw += power_draw
                    total_power_cap += cap
                    total_util += mem_util
                    if mem_util < 10:
                        idle_power += power_draw
                    else:
                        active_power += power_draw

        avg_util = total_util / max(total_gpus, 1)
        util_pct = round(avg_util, 1)
        power_waste_watts = max(idle_power - active_power * 0.1, 0.0) if active_power > 0 else idle_power
        waste_kwh_daily = power_waste_watts * _HOURS_PER_DAY * _KWH_PER_WATT_HOUR
        waste_cost_monthly = power_waste_watts * _HOURS_PER_MONTH * _KWH_PER_WATT_HOUR * _ELECTRICITY_COST_PER_KWH
        idle_cost_monthly = idle_power * _HOURS_PER_MONTH * _KWH_PER_WATT_HOUR * _ELECTRICITY_COST_PER_KWH
        active_cost_monthly = active_power * _HOURS_PER_MONTH * _KWH_PER_WATT_HOUR * _ELECTRICITY_COST_PER_KWH

        efficiency = max(0.0, min(100.0, (1 - power_waste_watts / max(total_power_draw, 1)) * 100))
        recommended_cap = total_power_cap * 0.8
        cap_savings = 15.0

        annual_cost = (total_power_draw * _HOURS_PER_MONTH * 12 * _KWH_PER_WATT_HOUR * _ELECTRICITY_COST_PER_KWH)
        annual_kwh = total_power_draw * _HOURS_PER_MONTH * 12 * _KWH_PER_WATT_HOUR
        annual_carbon_kg = annual_kwh * (_GRID_CARBON_INTENSITY_G_PER_KWH / 1000)

        recs: list[str] = []
        if power_waste_watts > total_power_draw * 0.2:
            recs.append(f"Power waste {power_waste_watts:.0f}W ({(power_waste_watts/max(total_power_draw,1)*100):.0f}% of total) — consolidate idle GPUs")
        if idle_power > active_power and total_gpus > 1:
            recs.append("More power used by idle than active GPUs — consider powering down unused nodes")
        if util_pct < 30 and total_gpus > 4:
            recs.append("Low average utilization — right-size cluster to reduce base power consumption")
        if total_power_draw > total_power_cap * 0.85:
            recs.append(f"Cluster at {total_power_draw/max(total_power_cap,1)*100:.0f}% of power capacity — consider expansion")
        if annual_carbon_kg > 10000:
            recs.append(f"Estimated {annual_carbon_kg/1000:.1f} tonnes CO2/year — consider renewable energy or offsets")
        recs.append(f"Power capping to {recommended_cap:.0f}W (80%) could save ~{cap_savings:.0f}% without significant perf impact")
        recs.append("Enable GPU power capping via nvidia-smi -pm ENABLED and nvidia-smi -pl <power_limit>")

        return PowerAnalysisResult(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            total_gpus=total_gpus,
            total_power_draw_watts=round(total_power_draw, 1),
            total_power_capacity_watts=round(total_power_cap, 1),
            utilization_percent=util_pct,
            idle_power_watts=round(idle_power, 1),
            idle_power_cost_monthly=round(idle_cost_monthly, 2),
            active_power_watts=round(active_power, 1),
            active_power_cost_monthly=round(active_cost_monthly, 2),
            power_waste_watts=round(power_waste_watts, 1),
            power_waste_kwh_daily=round(waste_kwh_daily, 2),
            power_waste_cost_monthly=round(waste_cost_monthly, 2),
            power_efficiency_score=round(efficiency, 1),
            recommended_power_cap_watts=round(recommended_cap, 1),
            power_cap_savings_percent=cap_savings,
            estimated_annual_power_cost=round(annual_cost, 2),
            estimated_annual_carbon_kg=round(annual_carbon_kg, 1),
            recommendations=recs,
        )

    def estimate_carbon(self, cluster_id: UUID) -> CarbonEmissionsEstimate:
        cluster = self._repository.get_cluster(cluster_id)
        cluster_name = cluster.name if cluster else "unknown"
        power = self.analyze_power(cluster_id)
        total_kwh = power.total_power_draw_watts * _HOURS_PER_MONTH * _KWH_PER_WATT_HOUR
        carbon_kg = total_kwh * (_GRID_CARBON_INTENSITY_G_PER_KWH / 1000)
        carbon_tons = carbon_kg / 1000
        miles = carbon_kg / _CARBON_PER_MILE
        homes = total_kwh * 12 / max(_HOMES_ENERGY_PER_YEAR_KWH, 1)
        offset_cost = carbon_tons * 15.0

        recs: list[str] = []
        if carbon_tons > 10:
            recs.append(f"Carbon footprint equivalent to driving {miles:,.0f} miles — consider offsets")
        if carbon_tons > 100:
            recs.append("Significant carbon emitter — evaluate renewable energy procurement")
        recs.append(f"Estimated offset cost: ${offset_cost:.0f} at $15/tonne CO2")
        if _GRID_CARBON_INTENSITY_G_PER_KWH > 300:
            recs.append("High grid carbon intensity — consider locating workloads in greener regions")

        return CarbonEmissionsEstimate(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            total_energy_kwh=round(total_kwh, 2),
            grid_carbon_intensity_g_per_kwh=_GRID_CARBON_INTENSITY_G_PER_KWH,
            carbon_footprint_kg_co2=round(carbon_kg, 2),
            carbon_footprint_tons_co2=round(carbon_tons, 3),
            equivalent_miles_driven=round(miles, 1),
            equivalent_homes_energy=round(homes, 2),
            low_carbon_energy_percent=25.0,
            recommended_offset_cost_usd=round(offset_cost, 2),
            recommendations=recs,
        )

    @staticmethod
    def suggest_power_cap(
        gpu_model: str = "a100",
        gpu_count: int = 8,
        current_power_watts: float = 0.0,
    ) -> PowerCapSuggestion:
        profile = None
        key = gpu_model.lower().replace("nvidia ", "")
        for p in GPU_POWER_PROFILES:
            if p.gpu_model == key:
                profile = p
                break

        if profile is None:
            profile_type = type("profile", (), {"tdp_watts": 300.0, "typical_load_power_watts": 250.0, "idle_power_watts": 40.0})
            current = current_power_watts if current_power_watts > 0 else 250.0
            tdp = 300.0
        else:
            current = current_power_watts if current_power_watts > 0 else profile.typical_load_power_watts
            tdp = profile.tdp_watts

        tdp_pct = round(current / max(tdp, 1) * 100, 1)

        cap_80 = tdp * 0.8
        cap_70 = tdp * 0.7
        cap_60 = tdp * 0.6

        if tdp_pct > 90:
            recommended_cap = cap_80
            perf_impact = 5.0
            risk = "low"
            temp_reduction = 8.0
        elif tdp_pct > 75:
            recommended_cap = cap_70
            perf_impact = 10.0
            risk = "medium"
            temp_reduction = 12.0
        else:
            recommended_cap = cap_70
            perf_impact = 8.0
            risk = "low"
            temp_reduction = 10.0

        power_savings_w = max(current - recommended_cap, 0)
        power_savings_monthly_kwh = power_savings_w * _HOURS_PER_MONTH * _KWH_PER_WATT_HOUR
        cost_savings = power_savings_monthly_kwh * _ELECTRICITY_COST_PER_KWH

        recs: list[str] = [
            f"Cap power from {current:.0f}W ({tdp_pct:.0f}% TDP) to {recommended_cap:.0f}W ({recommended_cap/tdp*100:.0f}% TDP)",
        ]
        if perf_impact <= 5:
            recs.append(f"Minimal performance impact expected ({perf_impact:.0f}%) — safe for most workloads")
        elif perf_impact <= 10:
            recs.append(f"Moderate performance impact ({perf_impact:.0f}%) — test with representative workload first")
        if power_savings_w > 0:
            recs.append(f"Estimated power reduction {power_savings_w:.0f}W per GPU — "
                         f"save ${cost_savings:.0f}/mo across {gpu_count} GPUs")
        recs.append(f"Apply via: nvidia-smi -pl {int(recommended_cap)}")
        recs.append("Monitor performance regression with power capping enabled")

        return PowerCapSuggestion(
            gpu_model=gpu_model,
            gpu_count=gpu_count,
            current_power_watts=round(current, 1),
            current_tdp_percent=tdp_pct,
            recommended_cap_watts=round(recommended_cap, 1),
            recommended_cap_percent=round(recommended_cap / max(tdp, 1) * 100, 1),
            estimated_performance_impact_percent=perf_impact,
            estimated_power_savings_watts=round(power_savings_w, 1),
            estimated_power_savings_monthly_kwh=round(power_savings_monthly_kwh, 2),
            estimated_cost_savings_monthly=round(cost_savings * gpu_count, 2),
            estimated_temperature_reduction_c=temp_reduction,
            risk_level=risk,
            recommendations=recs,
        )

    def generate_power_recommendations(self, cluster_id: UUID) -> list[ResourceRecommendation]:
        recs: list[ResourceRecommendation] = []
        cluster = self._repository.get_cluster(cluster_id)
        if cluster is None:
            return recs

        power = self.analyze_power(cluster_id)

        if power.power_waste_watts > power.total_power_draw_watts * 0.15:
            recs.append(ResourceRecommendation(
                type=RecommendationType.EFFICIENCY,
                severity=RecommendationSeverity.HIGH,
                title="Reduce GPU power waste from idle resources",
                description=f"{power.power_waste_watts:.0f}W power waste ({(power.power_waste_watts/max(power.total_power_draw_watts,1)*100):.0f}% of draw) "
                            f"costs ${power.power_waste_cost_monthly:.0f}/mo.",
                reasoning=f"Idle GPUs consume {power.idle_power_watts:.0f}W, wasting {power.power_waste_kwh_daily:.1f}kWh/day. "
                          "Consolidating workloads or powering down idle nodes can eliminate most of this waste.",
                expected_impact=f"Up to ${power.power_waste_cost_monthly*12:.0f}/yr savings and "
                                f"{power.estimated_annual_carbon_kg/1000:.1f} tonnes CO2 reduction.",
                confidence=0.85,
                risk_level="low",
                affected_resources=[f"cluster/{cluster_id}"],
                actions=[
                    "Identify and power down idle GPU nodes",
                    "Consolidate workloads onto fewer nodes",
                    "Implement node auto-scaling based on demand",
                ],
                estimated_savings={
                    "annual_power_cost_savings": round(power.power_waste_cost_monthly * 12, 2),
                    "annual_kwh_savings": round(power.power_waste_watts * _HOURS_PER_MONTH * 12 * _KWH_PER_WATT_HOUR, 2),
                },
            ))

        if power.utilization_percent < 40:
            recs.append(ResourceRecommendation(
                type=RecommendationType.RIGHT_SIZING,
                severity=RecommendationSeverity.MEDIUM,
                title="Right-size cluster to reduce power consumption",
                description=f"Average GPU utilization only {power.utilization_percent:.0f}% — cluster is over-provisioned.",
                reasoning=f"At {power.utilization_percent:.0f}% average utilization, the cluster could be right-sized "
                          f"to reduce power consumption by up to {100 - power.utilization_percent:.0f}%.",
                expected_impact=f"Estimated {power.estimated_annual_power_cost:.0f}/yr at current size.",
                confidence=0.75,
                risk_level="medium",
                affected_resources=[f"cluster/{cluster_id}"],
                actions=[
                    "Analyze peak vs average GPU demand",
                    "Reduce cluster size to match 80th percentile demand",
                    "Use burstable/spot instances for peak handling",
                ],
                estimated_savings={
                    "potential_annual_savings": round(power.estimated_annual_power_cost * 0.3, 2),
                },
            ))

        carbon = self.estimate_carbon(cluster_id)
        if carbon.carbon_footprint_tons_co2 > 5:
            recs.append(ResourceRecommendation(
                type=RecommendationType.RISK_MITIGATION,
                severity=RecommendationSeverity.LOW,
                title="Reduce GPU carbon footprint",
                description=f"Cluster estimated to emit {carbon.carbon_footprint_tons_co2:.1f} tonnes CO2/year — "
                            f"equivalent to driving {carbon.equivalent_miles_driven:,.0f} miles.",
                reasoning="GPU computing has significant carbon impact. Reducing power waste, using renewable energy, "
                          "and optimizing workload efficiency can reduce emissions.",
                expected_impact=f"Offsets estimated at ${carbon.recommended_offset_cost_usd:.0f}/year.",
                confidence=0.7,
                risk_level="low",
                affected_resources=[f"cluster/{cluster_id}"],
                actions=[
                    "Purchase carbon offsets for GPU emissions",
                    "Schedule workloads during periods of low-carbon grid energy",
                    "Evaluate renewable energy options for data center",
                ],
                estimated_savings={"carbon_tons_per_year": round(carbon.carbon_footprint_tons_co2, 1)},
            ))

        return recs
