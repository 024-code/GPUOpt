from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from .ml.recommendation_model import RecommendationModel
from .repository import ClusterRepository
from .schemas import (
    ClusterStateData,
    RecommendationSeverity,
    RecommendationSet,
    RecommendationStatus,
    RecommendationType,
    ResourceRecommendation,
    WhatIfProjection,
    WorkloadAnalysisResult,
)

logger = logging.getLogger(__name__)


class RecommendationEngine:
    def __init__(self, repository: ClusterRepository, ml_model: RecommendationModel | None = None) -> None:
        self.repository = repository
        self.ml_model = ml_model or RecommendationModel()

    def generate(self, cluster_id: UUID) -> RecommendationSet:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")

        state = self.repository.latest_state(cluster_id)
        analysis = self.repository.latest_analysis(cluster_id)

        if state is None and analysis is None:
            raise KeyError("No state or analysis data available to generate recommendations")

        recs: list[ResourceRecommendation] = []
        now = datetime.now(timezone.utc)

        if state:
            recs.extend(self._gpu_placement_recs(state, now))
            recs.extend(self._risk_mitigation_recs(state, now))

        if analysis:
            recs.extend(self._efficiency_recs(analysis, now))
            recs.extend(self._scaling_recs(analysis, now))

        recs.extend(self._right_sizing_recs(state, analysis, now))

        temp_set = RecommendationSet(
            cluster_id=cluster.id, cluster_name=cluster.name,
            environment=cluster.environment, recommendations=recs,
        )
        if self.ml_model.get_training_count() > 0:
            recs = self.ml_model.score_recommendation_set(temp_set, state, analysis)
        else:
            recs = [self._score_rec(r, state, analysis) for r in recs]
            recs.sort(key=lambda r: (-r.score, -r.confidence))

        critical = sum(1 for r in recs if r.severity == RecommendationSeverity.CRITICAL)
        high = sum(1 for r in recs if r.severity == RecommendationSeverity.HIGH)
        medium = sum(1 for r in recs if r.severity == RecommendationSeverity.MEDIUM)

        avg_score = sum(r.score for r in recs) / max(len(recs), 1)
        total_gpu_hours = sum(
            r.estimated_savings.get("potential_idle_reduction_hours", 0)
            for r in recs
        )

        summary_parts = [f"{len(recs)} recommendation(s)"]
        if critical:
            summary_parts.append(f"{critical} critical")
        if high:
            summary_parts.append(f"{high} high")
        if medium:
            summary_parts.append(f"{medium} medium")
        summary_parts.append(f"avg score {avg_score:.0f}/100")

        top_rec = max(recs, key=lambda r: r.score) if recs else None

        result = RecommendationSet(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            based_on_state_at=state.collected_at if state else None,
            based_on_analysis_at=analysis.generated_at if analysis else None,
            recommendation_count=len(recs),
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            recommendations=recs,
            summary="; ".join(summary_parts),
            avg_score=round(avg_score, 1),
            total_estimated_savings_gpu_hours=round(total_gpu_hours, 1),
            top_recommendation=top_rec.title if top_rec else "",
        )

        self.repository.save_recommendations(result)
        return result

    def update_status(self, cluster_id: UUID, rec_id: UUID, status: str, reason: str = "") -> ResourceRecommendation:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        rec_set = self.repository.latest_recommendations(cluster_id)
        old_status = "pending"
        found_rec = None
        if rec_set:
            for r in rec_set.recommendations:
                if r.id == rec_id:
                    old_status = r.status.value
                    found_rec = r
                    break
        updated = self.repository.update_rec_status(cluster_id, rec_id, status, reason)
        if updated is None:
            raise KeyError(f"Recommendation not found: {rec_id}")
        if found_rec and old_status != status:
            state = self.repository.latest_state(cluster_id)
            analysis = self.repository.latest_analysis(cluster_id)
            self.ml_model.train_from_status(found_rec, old_status, status, state, analysis)
        return updated

    def what_if(self, cluster_id: UUID) -> WhatIfProjection:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        state = self.repository.latest_state(cluster_id)
        analysis = self.repository.latest_analysis(cluster_id)
        rec_set = self.repository.latest_recommendations(cluster_id)

        if rec_set is None:
            raise KeyError("No recommendations available for what-if analysis")

        if state is None:
            projected_util = self.ml_model.score_recommendation_set(rec_set, None, None)[0].score if rec_set.recommendations else 0.0
        else:
            used_gpus = sum(1 for n in state.nodes for g in n.gpu_devices if g.memory_used_bytes > 0)
            total_gpus = max(sum(1 for n in state.nodes for _ in n.gpu_devices), 1)
            current_util = used_gpus / total_gpus * 100

            approved = [r for r in rec_set.recommendations if r.status.value == "pending" or r.status.value == "approved"]
            improvement = min(len(approved) * 5, 40)
            projected_util = min(current_util + improvement, 100)

        if analysis is None:
            projected_efficiency = 50.0
            projected_idle_reduction = 0.0
            projected_power_savings = 0.0
        else:
            projected_efficiency = min(analysis.overall_efficiency_score + 15, 100)
            projected_idle_reduction = analysis.total_idle_gpu_hours * 0.4
            projected_power_savings = analysis.estimated_power_waste_kwh * 0.35

        cost_savings = projected_power_savings * 0.12

        fragmentation_improvement = min(len(rec_set.recommendations) * 3, 25)

        reservations_freed = sum(
            1 for r in rec_set.recommendations
            if r.type in (RecommendationType.PLACEMENT, RecommendationType.RIGHT_SIZING)
        )

        active_recs = len([r for r in rec_set.recommendations if r.status.value != "dismissed"])
        risk_reduction = min(active_recs * 8, 80)

        summary_parts = [
            f"Projected utilization: {projected_util:.0f}%",
            f"efficiency: {projected_efficiency:.0f}/100",
            f"savings: {projected_power_savings:.1f} kWh",
        ]

        return WhatIfProjection(
            cluster_id=cluster.id,
            cluster_name=cluster.name,
            environment=cluster.environment,
            recommendation_set_id=rec_set.id,
            projected_gpu_utilization_percent=round(projected_util, 1),
            projected_efficiency_score=round(projected_efficiency, 1),
            projected_idle_gpu_hours_reduction=round(projected_idle_reduction, 1),
            projected_power_savings_kwh=round(projected_power_savings, 1),
            estimated_cost_savings_usd=round(cost_savings, 2),
            fragmentation_improvement_percent=round(fragmentation_improvement, 1),
            reservations_freed=reservations_freed,
            risk_reduction_score=round(risk_reduction, 1),
            summary="; ".join(summary_parts),
        )

    def get_latest(self, cluster_id: UUID) -> RecommendationSet | None:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        return self.repository.latest_recommendations(cluster_id)

    def list_recommendations(self, cluster_id: UUID, limit: int = 10) -> list[RecommendationSet]:
        cluster = self.repository.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"Cluster not found: {cluster_id}")
        return self.repository.list_recommendations(cluster_id, limit=limit)

    @staticmethod
    def _score_rec(rec: ResourceRecommendation, state: ClusterStateData | None,
                   analysis: WorkloadAnalysisResult | None) -> ResourceRecommendation:
        severity_scores = {"critical": 90, "high": 70, "medium": 50, "low": 30, "info": 10}
        base = severity_scores.get(rec.severity.value, 30)
        confidence_component = rec.confidence * 100 * 0.3
        risk = 0.0
        if rec.risk_level == "high":
            risk = 10.0
        elif rec.risk_level == "medium":
            risk = 5.0
        savings = 0.0
        for v in rec.estimated_savings.values():
            if isinstance(v, (int, float)):
                savings += min(abs(v), 20)
        score = min(base + confidence_component + risk + savings, 100)
        rec.score = round(score, 1)
        return rec

    @staticmethod
    def _gpu_placement_recs(state: ClusterStateData, now: datetime) -> list[ResourceRecommendation]:
        recs: list[ResourceRecommendation] = []
        for node in state.nodes:
            free_gpus = [g for g in node.gpu_devices if g.memory_used_bytes < g.memory_total_bytes * 0.1]
            hot_gpus = [g for g in node.gpu_devices if g.memory_used_bytes > g.memory_total_bytes * 0.85]
            if free_gpus and hot_gpus:
                for hot in hot_gpus:
                    free = free_gpus[0]
                    ratio = round(hot.memory_used_bytes / max(free.memory_total_bytes, 1) * 100, 1)
                    recs.append(ResourceRecommendation(
                        type=RecommendationType.PLACEMENT,
                        severity=RecommendationSeverity.MEDIUM,
                        title="GPU memory imbalance across devices",
                        description=f"GPU {hot.index} on {node.name} is at {ratio}% memory usage "
                                    f"while GPU {free.index} on same node is nearly idle.",
                        reasoning=f"GPU {hot.index} ({hot.uuid}): {hot.memory_used_bytes / (1024**3):.0f}GiB used / "
                                  f"{hot.memory_total_bytes / (1024**3):.0f}GiB total. "
                                  f"GPU {free.index} ({free.uuid}): {free.memory_used_bytes / (1024**3):.0f}GiB used.",
                        expected_impact="Better memory-balanced GPU utilization across available devices.",
                        confidence=0.65,
                        risk_level="low",
                        affected_resources=[hot.uuid, free.uuid],
                        actions=[f"Review pod placement for GPU {hot.index}", f"Consider rebalancing workloads between GPU {hot.index} and GPU {free.index}"],
                        estimated_savings={"memory_balance_improvement": round(ratio - 10, 1)},
                    ))
            if free_gpus and len(free_gpus) == len(node.gpu_devices):
                recs.append(ResourceRecommendation(
                    type=RecommendationType.PLACEMENT,
                    severity=RecommendationSeverity.LOW,
                    title=f"All GPUs idle on {node.name}",
                    description=f"All {len(node.gpu_devices)} GPU(s) on {node.name} are below 10% memory utilization.",
                    reasoning="Entire node's GPU capacity is unused. Workloads could be placed here without contention.",
                    expected_impact="Utilize available GPU capacity.",
                    confidence=0.9,
                    risk_level="low",
                    affected_resources=[g.uuid for g in free_gpus],
                    actions=[f"Schedule pending GPU workloads to {node.name}"],
                    estimated_savings={"unused_gpu_count": len(free_gpus)},
                ))
        return recs

    @staticmethod
    def _risk_mitigation_recs(state: ClusterStateData, now: datetime) -> list[ResourceRecommendation]:
        recs: list[ResourceRecommendation] = []
        for node in state.nodes:
            for gpu in node.gpu_devices:
                if gpu.memory_total_bytes > 0:
                    mem_pct = gpu.memory_used_bytes / gpu.memory_total_bytes * 100
                    if mem_pct > 90:
                        recs.append(ResourceRecommendation(
                            type=RecommendationType.RISK_MITIGATION,
                            severity=RecommendationSeverity.CRITICAL,
                            title=f"OOM risk on {node.name} GPU {gpu.index}",
                            description=f"GPU memory usage at {mem_pct:.0f}% ({gpu.memory_used_bytes / (1024**3):.0f}/{gpu.memory_total_bytes / (1024**3):.0f} GiB).",
                            reasoning=f"Memory pressure exceeds 90% threshold. Workload may OOM if demand increases.",
                            expected_impact="Prevent GPU OOM failures and workload disruption.",
                            confidence=0.85,
                            risk_level="high",
                            affected_resources=[gpu.uuid],
                            actions=["Migrate or resize workload consuming this GPU", "Consider a GPU with larger memory capacity"],
                            estimated_savings={"oom_risk_reduction": 1},
                        ))
                    elif mem_pct > 75:
                        recs.append(ResourceRecommendation(
                            type=RecommendationType.RISK_MITIGATION,
                            severity=RecommendationSeverity.HIGH,
                            title=f"Elevated memory pressure on {node.name} GPU {gpu.index}",
                            description=f"GPU memory at {mem_pct:.0f}% — approaching critical threshold.",
                            reasoning="Sustained high memory usage increases OOM risk during workload spikes.",
                            expected_impact="Reduce memory-related failure risk.",
                            confidence=0.7,
                            risk_level="medium",
                            affected_resources=[gpu.uuid],
                            actions=["Monitor memory growth trend", "Evaluate if workload can use memory more efficiently"],
                            estimated_savings={"risk_reduction": 1},
                        ))
        return recs

    @staticmethod
    def _efficiency_recs(analysis: WorkloadAnalysisResult, now: datetime) -> list[ResourceRecommendation]:
        recs: list[ResourceRecommendation] = []
        for ne in analysis.node_efficiencies:
            if ne.gpu_idle_percent > 60 and ne.gpu_count > 0:
                recs.append(ResourceRecommendation(
                    type=RecommendationType.EFFICIENCY,
                    severity=RecommendationSeverity.HIGH,
                    title=f"GPUs on {ne.node_name} idle {ne.gpu_idle_percent:.0f}% of the time",
                    description=f"{ne.gpu_count} GPU(s) on {ne.node_name} are idle over {ne.gpu_idle_percent:.0f}% of observed period.",
                    reasoning=f"Efficiency score: {ne.efficiency_score}/100. Idle GPUs waste power without contributing to throughput.",
                    expected_impact="Reduce power consumption and free GPU capacity for other workloads.",
                    confidence=0.8,
                    risk_level="low",
                    affected_resources=[f"{ne.node_name}/gpu-{i}" for i in range(ne.gpu_count)],
                    actions=["Consolidate workloads to fewer nodes", "Power-off idle GPUs or node", "Reschedule batch workloads to fill gaps"],
                    estimated_savings={"potential_idle_reduction_hours": round(analysis.total_idle_gpu_hours, 1)},
                ))
            if ne.avg_gpu_utilization_percent < 20 and ne.gpu_count > 0:
                recs.append(ResourceRecommendation(
                    type=RecommendationType.EFFICIENCY,
                    severity=RecommendationSeverity.MEDIUM,
                    title=f"Very low GPU utilization on {ne.node_name}",
                    description=f"Average GPU utilization is only {ne.avg_gpu_utilization_percent:.0f}% across {ne.gpu_count} GPU(s).",
                    reasoning="GPUs are significantly underutilized. This may indicate oversized GPU allocations.",
                    expected_impact="Improve resource utilization and reduce cost.",
                    confidence=0.75,
                    risk_level="low",
                    affected_resources=[f"{ne.node_name}/gpu-{i}" for i in range(ne.gpu_count)],
                    actions=["Investigate if workloads need the current GPU count", "Consider GPU sharing or MIG partitioning"],
                    estimated_savings={"utilization_gain": round(80 - ne.avg_gpu_utilization_percent, 1)},
                ))
        return recs

    @staticmethod
    def _scaling_recs(analysis: WorkloadAnalysisResult, now: datetime) -> list[ResourceRecommendation]:
        recs: list[ResourceRecommendation] = []
        if analysis.overall_efficiency_score < 30 and analysis.gpu_count >= 4:
            recs.append(ResourceRecommendation(
                type=RecommendationType.SCALING,
                severity=RecommendationSeverity.MEDIUM,
                title=f"Cluster-wide efficiency below 30% across {analysis.gpu_count} GPUs",
                description=f"Overall efficiency score: {analysis.overall_efficiency_score}/100. "
                            f"Estimated power waste: {analysis.estimated_power_waste_kwh} kWh.",
                reasoning="Persistent low efficiency across the cluster suggests over-provisioning relative to workload demand.",
                expected_impact="Reduce operational cost and power consumption.",
                confidence=0.7,
                risk_level="medium",
                affected_resources=[f"{ne.node_name}" for ne in analysis.node_efficiencies if ne.efficiency_score < 40],
                actions=["Evaluate reducing GPU node count", "Consolidate workloads before adding new capacity", "Review GPU allocation policies"],
                estimated_savings={"estimated_kwh_savings": analysis.estimated_power_waste_kwh},
            ))
        return recs

    @staticmethod
    def _right_sizing_recs(state: ClusterStateData | None, analysis: WorkloadAnalysisResult | None,
                           now: datetime) -> list[ResourceRecommendation]:
        recs: list[ResourceRecommendation] = []
        if state and analysis:
            for trend in analysis.gpu_trends:
                if trend.memory_total_bytes > 0 and trend.avg_memory_used_bytes > 0:
                    util_ratio = trend.avg_memory_used_bytes / trend.memory_total_bytes
                    if util_ratio < 0.3 and trend.sample_count > 1:
                        recs.append(ResourceRecommendation(
                            type=RecommendationType.RIGHT_SIZING,
                            severity=RecommendationSeverity.LOW,
                            title=f"GPU {trend.gpu_uuid[:16]}... may be over-provisioned",
                            description=f"Average memory utilization: {util_ratio*100:.0f}% of {trend.memory_total_bytes / (1024**3):.0f}GiB.",
                            reasoning=f"GPU {trend.model} has {trend.memory_total_bytes / (1024**3):.0f}GiB but "
                                      f"workloads only use ~{trend.avg_memory_used_bytes / (1024**3):.0f}GiB on average.",
                            expected_impact="Lower cost by matching GPU memory to workload requirements.",
                            confidence=0.6,
                            risk_level="low",
                            affected_resources=[trend.gpu_uuid],
                            actions=[f"Evaluate if a smaller GPU (e.g., {trend.memory_total_bytes // 2 // (1024**3)}GiB) would suffice",
                                     "Consider MIG partitioning to share this GPU"],
                            estimated_savings={"potential_memory_reduction_gib": round(trend.memory_total_bytes / (1024**3) * (1 - util_ratio), 1)},
                        ))
        return recs
