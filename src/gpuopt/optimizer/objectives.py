from __future__ import annotations

import math

from .models import (
    NodeCandidate,
    ObjectiveScore,
    ObjectiveWeight,
    OptimizationRequest,
    WorkloadSpec,
)


class ObjectiveScorer:
    def score_all(
        self,
        request: OptimizationRequest,
        workload: WorkloadSpec,
        node: NodeCandidate,
        tenant_weights: ObjectiveWeight | None = None,
    ) -> list[ObjectiveScore]:
        weights = tenant_weights or request.global_weights
        scorers = [
            ("gpu_utilization", self._score_gpu_utilization, weights.gpu_utilization),
            ("throughput", self._score_throughput, weights.throughput),
            ("queue_time_reduction", self._score_queue_time_reduction, weights.queue_time_reduction),
            ("job_completion_time", self._score_job_completion_time, weights.job_completion_time),
            ("gpu_hours_per_token", self._score_gpu_hours_per_token, weights.gpu_hours_per_token),
            ("power_efficiency", self._score_power_efficiency, weights.power_efficiency),
            ("carbon_footprint", self._score_carbon_footprint, weights.carbon_footprint),
            ("fairness", self._score_fairness, weights.fairness),
            ("starvation_reduction", self._score_starvation_reduction, weights.starvation_reduction),
            ("minimal_movement", self._score_minimal_movement, weights.minimal_movement),
            ("operational_churn", self._score_operational_churn, weights.operational_churn),
        ]
        results: list[ObjectiveScore] = []
        for name, scorer_fn, weight in scorers:
            raw = scorer_fn(workload, node)
            score = max(0.0, min(1.0, raw))
            results.append(ObjectiveScore(
                objective=name,
                score=score,
                weight=weight,
                weighted_score=score * weight,
            ))
        return results

    def total_utility(self, scores: list[ObjectiveScore]) -> float:
        if not scores:
            return 0.0
        total_weight = sum(s.weight for s in scores)
        if total_weight == 0:
            return 0.0
        return round(sum(s.weighted_score for s in scores) / total_weight * 100, 2)

    # ── Individual scorers (normalized 0-1) ───────────────────

    def _score_gpu_utilization(self, wl: WorkloadSpec, node: NodeCandidate) -> float:
        util = node.current_gpu_utilization_pct
        if util <= 0:
            return 0.3
        if util < 30:
            return 0.1 + util / 30 * 0.3
        if util < 70:
            return 0.4 + (util - 30) / 40 * 0.4
        if util < 90:
            return 0.8 + (util - 70) / 20 * 0.15
        return 0.95

    def _score_throughput(self, wl: WorkloadSpec, node: NodeCandidate) -> float:
        if wl.estimated_tokens_per_step <= 0 and wl.estimated_samples_per_step <= 0:
            return 0.5
        gpu_ratio = node.free_gpus / max(node.total_gpus, 1)
        base = min(gpu_ratio * 2, 1.0)
        mem_ratio = min(node.gpu_memory_per_gpu_gb / 80.0, 1.5) if node.gpu_memory_per_gpu_gb > 0 else 1.0
        return min(base * mem_ratio, 1.0)

    def _score_queue_time_reduction(self, wl: WorkloadSpec, node: NodeCandidate) -> float:
        free_ratio = node.free_gpus / max(node.total_gpus, 1)
        if free_ratio >= 0.5:
            return 1.0
        if free_ratio >= 0.25:
            return 0.7 + (free_ratio - 0.25) / 0.25 * 0.3
        if free_ratio >= 0.1:
            return 0.3 + (free_ratio - 0.1) / 0.15 * 0.4
        return free_ratio / 0.1 * 0.3

    def _score_job_completion_time(self, wl: WorkloadSpec, node: NodeCandidate) -> float:
        free_gpus = node.free_gpus
        needed = wl.gpu_count
        if needed <= 0:
            return 0.5
        ratio = free_gpus / needed
        if ratio >= 2:
            return 0.9
        if ratio >= 1:
            return 0.7 + (ratio - 1) * 0.2
        if ratio >= 0.5:
            return 0.3 + (ratio - 0.5) / 0.5 * 0.4
        return ratio / 0.5 * 0.3

    def _score_gpu_hours_per_token(self, wl: WorkloadSpec, node: NodeCandidate) -> float:
        if wl.estimated_tokens_per_step <= 0:
            return 0.5
        gpu_mem = node.gpu_memory_per_gpu_gb
        if gpu_mem <= 0:
            return 0.5
        if gpu_mem >= 80:
            return 0.9
        if gpu_mem >= 40:
            return 0.5 + (gpu_mem - 40) / 40 * 0.4
        return gpu_mem / 40 * 0.5

    def _score_power_efficiency(self, wl: WorkloadSpec, node: NodeCandidate) -> float:
        if node.current_power_watts <= 0:
            return 0.5
        util = node.current_gpu_utilization_pct
        if util <= 0:
            return 0.3
        efficiency = util / max(node.current_power_watts, 1) * 10
        return min(efficiency, 1.0)

    def _score_carbon_footprint(self, wl: WorkloadSpec, node: NodeCandidate) -> float:
        carbon = node.carbon_intensity_g_per_kwh
        if carbon <= 0:
            return 0.5
        if carbon < 100:
            return 0.9
        if carbon < 300:
            return 0.7
        if carbon < 500:
            return 0.4
        return 0.2

    def _score_fairness(self, wl: WorkloadSpec, node: NodeCandidate) -> float:
        running = node.running_jobs
        if running <= 0:
            return 0.8
        load = running / max(node.total_gpus, 1)
        if load < 0.3:
            return 0.9
        if load < 0.6:
            return 0.7
        if load < 0.8:
            return 0.4
        return 0.2

    def _score_starvation_reduction(self, wl: WorkloadSpec, node: NodeCandidate) -> float:
        free = node.free_gpus
        total = node.total_gpus
        if total <= 0:
            return 0.5
        free_ratio = free / total
        return 1.0 - free_ratio

    def _score_minimal_movement(self, wl: WorkloadSpec, node: NodeCandidate) -> float:
        if wl.gpu_count <= 0:
            return 1.0
        if node.free_gpus >= wl.gpu_count:
            return 1.0
        ratio = node.free_gpus / wl.gpu_count
        return max(0.1, ratio)

    def _score_operational_churn(self, wl: WorkloadSpec, node: NodeCandidate) -> float:
        running = node.running_jobs
        if running <= 0:
            return 1.0
        if running < 3:
            return 0.8
        if running < 10:
            return 0.5
        return 0.2
