from __future__ import annotations

import logging
from typing import Any

from .constraints import ConstraintEngine
from .models import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    ObjectiveWeight,
    OptimizationCandidate,
    OptimizationRequest,
    OptimizationResult,
    WorkloadSpec,
)
from .objectives import ObjectiveScorer

logger = logging.getLogger(__name__)


class Optimizer:
    def __init__(
        self,
        constraint_engine: ConstraintEngine | None = None,
        objective_scorer: ObjectiveScorer | None = None,
    ) -> None:
        self._constraints = constraint_engine or ConstraintEngine()
        self._objectives = objective_scorer or ObjectiveScorer()
        self._weights: dict[str, float] = dict(DEFAULT_OBJECTIVE_WEIGHTS)

    def set_global_weights(self, weights: ObjectiveWeight) -> None:
        self._weights = weights.model_dump()

    def optimize(self, request: OptimizationRequest) -> OptimizationResult:
        candidates: list[OptimizationCandidate] = []

        for workload in request.workloads:
            for node in request.candidates:
                tenant_profile = request.tenant_profiles.get(workload.tenant_id)
                tenant_weights = self._profile_weights(tenant_profile) if tenant_profile else None

                constraint_results = self._constraints.evaluate(request, workload, node, tenant_profile)
                feasible = self._constraints.all_feasible(constraint_results)

                objective_scores = self._objectives.score_all(request, workload, node, tenant_weights)
                utility = self._objectives.total_utility(objective_scores) if feasible else 0.0

                candidates.append(OptimizationCandidate(
                    workload=workload,
                    target_node=node,
                    constraints=constraint_results,
                    objective_scores=objective_scores,
                    total_utility=utility,
                    feasible=feasible,
                    action=self._suggest_action(workload, node, feasible),
                ))

        feasible = [c for c in candidates if c.feasible]
        infeasible = [c for c in candidates if not c.feasible]
        feasible.sort(key=lambda c: -c.total_utility)

        best = feasible[0] if feasible else None
        return OptimizationResult(
            request_id=request.id,
            candidates=candidates,
            feasible_count=len(feasible),
            infeasible_count=len(infeasible),
            best_candidate=best,
            summary=self._build_summary(request, best, feasible, infeasible),
        )

    def evaluate_workload(
        self,
        workload: WorkloadSpec,
        nodes: list[Any],
        global_weights: ObjectiveWeight | None = None,
    ) -> OptimizationResult:
        req = OptimizationRequest(
            workloads=[workload],
            candidates=nodes,
            global_weights=global_weights or ObjectiveWeight(),
        )
        return self.optimize(req)

    def _profile_weights(self, profile: Any) -> ObjectiveWeight:
        return ObjectiveWeight(
            gpu_utilization=profile.weights.gpu_utilization if hasattr(profile, 'weights') else 1.0,
            throughput=profile.weights.throughput if hasattr(profile, 'weights') else 1.0,
            queue_time_reduction=profile.weights.queue_time_reduction if hasattr(profile, 'weights') else 1.0,
            job_completion_time=profile.weights.job_completion_time if hasattr(profile, 'weights') else 1.0,
            gpu_hours_per_token=profile.weights.gpu_hours_per_token if hasattr(profile, 'weights') else 1.0,
            power_efficiency=profile.weights.power_efficiency if hasattr(profile, 'weights') else 1.0,
            carbon_footprint=profile.weights.carbon_footprint if hasattr(profile, 'weights') else 1.0,
            fairness=profile.weights.fairness if hasattr(profile, 'weights') else 1.0,
            starvation_reduction=profile.weights.starvation_reduction if hasattr(profile, 'weights') else 1.0,
            minimal_movement=profile.weights.minimal_movement if hasattr(profile, 'weights') else 1.0,
            operational_churn=profile.weights.operational_churn if hasattr(profile, 'weights') else 1.0,
        )

    @staticmethod
    def _suggest_action(wl: WorkloadSpec, node: Any, feasible: bool) -> str:
        if not feasible:
            return "reject"
        if wl.inference_deployment:
            return "deploy"
        if wl.preemptible:
            return "schedule_preemptible"
        return "schedule"

    @staticmethod
    def _build_summary(
        req: OptimizationRequest,
        best: OptimizationCandidate | None,
        feasible: list[OptimizationCandidate],
        infeasible: list[OptimizationCandidate],
    ) -> str:
        parts = [
            f"Optimized {len(req.workloads)} workload(s) across {len(req.candidates)} node(s)",
            f"feasible={len(feasible)} infeasible={len(infeasible)}",
        ]
        if best:
            parts.append(
                f"best utility={best.total_utility:.1f} "
                f"node={best.target_node.node_id if best.target_node else '?'} "
                f"action={best.action}"
            )
        return " | ".join(parts)
