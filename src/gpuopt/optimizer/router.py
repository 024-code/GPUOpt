from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from .constraints import ConstraintEngine
from .models import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    OptimizationRequest,
    OptimizationResult,
    ObjectiveWeight,
    WorkloadSpec,
    NodeCandidate,
    TenantObjectiveProfile,
)
from .optimizer import Optimizer

logger = logging.getLogger(__name__)

optimizer_router = APIRouter(prefix="/api/v1/optimizer", tags=["optimizer"])


def _optimizer() -> Optimizer:
    return Optimizer()


@optimizer_router.post("/optimize")
def optimize(
    request: OptimizationRequest,
    optimizer: Optimizer = Depends(_optimizer),
) -> OptimizationResult:
    return optimizer.optimize(request)


@optimizer_router.post("/evaluate")
def evaluate_workload(
    workload: WorkloadSpec,
    nodes: list[NodeCandidate],
    optimizer: Optimizer = Depends(_optimizer),
) -> OptimizationResult:
    return optimizer.evaluate_workload(workload, nodes)


@optimizer_router.post("/weigh")
def set_weights(
    weights: ObjectiveWeight,
    optimizer: Optimizer = Depends(_optimizer),
) -> dict[str, object]:
    optimizer.set_global_weights(weights)
    return {"status": "ok", "weights": weights.model_dump()}


@optimizer_router.get("/default-weights")
def default_weights() -> dict[str, float]:
    return dict(DEFAULT_OBJECTIVE_WEIGHTS)


@optimizer_router.post("/batch")
def batch_optimize(
    requests: list[OptimizationRequest],
    optimizer: Optimizer = Depends(_optimizer),
) -> list[OptimizationResult]:
    return [optimizer.optimize(req) for req in requests]


@optimizer_router.post("/check-constraints")
def check_constraints(
    workload: WorkloadSpec,
    node: NodeCandidate,
    tenant_profile: TenantObjectiveProfile | None = None,
) -> list[object]:
    engine = ConstraintEngine()
    req = OptimizationRequest(workloads=[workload], candidates=[node])
    return engine.evaluate(req, workload, node, tenant_profile)
