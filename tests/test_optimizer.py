from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gpuopt.optimizer.constraints import ConstraintEngine
from gpuopt.optimizer.models import (
    HardConstraint,
    NodeCandidate,
    ObjectiveWeight,
    OptimizationRequest,
    TenantObjectiveProfile,
    WorkloadSpec,
)
from gpuopt.optimizer.objectives import ObjectiveScorer
from gpuopt.optimizer.optimizer import Optimizer


@pytest.fixture()
def sample_workload() -> WorkloadSpec:
    return WorkloadSpec(
        id="wl-1", tenant_id="team-a", job_name="train-llm",
        gpu_count=8, gpu_model="A100", memory_per_gpu_gb=80,
        requires_nvlink=True, estimated_runtime_minutes=1440,
        checkpoint_interval_minutes=120,
    )


@pytest.fixture()
def sample_nodes() -> list[NodeCandidate]:
    return [
        NodeCandidate(
            node_id="gpu-a1", cluster_id="c1", zone="us-east-1a",
            gpu_model="NVIDIA A100-SXM-80GB", total_gpus=8, free_gpus=8,
            gpu_memory_per_gpu_gb=80, has_nvlink=True,
            current_gpu_utilization_pct=45, current_power_watts=350,
            carbon_intensity_g_per_kwh=200, running_jobs=2,
        ),
        NodeCandidate(
            node_id="gpu-b1", cluster_id="c1", zone="us-east-1b",
            gpu_model="NVIDIA H100-SXM-80GB", total_gpus=8, free_gpus=2,
            gpu_memory_per_gpu_gb=80, has_nvlink=True,
            current_gpu_utilization_pct=75, current_power_watts=450,
            carbon_intensity_g_per_kwh=400, running_jobs=6,
        ),
        NodeCandidate(
            node_id="gpu-c1", cluster_id="c1", zone="us-east-1c",
            gpu_model="NVIDIA A100-SXM-40GB", total_gpus=4, free_gpus=4,
            gpu_memory_per_gpu_gb=40, has_nvlink=False,
            current_gpu_utilization_pct=20, current_power_watts=200,
            carbon_intensity_g_per_kwh=150, running_jobs=1,
        ),
    ]


# ── Constraint Tests ─────────────────────────────────────────

class TestConstraintEngine:
    def test_gpu_memory_pass(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        result = e._check_gpu_memory(sample_workload, sample_nodes[0])
        assert result.passed

    def test_gpu_memory_fail(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        heavy = sample_workload.model_copy(update={"memory_per_gpu_gb": 100})
        result = e._check_gpu_memory(heavy, sample_nodes[2])
        assert not result.passed
        assert "95% threshold" in result.reason

    def test_gpu_topology_pass(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        result = e._check_gpu_topology(sample_workload, sample_nodes[0])
        assert result.passed

    def test_gpu_topology_fail(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        result = e._check_gpu_topology(sample_workload, sample_nodes[2])
        assert not result.passed
        assert result.constraint == HardConstraint.GPU_TOPOLOGY

    def test_gpu_compatibility(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        result = e._check_gpu_compatibility(sample_workload, sample_nodes[1])
        assert not result.passed

    def test_tenant_quota_pass(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        profile = TenantObjectiveProfile(tenant_id="team-a", gpu_quota=32)
        result = e._check_tenant_quota(sample_workload, sample_nodes[0], profile)
        assert result.passed

    def test_tenant_quota_fail(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        profile = TenantObjectiveProfile(tenant_id="team-a", gpu_quota=4)
        result = e._check_tenant_quota(sample_workload, sample_nodes[0], profile)
        assert not result.passed

    def test_inference_slo_latency(self, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        wl = WorkloadSpec(id="inf-1", inference_deployment=True, memory_per_gpu_gb=160)
        profile = TenantObjectiveProfile(tenant_id="team-a", slo_max_latency_ms=50)
        result = e._check_inference_slo_latency(wl, profile)
        assert not result.passed
        assert "SLO" in result.reason

    def test_data_locality(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        wl = sample_workload.model_copy(update={"data_location": "us-east-1a"})
        result = e._check_data_locality(wl, sample_nodes[0])
        assert result.passed
        result = e._check_data_locality(wl, sample_nodes[1])
        assert not result.passed

    def test_approved_zones(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        wl = sample_workload.model_copy(update={"approved_zones": ["us-east-1a"]})
        result = e._check_approved_zones(wl, sample_nodes[0], None)
        assert result.passed
        result = e._check_approved_zones(wl, sample_nodes[1], None)
        assert not result.passed

    def test_blast_radius(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        req = OptimizationRequest(workloads=[sample_workload], candidates=sample_nodes)
        result = e._check_action_blast_radius(sample_workload, sample_nodes[0], req)
        assert result.passed
        tight = sample_workload.model_copy(update={"gpu_count": 16})
        result = e._check_action_blast_radius(tight, sample_nodes[0], req)
        assert not result.passed

    def test_all_constraints_on_best_node(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        req = OptimizationRequest(workloads=[sample_workload], candidates=sample_nodes)
        results = e.evaluate(req, sample_workload, sample_nodes[0])
        passed = [r for r in results if r.passed]
        assert len(passed) == len(results)

    def test_all_constraints_on_incompatible_node(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        e = ConstraintEngine()
        req = OptimizationRequest(workloads=[sample_workload], candidates=sample_nodes)
        results = e.evaluate(req, sample_workload, sample_nodes[2])
        failing = [r for r in results if not r.passed]
        assert len(failing) >= 2


# ── Objective Scorer Tests ───────────────────────────────────

class TestObjectiveScorer:
    def test_gpu_utilization_score(self, sample_nodes: list[NodeCandidate]):
        o = ObjectiveScorer()
        wl = WorkloadSpec(id="wl-1")
        assert 0 < o._score_gpu_utilization(wl, sample_nodes[0]) < 1
        assert o._score_gpu_utilization(wl, sample_nodes[2]) < o._score_gpu_utilization(wl, sample_nodes[1])

    def test_all_scores_return_normalized(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        o = ObjectiveScorer()
        req = OptimizationRequest(workloads=[sample_workload], candidates=sample_nodes)
        scores = o.score_all(req, sample_workload, sample_nodes[0])
        assert len(scores) == 11
        for s in scores:
            assert 0 <= s.score <= 1
            assert s.weight > 0
            assert s.weighted_score == s.score * s.weight

    def test_total_utility(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        o = ObjectiveScorer()
        req = OptimizationRequest(workloads=[sample_workload], candidates=sample_nodes)
        scores = o.score_all(req, sample_workload, sample_nodes[0])
        utility = o.total_utility(scores)
        assert 0 < utility <= 100

    def test_tenant_weights_apply(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        o = ObjectiveScorer()
        req = OptimizationRequest(workloads=[sample_workload], candidates=sample_nodes)
        tenant_weights = ObjectiveWeight(gpu_utilization=2.0, throughput=0.5)
        scores = o.score_all(req, sample_workload, sample_nodes[0], tenant_weights)
        util_score = next(s for s in scores if s.objective == "gpu_utilization")
        assert util_score.weight == 2.0
        assert util_score.weighted_score == util_score.score * 2.0


# ── Optimizer Tests ──────────────────────────────────────────

class TestOptimizer:
    def test_optimize_best_node_selected(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        opt = Optimizer()
        req = OptimizationRequest(workloads=[sample_workload], candidates=sample_nodes)
        result = opt.optimize(req)
        assert result.feasible_count >= 1
        assert result.best_candidate is not None
        assert result.best_candidate.feasible

    def test_infeasible_detected(self, sample_nodes: list[NodeCandidate]):
        opt = Optimizer()
        wl = WorkloadSpec(id="wl-bad", gpu_count=64, memory_per_gpu_gb=500, requires_nvlink=False)
        req = OptimizationRequest(workloads=[wl], candidates=sample_nodes)
        result = opt.optimize(req)
        assert result.infeasible_count > 0

    def test_best_utility_is_highest(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        opt = Optimizer()
        req = OptimizationRequest(workloads=[sample_workload], candidates=sample_nodes)
        result = opt.optimize(req)
        feasible = [c for c in result.candidates if c.feasible]
        if len(feasible) >= 2:
            assert feasible[0].total_utility >= feasible[1].total_utility

    def test_evaluate_workload(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        opt = Optimizer()
        result = opt.evaluate_workload(sample_workload, sample_nodes)
        assert result.feasible_count >= 1

    def test_set_global_weights(self):
        opt = Optimizer()
        weights = ObjectiveWeight(gpu_utilization=3.0, throughput=0.0)
        opt.set_global_weights(weights)
        assert opt._weights["gpu_utilization"] == 3.0
        assert opt._weights["throughput"] == 0.0

    def test_tenant_profile_weights(self, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        opt = Optimizer()
        profile = TenantObjectiveProfile(
            tenant_id="team-a", gpu_quota=32,
            weights=ObjectiveWeight(gpu_utilization=0.5, power_efficiency=3.0),
        )
        req = OptimizationRequest(
            workloads=[sample_workload], candidates=sample_nodes,
            tenant_profiles={"team-a": profile},
        )
        result = opt.optimize(req)
        assert result.feasible_count >= 1

    def test_default_weights(self):
        from gpuopt.optimizer.models import DEFAULT_OBJECTIVE_WEIGHTS
        assert len(DEFAULT_OBJECTIVE_WEIGHTS) == 11
        total = sum(DEFAULT_OBJECTIVE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01


# ── API Tests ────────────────────────────────────────────────

class TestOptimizerAPI:
    def test_optimize_endpoint(self, client: TestClient, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        payload = OptimizationRequest(workloads=[sample_workload], candidates=sample_nodes)
        resp = client.post("/api/v1/optimizer/optimize", json=payload.model_dump(mode="json"))
        assert resp.status_code == 200
        data = resp.json()
        assert data["feasible_count"] >= 1
        assert data["best_candidate"] is not None

    def test_evaluate_endpoint(self, client: TestClient, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        resp = client.post(
            "/api/v1/optimizer/evaluate",
            params={},
            json={
                "workload": sample_workload.model_dump(mode="json"),
                "nodes": [n.model_dump(mode="json") for n in sample_nodes],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["feasible_count"] >= 1

    def test_default_weights_endpoint(self, client: TestClient):
        resp = client.get("/api/v1/optimizer/default-weights")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 11

    def test_weigh_endpoint(self, client: TestClient):
        resp = client.post("/api/v1/optimizer/weigh", json={"gpu_utilization": 2.0, "fairness": 0.5})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_check_constraints_endpoint(self, client: TestClient, sample_workload: WorkloadSpec, sample_nodes: list[NodeCandidate]):
        resp = client.post(
            "/api/v1/optimizer/check-constraints",
            params={},
            json={
                "workload": sample_workload.model_dump(mode="json"),
                "node": sample_nodes[0].model_dump(mode="json"),
            },
        )
        assert resp.status_code == 200
        constraints = resp.json()
        assert len(constraints) == 12
        assert all(c["passed"] for c in constraints)
