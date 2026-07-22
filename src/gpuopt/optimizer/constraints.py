from __future__ import annotations

from .models import (
    ConstraintResult,
    HardConstraint,
    NodeCandidate,
    OptimizationRequest,
    TenantObjectiveProfile,
    WorkloadSpec,
)


class ConstraintEngine:
    def evaluate(
        self,
        request: OptimizationRequest,
        workload: WorkloadSpec,
        node: NodeCandidate,
        tenant_profile: TenantObjectiveProfile | None = None,
    ) -> list[ConstraintResult]:
        results: list[ConstraintResult] = []
        results.append(self._check_gpu_memory(workload, node))
        results.append(self._check_gpu_topology(workload, node))
        results.append(self._check_gpu_compatibility(workload, node))
        results.append(self._check_tenant_quota(workload, node, tenant_profile))
        results.append(self._check_tenant_priority(workload, tenant_profile))
        results.append(self._check_inference_slo_latency(workload, tenant_profile))
        results.append(self._check_inference_slo_error(workload, tenant_profile))
        results.append(self._check_data_locality(workload, node))
        results.append(self._check_approved_zones(workload, node, tenant_profile))
        results.append(self._check_checkpoint_policy(workload))
        results.append(self._check_preemption_policy(workload))
        results.append(self._check_action_blast_radius(workload, node, request))
        return results

    def all_feasible(self, results: list[ConstraintResult]) -> bool:
        return all(r.passed for r in results)

    def failing(self, results: list[ConstraintResult]) -> list[ConstraintResult]:
        return [r for r in results if not r.passed]

    # ── Individual checks ─────────────────────────────────────

    def _check_gpu_memory(self, wl: WorkloadSpec, node: NodeCandidate) -> ConstraintResult:
        needed = wl.memory_per_gpu_gb
        available = node.gpu_memory_per_gpu_gb
        if needed <= 0:
            return ConstraintResult(constraint=HardConstraint.GPU_MEMORY, passed=True)
        if available <= 0:
            return ConstraintResult(
                constraint=HardConstraint.GPU_MEMORY, passed=False,
                reason="Node has no GPU memory info",
            )
        passed = needed <= available * 1.02
        return ConstraintResult(
            constraint=HardConstraint.GPU_MEMORY, passed=passed,
            reason=f"Need {needed} GB, have {available} GB (95% threshold)"
            if not passed else "",
            detail={"needed_gb": needed, "available_gb": available},
        )

    def _check_gpu_topology(self, wl: WorkloadSpec, node: NodeCandidate) -> ConstraintResult:
        if not wl.requires_nvlink:
            return ConstraintResult(constraint=HardConstraint.GPU_TOPOLOGY, passed=True)
        passed = node.has_nvlink
        return ConstraintResult(
            constraint=HardConstraint.GPU_TOPOLOGY, passed=passed,
            reason="Workload requires NVLink but node does not support it" if not passed else "",
            detail={"requires_nvlink": True, "node_has_nvlink": node.has_nvlink},
        )

    def _check_gpu_compatibility(self, wl: WorkloadSpec, node: NodeCandidate) -> ConstraintResult:
        if not wl.gpu_model:
            return ConstraintResult(constraint=HardConstraint.GPU_COMPATIBILITY, passed=True)
        passed = wl.gpu_model.lower() in node.gpu_model.lower()
        return ConstraintResult(
            constraint=HardConstraint.GPU_COMPATIBILITY, passed=passed,
            reason=f"Workload requires '{wl.gpu_model}', node has '{node.gpu_model}'" if not passed else "",
            detail={"required": wl.gpu_model, "actual": node.gpu_model},
        )

    def _check_tenant_quota(
        self, wl: WorkloadSpec, node: NodeCandidate, profile: TenantObjectiveProfile | None,
    ) -> ConstraintResult:
        if profile is None or profile.gpu_quota <= 0:
            return ConstraintResult(constraint=HardConstraint.TENANT_QUOTA, passed=True)
        quota_remaining = profile.gpu_quota - node.running_jobs
        passed = wl.gpu_count <= quota_remaining or quota_remaining < 0
        return ConstraintResult(
            constraint=HardConstraint.TENANT_QUOTA, passed=passed,
            reason=f"Tenant quota {profile.gpu_quota} would be exceeded by {wl.gpu_count} GPUs"
            if not passed else "",
            detail={"quota": profile.gpu_quota, "requested": wl.gpu_count, "running": node.running_jobs},
        )

    def _check_tenant_priority(self, wl: WorkloadSpec, profile: TenantObjectiveProfile | None) -> ConstraintResult:
        if profile is None:
            return ConstraintResult(constraint=HardConstraint.TENANT_PRIORITY, passed=True)
        if wl.priority >= profile.priority_class:
            return ConstraintResult(constraint=HardConstraint.TENANT_PRIORITY, passed=True)
        return ConstraintResult(
            constraint=HardConstraint.TENANT_PRIORITY, passed=False,
            reason=f"Workload priority {wl.priority} below tenant floor {profile.priority_class}",
            detail={"workload_priority": wl.priority, "tenant_floor": profile.priority_class},
        )

    def _check_inference_slo_latency(
        self, wl: WorkloadSpec, profile: TenantObjectiveProfile | None,
    ) -> ConstraintResult:
        if not wl.inference_deployment:
            return ConstraintResult(constraint=HardConstraint.INFERENCE_SLO_LATENCY, passed=True)
        if profile is None or profile.slo_max_latency_ms is None:
            return ConstraintResult(constraint=HardConstraint.INFERENCE_SLO_LATENCY, passed=True)
        estimated_latency = self._estimate_inference_latency(wl)
        passed = estimated_latency <= profile.slo_max_latency_ms
        return ConstraintResult(
            constraint=HardConstraint.INFERENCE_SLO_LATENCY, passed=passed,
            reason=f"Estimated latency {estimated_latency}ms exceeds SLO {profile.slo_max_latency_ms}ms"
            if not passed else "",
            detail={"estimated_ms": estimated_latency, "slo_ms": profile.slo_max_latency_ms},
        )

    def _check_inference_slo_error(
        self, wl: WorkloadSpec, profile: TenantObjectiveProfile | None,
    ) -> ConstraintResult:
        if not wl.inference_deployment:
            return ConstraintResult(constraint=HardConstraint.INFERENCE_SLO_ERROR, passed=True)
        if profile is None or profile.slo_max_error_rate_pct is None:
            return ConstraintResult(constraint=HardConstraint.INFERENCE_SLO_ERROR, passed=True)
        return ConstraintResult(
            constraint=HardConstraint.INFERENCE_SLO_ERROR, passed=True,
            detail={"slo_max_error_pct": profile.slo_max_error_rate_pct},
        )

    def _check_data_locality(self, wl: WorkloadSpec, node: NodeCandidate) -> ConstraintResult:
        if not wl.data_location:
            return ConstraintResult(constraint=HardConstraint.DATA_LOCALITY, passed=True)
        zone = node.zone
        locality_match = wl.data_location in zone
        return ConstraintResult(
            constraint=HardConstraint.DATA_LOCALITY, passed=locality_match,
            reason=f"Data in '{wl.data_location}' not local to zone '{zone}'" if not locality_match else "",
            detail={"data_location": wl.data_location, "node_zone": zone},
        )

    def _check_approved_zones(
        self, wl: WorkloadSpec, node: NodeCandidate, profile: TenantObjectiveProfile | None,
    ) -> ConstraintResult:
        approved = wl.approved_zones or (profile.approved_zones if profile else [])
        if not approved:
            return ConstraintResult(constraint=HardConstraint.APPROVED_ZONES, passed=True)
        passed = node.zone in approved
        return ConstraintResult(
            constraint=HardConstraint.APPROVED_ZONES, passed=passed,
            reason=f"Zone '{node.zone}' not in approved zones {approved}" if not passed else "",
            detail={"node_zone": node.zone, "approved_zones": approved},
        )

    def _check_checkpoint_policy(self, wl: WorkloadSpec) -> ConstraintResult:
        if wl.checkpoint_interval_minutes <= 0:
            return ConstraintResult(constraint=HardConstraint.CHECKPOINT_POLICY, passed=True)
        if wl.estimated_runtime_minutes <= 0:
            return ConstraintResult(constraint=HardConstraint.CHECKPOINT_POLICY, passed=True)
        if wl.checkpoint_interval_minutes * 2 <= wl.estimated_runtime_minutes:
            return ConstraintResult(constraint=HardConstraint.CHECKPOINT_POLICY, passed=True)
        return ConstraintResult(
            constraint=HardConstraint.CHECKPOINT_POLICY, passed=False,
            reason=f"Checkpoint interval {wl.checkpoint_interval_minutes}m exceeds runtime {wl.estimated_runtime_minutes}m",
            detail={"interval_min": wl.checkpoint_interval_minutes, "runtime_min": wl.estimated_runtime_minutes},
        )

    def _check_preemption_policy(self, wl: WorkloadSpec) -> ConstraintResult:
        if wl.preemptible and wl.priority > 0:
            return ConstraintResult(constraint=HardConstraint.PREEMPTION_POLICY, passed=True)
        if not wl.preemptible:
            return ConstraintResult(constraint=HardConstraint.PREEMPTION_POLICY, passed=True)
        return ConstraintResult(
            constraint=HardConstraint.PREEMPTION_POLICY, passed=False,
            reason="Preemptible workload has no priority",
            detail={"preemptible": True, "priority": wl.priority},
        )

    def _check_action_blast_radius(
        self, wl: WorkloadSpec, node: NodeCandidate, request: OptimizationRequest,
    ) -> ConstraintResult:
        max_gpus = wl.gpu_count
        if max_gpus <= 0:
            return ConstraintResult(constraint=HardConstraint.ACTION_BLAST_RADIUS, passed=True)
        if node.free_gpus >= max_gpus:
            return ConstraintResult(constraint=HardConstraint.ACTION_BLAST_RADIUS, passed=True)
        return ConstraintResult(
            constraint=HardConstraint.ACTION_BLAST_RADIUS, passed=False,
            reason=f"Request {max_gpus} GPUs but only {node.free_gpus} free on node",
            detail={"requested": max_gpus, "free": node.free_gpus, "node_id": node.node_id},
        )

    @staticmethod
    def _estimate_inference_latency(wl: WorkloadSpec) -> float:
        base_latency = 50.0
        memory_factor = max(wl.memory_per_gpu_gb / 80.0, 1.0) if wl.memory_per_gpu_gb > 0 else 1.0
        return round(base_latency * memory_factor, 1)
