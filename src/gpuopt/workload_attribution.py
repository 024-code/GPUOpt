from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class ResourceOwner(str, Enum):
    TRAINING_JOB = "training_job"
    INFERENCE_ENDPOINT = "inference_endpoint"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class PodOwnership:
    pod_name: str
    namespace: str
    owner_kind: str
    owner_name: str
    owner_uid: str
    resource_owner: ResourceOwner
    tenant_id: str = ""
    project_id: str = ""
    user_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class GPUAllocationRecord:
    allocation_id: str
    gpu_uuid: str
    gpu_index: int
    node_name: str
    pod_name: str
    namespace: str
    tenant_id: str
    project_id: str
    owner_kind: str
    owner_name: str
    allocated_at: str
    released_at: str | None = None
    duration_hours: float = 0.0
    memory_allocated_gib: float = 0.0
    gpu_utilization_avg: float = 0.0


@dataclass
class TenantIsolationPolicy:
    tenant_id: str
    dedicated_nodes: list[str] = field(default_factory=list)
    dedicated_gpus: list[str] = field(default_factory=list)
    network_isolation: bool = False
    namespace_isolation: bool = True
    resource_quota: dict[str, float] = field(default_factory=dict)
    allowed_images: list[str] = field(default_factory=list)
    allowed_namespaces: list[str] = field(default_factory=list)


class WorkloadAttributionEngine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pod_ownership: dict[str, PodOwnership] = {}
        self._gpu_allocations: list[GPUAllocationRecord] = []
        self._tenant_policies: dict[str, TenantIsolationPolicy] = {}
        self._allocation_limit = 50000

    def register_pod(self, pod_name: str, namespace: str, owner_kind: str,
                     owner_name: str, owner_uid: str, labels: dict[str, str] | None = None,
                     tenant_id: str = "", project_id: str = "",
                     user_id: str = "") -> PodOwnership:
        res_owner = self._determine_resource_owner(owner_kind, labels or {})
        ownership = PodOwnership(
            pod_name=pod_name, namespace=namespace,
            owner_kind=owner_kind, owner_name=owner_name,
            owner_uid=owner_uid, resource_owner=res_owner,
            tenant_id=tenant_id, project_id=project_id,
            user_id=user_id, labels=labels or {},
        )
        with self._lock:
            self._pod_ownership[pod_name] = ownership
        return ownership

    def record_gpu_allocation(self, gpu_uuid: str, gpu_index: int, node_name: str,
                               pod_name: str, namespace: str, tenant_id: str,
                               project_id: str, owner_kind: str, owner_name: str,
                               memory_gib: float = 0.0) -> GPUAllocationRecord:
        record = GPUAllocationRecord(
            allocation_id=str(uuid4()),
            gpu_uuid=gpu_uuid, gpu_index=gpu_index, node_name=node_name,
            pod_name=pod_name, namespace=namespace,
            tenant_id=tenant_id, project_id=project_id,
            owner_kind=owner_kind, owner_name=owner_name,
            allocated_at=datetime.now(timezone.utc).isoformat(),
            memory_allocated_gib=memory_gib,
        )
        with self._lock:
            self._gpu_allocations.append(record)
            if len(self._gpu_allocations) > self._allocation_limit:
                self._gpu_allocations = self._gpu_allocations[-self._allocation_limit:]
        return record

    def release_gpu(self, allocation_id: str) -> bool:
        with self._lock:
            for rec in self._gpu_allocations:
                if rec.allocation_id == allocation_id and rec.released_at is None:
                    now = datetime.now(timezone.utc)
                    allocated = datetime.fromisoformat(rec.allocated_at)
                    rec.released_at = now.isoformat()
                    rec.duration_hours = (now - allocated).total_seconds() / 3600
                    return True
            return False

    def set_tenant_policy(self, policy: TenantIsolationPolicy) -> None:
        with self._lock:
            self._tenant_policies[policy.tenant_id] = policy

    def get_tenant_policy(self, tenant_id: str) -> TenantIsolationPolicy | None:
        with self._lock:
            return self._tenant_policies.get(tenant_id)

    def get_pod_ownership(self, pod_name: str) -> PodOwnership | None:
        with self._lock:
            return self._pod_ownership.get(pod_name)

    def list_pods_by_tenant(self, tenant_id: str) -> list[PodOwnership]:
        with self._lock:
            return [p for p in self._pod_ownership.values() if p.tenant_id == tenant_id]

    def list_pods_by_owner(self, owner_kind: str, owner_name: str) -> list[PodOwnership]:
        with self._lock:
            return [p for p in self._pod_ownership.values()
                    if p.owner_kind == owner_kind and p.owner_name == owner_name]

    def get_gpu_allocations(self, tenant_id: str | None = None,
                            pod_name: str | None = None,
                            owner_name: str | None = None) -> list[GPUAllocationRecord]:
        with self._lock:
            results = list(self._gpu_allocations)
            if tenant_id:
                results = [r for r in results if r.tenant_id == tenant_id]
            if pod_name:
                results = [r for r in results if r.pod_name == pod_name]
            if owner_name:
                results = [r for r in results if r.owner_name == owner_name]
            return results

    def get_attribution_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            allocs = self._gpu_allocations if not tenant_id else \
                [r for r in self._gpu_allocations if r.tenant_id == tenant_id]
            active = [r for r in allocs if r.released_at is None]
            total_hours = sum(r.duration_hours for r in allocs if r.duration_hours > 0)
            by_owner: dict[str, int] = {}
            by_owner_hours: dict[str, float] = {}
            for r in allocs:
                key = f"{r.owner_kind}:{r.owner_name}"
                by_owner[key] = by_owner.get(key, 0) + 1
                by_owner_hours[key] = by_owner_hours.get(key, 0.0) + r.duration_hours

            return {
                "total_allocations": len(allocs),
                "active_allocations": len(active),
                "total_gpu_hours": round(total_hours, 1),
                "by_owner": by_owner,
                "by_owner_hours": {k: round(v, 1) for k, v in by_owner_hours.items()},
                "tenant_id": tenant_id or "all",
            }

    def _determine_resource_owner(self, owner_kind: str,
                                   labels: dict[str, str]) -> ResourceOwner:
        if owner_kind in ("TrainingJob", "Job", "CronJob"):
            return ResourceOwner.TRAINING_JOB
        if owner_kind in ("InferenceEndpoint", "Deployment", "StatefulSet", "DaemonSet"):
            return ResourceOwner.INFERENCE_ENDPOINT
        if owner_kind in ("Node", "System"):
            return ResourceOwner.SYSTEM
        return ResourceOwner.UNKNOWN


_attribution_engine = WorkloadAttributionEngine()


def get_attribution_engine() -> WorkloadAttributionEngine:
    return _attribution_engine
