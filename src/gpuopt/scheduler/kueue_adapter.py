from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .rl_scheduler import Job, Node, RLScheduler
from .resource_flavors import ResourceFlavorManager, ResourceFlavorTier, get_flavor_manager
from .fairness import DominantResourceFairness, ProportionalFairnessScheduler

logger = logging.getLogger(__name__)


class KueueAdapter:
    def __init__(self, k8s_client: Any, rl_scheduler: RLScheduler | None = None,
                 flavor_manager: ResourceFlavorManager | None = None) -> None:
        self._client = k8s_client
        self._rl = rl_scheduler or RLScheduler()
        self._flavor_manager = flavor_manager or get_flavor_manager()
        self._drf = DominantResourceFairness()
        self._ps = ProportionalFairnessScheduler()
        self._detected = False
        self._version = ""

    def detect(self) -> dict[str, Any]:
        try:
            crds = self._client._custom.list_cluster_custom_object(
                "apiextensions.k8s.io", "v1", "customresourcedefinitions",
            )
            for crd in crds.get("items", []):
                name = crd.get("metadata", {}).get("name", "")
                if "clusterqueues.kueue.x-k8s.io" in name:
                    self._detected = True
                    self._version = self._get_kueue_version()
                    return {"detected": True, "version": self._version}
            self._detected = False
            return {"detected": False, "version": ""}
        except Exception as exc:
            logger.debug("Kueue CRD detection failed: %s", exc)
            self._detected = False
            return {"detected": False, "error": str(exc)}

    def _get_kueue_version(self) -> str:
        try:
            queues = self._client._custom.list_cluster_custom_object(
                "kueue.x-k8s.io", "v1beta1", "clusterqueues",
            )
            for q in queues.get("items", []):
                return q.get("apiVersion", "v1beta1")
        except Exception:
            pass
        return "v1beta1"

    def discover_flavors_from_queues(self) -> list[dict[str, Any]]:
        discovered: list[dict[str, Any]] = []
        if not self._detected:
            return discovered
        try:
            queues = self._client._custom.list_cluster_custom_object(
                "kueue.x-k8s.io", self._version, "clusterqueues",
            )
            for q in queues.get("items", []):
                spec = q.get("spec", {})
                for flavor_spec in spec.get("flavors", []):
                    name = flavor_spec.get("name", "")
                    resources = {}
                    for r in flavor_spec.get("resources", []):
                        resources[r.get("name", "")] = r.get("nominalQuota", "")
                    node_labels = flavor_spec.get("nodeLabels", {})
                    tier = self._infer_tier(name, node_labels)
                    self._flavor_manager.create_flavor(
                        name=name,
                        node_labels=node_labels,
                        resources=resources,
                        tier=tier,
                    )
                    discovered.append({
                        "name": name, "resources": resources,
                        "node_labels": node_labels, "tier": tier.value,
                    })
        except Exception as exc:
            logger.error("Flavor discovery failed: %s", exc)
        return discovered

    def list_flavors(self) -> list[dict[str, Any]]:
        return [
            {
                "name": f.name,
                "tier": f.tier.value,
                "priority": f.priority,
                "node_labels": f.node_labels,
                "resources": f.resources,
                "active": f.active,
            }
            for f in self._flavor_manager.list_flavors()
        ]

    def _infer_tier(self, name: str, labels: dict[str, str]) -> ResourceFlavorTier:
        name_lower = name.lower()
        if "spot" in name_lower or "preempt" in name_lower:
            return ResourceFlavorTier.SPOT
        if "reserved" in name_lower or "reservation" in name_lower:
            return ResourceFlavorTier.RESERVED
        if "premium" in name_lower or "high" in name_lower or "a100" in name_lower or "h100" in name_lower:
            return ResourceFlavorTier.PREMIUM
        return ResourceFlavorTier.STANDARD

    def compute_fairness(self, tenants: dict[str, dict[str, Any]],
                         total_gpus: int) -> dict[str, Any]:
        result = self._drf.compute(tenants, total_gpus)
        return {
            "allocations": [
                {
                    "tenant_id": a.tenant_id,
                    "fair_share": a.fair_share,
                    "gpus_allocated": a.gpus_allocated,
                    "gpu_quota": a.gpu_quota,
                    "dominant_share": a.dominant_share,
                    "usage_ratio": a.usage_ratio,
                    "fair_share_ratio": a.fair_share_ratio,
                    "preemptible": a.preemptible,
                    "adjustment": a.adjustment,
                }
                for a in result.allocations
            ],
            "total_gpus": result.total_gpus,
            "total_allocated": result.total_allocated,
            "dominant_share_threshold": result.dominant_share_threshold,
            "over_allocated": result.over_allocated,
            "under_allocated": result.under_allocated,
            "rebalance_actions": self._drf.suggest_rebalance(result),
        }

    def list_cluster_queues(self) -> list[dict[str, Any]]:
        if not self._detected:
            return self._mock_queues()
        try:
            queues = self._client._custom.list_cluster_custom_object(
                "kueue.x-k8s.io", self._version, "clusterqueues",
            )
            results = []
            for q in queues.get("items", []):
                meta = q.get("metadata", {})
                spec = q.get("spec", {})
                status = q.get("status", {})
                flavor_names = [f.get("name") for f in spec.get("flavors", [])]
                flavor_details = [
                    self._flavor_manager.get_flavor(fn)
                    for fn in flavor_names
                ]
                fair_sharing = spec.get("fairSharing", {})
                results.append({
                    "name": meta.get("name", ""),
                    "namespace": meta.get("namespace", ""),
                    "cohort": spec.get("cohort", ""),
                    "resourceGroups": spec.get("resourceGroups", []),
                    "fairSharing": fair_sharing,
                    "flavors": flavor_names,
                    "flavor_details": [
                        {
                            "name": f.name, "tier": f.tier.value,
                            "resources": f.resources,
                        }
                        for f in flavor_details if f
                    ],
                    "status": {
                        "admitted": status.get("admittedWorkloads", 0),
                        "pending": status.get("pendingWorkloads", 0),
                        "reserving": status.get("reservingWorkloads", 0),
                    },
                })
            return results
        except Exception as exc:
            logger.error("list_cluster_queues failed: %s", exc)
            return self._mock_queues()

    def submit_workload(self, job: Job, cluster_queue: str, namespace: str = "default",
                        priority_class: str = "", nodes: list[Node] | None = None,
                        flavor_name: str = "") -> dict[str, Any]:
        if nodes and not self._rl.q_table:
            self._rl.train_from_history(50)

        selected_flavor = None
        if flavor_name:
            selected_flavor = self._flavor_manager.get_flavor(flavor_name)
        if selected_flavor is None and self._flavor_manager.list_flavors():
            selected_flavor = self._flavor_manager.select_flavor(job.required_gpus)

        if nodes:
            result = self._rl.schedule(job, nodes)
            node_label = result.node_id if result.node else ""
            q_val = result.q_value
        else:
            node_label = ""
            q_val = 0.0

        workload_name = f"gpuopt-{job.id[:8] if job.id else str(uuid4())[:8]}"
        labels = {
            "app.kubernetes.io/managed-by": "gpuopt",
            "gpuopt.ai/job-id": job.id,
            "gpuopt.ai/rl-q-value": f"{q_val:.4f}",
        }
        if priority_class:
            labels["kueue.x-k8s.io/priority-class"] = priority_class
        if node_label:
            labels["gpuopt.ai/placement-node"] = node_label
        if selected_flavor:
            labels["kueue.x-k8s.io/flavor"] = selected_flavor.name

        annotations = {
            "gpuopt.ai/created-at": datetime.now(timezone.utc).isoformat(),
            "gpuopt.ai/required-gpus": str(job.required_gpus),
            "gpuopt.ai/priority": str(job.priority),
            "gpuopt.ai/rl-reward": f"{q_val:.4f}",
        }
        if selected_flavor:
            annotations["gpuopt.ai/flavor"] = selected_flavor.name
            annotations["gpuopt.ai/flavor-tier"] = selected_flavor.tier.value

        pod_set_labels = {}
        if selected_flavor:
            pod_set_labels.update(selected_flavor.node_labels)

        body = {
            "apiVersion": "kueue.x-k8s.io/v1beta1",
            "kind": "Workload",
            "metadata": {
                "name": workload_name,
                "namespace": namespace,
                "labels": labels,
                "annotations": annotations,
            },
            "spec": {
                "queueName": cluster_queue,
                "priority": job.priority,
                "priorityClassName": priority_class or "",
                "podSets": [
                    {
                        "name": "gpu-workers",
                        "count": 1,
                        "spec": {
                            "containers": [
                                {
                                    "name": "worker",
                                    "image": "gpuopt/worker:latest",
                                    "resources": {
                                        "requests": {
                                            "nvidia.com/gpu": job.required_gpus,
                                            "memory": f"{int(job.memory_gb * 1024)}Mi",
                                        },
                                        "limits": {
                                            "nvidia.com/gpu": job.required_gpus,
                                        },
                                    },
                                },
                            ],
                        },
                    },
                ],
            },
        }

        if selected_flavor:
            body["spec"]["podSets"][0]["nodeSelector"] = selected_flavor.node_labels

        if self._client._ensure_client():
            try:
                self._client._custom.create_namespaced_custom_object(
                    "kueue.x-k8s.io", "v1beta1", namespace, "workloads", body,
                )
                logger.info("Kueue Workload %s submitted to queue %s (flavor=%s)",
                            workload_name, cluster_queue,
                            selected_flavor.name if selected_flavor else "default")
                return {
                    "status": "submitted",
                    "workload_name": workload_name,
                    "namespace": namespace,
                    "cluster_queue": cluster_queue,
                    "flavor": selected_flavor.name if selected_flavor else "default",
                    "flavor_tier": selected_flavor.tier.value if selected_flavor else "",
                    "rl_placement": node_label or "pending",
                    "rl_q_value": q_val,
                }
            except Exception as exc:
                logger.error("Kueue Workload submission failed: %s", exc)
                return {"status": "error", "error": str(exc), "workload_name": workload_name}
        return {"status": "dry_run", "workload": body, "workload_name": workload_name}

    def list_workloads(self, namespace: str = "") -> list[dict[str, Any]]:
        if not self._detected:
            return self._mock_workloads()
        try:
            if namespace:
                items = self._client._custom.list_namespaced_custom_object(
                    "kueue.x-k8s.io", self._version, namespace, "workloads",
                )
            else:
                items = self._client._custom.list_cluster_custom_object(
                    "kueue.x-k8s.io", self._version, "workloads",
                )
            results = []
            for w in items.get("items", []):
                meta = w.get("metadata", {})
                spec = w.get("spec", {})
                status = w.get("status", {})
                annotations = meta.get("annotations", {})
                results.append({
                    "name": meta.get("name", ""),
                    "namespace": meta.get("namespace", ""),
                    "queue": spec.get("queueName", ""),
                    "priority": spec.get("priority", 0),
                    "gpus": self._extract_gpus(spec),
                    "phase": self._determine_phase(status),
                    "conditions": status.get("conditions", []),
                    "flavor": annotations.get("gpuopt.ai/flavor", ""),
                    "flavor_tier": annotations.get("gpuopt.ai/flavor-tier", ""),
                })
            return results
        except Exception as exc:
            logger.error("list_workloads failed: %s", exc)
            return self._mock_workloads()

    def _extract_gpus(self, spec: dict) -> int:
        for pset in spec.get("podSets", []):
            for ctr in pset.get("spec", {}).get("containers", []):
                limits = ctr.get("resources", {}).get("limits", {})
                gpus = limits.get("nvidia.com/gpu", 0)
                if gpus:
                    return int(gpus)
        return 0

    def _determine_phase(self, status: dict) -> str:
        conds = status.get("conditions", [])
        for c in conds:
            if c.get("type") == "Admitted" and c.get("status") == "True":
                return "admitted"
            if c.get("type") == "QuotaReserved" and c.get("status") == "True":
                return "reserved"
        if status.get("admissionCheckState"):
            return "pending"
        return "queued"

    def _mock_queues(self) -> list[dict[str, Any]]:
        return [
            {"name": "gpu-queue", "namespace": "default", "cohort": "gpu-cohort",
             "resourceGroups": [], "fairSharing": {},
             "flavors": ["default-flavor", "spot-flavor"],
             "flavor_details": [
                 {"name": "default-flavor", "tier": "standard", "resources": {"nvidia.com/gpu": "8"}},
                 {"name": "spot-flavor", "tier": "spot", "resources": {"nvidia.com/gpu": "4"}},
             ],
             "status": {"admitted": 5, "pending": 3, "reserving": 1}},
            {"name": "inference-queue", "namespace": "default", "cohort": "gpu-cohort",
             "resourceGroups": [], "fairSharing": {},
             "flavors": ["premium-flavor"],
             "flavor_details": [
                 {"name": "premium-flavor", "tier": "premium", "resources": {"nvidia.com/gpu": "4"}},
             ],
             "status": {"admitted": 2, "pending": 1, "reserving": 0}},
        ]

    def _mock_workloads(self) -> list[dict[str, Any]]:
        return [
            {"name": "mock-workload-1", "namespace": "default", "queue": "gpu-queue",
             "priority": 5, "gpus": 4, "phase": "admitted",
             "flavor": "default-flavor", "flavor_tier": "standard"},
            {"name": "mock-workload-2", "namespace": "default", "queue": "gpu-queue",
             "priority": 3, "gpus": 2, "phase": "queued",
             "flavor": "spot-flavor", "flavor_tier": "spot"},
        ]
