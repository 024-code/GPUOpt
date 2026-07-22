from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .rl_scheduler import Job, Node, RLScheduler

logger = logging.getLogger(__name__)


class KueueAdapter:
    def __init__(self, k8s_client: Any, rl_scheduler: RLScheduler | None = None) -> None:
        self._client = k8s_client
        self._rl = rl_scheduler or RLScheduler()
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

    def list_cluster_queues(self) -> list[dict[str, Any]]:
        if not self._detected:
            return []
        try:
            queues = self._client._custom.list_cluster_custom_object(
                "kueue.x-k8s.io", self._version, "clusterqueues",
            )
            results = []
            for q in queues.get("items", []):
                meta = q.get("metadata", {})
                spec = q.get("spec", {})
                status = q.get("status", {})
                results.append({
                    "name": meta.get("name", ""),
                    "namespace": meta.get("namespace", ""),
                    "cohort": spec.get("cohort", ""),
                    "resourceGroups": spec.get("resourceGroups", []),
                    "fairSharing": spec.get("fairSharing", {}),
                    "flavors": [f.get("name") for f in spec.get("flavors", [])],
                    "status": {
                        "admitted": status.get("admittedWorkloads", 0),
                        "pending": status.get("pendingWorkloads", 0),
                        "reserving": status.get("reservingWorkloads", 0),
                    },
                })
            return results
        except Exception as exc:
            logger.error("list_cluster_queues failed: %s", exc)
            return []

    def submit_workload(self, job: Job, cluster_queue: str, namespace: str = "default",
                        priority_class: str = "", nodes: list[Node] | None = None) -> dict[str, Any]:
        if nodes and not self._rl.q_table:
            self._rl.train_from_history(50)

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

        body = {
            "apiVersion": "kueue.x-k8s.io/v1beta1",
            "kind": "Workload",
            "metadata": {
                "name": workload_name,
                "namespace": namespace,
                "labels": labels,
                "annotations": {
                    "gpuopt.ai/created-at": datetime.now(timezone.utc).isoformat(),
                    "gpuopt.ai/required-gpus": str(job.required_gpus),
                    "gpuopt.ai/priority": str(job.priority),
                    "gpuopt.ai/rl-reward": f"{q_val:.4f}",
                },
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

        if self._client._ensure_client():
            try:
                self._client._custom.create_namespaced_custom_object(
                    "kueue.x-k8s.io", "v1beta1", namespace, "workloads", body,
                )
                logger.info("Kueue Workload %s submitted to queue %s", workload_name, cluster_queue)
                return {
                    "status": "submitted",
                    "workload_name": workload_name,
                    "namespace": namespace,
                    "cluster_queue": cluster_queue,
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
                results.append({
                    "name": meta.get("name", ""),
                    "namespace": meta.get("namespace", ""),
                    "queue": spec.get("queueName", ""),
                    "priority": spec.get("priority", 0),
                    "gpus": self._extract_gpus(spec),
                    "phase": self._determine_phase(status),
                    "conditions": status.get("conditions", []),
                })
            return results
        except Exception as exc:
            logger.error("list_workloads failed: %s", exc)
            return []

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

    def _mock_workloads(self) -> list[dict[str, Any]]:
        return [
            {"name": "mock-workload-1", "namespace": "default", "queue": "gpu-queue",
             "priority": 5, "gpus": 4, "phase": "admitted"},
            {"name": "mock-workload-2", "namespace": "default", "queue": "gpu-queue",
             "priority": 3, "gpus": 2, "phase": "queued"},
        ]
