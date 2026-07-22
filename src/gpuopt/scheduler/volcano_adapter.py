from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .rl_scheduler import Job, Node, RLScheduler

logger = logging.getLogger(__name__)


class VolcanoAdapter:
    def __init__(self, k8s_client: Any, rl_scheduler: RLScheduler | None = None) -> None:
        self._client = k8s_client
        self._rl = rl_scheduler or RLScheduler()
        self._detected = False
        self._version = "v1beta1"

    def detect(self) -> dict[str, Any]:
        try:
            crds = self._client._custom.list_cluster_custom_object(
                "apiextensions.k8s.io", "v1", "customresourcedefinitions",
            )
            for crd in crds.get("items", []):
                name = crd.get("metadata", {}).get("name", "")
                if "queues.scheduling.volcano.sh" in name:
                    self._detected = True
                    return {"detected": True, "version": self._version}
            self._detected = False
            return {"detected": False, "version": ""}
        except Exception as exc:
            logger.debug("Volcano CRD detection failed: %s", exc)
            self._detected = False
            return {"detected": False, "error": str(exc)}

    def list_queues(self) -> list[dict[str, Any]]:
        if not self._detected:
            return []
        try:
            queues = self._client._custom.list_cluster_custom_object(
                "scheduling.volcano.sh", self._version, "queues",
            )
            results = []
            for q in queues.get("items", []):
                meta = q.get("metadata", {})
                spec = q.get("spec", {})
                status = q.get("status", {})
                results.append({
                    "name": meta.get("name", ""),
                    "weight": spec.get("weight", 1),
                    "capabilities": spec.get("capabilities", {}),
                    "state": status.get("state", ""),
                    "running": status.get("running", 0),
                    "pending": status.get("pending", 0),
                })
            return results
        except Exception as exc:
            logger.error("list_volcano_queues failed: %s", exc)
            return []

    def submit_podgroup(self, job: Job, queue: str, namespace: str = "default",
                        priority: int = 5, nodes: list[Node] | None = None) -> dict[str, Any]:
        if nodes and not self._rl.q_table:
            self._rl.train_from_history(50)

        if nodes:
            result = self._rl.schedule(job, nodes)
            node_label = result.node_id if result.node else ""
        else:
            node_label = ""

        pg_name = f"gpuopt-{job.id[:8] if job.id else str(uuid4())[:8]}"
        labels = {
            "app.kubernetes.io/managed-by": "gpuopt",
            "gpuopt.ai/job-id": job.id,
        }
        if node_label:
            labels["gpuopt.ai/placement-node"] = node_label

        body = {
            "apiVersion": f"scheduling.volcano.sh/{self._version}",
            "kind": "PodGroup",
            "metadata": {
                "name": pg_name,
                "namespace": namespace,
                "labels": labels,
                "annotations": {
                    "gpuopt.ai/created-at": datetime.now(timezone.utc).isoformat(),
                    "gpuopt.ai/required-gpus": str(job.required_gpus),
                    "gpuopt.ai/priority": str(job.priority),
                },
            },
            "spec": {
                "queue": queue,
                "minMember": 1,
                "priority": priority,
                "minResources": {
                    "nvidia.com/gpu": job.required_gpus,
                    "memory": f"{int(job.memory_gb * 1024)}Mi",
                },
            },
        }

        if self._client._ensure_client():
            try:
                self._client._custom.create_namespaced_custom_object(
                    "scheduling.volcano.sh", self._version, namespace, "podgroups", body,
                )
                logger.info("Volcano PodGroup %s submitted to queue %s", pg_name, queue)
                return {
                    "status": "submitted",
                    "podgroup_name": pg_name,
                    "namespace": namespace,
                    "queue": queue,
                    "rl_placement": node_label or "pending",
                }
            except Exception as exc:
                logger.error("Volcano PodGroup submission failed: %s", exc)
                return {"status": "error", "error": str(exc), "podgroup_name": pg_name}
        return {"status": "dry_run", "podgroup": body, "podgroup_name": pg_name}

    def list_podgroups(self, namespace: str = "") -> list[dict[str, Any]]:
        if not self._detected:
            return []
        try:
            if namespace:
                items = self._client._custom.list_namespaced_custom_object(
                    "scheduling.volcano.sh", self._version, namespace, "podgroups",
                )
            else:
                items = self._client._custom.list_cluster_custom_object(
                    "scheduling.volcano.sh", self._version, "podgroups",
                )
            results = []
            for pg in items.get("items", []):
                meta = pg.get("metadata", {})
                spec = pg.get("spec", {})
                status = pg.get("status", {})
                results.append({
                    "name": meta.get("name", ""),
                    "namespace": meta.get("namespace", ""),
                    "queue": spec.get("queue", ""),
                    "minMember": spec.get("minMember", 0),
                    "phase": status.get("phase", ""),
                    "running": status.get("running", 0),
                    "failed": status.get("failed", 0),
                })
            return results
        except Exception as exc:
            logger.error("list_podgroups failed: %s", exc)
            return []
