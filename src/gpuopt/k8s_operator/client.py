from __future__ import annotations

import logging
from typing import Any

from .models import GPUOptimizationAction, GPUOptimizationProfile, GPUWorkloadProfile

logger = logging.getLogger(__name__)


class K8sClientWrapper:
    def __init__(self, in_cluster: bool = False, kubeconfig: str | None = None) -> None:
        self._in_cluster = in_cluster
        self._kubeconfig = kubeconfig
        self._initialized = False
        self._core = None
        self._apps = None
        self._custom = None

    def _ensure_client(self) -> bool:
        if self._initialized:
            return True
        try:
            from kubernetes import client, config

            if self._in_cluster:
                config.load_incluster_config()
            else:
                config.load_kube_config(config_file=self._kubeconfig)
            self._core = client.CoreV1Api()
            self._apps = client.AppsV1Api()
            self._custom = client.CustomObjectsApi()
            self._initialized = True
            logger.info("K8s client initialized (in_cluster=%s)", self._in_cluster)
            return True
        except ImportError:
            logger.warning("Kubernetes Python client not installed; running in mock mode")
            return False
        except Exception as exc:
            logger.warning("Failed to init K8s client: %s", exc)
            return False

    def list_pods(self, namespace: str = "", label_selector: str = "") -> list[dict[str, Any]]:
        if not self._ensure_client():
            return []
        try:
            if namespace:
                pods = self._core.list_namespaced_pod(namespace, label_selector=label_selector)
            else:
                pods = self._core.list_pod_for_all_namespaces(label_selector=label_selector)
            results = []
            for p in pods.items:
                results.append({
                    "name": p.metadata.name,
                    "namespace": p.metadata.namespace,
                    "phase": p.status.phase if p.status else "Unknown",
                    "node": p.spec.node_name if p.spec else "",
                    "labels": dict(p.metadata.labels or {}),
                })
            return results
        except Exception as exc:
            logger.error("list_pods failed: %s", exc)
            return []

    def patch_deployment(self, namespace: str, name: str, body: dict[str, Any]) -> bool:
        if not self._ensure_client():
            return False
        try:
            self._apps.patch_namespaced_deployment(name, namespace, body)
            logger.info("Patched deployment %s/%s", namespace, name)
            return True
        except Exception as exc:
            logger.error("patch_deployment failed: %s", exc)
            return False

    def patch_statefulset(self, namespace: str, name: str, body: dict[str, Any]) -> bool:
        if not self._ensure_client():
            return False
        try:
            self._apps.patch_namespaced_stateful_set(name, namespace, body)
            logger.info("Patched statefulset %s/%s", namespace, name)
            return True
        except Exception as exc:
            logger.error("patch_statefulset failed: %s", exc)
            return False

    def list_nodes(self, label_selector: str = "") -> list[dict[str, Any]]:
        if not self._ensure_client():
            return []
        try:
            nodes = self._core.list_node(label_selector=label_selector)
            results = []
            for n in nodes.items:
                capacity = dict(n.status.capacity or {})
                labels = dict(n.metadata.labels or {})
                results.append({
                    "name": n.metadata.name,
                    "gpu_capacity": int(capacity.get("nvidia.com/gpu", 0)),
                    "gpu_model": labels.get("nvidia.com/gpu.product", ""),
                    "labels": labels,
                    "ready": any(c.type == "Ready" and c.status == "True" for c in (n.status.conditions or [])),
                })
            return results
        except Exception as exc:
            logger.error("list_nodes failed: %s", exc)
            return []

    def taint_node(self, node_name: str, key: str, value: str, effect: str) -> bool:
        if not self._ensure_client():
            return False
        try:
            body = {
                "spec": {
                    "taints": [{"key": key, "value": value, "effect": effect}]
                }
            }
            self._core.patch_node(node_name, body)
            logger.info("Tainted node %s with %s=%s:%s", node_name, key, value, effect)
            return True
        except Exception as exc:
            logger.error("taint_node failed: %s", exc)
            return False

    def label_node(self, node_name: str, labels: dict[str, str]) -> bool:
        if not self._ensure_client():
            return False
        try:
            body = {"metadata": {"labels": labels}}
            self._core.patch_node(node_name, body)
            logger.info("Labeled node %s with %s", node_name, labels)
            return True
        except Exception as exc:
            logger.error("label_node failed: %s", exc)
            return False

    def list_gpuoptimization_profiles(self, namespace: str = "") -> list[dict[str, Any]]:
        if not self._ensure_client():
            return []
        try:
            if namespace:
                items = self._custom.list_namespaced_custom_object(
                    "gpuopt.ai", "v1alpha1", namespace, "gpuoptimizationprofiles"
                )
            else:
                items = self._custom.list_cluster_custom_object(
                    "gpuopt.ai", "v1alpha1", "gpuoptimizationprofiles"
                )
            return items.get("items", [])
        except Exception as exc:
            logger.error("list_gpuoptimization_profiles failed: %s", exc)
            return []

    def list_gpuoptimization_actions(self, namespace: str = "") -> list[dict[str, Any]]:
        if not self._ensure_client():
            return []
        try:
            if namespace:
                items = self._custom.list_namespaced_custom_object(
                    "gpuopt.ai", "v1alpha1", namespace, "gpuoptimizationactions"
                )
            else:
                items = self._custom.list_cluster_custom_object(
                    "gpuopt.ai", "v1alpha1", "gpuoptimizationactions"
                )
            return items.get("items", [])
        except Exception as exc:
            logger.error("list_gpuoptimization_actions failed: %s", exc)
            return []

    def update_action_status(self, namespace: str, name: str, status: dict[str, Any]) -> bool:
        if not self._ensure_client():
            return False
        try:
            self._custom.patch_namespaced_custom_object_status(
                "gpuopt.ai", "v1alpha1", namespace, "gpuoptimizationactions", name,
                {"status": status},
            )
            return True
        except Exception as exc:
            logger.error("update_action_status failed: %s", exc)
            return False

    def update_profile_status(self, namespace: str, name: str, status: dict[str, Any]) -> bool:
        if not self._ensure_client():
            return False
        try:
            self._custom.patch_namespaced_custom_object_status(
                "gpuopt.ai", "v1alpha1", namespace, "gpuoptimizationprofiles", name,
                {"status": status},
            )
            return True
        except Exception as exc:
            logger.error("update_profile_status failed: %s", exc)
            return False
