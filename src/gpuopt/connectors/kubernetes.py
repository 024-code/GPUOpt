from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from gpuopt.schemas import (
    CheckItem,
    CheckStatus,
    ClusterTelemetry,
    GPUDeviceTelemetry,
    NodeTelemetry,
)

from .base import ClusterConnector


class KubernetesConnector(ClusterConnector):
    """Read-only Kubernetes environment connector.

    Import of the official Kubernetes Python client is deferred so mock-mode remains runnable
    without cluster dependencies installed.
    """

    def _load_k8s_client(self) -> tuple[Any, Any, Any, Any, Any] | None:
        """Load Kubernetes client configuration and return API clients."""
        try:
            from kubernetes import client, config
            from kubernetes.client import ApiException  # noqa: F401
        except ImportError:
            return None
        try:
            if self.cluster.in_cluster:
                config.load_incluster_config()
            else:
                config.load_kube_config(
                    config_file=self.cluster.kubeconfig_path,
                    context=self.cluster.kube_context,
                )
        except Exception:
            return None
        return (client.CoreV1Api(), client.VersionApi(), client.AppsV1Api(), client.ApiextensionsV1Api(), client.AuthorizationV1Api())

    def collect_telemetry(self) -> ClusterTelemetry:
        clients = self._load_k8s_client()
        if clients is None:
            return ClusterTelemetry(cluster_id=self.cluster.id, cluster_name=self.cluster.name)
        core = clients[0]
        collected_at = datetime.now(timezone.utc)
        telemetry_nodes: list[NodeTelemetry] = []
        gpu_count = 0

        try:
            node_list = core.list_node().items
        except Exception:
            return ClusterTelemetry(
                cluster_id=self.cluster.id,
                cluster_name=self.cluster.name,
                collected_at=collected_at,
            )

        for k8s_node in node_list:
            conditions = {c.type: c.status for c in (k8s_node.status.conditions or [])}
            status = "Ready" if conditions.get("Ready") == "True" else "NotReady"
            capacity = dict(k8s_node.status.capacity or {})
            allocatable = dict(k8s_node.status.allocatable or {})

            cpu_cap = int(capacity.get("cpu", 0)) * 1000 if capacity.get("cpu") else 0
            cpu_alloc = int(allocatable.get("cpu", 0)) * 1000 if allocatable.get("cpu") else 0
            mem_cap_str = capacity.get("memory", "0").replace("Ki", "").replace("Mi", "").replace("Gi", "")
            mem_alloc_str = allocatable.get("memory", "0").replace("Ki", "").replace("Mi", "").replace("Gi", "")
            mem_cap = int(mem_cap_str) if mem_cap_str.isdigit() else 0
            mem_alloc = int(mem_alloc_str) if mem_alloc_str.isdigit() else 0

            gpu_cap = int(capacity.get("nvidia.com/gpu", 0))
            pod_cap = int(capacity.get("pods", 110))

            gpu_devices: list[GPUDeviceTelemetry] = []
            labels = dict(k8s_node.metadata.labels or {})
            gpu_model = labels.get("nvidia.com/gpu.product", labels.get("gpuopt.ai/gpu-model", ""))
            for i in range(gpu_cap):
                gpu_devices.append(
                    GPUDeviceTelemetry(
                        index=i,
                        uuid=f"{k8s_node.metadata.name}/gpu-{i}",
                        model=gpu_model or "unknown",
                        memory_total_bytes=0,
                        memory_used_bytes=0,
                    )
                )
            gpu_count += gpu_cap

            pod_count = 0
            try:
                field_selector = f"spec.nodeName={k8s_node.metadata.name}"
                pods = core.list_pod_for_all_namespaces(field_selector=field_selector).items
                pod_count = len([p for p in pods if p.status.phase in ("Running", "Pending")])
            except Exception:
                pass

            telemetry_nodes.append(
                NodeTelemetry(
                    node_name=k8s_node.metadata.name,
                    status=status,
                    cpu_capacity_millicores=cpu_cap,
                    cpu_usage_millicores=0,
                    memory_capacity_bytes=mem_cap,
                    memory_usage_bytes=0,
                    pod_count=pod_count,
                    pod_capacity=pod_cap,
                    gpu_devices=gpu_devices,
                )
            )

        return ClusterTelemetry(
            cluster_id=self.cluster.id,
            cluster_name=self.cluster.name,
            collected_at=collected_at,
            node_count=len(telemetry_nodes),
            gpu_count=gpu_count,
            nodes=telemetry_nodes,
            freshness_seconds=0.0,
        )

    def run_checks(self) -> list[CheckItem]:
        try:
            from kubernetes import client, config
            from kubernetes.client import ApiException
        except ImportError:
            return [
                CheckItem(
                    name="kubernetes_client",
                    status=CheckStatus.FAIL,
                    message="The Kubernetes Python client is not installed.",
                    remediation="Install the project dependencies: pip install -e '.[dev]'",
                )
            ]

        try:
            if self.cluster.in_cluster:
                config.load_incluster_config()
            else:
                config.load_kube_config(
                    config_file=self.cluster.kubeconfig_path,
                    context=self.cluster.kube_context,
                )
        except Exception as exc:  # configuration exceptions vary by client version
            return [
                CheckItem(
                    name="kubernetes_config",
                    status=CheckStatus.FAIL,
                    message=f"Unable to load Kubernetes credentials: {exc}",
                    remediation="Verify kubeconfig path/context or in-cluster ServiceAccount mounting.",
                )
            ]

        core = client.CoreV1Api()
        version_api = client.VersionApi()
        apps = client.AppsV1Api()
        apiext = client.ApiextensionsV1Api()
        auth = client.AuthorizationV1Api()

        checks: list[CheckItem] = []
        checks.append(self._timed("api_server", lambda: self._check_api_server(version_api), ApiException))
        checks.append(self._timed("rbac_permissions", lambda: self._check_permissions(auth), ApiException))
        checks.append(self._timed("node_inventory", lambda: self._check_nodes(core), ApiException))
        checks.append(self._timed("gpu_inventory", lambda: self._check_gpus(core), ApiException))
        checks.append(
            self._timed(
                "gpu_operator",
                lambda: self._check_component_pods(
                    core,
                    namespaces=("gpu-operator", "nvidia-gpu-operator"),
                    label_selectors=("app.kubernetes.io/name=gpu-operator", "app=gpu-operator"),
                    optional=True,
                ),
                ApiException,
            )
        )
        checks.append(
            self._timed(
                "dcgm_exporter",
                lambda: self._check_component_pods(
                    core,
                    namespaces=("gpu-operator", "nvidia-gpu-operator", "monitoring"),
                    label_selectors=(
                        "app=nvidia-dcgm-exporter",
                        "app.kubernetes.io/name=dcgm-exporter",
                    ),
                    optional=False,
                ),
                ApiException,
            )
        )
        checks.append(self._timed("batch_scheduler", lambda: self._check_batch_scheduler(apiext), ApiException))
        checks.append(self._timed("prometheus", lambda: self._check_prometheus(core, apps), ApiException))
        return checks

    def _timed(
        self,
        name: str,
        operation: Callable[[], CheckItem],
        api_exception_type: type[Exception],
    ) -> CheckItem:
        started = time.perf_counter()
        try:
            result = operation()
        except api_exception_type as exc:
            result = CheckItem(
                name=name,
                status=CheckStatus.FAIL,
                message=f"Kubernetes API request failed: {exc}",
                remediation="Verify API reachability, credentials, RBAC and network policy.",
            )
        except Exception as exc:
            result = CheckItem(
                name=name,
                status=CheckStatus.FAIL,
                message=f"Unexpected check failure: {exc}",
                remediation="Review GPUOpt connector logs and cluster configuration.",
            )
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    @staticmethod
    def _check_api_server(version_api: Any) -> CheckItem:
        version = version_api.get_code()
        return CheckItem(
            name="api_server",
            status=CheckStatus.PASS,
            message=f"Kubernetes API server is reachable ({version.git_version}).",
            details={
                "git_version": version.git_version,
                "platform": version.platform,
                "major": version.major,
                "minor": version.minor,
            },
        )

    @staticmethod
    def _check_permissions(auth: Any) -> CheckItem:
        from kubernetes import client

        required = [
            ("list", "", "nodes"),
            ("list", "", "pods"),
            ("list", "apiextensions.k8s.io", "customresourcedefinitions"),
        ]
        results: dict[str, bool] = {}
        for verb, group, resource in required:
            review = client.V1SelfSubjectAccessReview(
                spec=client.V1SelfSubjectAccessReviewSpec(
                    resource_attributes=client.V1ResourceAttributes(
                        verb=verb,
                        group=group,
                        resource=resource,
                    )
                )
            )
            response = auth.create_self_subject_access_review(review)
            results[f"{verb}:{group or 'core'}:{resource}"] = bool(response.status.allowed)
        missing = [key for key, value in results.items() if not value]
        return CheckItem(
            name="rbac_permissions",
            status=CheckStatus.PASS if not missing else CheckStatus.FAIL,
            message="Required read-only permissions are available." if not missing else "Required RBAC permissions are missing.",
            details={"permissions": results, "missing": missing},
            remediation=None if not missing else "Apply infra/k8s/base/rbac.yaml or equivalent least-privilege RBAC.",
        )

    @staticmethod
    def _check_nodes(core: Any) -> CheckItem:
        nodes = core.list_node().items
        details = []
        ready_count = 0
        for node in nodes:
            conditions = {condition.type: condition.status for condition in node.status.conditions or []}
            ready = conditions.get("Ready") == "True"
            ready_count += int(ready)
            details.append(
                {
                    "name": node.metadata.name,
                    "ready": ready,
                    "capacity": dict(node.status.capacity or {}),
                    "allocatable": dict(node.status.allocatable or {}),
                    "labels": {
                        key: value
                        for key, value in (node.metadata.labels or {}).items()
                        if key.startswith(("nvidia.com/", "gpuopt.ai/", "node.kubernetes.io/instance-type"))
                    },
                }
            )
        if not nodes:
            return CheckItem(
                name="node_inventory",
                status=CheckStatus.FAIL,
                message="No Kubernetes nodes were returned.",
                remediation="Verify cluster registration and RBAC.",
            )
        status = CheckStatus.PASS if ready_count == len(nodes) else CheckStatus.WARN
        return CheckItem(
            name="node_inventory",
            status=status,
            message=f"Discovered {len(nodes)} node(s); {ready_count} Ready.",
            details={"nodes": details},
            remediation=None if status == CheckStatus.PASS else "Investigate NotReady nodes before enabling automation.",
        )

    def _check_gpus(self, core: Any) -> CheckItem:
        nodes = core.list_node().items
        gpu_total = 0
        mock_total = 0
        per_node: list[dict[str, Any]] = []
        for node in nodes:
            capacity = dict(node.status.capacity or {})
            allocatable = dict(node.status.allocatable or {})
            labels = node.metadata.labels or {}
            gpu_capacity = int(capacity.get("nvidia.com/gpu", 0))
            gpu_allocatable = int(allocatable.get("nvidia.com/gpu", 0))
            mock_count = int(labels.get("gpuopt.ai/mock-gpu-count", 0))
            gpu_total += gpu_capacity
            mock_total += mock_count
            per_node.append(
                {
                    "name": node.metadata.name,
                    "gpu_capacity": gpu_capacity,
                    "gpu_allocatable": gpu_allocatable,
                    "mock_gpu_count": mock_count,
                    "gpu_model": labels.get("nvidia.com/gpu.product") or labels.get("gpuopt.ai/gpu-model"),
                }
            )
        if gpu_total > 0:
            return CheckItem(
                name="gpu_inventory",
                status=CheckStatus.PASS,
                message=f"Discovered {gpu_total} allocatable NVIDIA GPU resource(s).",
                details={"real_gpu_count": gpu_total, "nodes": per_node},
            )
        if mock_total > 0 and self.cluster.options.get("allow_mock_gpu", True):
            return CheckItem(
                name="gpu_inventory",
                status=CheckStatus.WARN,
                message=f"No real nvidia.com/gpu resources; using {mock_total} mock GPU label(s).",
                details={"mock_gpu_count": mock_total, "nodes": per_node},
                remediation="Use this only in sandbox. Install NVIDIA GPU Operator/device plugin on real GPU nodes.",
            )
        return CheckItem(
            name="gpu_inventory",
            status=CheckStatus.FAIL,
            message="No NVIDIA GPU extended resources were detected.",
            details={"nodes": per_node},
            remediation="Install drivers and NVIDIA GPU Operator/device plugin, then verify nvidia.com/gpu capacity.",
        )

    @staticmethod
    def _check_component_pods(
        core: Any,
        namespaces: tuple[str, ...],
        label_selectors: tuple[str, ...],
        optional: bool,
    ) -> CheckItem:
        found: list[dict[str, Any]] = []
        for namespace in namespaces:
            for selector in label_selectors:
                try:
                    pods = core.list_namespaced_pod(namespace=namespace, label_selector=selector).items
                except Exception as exc:
                    if getattr(exc, "status", None) == 404:
                        continue
                    raise
                for pod in pods:
                    ready = all(status.ready for status in (pod.status.container_statuses or []))
                    found.append(
                        {
                            "namespace": namespace,
                            "name": pod.metadata.name,
                            "phase": pod.status.phase,
                            "ready": ready,
                        }
                    )
        component_name = "component"
        if any("dcgm" in selector for selector in label_selectors):
            component_name = "DCGM exporter"
        elif any("gpu-operator" in selector for selector in label_selectors):
            component_name = "NVIDIA GPU Operator"
        if not found:
            return CheckItem(
                name="dcgm_exporter" if "DCGM" in component_name else "gpu_operator",
                status=CheckStatus.WARN if optional else CheckStatus.FAIL,
                message=f"{component_name} pods were not detected.",
                remediation=f"Install or label {component_name} consistently so GPUOpt can discover it.",
            )
        not_ready = [pod for pod in found if not pod["ready"] or pod["phase"] != "Running"]
        return CheckItem(
            name="dcgm_exporter" if "DCGM" in component_name else "gpu_operator",
            status=CheckStatus.PASS if not not_ready else CheckStatus.WARN,
            message=f"Detected {len(found)} {component_name} pod(s).",
            details={"pods": found},
            remediation=None if not not_ready else f"Investigate non-ready {component_name} pods.",
        )

    @staticmethod
    def _check_batch_scheduler(apiext: Any) -> CheckItem:
        detected: list[str] = []
        known_crds = {
            "kueue": "clusterqueues.kueue.x-k8s.io",
            "volcano": "queues.scheduling.volcano.sh",
        }
        for name, crd in known_crds.items():
            try:
                apiext.read_custom_resource_definition(crd)
                detected.append(name)
            except Exception as exc:
                if getattr(exc, "status", None) != 404:
                    raise
        return CheckItem(
            name="batch_scheduler",
            status=CheckStatus.PASS if detected else CheckStatus.WARN,
            message=(
                f"Detected optional batch scheduling integration(s): {', '.join(detected)}."
                if detected
                else "No Kueue or Volcano CRDs detected."
            ),
            details={"detected": detected},
            remediation=None if detected else "Install Kueue or Volcano if queued/gang-scheduled AI workloads are required.",
        )

    @staticmethod
    def _check_prometheus(core: Any, apps: Any) -> CheckItem:
        candidates: list[dict[str, str]] = []
        for namespace in ("monitoring", "prometheus", "observability"):
            try:
                services = core.list_namespaced_service(namespace=namespace).items
            except Exception as exc:
                if getattr(exc, "status", None) == 404:
                    continue
                raise
            for service in services:
                name = service.metadata.name.lower()
                if "prometheus" in name:
                    candidates.append({"namespace": namespace, "service": service.metadata.name})
        return CheckItem(
            name="prometheus",
            status=CheckStatus.PASS if candidates else CheckStatus.WARN,
            message=(
                f"Detected {len(candidates)} Prometheus service candidate(s)."
                if candidates
                else "No Prometheus service was discovered in common namespaces."
            ),
            details={"services": candidates},
            remediation=None if candidates else "Install Prometheus Operator or configure an external Prometheus endpoint.",
        )
