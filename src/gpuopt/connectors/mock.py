from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from gpuopt.schemas import (
    CheckItem,
    CheckStatus,
    ClusterTelemetry,
    GPUDeviceTelemetry,
    NodeTelemetry,
)

from .base import ClusterConnector


class MockConnector(ClusterConnector):
    """Connector used for local development when no physical GPU is available."""

    def run_checks(self) -> list[CheckItem]:
        snapshot = self._load_snapshot()
        return [
            self._component_check("api_server", snapshot, required=True),
            self._node_inventory(snapshot),
            self._gpu_inventory(snapshot),
            self._component_check("gpu_operator", snapshot, required=False),
            self._component_check("dcgm_exporter", snapshot, required=True),
            self._component_check("prometheus", snapshot, required=True),
            self._batch_scheduler(snapshot),
            self._permissions(snapshot),
        ]

    def collect_telemetry(self) -> ClusterTelemetry:
        snapshot = self._load_snapshot()
        nodes_data = snapshot.get("nodes", [])
        telemetry_nodes: list[NodeTelemetry] = []
        for node in nodes_data:
            gpu_devices_raw = node.get("gpu_devices", [])
            gpu_devices = [
                GPUDeviceTelemetry(
                    index=gpu.get("index", i),
                    uuid=gpu.get("uuid", ""),
                    model=gpu.get("model", "mock-gpu"),
                    memory_total_bytes=gpu.get("memory_total", 0),
                    memory_used_bytes=gpu.get("memory_used", 0),
                    utilization_gpu_percent=gpu.get("utilization_gpu", 0.0),
                    utilization_memory_percent=gpu.get("utilization_memory", 0.0),
                    temperature_gpu_celsius=gpu.get("temperature_gpu", 0.0),
                    power_draw_watts=gpu.get("power_draw", 0.0),
                    power_limit_watts=gpu.get("power_limit", 0.0),
                    ecc_errors_volatile=gpu.get("ecc_errors_volatile", 0),
                    ecc_errors_aggregate=gpu.get("ecc_errors_aggregate", 0),
                    clock_sm_mhz=gpu.get("clock_sm", 0),
                    clock_mem_mhz=gpu.get("clock_mem", 0),
                )
                for i, gpu in enumerate(gpu_devices_raw)
            ]
            telemetry_nodes.append(
                NodeTelemetry(
                    node_name=node.get("name", "unknown"),
                    status="Ready" if node.get("ready", False) else "NotReady",
                    gpu_devices=gpu_devices,
                    pod_count=node.get("pods", 0),
                    pod_capacity=node.get("pod_capacity", 110),
                )
            )
        collected_at = datetime.now(timezone.utc)
        return ClusterTelemetry(
            cluster_id=self.cluster.id,
            cluster_name=self.cluster.name,
            collected_at=collected_at,
            node_count=len(telemetry_nodes),
            gpu_count=sum(len(n.gpu_devices) for n in telemetry_nodes),
            nodes=telemetry_nodes,
            freshness_seconds=0.0,
        )

    def _load_snapshot(self) -> dict[str, Any]:
        started = perf_counter()
        snapshot_path = self.cluster.options.get("snapshot_path")
        if not snapshot_path:
            return self._default_snapshot()
        path = Path(snapshot_path)
        if not path.exists():
            raise FileNotFoundError(f"Mock snapshot does not exist: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_load_latency_ms"] = round((perf_counter() - started) * 1000, 2)
        return data

    @staticmethod
    def _default_snapshot() -> dict[str, Any]:
        return {
            "api_server": {"ready": True, "version": "mock-v1"},
            "nodes": [{
                "name": "mock-gpu-worker",
                "ready": True,
                "gpu_count": 4,
                "pod_capacity": 110,
                "gpu_devices": [
                    {
                        "index": i,
                        "uuid": f"mock-gpu-{i:04d}",
                        "model": "RTX 4090",
                        "memory_total": 25769803776,
                        "memory_used": 0,
                        "utilization_gpu": 0.0,
                        "utilization_memory": 0.0,
                        "temperature_gpu": 35.0,
                        "power_draw": 30.0,
                        "power_limit": 450.0,
                        "ecc_errors_volatile": 0,
                        "ecc_errors_aggregate": 0,
                        "clock_sm": 210,
                        "clock_mem": 1013,
                    }
                    for i in range(4)
                ],
            }],
            "components": {
                "gpu_operator": {"ready": True},
                "dcgm_exporter": {"ready": True, "metrics_endpoint": "mock://dcgm"},
                "prometheus": {"ready": True},
                "kueue": {"ready": True},
            },
            "permissions": {"list_nodes": True, "list_pods": True, "read_crds": True},
        }

    @staticmethod
    def _component_check(name: str, snapshot: dict[str, Any], required: bool) -> CheckItem:
        source = snapshot.get(name) or snapshot.get("components", {}).get(name, {})
        ready = bool(source.get("ready"))
        if ready:
            return CheckItem(
                name=name,
                status=CheckStatus.PASS,
                message=f"{name.replace('_', ' ').title()} is available.",
                details=source,
            )
        return CheckItem(
            name=name,
            status=CheckStatus.FAIL if required else CheckStatus.WARN,
            message=f"{name.replace('_', ' ').title()} is not ready in the mock snapshot.",
            details=source,
            remediation=f"Set components.{name}.ready=true after installing or configuring it.",
        )

    @staticmethod
    def _node_inventory(snapshot: dict[str, Any]) -> CheckItem:
        nodes = snapshot.get("nodes", [])
        ready_nodes = [node for node in nodes if node.get("ready")]
        status = CheckStatus.PASS if nodes and len(ready_nodes) == len(nodes) else CheckStatus.WARN
        return CheckItem(
            name="node_inventory",
            status=status,
            message=f"Discovered {len(nodes)} node(s); {len(ready_nodes)} Ready.",
            details={"nodes": nodes},
            remediation=None if status == CheckStatus.PASS else "Investigate NotReady nodes.",
        )

    @staticmethod
    def _gpu_inventory(snapshot: dict[str, Any]) -> CheckItem:
        nodes = snapshot.get("nodes", [])
        gpu_count = sum(int(node.get("gpu_count", 0)) for node in nodes)
        status = CheckStatus.PASS if gpu_count > 0 else CheckStatus.WARN
        return CheckItem(
            name="gpu_inventory",
            status=status,
            message=f"Discovered {gpu_count} mock GPU(s).",
            details={"gpu_count": gpu_count},
            remediation=None if gpu_count else "Label mock nodes or connect a real GPU cluster.",
        )

    @staticmethod
    def _batch_scheduler(snapshot: dict[str, Any]) -> CheckItem:
        components = snapshot.get("components", {})
        ready = [name for name in ("kueue", "volcano") if components.get(name, {}).get("ready")]
        return CheckItem(
            name="batch_scheduler",
            status=CheckStatus.PASS if ready else CheckStatus.WARN,
            message=(
                f"Detected batch scheduler integration: {', '.join(ready)}."
                if ready
                else "No optional Kubernetes batch scheduler detected."
            ),
            details={"detected": ready},
            remediation=None if ready else "Install Kueue or Volcano for queued AI jobs.",
        )

    @staticmethod
    def _permissions(snapshot: dict[str, Any]) -> CheckItem:
        permissions = snapshot.get("permissions", {})
        missing = [name for name, allowed in permissions.items() if not allowed]
        return CheckItem(
            name="rbac_permissions",
            status=CheckStatus.PASS if not missing else CheckStatus.FAIL,
            message="Read-only RBAC permissions are sufficient." if not missing else "Missing RBAC permissions.",
            details={"permissions": permissions, "missing": missing},
            remediation=None if not missing else "Update the GPUOpt ClusterRole and binding.",
        )
