from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from kubernetes import client as k8s_client

from gpuopt.k8s_operator.client import K8sClientWrapper

logger = logging.getLogger(__name__)


class ScaleDirection(str, enum.Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingPolicy(str, enum.Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    SCHEDULED = "scheduled"


@dataclass
class NodeGroupConfig:
    name: str = ""
    min_size: int = 0
    max_size: int = 10
    current_size: int = 1
    gpu_type: str = ""
    gpus_per_node: int = 8
    instance_type: str = ""
    region: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    taints: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ScaleEvent:
    direction: ScaleDirection = ScaleDirection.NONE
    node_group: str = ""
    delta: int = 0
    reason: str = ""
    initiated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str = ""
    status: str = "pending"
    target_size: int = 0


@dataclass
class AutoscalingConfig:
    policy: ScalingPolicy = ScalingPolicy.AUTOMATIC
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_increment: int = 1
    scale_down_decrement: int = 1
    cooldown_seconds: int = 300
    min_nodes: int = 1
    max_nodes: int = 20
    node_groups: list[NodeGroupConfig] = field(default_factory=list)


@dataclass
class AutoscalerStatus:
    running: bool = False
    current_config: AutoscalingConfig | None = None
    last_event: ScaleEvent | None = None
    event_count: int = 0


class ClusterAutoscalerClient:
    def __init__(self, k8s_client_wrapper: K8sClientWrapper | None = None) -> None:
        self._k8s = k8s_client_wrapper
        self._use_mock = k8s_client_wrapper is None

    def set_node_group_size(self, node_group: str, target_size: int) -> bool:
        if self._use_mock:
            logger.info("Mock: set node group %s to size %d", node_group, target_size)
            return True
        try:
            self._k8s._ensure_client()
            core = k8s_client.CoreV1Api()
            autoscaling_v1 = k8s_client.AutoscalingV1Api()
            nodegroup_label = f"node-group={node_group}"
            nodes = core.list_nodes(label_selector=nodegroup_label).items
            current = len(nodes)
            delta = target_size - current
            if delta > 0:
                self._create_dummy_nodes(core, node_group, delta, {})
            elif delta < 0:
                self._remove_nodes(core, nodes[:abs(delta)])
            return True
        except Exception as exc:
            logger.error("Failed to set node group %s size to %d: %s", node_group, target_size, exc)
            return False

    def _create_dummy_nodes(self, core: k8s_client.CoreV1Api, node_group: str, count: int, labels: dict[str, str]) -> None:
        for i in range(count):
            node_name = f"gpu-{node_group}-scale-{int(datetime.now().timestamp())}-{i}"
            body = {
                "apiVersion": "v1",
                "kind": "Node",
                "metadata": {
                    "name": node_name,
                    "labels": {"node-group": node_group, "gpu-node": "true", **labels},
                },
                "spec": {"unschedulable": False},
            }
            core.create_node(body)

    def _remove_nodes(self, core: k8s_client.CoreV1Api, nodes: list[Any]) -> None:
        for node in nodes:
            name = node.metadata.name if hasattr(node.metadata, "name") else node["metadata"]["name"]
            core.delete_node(name)

    def get_node_group_size(self, node_group: str) -> int:
        if self._use_mock:
            return 3
        try:
            self._k8s._ensure_client()
            core = k8s_client.CoreV1Api()
            nodes = core.list_nodes(label_selector=f"node-group={node_group}").items
            return len(nodes)
        except Exception:
            return 0

    def annotate_node_group(self, node_group: str, annotations: dict[str, str]) -> bool:
        if self._use_mock:
            return True
        try:
            self._k8s._ensure_client()
            core = k8s_client.CoreV1Api()
            nodes = core.list_nodes(label_selector=f"node-group={node_group}").items
            for node in nodes:
                name = node.metadata.name if hasattr(node.metadata, "name") else node["metadata"]["name"]
                existing = node.metadata.annotations or {} if hasattr(node.metadata, "annotations") else node.get("metadata", {}).get("annotations", {})
                merged = {**existing, **annotations}
                core.patch_node(name, {"metadata": {"annotations": merged}})
            return True
        except Exception as exc:
            logger.error("Failed to annotate node group %s: %s", node_group, exc)
            return False


class AutoscalerEngine:
    def __init__(
        self,
        k8s_client_wrapper: K8sClientWrapper | None = None,
        config: AutoscalingConfig | None = None,
    ) -> None:
        self._k8s = k8s_client_wrapper
        self._ca_client = ClusterAutoscalerClient(k8s_client_wrapper)
        self._config = config or AutoscalingConfig()
        self._events: list[ScaleEvent] = []
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_scale_time: float = 0.0

    @property
    def config(self) -> AutoscalingConfig:
        return self._config

    @config.setter
    def config(self, value: AutoscalingConfig) -> None:
        with self._lock:
            self._config = value

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="autoscaler-engine")
        self._thread.start()
        logger.info("Autoscaler engine started (policy=%s)", self._config.policy.value)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Autoscaler engine stopped")

    def _loop(self) -> None:
        while self._running:
            try:
                if self._config.policy == ScalingPolicy.AUTOMATIC:
                    self._autoscale_cycle()
            except Exception as exc:
                logger.error("Autoscaler cycle error: %s", exc)
            threading.Event().wait(60)

    def _autoscale_cycle(self) -> None:
        now = datetime.now().timestamp()
        if now - self._last_scale_time < self._config.cooldown_seconds:
            return
        utilization = self._compute_cluster_utilization()
        if utilization >= self._config.scale_up_threshold:
            self._scale_up(utilization)
        elif utilization <= self._config.scale_down_threshold and self._get_total_nodes() > self._config.min_nodes:
            self._scale_down(utilization)

    def _compute_cluster_utilization(self) -> float:
        if self._k8s is None:
            import random
            return random.uniform(20, 95)
        try:
            self._k8s._ensure_client()
            core = k8s_client.CoreV1Api()
            nodes = core.list_nodes(label_selector="gpu-node=true").items
            if not nodes:
                return 0.0
            pods = core.list_pod_for_all_namespaces(field_selector="spec.nodeName!=,").items
            allocatable_gpu = 0
            requested_gpu = 0
            for node in nodes:
                caps = node.status.capacity or {}
                gpu_count = int(caps.get("nvidia.com/gpu", 0))
                allocatable_gpu += gpu_count
            for pod in pods:
                for container in pod.spec.containers:
                    limits = container.resources.limits or {}
                    gpu_req = int(limits.get("nvidia.com/gpu", 0))
                    requested_gpu += gpu_req
            if allocatable_gpu == 0:
                return 0.0
            return (requested_gpu / allocatable_gpu) * 100.0
        except Exception as exc:
            logger.warning("Utilization compute failed: %s", exc)
            return 50.0

    def _get_total_nodes(self) -> int:
        if self._k8s is None:
            return len(self._config.node_groups) if self._config.node_groups else 3
        try:
            self._k8s._ensure_client()
            core = k8s_client.CoreV1Api()
            return len(core.list_nodes(label_selector="gpu-node=true").items)
        except Exception:
            return 0

    def _scale_up(self, utilization: float) -> ScaleEvent | None:
        total = self._get_total_nodes()
        if total >= self._config.max_nodes:
            return None
        delta = min(self._config.scale_up_increment, self._config.max_nodes - total)
        node_group = self._select_node_group()
        event = ScaleEvent(
            direction=ScaleDirection.UP,
            node_group=node_group,
            delta=delta,
            reason=f"Utilization {utilization:.1f}% > threshold {self._config.scale_up_threshold}%",
            target_size=total + delta,
            status="executed",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        success = self._ca_client.set_node_group_size(node_group, total + delta)
        if not success:
            event.status = "failed"
        with self._lock:
            self._events.append(event)
            self._last_scale_time = datetime.now().timestamp()
        logger.info("Scale up: %s (delta=%d, util=%.1f%%)", node_group, delta, utilization)
        return event

    def _scale_down(self, utilization: float) -> ScaleEvent | None:
        total = self._get_total_nodes()
        if total <= self._config.min_nodes:
            return None
        delta = min(self._config.scale_down_decrement, total - self._config.min_nodes)
        node_group = self._select_node_group()
        event = ScaleEvent(
            direction=ScaleDirection.DOWN,
            node_group=node_group,
            delta=delta,
            reason=f"Utilization {utilization:.1f}% < threshold {self._config.scale_down_threshold}%",
            target_size=total - delta,
            status="executed",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        success = self._ca_client.set_node_group_size(node_group, total - delta)
        if not success:
            event.status = "failed"
        with self._lock:
            self._events.append(event)
            self._last_scale_time = datetime.now().timestamp()
        logger.info("Scale down: %s (delta=%d, util=%.1f%%)", node_group, delta, utilization)
        return event

    def _select_node_group(self) -> str:
        if self._config.node_groups:
            return self._config.node_groups[0].name
        return "default-gpu-group"

    def scale_manual(self, node_group: str, target_size: int) -> ScaleEvent:
        event = ScaleEvent(
            direction=ScaleDirection.UP if target_size > self._ca_client.get_node_group_size(node_group) else ScaleDirection.DOWN,
            node_group=node_group,
            delta=abs(target_size - self._ca_client.get_node_group_size(node_group)),
            reason="Manual scaling request",
            target_size=target_size,
            status="executed" if self._ca_client.set_node_group_size(node_group, target_size) else "failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._events.append(event)
        return event

    def get_events(self, limit: int = 50) -> list[ScaleEvent]:
        with self._lock:
            return list(self._events[-limit:])

    def get_status(self) -> AutoscalerStatus:
        with self._lock:
            return AutoscalerStatus(
                running=self._running,
                current_config=self._config,
                last_event=self._events[-1] if self._events else None,
                event_count=len(self._events),
            )
