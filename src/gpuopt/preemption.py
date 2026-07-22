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


class PreemptionPolicy(str, enum.Enum):
    NEVER = "Never"
    PREEMPT_LOWER_PRIORITY = "PreemptLowerPriority"


class PriorityClass(str, enum.Enum):
    BATCH = "batch"
    INTERACTIVE = "interactive"
    PRODUCTION = "production"
    CRITICAL = "critical"


_PRIORITY_SORT = {
    PriorityClass.CRITICAL: 1000,
    PriorityClass.PRODUCTION: 500,
    PriorityClass.INTERACTIVE: 200,
    PriorityClass.BATCH: 100,
}


@dataclass
class PreemptionCandidate:
    workload_name: str
    namespace: str
    priority: PriorityClass
    reason: str
    gpu_claimed: int = 0
    age_minutes: float = 0.0


@dataclass
class PreemptionAction:
    workload_name: str
    namespace: str
    priority: PriorityClass
    reason: str
    eviction_strategy: str
    initiated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "pending"


@dataclass
class PreemptionPolicyConfig:
    policy: PreemptionPolicy = PreemptionPolicy.PREEMPT_LOWER_PRIORITY
    min_priority_delta: int = 100
    preempt_oldest_first: bool = True
    max_preemptions_per_cycle: int = 5


class PreemptionEngine:
    def __init__(
        self,
        k8s_client_wrapper: K8sClientWrapper | None = None,
        config: PreemptionPolicyConfig | None = None,
    ) -> None:
        self._k8s = k8s_client_wrapper
        self._config = config or PreemptionPolicyConfig()
        self._history: list[PreemptionAction] = []
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def config(self) -> PreemptionPolicyConfig:
        return self._config

    @config.setter
    def config(self, value: PreemptionPolicyConfig) -> None:
        with self._lock:
            self._config = value

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._cycle_loop, daemon=True, name="preemption-engine")
        self._thread.start()
        logger.info("Preemption engine started (policy=%s)", self._config.policy.value)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Preemption engine stopped")

    def _cycle_loop(self) -> None:
        while self._running:
            try:
                self.cycle()
            except Exception as exc:
                logger.error("Preemption cycle error: %s", exc)
            threading.Event().wait(30)

    def cycle(self) -> list[PreemptionAction]:
        if self._config.policy == PreemptionPolicy.NEVER:
            return []
        workloads = self._discover_workloads()
        collisions = self._detect_collisions(workloads)
        actions: list[PreemptionAction] = []
        for high_pri, low_pri in collisions:
            action = self._preempt(high_pri, low_pri)
            if action:
                actions.append(action)
                with self._lock:
                    self._history.append(action)
                if len(actions) >= self._config.max_preemptions_per_cycle:
                    break
        return actions

    def _discover_workloads(self) -> list[dict[str, Any]]:
        if self._k8s is None:
            return self._mock_workloads()
        try:
            pods = self._k8s.list_pods(namespace="")
            workloads: dict[str, dict[str, Any]] = {}
            for pod in pods.get("items", pods) if isinstance(pods, dict) else pods:
                meta = pod.metadata if hasattr(pod, "metadata") else pod.get("metadata", {})
                ns = meta.namespace if hasattr(meta, "namespace") else meta.get("namespace", "default")
                owner = meta.owner_references[0].name if (hasattr(meta, "owner_references") and meta.owner_references) else (meta.get("owner_references", [{}])[0].get("name", meta.get("name", "unknown")))
                wl_key = f"{ns}/{owner}"
                if wl_key not in workloads:
                    priority_str = (meta.labels or {}).get("priority-class", "") if hasattr(meta, "labels") else (meta.get("labels", {}).get("priority-class", ""))
                    try:
                        priority = PriorityClass(priority_str) if priority_str else PriorityClass.BATCH
                    except ValueError:
                        priority = PriorityClass.BATCH
                    creation = meta.creation_timestamp if hasattr(meta, "creation_timestamp") else meta.get("creation_timestamp")
                    age = 0.0
                    if creation:
                        if hasattr(creation, "timestamp"):
                            age = (datetime.now(timezone.utc) - creation).total_seconds() / 60.0
                        elif isinstance(creation, str):
                            try:
                                age = (datetime.now(timezone.utc) - datetime.fromisoformat(creation.replace("Z", "+00:00"))).total_seconds() / 60.0
                            except Exception:
                                pass
                    gpu_str = (meta.labels or {}).get("gpu-count", "0") if hasattr(meta, "labels") else (meta.get("labels", {}).get("gpu-count", "0"))
                    try:
                        gpu_count = int(gpu_str)
                    except (ValueError, TypeError):
                        gpu_count = 0
                    workloads[wl_key] = {
                        "workload_name": owner,
                        "namespace": ns,
                        "priority": priority,
                        "gpu_claimed": gpu_count,
                        "age_minutes": age,
                    }
            return list(workloads.values())
        except Exception as exc:
            logger.warning("K8s discovery failed: %s", exc)
            return self._mock_workloads()

    def _mock_workloads(self) -> list[dict[str, Any]]:
        import random
        return [
            {
                "workload_name": f"train-{i}-{random.randint(100,999)}",
                "namespace": "default",
                "priority": random.choice(list(PriorityClass)),
                "gpu_claimed": random.choice([1, 2, 4, 8]),
                "age_minutes": random.uniform(1, 120),
            }
            for i in range(random.randint(4, 8))
        ]

    def _detect_collisions(self, workloads: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        sorted_wl = sorted(workloads, key=lambda w: _PRIORITY_SORT.get(w["priority"], 0), reverse=True)
        collisions: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for i, high in enumerate(sorted_wl):
            for low in sorted_wl[i + 1:]:
                high_score = _PRIORITY_SORT.get(high["priority"], 0)
                low_score = _PRIORITY_SORT.get(low["priority"], 0)
                if high_score - low_score >= self._config.min_priority_delta:
                    collisions.append((high, low))
        return collisions

    def _preempt(self, higher: dict[str, Any], lower: dict[str, Any]) -> PreemptionAction | None:
        wl_name = lower["workload_name"]
        ns = lower["namespace"]
        reason = f"Preempted by {higher['workload_name']} (priority delta={_PRIORITY_SORT.get(higher['priority'], 0) - _PRIORITY_SORT.get(lower['priority'], 0)})"
        if self._k8s is not None:
            try:
                self._evict_with_k8s(wl_name, ns)
            except Exception as exc:
                logger.error("Preemption eviction failed for %s/%s: %s", ns, wl_name, exc)
                return PreemptionAction(workload_name=wl_name, namespace=ns, priority=lower["priority"], reason=reason, eviction_strategy="k8s-evict", status="failed")
        logger.info("Preempting %s/%s: %s", ns, wl_name, reason)
        return PreemptionAction(workload_name=wl_name, namespace=ns, priority=lower["priority"], reason=reason, eviction_strategy="k8s-evict", status="executed")

    def _evict_with_k8s(self, name: str, namespace: str) -> None:
        core = k8s_client.CoreV1Api()
        pods = core.list_namespaced_pod(namespace, label_selector=f"app={name}").items
        for pod in pods:
            core.delete_namespaced_pod(pod.metadata.name, namespace, grace_period_seconds=0)

    def get_history(self, limit: int = 50) -> list[PreemptionAction]:
        with self._lock:
            return list(self._history[-limit:])

    def apply_preemption_policy(self, workload_name: str, namespace: str, priority: PriorityClass, policy: PreemptionPolicy) -> bool:
        if self._k8s is None:
            return True
        try:
            label_priority = {"batch": "batch", "interactive": "interactive", "production": "production", "critical": "critical"}.get(priority.value, "batch")
            self._k8s._ensure_client()
            core = k8s_client.CoreV1Api()
            pods = core.list_namespaced_pod(namespace, label_selector=f"app={workload_name}").items
            for pod in pods:
                patch = {
                    "metadata": {
                        "labels": {
                            "priority-class": label_priority,
                            "preemption-policy": policy.value,
                        }
                    }
                }
                core.patch_namespaced_pod(pod.metadata.name, namespace, patch)
            return True
        except Exception as exc:
            logger.error("Failed to apply preemption policy to %s/%s: %s", namespace, workload_name, exc)
            return False
