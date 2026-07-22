from __future__ import annotations

import logging
from typing import Any

from .schemas import WorkloadCapability, WorkloadClassResult

logger = logging.getLogger(__name__)

FRAMEWORK_CAPABILITIES: dict[str, dict[str, Any]] = {
    "pytorch": {
        "supports_checkpoint": True, "supports_preemption": True,
        "supports_elastic": True, "supports_migration": True,
        "destructive_actions_allowed": False, "recovery_time_seconds": 60.0,
    },
    "tensorflow": {
        "supports_checkpoint": True, "supports_preemption": True,
        "supports_elastic": True, "supports_migration": True,
        "destructive_actions_allowed": False, "recovery_time_seconds": 90.0,
    },
    "jax": {
        "supports_checkpoint": True, "supports_preemption": True,
        "supports_elastic": True, "supports_migration": False,
        "destructive_actions_allowed": False, "recovery_time_seconds": 120.0,
    },
    "onnx": {
        "supports_checkpoint": False, "supports_preemption": False,
        "supports_elastic": True, "supports_migration": True,
        "destructive_actions_allowed": True, "recovery_time_seconds": 10.0,
    },
    "inference": {
        "supports_checkpoint": False, "supports_preemption": False,
        "supports_elastic": True, "supports_migration": True,
        "destructive_actions_allowed": False, "recovery_time_seconds": 5.0,
    },
    "unknown": {
        "supports_checkpoint": False, "supports_preemption": False,
        "supports_elastic": False, "supports_migration": False,
        "destructive_actions_allowed": False, "recovery_time_seconds": 300.0,
    },
}

DESTRUCTIVE_ACTIONS = ["preempt", "kill", "evict", "terminate", "delete"]
NON_DESTRUCTIVE_ACTIONS = ["placement", "scale_up", "scale_down", "right_size", "migrate"]


class WorkloadClassifier:
    def classify(self, workload: dict) -> WorkloadClassResult:
        wid = workload.get("job_id") or workload.get("id", "unknown")
        name = workload.get("name", wid)
        framework = (workload.get("framework") or "unknown").lower()
        duration = workload.get("max_duration_minutes", 120)

        caps = dict(FRAMEWORK_CAPABILITIES.get(framework, FRAMEWORK_CAPABILITIES["unknown"]))
        if kwargs := workload.get("capability_overrides"):
            caps.update(kwargs)

        if duration and duration < 5:
            caps["supports_checkpoint"] = False
            caps["supports_preemption"] = False

        if workload.get("checkpoint_enabled") is False:
            caps["supports_checkpoint"] = False

        classification = self._determine_classification(caps, framework)
        capability = WorkloadCapability(
            workload_type=classification,
            **{k: v for k, v in caps.items() if k in WorkloadCapability.model_fields},
        )

        recs, unsafe = self._generate_recommendations(capability)
        return WorkloadClassResult(
            workload_id=wid, name=name, framework=framework,
            capability=capability, recommended_actions=recs,
            unsafe_actions=unsafe,
        )

    def _determine_classification(self, caps: dict, framework: str) -> str:
        if caps["supports_checkpoint"] and caps["supports_preemption"] and caps["supports_elastic"]:
            return "resilient"
        if caps["supports_checkpoint"] and caps["supports_preemption"]:
            return "preemptible"
        if caps["supports_migration"]:
            return "migratable"
        return "fragile"

    def _generate_recommendations(self, cap: WorkloadCapability) -> tuple[list[str], list[str]]:
        recs = []
        unsafe = []
        if cap.supports_checkpoint:
            recs.append("checkpoint")
            recs.append("preempt")
        if cap.supports_elastic:
            recs.append("elastic_scale")
        if cap.supports_migration:
            recs.append("migration")
        if cap.supports_preemption:
            recs.append("preempt")
        if cap.destructive_actions_allowed:
            unsafe.extend(DESTRUCTIVE_ACTIONS)
        if not cap.supports_preemption:
            unsafe.append("preempt")
            unsafe.append("evict")
        if not cap.supports_migration:
            unsafe.append("migrate")
        if not cap.supports_checkpoint:
            unsafe.append("kill")
        return recs, unsafe

    def is_action_safe(self, workload: dict, action: str) -> tuple[bool, str]:
        result = self.classify(workload)
        if action in result.unsafe_actions:
            return False, f"Action '{action}' is unsafe for {result.capability.classification} workload '{result.name}'"
        if action not in result.recommended_actions and action not in NON_DESTRUCTIVE_ACTIONS:
            return False, f"Action '{action}' not recommended for this workload type"
        return True, "Action is safe"
