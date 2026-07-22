from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from typing import Any

from .schemas import CanaryDeployment, CanaryStep
from .governance_extended import RollbackManager

logger = logging.getLogger(__name__)

CANARY_TEMPLATES: dict[str, dict] = {
    "placement": {
        "steps": [
            {"name": "1% canary", "traffic_percent": 1.0, "duration_minutes": 10.0,
             "success_criteria": ["error_rate_below_0.01", "latency_p99_below_2x"]},
            {"name": "5% canary", "traffic_percent": 5.0, "duration_minutes": 15.0,
             "success_criteria": ["error_rate_below_0.01", "latency_p99_below_1.5x"]},
            {"name": "25% canary", "traffic_percent": 25.0, "duration_minutes": 20.0,
             "success_criteria": ["error_rate_below_0.005", "latency_p99_below_1.2x"]},
            {"name": "50% canary", "traffic_percent": 50.0, "duration_minutes": 30.0,
             "success_criteria": ["error_rate_below_0.005", "latency_p99_below_1.1x"]},
            {"name": "100% rollout", "traffic_percent": 100.0, "duration_minutes": 60.0,
             "success_criteria": ["error_rate_below_0.001", "latency_p99_below_1.0x"]},
        ]
    },
    "scale_down": {
        "steps": [
            {"name": "drain 1 node", "traffic_percent": 10.0, "duration_minutes": 15.0,
             "success_criteria": ["workloads_migrated", "no_errors"]},
            {"name": "drain 25% nodes", "traffic_percent": 25.0, "duration_minutes": 20.0,
             "success_criteria": ["workloads_migrated", "no_errors", "capacity_sufficient"]},
            {"name": "drain 50% nodes", "traffic_percent": 50.0, "duration_minutes": 30.0,
             "success_criteria": ["workloads_migrated", "no_errors"]},
            {"name": "full scale", "traffic_percent": 100.0, "duration_minutes": 30.0,
             "success_criteria": ["target_count_reached", "all_workloads_stable"]},
        ]
    },
    "preempt": {
        "steps": [
            {"name": "preempt 1 job", "traffic_percent": 5.0, "duration_minutes": 5.0,
             "success_criteria": ["job_preempted", "checkpoint_saved"]},
            {"name": "preempt 10%", "traffic_percent": 10.0, "duration_minutes": 10.0,
             "success_criteria": ["all_preempted_ckpt_saved", "no_crashes"]},
            {"name": "preempt 25%", "traffic_percent": 25.0, "duration_minutes": 15.0,
             "success_criteria": ["cluster_stable", "preempted_jobs_requeued"]},
        ]
    },
}


class CanaryManager:
    def __init__(self) -> None:
        self._deployments: dict[str, CanaryDeployment] = {}
        self._rollback = RollbackManager()

    def create(self, action_id: str, action_type: str) -> CanaryDeployment:
        template = CANARY_TEMPLATES.get(action_type, CANARY_TEMPLATES["placement"])
        steps = [CanaryStep(**s, status="pending") for s in template["steps"]]
        dep = CanaryDeployment(
            action_id=action_id,
            action_type=action_type,
            steps=steps,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._deployments[dep.deployment_id] = dep
        logger.info("Canary %s created for %s action %s", dep.deployment_id, action_type, action_id)
        return dep

    def advance_step(self, deployment_id: str, metrics: dict | None = None) -> dict:
        dep = self._deployments.get(deployment_id)
        if not dep:
            return {"error": "Deployment not found"}

        step = dep.steps[dep.current_step]
        step.status = "running"
        passed = self._evaluate_step(step, metrics or {})
        if passed:
            step.status = "passed"
            dep.current_step += 1
            if dep.current_step >= len(dep.steps):
                dep.status = "completed"
                dep.completed_at = datetime.now(timezone.utc).isoformat()
                return {"deployment_id": deployment_id, "status": "completed", "step": dep.current_step}
            next_step = dep.steps[dep.current_step]
            next_step.status = "active"
            return {
                "deployment_id": deployment_id, "status": "advancing",
                "completed_step": step.name, "next_step": next_step.name,
                "traffic_percent": next_step.traffic_percent,
            }
        step.status = "failed"
        return self._trigger_rollback(deployment_id, f"Step '{step.name}' failed criteria")

    def _evaluate_step(self, step: CanaryStep, metrics: dict) -> bool:
        for criterion in step.success_criteria:
            if criterion.startswith("error_rate_below_"):
                threshold = float(criterion.split("_")[-1])
                if metrics.get("error_rate", random.random() * 0.02) > threshold:
                    return False
            elif criterion.startswith("latency_p99_below_"):
                mult = float(criterion.split("_")[-1].replace("x", ""))
                if metrics.get("latency_p99", random.uniform(50, 200)) > 100 * mult:
                    return False
            elif criterion == "no_errors":
                if metrics.get("error_count", random.randint(0, 3)) > 0:
                    return False
            elif criterion == "job_preempted" or criterion == "checkpoint_saved":
                ok = metrics.get(criterion, random.random() > 0.2)
                if not ok:
                    return False
            else:
                ok = metrics.get(criterion, random.random() > 0.1)
                if not ok:
                    return False
        return True

    def _trigger_rollback(self, deployment_id: str, reason: str) -> dict:
        dep = self._deployments[deployment_id]
        dep.status = "rolling_back"
        dep.rollback_triggered = True
        plan = self._rollback.create_plan(dep.action_id, dep.action_type, reason)
        self._rollback.execute(plan)
        dep.status = "rolled_back"
        dep.completed_at = datetime.now(timezone.utc).isoformat()
        return {
            "deployment_id": deployment_id, "status": "rolled_back",
            "reason": reason, "rollback_plan_id": plan.plan_id,
        }

    def get_deployment(self, deployment_id: str) -> CanaryDeployment | None:
        return self._deployments.get(deployment_id)

    def list_deployments(self) -> list[CanaryDeployment]:
        return list(self._deployments.values())
