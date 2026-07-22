from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from .adapters import ActionAdapter
from .client import K8sClientWrapper
from .models import ActionPhase, ActionSpec, ActionStatus, ActionType, GPUOptimizationAction

logger = logging.getLogger(__name__)


class GPUOptimizationController:
    def __init__(
        self,
        client: K8sClientWrapper | None = None,
        adapter: ActionAdapter | None = None,
        poll_interval: float = 30.0,
    ) -> None:
        self._client = client or K8sClientWrapper()
        self._adapter = adapter or ActionAdapter(self._client)
        self._poll_interval = poll_interval
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.info("Controller already running")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="k8s-operator")
        self._thread.start()
        logger.info(
            "GPUOptimizationController started (poll_interval=%ss, thread=%s)",
            self._poll_interval, self._thread.name,
        )

    def _run_loop(self) -> None:
        while self._running:
            try:
                self._reconcile_actions()
                self._reconcile_profiles()
            except Exception as exc:
                logger.exception("Reconciliation error: %s", exc)
            time.sleep(self._poll_interval)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("GPUOptimizationController stopped")

    def reconcile_once(self) -> dict[str, Any]:
        action_count = self._reconcile_actions()
        profile_count = self._reconcile_profiles()
        return {
            "actions_reconciled": action_count,
            "profiles_reconciled": profile_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _reconcile_actions(self) -> int:
        items = self._client.list_gpuoptimization_actions()
        count = 0
        for item in items:
            try:
                self._reconcile_action(item)
                count += 1
            except Exception as exc:
                logger.error("Failed to reconcile action %s: %s", item.get("metadata", {}).get("name", "?"), exc)
        return count

    def _reconcile_action(self, item: dict[str, Any]) -> None:
        metadata = item.get("metadata", {})
        spec_data = item.get("spec", {})
        status_data = item.get("status", {})
        name = metadata.get("name", "")
        namespace = metadata.get("namespace", "default")
        resource_version = metadata.get("resourceVersion", "")

        action = self._build_action(name, namespace, spec_data, status_data)
        if action is None:
            return

        if action.status.phase in (ActionPhase.COMPLETED, ActionPhase.FAILED, ActionPhase.ROLLED_BACK):
            return

        logger.info("Reconciling action %s/%s (phase=%s)", namespace, name, action.status.phase)

        if action.status.phase == ActionPhase.PENDING:
            self._client.update_action_status(namespace, name, {
                "phase": ActionPhase.RUNNING.value,
                "startTime": datetime.now(timezone.utc).isoformat(),
                "observedGeneration": 1,
                "conditions": [
                    {
                        "type": "Reconciled",
                        "status": "True",
                        "reason": "ExecutionStarted",
                        "message": f"Action {name} execution started",
                        "lastTransitionTime": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            })
            action.status.phase = ActionPhase.RUNNING

        if action.status.phase == ActionPhase.RUNNING:
            result = self._adapter.execute(action)
            new_phase = ActionPhase.COMPLETED if result.get("success") else ActionPhase.FAILED
            self._client.update_action_status(namespace, name, {
                "phase": new_phase.value,
                "completionTime": datetime.now(timezone.utc).isoformat() if new_phase in (
                    ActionPhase.COMPLETED, ActionPhase.FAILED
                ) else None,
                "result": json.dumps(result),
                "observedGeneration": 1,
                "conditions": [
                    {
                        "type": "Reconciled",
                        "status": "True" if new_phase == ActionPhase.COMPLETED else "False",
                        "reason": "ExecutionCompleted" if new_phase == ActionPhase.COMPLETED else "ExecutionFailed",
                        "message": result.get("message", result.get("error", "Unknown")),
                        "lastTransitionTime": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            })
            logger.info("Action %s/%s completed with phase %s", namespace, name, new_phase)

    def _build_action(
        self, name: str, namespace: str, spec_data: dict[str, Any], status_data: dict[str, Any]
    ) -> GPUOptimizationAction | None:
        try:
            action_type = spec_data.get("actionType", "")
            params = spec_data.get("parameters", {})
            approval = spec_data.get("approvalRequired", False)
            target = spec_data.get("targetCluster", "")
            rec_ref = spec_data.get("recommendationRef", "")

            phase_str = status_data.get("phase", ActionPhase.PENDING.value)
            try:
                phase = ActionPhase(phase_str)
            except ValueError:
                phase = ActionPhase.PENDING

            return GPUOptimizationAction(
                name=name,
                namespace=namespace,
                spec=ActionSpec(
                    actionType=ActionType(action_type) if action_type else ActionType.APPLY_RECOMMENDATION,
                    targetCluster=target,
                    recommendationRef=rec_ref,
                    parameters={
                        "gpuCount": params.get("gpuCount", 0),
                        "powerCapWatts": params.get("powerCapWatts", 0),
                        "nodeSelector": params.get("nodeSelector", {}),
                        "reason": params.get("reason", ""),
                        "dryRun": params.get("dryRun", True),
                    },
                    approvalRequired=approval,
                ),
                status=ActionStatus(phase=phase),
            )
        except Exception as exc:
            logger.error("Failed to build action from CRD data: %s", exc)
            return None

    def _reconcile_profiles(self) -> int:
        items = self._client.list_gpuoptimization_profiles()
        count = 0
        for item in items:
            try:
                self._reconcile_profile(item)
                count += 1
            except Exception as exc:
                logger.error("Failed to reconcile profile %s: %s", item.get("metadata", {}).get("name", "?"), exc)
        return count

    def _reconcile_profile(self, item: dict[str, Any]) -> None:
        metadata = item.get("metadata", {})
        spec_data = item.get("spec", {})
        status_data = item.get("status", {})
        name = metadata.get("name", "")
        namespace = metadata.get("namespace", "default")

        generation = metadata.get("generation", 0)
        observed = status_data.get("observedGeneration", 0)

        if generation <= observed:
            return

        rules = spec_data.get("optimizationRules", [])
        target = spec_data.get("targetRef", {"kind": "Namespace", "name": "default"})
        logger.info(
            "Reconciling profile %s/%s with %d rules for %s/%s",
            namespace, name, len(rules), target.get("kind"), target.get("name"),
        )

        self._client.update_profile_status(namespace, name, {
            "observedGeneration": generation,
            "conditions": [
                {
                    "type": "Reconciled",
                    "status": "True",
                    "reason": "ProfileApplied",
                    "message": f"Profile {name} reconciled with {len(rules)} rules",
                    "lastTransitionTime": datetime.now(timezone.utc).isoformat(),
                }
            ],
        })
        logger.info("Profile %s/%s reconciled", namespace, name)


def run_controller(poll_interval: float = 30.0, in_cluster: bool = True) -> None:
    client = K8sClientWrapper(in_cluster=in_cluster)
    controller = GPUOptimizationController(client=client, poll_interval=poll_interval)
    controller.start()
