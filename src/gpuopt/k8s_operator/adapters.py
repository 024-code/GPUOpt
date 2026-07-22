from __future__ import annotations

import logging
from typing import Any

from .client import K8sClientWrapper
from .models import ActionType, GPUOptimizationAction

logger = logging.getLogger(__name__)


class ActionAdapter:
    def __init__(self, client: K8sClientWrapper) -> None:
        self._client = client

    def execute(self, action: GPUOptimizationAction) -> dict[str, Any]:
        action_type = action.spec.actionType
        params = action.spec.parameters
        cluster = action.spec.targetCluster

        logger.info("Executing action %s (%s) on cluster %s", action.name, action_type, cluster)

        handlers = {
            ActionType.SCALE_GPU_COUNT: self._scale_gpu_count,
            ActionType.ADJUST_POWER_CAP: self._adjust_power_cap,
            ActionType.MIGRATE_WORKLOAD: self._migrate_workload,
            ActionType.APPLY_RECOMMENDATION: self._apply_recommendation,
            ActionType.REQUEST_APPROVAL: self._request_approval,
        }
        handler = handlers.get(action_type)
        if handler is None:
            return {"success": False, "error": f"Unknown action type: {action_type}"}

        try:
            if params.dryRun:
                return self._dry_run(action)
            return handler(action)
        except Exception as exc:
            logger.exception("Action %s failed", action.name)
            return {"success": False, "error": str(exc)}

    def _dry_run(self, action: GPUOptimizationAction) -> dict[str, Any]:
        return {
            "success": True,
            "dryRun": True,
            "message": f"Would execute {action.spec.actionType} on {action.spec.targetCluster} "
                       f"with params: {action.spec.parameters.model_dump(mode='json')}",
        }

    def _scale_gpu_count(self, action: GPUOptimizationAction) -> dict[str, Any]:
        params = action.spec.parameters
        namespace = action.namespace
        gpu_count = params.gpuCount
        results: list[dict[str, Any]] = []

        pods = self._client.list_pods(namespace, "gpuopt.ai/managed=true")
        for pod in pods[:5]:
            if gpu_count > 0:
                body = {
                    "metadata": {"annotations": {"gpuopt.ai/requested-gpus": str(gpu_count)}},
                }
            else:
                body = {
                    "metadata": {"annotations": {"gpuopt.ai/requested-gpus": None}},
                }
            success = self._client.patch_deployment(
                pod["namespace"], pod["name"].rsplit("-", 1)[0], body
            )
            results.append({"pod": pod["name"], "patched": success})

        return {"success": True, "action": "scale_gpu_count", "gpuCount": gpu_count, "results": results}

    def _adjust_power_cap(self, action: GPUOptimizationAction) -> dict[str, Any]:
        params = action.spec.parameters
        power_watts = params.powerCapWatts
        results: list[dict[str, Any]] = []

        nodes = self._client.list_nodes()
        for node in nodes:
            if node["gpu_capacity"] > 0:
                labels = {
                    "gpuopt.ai/power-cap-watts": str(power_watts),
                    "gpuopt.ai/power-cap-requested": str(action.name),
                }
                success = self._client.label_node(node["name"], labels)
                results.append({"node": node["name"], "labeled": success})

        return {"success": True, "action": "adjust_power_cap", "powerWatts": power_watts, "results": results}

    def _migrate_workload(self, action: GPUOptimizationAction) -> dict[str, Any]:
        params = action.spec.parameters
        results: list[dict[str, Any]] = []

        pods = self._client.list_pods(action.namespace, "gpuopt.ai/migratable=true")
        for pod in pods[:3]:
            body = {
                "spec": {
                    "nodeSelector": params.nodeSelector if params.nodeSelector else None,
                    "tolerations": [{"key": "gpuopt.ai/migrated", "operator": "Exists"}],
                }
            }
            success = self._client.patch_deployment(
                pod["namespace"], pod["name"].rsplit("-", 1)[0], body
            )
            results.append({"pod": pod["name"], "patched": success})

        return {"success": True, "action": "migrate_workload", "results": results}

    def _apply_recommendation(self, action: GPUOptimizationAction) -> dict[str, Any]:
        recommendation_id = action.spec.recommendationRef
        return {
            "success": True,
            "action": "apply_recommendation",
            "recommendationRef": recommendation_id,
            "message": f"Recommendation {recommendation_id} would be applied via actuation service",
        }

    def _request_approval(self, action: GPUOptimizationAction) -> dict[str, Any]:
        return {
            "success": True,
            "action": "request_approval",
            "message": f"Approval request created for action {action.name}",
            "approvalRequired": True,
        }
