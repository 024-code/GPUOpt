from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from gpuopt.k8s_operator.adapters import ActionAdapter
from gpuopt.k8s_operator.client import K8sClientWrapper
from gpuopt.k8s_operator.controller import GPUOptimizationController
from gpuopt.k8s_operator.models import (
    ActionParameters,
    ActionPhase,
    ActionSpec,
    ActionStatus,
    ActionType,
    GPUOptimizationAction,
    GPUOptimizationProfile,
    GPUWorkloadProfile,
    OptimizationProfileSpec,
    OptimizationRule,
    OptimizationRuleType,
    WorkloadProfileSpec,
    WorkloadSelector,
)


class TestK8sModels:
    def test_create_profile(self):
        profile = GPUOptimizationProfile(
            name="test-profile",
            spec=OptimizationProfileSpec(
                optimizationRules=[
                    OptimizationRule(ruleType=OptimizationRuleType.GPU_QUOTA, maxGpus=8),
                    OptimizationRule(ruleType=OptimizationRuleType.UTILIZATION_TARGET, minUtilizationPercent=60.0),
                ]
            ),
        )
        assert profile.name == "test-profile"
        assert len(profile.spec.optimizationRules) == 2
        assert profile.spec.optimizationRules[0].ruleType == OptimizationRuleType.GPU_QUOTA
        assert profile.spec.optimizationRules[0].maxGpus == 8

    def test_create_action(self):
        action = GPUOptimizationAction(
            name="test-action",
            spec=ActionSpec(
                actionType=ActionType.SCALE_GPU_COUNT,
                targetCluster="cluster-1",
                parameters=ActionParameters(gpuCount=4, dryRun=True),
            ),
        )
        assert action.spec.actionType == ActionType.SCALE_GPU_COUNT
        assert action.spec.parameters.gpuCount == 4
        assert action.status.phase == ActionPhase.PENDING

    def test_create_workload_profile(self):
        wl = GPUWorkloadProfile(
            name="test-wl",
            spec=WorkloadProfileSpec(
                workloadSelector=WorkloadSelector(matchLabels={"app": "training"}),
                requirements={"gpuCount": 4, "minGpuMemoryGb": 40},
            ),
        )
        assert wl.spec.workloadSelector.matchLabels["app"] == "training"
        assert wl.spec.requirements.gpuCount == 4


class TestActionAdapter:
    def test_dry_run(self):
        client = K8sClientWrapper()
        adapter = ActionAdapter(client)
        action = GPUOptimizationAction(
            name="dry-test",
            spec=ActionSpec(
                actionType=ActionType.SCALE_GPU_COUNT,
                parameters=ActionParameters(gpuCount=8, dryRun=True),
            ),
        )
        result = adapter.execute(action)
        assert result["success"] is True
        assert result["dryRun"] is True

    def test_adapter_handles_error(self):
        client = K8sClientWrapper()
        adapter = ActionAdapter(client)
        action = GPUOptimizationAction(
            name="error-test",
            spec=ActionSpec(actionType=ActionType.APPLY_RECOMMENDATION),
        )
        result = adapter.execute(action)
        assert "success" in result


class TestGPUOptimizationController:
    def test_reconcile_once(self):
        client = K8sClientWrapper()
        controller = GPUOptimizationController(client=client, poll_interval=999)
        result = controller.reconcile_once()
        assert "actions_reconciled" in result
        assert "profiles_reconciled" in result

    def test_reconcile_with_action(self):
        client = K8sClientWrapper()
        controller = GPUOptimizationController(client=client, poll_interval=999)
        result = controller.reconcile_once()
        assert isinstance(result["actions_reconciled"], int)


class TestK8sAPI:
    def test_list_nodes_no_k8s(self, client: TestClient):
        resp = client.get("/api/v1/k8s/nodes")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_pods_no_k8s(self, client: TestClient):
        resp = client.get("/api/v1/k8s/pods")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_actions_no_k8s(self, client: TestClient):
        resp = client.get("/api/v1/k8s/actions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_profiles_no_k8s(self, client: TestClient):
        resp = client.get("/api/v1/k8s/profiles")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_trigger_reconcile(self, client: TestClient):
        resp = client.post("/api/v1/k8s/reconcile")
        assert resp.status_code == 200
        data = resp.json()
        assert "actions_reconciled" in data

    def test_execute_action(self, client: TestClient):
        resp = client.post("/api/v1/k8s/actions/execute", json={
            "name": "api-test-action",
            "namespace": "default",
            "spec": {
                "actionType": "scale_gpu_count",
                "targetCluster": "test-cluster",
                "parameters": {"gpuCount": 4, "dryRun": True},
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
