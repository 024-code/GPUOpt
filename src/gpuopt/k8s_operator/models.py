from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class OptimizationRuleType(StrEnum):
    GPU_QUOTA = "gpu_quota"
    UTILIZATION_TARGET = "utilization_target"
    POWER_CAP = "power_cap"
    SCHEDULING_PREFERENCE = "scheduling_preference"
    COST_BUDGET = "cost_budget"


class SchedulingPreference(BaseModel):
    allowPreemption: bool = False
    preferredNodeSelector: dict[str, str] = Field(default_factory=dict)
    taints: list[dict[str, str]] = Field(default_factory=list)


class OptimizationRule(BaseModel):
    ruleType: OptimizationRuleType
    enabled: bool = True
    maxGpus: int = 0
    minUtilizationPercent: float = 0.0
    maxPowerWatts: int = 0
    preferredGpuModel: str = ""
    budgetMonthlyUsd: float = 0.0
    scheduling: SchedulingPreference = Field(default_factory=SchedulingPreference)


class OptimizationProfileSpec(BaseModel):
    targetRef: dict[str, str] = Field(default_factory=lambda: {"kind": "Namespace", "name": "default"})
    optimizationRules: list[OptimizationRule] = Field(default_factory=list)


class OptimizationProfileStatus(BaseModel):
    observedGeneration: int = 0
    conditions: list[dict[str, Any]] = Field(default_factory=list)


class GPUOptimizationProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""
    namespace: str = "default"
    spec: OptimizationProfileSpec = Field(default_factory=OptimizationProfileSpec)
    status: OptimizationProfileStatus = Field(default_factory=OptimizationProfileStatus)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ActionType(StrEnum):
    SCALE_GPU_COUNT = "scale_gpu_count"
    ADJUST_POWER_CAP = "adjust_power_cap"
    MIGRATE_WORKLOAD = "migrate_workload"
    APPLY_RECOMMENDATION = "apply_recommendation"
    REQUEST_APPROVAL = "request_approval"


class ActionPhase(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ActionParameters(BaseModel):
    gpuCount: int = 0
    powerCapWatts: int = 0
    nodeSelector: dict[str, str] = Field(default_factory=dict)
    reason: str = ""
    dryRun: bool = True


class ActionSpec(BaseModel):
    actionType: ActionType
    targetCluster: str = ""
    recommendationRef: str = ""
    parameters: ActionParameters = Field(default_factory=ActionParameters)
    approvalRequired: bool = False


class ActionStatus(BaseModel):
    phase: ActionPhase = ActionPhase.PENDING
    observedGeneration: int = 0
    startTime: datetime | None = None
    completionTime: datetime | None = None
    result: str = ""
    actuationId: str = ""
    conditions: list[dict[str, Any]] = Field(default_factory=list)


class GPUOptimizationAction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""
    namespace: str = "default"
    spec: ActionSpec = Field(default_factory=ActionSpec)
    status: ActionStatus = Field(default_factory=ActionStatus)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkloadSelector(BaseModel):
    matchLabels: dict[str, str] = Field(default_factory=dict)
    matchNames: list[str] = Field(default_factory=list)


class WorkloadRequirements(BaseModel):
    minGpuMemoryGb: int = 0
    preferredGpuModel: str = ""
    gpuCount: int = 1
    nodeCount: int = 1
    gpuMemoryFraction: float = 0.0


class WorkloadOptimization(BaseModel):
    enableElasticTraining: bool = False
    enablePreemption: bool = False
    preferredBatchSize: int = 0
    tensorParallelism: int = 0
    pipelineParallelism: int = 0
    dataParallelism: int = 0
    targetLatencyMs: int = 0
    targetThroughputTokensPerSec: int = 0
    powerCapWatts: int = 0


class WorkloadProfileSpec(BaseModel):
    workloadSelector: WorkloadSelector = Field(default_factory=WorkloadSelector)
    requirements: WorkloadRequirements = Field(default_factory=WorkloadRequirements)
    optimization: WorkloadOptimization = Field(default_factory=WorkloadOptimization)


class WorkloadProfileStatus(BaseModel):
    observedGeneration: int = 0
    conditions: list[dict[str, Any]] = Field(default_factory=list)


class GPUWorkloadProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""
    namespace: str = "default"
    spec: WorkloadProfileSpec = Field(default_factory=WorkloadProfileSpec)
    status: WorkloadProfileStatus = Field(default_factory=WorkloadProfileStatus)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
