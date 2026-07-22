from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ClusterHealth(StrEnum):
    UNKNOWN = "unknown"
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class FederationRole(StrEnum):
    ACTIVE = "active"
    STANDBY = "standby"
    DRAINING = "draining"


class FederatedCluster(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    endpoint: str = ""
    environment: str = "sandbox"
    health: ClusterHealth = ClusterHealth.UNKNOWN
    role: FederationRole = FederationRole.ACTIVE
    total_gpus: int = 0
    free_gpus: int = 0
    gpu_models: list[str] = Field(default_factory=list)
    avg_utilization: float = 0.0
    region: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    options: dict[str, Any] = Field(default_factory=dict)


class WorkloadState(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"


class FederatedWorkload(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    required_gpus: int = 1
    priority: int = 5
    memory_gb: float = 8.0
    state: WorkloadState = WorkloadState.PENDING
    assigned_cluster: str = ""
    assigned_node: str = ""
    scheduler_type: str = "rl"
    original_workload_ref: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FederationState(BaseModel):
    clusters: list[FederatedCluster] = Field(default_factory=list)
    workloads: list[FederatedWorkload] = Field(default_factory=list)
    total_gpus_across_clusters: int = 0
    total_free_gpus: int = 0
    total_pending_workloads: int = 0
    total_running_workloads: int = 0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
