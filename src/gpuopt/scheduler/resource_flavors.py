from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class ResourceFlavorTier(str, Enum):
    PREMIUM = "premium"
    STANDARD = "standard"
    SPOT = "spot"
    RESERVED = "reserved"


@dataclass
class ResourceFlavor:
    name: str
    node_labels: dict[str, str] = field(default_factory=dict)
    node_taints: list[dict[str, str]] = field(default_factory=list)
    resources: dict[str, str] = field(default_factory=dict)
    tier: ResourceFlavorTier = ResourceFlavorTier.STANDARD
    priority: int = 0
    max_workloads: int = 0
    active: bool = True
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class FlavorUsage:
    flavor_name: str
    total_gpus: int
    used_gpus: int
    available_gpus: int
    total_memory_gib: float
    used_memory_gib: float
    utilization_percent: float
    workloads_admitted: int
    workloads_pending: int


class ResourceFlavorManager:
    def __init__(self) -> None:
        self._flavors: dict[str, ResourceFlavor] = {}
        self._usage: dict[str, FlavorUsage] = {}

    def create_flavor(self, name: str, node_labels: dict[str, str] | None = None,
                       node_taints: list[dict[str, str]] | None = None,
                       resources: dict[str, str] | None = None,
                       tier: ResourceFlavorTier = ResourceFlavorTier.STANDARD,
                       priority: int = 0, max_workloads: int = 0) -> ResourceFlavor:
        flavor = ResourceFlavor(
            name=name,
            node_labels=node_labels or {},
            node_taints=node_taints or [],
            resources=resources or {"nvidia.com/gpu": "1"},
            tier=tier, priority=priority, max_workloads=max_workloads,
        )
        self._flavors[name] = flavor
        return flavor

    def get_flavor(self, name: str) -> ResourceFlavor | None:
        return self._flavors.get(name)

    def list_flavors(self, tier: ResourceFlavorTier | None = None) -> list[ResourceFlavor]:
        if tier:
            return [f for f in self._flavors.values() if f.tier == tier]
        return list(self._flavors.values())

    def update_flavor(self, name: str, **kwargs: Any) -> ResourceFlavor | None:
        flavor = self._flavors.get(name)
        if flavor is None:
            return None
        for key, value in kwargs.items():
            if hasattr(flavor, key):
                setattr(flavor, key, value)
        return flavor

    def delete_flavor(self, name: str) -> bool:
        if name in self._flavors:
            del self._flavors[name]
            if name in self._usage:
                del self._usage[name]
            return True
        return False

    def update_usage(self, flavor_name: str, total_gpus: int, used_gpus: int,
                      total_memory_gib: float, used_memory_gib: float,
                      workloads_admitted: int, workloads_pending: int) -> None:
        self._usage[flavor_name] = FlavorUsage(
            flavor_name=flavor_name,
            total_gpus=total_gpus, used_gpus=used_gpus,
            available_gpus=total_gpus - used_gpus,
            total_memory_gib=total_memory_gib, used_memory_gib=used_memory_gib,
            utilization_percent=round(used_gpus / max(total_gpus, 1) * 100, 1),
            workloads_admitted=workloads_admitted, workloads_pending=workloads_pending,
        )

    def get_usage(self, flavor_name: str) -> FlavorUsage | None:
        return self._usage.get(flavor_name)

    def get_all_usage(self) -> list[FlavorUsage]:
        return list(self._usage.values())

    def select_flavor(self, required_gpus: int, preferred_tier: ResourceFlavorTier | None = None,
                       exclude_flavors: list[str] | None = None) -> ResourceFlavor | None:
        exclude = set(exclude_flavors or [])
        candidates = sorted(
            [f for f in self._flavors.values() if f.active and f.name not in exclude],
            key=lambda f: (-f.priority, f.name),
        )
        if preferred_tier:
            tier_candidates = [f for f in candidates if f.tier == preferred_tier]
            if tier_candidates:
                candidates = tier_candidates

        for flavor in candidates:
            usage = self._usage.get(flavor.name)
            if usage and usage.available_gpus >= required_gpus:
                return flavor
            if usage is None:
                return flavor
        return candidates[0] if candidates else None

    def to_kueue_spec(self, flavor: ResourceFlavor) -> dict[str, Any]:
        return {
            "name": flavor.name,
            "nodeLabels": flavor.node_labels,
            "nodeTaints": flavor.node_taints,
            "resources": [
                {"name": k, "nominalQuota": v}
                for k, v in flavor.resources.items()
            ],
            "tier": flavor.tier.value,
            "priority": flavor.priority,
        }


_flavor_manager = ResourceFlavorManager()


def get_flavor_manager() -> ResourceFlavorManager:
    return _flavor_manager
