from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FairShareAllocation:
    tenant_id: str
    fair_share: float
    gpus_allocated: int
    gpus_requested: int
    gpu_quota: int
    usage_ratio: float
    fair_share_ratio: float
    dominant_share: float
    priority: float
    preemptible: bool
    adjustment: float = 0.0


@dataclass
class DominantResourceFairnessResult:
    allocations: list[FairShareAllocation]
    total_gpus: int
    total_allocated: int
    dominant_share_threshold: float
    over_allocated: list[str]
    under_allocated: list[str]


class DominantResourceFairness:
    def __init__(self) -> None:
        self._default_weight = 1.0
        self._preemption_threshold = 0.9
        self._min_fair_share = 0.05

    def compute(
        self,
        tenants: dict[str, dict[str, Any]],
        total_gpus: int,
    ) -> DominantResourceFairnessResult:
        if not tenants or total_gpus <= 0:
            return DominantResourceFairnessResult(
                allocations=[], total_gpus=total_gpus, total_allocated=0,
                dominant_share_threshold=0.0,
                over_allocated=[], under_allocated=[],
            )

        total_weight = sum(
            t.get("weight", self._default_weight) for t in tenants.values()
        )

        allocations: list[FairShareAllocation] = []
        total_requested = sum(t.get("requested_gpus", 0) for t in tenants.values())
        total_allocated = sum(t.get("allocated_gpus", 0) for t in tenants.values())

        for tenant_id, info in tenants.items():
            fair_share = (info.get("weight", self._default_weight) / total_weight) if total_weight > 0 else 0
            requested = info.get("requested_gpus", 0)
            allocated = info.get("allocated_gpus", 0)
            quota = info.get("quota", total_gpus)
            priority = info.get("priority", 1.0)
            preemptible = info.get("preemptible", False)

            fair_share = max(fair_share, self._min_fair_share)
            ideal_share = fair_share * total_gpus

            if ideal_share > 0:
                dominant_share = allocated / ideal_share
            else:
                dominant_share = 0.0

            usage_ratio = allocated / max(quota, 1)
            fair_share_ratio = allocated / max(ideal_share, 1)

            adjustment = 0.0
            if dominant_share > self._preemption_threshold and preemptible:
                adjustment = -0.1
            elif dominant_share < 0.5 and priority > 1.0:
                adjustment = 0.05

            allocations.append(FairShareAllocation(
                tenant_id=tenant_id,
                fair_share=round(fair_share, 4),
                gpus_allocated=allocated,
                gpus_requested=requested,
                gpu_quota=quota,
                usage_ratio=round(usage_ratio, 4),
                fair_share_ratio=round(fair_share_ratio, 4),
                dominant_share=round(dominant_share, 4),
                priority=priority,
                preemptible=preemptible,
                adjustment=round(adjustment, 4),
            ))

        dominant_shares = [a.dominant_share for a in allocations]
        threshold = sum(dominant_shares) / max(len(dominant_shares), 1) if dominant_shares else 0
        threshold = max(threshold, 1.0)

        over = [a.tenant_id for a in allocations if a.dominant_share > threshold and a.gpus_allocated > 0]
        under = [a.tenant_id for a in allocations if a.dominant_share < 0.8 and a.gpus_requested > a.gpus_allocated]

        return DominantResourceFairnessResult(
            allocations=sorted(allocations, key=lambda a: -a.dominant_share),
            total_gpus=total_gpus,
            total_allocated=total_allocated,
            dominant_share_threshold=round(threshold, 4),
            over_allocated=over,
            under_allocated=under,
        )

    def suggest_rebalance(self, result: DominantResourceFairnessResult) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for over in result.over_allocated:
            actions.append({
                "type": "reduce",
                "tenant_id": over,
                "reason": f"Dominant share exceeds threshold ({result.dominant_share_threshold:.2f})",
                "suggested_reduction": "10-20%",
            })
        for under in result.under_allocated:
            actions.append({
                "type": "increase",
                "tenant_id": under,
                "reason": "Fair share below 80% of entitlement",
                "suggested_increase": "consider priority boost or quota increase",
            })
        return actions


class ProportionalFairnessScheduler:
    def __init__(self) -> None:
        self._drf = DominantResourceFairness()

    def schedule(self, tenants: dict[str, dict[str, Any]],
                 total_gpus: int) -> dict[str, int]:
        drf_result = self._drf.compute(tenants, total_gpus)
        allocation: dict[str, int] = {}
        remaining = total_gpus

        for alloc in sorted(drf_result.allocations, key=lambda a: a.priority, reverse=True):
            fair = int(alloc.fair_share * total_gpus)
            adjusted = max(0, fair + int(alloc.adjustment * total_gpus))
            allocated = min(adjusted + alloc.gpus_allocated, alloc.gpu_quota, remaining)
            allocation[alloc.tenant_id] = allocated
            remaining -= allocated

        if remaining > 0:
            for alloc in sorted(drf_result.allocations, key=lambda a: a.dominant_share):
                if remaining <= 0:
                    break
                extra = min(remaining, alloc.gpu_quota - allocation.get(alloc.tenant_id, 0))
                if extra > 0:
                    allocation[alloc.tenant_id] = allocation.get(alloc.tenant_id, 0) + extra
                    remaining -= extra

        return allocation


drf_engine = DominantResourceFairness()
proportional_scheduler = ProportionalFairnessScheduler()
