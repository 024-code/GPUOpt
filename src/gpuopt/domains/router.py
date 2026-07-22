from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query

from .collectors import DomainCollector
from .stores import DomainStore, get_domain_store

logger = logging.getLogger(__name__)

domain_router = APIRouter(prefix="/api/v1/domains", tags=["domains"])


def _store() -> DomainStore:
    return get_domain_store()


def _ts_or_none(value: str | None) -> datetime | None:
    if value:
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None
    return None


# ── 1. GPU & Node ──────────────────────────────────────────────

@domain_router.get("/gpu-node/telemetry")
def list_gpu_node_telemetry(
    cluster_id: str | None = Query(None),
    node_name: str | None = Query(None),
    after: str | None = Query(None),
    before: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    def fn(e: Any) -> bool:
        if cluster_id and e.cluster_id != cluster_id:
            return False
        if node_name and e.node_name != node_name:
            return False
        return True
    return store.gpu_node.query(
        after=_ts_or_none(after), before=_ts_or_none(before),
        filter_fn=fn, limit=limit,
    )


@domain_router.get("/gpu-node/summary")
def gpu_node_summary(
    cluster_id: str | None = Query(None),
    store: DomainStore = Depends(_store),
):
    entries = store.gpu_node.list(limit=1000)
    if cluster_id:
        entries = [e for e in entries if e.cluster_id == cluster_id]
    if not entries:
        return {}
    latest = entries[-1]
    total_gpus = sum(len(n.gpus) for n in entries if n.gpus)
    return {
        "node_count": len(set(e.node_name for e in entries)),
        "latest_timestamp": latest.timestamp.isoformat(),
        "total_gpu_samples": total_gpus,
        "avg_gpu_util": round(
            sum(g.utilization_gpu_pct for n in entries for g in (n.gpus or [])) / max(total_gpus, 1), 1
        ),
    }


# ── 2. Fabric & Storage ────────────────────────────────────────

@domain_router.get("/fabric-storage/telemetry")
def list_fabric_storage_telemetry(
    cluster_id: str | None = Query(None),
    after: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    def fn(e: Any) -> bool:
        if cluster_id and e.cluster_id != cluster_id:
            return False
        return True
    return store.fabric_storage.query(
        after=_ts_or_none(after), filter_fn=fn, limit=limit,
    )


@domain_router.get("/fabric-storage/nccl-events")
def list_nccl_events(
    cluster_id: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    result: list[dict[str, Any]] = []
    for entry in store.fabric_storage.list(limit=500):
        if cluster_id and entry.cluster_id != cluster_id:
            continue
        for ev in entry.nccl_events:
            result.append({
                "cluster_id": entry.cluster_id,
                "timestamp": ev.timestamp.isoformat(),
                "collective_type": ev.collective_type,
                "message_size_bytes": ev.message_size_bytes,
                "duration_us": ev.duration_us,
                "bus_bw_gbps": ev.bus_bw_gbps,
            })
    return result[-limit:]


# ── 3. Scheduler & Jobs ────────────────────────────────────────

@domain_router.get("/scheduler/state")
def list_scheduler_state(
    cluster_id: str | None = Query(None),
    after: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    def fn(e: Any) -> bool:
        if cluster_id and e.cluster_id != cluster_id:
            return False
        return True
    return store.scheduler_states.query(
        after=_ts_or_none(after), filter_fn=fn, limit=limit,
    )


@domain_router.get("/scheduler/events")
def list_scheduler_events(
    cluster_id: str | None = Query(None),
    job_id: str | None = Query(None),
    event_type: str | None = Query(None),
    after: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    def fn(e: Any) -> bool:
        if cluster_id and e.cluster_id != cluster_id:
            return False
        if job_id and e.job_id != job_id:
            return False
        if event_type and e.event_type != event_type:
            return False
        return True
    return store.scheduler_events.query(
        after=_ts_or_none(after), filter_fn=fn, limit=limit,
    )


# ── 4. Training Runtime ────────────────────────────────────────

@domain_router.get("/training/steps")
def list_training_steps(
    job_id: str | None = Query(None),
    after: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    def fn(e: Any) -> bool:
        if job_id and e.job_id != job_id:
            return False
        return True
    return store.training_steps.query(
        after=_ts_or_none(after), filter_fn=fn, limit=limit,
    )


@domain_router.get("/training/runs")
def list_training_runs(
    job_id: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    entries = store.training_runs.list(limit=1000)
    if job_id:
        entries = [e for e in entries if e.job_id == job_id]
    return entries[-limit:]


# ── 5. Inference Runtime ───────────────────────────────────────

@domain_router.get("/inference/samples")
def list_inference_samples(
    model_id: str | None = Query(None),
    after: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    def fn(e: Any) -> bool:
        if model_id and e.model_id != model_id:
            return False
        return True
    return store.inference_samples.query(
        after=_ts_or_none(after), filter_fn=fn, limit=limit,
    )


@domain_router.get("/inference/summaries")
def list_inference_summaries(
    model_id: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    entries = store.inference_summaries.list(limit=1000)
    if model_id:
        entries = [e for e in entries if e.model_id == model_id]
    return entries[-limit:]


# ── 6. Tenant & Cost ───────────────────────────────────────────

@domain_router.get("/tenant/quota")
def list_tenant_quotas(
    tenant_id: str | None = Query(None),
    after: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    def fn(e: Any) -> bool:
        if tenant_id and e.tenant_id != tenant_id:
            return False
        return True
    return store.tenant_quotas.query(
        after=_ts_or_none(after), filter_fn=fn, limit=limit,
    )


@domain_router.get("/tenant/costs")
def list_cost_allocations(
    tenant_id: str | None = Query(None),
    period: str | None = Query(None),
    after: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    def fn(e: Any) -> bool:
        if tenant_id and e.tenant_id != tenant_id:
            return False
        if period and e.period != period:
            return False
        return True
    return store.cost_allocations.query(
        after=_ts_or_none(after), filter_fn=fn, limit=limit,
    )


# ── 7. Actions & Outcomes ──────────────────────────────────────

@domain_router.get("/actions/events")
def list_action_events(
    cluster_id: str | None = Query(None),
    action_type: str | None = Query(None),
    status: str | None = Query(None),
    target_resource: str | None = Query(None),
    after: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    def fn(e: Any) -> bool:
        if cluster_id and e.cluster_id != cluster_id:
            return False
        if action_type and e.action_type.value != action_type:
            return False
        if status and e.status.value != status:
            return False
        if target_resource and e.target_resource != target_resource:
            return False
        return True
    return store.action_events.query(
        after=_ts_or_none(after), filter_fn=fn, limit=limit,
    )


@domain_router.get("/actions/outcomes")
def list_action_outcomes(
    event_id: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, le=1000),
    store: DomainStore = Depends(_store),
):
    def fn(e: Any) -> bool:
        if status and e.status.value != status:
            return False
        if event_id:
            return str(e.event_id) == event_id
        return True
    return store.action_outcomes.query(filter_fn=fn, limit=limit)


@domain_router.get("/actions/chain/{event_id}")
def action_chain(
    event_id: str,
    store: DomainStore = Depends(_store),
):
    events = store.action_events.list(limit=5000)
    chain: list[dict[str, Any]] = []
    visited: set[str] = set()
    target = event_id
    while target and target not in visited:
        visited.add(target)
        for e in events:
            if str(e.id) == target:
                chain.append(e.model_dump(mode="json"))
                parent = e.parent_event_id
                target = str(parent) if parent else None
                break
        else:
            break
    return chain


# ── Collection trigger ─────────────────────────────────────────

@domain_router.post("/collect")
def trigger_collection(
    cluster_id: str = "sandbox",
    minutes: int = Query(1, le=60),
    store: DomainStore = Depends(_store),
):
    collector = DomainCollector(store)
    collector.seed_historical(cluster_id, minutes=minutes)
    return {
        "collected": True,
        "cluster_id": cluster_id,
        "minutes": minutes,
        "store_counts": {
            "gpu_node": store.gpu_node.count(),
            "fabric_storage": store.fabric_storage.count(),
            "scheduler_events": store.scheduler_events.count(),
            "scheduler_states": store.scheduler_states.count(),
            "training_steps": store.training_steps.count(),
            "training_runs": store.training_runs.count(),
            "inference_samples": store.inference_samples.count(),
            "inference_summaries": store.inference_summaries.count(),
            "tenant_quotas": store.tenant_quotas.count(),
            "cost_allocations": store.cost_allocations.count(),
            "action_events": store.action_events.count(),
            "action_outcomes": store.action_outcomes.count(),
        },
    }


@domain_router.get("/counts")
def domain_counts(store: DomainStore = Depends(_store)):
    return {
        "gpu_node": store.gpu_node.count(),
        "fabric_storage": store.fabric_storage.count(),
        "scheduler_events": store.scheduler_events.count(),
        "scheduler_states": store.scheduler_states.count(),
        "training_steps": store.training_steps.count(),
        "training_runs": store.training_runs.count(),
        "inference_samples": store.inference_samples.count(),
        "inference_summaries": store.inference_summaries.count(),
        "tenant_quotas": store.tenant_quotas.count(),
        "cost_allocations": store.cost_allocations.count(),
        "action_events": store.action_events.count(),
        "action_outcomes": store.action_outcomes.count(),
    }
