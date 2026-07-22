from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from .models import (
    GpuNodeTelemetry,
    FabricStorageTelemetry,
    SchedulerJobEvent,
    SchedulerState,
    TrainingStepMetric,
    TrainingRunSummary,
    InferenceRequestSample,
    InferenceSummary,
    TenantQuota,
    CostAllocation,
    ActionEvent,
    ActionOutcome,
    ActionStatus,
)

T = TypeVar("T")


def _extract_field(obj: Any, *names: str) -> Any:
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            if isinstance(v, datetime):
                return v.isoformat()
            if isinstance(v, (ActionStatus,)):
                return v.value
            return v
    return None


def _timestamp(obj: Any) -> str:
    ts = _extract_field(obj, "timestamp", "collected_at")
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()
    return ts


def _event_type(obj: Any) -> str | None:
    et = _extract_field(obj, "event_type", "action_type")
    if et is not None:
        return str(et)
    return None


class RingStore(Generic[T]):
    def __init__(self, max_entries: int = 50000) -> None:
        self._entries: list[T] = []
        self._max = max_entries

    def add(self, item: T) -> None:
        self._entries.append(item)
        if len(self._entries) > self._max:
            self._entries.pop(0)

    def list(self, limit: int = 100, offset: int = 0) -> list[T]:
        return self._entries[offset:offset + limit]

    def query(
        self,
        *,
        after: datetime | None = None,
        before: datetime | None = None,
        filter_fn: callable | None = None,
        limit: int = 100,
    ) -> list[T]:
        results = self._entries
        if after:
            results = [e for e in results if hasattr(e, 'timestamp') and e.timestamp >= after]
        if before:
            results = [e for e in results if hasattr(e, 'timestamp') and e.timestamp <= before]
        if filter_fn:
            results = [e for e in results if filter_fn(e)]
        return results[-limit:] if len(results) > limit else results

    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


class PersistentStore(RingStore[T]):
    DB_DOMAINS: set[str] = set()

    def __init__(
        self,
        domain: str,
        max_entries: int = 50000,
        *,
        persistence_enabled: bool = True,
        db_conn: Any = None,
    ) -> None:
        super().__init__(max_entries=max_entries)
        self._domain = domain
        self._persistence_enabled = persistence_enabled
        self._db_conn = db_conn

    def add(self, item: T) -> None:
        super().add(item)
        if self._persistence_enabled and self._db_conn is not None:
            self._persist(item)

    def _persist(self, item: T) -> None:
        try:
            payload = json.dumps(item.model_dump(mode="json"), default=str)
            ts = _timestamp(item)
            cluster_id = _extract_field(item, "cluster_id")
            node_id = _extract_field(item, "node_id", "node_name")
            job_id = _extract_field(item, "job_id")
            tenant_id = _extract_field(item, "tenant_id")
            model_id = _extract_field(item, "model_id")
            event_type = _event_type(item)
            self._db_conn.execute(
                """INSERT INTO domain_events
                   (domain, payload, timestamp, cluster_id, node_id, job_id, tenant_id, model_id, event_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self._domain, payload, ts, cluster_id, node_id, job_id, tenant_id, model_id, event_type),
            )
            self._db_conn.commit()
        except Exception:
            pass

    def reload(self, limit: int = 50000) -> None:
        if self._db_conn is None:
            return
        try:
            cur = self._db_conn.execute(
                "SELECT payload FROM domain_events WHERE domain = ? ORDER BY id DESC LIMIT ?",
                (self._domain, limit),
            )
            self._entries.clear()
            for row in cur.fetchall():
                raw = row["payload"] if isinstance(row, dict) else row[0]
                d = json.loads(raw)
                try:
                    from .models import (
                        GpuNodeTelemetry, FabricStorageTelemetry,
                        SchedulerJobEvent, SchedulerState,
                        TrainingStepMetric, TrainingRunSummary,
                        InferenceRequestSample, InferenceSummary,
                        TenantQuota, CostAllocation,
                        ActionEvent, ActionOutcome,
                    )
                    MODEL_MAP: dict[str, type] = {
                        "gpu_node": GpuNodeTelemetry,
                        "fabric_storage": FabricStorageTelemetry,
                        "scheduler_events": SchedulerJobEvent,
                        "scheduler_states": SchedulerState,
                        "training_steps": TrainingStepMetric,
                        "training_runs": TrainingRunSummary,
                        "inference_samples": InferenceRequestSample,
                        "inference_summaries": InferenceSummary,
                        "tenant_quotas": TenantQuota,
                        "cost_allocations": CostAllocation,
                        "action_events": ActionEvent,
                        "action_outcomes": ActionOutcome,
                    }
                    model_cls = MODEL_MAP.get(self._domain)
                    if model_cls is not None:
                        self._entries.append(model_cls.model_validate(d))
                except Exception:
                    pass
            if self._entries and len(self._entries) > self._max:
                self._entries = self._entries[-self._max:]
        except Exception:
            pass

    def clear(self) -> None:
        super().clear()
        if self._persistence_enabled and self._db_conn is not None:
            try:
                self._db_conn.execute(
                    "DELETE FROM domain_events WHERE domain = ?", (self._domain,)
                )
                self._db_conn.commit()
            except Exception:
                pass


class DomainStore:
    def __init__(self, db_conn: Any = None, persistence_enabled: bool = True) -> None:
        kw = {"persistence_enabled": persistence_enabled, "db_conn": db_conn}
        self.gpu_node: PersistentStore[GpuNodeTelemetry] = PersistentStore("gpu_node", max_entries=20000, **kw)
        self.fabric_storage: PersistentStore[FabricStorageTelemetry] = PersistentStore("fabric_storage", max_entries=20000, **kw)
        self.scheduler_events: PersistentStore[SchedulerJobEvent] = PersistentStore("scheduler_events", max_entries=50000, **kw)
        self.scheduler_states: PersistentStore[SchedulerState] = PersistentStore("scheduler_states", max_entries=20000, **kw)
        self.training_steps: PersistentStore[TrainingStepMetric] = PersistentStore("training_steps", max_entries=100000, **kw)
        self.training_runs: PersistentStore[TrainingRunSummary] = PersistentStore("training_runs", max_entries=5000, **kw)
        self.inference_samples: PersistentStore[InferenceRequestSample] = PersistentStore("inference_samples", max_entries=100000, **kw)
        self.inference_summaries: PersistentStore[InferenceSummary] = PersistentStore("inference_summaries", max_entries=5000, **kw)
        self.tenant_quotas: PersistentStore[TenantQuota] = PersistentStore("tenant_quotas", max_entries=20000, **kw)
        self.cost_allocations: PersistentStore[CostAllocation] = PersistentStore("cost_allocations", max_entries=50000, **kw)
        self.action_events: PersistentStore[ActionEvent] = PersistentStore("action_events", max_entries=50000, **kw)
        self.action_outcomes: PersistentStore[ActionOutcome] = PersistentStore("action_outcomes", max_entries=50000, **kw)

    def reload_all(self) -> None:
        for store in vars(self).values():
            if isinstance(store, PersistentStore):
                store.reload()

    def clear_all(self) -> None:
        for store in vars(self).values():
            if isinstance(store, (RingStore, PersistentStore)):
                store.clear()


_domain_store = DomainStore(persistence_enabled=False)


def get_domain_store() -> DomainStore:
    return _domain_store


def create_domain_store(db_conn: Any = None, persistence_enabled: bool = True) -> DomainStore:
    global _domain_store
    store = DomainStore(db_conn=db_conn, persistence_enabled=persistence_enabled)
    if persistence_enabled and db_conn is not None:
        store.reload_all()
    _domain_store = store
    return store
