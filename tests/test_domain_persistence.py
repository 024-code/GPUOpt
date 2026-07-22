from __future__ import annotations

import json
import sqlite3

import pytest

from gpuopt.domains.models import (
    GpuNodeTelemetry,
    SchedulerJobEvent,
    ActionEvent,
    ActionOutcome,
    ActionType,
    ActionStatus,
)
from gpuopt.domains.stores import (
    DomainStore,
    PersistentStore,
    RingStore,
    create_domain_store,
    get_domain_store,
)


@pytest.fixture()
def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS domain_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT NOT NULL,
            payload TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            cluster_id TEXT,
            node_id TEXT,
            job_id TEXT,
            tenant_id TEXT,
            model_id TEXT,
            event_type TEXT
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_domain_ts ON domain_events(domain, timestamp DESC)"
    )
    return conn


class TestPersistentStore:
    def test_persist_telemetry(self, db_conn: sqlite3.Connection):
        store: PersistentStore[GpuNodeTelemetry] = PersistentStore(
            "gpu_node", max_entries=20000, persistence_enabled=True, db_conn=db_conn
        )
        telemetry = GpuNodeTelemetry(cluster_id="c1", node_name="node-a")
        store.add(telemetry)

        cur = db_conn.execute(
            "SELECT domain, payload, cluster_id FROM domain_events WHERE domain = ?",
            ("gpu_node",),
        )
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0]["domain"] == "gpu_node"
        assert rows[0]["cluster_id"] == "c1"
        payload = json.loads(rows[0]["payload"])
        assert payload["cluster_id"] == "c1"
        assert payload["node_name"] == "node-a"

    def test_persist_scheduler_event(self, db_conn: sqlite3.Connection):
        store: PersistentStore[SchedulerJobEvent] = PersistentStore(
            "scheduler_events", persistence_enabled=True, db_conn=db_conn
        )
        event = SchedulerJobEvent(
            cluster_id="c1", job_id="job-1", event_type="submitted"
        )
        store.add(event)

        cur = db_conn.execute(
            "SELECT event_type, job_id FROM domain_events WHERE domain = ?",
            ("scheduler_events",),
        )
        row = cur.fetchone()
        assert row is not None
        assert row["event_type"] == "submitted"
        assert row["job_id"] == "job-1"

    def test_persist_action_event(self, db_conn: sqlite3.Connection):
        store: PersistentStore[ActionEvent] = PersistentStore(
            "action_events", persistence_enabled=True, db_conn=db_conn
        )
        event = ActionEvent(
            action_type=ActionType.RECOMMENDATION,
            status=ActionStatus.PROPOSED,
            cluster_id="c1",
        )
        store.add(event)

        cur = db_conn.execute(
            "SELECT event_type FROM domain_events WHERE domain = ?",
            ("action_events",),
        )
        row = cur.fetchone()
        assert row is not None
        assert row["event_type"] == "recommendation"

    def test_persist_multiple_items(self, db_conn: sqlite3.Connection):
        store: PersistentStore[SchedulerJobEvent] = PersistentStore(
            "scheduler_events", max_entries=50000, persistence_enabled=True, db_conn=db_conn
        )
        for i in range(10):
            store.add(SchedulerJobEvent(
                cluster_id="c1", job_id=f"job-{i}", event_type="submitted"
            ))

        cur = db_conn.execute(
            "SELECT COUNT(*) as cnt FROM domain_events WHERE domain = ?",
            ("scheduler_events",),
        )
        assert cur.fetchone()["cnt"] == 10
        assert store.count() == 10

    def test_no_persistence_when_disabled(self, db_conn: sqlite3.Connection):
        store: PersistentStore[GpuNodeTelemetry] = PersistentStore(
            "gpu_node", persistence_enabled=False, db_conn=db_conn
        )
        store.add(GpuNodeTelemetry(cluster_id="c1", node_name="node-a"))

        cur = db_conn.execute(
            "SELECT COUNT(*) as cnt FROM domain_events"
        )
        assert cur.fetchone()["cnt"] == 0
        # Ring buffer should still work
        assert store.count() == 1

    def test_no_persistence_without_conn(self):
        store: PersistentStore[GpuNodeTelemetry] = PersistentStore(
            "gpu_node", persistence_enabled=True, db_conn=None
        )
        store.add(GpuNodeTelemetry(cluster_id="c1", node_name="node-a"))
        assert store.count() == 1

    def test_clear_also_clears_db(self, db_conn: sqlite3.Connection):
        store: PersistentStore[SchedulerJobEvent] = PersistentStore(
            "scheduler_events", persistence_enabled=True, db_conn=db_conn
        )
        store.add(SchedulerJobEvent(cluster_id="c1", job_id="j1", event_type="submitted"))
        store.clear()

        cur = db_conn.execute(
            "SELECT COUNT(*) as cnt FROM domain_events WHERE domain = ?",
            ("scheduler_events",),
        )
        assert cur.fetchone()["cnt"] == 0
        assert store.count() == 0

    def test_reload_from_db(self, db_conn: sqlite3.Connection):
        store1: PersistentStore[GpuNodeTelemetry] = PersistentStore(
            "gpu_node", persistence_enabled=True, db_conn=db_conn
        )
        store1.add(GpuNodeTelemetry(cluster_id="c1", node_name="node-a"))
        store1.add(GpuNodeTelemetry(cluster_id="c1", node_name="node-b"))

        store2: PersistentStore[GpuNodeTelemetry] = PersistentStore(
            "gpu_node", persistence_enabled=True, db_conn=db_conn
        )
        store2.reload()
        assert store2.count() == 2
        names = sorted(e.node_name for e in store2.list(limit=10))
        assert names == ["node-a", "node-b"]

    def test_reload_empty_db(self, db_conn: sqlite3.Connection):
        store: PersistentStore[GpuNodeTelemetry] = PersistentStore(
            "gpu_node", persistence_enabled=True, db_conn=db_conn
        )
        store.reload()
        assert store.count() == 0

    def test_reload_from_db_separate_domains(self, db_conn: sqlite3.Connection):
        gpu_store: PersistentStore[GpuNodeTelemetry] = PersistentStore(
            "gpu_node", persistence_enabled=True, db_conn=db_conn
        )
        gpu_store.add(GpuNodeTelemetry(cluster_id="c1", node_name="node-a"))

        sched_store: PersistentStore[SchedulerJobEvent] = PersistentStore(
            "scheduler_events", persistence_enabled=True, db_conn=db_conn
        )
        sched_store.add(SchedulerJobEvent(cluster_id="c1", job_id="j1", event_type="submitted"))

        gpu_store2: PersistentStore[GpuNodeTelemetry] = PersistentStore(
            "gpu_node", persistence_enabled=True, db_conn=db_conn
        )
        gpu_store2.reload()
        assert gpu_store2.count() == 1
        assert gpu_store2.list(limit=10)[0].node_name == "node-a"

        sched_store2: PersistentStore[SchedulerJobEvent] = PersistentStore(
            "scheduler_events", persistence_enabled=True, db_conn=db_conn
        )
        sched_store2.reload()
        assert sched_store2.count() == 1

    def test_ring_buffer_limit_respected(self, db_conn: sqlite3.Connection):
        store: PersistentStore[SchedulerJobEvent] = PersistentStore(
            "scheduler_events", max_entries=5, persistence_enabled=True, db_conn=db_conn
        )
        for i in range(20):
            store.add(SchedulerJobEvent(
                cluster_id="c1", job_id=f"job-{i}", event_type="submitted"
            ))
        # Ring buffer should be capped at 5
        assert store.count() == 5
        # DB should have all 20
        cur = db_conn.execute(
            "SELECT COUNT(*) as cnt FROM domain_events WHERE domain = ?",
            ("scheduler_events",),
        )
        assert cur.fetchone()["cnt"] == 20


class TestDomainStorePersistence:
    def test_create_domain_store_with_db(self, db_conn: sqlite3.Connection):
        store = DomainStore(db_conn=db_conn, persistence_enabled=True)
        store.gpu_node.add(GpuNodeTelemetry(cluster_id="c1", node_name="node-a"))

        cur = db_conn.execute(
            "SELECT COUNT(*) as cnt FROM domain_events WHERE domain = ?",
            ("gpu_node",),
        )
        assert cur.fetchone()["cnt"] == 1
        assert store.gpu_node.count() == 1

    def test_create_domain_store_persistence_disabled(self, db_conn: sqlite3.Connection):
        store = DomainStore(db_conn=db_conn, persistence_enabled=False)
        store.gpu_node.add(GpuNodeTelemetry(cluster_id="c1", node_name="node-a"))

        cur = db_conn.execute("SELECT COUNT(*) as cnt FROM domain_events")
        assert cur.fetchone()["cnt"] == 0
        assert store.gpu_node.count() == 1

    def test_reload_all(self, db_conn: sqlite3.Connection):
        store1 = DomainStore(db_conn=db_conn, persistence_enabled=True)
        store1.gpu_node.add(GpuNodeTelemetry(cluster_id="c1", node_name="node-a"))
        store1.scheduler_events.add(SchedulerJobEvent(
            cluster_id="c1", job_id="j1", event_type="submitted"
        ))

        store2 = DomainStore(db_conn=db_conn, persistence_enabled=True)
        store2.reload_all()
        assert store2.gpu_node.count() == 1
        assert store2.scheduler_events.count() == 1
        assert store2.gpu_node.list(limit=10)[0].node_name == "node-a"

    def test_clear_all_clears_db(self, db_conn: sqlite3.Connection):
        store = DomainStore(db_conn=db_conn, persistence_enabled=True)
        store.gpu_node.add(GpuNodeTelemetry(cluster_id="c1", node_name="node-a"))
        store.clear_all()

        cur = db_conn.execute("SELECT COUNT(*) as cnt FROM domain_events")
        assert cur.fetchone()["cnt"] == 0
        assert store.gpu_node.count() == 0

    def test_create_domain_store_function(self, db_conn: sqlite3.Connection):
        from gpuopt.domains.stores import _domain_store as orig
        store = create_domain_store(db_conn=db_conn, persistence_enabled=True)
        assert store is get_domain_store()
        store.gpu_node.add(GpuNodeTelemetry(cluster_id="c1", node_name="node-a"))

        cur = db_conn.execute(
            "SELECT COUNT(*) as cnt FROM domain_events WHERE domain = ?",
            ("gpu_node",),
        )
        assert cur.fetchone()["cnt"] == 1
        create_domain_store(db_conn=None, persistence_enabled=False)

    def test_create_domain_store_reloads(self, db_conn: sqlite3.Connection):
        store1 = DomainStore(db_conn=db_conn, persistence_enabled=True)
        store1.gpu_node.add(GpuNodeTelemetry(cluster_id="c1", node_name="node-a"))

        store2 = create_domain_store(db_conn=db_conn, persistence_enabled=True)
        assert store2.gpu_node.count() == 1
        create_domain_store(db_conn=None, persistence_enabled=False)

    def test_ring_store_subclass_isinstance(self, db_conn: sqlite3.Connection):
        store = DomainStore(db_conn=db_conn, persistence_enabled=True)
        for attr_name in vars(store):
            val = getattr(store, attr_name)
            assert isinstance(val, RingStore), f"{attr_name} is not a RingStore"

    def test_all_12_stores_persist(self, db_conn: sqlite3.Connection):
        store = DomainStore(db_conn=db_conn, persistence_enabled=True)
        for attr_name in vars(store):
            val = getattr(store, attr_name)
            assert isinstance(val, PersistentStore), f"{attr_name} is not a PersistentStore"
            assert val._persistence_enabled is True


class TestPersistenceCrossDomain:
    def test_disabled_persistence_still_uses_ring_buffer(self):
        store = DomainStore(persistence_enabled=False)
        store.gpu_node.add(GpuNodeTelemetry(cluster_id="c1", node_name="node-a"))
        assert store.gpu_node.count() == 1
        assert isinstance(store.gpu_node, RingStore)

    def test_default_store_no_persistence(self):
        store = DomainStore(persistence_enabled=False)
        assert store.gpu_node._persistence_enabled is False
        assert store.gpu_node._db_conn is None
